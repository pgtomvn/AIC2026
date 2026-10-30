"""ASR topic graph: ordered transcript topics, relation edges, and bounded graph search.

Each video transcript is condensed into ordered topics ``u_i = (v, d_i, s_i, e_i)``.
Topics are linked by temporal ``NEXT`` edges and semantic ``GRAPH_RELATION`` edges
(``E = E_next ∪ E_sem``); entity mentions add ``MENTIONS``/``SHARES_ENTITY`` evidence.
A query is scored as ``S(u_i, q) = cos(z_q, z_i) + min(Σ_{j∈C} b(r_ij), τ)`` where ``C``
is the top-K stage candidate set.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

TOPIC_GRAPH_VERSION = "asr-topic-graph-v1"

RELATION_NEXT = "NEXT"
RELATION_MENTIONS = "MENTIONS"
RELATION_SHARES_ENTITY = "SHARES_ENTITY"
RELATION_GRAPH = "GRAPH_RELATION"
RELATIONS = (RELATION_NEXT, RELATION_MENTIONS, RELATION_SHARES_ENTITY, RELATION_GRAPH)
# Stored topic-topic relation codes.
_EDGE_CODES = {RELATION_NEXT: 0, RELATION_GRAPH: 1}
_EDGE_NAMES = {code: name for name, code in _EDGE_CODES.items()}

_CAPITALIZED_RUN = re.compile(
    r"(?<![\wÀ-ỹ])((?:[A-ZÀ-ỸĐ][a-zà-ỹđ]+)(?:\s+[A-ZÀ-ỸĐ][a-zà-ỹđ]+)+)(?![\wÀ-ỹ])"
)
_ACRONYM = re.compile(r"(?<![\wÀ-ỹ])([A-ZĐ]{2,}(?:\.[A-ZĐ]{2,})*)(?![\wÀ-ỹ])")
_SENTENCE_START = re.compile(r"(?:^|[.!?…]\s+)$")


@dataclass(frozen=True, slots=True)
class AsrSegment:
    """One timed transcript segment of a video."""

    video_id: str
    start_ms: int
    end_ms: int
    text: str
    source_frame_idx: int | None = None


@dataclass(frozen=True, slots=True)
class Keyframe:
    """One organizer keyframe: 1-based ordinal, source frame index, and time."""

    ordinal: int
    frame_idx: int
    pts_ms: int


@dataclass(frozen=True, slots=True)
class TopicGraphParams:
    """Build and search parameters; defaults follow the GRASP topic graph."""

    max_gap_ms: int = 4_000
    max_duration_ms: int = 60_000
    max_chars: int = 900
    semantic_neighbors: int = 5
    semantic_min_similarity: float = 0.60
    entity_max_topics: int = 300
    candidate_k: int = 50
    final_k: int = 10
    tau: float = 0.20
    next_bonus: float = 0.10
    semantic_bonus: float = 0.08
    hops: int = 1
    max_display_topics: int = 60
    max_display_entities: int = 12
    max_search_topics: int = 500
    min_similarity: float = 0.20

    def __post_init__(self) -> None:
        for name in ("max_duration_ms", "max_chars", "entity_max_topics", "candidate_k",
                     "final_k", "max_display_topics", "max_search_topics"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_gap_ms", "semantic_neighbors", "hops", "max_display_entities"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("semantic_min_similarity", "tau", "next_bonus", "semantic_bonus", "min_similarity"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.max_search_topics < self.candidate_k:
            raise ValueError("max_search_topics must cover candidate_k")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> TopicGraphParams:
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in dict(value or {}).items() if key in allowed})


@dataclass(frozen=True, slots=True)
class Topic:
    """One condensed transcript topic and its keyframe span."""

    topic_id: int
    batch_id: str
    video_id: str
    ordinal: int
    start_ms: int
    end_ms: int
    text: str
    source_frame_idx: int
    keyframe_ordinal: int | None = None
    shot_start: int | None = None
    shot_end: int | None = None
    entity_ids: tuple[int, ...] = ()
    transcript: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["entity_ids"] = list(self.entity_ids)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Topic:
        data = dict(value)
        data["entity_ids"] = tuple(int(item) for item in data.get("entity_ids", ()))
        return cls(**data)


def normalize_text(value: str) -> str:
    """Return NFC text with collapsed whitespace."""

    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def segment_topics(
    segments: Sequence[AsrSegment], params: TopicGraphParams = TopicGraphParams()
) -> list[list[AsrSegment]]:
    """Group one video's ordered segments into contiguous topics.

    A new topic starts after a silence gap, or when the running topic would exceed the
    duration or character budget.  Empty segments are ignored.
    """

    ordered = sorted(
        (item for item in segments if normalize_text(item.text)),
        key=lambda item: (item.start_ms, item.end_ms),
    )
    if len({item.video_id for item in ordered}) > 1:
        raise ValueError("segment_topics accepts one video at a time")
    if any(item.start_ms < 0 or item.end_ms < item.start_ms for item in ordered):
        raise ValueError("ASR segment timestamps must be nonnegative and ordered")
    groups: list[list[AsrSegment]] = []
    current: list[AsrSegment] = []
    chars = 0
    for segment in ordered:
        text = normalize_text(segment.text)
        if current:
            gap = segment.start_ms - max(item.end_ms for item in current)
            duration = segment.end_ms - current[0].start_ms
            if (
                gap > params.max_gap_ms
                or duration > params.max_duration_ms
                or chars + len(text) + 1 > params.max_chars
            ):
                groups.append(current)
                current, chars = [], 0
        current.append(segment)
        chars += len(text) + 1
    if current:
        groups.append(current)
    return groups


def keyframe_span(
    start_ms: int, end_ms: int, keyframes: Sequence[Keyframe]
) -> tuple[Keyframe, int, int] | None:
    """Return (representative keyframe, first ordinal, last ordinal) for a time span."""

    if not keyframes:
        return None
    inside = [item for item in keyframes if start_ms <= item.pts_ms <= end_ms]
    if not inside:
        middle = (start_ms + end_ms) // 2
        nearest = min(keyframes, key=lambda item: abs(item.pts_ms - middle))
        return nearest, nearest.ordinal, nearest.ordinal
    return inside[len(inside) // 2], inside[0].ordinal, inside[-1].ordinal


def extract_mentions(text: str, extractor: Any | None = None) -> tuple[str, ...]:
    """Extract entity mentions: dictionary/rule entities plus proper-noun runs."""

    normalized = normalize_text(text)
    found: dict[str, str] = {}

    def add(value: str) -> None:
        value = value.strip(" ,.;:!?\"'()[]")
        if len(value) < 2 or len(value) > 64:
            return
        found.setdefault(value.casefold(), value)

    if extractor is not None:
        for entity in extractor.extract(normalized):
            add(entity.text)
    for match in _CAPITALIZED_RUN.finditer(normalized):
        words = match.group(1).split()
        # A sentence-initial capital is grammar, not a name; drop that first word.
        if _SENTENCE_START.search(normalized[: match.start(1)]):
            words = words[1:]
        if len(words) >= 2:
            add(" ".join(words))
    for match in _ACRONYM.finditer(normalized):
        add(match.group(1))
    return tuple(found[key] for key in sorted(found))


@dataclass(slots=True)
class TopicGraphArtifact:
    """In-memory topic graph prior to serialization."""

    batch_id: str
    topics: list[Topic]
    embeddings: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_rel: np.ndarray
    edge_weight: np.ndarray
    entities: list[str]
    params: TopicGraphParams
    encoder_fingerprint: str = ""
    summarizer: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)


def _semantic_edges(
    embeddings: np.ndarray, topics: Sequence[Topic], params: TopicGraphParams
) -> list[tuple[int, int, float]]:
    count = embeddings.shape[0]
    k = max(0, int(params.semantic_neighbors))
    if count < 2 or k == 0:
        return []
    matrix = embeddings.astype(np.float32, copy=False)
    pairs: dict[tuple[int, int], float] = {}
    block = 1024
    for start in range(0, count, block):
        scores = matrix[start : start + block] @ matrix.T
        for row in range(scores.shape[0]):
            index = start + row
            scores[row, index] = -np.inf
        take = min(k + 2, count - 1)
        top = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        for row in range(scores.shape[0]):
            index = start + row
            ranked = sorted(top[row], key=lambda item: -scores[row, item])
            kept = 0
            for other in ranked:
                similarity = float(scores[row, other])
                if similarity < params.semantic_min_similarity or kept >= k:
                    break
                left, right = topics[index], topics[int(other)]
                if left.video_id == right.video_id and abs(left.ordinal - right.ordinal) == 1:
                    continue
                key = (min(index, int(other)), max(index, int(other)))
                # Float rounding can put near-duplicate topics just above 1.
                pairs[key] = max(pairs.get(key, 0.0), min(similarity, 1.0))
                kept += 1
    return [(left, right, weight) for (left, right), weight in sorted(pairs.items())]


def build_topic_graph(
    batch_id: str,
    segments_by_video: Mapping[str, Sequence[AsrSegment]],
    encode: Callable[[list[str]], np.ndarray],
    *,
    keyframes_by_video: Mapping[str, Sequence[Keyframe]] | None = None,
    params: TopicGraphParams = TopicGraphParams(),
    summarize: Callable[[str], str | None] | None = None,
    summarizer_name: str = "none",
    entity_extractor: Any | None = None,
    encoder_fingerprint: str = "",
    progress: Callable[[str, int, int], None] | None = None,
) -> TopicGraphArtifact:
    """Build topics, embeddings, and edges for one batch."""

    keyframes_by_video = keyframes_by_video or {}
    drafts: list[dict[str, Any]] = []
    videos = sorted(segments_by_video)
    for position, video_id in enumerate(videos, start=1):
        frames = sorted(keyframes_by_video.get(video_id, ()), key=lambda item: item.pts_ms)
        for ordinal, group in enumerate(segment_topics(segments_by_video[video_id], params)):
            transcript = " ".join(normalize_text(item.text) for item in group)
            start_ms, end_ms = group[0].start_ms, max(item.end_ms for item in group)
            span = keyframe_span(start_ms, end_ms, frames)
            if span is not None:
                keyframe, shot_start, shot_end = span
                source_frame_idx, keyframe_ordinal = keyframe.frame_idx, keyframe.ordinal
            else:
                indices = [item.source_frame_idx for item in group if item.source_frame_idx is not None]
                if not indices:
                    continue
                source_frame_idx = int(indices[len(indices) // 2])
                keyframe_ordinal = shot_start = shot_end = None
            drafts.append(
                {
                    "video_id": video_id,
                    "ordinal": ordinal,
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "transcript": transcript,
                    "source_frame_idx": int(source_frame_idx),
                    "keyframe_ordinal": keyframe_ordinal,
                    "shot_start": shot_start,
                    "shot_end": shot_end,
                }
            )
        if progress is not None:
            progress("segment", position, len(videos))

    texts: list[str] = []
    for position, draft in enumerate(drafts, start=1):
        description = summarize(draft["transcript"]) if summarize is not None else None
        texts.append(normalize_text(description) if description else draft["transcript"])
        if progress is not None and summarize is not None:
            progress("summarize", position, len(drafts))

    mentions = [extract_mentions(draft["transcript"], entity_extractor) for draft in drafts]
    frequency = Counter(name.casefold() for names in mentions for name in set(names))
    canonical: dict[str, str] = {}
    for names in mentions:
        for name in names:
            canonical.setdefault(name.casefold(), name)
    entities = sorted(
        canonical[key] for key, count in frequency.items() if 2 <= count <= params.entity_max_topics
    )
    entity_index = {name.casefold(): index for index, name in enumerate(entities)}

    topics = [
        Topic(
            topic_id=index,
            batch_id=batch_id,
            video_id=draft["video_id"],
            ordinal=draft["ordinal"],
            start_ms=draft["start_ms"],
            end_ms=draft["end_ms"],
            text=texts[index],
            transcript=draft["transcript"],
            source_frame_idx=draft["source_frame_idx"],
            keyframe_ordinal=draft["keyframe_ordinal"],
            shot_start=draft["shot_start"],
            shot_end=draft["shot_end"],
            entity_ids=tuple(
                sorted({entity_index[name.casefold()] for name in mentions[index] if name.casefold() in entity_index})
            ),
        )
        for index, draft in enumerate(drafts)
    ]
    parts = []
    for start in range(0, len(texts), 512):
        parts.append(np.asarray(encode(texts[start : start + 512]), dtype=np.float32))
        if progress is not None:
            progress("encode", min(start + 512, len(texts)), len(texts))
    embeddings = _normalize_rows(np.concatenate(parts, axis=0)) if texts else np.zeros((0, 1), np.float32)
    if embeddings.shape[0] != len(topics):
        raise ValueError("encoder returned a different number of vectors than topics")

    edges: list[tuple[int, int, int, float]] = []
    for left, right in zip(topics, topics[1:]):
        if left.video_id == right.video_id:
            edges.append((left.topic_id, right.topic_id, _EDGE_CODES[RELATION_NEXT], 1.0))
    for left, right, weight in _semantic_edges(embeddings, topics, params):
        edges.append((left, right, _EDGE_CODES[RELATION_GRAPH], weight))
    array = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    return TopicGraphArtifact(
        batch_id=batch_id,
        topics=topics,
        embeddings=embeddings,
        edge_src=array[:, 0].astype(np.int32),
        edge_dst=array[:, 1].astype(np.int32),
        edge_rel=array[:, 2].astype(np.int8),
        edge_weight=array[:, 3].astype(np.float32),
        entities=entities,
        params=params,
        encoder_fingerprint=encoder_fingerprint,
        summarizer=summarizer_name,
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2-D matrix")
    if matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must have finite values and a positive dimension")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("embedding vectors must not be zero")
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def write_topic_graph(artifact: TopicGraphArtifact, directory: str | Path) -> dict[str, Any]:
    """Write a new local artifact; refuse to replace existing generated state."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    with gzip.open(root / "topics.jsonl.gz", "wt", encoding="utf-8") as handle:
        for topic in artifact.topics:
            handle.write(json.dumps(topic.to_dict(), ensure_ascii=False) + "\n")
    np.save(root / "embeddings.npy", artifact.embeddings.astype(np.float16))
    np.savez_compressed(
        root / "edges.npz",
        src=artifact.edge_src,
        dst=artifact.edge_dst,
        rel=artifact.edge_rel,
        weight=artifact.edge_weight,
    )
    (root / "entities.json").write_text(
        json.dumps(artifact.entities, ensure_ascii=False), encoding="utf-8"
    )
    digest = hashlib.sha256()
    for name in ("topics.jsonl.gz", "embeddings.npy", "edges.npz", "entities.json"):
        digest.update((root / name).read_bytes())
    manifest = {
        "version": TOPIC_GRAPH_VERSION,
        "batch_id": artifact.batch_id,
        "topics": len(artifact.topics),
        "videos": len({topic.video_id for topic in artifact.topics}),
        "entities": len(artifact.entities),
        "edges": {
            name: int(np.count_nonzero(artifact.edge_rel == code)) for name, code in _EDGE_CODES.items()
        },
        "embedding_dim": int(artifact.embeddings.shape[1]) if artifact.embeddings.size else 0,
        "encoder_fingerprint": artifact.encoder_fingerprint,
        "summarizer": artifact.summarizer,
        "params": asdict(artifact.params),
        "content_sha256": digest.hexdigest(),
        **artifact.extra,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


@dataclass(frozen=True, slots=True)
class _Neighbor:
    topic: int
    relation: str
    weight: float


class TopicGraph:
    """Loaded, searchable topic graph for one batch."""

    def __init__(
        self,
        batch_id: str,
        topics: Sequence[Topic],
        embeddings: np.ndarray,
        edges: Iterable[tuple[int, int, str, float]],
        entities: Sequence[str],
        params: TopicGraphParams = TopicGraphParams(),
    ) -> None:
        self.batch_id = batch_id
        self.topics = list(topics)
        self.embeddings = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        if self.embeddings.shape[0] != len(self.topics):
            raise ValueError("topic and embedding counts differ")
        self.entities = list(entities)
        self.params = params
        if any(topic.topic_id != index or topic.batch_id != batch_id
               for index, topic in enumerate(self.topics)):
            raise ValueError("topic ids must be contiguous and belong to this batch")
        if any(entity < 0 or entity >= len(self.entities)
               for topic in self.topics for entity in topic.entity_ids):
            raise ValueError("topic refers to an unknown entity")
        self._adjacency: dict[int, list[_Neighbor]] = defaultdict(list)
        self._next: dict[int, int] = {}
        self._previous: dict[int, int] = {}
        for src, dst, relation, weight in edges:
            src, dst = int(src), int(dst)
            if math.isfinite(weight) and 1.0 < weight <= 1.0 + 1e-3:
                weight = 1.0  # float rounding of near-duplicate cosine similarities
            if not (0 <= src < len(self.topics) and 0 <= dst < len(self.topics)) or src == dst:
                raise ValueError("edge endpoints must refer to distinct existing topics")
            if relation not in _EDGE_CODES or not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("invalid graph relation or weight")
            if relation == RELATION_NEXT and (
                self.topics[src].video_id != self.topics[dst].video_id
                or self.topics[src].ordinal >= self.topics[dst].ordinal
                or src in self._next or dst in self._previous
            ):
                raise ValueError("NEXT must be an ordered, nonbranching chain within one video")
            self._adjacency[src].append(_Neighbor(dst, relation, float(weight)))
            self._adjacency[dst].append(_Neighbor(src, relation, float(weight)))
            if relation == RELATION_NEXT:
                self._next[src] = dst
                self._previous[dst] = src
        self._entity_topics: dict[int, list[int]] = defaultdict(list)
        for topic in self.topics:
            for entity in topic.entity_ids:
                self._entity_topics[entity].append(topic.topic_id)

    @classmethod
    def load(cls, directory: str | Path, params: TopicGraphParams | None = None) -> TopicGraph:
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != TOPIC_GRAPH_VERSION:
            raise ValueError("unsupported topic graph version")
        digest = hashlib.sha256()
        for name in ("topics.jsonl.gz", "embeddings.npy", "edges.npz", "entities.json"):
            with (root / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        if digest.hexdigest() != manifest.get("content_sha256"):
            raise ValueError("topic graph integrity mismatch; rebuild into a new directory")
        with gzip.open(root / "topics.jsonl.gz", "rt", encoding="utf-8") as handle:
            topics = [Topic.from_dict(json.loads(line)) for line in handle if line.strip()]
        embeddings = np.load(root / "embeddings.npy", allow_pickle=False)
        with np.load(root / "edges.npz", allow_pickle=False) as data:
            if len({len(data[name]) for name in ("src", "dst", "rel", "weight")}) != 1:
                raise ValueError("graph edge arrays have different lengths")
            edges = [
                (int(src), int(dst), _EDGE_NAMES[int(rel)], float(weight))
                for src, dst, rel, weight in zip(data["src"], data["dst"], data["rel"], data["weight"])
            ]
        entities = json.loads((root / "entities.json").read_text(encoding="utf-8"))
        stored = TopicGraphParams.from_mapping(manifest.get("params"))
        result = cls(manifest["batch_id"], topics, embeddings, edges, entities, params or stored)
        result.encoder_fingerprint = str(manifest.get("encoder_fingerprint", ""))
        return result

    def __len__(self) -> int:
        return len(self.topics)

    def neighbors(self, topic_id: int) -> list[_Neighbor]:
        return list(self._adjacency.get(int(topic_id), ()))

    def chain(self, topic_id: int, before: int = 2, after: int = 2) -> list[Topic]:
        """Return the temporal NEXT chain around one topic."""

        ids = [int(topic_id)]
        cursor = int(topic_id)
        for _ in range(max(0, before)):
            if cursor not in self._previous:
                break
            cursor = self._previous[cursor]
            ids.insert(0, cursor)
        cursor = int(topic_id)
        for _ in range(max(0, after)):
            if cursor not in self._next:
                break
            cursor = self._next[cursor]
            ids.append(cursor)
        return [self.topics[item] for item in ids]

    def _bonus(self, scores: Mapping[int, float], candidates: set[int], node: int) -> float:
        total = 0.0
        for neighbor in self._adjacency.get(node, ()):
            if neighbor.topic not in candidates or neighbor.topic == node:
                continue
            if neighbor.relation == RELATION_NEXT:
                beta = self.params.next_bonus
            elif neighbor.relation == RELATION_GRAPH:
                beta = self.params.semantic_bonus * neighbor.weight
            else:
                continue
            total += beta * max(scores.get(neighbor.topic, 0.0), 0.0)
        return min(total, self.params.tau)

    def _expand(self, seeds: Iterable[int], hops: int, allowed: set[int]) -> set[int]:
        found = set(int(item) for item in seeds)
        frontier = set(found)
        for _ in range(max(0, hops)):
            following: set[int] = set()
            for node in sorted(frontier):
                for neighbor in self._adjacency.get(node, ()):
                    if len(found) + len(following) >= self.params.max_search_topics:
                        break
                    if neighbor.topic not in found and neighbor.topic in allowed:
                        following.add(neighbor.topic)
            found |= following
            frontier = following
        return found

    def _rank(
        self, query: np.ndarray, pool: np.ndarray, candidate_k: int
    ) -> tuple[dict[int, float], dict[int, float], set[int]]:
        cosine = self.embeddings[pool] @ query
        take = min(candidate_k, len(pool))
        order = np.argsort(-cosine)[:take]
        candidates = {int(pool[item]) for item in order}
        scores = {int(node): float(value) for node, value in zip(pool, cosine)}
        bonuses = {node: self._bonus(scores, candidates, node) for node in scores}
        return scores, bonuses, candidates

    def _path(self, start: int, anchors: set[int], allowed: set[int], limit: int = 3) -> list[dict[str, Any]]:
        if start in anchors:
            return [{"topic_id": start, "relation": None}]
        previous: dict[int, tuple[int, str]] = {}
        queue: deque[tuple[int, int]] = deque([(start, 0)])
        seen = {start}
        while queue:
            node, depth = queue.popleft()
            if depth >= limit:
                continue
            for neighbor in self._adjacency.get(node, ()):
                if neighbor.topic in seen or neighbor.topic not in allowed:
                    continue
                seen.add(neighbor.topic)
                previous[neighbor.topic] = (node, neighbor.relation)
                if neighbor.topic in anchors:
                    steps = [{"topic_id": neighbor.topic, "relation": None}]
                    cursor = neighbor.topic
                    while cursor != start:
                        parent, relation = previous[cursor]
                        steps.append({"topic_id": parent, "relation": relation})
                        cursor = parent
                    return steps
                queue.append((neighbor.topic, depth + 1))
        return [{"topic_id": start, "relation": None}]

    def search(
        self,
        stage1: np.ndarray,
        stage2: np.ndarray | None = None,
        *,
        final_k: int | None = None,
        video_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Two-stage graph search returning matches, paths, and a display subgraph."""

        params = self.params
        final_k = params.final_k if final_k is None else int(final_k)
        if not 1 <= final_k <= 1000:
            raise ValueError("final_k must be between 1 and 1000")
        empty = {"subgraph_size": 0, "anchors": [], "matches": [], "nodes": [], "edges": [],
                 "refined": stage2 is not None, "stage2_candidates": 0}
        if not self.topics:
            return {**empty, "fallback_reason": "topic_graph_empty"}
        q1 = _normalize_rows(np.asarray(stage1, dtype=np.float32).reshape(1, -1))[0]
        if len(q1) != self.embeddings.shape[1]:
            raise ValueError("query and topic embedding dimensions differ")
        everything = np.arange(len(self.topics))
        if video_ids is not None:
            wanted = set(video_ids)
            everything = np.asarray([t.topic_id for t in self.topics if t.video_id in wanted], dtype=np.int64)
        cosine1 = self.embeddings[everything] @ q1
        take = min(params.candidate_k, len(everything))
        anchors_order = [int(everything[item]) for item in np.argsort(-cosine1)[:take]
                         if cosine1[item] >= params.min_similarity]
        if not anchors_order:
            return {**empty, "fallback_reason": "no_topic_evidence"}
        anchors = set(anchors_order)
        subgraph = self._expand(anchors, params.hops, set(int(item) for item in everything))
        pool = np.asarray(sorted(subgraph), dtype=np.int64)
        s1, b1, _ = self._rank(q1, pool, params.candidate_k)
        if stage2 is not None:
            q2 = _normalize_rows(np.asarray(stage2, dtype=np.float32).reshape(1, -1))[0]
            if len(q2) != self.embeddings.shape[1]:
                raise ValueError("query and topic embedding dimensions differ")
            s2, b2, stage2_candidates = self._rank(q2, pool, params.candidate_k)
            final = {node: s2[node] + b2[node] for node in s2 if s2[node] >= params.min_similarity}
        else:
            s2, b2, stage2_candidates = {}, {}, set()
            final = {node: s1[node] + b1[node] for node in s1 if s1[node] >= params.min_similarity}
        ranked = sorted(final, key=lambda node: (-final[node], node))[:final_k]
        matches = []
        for rank, node in enumerate(ranked, start=1):
            matches.append(
                {
                    "rank": rank,
                    "score": round(final[node], 6),
                    "stage1_score": round(s1[node] + b1[node], 6),
                    "stage2_score": round(s2[node] + b2[node], 6) if s2 else None,
                    "cosine": round(s2.get(node, s1[node]), 6),
                    "graph_bonus": round(b2.get(node, b1[node]), 6),
                    "path": self._path(node, anchors, subgraph),
                    **self.topic_payload(node),
                }
            )
        nodes, edges = self._display(anchors_order, ranked, matches, subgraph)
        return {
            "subgraph_size": len(subgraph),
            "anchors": anchors_order[:10],
            "matches": matches,
            "nodes": nodes,
            "edges": edges,
            "refined": stage2 is not None,
            "stage2_candidates": len(stage2_candidates),
            "fallback_reason": None if matches else "no_refinement_evidence",
        }

    def topic_payload(self, topic_id: int) -> dict[str, Any]:
        topic = self.topics[int(topic_id)]
        result = topic.to_dict()
        result["entities"] = [self.entities[item] for item in topic.entity_ids if item < len(self.entities)]
        result["evidence_source"] = "asr"
        result.pop("entity_ids", None)
        return result

    def _display(
        self,
        anchors_order: Sequence[int],
        ranked: Sequence[int],
        matches: Sequence[Mapping[str, Any]],
        subgraph: set[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        params = self.params
        roles: dict[int, str] = {}
        for node in ranked[:params.max_display_topics]:
            roles[node] = "match"
        for node in anchors_order[:10]:
            if len(roles) >= params.max_display_topics:
                break
            roles.setdefault(node, "anchor")
        for match in matches:
            for step in match["path"]:
                if len(roles) >= params.max_display_topics:
                    break
                roles.setdefault(int(step["topic_id"]), "related")
        for node in list(ranked):
            for neighbor in self._adjacency.get(node, ()):
                if len(roles) >= params.max_display_topics:
                    break
                if neighbor.topic in subgraph:
                    roles.setdefault(neighbor.topic, "related")
        shown = set(roles)
        nodes = [
            {"id": f"t{node}", "kind": "topic", "role": role, **self.topic_payload(node)}
            for node, role in roles.items()
        ]
        edges: list[dict[str, Any]] = []
        seen_pairs: set[tuple[int, int, str]] = set()
        for node in shown:
            for neighbor in self._adjacency.get(node, ()):
                if neighbor.topic not in shown:
                    continue
                left, right = (node, neighbor.topic)
                if neighbor.relation == RELATION_NEXT:
                    if self._next.get(left) != right:
                        continue
                else:
                    left, right = min(left, right), max(left, right)
                key = (left, right, neighbor.relation)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                edges.append(
                    {
                        "source": f"t{left}",
                        "target": f"t{right}",
                        "relation": neighbor.relation,
                        "weight": round(neighbor.weight, 4),
                    }
                )
        mention_counts = Counter(
            entity for node in shown for entity in self.topics[node].entity_ids
        )
        top_entities = [
            entity for entity, count in mention_counts.most_common() if count >= 1
        ][: params.max_display_entities]
        for entity in top_entities:
            nodes.append(
                {"id": f"e{entity}", "kind": "entity", "role": "entity", "label": self.entities[entity]}
            )
            members = sorted(node for node in shown if entity in self.topics[node].entity_ids)
            for node in members:
                edges.append(
                    {"source": f"t{node}", "target": f"e{entity}", "relation": RELATION_MENTIONS, "weight": 1.0}
                )
            for left, right in zip(members, members[1:]):
                edges.append(
                    {
                        "source": f"t{left}",
                        "target": f"t{right}",
                        "relation": RELATION_SHARES_ENTITY,
                        "weight": 1.0,
                        "entity": self.entities[entity],
                    }
                )
        return nodes, edges
