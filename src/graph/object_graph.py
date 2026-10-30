"""Frame object graph: detections as frame-local nodes, box predicates as edges.

A query is canonicalized into a query graph ``(V_q, E_q)`` of typed object nodes (with
optional counts) and predicate edges.  Each frame is ranked by
``S(f, q) = |C_sat(f, q)| / (|V_q| + |E_q|)`` under the best injective mapping
``φ: V_q → V_f``.  Detection ids are frame-local, and an undetected object is treated as
unobserved rather than absent: it only fails its own constraints.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

OBJECT_GRAPH_VERSION = "detection-object-graph-v1"

PREDICATES = (
    "left_of",
    "right_of",
    "above",
    "below",
    "on",
    "near",
    "behind",
    "in_front_of",
    "wears",
    "holds",
    "inside",
)
# These concepts cannot be established from axis-aligned 2-D detection boxes.
UNVERIFIED_PREDICATES = frozenset({"on", "behind", "in_front_of", "wears", "holds"})

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6, "bảy": 7,
    "tám": 8, "chín": 9, "mười": 10,
}
_COLORS = {
    "red": "red", "blue": "blue", "green": "green", "yellow": "yellow", "white": "white",
    "black": "black", "orange": "orange", "pink": "pink", "purple": "purple", "brown": "brown",
    "gray": "gray", "grey": "gray",
    "đỏ": "red", "xanh dương": "blue", "xanh lá": "green", "xanh": "blue", "vàng": "yellow",
    "trắng": "white", "đen": "black", "cam": "orange", "hồng": "pink", "tím": "purple",
    "nâu": "brown", "xám": "gray",
}
# Relation phrases, longest first so "to the left of" wins over "of"-less forms.
_RELATION_PHRASES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            "to the left of": "left_of", "left of": "left_of", "on the left of": "left_of",
            "to the right of": "right_of", "right of": "right_of", "on the right of": "right_of",
            "above": "above", "over": "above", "on top of": "on", "on": "on", "atop": "on",
            "below": "below", "under": "below", "underneath": "below", "beneath": "below",
            "next to": "near", "beside": "near", "near": "near", "close to": "near", "by": "near",
            "behind": "behind", "in front of": "in_front_of",
            "wearing": "wears", "wears": "wears",
            "holding": "holds", "holds": "holds", "carrying": "holds",
            "inside": "inside", "in": "inside", "within": "inside",
            "bên trái": "left_of", "phía bên trái": "left_of", "bên phải": "right_of",
            "phía bên phải": "right_of", "phía trên": "above", "bên trên": "above",
            "trên": "on", "phía dưới": "below", "bên dưới": "below", "dưới": "below",
            "bên cạnh": "near", "cạnh": "near", "gần": "near", "kế bên": "near",
            "phía sau": "behind", "đằng sau": "behind", "sau lưng": "behind",
            "phía trước": "in_front_of", "đằng trước": "in_front_of", "trước": "in_front_of",
            "mặc": "wears", "đội": "wears", "đeo": "wears", "cầm": "holds", "bế": "holds",
            "ôm": "holds", "trong": "inside", "bên trong": "inside",
        }.items(),
        key=lambda item: -len(item[0]),
    )
)
# Vietnamese nouns -> English object labels (Open Images vocabulary terms).
VI_OBJECT_LEXICON: Mapping[str, str] = {
    "người đàn ông": "man", "đàn ông": "man", "nam": "man", "người phụ nữ": "woman",
    "phụ nữ": "woman", "cô gái": "girl", "bé gái": "girl", "bé trai": "boy", "cậu bé": "boy",
    "em bé": "baby", "trẻ em": "child", "đứa trẻ": "child", "người": "person",
    "khuôn mặt": "human face", "gương mặt": "human face", "tay": "human hand",
    "xe hơi": "car", "ô tô": "car", "xe ô tô": "car", "xe máy": "motorcycle", "xe đạp": "bicycle",
    "xe buýt": "bus", "xe tải": "truck", "tàu": "boat", "thuyền": "boat", "máy bay": "airplane",
    "chó": "dog", "mèo": "cat", "gà": "chicken", "chim": "bird", "bò": "cattle", "ngựa": "horse",
    "cá": "fish", "chuối": "banana", "táo": "apple", "cam quả": "orange", "cà chua": "tomato",
    "dưa hấu": "watermelon", "rau": "vegetable", "trái cây": "fruit", "hoa": "flower",
    "cây": "tree", "bàn": "table", "ghế": "chair", "giường": "bed", "sofa": "sofa",
    "tủ": "cabinetry", "cửa": "door", "cửa sổ": "window", "tòa nhà": "building", "nhà": "house",
    "đèn": "lamp", "đồng hồ": "clock", "tivi": "television", "ti vi": "television",
    "điện thoại": "mobile phone", "máy tính": "computer monitor", "laptop": "laptop",
    "bàn phím": "computer keyboard", "micro": "microphone", "mic": "microphone",
    "sách": "book", "chai": "bottle", "ly": "glass", "cốc": "mug", "tách": "coffee cup",
    "bát": "bowl", "tô": "bowl", "chén": "bowl", "đĩa": "plate", "dĩa": "plate", "nồi": "pot",
    "chảo": "frying pan", "dao": "knife", "muỗng": "spoon", "thìa": "spoon", "nĩa": "fork",
    "thức ăn": "food", "món ăn": "food", "bánh": "baked goods", "bánh mì": "bread",
    "mũ": "hat", "nón": "hat", "kính": "glasses", "áo": "shirt", "áo sơ mi": "shirt",
    "áo khoác": "jacket", "váy": "dress", "giày": "footwear", "túi": "handbag", "balo": "backpack",
    "cờ": "flag", "biển báo": "traffic sign", "đèn giao thông": "traffic light",
    "bóng": "ball", "đường": "road", "biển": "sea", "núi": "mountain",
}
_PERSON_LABELS = frozenset({"person", "man", "woman", "boy", "girl", "child", "human body"})


def fold(value: str) -> str:
    """Case-fold and NFC-normalize text for lexical matching."""

    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


@dataclass(frozen=True, slots=True)
class FrameRef:
    video_id: str
    keyframe_ordinal: int
    source_frame_idx: int


class ObjectVocabulary:
    """Maps surface mentions to detector class ids."""

    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self._by_name: dict[str, set[int]] = {}
        for index, name in enumerate(self.classes):
            key = fold(name)
            self._by_name.setdefault(key, set()).add(index)
            self._by_name.setdefault(_singular(key), set()).add(index)
        person = {index for index, name in enumerate(self.classes) if fold(name) in _PERSON_LABELS}
        for alias in ("person", "people", "human", "persons"):
            self._by_name.setdefault(alias, set()).update(person)
        for alias in ("man", "men"):
            self._by_name.setdefault(alias, set()).update(
                {index for index, name in enumerate(self.classes) if fold(name) == "man"}
            )
        for alias in ("woman", "women"):
            self._by_name.setdefault(alias, set()).update(
                {index for index, name in enumerate(self.classes) if fold(name) == "woman"}
            )
        self.max_words = max((len(key.split()) for key in self._by_name), default=1)

    def lookup(self, phrase: str) -> frozenset[int]:
        key = fold(phrase)
        found = self._by_name.get(key) or self._by_name.get(_singular(key))
        return frozenset(found or ())


@dataclass(frozen=True, slots=True)
class QueryNode:
    node_id: str
    label: str
    classes: frozenset[int]
    count: int | None = None
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryEdge:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True, slots=True)
class QueryGraph:
    nodes: tuple[QueryNode, ...]
    edges: tuple[QueryEdge, ...]
    unverified: tuple[str, ...] = ()

    @property
    def constraint_count(self) -> int:
        return sum(1 + (node.count is not None) + len(node.attributes)
                   for node in self.nodes) + len(self.edges)

    def to_dict(self, classes: Sequence[str]) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "label": node.label,
                    "classes": sorted(classes[item] for item in node.classes),
                    "count": node.count,
                    "attributes": list(node.attributes),
                }
                for node in self.nodes
            ],
            "edges": [
                {"subject": edge.subject, "predicate": edge.predicate, "object": edge.object}
                for edge in self.edges
            ],
            "unverified": list(self.unverified),
        }


_TOKEN = re.compile(r"[\wÀ-ỹ]+|[^\w\s]", re.UNICODE)


def parse_query_graph(text: str, vocabulary: ObjectVocabulary) -> QueryGraph:
    """Canonicalize English or Vietnamese text into an object query graph."""

    tokens = [fold(item) for item in _TOKEN.findall(unicodedata.normalize("NFC", text))]
    mentions: list[tuple[int, int, str, frozenset[int]]] = []
    position = 0
    vi_max = max(len(key.split()) for key in VI_OBJECT_LEXICON)
    longest = max(vocabulary.max_words, vi_max)
    while position < len(tokens):
        matched = False
        for size in range(min(longest, len(tokens) - position), 0, -1):
            phrase = " ".join(tokens[position : position + size])
            english = VI_OBJECT_LEXICON.get(phrase, phrase)
            classes = vocabulary.lookup(english)
            if classes and phrase not in _COLORS and phrase not in _NUMBER_WORDS:
                if any(phrase == relation for relation, _ in _RELATION_PHRASES):
                    continue
                mentions.append((position, position + size, english, classes))
                position += size
                matched = True
                break
        if not matched:
            position += 1

    nodes: list[QueryNode] = []
    unverified: list[str] = []
    if re.search(r"\b(?:without|no|not|không|chẳng|chưa)\b", fold(text)):
        unverified.append("negation: missing detections do not establish absence")
    if len(mentions) > 8:
        unverified.append("query exceeds eight object nodes; refine the query")
        mentions = mentions[:8]
    for index, (start, end, label, classes) in enumerate(mentions):
        count = None
        window = tokens[max(0, start - 3) : start]
        for token in reversed(window):
            if token.isdigit() and 0 < int(token) <= 50:
                count = int(token)
                break
            if token in _NUMBER_WORDS:
                count = _NUMBER_WORDS[token]
                break
        following_limit = mentions[index + 1][0] if index + 1 < len(mentions) else len(tokens)
        neighborhood = tokens[max(0, start - 2) : start] + tokens[end : min(end + 3, following_limit)]
        colors = []
        joined = " ".join(neighborhood)
        for word, color in sorted(_COLORS.items(), key=lambda item: -len(item[0])):
            if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", joined) and color not in colors:
                colors.append(color)
                joined = joined.replace(word, " ")
        if colors:
            unverified.extend(f"{color} {label}" for color in colors)
        nodes.append(QueryNode(f"n{index}", label, classes, count, tuple(colors)))

    edges: list[QueryEdge] = []
    for index in range(len(mentions) - 1):
        between = " ".join(tokens[mentions[index][1] : mentions[index + 1][0]])
        predicate = None
        for phrase, name in _RELATION_PHRASES:
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", between):
                predicate = name
                break
        if predicate is not None:
            edges.append(QueryEdge(nodes[index].node_id, predicate, nodes[index + 1].node_id))
            if predicate in UNVERIFIED_PREDICATES:
                unverified.append(f"{predicate}: requires visual or caption verification")
    return QueryGraph(tuple(nodes), tuple(edges), tuple(unverified))


def _box_center(box: np.ndarray) -> tuple[float, float]:
    return (float(box[1] + box[3]) / 2.0, float(box[0] + box[2]) / 2.0)


def _area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _intersection(left: np.ndarray, right: np.ndarray) -> float:
    height = min(left[2], right[2]) - max(left[0], right[0])
    width = min(left[3], right[3]) - max(left[1], right[1])
    return max(0.0, float(height)) * max(0.0, float(width))


def _gap(left: np.ndarray, right: np.ndarray) -> float:
    dy = max(0.0, float(max(left[0], right[0]) - min(left[2], right[2])))
    dx = max(0.0, float(max(left[1], right[1]) - min(left[3], right[3])))
    return max(dx, dy)


def predicate_holds(predicate: str, subject: np.ndarray, obj: np.ndarray, margin: float = 0.03) -> bool:
    """Evaluate one predicate over ``[ymin, xmin, ymax, xmax]`` normalized boxes."""

    if predicate in UNVERIFIED_PREDICATES:
        return False
    sx, sy = _box_center(subject)
    ox, oy = _box_center(obj)
    inter = _intersection(subject, obj)
    if predicate == "left_of":
        return sx + margin < ox
    if predicate == "right_of":
        return sx > ox + margin
    if predicate == "above":
        return sy + margin < oy
    if predicate == "below":
        return sy > oy + margin
    if predicate == "near":
        return inter > 0 or _gap(subject, obj) <= 0.10
    if predicate == "inside":
        return _area(subject) > 0 and inter / _area(subject) >= 0.9
    raise ValueError(f"unknown predicate: {predicate}")


@dataclass(slots=True)
class ObjectIndexArtifact:
    batch_id: str
    classes: list[str]
    frames: list[FrameRef]
    det_frame: np.ndarray
    det_class: np.ndarray
    det_score: np.ndarray
    det_box: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


def build_object_index(
    batch_id: str,
    archive_paths: Sequence[str | Path],
    frame_lookup: Callable[[str, int], int | None],
    *,
    video_id_from_member: Callable[[str], str] = lambda value: value,
    min_confidence: float = 0.30,
    max_per_frame: int = 30,
    progress: Callable[[str, int, int], None] | None = None,
) -> ObjectIndexArtifact:
    """Read organizer detection JSON members into compact arrays.

    ``frame_lookup(video_id, keyframe_ordinal)`` returns the source frame index or None
    when the keyframe is unknown (such members are skipped).
    """

    classes: list[str] = []
    class_ids: dict[str, int] = {}
    frames: list[FrameRef] = []
    rows: list[tuple[int, int, float, float, float, float, float]] = []
    if not 0 <= min_confidence <= 1 or max_per_frame < 1:
        raise ValueError("invalid object build bounds")
    seen: set[tuple[str, int]] = set()
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".json")]
            for position, member in enumerate(members, start=1):
                parts = member.replace("\\", "/").split("/")
                if len(parts) < 2:
                    continue
                try:
                    ordinal = int(Path(parts[-1]).stem)
                except ValueError:
                    continue
                video_id = video_id_from_member(parts[-2])
                source = frame_lookup(video_id, ordinal)
                if source is None:
                    continue
                if (video_id, ordinal) in seen:
                    raise ValueError(f"duplicate object frame: {video_id}/{ordinal}")
                seen.add((video_id, ordinal))
                payload = json.loads(archive.read(member).decode("utf-8"))
                scores = [float(item) for item in payload.get("detection_scores", ())]
                names = list(payload.get("detection_class_entities", ()))
                boxes = list(payload.get("detection_boxes", ()))
                if not len(scores) == len(names) == len(boxes):
                    raise ValueError(f"inconsistent detection arrays: {member}")
                kept = []
                for index in np.argsort(-np.asarray(scores, dtype=np.float32)):
                    box = np.asarray(boxes[index], dtype=np.float32)
                    if (not math.isfinite(scores[index]) or not 0 <= scores[index] <= 1
                            or box.shape != (4,) or not np.isfinite(box).all()
                            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                        raise ValueError(f"invalid detection box or confidence: {member}")
                    if scores[index] >= min_confidence and str(names[index]).strip():
                        kept.append(index)
                    if len(kept) >= max_per_frame:
                        break
                if not kept:
                    continue
                frame_row = len(frames)
                frames.append(FrameRef(video_id, ordinal, int(source)))
                for index in kept:
                    name = str(names[index])
                    if name not in class_ids:
                        class_ids[name] = len(classes)
                        classes.append(name)
                    box = [float(value) for value in boxes[index]]
                    rows.append((frame_row, class_ids[name], scores[index], *box))
                if progress is not None and position % 5000 == 0:
                    progress(str(archive_path), position, len(members))
    array = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    return ObjectIndexArtifact(
        batch_id=batch_id,
        classes=classes,
        frames=frames,
        det_frame=array[:, 0].astype(np.int32),
        det_class=array[:, 1].astype(np.int16),
        det_score=array[:, 2].astype(np.float16),
        det_box=array[:, 3:7].astype(np.float32),
    )


def write_object_index(artifact: ObjectIndexArtifact, directory: str | Path) -> dict[str, Any]:
    """Write a new local artifact without replacing existing generated state."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        root / "detections.npz",
        frame=artifact.det_frame,
        cls=artifact.det_class,
        score=artifact.det_score,
        box=artifact.det_box,
    )
    with gzip.open(root / "frames.jsonl.gz", "wt", encoding="utf-8") as handle:
        for frame in artifact.frames:
            handle.write(json.dumps([frame.video_id, frame.keyframe_ordinal, frame.source_frame_idx]) + "\n")
    (root / "classes.json").write_text(json.dumps(artifact.classes, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256()
    for name in ("detections.npz", "frames.jsonl.gz", "classes.json"):
        digest.update((root / name).read_bytes())
    manifest = {
        "version": OBJECT_GRAPH_VERSION,
        "batch_id": artifact.batch_id,
        "frames": len(artifact.frames),
        "detections": int(artifact.det_frame.shape[0]),
        "classes": len(artifact.classes),
        "content_sha256": digest.hexdigest(),
        **artifact.extra,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class ObjectGraphIndex:
    """Searchable detection index for one batch."""

    def __init__(
        self,
        batch_id: str,
        classes: Sequence[str],
        frames: Sequence[FrameRef],
        det_frame: np.ndarray,
        det_class: np.ndarray,
        det_score: np.ndarray,
        det_box: np.ndarray,
        *,
        match_confidence: float = 0.40,
        max_candidates: int = 4_000,
        max_detections_per_node: int = 6,
        beam_width: int = 128,
    ) -> None:
        self.batch_id = batch_id
        self.classes = list(classes)
        self.frames = list(frames)
        arrays = [np.asarray(value) for value in (det_frame, det_class, det_score, det_box)]
        if len({len(value) for value in arrays}) != 1 or arrays[3].shape != (len(arrays[0]), 4):
            raise ValueError("inconsistent object detection arrays")
        if any(value.ndim != 1 for value in arrays[:3]):
            raise ValueError("object frame/class/score arrays must be one-dimensional")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("object detections must be finite")
        if (np.any(arrays[0] != arrays[0].astype(np.int64))
                or np.any(arrays[1] != arrays[1].astype(np.int64))
                or np.any(arrays[0] < 0) or np.any(arrays[0] >= len(self.frames))
                or np.any(arrays[1] < 0) or np.any(arrays[1] >= len(self.classes))):
            raise ValueError("object detections refer to unknown frame/class ids")
        box = arrays[3]
        if (np.any(box < 0) or np.any(box > 1) or np.any(box[:, 0] >= box[:, 2])
                or np.any(box[:, 1] >= box[:, 3]) or np.any(arrays[2] < 0)
                or np.any(arrays[2] > 1)):
            raise ValueError("invalid normalized object boxes or confidences")
        if not 0 <= match_confidence <= 1 or min(max_candidates, max_detections_per_node, beam_width) < 1:
            raise ValueError("invalid object search bounds")
        order = np.argsort(det_frame, kind="stable")
        self.det_frame = np.asarray(det_frame, dtype=np.int64)[order]
        self.det_class = np.asarray(det_class, dtype=np.int64)[order]
        self.det_score = np.asarray(det_score, dtype=np.float32)[order]
        self.det_box = np.asarray(det_box, dtype=np.float32)[order]
        self.match_confidence = float(match_confidence)
        self.max_candidates = int(max_candidates)
        self.max_detections_per_node = int(max_detections_per_node)
        self.beam_width = int(beam_width)
        self.vocabulary = ObjectVocabulary(self.classes)
        boundaries = np.searchsorted(self.det_frame, np.arange(len(self.frames) + 1))
        self._starts = boundaries[:-1]
        self._ends = boundaries[1:]
        self._frame_lookup = {
            (frame.video_id, frame.source_frame_idx): index for index, frame in enumerate(self.frames)
        }

    @classmethod
    def load(cls, directory: str | Path, **kwargs: Any) -> ObjectGraphIndex:
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != OBJECT_GRAPH_VERSION:
            raise ValueError("unsupported object graph version")
        digest = hashlib.sha256()
        for name in ("detections.npz", "frames.jsonl.gz", "classes.json"):
            with (root / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        if digest.hexdigest() != manifest.get("content_sha256"):
            raise ValueError("object graph integrity mismatch; rebuild into a new directory")
        with np.load(root / "detections.npz", allow_pickle=False) as data:
            arrays = (data["frame"], data["cls"], data["score"], data["box"])
        with gzip.open(root / "frames.jsonl.gz", "rt", encoding="utf-8") as handle:
            frames = [FrameRef(str(v), int(o), int(s)) for v, o, s in (json.loads(line) for line in handle)]
        classes = json.loads((root / "classes.json").read_text(encoding="utf-8"))
        return cls(manifest["batch_id"], classes, frames, *arrays, **kwargs)

    def parse(self, text: str) -> QueryGraph:
        return parse_query_graph(text, self.vocabulary)

    def _frame_detections(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, end = int(self._starts[frame]), int(self._ends[frame])
        return self.det_class[start:end], self.det_score[start:end], self.det_box[start:end]

    def evaluate(self, frame: int, query: QueryGraph) -> dict[str, Any]:
        """Return bounded, injective box assignments with explicit unknown constraints."""

        if len(query.nodes) > 8:
            raise ValueError("object queries support at most eight nodes")
        classes, scores, boxes = self._frame_detections(frame)
        options: list[list[int]] = []
        for node in query.nodes:
            mask = np.isin(classes, list(node.classes)) & (scores >= self.match_confidence)
            ranked = [int(item) for item in np.flatnonzero(mask)[np.argsort(-scores[mask])]]
            deduplicated: list[int] = []
            for detection in ranked:
                duplicate = False
                for kept in deduplicated:
                    intersection = _intersection(boxes[detection], boxes[kept])
                    union = _area(boxes[detection]) + _area(boxes[kept]) - intersection
                    if union > 0 and intersection / union >= 0.5:
                        duplicate = True
                        break
                if not duplicate:
                    deduplicated.append(detection)
            options.append(deduplicated)
        index = {node.node_id: position for position, node in enumerate(query.nodes)}
        if len(index) != len(query.nodes) or any(
            edge.subject not in index or edge.object not in index or edge.predicate not in PREDICATES
            for edge in query.edges
        ):
            raise ValueError("invalid object query node ids or relations")

        def edge_states(mapping: Sequence[int | None]) -> list[bool]:
            states = []
            for edge in query.edges:
                left_index, right_index = index[edge.subject], index[edge.object]
                left = mapping[left_index] if left_index < len(mapping) else None
                right = mapping[right_index] if right_index < len(mapping) else None
                states.append(left is not None and right is not None
                              and predicate_holds(edge.predicate, boxes[left], boxes[right]))
            return states

        def objective(mapping: Sequence[int | None]) -> tuple[int, float]:
            nodes = sum(item is not None for item in mapping)
            counts = sum(item is not None and node.count is not None and len(options[pos]) == node.count
                         for pos, (node, item) in enumerate(zip(query.nodes, mapping)))
            confidence = sum(float(scores[item]) for item in mapping if item is not None)
            return nodes + counts + sum(edge_states(mapping)), confidence

        beam: list[tuple[int | None, ...]] = [()]
        for ranked in options:
            following = [(*mapping, choice) for mapping in beam
                         for choice in [*ranked[:self.max_detections_per_node], None]
                         if choice is None or choice not in mapping]
            following.sort(key=objective, reverse=True)
            beam = following[:self.beam_width]
        best_mapping = list(beam[0])
        best_edges = edge_states(best_mapping)
        node_satisfied = [int(item is not None) + int(item is not None and node.count is not None
                          and len(ranked) == node.count)
                          for node, item, ranked in zip(query.nodes, best_mapping, options)]
        satisfied_count = sum(node_satisfied) + sum(best_edges)
        total = max(1, query.constraint_count)
        frame_ref = self.frames[frame]
        evidence_nodes = []
        for node, detection, ranked in zip(query.nodes, best_mapping, options):
            evidence_nodes.append(
                {
                    "id": node.node_id,
                    "label": node.label,
                    "matched": detection is not None,
                    "class": self.classes[int(classes[detection])] if detection is not None else None,
                    "confidence": round(float(scores[detection]), 4) if detection is not None else None,
                    "box": [round(float(value), 4) for value in boxes[detection]] if detection is not None else None,
                    "count": len(ranked),
                    "count_required": node.count,
                    "detection_id": detection,
                    "attributes_verified": False if node.attributes else None,
                }
            )
        return {
            "score": satisfied_count / total,
            "satisfied": satisfied_count,
            "constraints": total,
            "video_id": frame_ref.video_id,
            "keyframe_ordinal": frame_ref.keyframe_ordinal,
            "source_frame_idx": frame_ref.source_frame_idx,
            "evidence": {
                "source": "organizer_detections",
                "box_format": "ymin,xmin,ymax,xmax",
                "count_semantics": "observed detections after overlap suppression",
                "assignment": "bounded_beam",
                "nodes": evidence_nodes,
                "edges": [
                    {"subject": edge.subject, "predicate": edge.predicate, "object": edge.object,
                     "satisfied": ok, "verified": edge.predicate not in UNVERIFIED_PREDICATES}
                    for edge, ok in zip(query.edges, best_edges)
                ],
                "unverified": list(query.unverified),
            },
            "confidence": float(
                np.mean([scores[item] for item in best_mapping if item is not None]) if any(
                    item is not None for item in best_mapping
                ) else 0.0
            ),
        }

    def candidate_frames(self, query: QueryGraph, video_ids: set[str] | None = None,
                         bound: int | None = None) -> np.ndarray:
        """Frames ordered by how many query node types they contain."""

        if not query.nodes:
            return np.zeros(0, dtype=np.int64)
        coverage = np.zeros(len(self.frames), dtype=np.int32)
        strength = np.zeros(len(self.frames), dtype=np.float32)
        confident = self.det_score >= self.match_confidence
        for node in query.nodes:
            mask = np.isin(self.det_class, list(node.classes)) & confident
            frames = np.unique(self.det_frame[mask])
            coverage[frames] += 1
            np.add.at(strength, self.det_frame[mask], self.det_score[mask])
        present = np.flatnonzero(coverage > 0)
        if video_ids is not None:
            present = np.asarray([frame for frame in present if self.frames[frame].video_id in video_ids],
                                 dtype=np.int64)
        order = np.lexsort((-strength[present], -coverage[present]))
        return present[order][: bound or self.max_candidates]

    def search(
        self,
        query: QueryGraph,
        *,
        limit: int = 100,
        video_ids: Iterable[str] | None = None,
        prior: Mapping[tuple[str, int], float] | None = None,
        max_candidates: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank frames by satisfied-constraint fraction.

        ``prior`` maps (video_id, source_frame_idx) to a conventional-retrieval score in
        [0, 1]; these frames are evaluated first within the candidate bound and break ties.
        """

        if not 1 <= limit <= 1000:
            raise ValueError("object search limit must be between 1 and 1000")
        wanted = set(video_ids) if video_ids is not None else None
        bound = min(int(max_candidates or self.max_candidates), self.max_candidates)
        if bound < 1:
            raise ValueError("max_candidates must be positive")
        candidates = [int(item) for item in self.candidate_frames(query, wanted, bound)]
        prior = dict(prior or {})
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in prior.values()):
            raise ValueError("conventional prior scores must be finite and in [0, 1]")
        extra = [self._frame_lookup[key] for key in prior if key in self._frame_lookup]
        seen: set[int] = set()
        results = []
        for frame in [*extra, *candidates]:
            if frame in seen:
                continue
            if wanted is not None and self.frames[frame].video_id not in wanted:
                continue
            if len(seen) >= bound:
                break
            seen.add(frame)
            outcome = self.evaluate(frame, query)
            if outcome["satisfied"] == 0:
                continue
            ref = self.frames[frame]
            outcome["prior"] = float(prior.get((ref.video_id, ref.source_frame_idx), 0.0))
            results.append(outcome)
        results.sort(
            key=lambda item: (-item["score"], -item["prior"], -item["confidence"], item["video_id"], item["source_frame_idx"])
        )
        for rank, item in enumerate(results[:limit], start=1):
            item["rank"] = rank
        return results[:limit]

    def search_with_evidence(self, text: str | QueryGraph, **kwargs: Any) -> dict[str, Any]:
        """Expose partial graph evidence and a reason to use conventional retrieval."""

        query = self.parse(text) if isinstance(text, str) else text
        matches = self.search(query, **kwargs)
        reason = None
        if not query.nodes:
            reason = "no_recognized_objects"
        elif query.unverified:
            reason = "unverified_object_constraints"
        elif not matches or matches[0]["score"] < 1.0:
            reason = "incomplete_object_evidence"
        return {"matches": matches, "query_graph": query.to_dict(self.classes),
                "fallback_reason": reason, "evidence_source": "detections"}

    def frame_objects(self, video_id: str, source_frame_idx: int) -> list[dict[str, Any]]:
        frame = self._frame_lookup.get((video_id, int(source_frame_idx)))
        if frame is None:
            return []
        classes, scores, boxes = self._frame_detections(frame)
        return [
            {
                "class": self.classes[int(cls)],
                "confidence": round(float(score), 4),
                "box": [round(float(value), 4) for value in box],
            }
            for cls, score, box in zip(classes, scores, boxes)
            if score >= self.match_confidence
        ]
