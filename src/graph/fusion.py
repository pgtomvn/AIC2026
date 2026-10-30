"""Unified ranking: conventional retrieval plus ASR-topic and object-graph evidence.

Every list contributes a weighted reciprocal-rank term (``w / (k + rank)``):

* conventional hits (visual, OCR, ASR, and metadata channels already fused by the pipeline);
* topic support: a frame whose time lies inside a matched transcript topic of the same video
  receives that topic's rank; the topic keyframe itself is added when no conventional frame
  falls inside it;
* object support: a frame ranked by the object graph contributes ``w * fraction / (k + rank)``
  where ``fraction`` is the share of query constraints it satisfies.

A frame supported by several sources rises above one supported by a single source; graph
evidence is attached to the hit so the reason for its rank stays inspectable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """Weights of the unified ranking; defaults are tuned on the train/validation split."""

    k: float = 60.0
    conventional: float = 1.0
    topic: float = 0.8
    object: float = 0.6
    topic_padding_ms: int = 2_000
    topic_matches: int = 20
    graph_only_limit: int = 10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> FusionWeights:
        allowed = set(cls.__dataclass_fields__)
        weights = cls(**{key: item for key, item in dict(value or {}).items() if key in allowed})
        if weights.k <= 0 or min(weights.conventional, weights.topic, weights.object) < 0:
            raise ValueError("fusion weights must be nonnegative and k positive")
        return weights


def _key(hit: Mapping[str, Any]) -> tuple[Any, ...]:
    return (hit.get("batch_id"), hit.get("video_id"), hit.get("source_frame_idx"), hit.get("keyframe_ordinal"))


def fuse(
    conventional: Sequence[Mapping[str, Any]],
    topics: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]] = (),
    *,
    weights: FusionWeights = FusionWeights(),
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return one ranked list.

    ``topics`` are topic-graph hits in rank order; each carries ``graph_evidence`` with
    ``start_ms``/``end_ms``. ``objects`` are object-graph hits in rank order with
    ``graph_evidence`` and a ``fraction`` of satisfied constraints.
    """

    k = weights.k
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    scores: dict[tuple[Any, ...], float] = {}
    sources: dict[tuple[Any, ...], set[str]] = {}

    def add(hit: Mapping[str, Any], term: float, source: str) -> tuple[Any, ...]:
        key = _key(hit)
        if key not in rows:
            rows[key] = dict(hit)
            scores[key] = 0.0
            sources[key] = set()
        scores[key] += term
        sources[key].add(source)
        return key

    for rank, hit in enumerate(conventional, 1):
        add(hit, weights.conventional / (k + rank), "conventional")

    by_video: dict[tuple[Any, Any], list[tuple[Any, ...]]] = {}
    for key, row in rows.items():
        by_video.setdefault((row.get("batch_id"), row.get("video_id")), []).append(key)

    graph_only = 0
    for rank, topic in enumerate(topics[: weights.topic_matches], 1):
        evidence = topic.get("graph_evidence") or {}
        start, end = evidence.get("start_ms"), evidence.get("end_ms")
        term = weights.topic / (k + rank)
        inside = []
        if isinstance(start, int) and isinstance(end, int):
            low, high = start - weights.topic_padding_ms, end + weights.topic_padding_ms
            inside = [key for key in by_video.get((topic.get("batch_id"), topic.get("video_id")), ())
                      if isinstance(rows[key].get("timestamp_ms"), int) and low <= rows[key]["timestamp_ms"] <= high]
        if inside:
            for key in inside:
                if "topic" in sources[key]:
                    continue  # one topic term per frame: the best-ranked supporting topic
                scores[key] += term
                sources[key].add("topic")
                rows[key].setdefault("graph_evidence", evidence)
        elif graph_only < weights.graph_only_limit:
            key = add(topic, term, "topic")
            rows[key]["graph_evidence"] = evidence
            graph_only += 1

    for rank, hit in enumerate(objects, 1):
        fraction = float(hit.get("fraction", 0.0))
        if fraction <= 0:
            continue
        key = _key(hit)
        if key not in rows and graph_only >= 2 * weights.graph_only_limit:
            continue
        if key not in rows:
            graph_only += 1
        key = add(hit, weights.object * fraction / (k + rank), "object")
        if hit.get("graph_evidence"):
            rows[key]["object_evidence"] = hit["graph_evidence"]

    ordered = sorted(rows, key=lambda key: (-scores[key], str(key)))[:limit]
    result = []
    for rank, key in enumerate(ordered, 1):
        row = rows[key]
        row["rank"] = rank
        row["fused_score"] = round(scores[key], 6)
        row["sources"] = sorted(sources[key])
        result.append(row)
    return result


def topic_keys(hits: Iterable[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
    """Identity keys of hits, used by tests and diagnostics."""
    return {_key(hit) for hit in hits}
