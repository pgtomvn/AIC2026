"""Rank fusion and temporal diversification primitives for Textual KIS."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChannelRanking:
    """One named, weighted ranking returned by a retrieval channel."""

    name: str
    results: tuple[dict[str, Any], ...]
    weight: float = 1.0


def canonical_key(result: Mapping[str, Any]) -> tuple[str, int]:
    """Extract the non-ambiguous frame key from a channel result."""

    payload = result.get("payload")
    source = payload if isinstance(payload, Mapping) else result
    video_id = source.get("video_id")
    frame = source.get("source_frame_idx")
    if frame is None:
        # Compatibility is intentionally input-only.  Returned records always
        # expose source_frame_idx.
        frame = source.get("frame_id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("retrieval result is missing video_id")
    if isinstance(frame, bool):
        raise ValueError("source_frame_idx must be an integer")
    try:
        frame_int = int(frame)
    except (TypeError, ValueError) as error:
        raise ValueError("retrieval result is missing source_frame_idx") from error
    if frame_int < 0:
        raise ValueError("source_frame_idx cannot be negative")
    return video_id, frame_int


def _flatten(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = result.get("payload")
    flattened = dict(payload) if isinstance(payload, Mapping) else {}
    flattened.update({key: value for key, value in result.items() if key != "payload"})
    video_id, source_frame_idx = canonical_key(result)
    flattened["video_id"] = video_id
    flattened["source_frame_idx"] = source_frame_idx
    flattened.pop("frame_id", None)
    return flattened


def weighted_rrf(
    rankings: Iterable[ChannelRanking],
    *,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse heterogeneous rankings using weighted Reciprocal Rank Fusion.

    Duplicate frame keys within one channel contribute only their best rank.
    Raw cosine/lexical scores are retained as evidence but never combined.
    """

    if k < 0:
        raise ValueError("RRF k must be non-negative")
    scores: dict[tuple[str, int], float] = defaultdict(float)
    records: dict[tuple[str, int], dict[str, Any]] = {}
    channel_ranks: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    evidence: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for ranking in rankings:
        if ranking.weight < 0:
            raise ValueError("channel weights must be non-negative")
        seen: set[tuple[str, int]] = set()
        for rank, result in enumerate(ranking.results, start=1):
            key = canonical_key(result)
            if key in seen:
                continue
            seen.add(key)
            scores[key] += ranking.weight / (k + rank)
            channel_ranks[key][ranking.name] = rank
            if key not in records:
                records[key] = _flatten(result)
            source = result.get("payload")
            source = source if isinstance(source, Mapping) else result
            evidence_item = {
                "channel": ranking.name,
                "rank": rank,
                "raw_score": result.get("score", result.get("_rankingScore")),
            }
            # Keep compact lexical provenance needed by the UI.  In
            # particular, OCR polygons must survive frame resolution and RRF
            # instead of existing only in the backing text document.
            for field in ("segment_id", "kind", "text_raw", "polygons"):
                if field in source:
                    evidence_item[field] = source[field]
            evidence[key].append(evidence_item)

    fused: list[dict[str, Any]] = []
    for key, score in scores.items():
        item = records[key]
        item["score"] = float(score)
        item["channel_ranks"] = channel_ranks[key]
        item["evidence"] = evidence[key]
        fused.append(item)
    fused.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["video_id"]),
            int(item["source_frame_idx"]),
        )
    )
    return fused


def aggregate_temporal_windows(
    candidates: list[dict[str, Any]],
    *,
    window_ms: int = 4_000,
) -> list[dict[str, Any]]:
    """Aggregate nearby evidence in real time without crossing shot boundaries."""

    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    timeless: list[dict[str, Any]] = []
    for item in candidates:
        timestamp = item.get("timestamp_ms")
        if timestamp is None:
            timeless.append(dict(item))
            continue
        shot = str(item.get("shot_id", f"frame:{item['source_frame_idx']}"))
        groups[(str(item["video_id"]), shot)].append(dict(item))

    aggregated: list[dict[str, Any]] = []
    half = window_ms / 2
    for group in groups.values():
        group.sort(key=lambda item: int(item["timestamp_ms"]))
        for anchor in group:
            nearby = [
                item
                for item in group
                if abs(int(item["timestamp_ms"]) - int(anchor["timestamp_ms"])) <= half
            ]
            ranked = sorted(nearby, key=lambda item: float(item["score"]), reverse=True)
            top_scores = [float(item["score"]) for item in ranked[:3]]
            representative = dict(ranked[0])
            representative["score"] = max(top_scores) + sum(top_scores) / len(top_scores)
            representative["window_start_ms"] = max(0, int(anchor["timestamp_ms"] - half))
            representative["window_end_ms"] = int(anchor["timestamp_ms"] + half)
            representative["window_evidence_count"] = len(nearby)
            aggregated.append(representative)
    aggregated.extend(timeless)

    # Multiple anchors can select the same representative; keep its strongest
    # window score before diversity filtering.
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for item in aggregated:
        key = canonical_key(item)
        if key not in unique or float(item["score"]) > float(unique[key]["score"]):
            unique[key] = item
    return sorted(
        unique.values(),
        key=lambda item: (
            -float(item["score"]),
            str(item["video_id"]),
            int(item["source_frame_idx"]),
        ),
    )


def diversify(
    candidates: list[dict[str, Any]],
    *,
    limit: int = 100,
    min_separation_ms: int = 3_000,
    early_cutoff: int = 20,
    early_per_video: int = 2,
    late_per_video: int = 5,
) -> list[dict[str, Any]]:
    """Apply temporal NMS, uniqueness, and rank-band per-video quotas."""

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    per_video: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        key = canonical_key(candidate)
        if key in seen:
            continue
        video_id = key[0]
        quota = early_per_video if len(selected) < early_cutoff else late_per_video
        if per_video[video_id] >= quota:
            continue
        timestamp = candidate.get("timestamp_ms")
        shot = candidate.get("shot_id")
        too_close = False
        if timestamp is not None:
            for previous in selected:
                if previous["video_id"] != video_id or previous.get("timestamp_ms") is None:
                    continue
                same_shot = (
                    shot is None
                    or previous.get("shot_id") is None
                    or previous.get("shot_id") == shot
                )
                if (
                    same_shot
                    and abs(int(previous["timestamp_ms"]) - int(timestamp)) < min_separation_ms
                ):
                    too_close = True
                    break
        if too_close:
            continue
        output = dict(candidate)
        output["rank"] = len(selected) + 1
        selected.append(output)
        seen.add(key)
        per_video[video_id] += 1
        if len(selected) >= limit:
            break
    return selected
