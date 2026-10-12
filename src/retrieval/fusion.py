"""Rank fusion and temporal diversification primitives for Textual KIS."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

FrameIdentity = tuple[str, int] | tuple[str, str, int]
VideoIdentity = str | tuple[str, str]
_B02_BATCH_ID = "aic26-b2"


@dataclass(frozen=True)
class ChannelRanking:
    """One named, weighted ranking returned by a retrieval channel."""

    name: str
    results: tuple[dict[str, Any], ...]
    weight: float = 1.0


def canonical_key(result: Mapping[str, Any]) -> tuple[str, int]:
    """Extract video ID and source-frame index for legacy callers/output."""

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


def frame_identity_key(result: Mapping[str, Any]) -> FrameIdentity:
    """Use B02 ordinal identity without changing the legacy frame key contract.

    Organizer ``source_frame_idx`` is advisory for B02 traffic video and can
    repeat. B01 keeps its original ``(video_id, source_frame_idx)`` key.
    """

    video_id, source_frame_idx = canonical_key(result)
    payload = result.get("payload")
    source = payload if isinstance(payload, Mapping) else result
    batch_id = source.get("batch_id", result.get("batch_id"))
    if batch_id != _B02_BATCH_ID:
        return video_id, source_frame_idx
    ordinal = source.get("keyframe_ordinal", result.get("keyframe_ordinal"))
    if isinstance(ordinal, bool) or not (
        isinstance(ordinal, int)
        or (isinstance(ordinal, str) and ordinal.isdecimal())
    ):
        raise ValueError("B02 retrieval result requires a positive keyframe_ordinal")
    ordinal_int = int(ordinal)
    if ordinal_int < 1:
        raise ValueError("B02 retrieval result requires a positive keyframe_ordinal")
    return _B02_BATCH_ID, video_id, ordinal_int


def _video_identity(result: Mapping[str, Any]) -> VideoIdentity:
    frame_identity = frame_identity_key(result)
    if len(frame_identity) == 3:
        return frame_identity[0], frame_identity[1]
    return frame_identity[0]


def _tie_breaker(item: Mapping[str, Any]) -> tuple[str, int, int]:
    video_id, source_frame_idx = canonical_key(item)
    identity = frame_identity_key(item)
    return video_id, source_frame_idx, identity[2] if len(identity) == 3 else -1


def _flatten(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = result.get("payload")
    flattened = dict(payload) if isinstance(payload, Mapping) else {}
    flattened.update({key: value for key, value in result.items() if key != "payload"})
    video_id, source_frame_idx = canonical_key(result)
    flattened["video_id"] = video_id
    flattened["source_frame_idx"] = source_frame_idx
    identity = frame_identity_key(result)
    if len(identity) == 3:
        flattened["batch_id"] = identity[0]
        flattened["keyframe_ordinal"] = identity[2]
    flattened.pop("frame_id", None)
    return flattened


def weighted_rrf(
    rankings: Iterable[ChannelRanking],
    *,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse heterogeneous rankings using weighted Reciprocal Rank Fusion.

    Duplicate frame identities within one channel contribute only their best rank.
    Raw cosine/lexical scores are retained as evidence but never combined.
    """

    if k < 0:
        raise ValueError("RRF k must be non-negative")
    scores: dict[FrameIdentity, float] = defaultdict(float)
    records: dict[FrameIdentity, dict[str, Any]] = {}
    channel_ranks: dict[FrameIdentity, dict[str, int]] = defaultdict(dict)
    evidence: dict[FrameIdentity, list[dict[str, Any]]] = defaultdict(list)

    for ranking in rankings:
        if ranking.weight < 0:
            raise ValueError("channel weights must be non-negative")
        seen: set[FrameIdentity] = set()
        for rank, result in enumerate(ranking.results, start=1):
            key = frame_identity_key(result)
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
            nested_ranks = result.get("channel_ranks")
            if isinstance(nested_ranks, Mapping):
                evidence_item["nested_channel_ranks"] = {
                    str(name): int(value) for name, value in nested_ranks.items()
                }
            # Keep compact lexical provenance needed by the UI.  In
            # particular, OCR polygons must survive frame resolution and RRF
            # instead of existing only in the backing text document.
            for field in (
                "segment_id",
                "kind",
                "text_raw",
                "start_ms",
                "end_ms",
                "confidence",
                "model_version",
                "polygons",
                "temporal_steps",
            ):
                if field in source:
                    evidence_item[field] = source[field]
            matches = result.get("entity_matches", source.get("entity_matches"))
            if isinstance(matches, (list, tuple)):
                evidence_item["entity_matches"] = [
                    dict(value) for value in matches if isinstance(value, Mapping)
                ]
            evidence[key].append(evidence_item)

    fused: list[dict[str, Any]] = []
    for key, score in scores.items():
        item = records[key]
        item["score"] = float(score)
        item["channel_ranks"] = channel_ranks[key]
        item["evidence"] = evidence[key]
        entity_matches: dict[str, dict[str, Any]] = {}
        for value in evidence[key]:
            for match in value.get("entity_matches", ()):
                identity = match.get("identity") if isinstance(match, Mapping) else None
                if isinstance(identity, str) and identity:
                    entity_matches.setdefault(identity, dict(match))
        if entity_matches:
            item["entity_matches"] = list(entity_matches.values())
        fused.append(item)
    fused.sort(key=lambda item: (-float(item["score"]), *_tie_breaker(item)))
    return fused


def aggregate_temporal_windows(
    candidates: list[dict[str, Any]],
    *,
    window_ms: int = 4_000,
) -> list[dict[str, Any]]:
    """Aggregate nearby evidence in real time without crossing shot boundaries."""

    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    groups: dict[tuple[VideoIdentity, str], list[dict[str, Any]]] = defaultdict(list)
    timeless: list[dict[str, Any]] = []
    for item in candidates:
        timestamp = item.get("timestamp_ms")
        if timestamp is None:
            timeless.append(dict(item))
            continue
        groups[(_video_identity(item), str(item.get("shot_id") or ""))].append(
            dict(item)
        )

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
            # Distinct B02 images may share both advisory frame_idx and PTS.
            # Keep each equal-time anchor as a distinct candidate rather than
            # repeatedly selecting only the highest-scoring image.
            same_time_collision = (
                len(frame_identity_key(anchor)) == 3
                and any(
                    int(item["timestamp_ms"]) == int(anchor["timestamp_ms"])
                    and frame_identity_key(item) != frame_identity_key(anchor)
                    for item in nearby
                )
            )
            representative = dict(anchor if same_time_collision else ranked[0])
            representative["score"] = max(top_scores) + sum(top_scores) / len(top_scores)
            representative["window_start_ms"] = max(0, int(anchor["timestamp_ms"] - half))
            representative["window_end_ms"] = int(anchor["timestamp_ms"] + half)
            representative["window_evidence_count"] = len(nearby)
            merged_evidence: list[dict[str, Any]] = []
            for item in nearby:
                for raw_evidence in item.get("evidence", ()):
                    if not isinstance(raw_evidence, Mapping):
                        continue
                    evidence_item = dict(raw_evidence)
                    evidence_item["timestamp_ms"] = int(item["timestamp_ms"])
                    merged_evidence.append(evidence_item)
            if merged_evidence:
                representative["evidence"] = merged_evidence
            entity_matches: dict[str, dict[str, Any]] = {}
            for item in nearby:
                for match in item.get("entity_matches", ()):
                    identity = match.get("identity") if isinstance(match, Mapping) else None
                    if isinstance(identity, str) and identity:
                        entity_matches.setdefault(identity, dict(match))
            if entity_matches:
                representative["entity_matches"] = list(entity_matches.values())
            aggregated.append(representative)
    aggregated.extend(timeless)

    # Multiple anchors can select the same representative; keep its strongest
    # window score before diversity filtering.
    unique: dict[FrameIdentity, dict[str, Any]] = {}
    for item in aggregated:
        key = frame_identity_key(item)
        if key not in unique or float(item["score"]) > float(unique[key]["score"]):
            unique[key] = item
    return sorted(
        unique.values(),
        key=lambda item: (-float(item["score"]), *_tie_breaker(item)),
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
    seen: set[FrameIdentity] = set()
    per_video: dict[VideoIdentity, int] = defaultdict(int)
    for candidate in candidates:
        key = frame_identity_key(candidate)
        if key in seen:
            continue
        video_id = _video_identity(candidate)
        quota = early_per_video if len(selected) < early_cutoff else late_per_video
        if per_video[video_id] >= quota:
            continue
        timestamp = candidate.get("timestamp_ms")
        shot = candidate.get("shot_id")
        too_close = False
        if timestamp is not None:
            for previous in selected:
                if (
                    _video_identity(previous) != video_id
                    or previous.get("timestamp_ms") is None
                ):
                    continue
                if (
                    len(key) == 3
                    and int(previous["timestamp_ms"]) == int(timestamp)
                    and frame_identity_key(previous) != key
                ):
                    # Equal PTS does not prove these B02 JPEGs are duplicates.
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
