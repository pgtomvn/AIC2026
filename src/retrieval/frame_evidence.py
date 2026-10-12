"""Read-only, bounded lexical evidence around a canonical frame."""

import json
from typing import Any


def _valid_source_window(begin: object, finish: object, start_ms: int, end_ms: int) -> bool:
    """Keep ordered nonnegative context times with the existing closed overlap.

    Point timestamps remain valid for OCR and legacy ASR navigation records.
    A zero-duration speech record is not evidence of a spoken interval/action.
    """
    return (type(begin) is int and type(finish) is int
            and 0 <= begin <= finish and begin <= end_ms and finish >= start_ms)


def _indexed_segments(
    store: Any, batch_id: str, video_id: str, kind: str,
    start_ms: int, end_ms: int, limit: int,
) -> list[dict[str, Any]]:
    """Read an imported OCR/ASR window without relying on the local ingest table."""
    filters = (
        f"batch_id = {json.dumps(batch_id)} AND video_id = {json.dumps(video_id)} "
        f"AND kind = {json.dumps(kind)} AND start_ms <= {end_ms} AND end_ms >= {start_ms}"
    )
    hits = store.search("", top_k=limit, filters=filters)
    segments: list[dict[str, Any]] = []
    for hit in hits[:limit]:
        payload = hit.get("payload", {})
        if not isinstance(payload, dict) or any(
            payload.get(key) != value
            for key, value in (("batch_id", batch_id), ("video_id", video_id), ("kind", kind))
        ):
            continue
        begin, finish = payload.get("start_ms"), payload.get("end_ms")
        if not _valid_source_window(begin, finish, start_ms, end_ms):
            continue
        text = payload.get("text_nfc") or payload.get("text")
        if not isinstance(text, str):
            continue
        frame_idx = payload.get("source_frame_idx")
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
            frame_idx = None
        segments.append({
            "segment_id": str(payload.get("segment_id") or hit.get("id", "")),
            "kind": kind, "start_ms": begin, "end_ms": finish,
            "text": text[:2048], "source_frame_idx": frame_idx,
        })
    return sorted(segments, key=lambda item: (item["start_ms"], item["end_ms"], item["segment_id"]))


def frame_evidence(
    repository: Any, batch_id: str, video_id: str, source_frame_idx: int, *,
    limit: int = 200, text_stores: dict[str, Any] | None = None,
    keyframe_ordinal: int | None = None,
) -> dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    if isinstance(source_frame_idx, bool) or source_frame_idx < 0:
        raise ValueError("source_frame_idx must be nonnegative")
    getter = getattr(repository, "get_source_frame", None) or repository.get_frame
    if batch_id == "aic26-b2":
        from src.ingest.frame_identity import resolve_image_frame

        frame = resolve_image_frame(repository, {"batch_id": batch_id, "video_id": video_id,
                                   "source_frame_idx": source_frame_idx, "keyframe_ordinal": keyframe_ordinal})
    else:
        frame = getter(batch_id, video_id, source_frame_idx)
    if frame is None:
        raise LookupError("canonical_frame_not_found")
    start, end = max(0, frame.timestamp_ms - 30_000), frame.timestamp_ms + 30_000
    segments: list[dict[str, Any]] = []
    # Reserve space for both channels; never read or return over 200 segments total.
    for kind, budget in (("asr", (limit + 1) // 2), ("ocr", limit // 2)):
        if not budget:
            continue
        indexed = _indexed_segments(text_stores[kind], batch_id, video_id, kind, start, end, budget) \
            if text_stores and kind in text_stores else []
        if indexed:
            segments.extend(indexed)
            continue
        for segment in repository.text_segments_in_range(
            batch_id, video_id, start_ms=start, end_ms=end, kind=kind,
            index_ready_only=True, limit=budget,
        )[:budget]:
            # Canonical TextSegment validates these fields, but optional
            # repository adapters must not bypass the same context boundary.
            if any(getattr(segment, key, value) != value for key, value in
                   (("batch_id", batch_id), ("video_id", video_id), ("kind", kind))):
                continue
            begin, finish = getattr(segment, "start_ms", None), getattr(segment, "end_ms", None)
            if not _valid_source_window(begin, finish, start, end):
                continue
            text = getattr(segment, "text_nfc", None)
            if not isinstance(text, str):
                continue
            frame_idx = getattr(segment, "source_frame_idx", None)
            if type(frame_idx) is not int or frame_idx < 0:
                frame_idx = None
            segments.append({
                "segment_id": segment.segment_id, "kind": kind,
                "start_ms": begin, "end_ms": finish,
                "text": text[:2048], "source_frame_idx": frame_idx,
            })
    return {
        "anchor": {"batch_id": batch_id, "video_id": video_id,
                   "source_frame_idx": source_frame_idx, "timestamp_ms": frame.timestamp_ms,
                   **({"keyframe_ordinal": keyframe_ordinal} if batch_id == "aic26-b2" else {})},
        "start_ms": start, "end_ms": end, "segments": segments,
        "limit": limit, "speech_is_not_visual_confirmation": True,
    }
