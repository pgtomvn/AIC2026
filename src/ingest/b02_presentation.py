"""Explicit ordinal image identity and submission coordinates for Batch 2."""
from __future__ import annotations

import re
from typing import Any

from src.ingest.b02_keyframe_mapping import is_vfr_traffic_video


def ordinal_thumbnail_url(batch_id: str, video_id: str, ordinal: int) -> str:
    """Build a B02 image URL without converting ordinal into source-frame index."""
    if batch_id != "aic26-b2" or not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("invalid B02 identity")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ValueError("keyframe ordinal must be a positive integer")
    return f"/v1/keyframes/{batch_id}/{video_id}/{ordinal}/image"


def present_ordinal_frame(frame: Any) -> dict[str, Any]:
    """Return safe canonical coordinates; no paths or inferred CFR timestamps."""
    url = ordinal_thumbnail_url(frame.batch_id, frame.video_id, frame.keyframe_ordinal)
    for value in (frame.source_frame_idx, frame.timestamp_ms):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid canonical frame coordinates")
    traffic = is_vfr_traffic_video(frame.video_id)
    return {
        "batch_id": frame.batch_id,
        "video_id": frame.video_id,
        "keyframe_ordinal": frame.keyframe_ordinal,
        "source_frame_idx": frame.source_frame_idx,
        "timestamp_ms": frame.timestamp_ms,
        "frame_semantics": "keyframe_ordinal",
        "thumbnail_url": url,
        "submission_unit": "timestamp_ms" if traffic else "source_frame_idx",
        "submission_value": frame.timestamp_ms if traffic else frame.source_frame_idx,
        "trake_eligible": not traffic and frame.video_id != "M08_V001",
    }


def ordinal_neighborhood(repository: Any, batch_id: str, video_id: str,
                         ordinal: int, radius: int = 5) -> dict[str, Any]:
    """Read at most eleven exact ordinals; preserve equal-PTS neighboring images."""
    ordinal_thumbnail_url(batch_id, video_id, ordinal)
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 5:
        raise ValueError("ordinal neighborhood radius must be between zero and five")
    anchor = repository.get_ordinal_frame(batch_id, video_id, ordinal)
    if anchor is None:
        raise LookupError("canonical keyframe not found")
    neighbors = []
    for index in range(max(1, ordinal - radius), ordinal + radius + 1):
        frame = anchor if index == ordinal else repository.get_ordinal_frame(batch_id, video_id, index)
        if frame is not None:
            neighbors.append({**present_ordinal_frame(frame), "offset": index - ordinal})
    return {"anchor": present_ordinal_frame(anchor), "neighbors": neighbors}
