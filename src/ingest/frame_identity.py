"""Explicit canonical image identity; never infer ordinal from a source index."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def value(frame: Any, name: str) -> Any:
    return frame.get(name) if isinstance(frame, Mapping) else getattr(frame, name, None)


def image_identity(frame: Any) -> tuple[str, str, int]:
    """B02 is ordinal keyed; all existing batches retain source coordinates."""
    batch, video = value(frame, "batch_id"), value(frame, "video_id")
    if not isinstance(batch, str) or not isinstance(video, str) or not video:
        raise ValueError("invalid image batch/video")
    ordinal = batch == "aic26-b2"
    coordinate = value(frame, "keyframe_ordinal" if ordinal else "source_frame_idx")
    if isinstance(coordinate, bool) or not isinstance(coordinate, int) or coordinate < int(ordinal):
        raise ValueError("canonical image coordinate is missing or invalid")
    return batch, video, coordinate


def candidate_identity(frame: Any) -> str:
    batch, video, coordinate = image_identity(frame)
    return f"{batch}:{video}:ordinal:{coordinate}" if batch == "aic26-b2" else f"{batch}:{video}:{coordinate}"


def resolve_image_frame(repository: Any, reference: Any) -> Any:
    """Resolve the exact image and reject stale source coordinates for B02."""
    batch, video, coordinate = image_identity(reference)
    getter = repository.get_ordinal_frame if batch == "aic26-b2" else repository.get_frame
    frame = getter(batch, video, coordinate)
    if frame is not None and batch == "aic26-b2":
        source = value(reference, "source_frame_idx")
        if source is not None and (
            isinstance(source, bool) or not isinstance(source, int)
            or source != frame.source_frame_idx
        ):
            raise ValueError("source coordinate does not match canonical ordinal")
        if image_identity(frame) != (batch, video, coordinate):
            raise ValueError("canonical identity mismatch")
    return frame
