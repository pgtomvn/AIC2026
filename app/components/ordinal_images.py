"""Safe image loading for ordinal-keyed batches, independent of UI state."""
from __future__ import annotations

from typing import Any

from src.ingest.b02_presentation import ordinal_thumbnail_url


def load_ordinal_image(context: Any, item: dict[str, Any]) -> bytes | None:
    """Never fall back to a potentially ambiguous source-frame image."""
    try:
        batch, video, ordinal = item.get("batch_id"), item.get("video_id"), item.get("keyframe_ordinal")
        ordinal_thumbnail_url(batch, video, ordinal)
        repository = getattr(context, "artifact_repository", None)
        resolver = getattr(context, "qa_image_resolver", None)
        getter = getattr(repository, "get_ordinal_frame", None)
        if not callable(getter) or resolver is None:
            return None
        frame = getter(batch, video, ordinal)
        if frame is None or (frame.batch_id, frame.video_id, frame.keyframe_ordinal) != (batch, video, ordinal):
            return None
        data = getattr(resolver.resolve(frame), "data", None)
        return data if isinstance(data, bytes) else None
    except Exception:
        # Missing/unavailable evidence must not show a different canonical image.
        return None
