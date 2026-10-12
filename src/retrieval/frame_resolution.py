"""Request-scoped canonical metadata batching; ranking evidence is never cached."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class BatchedFrameResolver:
    """Resolve text anchors using bounded repository reads, preserving legacy semantics."""

    def __init__(self, repository: Any, *, scalar_fallback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None) -> None:
        self.repository = repository
        self.scalar_fallback = scalar_fallback
        self._local = threading.local()

    def set_request_mode(self, mode: str) -> None:
        """A supplied legacy fallback limits batching to explicit ASR-only queries."""
        self._local.enabled = self.scalar_fallback is None or mode == "asr_only"

    def _enabled(self) -> bool:
        return getattr(self._local, "enabled", self.scalar_fallback is None)

    def begin_request(self) -> None:
        """Clear per-thread metadata, including negative lookups, between requests."""
        self._local.frames = {}
        self._local.nearest = {}

    def _ensure_cache(self) -> None:
        if not hasattr(self._local, "frames"):
            self.begin_request()

    @staticmethod
    def _identity(result: Mapping[str, Any]) -> tuple[str, str, int | None] | None:
        payload = result.get("payload", result)
        batch, video = payload.get("batch_id"), payload.get("video_id")
        if not isinstance(batch, str) or not isinstance(video, str):
            return None
        if batch == "aic26-b2":
            return None  # handled by the explicit ordinal path, never the source cache
        raw = payload.get("source_frame_idx")
        try:
            frame = -1 if isinstance(raw, bool) else int(raw)
        except (TypeError, ValueError):
            frame = -1
        return batch, video, frame if frame >= 0 else None

    def prefetch(self, results: Sequence[dict[str, Any]]) -> None:
        """Fetch only missing candidate identities in chunks of at most 100."""
        if not self._enabled():
            return
        self._ensure_cache()
        grouped: dict[str, dict[tuple[str, int], None]] = defaultdict(dict)
        for result in results:
            identity = self._identity(result)
            if identity is None or identity[2] is None or identity in self._local.frames:
                continue
            batch, video, frame = identity
            grouped[batch][(video, frame)] = None
        for batch, identities in grouped.items():
            keys = list(identities)
            for offset in range(0, len(keys), 100):
                chunk = keys[offset : offset + 100]
                found = self.repository.get_frames(batch, chunk)
                for video, frame in chunk:
                    self._local.frames[(batch, video, frame)] = found.get((video, frame))

    def __call__(self, result: dict[str, Any]) -> dict[str, Any] | None:
        payload = result.get("payload", result)
        if payload.get("batch_id") == "aic26-b2":
            from src.ingest.frame_identity import resolve_image_frame

            try:
                frame = resolve_image_frame(self.repository, payload)
            except ValueError:
                return None
            return {**result, "payload": {**payload, **frame.to_dict()}} if frame else None
        if not self._enabled():
            return self.scalar_fallback(result)
        self._ensure_cache()
        identity = self._identity(result)
        if identity is None:
            return None
        batch, video, frame_idx = identity
        frame = None
        if frame_idx is not None:
            if identity not in self._local.frames:
                self.prefetch([result])
            frame = self._local.frames[identity]
        payload = result.get("payload", result)
        if frame is None:
            try:
                start = int(payload.get("start_ms", payload.get("timestamp_ms", 0)))
                end = int(payload.get("end_ms", start))
            except (TypeError, ValueError):
                return None
            key = (batch, video, (start + end) // 2)
            if key not in self._local.nearest:
                self._local.nearest[key] = self.repository.nearest_frame(*key)
            frame = self._local.nearest[key]
        if frame is None:
            return None
        mapped_payload = dict(payload)
        mapped_payload.update(frame.to_dict())
        mapped = dict(result)
        mapped["payload"] = mapped_payload
        return mapped
