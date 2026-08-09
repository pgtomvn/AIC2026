"""Single-flight handoff between CUDA retrieval encoders and local VLMs."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import Any


class GPUModelCoordinator:
    """Coordinate heavy local models without unloading CPU query encoders."""

    def __init__(
        self,
        *,
        dense_channels: Iterable[Any],
        local_reasoner: Any,
        warmup: Any = None,
        clear_threshold: float = 0.80,
        monotonic: Any = time.monotonic,
    ) -> None:
        if not 0.0 < float(clear_threshold) <= 1.0:
            raise ValueError("clear_threshold must be in (0, 1]")
        self.dense_channels = tuple(dense_channels)
        self.local_reasoner = local_reasoner
        self.warmup = warmup
        self.clear_threshold = float(clear_threshold)
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._last_metrics: dict[str, int | float | bool] = {}

    @staticmethod
    def _encoder_uses_cuda(encoder: Any) -> bool:
        device = getattr(encoder, "device", None)
        if device is None:
            device = getattr(encoder, "_device", None)
        return device is not None and str(device).casefold().startswith("cuda")

    def _cuda_channels(self, batch_id: str | None = None) -> tuple[Any, ...]:
        preferred = None
        if isinstance(batch_id, str) and batch_id:
            preferred = "organizer_clip" if "reference" in batch_id.casefold() else "visual"
        return tuple(
            channel
            for channel in self.dense_channels
            if preferred is None or str(getattr(channel, "name", "")) == preferred
            if self._encoder_uses_cuda(getattr(channel, "encoder", None))
        )

    @property
    def last_metrics(self) -> dict[str, int | float | bool]:
        with self._lock:
            return dict(self._last_metrics)

    def before_vlm_inference(
        self, batch_id: str | None = None
    ) -> dict[str, int | float | bool]:
        """Unload CUDA encoders and conditionally clear the PyTorch allocator."""

        with self._lock:
            started = self._monotonic()
            evicted: list[str] = []
            for channel in self._cuda_channels(batch_id):
                encoder = getattr(channel, "encoder", None)
                unload = getattr(encoder, "unload", None)
                if callable(unload):
                    unload()
                    evicted.append(str(getattr(channel, "name", "cuda_dense")))

            used_ratio: float | None = None
            cache_cleared = False
            try:
                import torch

                if torch.cuda.is_available():
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    if total_bytes > 0:
                        used_ratio = 1.0 - (float(free_bytes) / float(total_bytes))
                    if used_ratio is not None and used_ratio >= self.clear_threshold:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        cache_cleared = True
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                used_ratio = None
                cache_cleared = False

            if evicted and self.warmup is not None:
                marker = getattr(self.warmup, "mark_evicted", None)
                if callable(marker):
                    marker()
            metrics: dict[str, int | float | bool] = {
                "handoff_ms": round((self._monotonic() - started) * 1_000, 3),
                "cuda_encoders_unloaded": len(evicted),
                "cuda_cache_cleared": cache_cleared,
            }
            if used_ratio is not None:
                metrics["vram_used_ratio_before_clear"] = round(used_ratio, 6)
            self._last_metrics = metrics
            return dict(metrics)

    def release_vlm_for_retrieval(self, batch_id: str) -> bool:
        """Release Qwen only when the selected batch needs a CUDA query encoder."""

        with self._lock:
            if not self._cuda_channels(batch_id):
                return False
            unload = getattr(self.local_reasoner, "unload", None)
            if callable(unload):
                unload()
                return True
            return False

    def release_local_vlm(self) -> None:
        """Release the local VLM before a general KIS/TRAKE workflow."""

        with self._lock:
            unload = getattr(self.local_reasoner, "unload", None)
            if callable(unload):
                unload()


__all__ = ["GPUModelCoordinator"]
