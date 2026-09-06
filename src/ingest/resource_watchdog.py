"""Fail-closed process/GPU resource budget checks for long ingestion runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.warmup import safe_resource_snapshot


class ResourceBudgetExceeded(RuntimeError):
    """Raised before publishing a batch that exceeded a configured budget."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResourcePeaks:
    process_rss_bytes: int
    gpu_process_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "process_rss_bytes": self.process_rss_bytes,
            "gpu_process_bytes": self.gpu_process_bytes,
        }


class ResourceBudgetGuard:
    """Sample inexpensive counters at each bounded unit-of-work boundary."""

    def __init__(
        self,
        *,
        rss_limit_bytes: int,
        gpu_limit_bytes: int,
        snapshot: Callable[[], Mapping[str, Any]] = safe_resource_snapshot,
    ) -> None:
        if rss_limit_bytes < 1 or gpu_limit_bytes < 1:
            raise ValueError("resource limits must be positive")
        self.rss_limit_bytes = int(rss_limit_bytes)
        self.gpu_limit_bytes = int(gpu_limit_bytes)
        self._snapshot = snapshot
        self._peak_rss = 0
        self._peak_gpu = 0

    @property
    def peaks(self) -> ResourcePeaks:
        return ResourcePeaks(self._peak_rss, self._peak_gpu)

    def check(self) -> ResourcePeaks:
        """Record current counters and raise stable errors at the hard limits."""

        values = self._snapshot()
        rss = int(values.get("process_rss_bytes") or 0)
        gpu = max(
            int(values.get("cuda_allocated_bytes") or 0),
            int(values.get("cuda_reserved_bytes") or 0),
        )
        self._peak_rss = max(self._peak_rss, rss)
        self._peak_gpu = max(self._peak_gpu, gpu)
        if rss >= self.rss_limit_bytes:
            raise ResourceBudgetExceeded("provisioning_rss_budget_exceeded")
        if gpu >= self.gpu_limit_bytes:
            raise ResourceBudgetExceeded("provisioning_gpu_budget_exceeded")
        return self.peaks

    @staticmethod
    def release_cuda() -> None:
        """Best-effort cleanup without importing torch when it is not loaded."""

        import gc
        import sys

        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except (AttributeError, RuntimeError):
            return


def current_process_id() -> int:
    """Expose the process identity for sanitized diagnostics and tests."""

    return os.getpid()


__all__ = [
    "ResourceBudgetExceeded",
    "ResourceBudgetGuard",
    "ResourcePeaks",
    "current_process_id",
]
