"""Sequential multi-batch warm-up sharing encoders, not trajectory state."""
from __future__ import annotations

import threading
import time
from typing import Any

from src.warmup import WarmupService


def scoped_snapshot(snapshot: dict[str, Any], batches: tuple[str, ...]) -> dict[str, Any]:
    """A ready single-batch filter remains usable if the other batch is down."""
    children = snapshot.get("batches", {})
    if len(batches) == 1 and batches[0] in children:
        return dict(children[batches[0]])
    return snapshot


class BatchWarmup:
    """Do not announce combined readiness before every selected batch is usable."""

    def __init__(self, services: dict[str, WarmupService], *, primary: str) -> None:
        self.services = services
        self.primary = primary
        self.trake_pipeline = getattr(services[primary], "trake_pipeline", None)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._done.set()

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if all(service.snapshot()["state"] in {"ready", "degraded"} for service in self.services.values()):
                return False
            self._done.clear()
            self._thread = threading.Thread(target=self._run, name="aic-batch-warmup", daemon=True)
            self._thread.start()
            return True

    def _run(self) -> None:
        try:
            for service in self.services.values():
                service.start()
                service.wait()
        finally:
            self._done.set()

    def snapshot(self) -> dict[str, Any]:
        snapshots = {batch: service.snapshot() for batch, service in self.services.items()}
        result = dict(snapshots[self.primary])
        workflows = {}
        for name in ("kis", "qa", "trake"):
            values = [item["workflows"][name] for item in snapshots.values()]
            usable = all(item.get("usable", False) for item in values)
            workflows[name] = {
                "usable": usable, "state": "ready" if usable else "warming" if not self._done.is_set() else "unavailable",
                "reason_code": "all_batches_ready" if usable else "batch_not_ready",
                "required_channels": [],
                "unavailable_channels": [batch for batch, item in snapshots.items()
                                         if not item["workflows"][name].get("usable", False)],
            }
        result.update(batches=snapshots, workflows=workflows,
                      search_usable=workflows["kis"]["usable"],
                      fully_ready=all(item["fully_ready"] for item in snapshots.values()),
                      state="warming" if not self._done.is_set() else (
                          "ready" if all(item["fully_ready"] for item in snapshots.values()) else "degraded"))
        if all(item["state"] == "idle" for item in snapshots.values()):
            result["state"] = "idle"
        if any(item["state"] == "evicted" for item in snapshots.values()) and self._done.is_set():
            result["state"] = "evicted"
        if result["state"] in {"idle", "evicted"}:
            for item in result["workflows"].values():
                item["state"] = result["state"]
        return result

    def workflow_snapshot(self, workflow: str) -> dict[str, Any]:
        return self.snapshot()["workflows"][workflow]

    def wait(self, timeout: float | None = None, *, workflow: str | None = None) -> dict[str, Any]:
        if workflow is None:
            self._done.wait(timeout)
            return self.snapshot()
        if workflow not in {"kis", "qa", "trake"}:
            raise ValueError("unknown workflow")
        started = time.monotonic()
        while not self.workflow_snapshot(workflow)["usable"] and not self._done.is_set():
            remaining = None if timeout is None else max(0, timeout - (time.monotonic() - started))
            if remaining == 0:
                break
            self._done.wait(0.05 if remaining is None else min(0.05, remaining))
        return self.snapshot()

    def readiness_report(self) -> Any:
        return self.services[self.primary].readiness_report()

    def mark_evicted(self) -> None:
        for service in self.services.values():
            service.mark_evicted()
