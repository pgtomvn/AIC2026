"""Explicit, single-flight runtime warm-up for KIS query paths."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from PIL import Image

_TERMINAL_STATES = frozenset({"ready", "degraded", "failed"})
_USABLE_CHANNEL_STATES = frozenset({"ready", "configured_unverified"})
_RETRIEVAL_CHANNELS = frozenset(
    {
        "visual",
        "organizer_clip",
        "accelerated_visual",
        "semantic_text",
        "ocr",
        "asr",
        "metadata",
    }
)
_WORKFLOW_NAMES = ("kis", "qa", "trake")


def _stdlib_rss_bytes() -> int | None:
    """Return process RSS without requiring psutil."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            get_process_memory_info.restype = wintypes.BOOL
            handle = get_current_process()
            succeeded = get_process_memory_info(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.working_set_size) if succeeded else None
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def safe_resource_snapshot() -> dict[str, int | None]:
    """Return inexpensive process/GPU counters without importing heavy models."""

    process_rss: int | None = None
    try:
        import psutil

        process_rss = int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        process_rss = _stdlib_rss_bytes()

    allocated = reserved = peak = None
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            peak = int(torch.cuda.max_memory_allocated())
    except (AttributeError, RuntimeError):
        pass
    return {
        "process_rss_bytes": process_rss,
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "cuda_peak_allocated_bytes": peak,
    }


def _failure_reason(error: Exception) -> str:
    if isinstance(error, OSError):
        return "warmup_io_failure"
    if isinstance(error, RuntimeError):
        return "warmup_runtime_failure"
    return "warmup_failed"


class WarmupService:
    """Warm only provisioned KIS channels and publish sanitized progress."""

    def __init__(
        self,
        *,
        pipeline: Any,
        readiness_service: Any,
        batch_id: str,
        profile: str,
        artifact_repository: Any = None,
        trake_pipeline: Any = None,
        warm_query: str = "một người đang đi bộ trong thành phố",
        include_raw_refiner: bool = True,
        startup_budget_seconds: float = 90.0,
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        resource_snapshot: Callable[
            [], dict[str, int | None]
        ] = safe_resource_snapshot,
    ) -> None:
        self.pipeline = pipeline
        self.readiness_service = readiness_service
        self.batch_id = batch_id
        self.profile = profile
        self.artifact_repository = artifact_repository
        self.trake_pipeline = trake_pipeline
        self.warm_query = warm_query
        self.include_raw_refiner = include_raw_refiner
        if startup_budget_seconds <= 0:
            raise ValueError("startup_budget_seconds must be positive")
        self.startup_budget_seconds = float(startup_budget_seconds)
        self._wall_clock = wall_clock
        self._monotonic = monotonic
        self._resource_snapshot = resource_snapshot
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._done.set()
        self._workflow_done = {name: threading.Event() for name in _WORKFLOW_NAMES}
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._search_usable = False
        self._started_at: float | None = None
        self._started_monotonic: float | None = None
        self._finished_at: float | None = None
        self._duration_ms: float | None = None
        self._channels: list[dict[str, Any]] = []
        self._workflows = self._initial_workflows()
        self._fully_ready = False
        self._evicted = False
        self._readiness_report: Any = None
        self._resources = self._resource_snapshot()

    def start(self) -> bool:
        """Start one daemon worker; repeated ready/running calls are no-ops."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._state in {"ready", "degraded"}:
                return False
            self._state = "warming"
            self._search_usable = False
            self._started_at = self._wall_clock()
            self._started_monotonic = self._monotonic()
            self._finished_at = None
            self._duration_ms = None
            self._channels = []
            self._workflows = {
                name: {
                    "state": "warming",
                    "usable": False,
                    "reason_code": "readiness_audit",
                    "required_channels": [],
                    "unavailable_channels": [],
                }
                for name in _WORKFLOW_NAMES
            }
            self._fully_ready = False
            self._evicted = False
            self._readiness_report = None
            self._done.clear()
            for event in self._workflow_done.values():
                event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="aic-kis-warmup",
                daemon=True,
            )
            self._thread.start()
            return True

    def wait(
        self,
        timeout: float | None = None,
        *,
        workflow: str | None = None,
    ) -> dict[str, Any]:
        """Wait for all stages or one workflow and return the latest snapshot."""

        if workflow is None:
            self._done.wait(timeout)
        else:
            if workflow not in _WORKFLOW_NAMES:
                raise ValueError("workflow must be one of: kis, qa, trake")
            self._workflow_done[workflow].wait(timeout)
        return self.snapshot()

    def mark_evicted(self) -> None:
        """Mark resident encoders unavailable after an explicit GPU handoff."""

        with self._lock:
            if self._state in {"ready", "degraded"}:
                self._state = "evicted"
                self._search_usable = False
                self._fully_ready = False
                self._evicted = True
                for name, workflow in self._workflows.items():
                    if workflow.get("state") != "not_configured":
                        self._workflows[name] = {
                            **workflow,
                            "state": "evicted",
                            "usable": False,
                            "reason_code": "encoder_evicted",
                        }
                if self._thread is None or not self._thread.is_alive():
                    self._finished_at = self._wall_clock()
                    self._done.set()
                    for event in self._workflow_done.values():
                        event.set()

    def snapshot(self) -> dict[str, Any]:
        """Return a nonblocking, provider-safe state snapshot."""

        with self._lock:
            duration_ms = self._duration_ms
            budget_exceeded = False
            if self._started_monotonic is not None:
                elapsed_ms = (
                    float(self._duration_ms)
                    if self._duration_ms is not None
                    else (self._monotonic() - self._started_monotonic) * 1_000
                )
                if duration_ms is None:
                    duration_ms = round(elapsed_ms, 3)
                budget_exceeded = elapsed_ms > self.startup_budget_seconds * 1_000
            return {
                "schema_version": 2,
                "state": self._state,
                "search_usable": self._search_usable,
                "fully_ready": self._fully_ready,
                "batch_id": self.batch_id,
                "profile": self.profile,
                "started_at_unix": self._started_at,
                "finished_at_unix": self._finished_at,
                "duration_ms": duration_ms,
                "budget_seconds": self.startup_budget_seconds,
                "budget_exceeded": budget_exceeded,
                "workflows": {
                    name: {
                        key: list(value) if isinstance(value, list) else value
                        for key, value in workflow.items()
                    }
                    for name, workflow in self._workflows.items()
                },
                "channels": [dict(item) for item in self._channels],
                "resources": dict(self._resources),
            }

    @staticmethod
    def _initial_workflows() -> dict[str, dict[str, Any]]:
        return {
            name: {
                "state": "idle",
                "usable": False,
                "reason_code": "not_started",
                "required_channels": [],
                "unavailable_channels": [],
            }
            for name in _WORKFLOW_NAMES
        }

    def workflow_snapshot(self, workflow: str) -> dict[str, Any]:
        """Return one sanitized workflow state without blocking."""

        if workflow not in _WORKFLOW_NAMES:
            raise ValueError("workflow must be one of: kis, qa, trake")
        with self._lock:
            selected = self._workflows[workflow]
            return {
                key: list(value) if isinstance(value, list) else value
                for key, value in selected.items()
            }

    def readiness_report(self) -> Any:
        """Return the exact report used by this warm-up without probing again."""

        with self._lock:
            return self._readiness_report

    def _channel_result(
        self,
        name: str,
        operation: Callable[[], None],
        *,
        unavailable_state: str | None = None,
    ) -> dict[str, Any]:
        started = self._monotonic()
        if unavailable_state is not None:
            return {
                "name": name,
                "state": unavailable_state,
                "reason_code": "channel_not_provisioned",
                "duration_ms": 0.0,
            }
        try:
            operation()
            return {
                "name": name,
                "state": "ready",
                "reason_code": "warmup_completed",
                "duration_ms": round((self._monotonic() - started) * 1_000, 3),
            }
        except Exception as error:
            return {
                "name": name,
                "state": "failed",
                "reason_code": _failure_reason(error),
                "duration_ms": round((self._monotonic() - started) * 1_000, 3),
            }

    def _workflow_contracts(self, report: Any) -> dict[str, dict[str, Any] | None]:
        contracts: dict[str, dict[str, Any] | None] = {}
        workflow_probe = getattr(self.readiness_service, "workflow_readiness", None)
        for name in _WORKFLOW_NAMES:
            selected = getattr(report, name, None)
            if selected is None and callable(workflow_probe):
                try:
                    selected = workflow_probe(report, name)
                except Exception:
                    selected = None
            if selected is None:
                contracts[name] = None
                continue
            contracts[name] = {
                "state": str(getattr(selected, "state", "blocked")),
                "required_channels": tuple(
                    str(value) for value in getattr(selected, "required_channels", ())
                ),
                "unavailable_channels": tuple(
                    str(value) for value in getattr(selected, "unavailable_channels", ())
                ),
            }
        if contracts["trake"] is None and self.trake_pipeline is not None:
            selected_channel = self.trake_pipeline.channel_for_batch(self.batch_id)
            contracts["trake"] = {
                "state": "ready",
                "required_channels": (selected_channel,),
                "unavailable_channels": (),
            }
        return contracts

    def _workflow_states(
        self,
        contracts: dict[str, dict[str, Any] | None],
        readiness_by_name: dict[str, str],
        channels: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        warmed = {str(item.get("name")): str(item.get("state")) for item in channels}
        warmable = {
            *(str(channel.name) for channel in self.pipeline.dense_channels),
            *(str(channel.name) for channel in self.pipeline.lexical_channels),
        }
        ready_retrieval = {
            name
            for name in warmable
            if name in _RETRIEVAL_CHANNELS and warmed.get(name) == "ready"
        }
        def channel_usable(name: str) -> bool:
            if name in warmable:
                return warmed.get(name) == "ready"
            return readiness_by_name.get(name) in _USABLE_CHANNEL_STATES

        results: dict[str, dict[str, Any]] = {}
        for name in _WORKFLOW_NAMES:
            contract = contracts.get(name)
            if contract is None:
                results[name] = {
                    "state": "not_configured",
                    "usable": False,
                    "reason_code": "workflow_not_configured",
                    "required_channels": [],
                    "unavailable_channels": [],
                }
                continue
            required = tuple(contract["required_channels"])
            unavailable = [value for value in required if not channel_usable(value)]
            if name == "kis":
                workflow_failure = any(warmed.get(value) == "failed" for value in required)
                workflow_failure = workflow_failure or warmed.get("raw_refiner") == "failed"
                usable = bool(ready_retrieval)
                if not warmable and warmed.get("raw_refiner") == "ready":
                    usable = True
                state = (
                    "ready"
                    if usable and not unavailable and not workflow_failure
                    else "degraded"
                    if usable
                    else "warming"
                    if any(warmed.get(value) == "warming" for value in required)
                    else "failed"
                )
                reason = (
                    "warmup_completed"
                    if state == "ready"
                    else "usable_with_missing_channels"
                    if state == "degraded"
                    else "channel_warming"
                    if state == "warming"
                    else "required_channel_unavailable"
                )
            elif name == "qa":
                workflow_failure = any(warmed.get(value) == "failed" for value in required)
                hard_unavailable = any(value in unavailable for value in ("gemini", "canonical_images"))
                usable = bool(ready_retrieval) and not hard_unavailable
                state = (
                    "ready"
                    if usable and not unavailable and not workflow_failure
                    else "degraded"
                    if usable
                    else "warming"
                    if any(warmed.get(value) == "warming" for value in required)
                    else "failed"
                )
                reason = (
                    "warmup_completed"
                    if state == "ready"
                    else "usable_with_missing_channels"
                    if state == "degraded"
                    else "channel_warming"
                    if state == "warming"
                    else "required_channel_unavailable"
                )
            else:
                selected_channel = required[0] if required else None
                trajectory_name = (
                    f"trake_{selected_channel}_trajectory" if selected_channel else None
                )
                trajectory_state = warmed.get(trajectory_name or "")
                dense_ready = bool(selected_channel and channel_usable(selected_channel))
                usable = dense_ready and trajectory_state == "ready"
                if usable:
                    state = "degraded" if contract["state"] == "degraded" else "ready"
                    reason = "warmup_completed"
                elif selected_channel and readiness_by_name.get(selected_channel) not in _USABLE_CHANNEL_STATES:
                    state = "failed"
                    reason = "required_channel_unavailable"
                elif trajectory_state == "failed":
                    state = "failed"
                    reason = "trajectory_cache_failed"
                else:
                    state = "warming"
                    reason = "trajectory_cache_warming"
                if trajectory_name and trajectory_state != "ready":
                    unavailable.append(trajectory_name)
                required = (*required, *((trajectory_name,) if trajectory_name else ()))
            results[name] = {
                "state": state,
                "usable": usable,
                "reason_code": reason,
                "required_channels": list(dict.fromkeys(required)),
                "unavailable_channels": list(dict.fromkeys(unavailable)),
            }
        return results

    def _publish(
        self,
        *,
        channels: list[dict[str, Any]],
        workflows: dict[str, dict[str, Any]],
        finished: bool,
        started: float,
    ) -> None:
        configured = [
            workflow
            for workflow in workflows.values()
            if workflow.get("state") != "not_configured"
        ]
        kis_usable = bool(workflows.get("kis", {}).get("usable", False))
        fully_ready = bool(configured) and all(
            workflow.get("state") == "ready" for workflow in configured
        )
        if not finished:
            state = "degraded" if kis_usable else "warming"
        elif kis_usable:
            state = "ready" if fully_ready else "degraded"
        else:
            state = "failed"
        with self._lock:
            if self._evicted:
                workflows = {
                    name: (
                        dict(workflow)
                        if workflow.get("state") == "not_configured"
                        else {
                            **workflow,
                            "state": "evicted",
                            "usable": False,
                            "reason_code": "encoder_evicted",
                        }
                    )
                    for name, workflow in workflows.items()
                }
                state = "evicted"
                kis_usable = False
                fully_ready = False
            self._state = state
            self._search_usable = kis_usable
            self._fully_ready = fully_ready
            self._channels = [dict(item) for item in channels]
            self._workflows = {
                name: {
                    key: list(value) if isinstance(value, (list, tuple)) else value
                    for key, value in workflow.items()
                }
                for name, workflow in workflows.items()
            }
            terminal_workflow_states = {
                "ready",
                "degraded",
                "failed",
                "not_configured",
                "blocked",
                "evicted",
            }
            for name, workflow in workflows.items():
                if workflow.get("state") in terminal_workflow_states:
                    self._workflow_done[name].set()
            if finished:
                self._finished_at = self._wall_clock()
                self._duration_ms = round((self._monotonic() - started) * 1_000, 3)
                self._resources = self._resource_snapshot()
                self._done.set()

    def _run(self) -> None:
        started = self._monotonic()
        channels: list[dict[str, Any]] = []
        try:
            report = self.readiness_service.check(self.batch_id, mode="app")
            with self._lock:
                self._readiness_report = report
            readiness_by_name = {
                str(item.name): str(item.state) for item in getattr(report, "channels", ())
            }
            contracts = self._workflow_contracts(report)
            vector_filter, lexical_filter = self.pipeline._batch_filters(
                {"batch_filters": (self.batch_id,)}
            )

            for channel in self.pipeline.dense_channels:
                state = readiness_by_name.get(channel.name)
                if state not in _USABLE_CHANNEL_STATES:
                    channels.append(
                        self._channel_result(
                            channel.name,
                            lambda: None,
                            unavailable_state=state or "not_provisioned",
                        )
                    )
                    continue

                def warm_dense(channel: Any = channel) -> None:
                    if not channel.store.is_available(vector_filter):
                        raise LookupError("channel is not provisioned")
                    channel.encoder.encode_text(self.warm_query)

                channels.append(self._channel_result(channel.name, warm_dense))

            for channel in self.pipeline.lexical_channels:
                state = readiness_by_name.get(channel.name)
                if state not in _USABLE_CHANNEL_STATES:
                    channels.append(
                        self._channel_result(
                            channel.name,
                            lambda: None,
                            unavailable_state=state or "not_provisioned",
                        )
                    )
                    continue

                def warm_lexical(channel: Any = channel) -> None:
                    if not channel.store.is_available(lexical_filter):
                        raise LookupError("channel is not provisioned")

                channels.append(self._channel_result(channel.name, warm_lexical))

            refiner = getattr(self.pipeline, "raw_refiner", None)
            has_refinable = False
            if (
                self.include_raw_refiner
                and refiner is not None
                and self.artifact_repository is not None
                and callable(getattr(self.artifact_repository, "has_refinable_frames", None))
            ):
                has_refinable = bool(
                    self.artifact_repository.has_refinable_frames(self.batch_id)
                )
            if has_refinable:

                def warm_refiner() -> None:
                    refiner.encoder.encode_text(self.warm_query)
                    image = Image.new("RGB", (224, 224), "black")
                    try:
                        refiner.encoder.encode_images([image])
                    finally:
                        image.close()

                channels.append(self._channel_result("raw_refiner", warm_refiner))

            workflows = self._workflow_states(contracts, readiness_by_name, channels)
            self._publish(
                channels=channels,
                workflows=workflows,
                finished=False,
                started=started,
            )

            selected_trake_channel: str | None = None
            if self.trake_pipeline is not None:
                selected_trake_channel = self.trake_pipeline.channel_for_batch(self.batch_id)
            if (
                selected_trake_channel is not None
                and readiness_by_name.get(selected_trake_channel) in _USABLE_CHANNEL_STATES
            ):
                trajectory_name = f"trake_{selected_trake_channel}_trajectory"
                trake_started = self._monotonic()
                trajectory = {
                    "name": trajectory_name,
                    "state": "warming",
                    "reason_code": "trajectory_cache_warming",
                    "duration_ms": 0.0,
                }
                channels.insert(0, trajectory)
                workflows = self._workflow_states(contracts, readiness_by_name, channels)
                self._publish(
                    channels=channels,
                    workflows=workflows,
                    finished=False,
                    started=started,
                )
                try:
                    cache_snapshot = self.trake_pipeline.warm(self.batch_id)
                    trajectory.update(
                        {
                            "state": "ready",
                            "reason_code": str(
                                cache_snapshot.get("reason_code", "warmup_completed")
                            ),
                        }
                    )
                    for field in (
                        "source",
                        "video_count",
                        "frame_count",
                        "source_point_count",
                        "nbytes",
                        "load_duration_ms",
                        "build_duration_ms",
                        "persist_duration_ms",
                    ):
                        if field in cache_snapshot:
                            trajectory[field] = cache_snapshot[field]
                except Exception as error:
                    trajectory.update(
                        {
                            "state": "failed",
                            "reason_code": _failure_reason(error),
                        }
                    )
                trajectory["duration_ms"] = round(
                    (self._monotonic() - trake_started) * 1_000,
                    3,
                )

            workflows = self._workflow_states(contracts, readiness_by_name, channels)
            self._publish(
                channels=channels,
                workflows=workflows,
                finished=True,
                started=started,
            )
        except Exception:
            channels.append(
                {
                    "name": "runtime",
                    "state": "failed",
                    "reason_code": "readiness_evaluation_failed",
                    "duration_ms": round((self._monotonic() - started) * 1_000, 3),
                }
            )
            workflows = {
                name: {
                    "state": "failed",
                    "usable": False,
                    "reason_code": "readiness_evaluation_failed",
                    "required_channels": [],
                    "unavailable_channels": [],
                }
                for name in _WORKFLOW_NAMES
            }
            self._publish(
                channels=channels,
                workflows=workflows,
                finished=True,
                started=started,
            )


__all__ = ["WarmupService", "safe_resource_snapshot"]
