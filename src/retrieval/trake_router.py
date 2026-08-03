"""Batch-aware routing for the single-space TRAKE alignment core.

Reference batches use only the verified organizer OpenCLIP space, while native
batches use only SigLIP2.  The router prevents a request from accidentally
querying a collection with an encoder from another vector space.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any, Callable

from src.contracts import TRAKEQueryRequest, TRAKEResponse
from src.retrieval.trake_alignment import TRAKEPipeline

_USABLE_CHANNEL_STATES = frozenset({"ready", "configured_unverified"})
_SAFE_STATUS_FIELDS = (
    "applicability",
    "state",
    "reason_code",
    "remediation_id",
    "count",
    "coverage",
    "expected_dimension",
)


class TRAKENotReadyError(RuntimeError):
    """A sanitized failure raised before an incompatible dense query can run."""

    def __init__(
        self,
        channel_name: str,
        reason_code: str,
        *,
        channel_statuses: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.channel_name = channel_name
        self.reason_code = reason_code
        self.channel_statuses = {
            str(name): dict(status) for name, status in (channel_statuses or {}).items()
        }
        super().__init__("the required TRAKE dense channel is not ready")


class TRAKEPipelineRouter:
    """Select exactly one compatible alignment core for a validated batch."""

    def __init__(
        self,
        pipelines: Mapping[str, TRAKEPipeline],
        config: Mapping[str, Any],
        *,
        readiness_service: Any = None,
        readiness_cache_seconds: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        allowed = {"visual", "organizer_clip"}
        unexpected = set(pipelines) - allowed
        if unexpected:
            raise ValueError(f"unsupported TRAKE dense channels: {sorted(unexpected)!r}")
        self.pipelines = dict(pipelines)
        self.config = config
        self.readiness_service = readiness_service
        if readiness_cache_seconds < 0:
            raise ValueError("readiness_cache_seconds must not be negative")
        self.readiness_cache_seconds = float(readiness_cache_seconds)
        self._monotonic = monotonic
        self._readiness_cache: dict[
            str, tuple[float, bool, dict[str, dict[str, Any]]]
        ] = {}
        self._readiness_lock = threading.RLock()

    def channel_for_batch(self, batch_id: str) -> str:
        """Return the sole compatible dense channel for one batch identity."""

        readiness = self.config.get("readiness", {})
        batches = readiness.get("batches", {}) if isinstance(readiness, Mapping) else {}
        specification = batches.get(batch_id, {}) if isinstance(batches, Mapping) else {}
        default_kind = (
            readiness.get("default_batch_kind", "native")
            if isinstance(readiness, Mapping)
            else "native"
        )
        kind = specification.get("kind", default_kind) if isinstance(specification, Mapping) else default_kind
        if kind == "reference":
            return "organizer_clip"
        if kind == "native":
            return "visual"
        raise ValueError("batch kind must be reference or native")

    @staticmethod
    def _safe_status(value: Any) -> dict[str, Any]:
        if hasattr(value, "to_dict"):
            raw = value.to_dict()
        elif isinstance(value, Mapping):
            raw = dict(value)
        else:
            raw = {}
        return {name: raw.get(name) for name in _SAFE_STATUS_FIELDS if name in raw}

    def channel_readiness(self, batch_id: str) -> tuple[bool, dict[str, dict[str, Any]]]:
        """Return readiness without loading a model or exposing service details."""

        channel_name = self.channel_for_batch(batch_id)
        pipeline = self.pipelines.get(channel_name)
        now = self._monotonic()
        with self._readiness_lock:
            cached = self._readiness_cache.get(batch_id)
            if cached is not None and now - cached[0] < self.readiness_cache_seconds:
                return cached[1], {
                    name: dict(status) for name, status in cached[2].items()
                }
        if self.readiness_service is None:
            state = "configured_unverified" if pipeline is not None else "not_provisioned"
            result = pipeline is not None, {
                channel_name: {
                    "state": state,
                    "reason_code": "readiness_probe_unavailable",
                }
            }
        else:
            try:
                report = self.readiness_service.check(batch_id, mode="app")
                channel = next(
                    (item for item in report.channels if item.name == channel_name),
                    None,
                )
            except Exception:
                result = False, {
                    channel_name: {
                        "state": "service_unavailable",
                        "reason_code": "readiness_probe_failed",
                    }
                }
            else:
                status = self._safe_status(channel)
                state = str(status.get("state", "not_provisioned"))
                usable = pipeline is not None and state in _USABLE_CHANNEL_STATES
                if pipeline is None and state in _USABLE_CHANNEL_STATES:
                    status = {
                        **status,
                        "state": "blocked_asset",
                        "reason_code": "compatible_encoder_unavailable",
                    }
                result = usable, {channel_name: status}
        with self._readiness_lock:
            if result[0]:
                self._readiness_cache[batch_id] = (
                    self._monotonic(),
                    True,
                    {name: dict(status) for name, status in result[1].items()},
                )
            else:
                # Provisioning and service recovery must be observable on the
                # next request; only successful gates receive the long TTL.
                self._readiness_cache.pop(batch_id, None)
        return result[0], {name: dict(status) for name, status in result[1].items()}

    def invalidate_readiness(self, batch_id: str | None = None) -> None:
        """Invalidate cached query gating after provisioning changes."""

        with self._readiness_lock:
            if batch_id is None:
                self._readiness_cache.clear()
            else:
                self._readiness_cache.pop(batch_id, None)

    def search(
        self,
        request: TRAKEQueryRequest,
        *,
        default_batch_id: str | None = None,
    ) -> TRAKEResponse:
        """Route a request without ever crossing the organizer/native spaces."""

        if not isinstance(request, TRAKEQueryRequest):
            request = TRAKEQueryRequest.model_validate(request)
        batch_id = request.resolve_batch(default_batch_id)
        channel_name = self.channel_for_batch(batch_id)
        usable, statuses = self.channel_readiness(batch_id)
        pipeline = self.pipelines.get(channel_name)
        if not usable or pipeline is None:
            reason = str(statuses[channel_name].get("reason_code") or "not_provisioned")
            raise TRAKENotReadyError(
                channel_name,
                reason,
                channel_statuses=statuses,
            )
        trajectory_cache = getattr(pipeline, "trajectory_cache", None)
        if trajectory_cache is not None and trajectory_cache.batch(batch_id) is None:
            cache_snapshot = trajectory_cache.snapshot()
            cache_state = str(cache_snapshot.get("state", "idle"))
            reason_code = (
                "trajectory_cache_warming"
                if cache_state in {"idle", "warming"}
                else "trajectory_cache_failed"
            )
            statuses[channel_name] = {
                **statuses[channel_name],
                "state": "warming" if cache_state in {"idle", "warming"} else "service_unavailable",
                "reason_code": reason_code,
            }
            raise TRAKENotReadyError(
                channel_name,
                reason_code,
                channel_statuses=statuses,
            )
        statuses[channel_name]["attempted"] = True
        return pipeline.search(
            request,
            default_batch_id=batch_id,
            channel_statuses=statuses,
        )

    def warm(self, batch_id: str) -> dict[str, Any]:
        """Warm only the one provisioned dense space selected for ``batch_id``."""

        channel_name = self.channel_for_batch(batch_id)
        usable, statuses = self.channel_readiness(batch_id)
        pipeline = self.pipelines.get(channel_name)
        if not usable or pipeline is None:
            reason = str(statuses[channel_name].get("reason_code") or "not_provisioned")
            raise TRAKENotReadyError(
                channel_name,
                reason,
                channel_statuses=statuses,
            )
        snapshot = pipeline.warm(batch_id)
        # A full trajectory warm may outlive the readiness TTL. Refresh the
        # already-sanitized successful gate at publication time so the first
        # post-warm query never repeats expensive service probes.
        with self._readiness_lock:
            self._readiness_cache[batch_id] = (
                self._monotonic(),
                True,
                {name: dict(status) for name, status in statuses.items()},
            )
        return snapshot


__all__ = ["TRAKENotReadyError", "TRAKEPipelineRouter"]
