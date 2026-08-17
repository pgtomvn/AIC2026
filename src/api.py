"""FastAPI transport for the shared KIS, Visual Q&A, and TRAKE cores."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import math
import re
import shutil
import sys
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

from src import PROJECT_ROOT
from src.contracts import (
    CompetitionSpec,
    QACandidateFrame,
    QACandidateRequest,
    QACompetitionSpec,
    QAQueryRequest,
    QASingleFrameAnswerRequest,
    QueryRequest,
    SearchHit,
    TRAKECompetitionSpec,
    TRAKEQueryRequest,
)
from src.export.bundle_builder import (
    BundleBuildError,
    build_codabench_bundle,
    load_bundle_manifest,
    re_fullmatch_run_id,
)
from src.export.preflight import (
    BundleEntry,
    BundleValidationContext,
    BundleValidationError,
)
from src.qa_submission import QASubmissionValidationError, export_qa_submission
from src.readiness import aggregate_channel_health
from src.retrieval.qa_session import QASessionStoreError
from src.retrieval.trake_router import TRAKENotReadyError
from src.retrieval.visual_qa import QACandidateMediaUnavailableError
from src.runtime import RuntimeContext, build_runtime
from src.submission import FrameIndexEvidence, SubmissionValidationError, export_submission
from src.trake_submission import (
    TRAKEFrameIndexEvidence,
    TRAKESubmissionValidationError,
    export_trake_submission,
)
from src.utils.io_helpers import sanitize_launch_identity
from src.vlm.qa_reasoner import QAImage, QAReasonerError

_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_FRAME_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_OPERATIONAL_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_QA_OUTCOME_STATUSES = frozenset(
    {
        "pending",
        "answered",
        "unanswerable",
        "media_failed",
        "provider_failed",
        "internal_failed",
    }
)
_QA_FAILURE_CODES = frozenset(
    {
        "media_unavailable",
        "media_window_too_large",
        "provider_unavailable",
        "provider_request_failed",
        "provider_response_invalid",
        "internal_reasoning_error",
    }
)


def _runtime_batch_filters(
    context: RuntimeContext,
    values: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve and validate request batches without contacting retrieval services."""

    validator = getattr(context, "validate_batch_filters", None)
    if callable(validator):
        return tuple(validator(values or None))
    if values:
        return values
    selected = getattr(context, "selected_batch_id", None)
    if isinstance(selected, str) and selected.strip():
        return (selected.strip(),)
    return ()


def _with_runtime_batches(
    context: RuntimeContext,
    request: QueryRequest | QACandidateRequest | QAQueryRequest | TRAKEQueryRequest,
) -> QueryRequest | QACandidateRequest | QAQueryRequest | TRAKEQueryRequest:
    batches = _runtime_batch_filters(context, request.batch_filters)
    if batches == request.batch_filters:
        return request
    if isinstance(request, TRAKEQueryRequest):
        if len(batches) != 1:
            raise ValueError("TRAKE requires exactly one resolved batch")
        return request.model_copy(update={"batch_filters": batches})
    return replace(request, batch_filters=batches)


_QA_FAILURE_STAGES = frozenset({"media", "provider", "internal", "retrieval"})
_QA_PROVIDER_ERROR_CATEGORIES = frozenset(
    {
        "auth_or_permission",
        "client_initialization",
        "client_unavailable",
        "gemini_response_invalid",
        "gemini_timeout",
        "invalid_request",
        "model_unavailable",
        "model_missing",
        "network_unreachable",
        "not_configured",
        "ollama_context_exceeded",
        "ollama_timeout",
        "ollama_response_invalid",
        "provider_chain_exhausted",
        "provider_request_failed",
        "rate_limited",
        "request_rejected",
        "response_invalid",
        "resource_exhausted",
        "server_unavailable",
    }
)
_QA_SEARCH_STATUSES = frozenset(
    {"completed", "completed_no_answer", "partial", "failed", "no_candidates", "incomplete"}
)
_QA_CHANNEL_STATES = frozenset(
    {
        "ready",
        "configured_unverified",
        "not_provisioned",
        "disabled_by_profile",
        "blocked_dependency",
        "blocked_asset",
        "incompatible_index",
        "service_unavailable",
    }
)
_QA_CHANNEL_APPLICABILITY = frozenset({"required", "optional", "not_applicable"})
_QA_RETRIEVAL_ERROR_CODES = frozenset(
    {
        "channel_unavailable",
        "not_provisioned",
        "disabled_by_profile",
        "blocked_dependency",
        "blocked_asset",
        "incompatible_index",
        "service_unavailable",
        "dependency_missing",
        "asset_missing",
        "collection_missing",
        "index_missing",
    }
)
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _directory_size(path: Path) -> int:
    """Return a best-effort byte count without following directory symlinks."""

    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            # A concurrently cleaned scratch file must not make status unavailable.
            continue
    return total


def _resource_snapshot(context: RuntimeContext) -> dict[str, int | None]:
    """Collect inexpensive local resource data without loading optional models."""

    data = context.config["data"]
    scratch = _project_path(str(data["scratch"]))
    scratch_limit = int(float(data["scratch_limit_gib"]) * 1024**3)
    reserve = int(float(data["free_space_reserve_gib"]) * 1024**3)
    disk = shutil.disk_usage(scratch.parent if scratch.parent.exists() else PROJECT_ROOT)
    process_rss: int | None = None
    try:
        import psutil

        process_rss = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass

    cuda_allocated = cuda_reserved = cuda_total = None
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            cuda_allocated = int(torch.cuda.memory_allocated())
            cuda_reserved = int(torch.cuda.memory_reserved())
            cuda_total = int(torch.cuda.get_device_properties(0).total_memory)
    except (AttributeError, RuntimeError):
        # CUDA status is optional and may be unavailable during driver teardown.
        pass

    scratch_used = _directory_size(scratch)
    return {
        "disk_free_bytes": int(disk.free),
        "disk_reserve_bytes": reserve,
        "scratch_used_bytes": scratch_used,
        "scratch_limit_bytes": scratch_limit,
        "scratch_remaining_bytes": max(0, scratch_limit - scratch_used),
        "process_rss_bytes": process_rss,
        "cuda_allocated_bytes": cuda_allocated,
        "cuda_reserved_bytes": cuda_reserved,
        "cuda_total_bytes": cuda_total,
    }


def _warmup_snapshot(context: RuntimeContext) -> dict[str, Any] | None:
    warmup = getattr(context, "warmup", None)
    snapshot = getattr(warmup, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        value = snapshot()
    except Exception:
        return {
            "state": "failed",
            "search_usable": False,
            "channels": [],
            "resources": {},
        }
    return dict(value) if isinstance(value, dict) else None


def _warmup_workflow(snapshot: Mapping[str, Any], workflow: str) -> dict[str, Any]:
    workflows = snapshot.get("workflows")
    if isinstance(workflows, Mapping):
        selected = workflows.get(workflow)
        if isinstance(selected, Mapping):
            return dict(selected)
    return {
        "state": str(snapshot.get("state", "idle")),
        "usable": bool(snapshot.get("search_usable", False)),
        "reason_code": "legacy_warmup_snapshot",
    }


def _require_search_usable(context: RuntimeContext, workflow: str = "kis") -> None:
    if workflow not in {"kis", "qa", "trake"}:
        raise ValueError("workflow must be one of: kis, qa, trake")
    warmup = getattr(context, "warmup", None)
    if warmup is None:
        return
    snapshot = _warmup_snapshot(context) or {}
    selected = _warmup_workflow(snapshot, workflow)
    state = str(selected.get("state", snapshot.get("state", "idle")))
    if state in {"idle", "evicted"}:
        start = getattr(warmup, "start", None)
        if callable(start):
            start()
        snapshot = _warmup_snapshot(context) or snapshot
        selected = _warmup_workflow(snapshot, workflow)
        state = str(selected.get("state", "warming"))
    if bool(selected.get("usable", False)):
        return
    retry_after = int(context.config.get("warmup", {}).get("retry_after_seconds", 2))
    warming = state in {"idle", "warming", "evicted"}
    raise HTTPException(
        status_code=503,
        detail={
            "code": (
                "runtime_warming"
                if warming
                else "trake_not_ready"
                if workflow == "trake"
                else "runtime_not_ready"
            ),
            "workflow": workflow,
            "state": state,
            "reason_code": selected.get("reason_code"),
            "message": "retrieval runtime is not ready",
        },
        headers={"Retry-After": str(max(1, retry_after))} if warming else None,
    )


def _release_local_vlm(context: RuntimeContext) -> None:
    coordinator = getattr(context, "gpu_coordinator", None)
    release = getattr(coordinator, "release_local_vlm", None)
    if callable(release):
        try:
            release()
            return
        except Exception:
            pass
    reasoner = getattr(context, "qa_reasoner", None)
    unload = getattr(reasoner, "unload", None)
    if callable(unload):
        try:
            unload()
        except Exception:
            pass


def _release_local_vlm_for_qa_retrieval(context: RuntimeContext, batch_id: str) -> None:
    """Release Qwen only when this batch needs a CUDA retrieval encoder."""

    coordinator = getattr(context, "gpu_coordinator", None)
    release = getattr(coordinator, "release_vlm_for_retrieval", None)
    if callable(release):
        try:
            release(batch_id)
            return
        except Exception:
            return
    _release_local_vlm(context)


def _require_qa_policy_usable(
    context: RuntimeContext,
    request: QAQueryRequest,
    *,
    require_retrieval: bool = True,
) -> None:
    """Gate the request-scoped provider and, when needed, QA retrieval media."""

    if require_retrieval:
        _require_search_usable(context, "kis")
    service = getattr(context, "readiness_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail={"code": "runtime_not_ready"})
    try:
        report = None
        warmup = getattr(context, "warmup", None)
        report_getter = getattr(warmup, "readiness_report", None)
        if callable(report_getter):
            warmed = report_getter()
            if getattr(warmed, "batch_id", None) == request.batch_filters[0]:
                report = warmed
        if report is None:
            report = service.check(request.batch_filters[0], mode="app")
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "runtime_not_ready"}) from None
    channels = {value.name: value for value in report.channels}
    if require_retrieval:
        canonical = channels.get("canonical_images")
        if canonical is None or not canonical.usable:
            raise HTTPException(
                status_code=503,
                detail={"code": "canonical_images_unavailable"},
            )
    local = bool(channels.get("ollama_vlm") and channels["ollama_vlm"].usable)
    gemini = bool(channels.get("gemini_vlm") and channels["gemini_vlm"].usable)
    usable = local if request.reasoning_policy == "local_only" else local or gemini
    if usable:
        return
    code = "ollama_unavailable" if request.reasoning_policy == "local_only" else "qa_all_providers_failed"
    raise HTTPException(status_code=503, detail={"code": code})


def _feedback_key(value: str) -> tuple[str, int]:
    video_id, separator, frame = value.rpartition(":")
    if not separator or not video_id:
        raise ValueError("feedback IDs must use VIDEO_ID:SOURCE_FRAME_IDX")
    source_frame_idx = int(frame)
    if source_frame_idx < 0:
        raise ValueError("feedback source frame IDs cannot be negative")
    return video_id, source_frame_idx


def _submission_identities(
    items: Any,
    *,
    normalize_video_id: Any,
) -> list[tuple[str, int]]:
    """Extract at most the submitted source identities without trusting payload shape."""

    if not isinstance(items, (list, tuple)):
        return []
    identities: list[tuple[str, int]] = []
    for item in items[:100]:
        if not isinstance(item, Mapping):
            continue
        raw_video_id = item.get("video_id", item.get("video_name"))
        try:
            video_id = normalize_video_id(raw_video_id)
        except (TypeError, ValueError):
            continue
        values: list[Any] = [item.get("source_frame_idx", item.get("frame_idx"))]
        matches = item.get("stage_matches")
        if isinstance(matches, (list, tuple)):
            values.extend(
                match.get("source_frame_idx") for match in matches if isinstance(match, Mapping)
            )
        frame_ids = item.get("frame_ids", item.get("source_frame_indices"))
        if isinstance(frame_ids, (list, tuple)):
            values.extend(frame_ids)
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                identities.append((video_id, value))
    return list(dict.fromkeys(identities))


def _bounded_frame_validation(
    repository: Any,
    batch_id: str,
    identities: list[tuple[str, int]],
) -> tuple[frozenset[tuple[str, int]], dict[str, dict[int, int]]]:
    """Resolve persisted frames and ordinal evidence using candidate-scoped queries."""

    bulk_lookup = getattr(repository, "existing_source_frames", None)
    scoped_evidence = getattr(repository, "frame_index_mappings_for_sources", None)
    if callable(bulk_lookup):
        existing = frozenset(bulk_lookup(batch_id, identities, include_refined=True))
    else:
        mappings = repository.frame_index_mappings(batch_id)
        canonical = {
            (video_id, source_frame_idx)
            for video_id, values in mappings.items()
            for source_frame_idx in values.values()
        }
        existing = frozenset(identity for identity in identities if identity in canonical)
    if callable(scoped_evidence):
        mappings = scoped_evidence(batch_id, identities)
    else:
        mappings = repository.frame_index_mappings(batch_id)
    return existing, mappings


def _safe_polygon_values(value: Any, *, depth: int = 0) -> Any:
    """Bound OCR geometry to finite numeric lists without arbitrary payload fields."""

    if depth > 3 or not isinstance(value, (list, tuple)) or len(value) > 256:
        return None
    result: list[Any] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            number = float(item)
            if not math.isfinite(number):
                return None
            result.append(item)
            continue
        nested = _safe_polygon_values(item, depth=depth + 1)
        if nested is None:
            return None
        result.append(nested)
    return result


def _safe_hit_evidence(value: Any) -> dict[str, Any]:
    """Allowlist retrieval evidence while dropping URIs, paths, and backend payloads."""

    raw_items = value.get("items", ()) if isinstance(value, dict) else value
    if not isinstance(raw_items, (list, tuple)):
        return {"items": []}
    items: list[dict[str, Any]] = []
    for raw in raw_items[:32]:
        if not isinstance(raw, dict):
            continue
        item: dict[str, Any] = {}
        for name in ("channel", "segment_id", "kind", "identity_method"):
            raw_value = raw.get(name)
            safe = (
                raw_value.strip()
                if isinstance(raw_value, str)
                and _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(raw_value.strip())
                else None
            )
            if safe is not None:
                item[name] = safe
        rank = raw.get("rank")
        if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1:
            item["rank"] = rank
        for name in ("raw_score", "sample_fps", "radius_seconds"):
            number = raw.get(name)
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                number = float(number)
                if math.isfinite(number):
                    item[name] = number
        text_value = raw.get("text_raw")
        if isinstance(text_value, str):
            item["text_raw"] = text_value[:2_048]
        polygons = _safe_polygon_values(raw.get("polygons"))
        if polygons is not None:
            item["polygons"] = polygons
        if item:
            items.append(item)
    return {"items": items}


def _public_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize internal retrieval dictionaries to the public SearchHit contract."""

    result: list[dict[str, Any]] = []
    for item in hits:
        raw_video_id = str(item.get("video_id", "")).strip()
        video_id = raw_video_id if _SAFE_FRAME_IDENTIFIER.fullmatch(raw_video_id) else None
        if video_id is None:
            continue
        source_frame_idx = int(item["source_frame_idx"])
        raw_batch_id = item.get("batch_id")
        batch_id = (
            raw_batch_id.strip()
            if isinstance(raw_batch_id, str)
            and _SAFE_FRAME_IDENTIFIER.fullmatch(raw_batch_id.strip())
            else None
        )
        thumbnail_uri = (
            f"/v1/frames/{batch_id}/{video_id}/{source_frame_idx}/image"
            if batch_id is not None
            else f"canonical://frame/{video_id}/{source_frame_idx}"
        )
        raw_channel_ranks = item.get("channel_ranks", {})
        raw_channel_ranks = raw_channel_ranks if isinstance(raw_channel_ranks, dict) else {}
        channel_ranks = {
            name: channel_rank
            for raw_name, channel_rank in raw_channel_ranks.items()
            if (
                name := (
                    str(raw_name).strip()
                    if _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(str(raw_name).strip())
                    else None
                )
            )
            is not None
            and isinstance(channel_rank, int)
            and not isinstance(channel_rank, bool)
            and channel_rank >= 1
        }
        raw_shot_id = item.get("shot_id")
        shot_id = (
            raw_shot_id.strip()
            if isinstance(raw_shot_id, str)
            and _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(raw_shot_id.strip())
            else f"frame:{source_frame_idx}"
        )
        hit = SearchHit(
            rank=len(result) + 1,
            video_id=video_id,
            source_frame_idx=source_frame_idx,
            timestamp_ms=int(item.get("timestamp_ms", 0)),
            shot_id=shot_id,
            fused_score=float(item.get("fused_score", item.get("score", 0.0))),
            thumbnail_uri=thumbnail_uri,
            channel_ranks=channel_ranks,
            evidence=_safe_hit_evidence(item.get("evidence", ())),
        )
        public = hit.to_dict()
        public["frame_semantics"] = "source_frame_idx"
        if batch_id is not None:
            public["batch_id"] = batch_id
        for field in ("keyframe_ordinal", "coarse_source_frame_idx"):
            field_value = item.get(field)
            if (
                isinstance(field_value, int)
                and not isinstance(field_value, bool)
                and field_value >= 0
            ):
                public[field] = field_value
        result.append(public)
    return result


def _search_payload(context: RuntimeContext, request: QueryRequest) -> dict[str, Any]:
    hits = context.pipeline.search(request, limit=request.limit)
    raw_errors = getattr(context.pipeline, "last_channel_errors", {})
    statuses = getattr(context.pipeline, "last_channel_statuses", {})
    safe_errors: dict[str, str] = {}
    if isinstance(raw_errors, dict):
        for raw_name in raw_errors:
            name = _safe_identifier(str(raw_name))
            if name is None:
                continue
            status = statuses.get(raw_name, {}) if isinstance(statuses, dict) else {}
            reason = status.get("reason_code") if isinstance(status, dict) else None
            safe_errors[name] = _safe_identifier(reason) or "channel_unavailable"
    safe_statuses = _safe_channel_statuses(statuses)
    attempted = [
        status
        for status in safe_statuses.values()
        if status.get("state") not in {"not_provisioned", "disabled_by_profile"}
    ]
    successful = any(status.get("state") == "ready" for status in attempted)
    if not hits and attempted and not successful:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "all_channels_failed",
                "message": "all provisioned retrieval channels failed",
                "channels": safe_errors,
                "channel_statuses": safe_statuses,
            },
        )
    return {
        "query_id": request.query_id,
        "query_plan": context.pipeline._plan(request).to_dict(),
        "hits": _public_hits(hits),
        "channel_errors": safe_errors,
        "channel_statuses": safe_statuses,
        "query_processing": _safe_query_processing(
            getattr(context.pipeline, "last_query_processing", {})
        ),
    }


def _qa_pipeline(context: RuntimeContext) -> Any:
    pipeline = getattr(context, "qa_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Visual Q&A is unavailable")
    return pipeline


def _trake_pipeline(context: RuntimeContext) -> Any:
    pipeline = getattr(context, "trake_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "trake_not_ready",
                "message": "TRAKE retrieval is unavailable",
                "channel_statuses": {},
            },
        )
    return pipeline


def _trake_failure_statuses(pipeline: Any, batch_id: str | None) -> dict[str, dict[str, Any]]:
    """Return allowlisted TRAKE readiness after a failed query."""

    if not batch_id:
        return {}
    probe = getattr(pipeline, "channel_readiness", None)
    if not callable(probe):
        return {}
    try:
        _, statuses = probe(batch_id)
    except (TypeError, ValueError, RuntimeError):
        return {}
    return _safe_channel_statuses(statuses)


def _safe_identifier(value: Any) -> str | None:
    """Return a bounded operational identifier, never arbitrary provider text."""

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate if _SAFE_IDENTIFIER.fullmatch(candidate) else None


def _safe_qa_frame(value: Any, *, outcome: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        source_frame_idx = int(value["source_frame_idx"])
        timestamp_ms = int(value.get("timestamp_ms", 0))
        position = int(value.get("position", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if source_frame_idx < 0 or timestamp_ms < 0 or position < 0:
        return None
    raw_batch_id = value.get("batch_id") or outcome.get("batch_id")
    raw_video_id = value.get("video_id") or outcome.get("video_id")
    batch_id = (
        raw_batch_id.strip()
        if isinstance(raw_batch_id, str) and _SAFE_FRAME_IDENTIFIER.fullmatch(raw_batch_id.strip())
        else None
    )
    video_id = (
        raw_video_id.strip()
        if isinstance(raw_video_id, str) and _SAFE_FRAME_IDENTIFIER.fullmatch(raw_video_id.strip())
        else None
    )
    if batch_id is None or video_id is None:
        return None
    return {
        "position": position,
        "batch_id": batch_id,
        "video_id": video_id,
        "source_frame_idx": source_frame_idx,
        "timestamp_ms": timestamp_ms,
    }


def _safe_qa_outcome(value: Any) -> dict[str, Any] | None:
    """Project a persisted outcome onto the public, secret-free contract."""

    if not isinstance(value, dict):
        return None
    raw_window_id = value.get("window_id")
    window_id = (
        raw_window_id.strip()
        if isinstance(raw_window_id, str)
        and _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(raw_window_id.strip())
        else None
    )
    status = value.get("status")
    if window_id is None or status not in _QA_OUTCOME_STATUSES:
        return None
    failure_code = value.get("failure_code")
    failure_stage = value.get("failure_stage")
    if str(status).endswith("_failed"):
        if failure_code not in _QA_FAILURE_CODES:
            status = "internal_failed"
            failure_code = "internal_reasoning_error"
            failure_stage = "internal"
        elif str(failure_code).startswith("media_"):
            status = "media_failed"
            failure_stage = "media"
        elif str(failure_code).startswith("provider_"):
            status = "provider_failed"
            failure_stage = "provider"
        else:
            status = "internal_failed"
            failure_code = "internal_reasoning_error"
            failure_stage = "internal"
    raw_batch_id = value.get("batch_id")
    raw_video_id = value.get("video_id")
    batch_id = (
        raw_batch_id.strip()
        if isinstance(raw_batch_id, str) and _SAFE_FRAME_IDENTIFIER.fullmatch(raw_batch_id.strip())
        else None
    )
    video_id = (
        raw_video_id.strip()
        if isinstance(raw_video_id, str) and _SAFE_FRAME_IDENTIFIER.fullmatch(raw_video_id.strip())
        else None
    )
    frames = (
        [
            frame
            for item in value.get("frames", ())
            if (frame := _safe_qa_frame(item, outcome=value)) is not None
        ]
        if isinstance(value.get("frames", ()), (list, tuple))
        else []
    )
    result: dict[str, Any] = {
        "window_id": window_id,
        "status": status,
        "frames": frames,
    }
    for name, item in (("batch_id", batch_id), ("video_id", video_id)):
        if item is not None:
            result[name] = item
    for name, minimum in (
        ("source_frame_idx", 0),
        ("timestamp_ms", 0),
        ("retrieval_rank", 1),
        ("attempts", 1),
    ):
        try:
            item = int(value[name])
        except (KeyError, TypeError, ValueError):
            continue
        if item >= minimum:
            result[name] = item
    for name in (
        "retrieval_score",
        "frame_prepare_ms",
        "provider_latency_ms",
        "served_latency_ms",
    ):
        try:
            item = float(value[name])
        except (KeyError, TypeError, ValueError):
            continue
        if item >= 0:
            result[name] = item
    for name in ("shot_id", "provider", "prompt_version", "cache_status"):
        raw_item = value.get(name)
        item = (
            raw_item.strip()
            if isinstance(raw_item, str)
            and _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(raw_item.strip())
            else None
        )
        if item is not None:
            result[name] = item
    model_name = _safe_model_identifier(value.get("model_name"))
    if model_name is not None:
        result["model_name"] = model_name
    for name in ("raw_answer", "normalized_answer", "answer"):
        item = value.get(name)
        if isinstance(item, str):
            result[name] = item
    answerable = value.get("answerable")
    if isinstance(answerable, bool):
        result["answerable"] = answerable
    evidence = value.get("evidence_frame_indices", ())
    if isinstance(evidence, (list, tuple)):
        result["evidence_frame_indices"] = [
            int(item)
            for item in evidence
            if isinstance(item, int) and not isinstance(item, bool) and item >= 1
        ]
    cache_layers = value.get("cache_layers")
    if isinstance(cache_layers, dict):
        safe_layers: dict[str, bool | int | float | str] = {}
        for name, item in cache_layers.items():
            safe_name = _safe_identifier(str(name))
            if safe_name is None or not isinstance(item, (bool, int, float, str)):
                continue
            if isinstance(item, str) and _safe_identifier(item) is None:
                continue
            safe_layers[safe_name] = item
        result["cache_layers"] = safe_layers
    frame_scores = value.get("frame_selection_scores")
    if isinstance(frame_scores, dict):
        selected_frames = {frame["source_frame_idx"] for frame in frames}
        safe_scores: dict[str, float] = {}
        for raw_frame, raw_score in frame_scores.items():
            try:
                source_frame_idx = int(raw_frame)
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if (
                source_frame_idx in selected_frames
                and isinstance(raw_score, (int, float))
                and not isinstance(raw_score, bool)
                and math.isfinite(score)
            ):
                safe_scores[str(source_frame_idx)] = score
        if safe_scores:
            result["frame_selection_scores"] = safe_scores
    if str(status).endswith("_failed"):
        result["failure_code"] = failure_code
        result["failure_stage"] = (
            failure_stage if failure_stage in _QA_FAILURE_STAGES else "internal"
        )
        category = value.get("provider_error_category")
        if category in _QA_PROVIDER_ERROR_CATEGORIES:
            result["provider_error_category"] = category
        provider_status = value.get("provider_status_code")
        if (
            isinstance(provider_status, int)
            and not isinstance(provider_status, bool)
            and 100 <= provider_status <= 599
        ):
            result["provider_status_code"] = provider_status
    return result


def _safe_model_identifier(value: Any) -> str | None:
    """Accept a bounded provider model ID while rejecting path-like values."""

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 255 or "\\" in candidate or candidate.startswith("/"):
        return None
    parts = candidate.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        return None
    if any(re.fullmatch(r"[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?", part) is None for part in parts):
        return None
    return candidate


def _safe_qa_prediction(value: Any) -> dict[str, Any] | None:
    """Project a persisted QA prediction without media URIs or arbitrary evidence."""

    if not isinstance(value, dict):
        return None
    batch_id = value.get("batch_id")
    video_id = value.get("video_id")
    window_id = value.get("window_id")
    if (
        not isinstance(batch_id, str)
        or _SAFE_FRAME_IDENTIFIER.fullmatch(batch_id.strip()) is None
        or not isinstance(video_id, str)
        or _SAFE_FRAME_IDENTIFIER.fullmatch(video_id.strip()) is None
        or not isinstance(window_id, str)
        or _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(window_id.strip()) is None
    ):
        return None
    try:
        rank = int(value["rank"])
        source_frame_idx = int(value["source_frame_idx"])
        timestamp_ms = int(value.get("timestamp_ms", 0))
        confidence_score = float(value["confidence_score"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        rank < 1
        or source_frame_idx < 0
        or timestamp_ms < 0
        or not math.isfinite(confidence_score)
        or not 0 <= confidence_score <= 1
    ):
        return None
    answer = value.get("answer")
    if not isinstance(answer, str) or not answer:
        return None
    evidence_value = value.get("evidence", {})
    evidence_value = evidence_value if isinstance(evidence_value, dict) else {}
    evidence: dict[str, Any] = {}
    for name, minimum in (
        ("representative_source_frame_idx", 0),
        ("retrieval_rank", 1),
        ("attempts", 1),
    ):
        item = evidence_value.get(name)
        if isinstance(item, int) and not isinstance(item, bool) and item >= minimum:
            evidence[name] = item
    for name in ("retrieval_score", "latency_ms", "answer_call_ms"):
        item = evidence_value.get(name)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            number = float(item)
            if math.isfinite(number) and number >= 0:
                evidence[name] = number
    for name in (
        "shot_id",
        "provider",
        "cache_status",
        "requested_policy",
        "fallback_reason_code",
    ):
        item = evidence_value.get(name)
        if isinstance(item, str) and _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(item.strip()):
            evidence[name] = item.strip()
    evidence["fallback_used"] = bool(evidence_value.get("fallback_used", False))
    for name in ("evidence_frame_indices", "evidence_source_frame_indices"):
        items = evidence_value.get(name, ())
        if isinstance(items, (list, tuple)):
            minimum = 1 if name == "evidence_frame_indices" else 0
            evidence[name] = [
                item
                for item in items[:5]
                if isinstance(item, int) and not isinstance(item, bool) and item >= minimum
            ]
    frames: list[dict[str, int]] = []
    raw_frames = evidence_value.get("window_frames", ())
    if isinstance(raw_frames, (list, tuple)):
        for fallback_position, frame in enumerate(raw_frames[:5], 1):
            if not isinstance(frame, dict):
                continue
            try:
                position = int(frame.get("position", fallback_position))
                frame_idx = int(frame["source_frame_idx"])
                frame_timestamp = int(frame.get("timestamp_ms", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if position >= 1 and frame_idx >= 0 and frame_timestamp >= 0:
                frames.append(
                    {
                        "position": position,
                        "source_frame_idx": frame_idx,
                        "timestamp_ms": frame_timestamp,
                    }
                )
    if frames:
        evidence["window_frames"] = frames
    provider = evidence.get("provider")
    model_name = _safe_model_identifier(value.get("model_name"))
    answer_source = (
        "/".join(item for item in (provider, model_name) if isinstance(item, str)) or "qa_reasoner"
    )
    if value.get("answer_source") == "operator":
        answer_source = "operator"
    result: dict[str, Any] = {
        "rank": rank,
        "batch_id": batch_id.strip(),
        "video_id": video_id.strip(),
        "source_frame_idx": source_frame_idx,
        "timestamp_ms": timestamp_ms,
        "answer": answer,
        "confidence_score": confidence_score,
        "answer_source": answer_source,
        "window_id": window_id.strip(),
        "evidence": evidence,
    }
    raw_answer = value.get("raw_answer")
    if isinstance(raw_answer, str):
        result["raw_answer"] = raw_answer[:4_096]
    if model_name is not None:
        result["model_name"] = model_name
    return result


def _safe_operator_edit(value: Any) -> dict[str, Any] | None:
    """Allowlist persisted edit provenance while retaining operator-visible answers."""

    if not isinstance(value, dict):
        return None
    window_id = value.get("window_id")
    if (
        not isinstance(window_id, str)
        or _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(window_id.strip()) is None
    ):
        return None
    result: dict[str, Any] = {"window_id": window_id.strip()}
    edit_id = value.get("edit_id")
    if isinstance(edit_id, int) and not isinstance(edit_id, bool) and edit_id >= 1:
        result["edit_id"] = edit_id
    for name in ("created_at", "edited_at"):
        item = value.get(name)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            number = float(item)
            if math.isfinite(number) and number >= 0:
                result[name] = number
    for name in (
        "original_raw_answer",
        "edited_raw_answer",
        "normalized_answer",
    ):
        item = value.get(name)
        if isinstance(item, str):
            result[name] = item[:4_096]
    previous = value.get("previous_normalized_answers", ())
    if isinstance(previous, (list, tuple)):
        result["previous_normalized_answers"] = [
            item[:4_096] for item in previous[:20] if isinstance(item, str)
        ]
    provider = value.get("provider")
    if isinstance(provider, str) and _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(provider.strip()):
        result["provider"] = provider.strip()
    model_name = _safe_model_identifier(value.get("model_name"))
    if model_name is not None:
        result["model_name"] = model_name
    return result


def _legacy_qa_outcomes(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthesize a minimal outcome view for pre-v2 sessions."""

    outcomes: dict[str, dict[str, Any]] = {}
    targets_by_window: dict[str, dict[str, Any]] = {}
    targets = state.get("targets", ())
    if isinstance(targets, (list, tuple)):
        for target in targets:
            if not isinstance(target, dict):
                continue
            window_id = target.get("window_id")
            if not isinstance(window_id, str) or not window_id:
                batch = target.get("batch_id")
                video = target.get("video_id")
                frame = target.get("source_frame_idx")
                if isinstance(batch, str) and isinstance(video, str) and isinstance(frame, int):
                    window_id = f"{batch}:{video}:{frame}"
            if isinstance(window_id, str) and window_id:
                targets_by_window[window_id] = target
    predictions = state.get("predictions", ())
    predictions_by_window: dict[str, dict[str, Any]] = {}
    if isinstance(predictions, (list, tuple)):
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            window_id = _safe_identifier(prediction.get("window_id"))
            batch_id = _safe_identifier(prediction.get("batch_id"))
            video_id = _safe_identifier(prediction.get("video_id"))
            if window_id is None or batch_id is None or video_id is None:
                continue
            predictions_by_window.setdefault(window_id, prediction)
            evidence = prediction.get("evidence", {})
            evidence = evidence if isinstance(evidence, dict) else {}
            frames = evidence.get("window_frames", ())
            outcome = outcomes.setdefault(
                window_id,
                {
                    "window_id": window_id,
                    "status": "answered",
                    "batch_id": batch_id,
                    "video_id": video_id,
                    "source_frame_idx": prediction.get("source_frame_idx"),
                    "timestamp_ms": prediction.get("timestamp_ms", 0),
                    "frames": frames if isinstance(frames, (list, tuple)) else [],
                    "raw_answer": prediction.get("raw_answer", ""),
                    "normalized_answer": prediction.get("answer", ""),
                    "answer": prediction.get("answer", ""),
                    "answerable": True,
                    "evidence_frame_indices": evidence.get("evidence_frame_indices", ()),
                    "provider": evidence.get("provider"),
                    "model_name": prediction.get("model_name"),
                    "provider_latency_ms": evidence.get("latency_ms"),
                    "attempts": evidence.get("attempts"),
                    "cache_status": evidence.get("cache_status"),
                },
            )
            if not outcome["frames"]:
                outcome["frames"] = [
                    {
                        "position": len(outcome["frames"]) + 1,
                        "batch_id": batch_id,
                        "video_id": video_id,
                        "source_frame_idx": prediction.get("source_frame_idx"),
                        "timestamp_ms": prediction.get("timestamp_ms", 0),
                    }
                ]
    results = state.get("results", {})
    if isinstance(results, dict):
        for target, reasoning in results.items():
            window_id = _safe_identifier(str(target))
            if window_id is None or window_id in outcomes or not isinstance(reasoning, dict):
                continue
            answerable = bool(reasoning.get("answerable", False))
            outcomes[window_id] = {
                "window_id": window_id,
                "status": "answered" if answerable else "unanswerable",
                "frames": [],
                "raw_answer": reasoning.get("raw_answer", ""),
                "normalized_answer": reasoning.get(
                    "normalized_answer", reasoning.get("answer", "")
                ),
                "answer": reasoning.get("answer", ""),
                "answerable": answerable,
                "evidence_frame_indices": reasoning.get("evidence_frame_indices", ()),
                "provider": reasoning.get("provider"),
                "model_name": reasoning.get("model_name"),
                "prompt_version": reasoning.get("prompt_version"),
                "provider_latency_ms": reasoning.get("latency_ms"),
                "attempts": reasoning.get("attempts"),
                "cache_status": reasoning.get("cache_status"),
            }
    failures = state.get("failures", {})
    if isinstance(failures, dict):
        for target, failure_value in failures.items():
            window_id = _safe_identifier(str(target))
            if window_id is not None and window_id not in outcomes:
                failure_code = (
                    failure_value
                    if failure_value in _QA_FAILURE_CODES
                    else "internal_reasoning_error"
                )
                if str(failure_code).startswith("media_"):
                    status, stage = "media_failed", "media"
                elif str(failure_code).startswith("provider_"):
                    status, stage = "provider_failed", "provider"
                else:
                    status, stage = "internal_failed", "internal"
                outcomes[window_id] = {
                    "window_id": window_id,
                    "status": status,
                    "frames": [],
                    "failure_code": failure_code,
                    "failure_stage": stage,
                }
    return [
        item
        for value in outcomes.values()
        if (
            item := _safe_qa_outcome(
                _enrich_legacy_qa_outcome(
                    value,
                    targets=targets_by_window,
                    predictions=predictions_by_window,
                )
            )
        )
    ]


def _enrich_legacy_qa_outcome(
    value: Any,
    *,
    targets: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> Any:
    """Recover a URI-free frame strip for a persisted pre-v2 outcome."""

    if not isinstance(value, dict):
        return value
    outcome = dict(value)
    window_id = str(outcome.get("window_id", ""))
    target = targets.get(window_id, {})
    prediction = predictions.get(window_id, {})
    for name in ("batch_id", "video_id", "source_frame_idx", "timestamp_ms", "shot_id"):
        if outcome.get(name) is None and target.get(name) is not None:
            outcome[name] = target[name]
    frames = outcome.get("frames")
    if not isinstance(frames, (list, tuple)) or not frames:
        evidence = prediction.get("evidence", {})
        window_frames = evidence.get("window_frames", ()) if isinstance(evidence, dict) else ()
        if isinstance(window_frames, (list, tuple)) and window_frames:
            outcome["frames"] = list(window_frames[:5])
        elif all(
            outcome.get(name) is not None for name in ("batch_id", "video_id", "source_frame_idx")
        ):
            outcome["frames"] = [
                {
                    "position": 1,
                    "batch_id": outcome["batch_id"],
                    "video_id": outcome["video_id"],
                    "source_frame_idx": outcome["source_frame_idx"],
                    "timestamp_ms": outcome.get("timestamp_ms", 0),
                }
            ]
    return outcome


def _qa_operational_trace(outcome: dict[str, Any]) -> dict[str, Any]:
    trace = {
        "window_id": outcome["window_id"],
        "outcome": outcome["status"],
        "provider": outcome.get("provider"),
        "model_name": outcome.get("model_name"),
        "cache_status": outcome.get("cache_status"),
        "provider_latency_ms": outcome.get("provider_latency_ms"),
        "served_latency_ms": outcome.get("served_latency_ms"),
        "attempts": outcome.get("attempts"),
    }
    if outcome.get("cache_layers"):
        trace["cache_layers"] = outcome["cache_layers"]
    if outcome["status"].endswith("_failed"):
        trace["failure_code"] = outcome.get("failure_code")
        trace["failure_stage"] = outcome.get("failure_stage")
        if outcome.get("provider_error_category") in _QA_PROVIDER_ERROR_CATEGORIES:
            trace["provider_error_category"] = outcome["provider_error_category"]
        if isinstance(outcome.get("provider_status_code"), int):
            trace["provider_status_code"] = outcome["provider_status_code"]
    return trace


def _safe_channel_statuses(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_status in value.items():
        name = _safe_identifier(str(raw_name))
        if name is None or not isinstance(raw_status, dict):
            continue
        status: dict[str, Any] = {}
        if raw_status.get("state") in _QA_CHANNEL_STATES:
            status["state"] = raw_status["state"]
        if raw_status.get("applicability") in _QA_CHANNEL_APPLICABILITY:
            status["applicability"] = raw_status["applicability"]
        for field in ("reason_code", "collection_name", "index_name", "remediation_id"):
            item = _safe_identifier(raw_status.get(field))
            if item is not None:
                status[field] = item
        for field in (
            "count",
            "expected_dimension",
            "token_count",
            "context_limit",
            "content_limit",
            "chunk_count",
            "processed_tokens",
            "dropped_tokens",
            "query_variant_count",
            "result_count",
        ):
            item = raw_status.get(field)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                status[field] = item
        for field in ("query_chunked", "query_truncated", "translation_applied"):
            item = raw_status.get(field)
            if isinstance(item, bool):
                status[field] = item
        coverage = raw_status.get("coverage")
        if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) and coverage >= 0:
            status["coverage"] = float(coverage)
        latency = raw_status.get("latency_ms")
        if (
            isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and 0 <= float(latency) <= 600_000
        ):
            status["latency_ms"] = round(float(latency), 3)
        if status:
            result[name] = status
    return result


def _safe_query_processing(value: Any) -> dict[str, Any]:
    """Allowlist query-routing diagnostics intended for the operator UI."""

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for field in (
        "detected_language",
        "translation_source",
        "translation_cache",
        "failure_code",
    ):
        item = _safe_identifier(value.get(field))
        if item is not None:
            result[field] = item
    for field in ("english_text", "original_text", "normalized_text"):
        item = value.get(field)
        if isinstance(item, str) and item:
            result[field] = item[:4096]
    latency = value.get("translation_latency_ms")
    if (
        isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and 0 <= float(latency) <= 600_000
    ):
        result["translation_latency_ms"] = round(float(latency), 3)
    applied = value.get("translation_applied")
    if isinstance(applied, bool):
        result["translation_applied"] = applied
    for field in ("prompt_count", "translated_prompt_count"):
        item = value.get(field)
        if isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 64:
            result[field] = item
    policies = value.get("channel_policies")
    if isinstance(policies, Mapping):
        safe_policies: dict[str, dict[str, Any]] = {}
        for raw_name, raw_policy in policies.items():
            name = _safe_identifier(str(raw_name))
            if name is None:
                continue
            policy: dict[str, Any] = {}
            if isinstance(raw_policy, (list, tuple)):
                values = [
                    item
                    for value in raw_policy
                    if (item := _safe_identifier(value)) is not None
                ]
                if values:
                    policy["policy"] = "+".join(values[:8])
                    policy["variant_count"] = len(values[:8])
                raw_policy = {}
            if not isinstance(raw_policy, Mapping):
                continue
            for field in ("policy", "failure_code"):
                item = _safe_identifier(raw_policy.get(field))
                if item is not None:
                    policy[field] = item
            count = raw_policy.get("variant_count")
            if isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 16:
                policy["variant_count"] = count
            if policy:
                safe_policies[name] = policy
        if safe_policies:
            result["channel_policies"] = safe_policies
    return result


def _safe_qa_snapshot(context: RuntimeContext, search_id: str) -> dict[str, Any]:
    """Return a restorable QA response without raw provider or retrieval errors."""

    pipeline = _qa_pipeline(context)
    snapshot = pipeline.get_search(search_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Visual Q&A search is missing or expired")
    if int(snapshot.get("state_version", 0)) >= 3:
        raw_candidates = snapshot.get("session_candidates", ())
        candidates: list[dict[str, Any]] = []
        if isinstance(raw_candidates, (list, tuple)):
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping):
                    continue
                try:
                    candidate = QACandidateFrame.from_dict(raw_candidate)
                except (TypeError, ValueError):
                    continue
                public = candidate.to_dict()
                public["thumbnail_url"] = (
                    f"/v1/frames/{candidate.batch_id}/{candidate.video_id}/"
                    f"{candidate.source_frame_idx}/image"
                )
                # Dense/lexical evidence can contain backend-specific payloads.
                # Public QA retrieval exposes only stable channel ranks.
                public["evidence"] = {"channel_ranks": dict(candidate.channel_ranks)}
                candidates.append(public)
        raw_predictions = snapshot.get("predictions", ())
        predictions = [
            prediction
            for value in raw_predictions
            if isinstance(value, dict)
            and (prediction := _safe_qa_prediction(value)) is not None
        ]
        state = snapshot.get("state", {})
        state = state if isinstance(state, Mapping) else {}
        answers = snapshot.get("answers", {})
        safe_answers: list[dict[str, Any]] = []
        if isinstance(answers, Mapping):
            for candidate_id, value in answers.items():
                if not isinstance(candidate_id, str) or not isinstance(value, Mapping):
                    continue
                reasoning = value.get("reasoning")
                reasoning = reasoning if isinstance(reasoning, Mapping) else {}
                timing = value.get("stage_timings")
                timing = timing if isinstance(timing, Mapping) else {}
                safe_answers.append(
                    {
                        "candidate_id": candidate_id,
                        "status": str(value.get("status") or "unanswerable"),
                        "answer": str(reasoning.get("answer") or ""),
                        "answerable": bool(reasoning.get("answerable", False)),
                        "provider": _safe_identifier(reasoning.get("provider")),
                        "model_name": _safe_identifier(reasoning.get("model_name")),
                        "requested_policy": _safe_identifier(
                            reasoning.get("requested_policy")
                        ),
                        "fallback_used": bool(reasoning.get("fallback_used", False)),
                        "fallback_reason_code": _safe_identifier(
                            reasoning.get("fallback_reason_code")
                        ),
                        "cache_status": _safe_identifier(reasoning.get("cache_status")),
                        "latency_ms": (
                            float(reasoning["latency_ms"])
                            if isinstance(reasoning.get("latency_ms"), (int, float))
                            and not isinstance(reasoning.get("latency_ms"), bool)
                            else None
                        ),
                        "stage_timings": {
                            str(name): round(float(number), 3)
                            for name, number in timing.items()
                            if isinstance(name, str)
                            and isinstance(number, (int, float))
                            and not isinstance(number, bool)
                            and float(number) >= 0
                        },
                    }
                )
        request = snapshot.get("request", {})
        safe_request = dict(request) if isinstance(request, Mapping) else {}
        if isinstance(safe_request.get("legacy_request"), Mapping):
            safe_request["legacy_request"] = {
                key: value
                for key, value in safe_request["legacy_request"].items()
                if key
                in {
                    "query_id",
                    "event_description",
                    "question",
                    "mode",
                    "answer_spec_id",
                    "batch_filters",
                    "limit",
                    "reasoning_policy",
                    "allow_external_frame_transfer",
                }
            }
        return {
            "search_id": search_id,
            "query_id": safe_request.get("query_id"),
            "state_version": 3,
            "status": str(snapshot.get("status") or "candidates_ready"),
            "request": safe_request,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "selected_candidate_id": snapshot.get("selected_candidate_id"),
            "answers": safe_answers,
            "predictions": predictions,
            "stage_timings": {
                "candidate_retrieval_ms": round(
                    float(snapshot.get("candidate_retrieval_ms", 0.0)), 3
                )
                if isinstance(snapshot.get("candidate_retrieval_ms"), (int, float))
                and not isinstance(snapshot.get("candidate_retrieval_ms"), bool)
                else 0.0
            },
            "candidate_cache_status": _safe_identifier(
                state.get("candidate_cache_status")
            ),
            "channel_statuses": _safe_channel_statuses(
                snapshot.get("channel_statuses", {})
            ),
            "query_processing": _safe_query_processing(
                snapshot.get("query_processing", state.get("query_processing", {}))
            ),
            "retrieval_errors": {
                str(name): "channel_unavailable"
                for name in snapshot.get("retrieval_errors", {})
            }
            if isinstance(snapshot.get("retrieval_errors"), Mapping)
            else {},
            "failures": {
                str(candidate_id): _safe_identifier(code) or "internal_reasoning_error"
                for candidate_id, code in snapshot.get("failures", {}).items()
            }
            if isinstance(snapshot.get("failures"), Mapping)
            else {},
            "expires_at": snapshot.get("expires_at"),
        }
    state = snapshot.get("state", {})
    state = state if isinstance(state, dict) else {}
    raw_predictions = [
        dict(item) for item in state.get("predictions", ()) if isinstance(item, dict)
    ]
    predictions = [
        prediction
        for value in raw_predictions
        if (prediction := _safe_qa_prediction(value)) is not None
    ]
    qa = context.config.get("retrieval", {}).get("visual_qa", {})
    maximum = int(qa.get("max_clusters", 20))
    step = int(qa.get("expansion_step", 5))
    target_count = int(snapshot.get("target_count", snapshot.get("requested_target_clusters", 0)))
    requested_target = int(snapshot.get("requested_target_clusters", target_count))
    retrieved_clusters = int(snapshot.get("retrieved_clusters", target_count))
    retrieval_exhausted = bool(snapshot.get("retrieval_exhausted", False))
    request = snapshot.get("request", {})
    try:
        safe_request = QAQueryRequest.from_dict(request).to_dict()
    except (TypeError, ValueError):
        safe_request = {}
    next_step = 1 if safe_request.get("mode", "interactive") == "interactive" else step
    next_target = (
        min(maximum, requested_target + next_step)
        if requested_target < maximum and not retrieval_exhausted
        else None
    )
    raw_outcomes = snapshot.get("window_outcomes", state.get("window_outcomes", ()))
    if isinstance(raw_outcomes, dict):
        raw_outcomes = list(raw_outcomes.values())
    raw_targets = snapshot.get("targets", state.get("targets", ()))
    targets_by_window: dict[str, dict[str, Any]] = {}
    if isinstance(raw_targets, (list, tuple)):
        for value in raw_targets:
            if not isinstance(value, dict):
                continue
            window_id = value.get("window_id")
            if not isinstance(window_id, str) or not window_id:
                batch = value.get("batch_id")
                video = value.get("video_id")
                frame = value.get("source_frame_idx")
                if isinstance(batch, str) and isinstance(video, str) and isinstance(frame, int):
                    window_id = f"{batch}:{video}:{frame}"
            if isinstance(window_id, str) and window_id:
                targets_by_window[window_id] = value
    predictions_by_window = {
        str(value["window_id"]): value
        for value in raw_predictions
        if isinstance(value.get("window_id"), str) and value.get("window_id")
    }
    outcomes = (
        [
            item
            for value in raw_outcomes
            if (
                item := _safe_qa_outcome(
                    _enrich_legacy_qa_outcome(
                        value,
                        targets=targets_by_window,
                        predictions=predictions_by_window,
                    )
                )
            )
        ]
        if isinstance(raw_outcomes, (list, tuple))
        else []
    )
    if not outcomes:
        legacy_state = dict(state)
        legacy_state.setdefault("targets", list(targets_by_window.values()))
        outcomes = _legacy_qa_outcomes(legacy_state)
    raw_candidate_windows = snapshot.get(
        "session_windows",
        state.get("session_windows", ()),
    )
    candidate_windows = []
    if isinstance(raw_candidate_windows, (list, tuple)):
        for value in raw_candidate_windows:
            if not isinstance(value, dict):
                continue
            candidate = _safe_qa_outcome({**value, "status": "pending"})
            if candidate is not None:
                candidate_windows.append(candidate)
    if not candidate_windows:
        candidate_windows = [dict(item) for item in outcomes]
    traces = [_qa_operational_trace(outcome) for outcome in outcomes]
    channel_statuses = _safe_channel_statuses(
        snapshot.get("channel_statuses", state.get("channel_statuses", {}))
    )
    retrieval_errors = snapshot.get("retrieval_errors", state.get("retrieval_errors", {}))
    if isinstance(retrieval_errors, dict):
        safe_retrieval_errors = {
            str(name): str(code) if code in _QA_RETRIEVAL_ERROR_CODES else "channel_unavailable"
            for name, code in retrieval_errors.items()
            if _safe_identifier(str(name)) is not None
        }
    elif isinstance(retrieval_errors, (list, tuple)):
        safe_retrieval_errors = {
            str(name): "channel_unavailable"
            for name in retrieval_errors
            if _safe_identifier(str(name)) is not None
        }
    else:
        safe_retrieval_errors = {}
    safe_provider_errors = {
        outcome["window_id"]: outcome.get("failure_code", "internal_reasoning_error")
        for outcome in outcomes
        if outcome["status"].endswith("_failed")
    }
    answerable_count = int(snapshot.get("answerable_count", 0))
    unanswerable_count = int(snapshot.get("unanswerable_count", 0))
    failed_count = int(snapshot.get("failed_count", 0))
    pending_count = int(snapshot.get("pending_count", 0))
    if outcomes:
        answerable_count = sum(item["status"] == "answered" for item in outcomes)
        unanswerable_count = sum(item["status"] == "unanswerable" for item in outcomes)
        failed_count = sum(item["status"].endswith("_failed") for item in outcomes)
        pending_count = sum(item["status"] == "pending" for item in outcomes)
    reasoned_count = answerable_count + unanswerable_count
    requested_policy = str(request.get("reasoning_policy", "local_only"))
    actual_providers = [
        str(item.get("provider"))
        for item in outcomes
        if isinstance(item.get("provider"), str) and item.get("provider")
    ]
    actual_provider = actual_providers[-1] if actual_providers else None
    status = snapshot.get("status")
    if status not in _QA_SEARCH_STATUSES:
        if retrieved_clusters == 0 and not outcomes:
            status = "no_candidates"
        elif failed_count and not answerable_count and not unanswerable_count:
            status = "failed"
        elif failed_count:
            status = "partial"
        elif pending_count:
            status = "incomplete"
        elif answerable_count:
            status = "completed"
        elif unanswerable_count:
            status = "completed_no_answer"
        else:
            status = "incomplete"
    return {
        "search_id": search_id,
        "request": safe_request,
        "query_id": request.get("query_id"),
        "mode": request.get("mode", "automatic"),
        "answer_spec_id": request.get("answer_spec_id"),
        "requested_policy": requested_policy,
        "actual_provider": actual_provider,
        "fallback_used": requested_policy == "local_then_openrouter"
        and "openrouter" in actual_providers,
        "predictions": predictions,
        "status": status,
        "reasoned_clusters": reasoned_count,
        "answerable_clusters": answerable_count,
        "unanswerable_clusters": unanswerable_count,
        "failed_clusters": failed_count,
        "pending_clusters": pending_count,
        "target_clusters": target_count,
        "requested_target_clusters": requested_target,
        "retrieved_clusters": retrieved_clusters,
        "retrieval_exhausted": retrieval_exhausted,
        "max_clusters": maximum,
        "expansion_available": next_target is not None,
        "next_target_clusters": next_target,
        "window_outcomes": outcomes,
        "candidate_windows": candidate_windows,
        "stage_timings": {
            "candidate_retrieval_ms": float(state.get("candidate_retrieval_ms", 0.0))
            if isinstance(state.get("candidate_retrieval_ms"), (int, float))
            and not isinstance(state.get("candidate_retrieval_ms"), bool)
            and float(state.get("candidate_retrieval_ms", 0.0)) >= 0
            else 0.0,
            "frame_prepare_ms": round(
                sum(
                    float(value)
                    for item in outcomes
                    if isinstance((value := item.get("frame_prepare_ms")), (int, float))
                    and not isinstance(value, bool)
                    and float(value) >= 0
                ),
                3,
            ),
            "ollama_inference_ms": round(
                sum(
                    float(value)
                    for item in outcomes
                    if isinstance((value := item.get("provider_latency_ms")), (int, float))
                    and not isinstance(value, bool)
                    and float(value) >= 0
                ),
                3,
            ),
        },
        "operational_traces": traces,
        "channel_statuses": channel_statuses,
        "retrieval_errors": safe_retrieval_errors,
        "provider_errors": safe_provider_errors,
        "operator_edits": [
            edit
            for value in snapshot.get("operator_edits", ())
            if (edit := _safe_operator_edit(value)) is not None
        ]
        if isinstance(snapshot.get("operator_edits", ()), (list, tuple))
        else [],
        "expires_at": snapshot.get("expires_at"),
    }


def _qa_failure_detail(search_id: str, response: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a recoverable total-failure response from already-sanitized state."""

    detail: dict[str, Any] = {
        "code": "qa_reasoning_unavailable",
        "message": "Visual Q&A providers could not produce a usable answer",
        "search_id": search_id,
        "recoverable": response is not None,
        "status": "failed",
    }
    if response is not None:
        for name in (
            "reasoned_clusters",
            "answerable_clusters",
            "unanswerable_clusters",
            "failed_clusters",
            "pending_clusters",
            "target_clusters",
            "requested_target_clusters",
            "retrieved_clusters",
            "retrieval_exhausted",
            "operational_traces",
            "retrieval_errors",
            "provider_errors",
            "stage_timings",
        ):
            detail[name] = response.get(name)
    return detail


def create_app(context: RuntimeContext) -> FastAPI:
    """Create an API around an injected runtime (convenient for offline tests)."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _release_local_vlm(context)
        warmup = getattr(context, "warmup", None)
        start = getattr(warmup, "start", None)
        if callable(start):
            start()
        yield

    app = FastAPI(
        title="AIC 2026 Multimodal Retrieval",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = context

    @app.post("/v1/kis/search")
    def search(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = _with_runtime_batches(context, QueryRequest.from_dict(payload))
            _release_local_vlm(context)
            _require_search_usable(context, "kis")
            return _search_payload(context, request)
        except HTTPException:
            raise
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail="KIS request is invalid or its selected batch is unavailable",
            ) from None

    @app.post("/v1/kis/refine")
    def refine(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = _with_runtime_batches(context, QueryRequest.from_dict(payload))
            _release_local_vlm(context)
            _require_search_usable(context, "kis")
            hits = context.pipeline.refine(
                request,
                positive_ids=tuple(_feedback_key(value) for value in request.positive_feedback_ids),
                negative_ids=tuple(_feedback_key(value) for value in request.negative_feedback_ids),
                limit=request.limit,
            )
            return {
                "query_id": request.query_id,
                "hits": _public_hits(hits),
                "channel_statuses": _safe_channel_statuses(
                    getattr(context.pipeline, "last_channel_statuses", {})
                ),
                "query_processing": _safe_query_processing(
                    getattr(context.pipeline, "last_query_processing", {})
                ),
            }
        except RuntimeError:
            raise HTTPException(
                status_code=409,
                detail="feedback refinement is unavailable for this request",
            ) from None
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422, detail="KIS refinement request is invalid"
            ) from None

    @app.post("/api/v1/search/trake", deprecated=True)
    @app.post("/v1/trake/search")
    def trake_search(payload: dict[str, Any]) -> dict[str, Any]:
        request: TRAKEQueryRequest | None = None
        pipeline: Any = None
        try:
            request = TRAKEQueryRequest.model_validate(payload)
            resolved = _with_runtime_batches(context, request)
            if not isinstance(resolved, TRAKEQueryRequest):
                raise TypeError("invalid TRAKE request type")
            request = resolved
            _release_local_vlm(context)
            _require_search_usable(context, "trake")
            pipeline = _trake_pipeline(context)
            response = pipeline.search(
                request,
                default_batch_id=getattr(context, "selected_batch_id", None),
            )
            public = response.model_dump(mode="json", by_alias=True)
            public["channel_statuses"] = _safe_channel_statuses(public.get("channel_statuses"))
            public["query_processing"] = _safe_query_processing(
                getattr(pipeline, "last_query_processing", {})
            )
            return public
        except TRAKENotReadyError as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "trake_not_ready",
                    "message": "the required TRAKE dense channel is not ready",
                    "channel_statuses": _safe_channel_statuses(error.channel_statuses),
                },
            ) from None
        except HTTPException:
            raise
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail="TRAKE request is invalid or its selected batch is unavailable",
            ) from None
        except Exception:
            batch_id = request.batch_filters[0] if request and request.batch_filters else None
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "trake_search_failed",
                    "message": "TRAKE dense retrieval failed",
                    "channel_statuses": _trake_failure_statuses(pipeline, batch_id),
                },
            ) from None

    def _qa_retrieve(payload: dict[str, Any]) -> dict[str, Any]:
        search_id: str | None = None
        try:
            legacy_request: dict[str, Any] | None = None
            if "question" in payload:
                legacy = QAQueryRequest.from_dict(payload)
                legacy = _with_runtime_batches(context, legacy)
                if not isinstance(legacy, QAQueryRequest):
                    raise TypeError("invalid legacy QA request")
                legacy_request = legacy.to_dict()
                request = QACandidateRequest(
                    query_id=legacy.query_id,
                    event_description=legacy.event_description,
                    batch_filters=legacy.batch_filters,
                    top_k=int(
                        context.config.get("retrieval", {})
                        .get("visual_qa", {})
                        .get("candidate_cards", 5)
                    ),
                    query_language=legacy.query_language,
                    event_description_en=legacy.event_description_en,
                )
            else:
                request = QACandidateRequest.from_dict(payload)
                request = _with_runtime_batches(context, request)
                if not isinstance(request, QACandidateRequest):
                    raise TypeError("invalid QA candidate request")
            _release_local_vlm_for_qa_retrieval(context, request.batch_filters[0])
            _require_search_usable(context, "kis")
            pipeline = _qa_pipeline(context)
            search_id = uuid4().hex
            pipeline.start_candidate_search(
                request,
                session_id=search_id,
                legacy_request=legacy_request,
            )
            return _safe_qa_snapshot(context, search_id)
        except HTTPException:
            raise
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail={"code": "qa_retrieval_request_invalid"},
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "qa_retrieval_unavailable",
                    "search_id": search_id,
                    "recoverable": search_id is not None,
                },
            ) from None

    @app.post("/v1/qa/retrieve")
    def qa_retrieve(payload: dict[str, Any]) -> dict[str, Any]:
        return _qa_retrieve(payload)

    @app.post("/api/v1/search/qa", deprecated=True)
    @app.post("/v1/qa/search", deprecated=True)
    def qa_search(payload: dict[str, Any]) -> dict[str, Any]:
        """Deprecated retrieval-only alias; it never invokes a VLM."""

        return _qa_retrieve(payload)

    @app.post("/v1/qa/search/{search_id}/answer")
    def qa_answer(search_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            answer_request = QASingleFrameAnswerRequest.from_dict(payload)
            pipeline = _qa_pipeline(context)
            snapshot = pipeline.get_search(search_id)
            if snapshot is None:
                raise KeyError(search_id)
            if int(snapshot.get("state_version", 0)) != 3:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "legacy_qa_session_read_only"},
                )
            candidates = snapshot.get("session_candidates", ())
            selected = next(
                (
                    QACandidateFrame.from_dict(value)
                    for value in candidates
                    if isinstance(value, Mapping)
                    and value.get("candidate_id") == answer_request.candidate_id
                ),
                None,
            )
            if selected is None:
                raise ValueError("candidate is outside the session")
            if answer_request.answer_engine == "vlm":
                request = QAQueryRequest(
                    query_id=search_id,
                    event_description=str(snapshot["request"]["event_description"]),
                    question=answer_request.question,
                    mode="interactive",
                    answer_spec_id=answer_request.answer_spec_id,
                    batch_filters=(selected.batch_id,),
                    limit=1,
                    reasoning_policy=answer_request.reasoning_policy,
                    allow_external_frame_transfer=(
                        answer_request.allow_external_frame_transfer
                    ),
                )
                _require_qa_policy_usable(context, request, require_retrieval=False)
            pipeline.answer_candidate(search_id, answer_request)
            return _safe_qa_snapshot(context, search_id)
        except HTTPException:
            raise
        except (KeyError, QASessionStoreError):
            raise HTTPException(
                status_code=404,
                detail={"code": "qa_search_missing"},
            ) from None
        except ValueError as error:
            code = (
                "florence_task_unsupported"
                if str(error) == "florence_task_unsupported"
                else "qa_single_frame_request_invalid"
            )
            raise HTTPException(
                status_code=422,
                detail={"code": code},
            ) from None
        except QAReasonerError as error:
            code = (
                "qa_all_providers_failed"
                if error.category == "provider_chain_exhausted"
                else error.category or "qa_provider_failed"
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": code,
                    "recoverable": error.category in {
                        "provider_chain_exhausted",
                        "network_unreachable",
                        "gemini_timeout",
                        "rate_limited",
                        "server_unavailable",
                        "resource_exhausted",
                    },
                },
            ) from None
        except QACandidateMediaUnavailableError as error:
            candidate_id = (
                error.candidate_id
                if _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(error.candidate_id)
                else None
            )
            detail: dict[str, Any] = {
                "code": "qa_candidate_media_unavailable",
                "recoverable": True,
            }
            if candidate_id is not None:
                detail["candidate_id"] = candidate_id
            raise HTTPException(status_code=409, detail=detail) from None
        except RuntimeError as error:
            code = str(error)
            if code == "legacy_qa_session_read_only":
                raise HTTPException(status_code=409, detail={"code": code}) from None
            if code == "florence_not_provisioned":
                raise HTTPException(status_code=503, detail={"code": code}) from None
            raise HTTPException(
                status_code=503,
                detail={"code": "qa_single_frame_unavailable", "recoverable": True},
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503,
                detail={"code": "qa_single_frame_unavailable", "recoverable": True},
            ) from None

    @app.post("/v1/qa/search/{search_id}/expand", deprecated=True)
    def qa_expand(search_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        del search_id, payload
        raise HTTPException(
            status_code=410,
            detail={
                "code": "qa_expand_removed",
                "message": "select a persisted candidate and call the single-frame answer endpoint",
            },
        )

    @app.get("/v1/qa/search/{search_id}")
    def qa_restore(search_id: str) -> dict[str, Any]:
        return _safe_qa_snapshot(context, search_id)

    @app.post("/v1/qa/search/{search_id}/edits")
    def qa_edit(search_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        window_id = payload.get("window_id")
        answer = payload.get("answer")
        if not isinstance(window_id, str) or not window_id.strip():
            raise HTTPException(status_code=422, detail="window_id is required")
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 256:
            raise HTTPException(status_code=422, detail="answer must be a bounded string")
        pipeline = _qa_pipeline(context)
        snapshot = pipeline.get_search(search_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Visual Q&A search is missing or expired")
        request_data = snapshot.get("request")
        try:
            request = QAQueryRequest.from_dict(request_data)
            spec = QACompetitionSpec.for_spec_id(request.answer_spec_id)
            normalized = spec.normalize_answer(answer, request.question)
            outcomes = snapshot.get("window_outcomes", ())
            if isinstance(outcomes, Mapping):
                outcomes = tuple(outcomes.values())
            selected = next(
                (
                    value
                    for value in outcomes
                    if isinstance(value, Mapping) and value.get("window_id") == window_id
                ),
                None,
            )
            if selected is None:
                raise ValueError("window does not exist")
            evidence = {
                "previous_normalized_answers": [str(selected.get("answer") or "")],
                "provider": selected.get("provider"),
                "model_name": selected.get("model_name"),
                "edited_raw_answer": answer,
                "normalized_answer": normalized,
            }
            store = getattr(context, "qa_session_store", None)
            if store is None:
                raise RuntimeError("QA audit store is unavailable")
            store.record_operator_edit(search_id, window_id, evidence)
            return _safe_qa_snapshot(context, search_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="QA operator edit is invalid") from None
        except QASessionStoreError:
            raise HTTPException(
                status_code=409, detail="QA session was concurrently modified"
            ) from None

    @app.post("/v1/qa/diagnostics/image")
    def qa_image_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
        encoded = payload.get("image_base64")
        mime_type = payload.get("mime_type")
        question = payload.get("question")
        event = payload.get("event_description") or "single uploaded diagnostic image"
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=422, detail="unsupported diagnostic image type")
        if not isinstance(encoded, str) or not isinstance(question, str) or not question.strip():
            raise HTTPException(status_code=422, detail="image and question are required")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="diagnostic image is invalid") from None
        if not raw or len(raw) > 8 * 1024 * 1024:
            raise HTTPException(status_code=422, detail="diagnostic image exceeds 8 MiB")
        try:
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                image = source.convert("RGB")
                image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format="WEBP", quality=88, method=6)
                canonical = output.getvalue()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            raise HTTPException(status_code=422, detail="diagnostic image is invalid") from None
        try:
            request = QAQueryRequest(
                query_id=f"diagnostic-{uuid4().hex}",
                event_description=str(event),
                question=question,
                answer_spec_id=str(
                    payload.get(
                        "answer_spec_id",
                        QACompetitionSpec.conservative_general_text().spec_id,
                    )
                ),
                batch_filters=(context.selected_batch_id,),
                limit=1,
                reasoning_policy=str(payload.get("reasoning_policy", "gemini_then_local")),
                allow_external_frame_transfer=payload.get("allow_external_frame_transfer", False),
            )
            if (
                request.reasoning_policy == "gemini_then_local"
                and not request.allow_external_frame_transfer
            ):
                raise ValueError("Gemini diagnostic requires frame-transfer consent")
            _require_qa_policy_usable(context, request, require_retrieval=False)
            result = context.qa_reasoner.reason(
                request,
                (
                    QAImage(
                        data=canonical,
                        mime_type="image/webp",
                        index=1,
                        source_frame_idx=0,
                        timestamp_ms=0,
                        sha256=hashlib.sha256(canonical).hexdigest(),
                    ),
                ),
                QACompetitionSpec.for_spec_id(request.answer_spec_id),
            )
            public = result.to_dict()
            public["requested_policy"] = request.reasoning_policy
            public["actual_provider"] = result.provider
            public["fallback_used"] = result.fallback_used
            public.pop("raw_answer", None)
            return public
        except QAReasonerError as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": error.category or "qa_provider_failed",
                    "provider": error.provider,
                    "model": error.model,
                },
            ) from None
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="diagnostic request is invalid") from None

    @app.get("/v1/frames/{batch_id}/{video_id}/{source_frame_idx}/neighbors")
    def neighbors(
        batch_id: str,
        video_id: str,
        source_frame_idx: int,
        radius_ms: int = 4_000,
        limit: int = 25,
    ) -> dict[str, Any]:
        repository = context.artifact_repository
        if repository is None:
            raise HTTPException(status_code=503, detail="artifact repository is unavailable")
        try:
            getter = getattr(repository, "get_source_frame", None)
            if not callable(getter):
                getter = repository.get_frame
            frame = getter(batch_id, video_id, source_frame_idx)
            if frame is None:
                raise HTTPException(status_code=404, detail="frame not found")
            values = repository.neighbors(
                batch_id,
                video_id,
                frame.timestamp_ms,
                radius_ms=radius_ms,
                limit=limit,
            )
            return {"frame": frame.to_dict(), "neighbors": [item.to_dict() for item in values]}
        except HTTPException:
            raise
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422, detail="frame neighborhood request is invalid"
            ) from None

    @app.get("/v1/frames/{batch_id}/{video_id}/{source_frame_idx}/image")
    def frame_image(batch_id: str, video_id: str, source_frame_idx: int) -> Response:
        repository = context.artifact_repository
        resolver = getattr(context, "qa_image_resolver", None)
        if repository is None or resolver is None:
            raise HTTPException(status_code=503, detail="canonical image service is unavailable")
        try:
            getter = getattr(repository, "get_source_frame", None)
            if not callable(getter):
                getter = repository.get_frame
            frame = getter(batch_id, video_id, source_frame_idx)
            if frame is None:
                raise HTTPException(status_code=404, detail="frame not found")
            image = resolver.resolve(frame)
            return Response(
                content=image.data,
                media_type=image.mime_type,
                headers={
                    "Cache-Control": "private, max-age=3600",
                    "ETag": f'"{image.content_sha256}"',
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=422, detail="canonical frame image is unavailable or invalid"
            ) from None

    @app.get("/v1/frames/{batch_id}/{video_id}/neighbors")
    def timestamp_neighbors(
        batch_id: str,
        video_id: str,
        timestamp_ms: int,
        radius_ms: int = 4_000,
        limit: int = 25,
    ) -> dict[str, Any]:
        repository = context.artifact_repository
        if repository is None:
            raise HTTPException(status_code=503, detail="frame repository is unavailable")
        try:
            values = repository.neighbors(
                batch_id,
                video_id,
                timestamp_ms,
                radius_ms=radius_ms,
                limit=limit,
            )
            return {"neighbors": [value.to_dict() for value in values]}
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422, detail="frame neighborhood request is invalid"
            ) from None

    @app.post("/v1/frames/validate")
    def validate_frames(payload: dict[str, Any]) -> dict[str, Any]:
        repository = context.artifact_repository
        batch_id = payload.get("batch_id")
        raw_identities = payload.get("identities")
        if repository is None:
            raise HTTPException(status_code=503, detail="frame repository is unavailable")
        if not isinstance(batch_id, str) or not isinstance(raw_identities, list):
            raise HTTPException(status_code=422, detail="frame validation request is invalid")
        try:
            selected_batch = _runtime_batch_filters(context, (batch_id,))[0]
            if len(raw_identities) > 100:
                raise ValueError("too many frame identities")
            identities: list[tuple[str, int]] = []
            for value in raw_identities:
                if not isinstance(value, Mapping):
                    raise ValueError("invalid frame identity")
                video_id = value.get("video_id")
                frame = value.get("source_frame_idx")
                if (
                    not isinstance(video_id, str)
                    or isinstance(frame, bool)
                    or not isinstance(frame, int)
                    or frame < 0
                ):
                    raise ValueError("invalid frame identity")
                identities.append((video_id, frame))
            existing = repository.existing_source_frames(
                selected_batch,
                identities,
                include_refined=True,
            )
            mappings = repository.frame_index_mappings_for_sources(
                selected_batch,
                identities,
            )
            timestamps: dict[str, int] = {}
            getter = getattr(repository, "get_source_frame", repository.get_frame)
            for video_id, source_frame_idx in existing:
                record = getter(selected_batch, video_id, source_frame_idx)
                if record is not None:
                    timestamps[f"{video_id}:{source_frame_idx}"] = int(record.timestamp_ms)
            return {
                "batch_id": selected_batch,
                "batch_active": bool(context.ledger.is_batch_active(selected_batch)),
                "existing": [
                    {"video_id": video_id, "source_frame_idx": source_frame_idx}
                    for video_id, source_frame_idx in sorted(existing)
                ],
                "ordinal_mappings": mappings,
                "timestamps_ms": timestamps,
            }
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422, detail="frame validation request is invalid"
            ) from None

    @app.post("/v1/submissions/export")
    def export(payload: dict[str, Any]) -> FileResponse:
        filename = str(payload.get("filename", "textual-kis.csv"))
        if not _SAFE_FILENAME.fullmatch(filename) or not filename.casefold().endswith(".csv"):
            raise HTTPException(status_code=422, detail="filename must be a simple .csv name")
        output_value = context.config["competition"]["output_dir"]
        output_dir = Path(output_value)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        spec = CompetitionSpec.from_config(context.config.get("competition", {}))
        batch_id = payload.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise HTTPException(
                status_code=422, detail="batch_id is required for active-batch validation"
            )

        try:
            batch_id = _runtime_batch_filters(context, (batch_id.strip(),))[0]
            if context.artifact_repository is None:
                raise SubmissionValidationError(
                    "canonical frame mappings are unavailable; export cannot validate frame IDs"
                )
            hits = payload.get("hits", [])
            identities = _submission_identities(
                hits,
                normalize_video_id=spec.normalize_video_id,
            )
            existing, mappings = _bounded_frame_validation(
                context.artifact_repository,
                batch_id,
                identities,
            )

            def is_canonical_source_frame(
                candidate_batch: str,
                video_id: str,
                source_frame_idx: int,
            ) -> bool:
                return candidate_batch == batch_id and (video_id, source_frame_idx) in existing

            target = export_submission(
                hits,
                output_dir / filename,
                spec=spec,
                batch_id=batch_id,
                is_batch_active=context.ledger.is_batch_active,
                is_canonical_source_frame=is_canonical_source_frame,
                frame_index_evidence=FrameIndexEvidence(mappings),
            )
        except SubmissionValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except (TypeError, ValueError, RuntimeError):
            raise HTTPException(
                status_code=422, detail="KIS submission export is invalid"
            ) from None
        return FileResponse(target, media_type="text/csv", filename=target.name)

    @app.post("/v1/qa/submissions/export")
    def qa_export(payload: dict[str, Any]) -> FileResponse:
        filename = str(payload.get("filename", "visual-qa.csv"))
        if not _SAFE_FILENAME.fullmatch(filename) or not filename.casefold().endswith(".csv"):
            raise HTTPException(status_code=422, detail="filename must be a simple .csv name")
        batch_id = payload.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise HTTPException(
                status_code=422, detail="batch_id is required for active-batch validation"
            )
        try:
            batch_id = _runtime_batch_filters(context, (batch_id.strip(),))[0]
            spec_id = str(
                payload.get(
                    "answer_spec_id",
                    context.config.get("retrieval", {})
                    .get("visual_qa", {})
                    .get("default_answer_spec_id", "aic-2024-visual-qa-numeric-v1"),
                )
            )
            spec = QACompetitionSpec.for_spec_id(spec_id)
            if context.artifact_repository is None:
                raise QASubmissionValidationError(
                    "canonical frame mappings are unavailable; export cannot validate frame IDs"
                )
            predictions = payload.get("predictions", payload.get("hits", []))
            identities = _submission_identities(
                predictions,
                normalize_video_id=spec.normalize_video_id,
            )
            existing, mappings = _bounded_frame_validation(
                context.artifact_repository,
                batch_id,
                identities,
            )

            def is_canonical_source_frame(
                candidate_batch: str,
                video_id: str,
                source_frame_idx: int,
            ) -> bool:
                return candidate_batch == batch_id and (video_id, source_frame_idx) in existing

            output_dir = _project_path(str(context.config["competition"]["output_dir"]))
            target = export_qa_submission(
                predictions,
                output_dir / filename,
                spec=spec,
                batch_id=batch_id,
                is_batch_active=context.ledger.is_batch_active,
                is_canonical_source_frame=is_canonical_source_frame,
                frame_index_evidence=FrameIndexEvidence(mappings),
            )
        except QASubmissionValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except (TypeError, ValueError, RuntimeError):
            raise HTTPException(status_code=422, detail="QA submission export is invalid") from None
        return FileResponse(target, media_type="text/csv", filename=target.name)

    @app.post("/v1/trake/submissions/export")
    def trake_export(payload: dict[str, Any]) -> FileResponse:
        query_id = payload.get("query_id")
        batch_id = payload.get("batch_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise HTTPException(status_code=422, detail="query_id is required")
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise HTTPException(
                status_code=422,
                detail="batch_id is required for active-batch validation",
            )
        try:
            batch_id = _runtime_batch_filters(context, (batch_id.strip(),))[0]
            repository = context.artifact_repository
            if repository is None:
                raise TRAKESubmissionValidationError(
                    "canonical frame mappings are unavailable; export cannot validate frame IDs"
                )
            results = payload.get("results", payload.get("hits", []))
            spec = TRAKECompetitionSpec.historical_2025_provisional()
            identities = _submission_identities(
                results,
                normalize_video_id=spec.normalize_video_id,
            )
            existing, mappings = _bounded_frame_validation(
                repository,
                batch_id,
                identities,
            )

            def is_canonical_source_frame(
                candidate_batch: str,
                video_id: str,
                source_frame_idx: int,
            ) -> bool:
                return candidate_batch == batch_id and (video_id, source_frame_idx) in existing

            def get_canonical_timestamp_ms(
                candidate_batch: str,
                video_id: str,
                source_frame_idx: int,
            ) -> int | None:
                if candidate_batch != batch_id:
                    return None
                getter = getattr(repository, "get_source_frame", None)
                if not callable(getter):
                    getter = repository.get_frame
                frame = getter(
                    candidate_batch,
                    video_id,
                    source_frame_idx,
                )
                return None if frame is None else frame.timestamp_ms

            output_dir = _project_path(str(context.config["competition"]["output_dir"]))
            target = export_trake_submission(
                results,
                output_dir,
                query_id=query_id,
                stage_count=payload.get("stage_count"),
                spec=spec,
                batch_id=batch_id,
                is_batch_active=context.ledger.is_batch_active,
                is_canonical_source_frame=is_canonical_source_frame,
                get_canonical_timestamp_ms=get_canonical_timestamp_ms,
                frame_index_evidence=TRAKEFrameIndexEvidence(mappings),
            )
        except TRAKESubmissionValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except (IndexError, TypeError, ValueError, RuntimeError):
            raise HTTPException(
                status_code=422, detail="TRAKE submission export is invalid"
            ) from None
        return FileResponse(target, media_type="text/csv", filename=target.name)

    def _bundle_output_dir() -> Path:
        return _project_path(str(context.config["competition"]["output_dir"]))

    @app.post("/v1/submissions/bundles")
    def create_submission_bundle(payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("mode", "all")
        batch_id = payload.get("batch_id")
        raw_entries = payload.get("entries")
        if mode not in {"all", "current"}:
            raise HTTPException(
                status_code=422,
                detail={"code": "bundle_mode_invalid", "issues": []},
            )
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise HTTPException(
                status_code=422,
                detail={"code": "bundle_batch_invalid", "issues": []},
            )
        if not isinstance(raw_entries, list) or not 1 <= len(raw_entries) <= 256:
            raise HTTPException(
                status_code=422,
                detail={"code": "bundle_entries_invalid", "issues": []},
            )
        try:
            batch_id = _runtime_batch_filters(context, (batch_id.strip(),))[0]
        except (IndexError, TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail={"code": "bundle_batch_invalid", "issues": []},
            ) from None

        entries: list[BundleEntry] = []
        total_bytes = 0
        try:
            for raw in raw_entries:
                if not isinstance(raw, Mapping):
                    raise ValueError("entry must be a mapping")
                encoded = raw.get("csv_base64")
                if not isinstance(encoded, str) or len(encoded) > 2_000_000:
                    raise ValueError("csv payload is invalid")
                try:
                    csv_bytes = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    raise ValueError("csv payload is invalid") from None
                if not csv_bytes or len(csv_bytes) > 1_000_000:
                    raise ValueError("csv payload is invalid")
                total_bytes += len(csv_bytes)
                if total_bytes > 8_000_000:
                    raise ValueError("bundle payload is too large")
                entries.append(
                    BundleEntry(
                        task=str(raw.get("task", "")),  # type: ignore[arg-type]
                        query_id=str(raw.get("query_id", "")),
                        csv_bytes=csv_bytes,
                        batch_id=batch_id,
                        source_fingerprint=str(raw.get("source_fingerprint", "")),
                        stage_count=raw.get("stage_count"),  # type: ignore[arg-type]
                        source_ranks=(
                            tuple(raw.get("source_ranks", ()))
                            if raw.get("source_ranks") is not None
                            else None
                        ),
                    )
                )
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail={"code": "bundle_entries_invalid", "issues": []},
            ) from None

        repository = context.artifact_repository
        if repository is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "bundle_build_unavailable"},
            )
        frame_cache: dict[tuple[str, int], Any] = {}

        def source_frame(video_id: str, frame_idx: int) -> Any:
            key = (video_id, frame_idx)
            if key not in frame_cache:
                getter = getattr(repository, "get_source_frame", None)
                if not callable(getter):
                    getter = repository.get_frame
                frame_cache[key] = getter(batch_id, video_id, frame_idx)
            return frame_cache[key]

        validation_context = BundleValidationContext(
            is_batch_active=lambda candidate: (
                candidate == batch_id and bool(context.ledger.is_batch_active(candidate))
            ),
            is_canonical_source_frame=lambda candidate, video, frame: (
                candidate == batch_id and source_frame(video, frame) is not None
            ),
            get_canonical_timestamp_ms=lambda candidate, video, frame: (
                None
                if candidate != batch_id or source_frame(video, frame) is None
                else int(source_frame(video, frame).timestamp_ms)
            ),
        )
        try:
            artifact = build_codabench_bundle(
                entries,
                _bundle_output_dir(),
                validation_context=validation_context,
                mode=mode,
            )
        except BundleValidationError as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "bundle_validation_failed",
                    "issues": [issue.to_dict() for issue in error.issues],
                },
            ) from None
        except (BundleBuildError, OSError):
            raise HTTPException(
                status_code=503,
                detail={"code": "bundle_build_unavailable"},
            ) from None
        result = artifact.to_dict()
        result["download_url"] = f"/v1/submissions/bundles/{result['run_id']}/download"
        return result

    @app.get("/v1/submissions/bundles/{run_id}")
    def submission_bundle_metadata(run_id: str) -> dict[str, Any]:
        if not re_fullmatch_run_id(run_id):
            raise HTTPException(status_code=404, detail={"code": "bundle_not_found"})
        manifest = load_bundle_manifest(_bundle_output_dir(), run_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail={"code": "bundle_not_found"})
        result = dict(manifest)
        result["download_url"] = f"/v1/submissions/bundles/{run_id}/download"
        return result

    @app.get("/v1/submissions/bundles/{run_id}/download")
    def download_submission_bundle(run_id: str) -> FileResponse:
        if not re_fullmatch_run_id(run_id):
            raise HTTPException(status_code=404, detail={"code": "bundle_not_found"})
        manifest = load_bundle_manifest(_bundle_output_dir(), run_id)
        filename = None if manifest is None else manifest.get("filename")
        if (
            not isinstance(filename, str)
            or _SAFE_FILENAME.fullmatch(filename) is None
            or not filename.casefold().endswith(".zip")
        ):
            raise HTTPException(status_code=404, detail={"code": "bundle_not_found"})
        target = _bundle_output_dir() / run_id / filename
        if not target.is_file():
            raise HTTPException(status_code=404, detail={"code": "bundle_not_found"})
        return FileResponse(target, media_type="application/zip", filename=filename)

    @app.get("/v1/ingest/status")
    def ingest_status(
        batch_id: str,
        preload_window_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            summary = context.ledger.status_summary(batch_id)
            status = summary.as_dict()
            status["resources"] = _resource_snapshot(context)
            ingestion = context.config.get("ingestion", {})
            configured_window = ingestion.get("preload_window_seconds")
            window = (
                preload_window_seconds if preload_window_seconds is not None else configured_window
            )
            projection = context.ledger.core_ingestion_projection(
                batch_id,
                total_videos=summary.total_videos,
                sample_videos=int(ingestion.get("projection_sample_videos", 10)),
                core_stages=ingestion.get("core_projection_stages", ("probe", "frames", "visual")),
                preload_window_seconds=None if window is None else float(window),
                deadline_fraction=float(ingestion.get("preload_deadline_fraction", 0.8)),
            )
            status["core_ingestion_projection"] = projection.as_dict()
            return status
        except ValueError:
            raise HTTPException(
                status_code=422, detail="ingest status request is invalid"
            ) from None
        except Exception:
            raise HTTPException(
                status_code=404, detail="batch ingest status is unavailable"
            ) from None

    @app.get("/v1/readiness")
    def readiness(batch_id: str | None = None, mode: str = "app") -> dict[str, Any]:
        """Return a sanitized, network-free readiness report for one batch."""

        service = getattr(context, "readiness_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="runtime readiness is unavailable")
        try:
            requested = (
                (batch_id.strip(),) if isinstance(batch_id, str) and batch_id.strip() else ()
            )
            selected_batch = _runtime_batch_filters(context, requested)[0]
            report = None
            warmup = getattr(context, "warmup", None)
            report_getter = getattr(warmup, "readiness_report", None)
            if mode == "app" and callable(report_getter):
                warmed = report_getter()
                if getattr(warmed, "batch_id", None) == selected_batch:
                    report = warmed
            if report is None:
                report = service.check(selected_batch, mode=mode)
            payload = report.to_dict()
            payload["selected_batch_id"] = getattr(context, "selected_batch_id", None)
            payload["launch_identity"] = sanitize_launch_identity(
                getattr(context, "launch_identity", {})
            )
            payload["warmup"] = _warmup_snapshot(context)
            channels = {item.name: item for item in report.channels}
            canonical_ready = bool(
                channels.get("canonical_images")
                and channels["canonical_images"].usable
            )
            local_ready = bool(
                channels.get("ollama_vlm") and channels["ollama_vlm"].usable
            )
            gemini_ready = bool(
                channels.get("gemini_vlm") and channels["gemini_vlm"].usable
            )
            answer_usable = canonical_ready and (local_ready or gemini_ready)
            answer_state = (
                "ready"
                if answer_usable and local_ready and gemini_ready
                else "degraded"
                if answer_usable
                else "blocked"
            )
            payload["qa_stages"] = {
                "retrieve": {
                    "state": payload.get("kis", {}).get("state", "blocked"),
                    "usable": payload.get("kis", {}).get("state") in {"ready", "degraded"},
                    "requires_vlm": False,
                },
                "answer": {
                    "state": answer_state,
                    "usable": answer_usable,
                    "requires_dense_encoder": False,
                    "providers": {
                        "ollama": local_ready,
                        "gemini": gemini_ready,
                    },
                },
            }
            return payload
        except ValueError:
            raise HTTPException(status_code=422, detail="readiness request is invalid") from None
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="runtime readiness could not be evaluated",
            ) from None

    @app.get("/api/v1/health/provisioning", deprecated=True)
    @app.get("/v1/health/provisioning")
    def provisioning(batch_id: str | None = None) -> dict[str, Any]:
        """Return batch channel readiness plus persisted ingestion progress."""

        service = getattr(context, "readiness_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="runtime readiness is unavailable")
        try:
            requested = (
                (batch_id.strip(),) if isinstance(batch_id, str) and batch_id.strip() else ()
            )
            selected_batch = _runtime_batch_filters(context, requested)[0]
            report = service.check(selected_batch, mode="app").to_dict()
            ingest: dict[str, Any] | None = None
            if context.ledger.is_batch_registered(selected_batch):
                ingest = ingest_status(selected_batch)
            return {
                "profile": report.get("profile"),
                "batch_id": selected_batch,
                "kis": report.get("kis"),
                "qa": report.get("qa"),
                "channels": report.get("channels", []),
                "checks": report.get("checks", []),
                "ingest": ingest,
                "warmup": _warmup_snapshot(context),
                "checked_at_unix": report.get("checked_at_unix"),
            }
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(status_code=422, detail="provisioning request is invalid") from None
        except Exception:
            raise HTTPException(
                status_code=503, detail="provisioning state could not be evaluated"
            ) from None

    @app.get("/api/v1/health/status", deprecated=True)
    @app.get("/v1/health/status")
    def system_health_status(batch_id: str | None = None) -> dict[str, Any]:
        """Return sanitized workflow and aggregate channel health for one batch."""

        service = getattr(context, "readiness_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="runtime readiness is unavailable")
        try:
            requested = (
                (batch_id.strip(),) if isinstance(batch_id, str) and batch_id.strip() else ()
            )
            selected_batch = _runtime_batch_filters(context, requested)[0]
            report = service.check(selected_batch, mode="app")
            payload = report.to_dict()
            workflow_probe = getattr(service, "workflow_readiness", None)
            if callable(workflow_probe):
                trake = workflow_probe(report, "trake").to_dict()
            else:
                trake = payload.get(
                    "trake",
                    {
                        "state": "blocked",
                        "required_channels": [],
                        "unavailable_channels": ["visual_vector"],
                    },
                )
            ingest: dict[str, Any] | None = None
            if context.ledger.is_batch_registered(selected_batch):
                ingest = ingest_status(selected_batch)
            return {
                "profile": payload.get("profile"),
                "batch_id": selected_batch,
                "kis": payload.get("kis"),
                "qa": payload.get("qa"),
                "trake": trake,
                "channel_statuses": aggregate_channel_health(report.channels),
                "ingest": ingest,
                "warmup": _warmup_snapshot(context),
                "checked_at_unix": payload.get("checked_at_unix"),
            }
        except HTTPException:
            raise
        except (IndexError, ValueError):
            raise HTTPException(
                status_code=422, detail="health status request is invalid"
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503, detail="runtime health could not be evaluated"
            ) from None

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/runtime/capabilities")
    def runtime_capabilities() -> dict[str, Any]:
        service = getattr(context, "readiness_service", None)
        channels: dict[str, dict[str, Any]] = {}
        warmup = getattr(context, "warmup", None)
        report = None
        report_getter = getattr(warmup, "readiness_report", None)
        if callable(report_getter):
            try:
                report = report_getter()
            except Exception:
                report = None
        elif service is not None:
            try:
                report = service.check(context.selected_batch_id, mode="app")
            except Exception:
                report = None
        if report is not None:
            for item in report.channels:
                if item.name in {"ollama_vlm", "gemini_vlm"}:
                    channels[item.name] = item.to_dict()
        ollama = context.config.get("services", {}).get("ollama_vlm", {})
        gemini = context.config.get("services", {}).get("gemini_vlm", {})
        if "ollama_vlm" not in channels:
            channels["ollama_vlm"] = {
                "name": "ollama_vlm",
                "state": "warming",
                "reason_code": "runtime_warming",
            }
        if "gemini_vlm" not in channels:
            configured = bool(str(gemini.get("api_key") or "").strip())
            channels["gemini_vlm"] = {
                "name": "gemini_vlm",
                "state": "configured_unverified" if configured else "blocked_asset",
                "reason_code": (
                    "external_check_not_run" if configured else "gemini_key_not_configured"
                ),
            }
        translation = context.config.get("models", {}).get("query_translation", {})
        translation_enabled = bool(translation.get("enabled", False))
        translation_files_ready = False
        if translation_enabled:
            converted = Path(str(translation.get("converted_path", "")))
            source = Path(str(translation.get("source_path", "")))
            converted = converted if converted.is_absolute() else PROJECT_ROOT / converted
            source = source if source.is_absolute() else PROJECT_ROOT / source
            translation_files_ready = (
                (converted / "model.bin").is_file()
                and (converted / "config.json").is_file()
                and (source / "source.spm").is_file()
                and (source / "target.spm").is_file()
            )
        translation_dependency_ready = importlib.util.find_spec("ctranslate2") is not None
        translation_ready = (
            translation_enabled and translation_files_ready and translation_dependency_ready
        )
        return {
            "profile": str(context.config.get("profile", "laptop")),
            "default_batch_id": context.selected_batch_id,
            "workflows": ["kis", "qa", "trake"],
            "query_translation": {
                "state": "ready" if translation_ready else (
                    "not_provisioned" if translation_enabled else "disabled_by_profile"
                ),
                "reason_code": None if translation_ready else (
                    "translator_asset_missing"
                    if translation_enabled and not translation_files_ready
                    else "translator_dependency_missing"
                    if translation_enabled
                    else "translator_disabled"
                ),
                "model": str(translation.get("model_name", "")),
                "revision": str(translation.get("revision", "")),
                "languages": ["vi", "en"],
                "offline_only": True,
            },
            "qa": {
                "default_policy": "gemini_then_local",
                "policies": ["gemini_then_local", "local_only"],
                "stages": {
                    "retrieve": {"requires_vlm": False},
                    "answer": {"requires_dense_encoder": False, "max_images": 1},
                },
                "local": {
                    "provider": "ollama-qwen",
                    "model": str(ollama.get("model", "qwen2.5vl:3b-q4_K_M")),
                    "status": channels.get("ollama_vlm", {}),
                },
                "cloud": {
                    "provider": "gemini",
                    "model": str(gemini.get("model", "gemini-3.6-flash")),
                    "status": channels.get("gemini_vlm", {}),
                    "requires_external_frame_transfer_consent": True,
                },
                "experimental": {
                    "florence2": {
                        "enabled": bool(
                            context.config.get("models", {})
                            .get("florence2", {})
                            .get("enabled", False)
                        ),
                        "tasks": ["ocr", "object_count"],
                        "offline_only": True,
                    }
                },
            },
        }

    @app.get("/v1/runtime/warmup")
    def runtime_warmup() -> dict[str, Any]:
        snapshot = _warmup_snapshot(context)
        if snapshot is None:
            raise HTTPException(status_code=503, detail="runtime warm-up is unavailable")
        return snapshot

    @app.post("/v1/runtime/warmup/start")
    def start_runtime_warmup() -> dict[str, Any]:
        """Release the local VLM and restart an idle or evicted warm-up."""

        warmup = getattr(context, "warmup", None)
        snapshot = _warmup_snapshot(context)
        if warmup is None or snapshot is None:
            raise HTTPException(status_code=503, detail="runtime warm-up is unavailable")
        if str(snapshot.get("state", "idle")) in {"idle", "evicted"}:
            _release_local_vlm(context)
            start = getattr(warmup, "start", None)
            if callable(start):
                start()
            snapshot = _warmup_snapshot(context) or snapshot
        return snapshot

    @app.get("/readyz")
    def ready(workflow: str = "kis", stage: str | None = None) -> dict[str, Any]:
        if workflow not in {"kis", "qa", "trake"}:
            raise HTTPException(status_code=422, detail="workflow must be kis, qa, or trake")
        if stage is not None and (workflow != "qa" or stage not in {"retrieve", "answer"}):
            raise HTTPException(
                status_code=422,
                detail="stage is supported only for qa: retrieve or answer",
            )
        if workflow == "qa" and stage == "answer":
            service = getattr(context, "readiness_service", None)
            if service is None:
                raise HTTPException(status_code=503, detail={"code": "runtime_not_ready"})
            try:
                report = None
                warmup = getattr(context, "warmup", None)
                report_getter = getattr(warmup, "readiness_report", None)
                if callable(report_getter):
                    warmed = report_getter()
                    if getattr(warmed, "batch_id", None) == context.selected_batch_id:
                        report = warmed
                if report is None:
                    report = service.check(context.selected_batch_id, mode="app")
                channels = {item.name: item for item in report.channels}
                canonical = bool(
                    channels.get("canonical_images") and channels["canonical_images"].usable
                )
                local = bool(channels.get("ollama_vlm") and channels["ollama_vlm"].usable)
                gemini = bool(channels.get("gemini_vlm") and channels["gemini_vlm"].usable)
                usable = canonical and (local or gemini)
            except Exception:
                usable = False
            if not usable:
                raise HTTPException(
                    status_code=503,
                    detail={"code": "qa_all_providers_failed", "workflow": workflow, "stage": stage},
                )
            return {"status": "ready", "workflow": workflow, "stage": stage}
        effective_workflow = "kis" if workflow == "qa" and stage == "retrieve" else workflow
        snapshot = _warmup_snapshot(context)
        if snapshot is None:
            return {
                "status": "ready",
                "workflow": workflow,
                "stage": stage,
                "warmup": None,
            }
        selected = _warmup_workflow(snapshot, effective_workflow)
        if not bool(selected.get("usable", False)):
            state = str(selected.get("state", "idle"))
            raise HTTPException(
                status_code=503,
                detail={
                    "code": (
                        "runtime_warming"
                        if state in {"idle", "warming", "evicted"}
                        else "trake_not_ready"
                        if effective_workflow == "trake"
                        else "runtime_not_ready"
                    ),
                    "workflow": workflow,
                    "stage": stage,
                    "warmup": snapshot,
                },
            )
        status = "ready" if selected.get("state") == "ready" else "degraded"
        return {
            "status": status,
            "workflow": workflow,
            "stage": stage,
            "warmup": snapshot,
        }

    return app


def create_default_app() -> FastAPI:
    """Uvicorn factory that explicitly initializes local state at startup."""

    context = build_runtime()
    context.ledger.initialize()
    if context.artifact_repository is not None:
        context.artifact_repository.initialize()
    if context.qa_session_store is not None:
        context.qa_session_store.initialize()
    return create_app(context)
