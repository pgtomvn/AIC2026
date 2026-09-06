"""Read-only, batch-aware runtime readiness reporting.

The probes in this module never initialize databases, create collections, load
models, or contact an external model provider.  Reports contain stable reason
codes instead of raw exception text so they are safe to expose through the API
and competition UI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import requests

from src import PROJECT_ROOT
from src.assets import verify_asset_manifest
from src.embedding.clip_encoder import CLIPEncoder
from src.indexing.vector_store import batch_collection_name
from src.organizer_verification import organizer_verification_passed
from src.retrieval.qa_media import CanonicalImageResolver
from src.tooling.ffmpeg import verify_ffmpeg_manifest
from src.utils.io_helpers import (
    configured_batch_id,
    load_launch_identity,
    sanitize_launch_identity,
)


class WorkflowState(StrEnum):
    """Overall availability for one user workflow."""

    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class ChannelApplicability(StrEnum):
    """Whether a channel participates in the selected batch."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_APPLICABLE = "not_applicable"


class ChannelState(StrEnum):
    """Sanitized channel state exposed to operators."""

    READY = "ready"
    CONFIGURED_UNVERIFIED = "configured_unverified"
    NOT_PROVISIONED = "not_provisioned"
    DISABLED_BY_PROFILE = "disabled_by_profile"
    BLOCKED_DEPENDENCY = "blocked_dependency"
    BLOCKED_ASSET = "blocked_asset"
    INCOMPATIBLE_INDEX = "incompatible_index"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass(frozen=True, slots=True)
class ChannelReadiness:
    """Readiness of one retrieval, media, or reasoning channel."""

    name: str
    applicability: ChannelApplicability
    state: ChannelState
    reason_code: str | None = None
    remediation_id: str | None = None
    count: int | None = None
    coverage: float | None = None
    resource_name: str | None = None
    expected_dimension: int | None = None

    @property
    def usable(self) -> bool:
        return self.state in {ChannelState.READY, ChannelState.CONFIGURED_UNVERIFIED}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkflowReadiness:
    """Aggregate workflow status and the channels needed to reach full readiness."""

    state: WorkflowState
    required_channels: tuple[str, ...]
    unavailable_channels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeCheck:
    """One local runtime/tooling prerequisite checked outside retrieval channels."""

    name: str
    state: ChannelState
    reason_code: str | None = None
    remediation_id: str | None = None

    @property
    def usable(self) -> bool:
        return self.state in _USABLE_STATES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Safe, immutable readiness snapshot for one runtime profile and batch."""

    profile: str
    batch_id: str
    selected_batch_id: str
    launch_identity: Mapping[str, Any]
    mode: str
    kis: WorkflowReadiness
    qa: WorkflowReadiness
    channels: tuple[ChannelReadiness, ...]
    checks: tuple[RuntimeCheck, ...]
    checked_at_unix: float
    multimodal: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "batch_id": self.batch_id,
            "selected_batch_id": self.selected_batch_id,
            "launch_identity": sanitize_launch_identity(self.launch_identity),
            "mode": self.mode,
            "kis": self.kis.to_dict(),
            "qa": self.qa.to_dict(),
            "channels": [channel.to_dict() for channel in self.channels],
            "checks": [check.to_dict() for check in self.checks],
            "checked_at_unix": self.checked_at_unix,
            "multimodal": dict(self.multimodal),
        }


class _Response(Protocol):
    status_code: int

    def json(self) -> Any: ...


class _HTTPClient(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> _Response: ...


_SAFE_BATCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_QDRANT_CHANNELS = {
    "visual": ("native_frames", "visual"),
    "organizer_clip": ("organizer_frames", "organizer_clip"),
    "accelerated_visual": ("accelerated_frames", "accelerated_visual"),
    "semantic_text": ("text_segments", "text"),
}
_MEILI_CHANNELS = {"ocr", "asr", "metadata"}
_USABLE_STATES = {ChannelState.READY, ChannelState.CONFIGURED_UNVERIFIED}
_HEALTH_CHANNEL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "visual_vector": ("visual", "organizer_clip", "accelerated_visual"),
    "ocr_text": ("ocr",),
    "asr_transcript": ("asr",),
    "metadata": ("metadata",),
}
_HEALTH_FAILURE_PRIORITY = {
    ChannelState.SERVICE_UNAVAILABLE: 0,
    ChannelState.INCOMPATIBLE_INDEX: 1,
    ChannelState.BLOCKED_DEPENDENCY: 2,
    ChannelState.BLOCKED_ASSET: 3,
    ChannelState.NOT_PROVISIONED: 4,
    ChannelState.DISABLED_BY_PROFILE: 5,
}


def aggregate_channel_health(
    channels: Iterable[ChannelReadiness],
) -> dict[str, dict[str, Any]]:
    """Collapse detailed readiness into four safe operator-facing channel groups.

    Required channels take precedence so an optional secondary encoder cannot mask
    an unavailable primary vector space.  No physical collection names, endpoints,
    paths, or backend exception details are copied into the aggregate.
    """

    by_name = {channel.name: channel for channel in channels}
    result: dict[str, dict[str, Any]] = {}
    for aggregate_name, member_names in _HEALTH_CHANNEL_GROUPS.items():
        members = [by_name[name] for name in member_names if name in by_name]
        applicable = [
            channel
            for channel in members
            if channel.applicability is not ChannelApplicability.NOT_APPLICABLE
        ]
        required = [
            channel
            for channel in applicable
            if channel.applicability is ChannelApplicability.REQUIRED
        ]
        candidates = required or applicable
        usable = [channel for channel in candidates if channel.usable]
        if usable:
            selected = min(
                usable,
                key=lambda channel: (
                    0 if channel.state is ChannelState.READY else 1,
                    member_names.index(channel.name),
                ),
            )
        elif candidates:
            selected = min(
                candidates,
                key=lambda channel: (
                    _HEALTH_FAILURE_PRIORITY.get(channel.state, 99),
                    member_names.index(channel.name),
                ),
            )
        else:
            selected = ChannelReadiness(
                aggregate_name,
                ChannelApplicability.NOT_APPLICABLE,
                ChannelState.NOT_PROVISIONED,
                "channel_not_applicable",
            )
        result[aggregate_name] = {
            "state": selected.state.value,
            "reason_code": selected.reason_code,
            "count": selected.count,
        }
    return result


def _path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _profile_enabled(model: Mapping[str, Any], profile: str) -> bool:
    if not bool(model.get("enabled", True)):
        return False
    enabled_profiles = tuple(str(item) for item in model.get("enabled_profiles", ()))
    return not enabled_profiles or profile in enabled_profiles


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def multimodal_coverage_snapshot(
    config: Mapping[str, Any],
    batch_id: str,
    channels: Iterable[ChannelReadiness],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Project exact artifact/ledger progress without creating generated state."""

    readiness = config.get("readiness", {})
    batches = readiness.get("batches", {}) if isinstance(readiness, Mapping) else {}
    specification = batches.get(batch_id, {}) if isinstance(batches, Mapping) else {}
    expected_videos = int(specification.get("expected_videos", 0) or 0)
    expected_frames = int(specification.get("expected_canonical_frames", 0) or 0)
    expected_mapping_rows = int(specification.get("expected_mapping_rows", 0) or 0)
    by_name = {channel.name: channel for channel in channels}
    visual_count = int(by_name.get("visual").count or 0) if by_name.get("visual") else 0
    organizer_count = (
        int(by_name.get("organizer_clip").count or 0) if by_name.get("organizer_clip") else 0
    )
    ocr_documents = int(by_name.get("ocr").count or 0) if by_name.get("ocr") else 0
    asr_documents = int(by_name.get("asr").count or 0) if by_name.get("asr") else 0
    data = config.get("data", {})
    ledger_path = _path(str(data.get("catalog_db", "")), project_root)
    artifact_path = _path(str(data.get("artifact_db", "")), project_root)
    frame_counts: dict[str, int] = {}
    stage_rows: dict[str, list[tuple[str, str, str | None]]] = {
        "ocr": [],
        "siglip2": [],
        "asr": [],
    }
    zero_transcript = 0
    asr_source_available = 0
    try:
        if artifact_path.is_file():
            with _read_only_connection(artifact_path) as connection:
                frame_counts = {
                    str(row[0]): int(row[1])
                    for row in connection.execute(
                        """
                        SELECT video_id, COUNT(*) FROM canonical_frames
                        WHERE batch_id = ? GROUP BY video_id
                        """,
                        (batch_id,),
                    )
                }
        if ledger_path.is_file():
            with _read_only_connection(ledger_path) as connection:
                for logical, stage_name in (
                    ("ocr", "reference_ocr"),
                    ("siglip2", "reference_siglip2"),
                    ("asr", "reference_asr"),
                ):
                    stage_rows[logical] = [
                        (str(row[0]), str(row[1]), None if row[2] is None else str(row[2]))
                        for row in connection.execute(
                            """
                            SELECT video_id, state, error FROM stages
                            WHERE batch_id = ? AND stage_name = ?
                            ORDER BY video_id
                            """,
                            (batch_id, stage_name),
                        )
                    ]
                zero_transcript = int(
                    connection.execute(
                        """
                        SELECT COUNT(DISTINCT video_id) FROM artifacts
                        WHERE batch_id = ? AND stage_name = 'reference_asr'
                          AND kind IN ('asr_no_audio', 'asr_zero_transcript')
                        """,
                        (batch_id,),
                    ).fetchone()[0]
                )
                asr_source_available = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM videos
                        WHERE batch_id = ? AND TRIM(source_uri) <> ''
                        """,
                        (batch_id,),
                    ).fetchone()[0]
                )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return {
            "status": "unknown",
            "reason_code": "multimodal_progress_unavailable",
            "expected_videos": expected_videos,
            "expected_canonical_frames": expected_frames,
        }

    def completed(stage: str) -> tuple[str, ...]:
        return tuple(video_id for video_id, state, _ in stage_rows[stage] if state == "completed")

    ocr_completed = completed("ocr")
    siglip_completed = completed("siglip2")
    asr_completed = completed("asr")
    ocr_scanned = sum(frame_counts.get(video_id, 0) for video_id in ocr_completed)
    # Source availability is independent from ASR processing completion.
    asr_failed = sum(state == "failed" for _, state, _ in stage_rows["asr"])
    asr_missing = max(0, expected_videos - asr_source_available)
    organizer_complete = expected_mapping_rows > 0 and organizer_count == expected_mapping_rows
    siglip_complete = (
        expected_frames > 0
        and visual_count == expected_frames
        and len(siglip_completed) == expected_videos
    )
    ocr_complete = expected_frames > 0 and ocr_scanned == expected_frames
    ocr_usable = ocr_documents > 0 and bool(ocr_completed)
    asr_complete = expected_videos > 0 and len(asr_completed) == expected_videos
    full_multimodal_counts = {
        "organizer_clip": organizer_count,
        "siglip2": visual_count,
        "ocr": ocr_documents,
        "asr": asr_documents,
    }
    channel_complete = {
        "organizer_clip": organizer_complete,
        "siglip2": siglip_complete,
        "ocr": ocr_complete and ocr_documents > 0,
        "asr": asr_complete and asr_documents > 0 and asr_failed == 0,
    }
    missing_full_multimodal = tuple(
        name for name in ("organizer_clip", "siglip2", "ocr", "asr") if not channel_complete[name]
    )
    advanced_kis = config.get("retrieval", {}).get("advanced_kis", {})
    object_batch_id = str(advanced_kis.get("object_batch_id", ""))
    object_archive_value = str(advanced_kis.get("object_archive", ""))
    object_archive_path = _path(object_archive_value, project_root)
    object_applicable = bool(object_batch_id == batch_id)
    object_available = bool(object_applicable and object_archive_path.is_file())
    object_state = (
        "ready"
        if object_available
        else "archive_missing"
        if object_applicable
        else "not_applicable"
    )
    coverage_healthy = (
        organizer_complete and siglip_complete and ocr_complete and asr_complete and asr_failed == 0
    )
    completion_value = readiness.get(
        "cloud_import_completion_receipt",
        "data/checkpoints/cloud-import/b01-import-complete.json",
    )
    completion_path = _path(str(completion_value), project_root)
    completion_valid = False
    completion_reason = "completion_receipt_missing"
    completion_manifest_sha256: str | None = None
    try:
        receipt = json.loads(completion_path.read_text(encoding="utf-8"))
        external_counts = receipt.get("external_counts")
        manifest_value = receipt.get("manifest_path")
        manifest_path = _path(str(manifest_value), project_root)
        digest = hashlib.sha256()
        with manifest_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        completion_manifest_sha256 = digest.hexdigest()
        completion_valid = bool(
            receipt.get("contract") == "cloud-import-completion-v2"
            and receipt.get("batch_id") == batch_id
            and receipt.get("complete") is True
            and tuple(receipt.get("completed_channels", ())) == ("siglip2", "ocr", "asr")
            and isinstance(external_counts, Mapping)
            and int(external_counts.get("siglip2", -1)) == visual_count
            and int(external_counts.get("ocr", -1)) == ocr_documents
            and int(external_counts.get("asr", -1)) == asr_documents
            and completion_manifest_sha256 == receipt.get("manifest_sha256")
        )
        completion_reason = None if completion_valid else "completion_receipt_mismatch"
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        completion_valid = False
    fully_healthy = coverage_healthy and completion_valid
    full_multimodal_healthy = fully_healthy and not missing_full_multimodal
    any_new_progress = bool(visual_count or ocr_completed or siglip_completed or asr_completed)
    return {
        "status": (
            "healthy" if fully_healthy else "partial" if any_new_progress else "not_provisioned"
        ),
        "reason_code": (
            None
            if fully_healthy
            else completion_reason
            if coverage_healthy and not completion_valid
            else "multimodal_coverage_incomplete"
        ),
        "capability_mode": ("full_multimodal" if full_multimodal_healthy else "multimodal_partial"),
        "status_label": (
            "HEALTHY · Full Multi-Modal"
            if full_multimodal_healthy
            else "DEGRADED · Multi-Modal Incomplete"
        ),
        "full_multimodal": {
            "healthy": full_multimodal_healthy,
            "required_channels": ("organizer_clip", "siglip2", "ocr", "asr"),
            "unavailable_channels": missing_full_multimodal,
            "usable_channels": tuple(
                name
                for name, usable in (
                    ("organizer_clip", organizer_count > 0),
                    ("siglip2", visual_count > 0),
                    ("ocr", ocr_usable),
                    ("asr", asr_documents > 0),
                )
                if usable
            ),
            "counts": full_multimodal_counts,
            "completion_receipt": {
                "valid": completion_valid,
                "reason_code": completion_reason,
                "manifest_sha256": completion_manifest_sha256,
            },
        },
        "objects": {
            "state": object_state,
            "available": object_available,
            "configured_batch_id": object_batch_id or None,
            "required_for": ("absolute_spatial", "relative_spatial", "negation"),
        },
        "expected_videos": expected_videos,
        "expected_canonical_frames": expected_frames,
        "organizer_clip": {
            "rows": organizer_count,
            "expected_rows": expected_mapping_rows,
            "consistent": organizer_complete,
        },
        "siglip2": {
            "canonical_frames": visual_count,
            "expected_frames": expected_frames,
            "completed_videos": len(siglip_completed),
            "consistent": siglip_complete,
        },
        "ocr": {
            "scanned_canonical_frames": ocr_scanned,
            "expected_frames": expected_frames,
            "completed_videos": len(ocr_completed),
            "text_documents": ocr_documents,
            "consistent": ocr_complete,
            "usable": ocr_usable,
            "coverage_mode": "complete" if ocr_complete else "partial" if ocr_usable else "missing",
            "coverage_ratio": (
                min(1.0, ocr_scanned / expected_frames) if expected_frames > 0 else 0.0
            ),
        },
        "asr": {
            "source_videos": asr_source_available,
            "completed_videos": len(asr_completed),
            "zero_transcript_videos": zero_transcript,
            # Compatibility alias for older clients. Zero transcript does not
            # imply that the source video lacks an audio stream.
            "no_audio_videos": zero_transcript,
            "failed_videos": asr_failed,
            "missing_source_videos": asr_missing,
            "text_documents": asr_documents,
            "consistent": asr_complete and asr_failed == 0,
        },
    }


class ReadinessService:
    """Probe local runtime prerequisites without mutating external or local state."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        project_root: Path = PROJECT_ROOT,
        http_client: _HTTPClient | None = None,
        clock: Callable[[], float] = time.time,
        selected_batch_id: str | None = None,
        launch_identity: Mapping[str, Any] | None = None,
        image_resolver: CanonicalImageResolver | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.http = http_client or requests.Session()
        self.clock = clock
        self.selected_batch_id = selected_batch_id or configured_batch_id(config)
        self.launch_identity = sanitize_launch_identity(
            launch_identity or load_launch_identity(config)
        )
        self.image_resolver = image_resolver or CanonicalImageResolver((project_root,))
        readiness = config.get("readiness", {})
        self.cache_seconds = max(0.0, float(readiness.get("cache_seconds", 10)))
        self._cache: dict[tuple[str, str, str], tuple[float, ReadinessReport]] = {}
        self._lock = threading.RLock()

    def invalidate(self) -> None:
        """Forget cached snapshots after ingestion or index activation."""

        with self._lock:
            self._cache.clear()

    def retrieval_fingerprint(self, batch_filters: tuple[str, ...]) -> str:
        """Return a stable, secret-free identity for candidate-cache invalidation.

        The identity includes retrieval/model configuration plus selected physical
        resource names, dimensions, counts, and health states.  Service endpoints,
        credentials, paths, and exception details are deliberately excluded.
        """

        batches = tuple(sorted(set(batch_filters)))
        if not batches:
            raise ValueError("at least one batch filter is required")
        reports = []
        retrieval_channels = {
            "visual",
            "organizer_clip",
            "accelerated_visual",
            "semantic_text",
            "ocr",
            "asr",
            "metadata",
        }
        for batch_id in batches:
            report = self.check(batch_id)
            reports.append(
                {
                    "batch_id": batch_id,
                    "channels": [
                        {
                            "name": channel.name,
                            "state": channel.state,
                            "count": channel.count,
                            "resource_name": channel.resource_name,
                            "expected_dimension": channel.expected_dimension,
                        }
                        for channel in report.channels
                        if channel.name in retrieval_channels
                    ],
                }
            )
        models = self.config.get("models", {})
        model_identity = {
            name: {
                key: model.get(key)
                for key in ("model_name", "pretrained", "revision", "embedding_dim")
            }
            for name in ("visual", "organizer_clip", "accelerated_visual", "text")
            if isinstance((model := models.get(name)), Mapping)
        }
        retrieval = self.config.get("retrieval", {})
        retrieval_identity = {
            key: retrieval.get(key)
            for key in (
                "rrf_k",
                "candidate_depths",
                "channel_weights",
                "tuning_record",
                "temporal_windows_ms",
                "diversity",
            )
        }
        payload = {
            "version": "qa-retrieval-readiness-v1",
            "profile": str(self.config.get("profile", "laptop")),
            "models": model_identity,
            "retrieval": retrieval_identity,
            "batches": reports,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def check(
        self,
        batch_id: str,
        *,
        mode: str = "app",
        force: bool = False,
    ) -> ReadinessReport:
        """Return a sanitized readiness report for ``batch_id``."""

        if not isinstance(batch_id, str) or not _SAFE_BATCH.fullmatch(batch_id):
            raise ValueError("batch_id must be a safe non-empty identifier")
        if mode not in {"app", "ingest", "full"}:
            raise ValueError("mode must be one of: app, ingest, full")
        profile = str(self.config.get("profile", "laptop"))
        if profile not in {"laptop", "accelerated"}:
            raise ValueError("profile must be laptop or accelerated")
        key = (profile, batch_id, mode)
        now = self.clock()
        if not force:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None and now - cached[0] <= self.cache_seconds:
                    return cached[1]

        specification = self._batch_specification(batch_id)
        applicability = {
            name: ChannelApplicability(str(value))
            for name, value in specification.get("channels", {}).items()
        }
        known_channels = (
            "visual",
            "organizer_clip",
            "accelerated_visual",
            "semantic_text",
            "ocr",
            "asr",
            "metadata",
            "canonical_images",
            "gemini_vlm",
            "ollama_vlm",
        )
        channels = tuple(
            self._probe_channel(
                name,
                batch_id,
                applicability.get(name, ChannelApplicability.NOT_APPLICABLE),
                specification,
            )
            for name in known_channels
        )
        by_name = {channel.name: channel for channel in channels}
        workflows = specification.get("workflows", {})
        kis = self._workflow(tuple(workflows.get("kis", ())), by_name, qa=False)
        qa = self._workflow(tuple(workflows.get("qa", ())), by_name, qa=True)
        checks = self._runtime_checks(mode)
        report = ReadinessReport(
            profile=profile,
            batch_id=batch_id,
            selected_batch_id=self.selected_batch_id,
            launch_identity=self.launch_identity,
            mode=mode,
            kis=kis,
            qa=qa,
            channels=channels,
            checks=checks,
            checked_at_unix=now,
            multimodal=multimodal_coverage_snapshot(
                self.config,
                batch_id,
                channels,
                project_root=self.project_root,
            ),
            # checked_at_unix is passed explicitly above.
        )
        with self._lock:
            self._cache[key] = (now, report)
        return report

    def workflow_readiness(
        self,
        report: ReadinessReport,
        workflow: str,
    ) -> WorkflowReadiness:
        """Aggregate another configured workflow from an existing channel snapshot.

        This avoids a second service probe and lets compatibility health endpoints
        expose newly configured workflows without changing ``ReadinessReport``.
        """

        if workflow not in {"kis", "qa", "trake"}:
            raise ValueError("workflow must be one of: kis, qa, trake")
        specification = self._batch_specification(report.batch_id)
        configured = specification.get("workflows", {})
        required = tuple(configured.get(workflow, ()))
        by_name = {channel.name: channel for channel in report.channels}
        if workflow == "trake":
            unavailable = tuple(
                name for name in required if name not in by_name or not by_name[name].usable
            )
            if unavailable:
                return WorkflowReadiness(WorkflowState.BLOCKED, required, unavailable)
            thumbnails = by_name.get("canonical_images")
            if thumbnails is None or not thumbnails.usable:
                return WorkflowReadiness(
                    WorkflowState.DEGRADED,
                    required,
                    ("canonical_images",),
                )
            return WorkflowReadiness(WorkflowState.READY, required, ())
        return self._workflow(required, by_name, qa=workflow == "qa")

    def _runtime_checks(self, mode: str) -> tuple[RuntimeCheck, ...]:
        translation = self._query_translation_check()
        if mode == "app":
            return (translation,)
        checks = [
            self._ffmpeg_check(),
            self._ingest_dependency_check(),
            self._cuda_check(),
            translation,
        ]
        if mode == "full":
            checks.append(self._asset_manifest_check())
        return tuple(checks)

    def _query_translation_check(self) -> RuntimeCheck:
        """Report the optional offline VI-to-EN adapter without loading it."""

        model = self.config.get("models", {}).get("query_translation", {})
        if not isinstance(model, Mapping) or not bool(model.get("enabled", False)):
            return RuntimeCheck(
                "query_translation",
                ChannelState.DISABLED_BY_PROFILE,
                "translator_disabled",
                "prepare_query_translation",
            )
        if not _module_available("ctranslate2"):
            return RuntimeCheck(
                "query_translation",
                ChannelState.BLOCKED_DEPENDENCY,
                "translator_dependency_missing",
                "bootstrap_runtime",
            )

        converted = _path(str(model.get("converted_path", "")), self.project_root)
        tokenizer = _path(str(model.get("tokenizer_path", "")), self.project_root)
        manifest = _path(str(model.get("asset_manifest", "")), self.project_root)
        required = (
            converted / "model.bin",
            converted / "config.json",
            tokenizer / "source.spm",
            tokenizer / "target.spm",
            manifest,
        )
        if any(not path.is_file() for path in required):
            return RuntimeCheck(
                "query_translation",
                ChannelState.BLOCKED_ASSET,
                "translator_asset_missing",
                "prepare_query_translation",
            )
        return RuntimeCheck("query_translation", ChannelState.READY)

    def _ffmpeg_check(self) -> RuntimeCheck:
        value = self.config.get("tools", {}).get("ffmpeg", {}).get("manifest")
        if not isinstance(value, str) or not value:
            return RuntimeCheck(
                "ffmpeg", ChannelState.BLOCKED_ASSET, "ffmpeg_not_configured", "install_ffmpeg"
            )
        try:
            verify_ffmpeg_manifest(_path(value, self.project_root))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return RuntimeCheck(
                "ffmpeg", ChannelState.BLOCKED_ASSET, "ffmpeg_verification_failed", "install_ffmpeg"
            )
        return RuntimeCheck("ffmpeg", ChannelState.READY)

    @staticmethod
    def _ingest_dependency_check() -> RuntimeCheck:
        if any(not _module_available(name) for name in ("av", "torch", "transformers")):
            return RuntimeCheck(
                "ingest_dependencies",
                ChannelState.BLOCKED_DEPENDENCY,
                "competition_dependency_missing",
                "bootstrap_runtime",
            )
        return RuntimeCheck("ingest_dependencies", ChannelState.READY)

    def _cuda_check(self) -> RuntimeCheck:
        visual = self.config.get("models", {}).get("visual", {})
        if str(visual.get("device", "cpu")) != "cuda":
            return RuntimeCheck("cuda", ChannelState.READY)
        try:
            import torch

            available = bool(torch.cuda.is_available())
        except (ImportError, RuntimeError):
            available = False
        if not available:
            return RuntimeCheck(
                "cuda",
                ChannelState.BLOCKED_ASSET,
                "cuda_unavailable",
                "repair_cuda_runtime",
            )
        return RuntimeCheck("cuda", ChannelState.READY)

    def _asset_manifest_check(self) -> RuntimeCheck:
        models = self.config.get("models", {})
        profile = str(self.config.get("profile", "laptop"))
        required: set[Path] = set()
        for key in ("visual", "text", "whisper", "ocr", "organizer_clip"):
            model = models.get(key, {})
            if key in {"organizer_clip"} and not _profile_enabled(model, profile):
                continue
            value = model.get("local_path", model.get("cache_dir"))
            if isinstance(value, str) and value:
                required.add(_path(value, self.project_root))
        for key in ("accelerated_visual",):
            model = models.get(key, {})
            if not _profile_enabled(model, profile):
                continue
            value = model.get("cache_dir")
            if isinstance(value, str) and value:
                required.add(_path(value, self.project_root))
        try:
            _checked, errors = verify_asset_manifest(
                self.project_root / "checksums" / "runtime-assets.sha256",
                root=self.project_root,
                required_paths=required,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            errors = ["verification failed"]
        if errors:
            return RuntimeCheck(
                "runtime_assets",
                ChannelState.BLOCKED_ASSET,
                "runtime_asset_manifest_failed",
                "prepare_models",
            )
        return RuntimeCheck("runtime_assets", ChannelState.READY)

    def _batch_specification(self, batch_id: str) -> Mapping[str, Any]:
        readiness = self.config.get("readiness", {})
        batches = readiness.get("batches", {})
        selected = batches.get(batch_id) if isinstance(batches, Mapping) else None
        if isinstance(selected, Mapping):
            return selected
        fallback = readiness.get("native_defaults", {})
        return fallback if isinstance(fallback, Mapping) else {}

    @staticmethod
    def _workflow(
        required: tuple[str, ...],
        channels: Mapping[str, ChannelReadiness],
        *,
        qa: bool,
    ) -> WorkflowReadiness:
        unavailable = tuple(
            name for name in required if name not in channels or not channels[name].usable
        )
        if qa:
            gemini_usable = bool(channels.get("gemini_vlm") and channels["gemini_vlm"].usable)
            local_usable = bool(channels.get("ollama_vlm") and channels["ollama_vlm"].usable)
            if "canonical_images" in unavailable or not (gemini_usable or local_usable):
                state = WorkflowState.BLOCKED
            elif unavailable:
                retrieval_names = {
                    "visual",
                    "organizer_clip",
                    "accelerated_visual",
                    "semantic_text",
                    "ocr",
                    "asr",
                    "metadata",
                }
                usable_retrieval = any(
                    name in retrieval_names and channel.state in _USABLE_STATES
                    for name, channel in channels.items()
                )
                state = WorkflowState.DEGRADED if usable_retrieval else WorkflowState.BLOCKED
            else:
                state = (
                    WorkflowState.READY
                    if gemini_usable and local_usable
                    else WorkflowState.DEGRADED
                )
        elif not unavailable:
            state = WorkflowState.READY
        else:
            retrieval_names = {
                "visual",
                "organizer_clip",
                "accelerated_visual",
                "semantic_text",
                "ocr",
                "asr",
                "metadata",
            }
            usable_retrieval = any(
                name in retrieval_names and channel.state in _USABLE_STATES
                for name, channel in channels.items()
            )
            state = WorkflowState.DEGRADED if usable_retrieval else WorkflowState.BLOCKED
        return WorkflowReadiness(state, required, unavailable)

    def _probe_channel(
        self,
        name: str,
        batch_id: str,
        applicability: ChannelApplicability,
        specification: Mapping[str, Any],
    ) -> ChannelReadiness:
        if applicability is ChannelApplicability.NOT_APPLICABLE:
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.DISABLED_BY_PROFILE,
                "channel_not_applicable",
            )
        if name in _QDRANT_CHANNELS:
            return self._probe_qdrant(name, batch_id, applicability)
        if name in _MEILI_CHANNELS:
            return self._probe_meili(name, batch_id, applicability)
        if name == "canonical_images":
            return self._probe_canonical_images(batch_id, applicability, specification)
        if name == "ollama_vlm":
            return self._probe_ollama_vlm(applicability)
        if name == "gemini_vlm":
            return self._probe_gemini_vlm(applicability)
        return ChannelReadiness(
            name,
            applicability,
            ChannelState.NOT_PROVISIONED,
            "unknown_channel",
        )

    def _request_json(self, method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
        response = self.http.request(method, url, timeout=3, **kwargs)
        return int(response.status_code), response.json()

    def _probe_qdrant(
        self,
        name: str,
        batch_id: str,
        applicability: ChannelApplicability,
    ) -> ChannelReadiness:
        collection_key, model_key = _QDRANT_CHANNELS[name]
        services = self.config.get("services", {})
        qdrant = services.get("qdrant", {})
        models = self.config.get("models", {})
        model = models.get(model_key, {})
        profile = str(self.config.get("profile", "laptop"))
        if name == "accelerated_visual" and not _profile_enabled(model, profile):
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.DISABLED_BY_PROFILE,
                "profile_does_not_enable_channel",
                "select_accelerated_profile",
            )
        template = qdrant.get("collections", {}).get(collection_key)
        expected_dimension = model.get("embedding_dim")
        if not isinstance(template, str) or not template:
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.NOT_PROVISIONED,
                "collection_not_configured",
                "init_indexes",
                expected_dimension=int(expected_dimension) if expected_dimension else None,
            )
        collection = (
            batch_collection_name(template, batch_id) if "{batch}" in template else template
        )
        base = str(qdrant.get("url", "")).rstrip("/")
        headers = {"api-key": qdrant["api_key"]} if qdrant.get("api_key") else {}
        try:
            status, payload = self._request_json(
                "GET", f"{base}/collections/{collection}", headers=headers
            )
            if status == 404:
                return ChannelReadiness(
                    name,
                    applicability,
                    ChannelState.NOT_PROVISIONED,
                    "collection_missing",
                    "init_indexes",
                    resource_name=collection,
                    expected_dimension=int(expected_dimension) if expected_dimension else None,
                )
            if status != 200:
                raise ValueError("unhealthy response")
            result = payload["result"]
            vector_config = result["config"]["params"]["vectors"]
            dimension = int(vector_config["size"])
            if expected_dimension is not None and dimension != int(expected_dimension):
                return ChannelReadiness(
                    name,
                    applicability,
                    ChannelState.INCOMPATIBLE_INDEX,
                    "vector_dimension_mismatch",
                    "rebuild_incompatible_index",
                    resource_name=collection,
                    expected_dimension=int(expected_dimension),
                )
            if result.get("status") != "green" or result.get("optimizer_status") != "ok":
                return ChannelReadiness(
                    name,
                    applicability,
                    ChannelState.INCOMPATIBLE_INDEX,
                    "collection_not_green",
                    "wait_for_index_optimization",
                    resource_name=collection,
                    expected_dimension=dimension,
                )
            count = int(result.get("points_count", 0))
            if "{batch}" not in template:
                count_status, count_payload = self._request_json(
                    "POST",
                    f"{base}/collections/{collection}/points/count",
                    headers=headers,
                    json={
                        "filter": {"must": [{"key": "batch_id", "match": {"value": batch_id}}]},
                        "exact": True,
                    },
                )
                if count_status != 200:
                    raise ValueError("count failed")
                count = int(count_payload["result"]["count"])
        except (KeyError, TypeError, ValueError, requests.RequestException, OSError):
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.SERVICE_UNAVAILABLE,
                "qdrant_unavailable",
                "start_services",
                resource_name=collection,
                expected_dimension=int(expected_dimension) if expected_dimension else None,
            )
        if count <= 0:
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.NOT_PROVISIONED,
                "collection_empty_for_batch",
                "ingest_or_import_batch",
                count=0,
                resource_name=collection,
                expected_dimension=int(expected_dimension) if expected_dimension else None,
            )
        dependency = self._model_dependency(name)
        if dependency is not None:
            return replace(
                dependency,
                applicability=applicability,
                count=count,
                resource_name=collection,
                expected_dimension=int(expected_dimension) if expected_dimension else None,
            )
        return ChannelReadiness(
            name,
            applicability,
            ChannelState.READY,
            count=count,
            resource_name=collection,
            expected_dimension=int(expected_dimension) if expected_dimension else None,
        )

    def _model_dependency(self, name: str) -> ChannelReadiness | None:
        module_sets = {
            "visual": ("torch", "transformers"),
            "semantic_text": ("torch", "transformers"),
            "organizer_clip": ("torch", "open_clip"),
            "accelerated_visual": ("torch", "open_clip"),
        }
        if any(not _module_available(module) for module in module_sets[name]):
            return ChannelReadiness(
                name,
                ChannelApplicability.OPTIONAL,
                ChannelState.BLOCKED_DEPENDENCY,
                "competition_dependency_missing",
                "bootstrap_runtime",
            )
        models = self.config.get("models", {})
        model = models.get(
            {
                "visual": "visual",
                "semantic_text": "text",
                "organizer_clip": "organizer_clip",
                "accelerated_visual": "accelerated_visual",
            }[name],
            {},
        )
        if name == "organizer_clip":
            encoder = CLIPEncoder(
                str(model.get("model_name", "")),
                str(model.get("pretrained", "")),
                device="cpu",
                batch_size=1,
                cache_dir=_path(model.get("cache_dir", "models/open_clip"), self.project_root),
                local_files_only=True,
            )
            if not organizer_verification_passed(
                model,
                expected_fingerprint=encoder.fingerprint,
                project_root=self.project_root,
            ):
                return ChannelReadiness(
                    name,
                    ChannelApplicability.OPTIONAL,
                    ChannelState.BLOCKED_ASSET,
                    "organizer_encoder_unverified",
                    "verify_organizer_encoder",
                )
        elif not self._model_snapshot_exists(model):
            return ChannelReadiness(
                name,
                ChannelApplicability.OPTIONAL,
                ChannelState.BLOCKED_ASSET,
                "pinned_model_snapshot_missing",
                "prepare_models",
            )
        return None

    def _model_snapshot_exists(self, model: Mapping[str, Any]) -> bool:
        cache_value = model.get("cache_dir")
        if not isinstance(cache_value, str) or not cache_value:
            return False
        cache = _path(cache_value, self.project_root)
        revision = model.get("revision")
        model_name = model.get("model_name")
        if isinstance(revision, str) and revision and isinstance(model_name, str):
            snapshot = cache / f"models--{model_name.replace('/', '--')}" / "snapshots" / revision
            return snapshot.is_dir() and any(snapshot.iterdir())
        return cache.is_dir() and any(cache.iterdir())

    def _probe_meili(
        self,
        name: str,
        batch_id: str,
        applicability: ChannelApplicability,
    ) -> ChannelReadiness:
        meili = self.config.get("services", {}).get("meilisearch", {})
        index = meili.get("indexes", {}).get(name)
        if not isinstance(index, str) or not index:
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.NOT_PROVISIONED,
                "index_not_configured",
                "init_indexes",
            )
        headers = {"Authorization": f"Bearer {meili['api_key']}"} if meili.get("api_key") else {}
        base = str(meili.get("url", "")).rstrip("/")
        try:
            status, payload = self._request_json(
                "POST",
                f"{base}/indexes/{index}/search",
                headers=headers,
                json={
                    "q": "",
                    "limit": 0,
                    "filter": f'batch_id = "{batch_id}"',
                    "facets": ["batch_id"],
                },
            )
            if status == 404:
                return ChannelReadiness(
                    name,
                    applicability,
                    ChannelState.NOT_PROVISIONED,
                    "index_missing",
                    "init_indexes",
                    resource_name=index,
                )
            if status != 200:
                raise ValueError("unhealthy response")
            distribution = payload.get("facetDistribution", {}).get("batch_id", {})
            count = int(
                distribution.get(
                    batch_id,
                    payload.get("estimatedTotalHits", payload.get("totalHits", 0)),
                )
            )
        except (AttributeError, TypeError, ValueError, requests.RequestException, OSError):
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.SERVICE_UNAVAILABLE,
                "meilisearch_unavailable",
                "start_services",
                resource_name=index,
            )
        if count <= 0:
            return ChannelReadiness(
                name,
                applicability,
                ChannelState.NOT_PROVISIONED,
                "index_empty_for_batch",
                "index_text",
                count=0,
                resource_name=index,
            )
        return ChannelReadiness(
            name,
            applicability,
            ChannelState.READY,
            count=count,
            resource_name=index,
        )

    def _probe_canonical_images(
        self,
        batch_id: str,
        applicability: ChannelApplicability,
        specification: Mapping[str, Any],
    ) -> ChannelReadiness:
        database_value = self.config.get("data", {}).get("artifact_db")
        if not isinstance(database_value, str) or not database_value:
            return ChannelReadiness(
                "canonical_images",
                applicability,
                ChannelState.NOT_PROVISIONED,
                "artifact_database_not_configured",
                "import_or_ingest_batch",
            )
        database = _path(database_value, self.project_root)
        if not database.is_file():
            return ChannelReadiness(
                "canonical_images",
                applicability,
                ChannelState.NOT_PROVISIONED,
                "artifact_database_missing",
                "import_or_ingest_batch",
            )
        try:
            uri = database.resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=1) as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT video_id)
                    FROM canonical_frames
                    WHERE batch_id = ? AND image_uri <> ''
                    """,
                    (batch_id,),
                ).fetchone()
            count, video_count = int(row[0]), int(row[1])
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return ChannelReadiness(
                "canonical_images",
                applicability,
                ChannelState.NOT_PROVISIONED,
                "canonical_artifacts_unavailable",
                "import_or_ingest_batch",
            )
        expected = int(
            specification.get(
                "expected_canonical_frames",
                specification.get("expected_frames", 0),
            )
            or 0
        )
        expected_videos = int(specification.get("expected_videos", 0) or 0)
        coverage = min(1.0, count / expected) if expected else None
        threshold_value = specification.get("canonical_image_min_coverage")
        if threshold_value is None:
            coverage_sufficient = (not expected or count >= expected) and (
                not expected_videos or video_count >= expected_videos
            )
        else:
            try:
                threshold = float(threshold_value)
            except (TypeError, ValueError):
                threshold = 1.0
            if not 0.0 <= threshold < 1.0:
                threshold = 1.0
            coverage_sufficient = coverage is not None and coverage > threshold
        if count <= 0:
            state = ChannelState.NOT_PROVISIONED
            reason = "no_canonical_images_for_batch"
        elif not coverage_sufficient:
            state = ChannelState.NOT_PROVISIONED
            reason = "canonical_image_coverage_incomplete"
        else:
            state = ChannelState.READY
            reason = None
        return ChannelReadiness(
            "canonical_images",
            applicability,
            state,
            reason,
            "import_or_ingest_batch" if reason else None,
            count=count,
            coverage=coverage,
            resource_name=database.name,
        )

    def _probe_ollama_vlm(self, applicability: ChannelApplicability) -> ChannelReadiness:
        service = self.config.get("services", {}).get("ollama_vlm", {})
        model = str(service.get("model", "qwen2.5vl:3b-q4_K_M"))
        base_url = str(service.get("url", "http://127.0.0.1:11434")).rstrip("/")
        try:
            status, tags = self._request_json("GET", f"{base_url}/api/tags")
            if status != 200 or not isinstance(tags, Mapping):
                raise ValueError("invalid local model catalog")
            models = tags.get("models")
            model_values = models if isinstance(models, list) else []
            record = next(
                (
                    value
                    for value in model_values
                    if isinstance(value, Mapping)
                    and str(value.get("name") or value.get("model") or "") == model
                ),
                None,
            )
            if record is None:
                return ChannelReadiness(
                    "ollama_vlm",
                    applicability,
                    ChannelState.BLOCKED_ASSET,
                    "ollama_model_missing",
                    "pull_local_vlm_model",
                    resource_name=model,
                )
            expected = str(service.get("expected_digest_prefix") or "").casefold()
            digest = str(record.get("digest") or "").casefold()
            if expected and not digest.startswith(expected):
                return ChannelReadiness(
                    "ollama_vlm",
                    applicability,
                    ChannelState.INCOMPATIBLE_INDEX,
                    "ollama_model_fingerprint_mismatch",
                    "pull_pinned_local_vlm_model",
                    resource_name=model,
                )
            show_status, details = self._request_json(
                "POST", f"{base_url}/api/show", json={"model": model}
            )
            capabilities = details.get("capabilities") if isinstance(details, Mapping) else None
            if (
                show_status != 200
                or not isinstance(capabilities, list)
                or "vision" not in capabilities
            ):
                return ChannelReadiness(
                    "ollama_vlm",
                    applicability,
                    ChannelState.BLOCKED_ASSET,
                    "ollama_model_not_vision_capable",
                    "configure_vision_model",
                    resource_name=model,
                )
        except (KeyError, TypeError, ValueError, requests.RequestException, OSError):
            return ChannelReadiness(
                "ollama_vlm",
                applicability,
                ChannelState.SERVICE_UNAVAILABLE,
                "ollama_service_unavailable",
                "start_ollama",
                resource_name=model,
            )
        return ChannelReadiness(
            "ollama_vlm",
            applicability,
            ChannelState.READY,
            resource_name=model,
        )

    def _probe_gemini_vlm(self, applicability: ChannelApplicability) -> ChannelReadiness:
        """Report configuration without contacting the external provider."""

        service = self.config.get("services", {}).get("gemini_vlm", {})
        model = str(service.get("model", "gemini-3.6-flash"))
        key = service.get("api_key")
        if not _module_available("google.genai"):
            return ChannelReadiness(
                "gemini_vlm",
                applicability,
                ChannelState.BLOCKED_DEPENDENCY,
                "gemini_sdk_missing",
                "bootstrap_runtime",
                resource_name=model,
            )
        if not isinstance(key, str) or not key.strip():
            return ChannelReadiness(
                "gemini_vlm",
                applicability,
                ChannelState.BLOCKED_ASSET,
                "gemini_key_not_configured",
                "configure_gemini_key",
                resource_name=model,
            )
        return ChannelReadiness(
            "gemini_vlm",
            applicability,
            ChannelState.CONFIGURED_UNVERIFIED,
            "external_check_not_run",
            "run_explicit_gemini_check",
            resource_name=model,
        )


def build_readiness_service(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    selected_batch_id: str | None = None,
    launch_identity: Mapping[str, Any] | None = None,
) -> ReadinessService:
    """Build the reusable, side-effect-free readiness service."""

    return ReadinessService(
        config,
        project_root=project_root,
        selected_batch_id=selected_batch_id,
        launch_identity=launch_identity,
    )
