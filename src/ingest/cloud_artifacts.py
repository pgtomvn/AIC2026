"""ZIP-streamed cloud extraction artifacts and bounded local importing.

The module is deliberately side-effect free at import and construction time.
Callers explicitly run extraction, finalization, validation, or import methods.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields
from fractions import Fraction
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np

from src.contracts import FrameRecord, TextSegment
from src.ingest.ledger import ArtifactSpec, JobLedger, StageFingerprintMismatch

# Receipt fingerprints intentionally keep the v1 contract so work produced by
# an older Colab runtime remains resumable. Progressive final artifacts use
# schema v3; the reader keeps accepting complete schema-v2 bundles.
CLOUD_ARTIFACT_SCHEMA_VERSION = 3
LEGACY_CLOUD_ARTIFACT_SCHEMA_VERSION = 2
CLOUD_ARTIFACT_CONTRACT = "aic-cloud-zip-artifacts-v1"
SIGLIP_FEATURES_NAME = "b01_siglip2_features.npy"
SIGLIP_MANIFEST_NAME = "b01_siglip2_manifest.json"
OCR_FEATURES_NAME = "b01_ocr_features.jsonl.gz"
ASR_TRANSCRIPTS_NAME = "b01_asr_transcripts.jsonl.gz"
ASR_AUDIO_EXTRACTION_CONTRACT = "zip-member-seekable-v2"
_STAGES = ("siglip2", "ocr", "asr")
_TEXT_STAGE_FILES = {"ocr": OCR_FEATURES_NAME, "asr": ASR_TRANSCRIPTS_NAME}
_SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TEXT_SEGMENT_FIELDS = frozenset(field.name for field in fields(TextSegment))


class CloudArtifactError(RuntimeError):
    """Raised when a cloud artifact or ZIP corpus violates its contract."""


def _normalized_channels(
    channels: Sequence[str], *, require_siglip2: bool = False
) -> tuple[str, ...]:
    requested = tuple(str(value).strip().casefold() for value in channels)
    selected = tuple(stage for stage in _STAGES if stage in requested)
    if (
        not selected
        or len(requested) != len(set(requested))
        or len(selected) != len(requested)
        or (require_siglip2 and "siglip2" not in selected)
    ):
        requirement = " including siglip2" if require_siglip2 else ""
        raise ValueError(f"channels must be a unique non-empty subset of {_STAGES}{requirement}")
    return selected


def _channel_file_name(stage: str) -> str:
    return SIGLIP_FEATURES_NAME if stage == "siglip2" else _TEXT_STAGE_FILES[stage]


@dataclass(frozen=True, slots=True)
class CloudFrameSource:
    """A canonical frame plus its portable ZIP member location."""

    frame: FrameRecord
    archive_path: Path
    member: str


@dataclass(frozen=True, slots=True)
class CloudImportReport:
    """Bounded import result suitable for JSON diagnostics."""

    batch_id: str
    channels: tuple[str, ...]
    vectors: int
    ocr_documents: int
    asr_documents: int
    elapsed_ms: float
    validate_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "channels": list(self.channels),
            "vectors": self.vectors,
            "ocr_documents": self.ocr_documents,
            "asr_documents": self.asr_documents,
            "elapsed_ms": self.elapsed_ms,
            "validate_only": self.validate_only,
        }


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _zip_directory_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint ZIP identity from its central directory without hashing payload bytes."""

    with zipfile.ZipFile(path) as archive:
        members = [
            {
                "name": info.filename.replace("\\", "/"),
                "size": int(info.file_size),
                "compressed_size": int(info.compress_size),
                "crc32": f"{int(info.CRC):08x}",
            }
            for info in archive.infolist()
        ]
    identity = {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "members": len(members),
        "central_directory_sha256": _stable_json_hash({"members": members}),
    }
    return identity


def _tool_versions() -> dict[str, str]:
    """Return bounded, secret-free versions relevant to artifact reproduction."""

    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for distribution in ("torch", "transformers", "rapidocr", "faster-whisper"):
        try:
            versions[distribution] = package_version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def asr_profile_settings(config: Mapping[str, Any], profile: str = "reference") -> dict[str, Any]:
    """Resolve one explicit ASR profile without silently changing devices/models."""

    normalized = str(profile).strip().casefold()
    if normalized == "production":
        values = config["models"]["whisper"]
    elif normalized == "reference":
        values = config["models"]["reference_provisioning"]["whisper"]
    else:
        raise ValueError("ASR profile must be production or reference")
    resolved = dict(values)
    resolved["profile"] = normalized
    resolved.setdefault("vad_filter", True)
    resolved.setdefault("vad_parameters", {})
    resolved.setdefault("condition_on_previous_text", True)
    resolved.setdefault("beam_size", 1)
    resolved.setdefault("retry_beam_size", resolved["beam_size"])
    resolved.setdefault("compute_type", "int8_float16" if normalized == "production" else "int8")
    return resolved


def cloud_model_contracts(
    config: Mapping[str, Any], *, asr_profile: str = "reference"
) -> dict[str, dict[str, Any]]:
    """Build the settings-derived model contract shared by cloud and importer CLIs."""

    models = config["models"]
    visual = models["visual"]
    reference = models["reference_provisioning"]
    rapidocr = reference["rapidocr"]
    whisper = asr_profile_settings(config, asr_profile)
    return {
        "siglip2": {
            "name": str(visual["model_name"]),
            "revision": str(visual["revision"]),
            "dimension": int(visual["embedding_dim"]),
            "dtype": "float16",
        },
        "ocr": {
            "name": "rapidocr-ppocrv6-small-multilingual",
            "revision": str(rapidocr["source_revision"]),
            "package_version": str(rapidocr["package_version"]),
            "confidence_threshold": max(0.5, float(rapidocr["confidence_threshold"])),
        },
        "asr": {
            "profile": str(whisper["profile"]),
            "name": str(whisper["hub_model_id"]),
            "revision": str(whisper["revision"]),
            "language": whisper.get("language"),
            "compute_type": str(whisper["compute_type"]),
            "beam_size": int(whisper["beam_size"]),
            "retry_beam_size": int(whisper["retry_beam_size"]),
            "vad_filter": bool(whisper["vad_filter"]),
            "vad_parameters": {
                str(key): int(value)
                for key, value in dict(whisper.get("vad_parameters", {})).items()
            },
            "condition_on_previous_text": bool(whisper.get("condition_on_previous_text", True)),
            "word_timestamps": True,
            "normalization": "nfc+accent-fold-v1",
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npy", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl_gzip(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    count = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
                count += 1
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _safe_members(archive: zipfile.ZipFile, suffixes: frozenset[str]) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise CloudArtifactError("unsafe_zip_member")
        if not info.is_dir() and path.suffix.casefold() in suffixes:
            members.append(info)
    return members


def _validated_video_id(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_VIDEO_ID.fullmatch(value):
        raise CloudArtifactError("invalid_video_id")
    return value


def _text_document(segment: TextSegment) -> dict[str, Any]:
    document = {"id": segment.segment_id, **segment.to_dict(), "text": segment.text_nfc}
    document[f"{segment.kind}_text"] = segment.text_nfc
    return document


def _segment_from_document(document: Mapping[str, Any]) -> TextSegment:
    values = {key: value for key, value in document.items() if key in _TEXT_SEGMENT_FIELDS}
    segment = TextSegment.from_dict(values)
    if str(document.get("id", segment.segment_id)) != segment.segment_id:
        raise CloudArtifactError("text_document_id_mismatch")
    return segment


def build_cloud_partition_manifest(
    *,
    map_archive: str | Path,
    keyframe_archives: Sequence[str | Path],
    video_archives: Sequence[str | Path] = (),
    batch_id: str,
    stage: str,
    max_frames: int = 8000,
    worker_count: int = 2,
) -> dict[str, Any]:
    """Build an immutable archive-local extraction plan without extracting ZIPs."""

    if stage not in {"ocr", "asr"}:
        raise ValueError("partition stage must be ocr or asr")
    if max_frames < 1 or worker_count < 1:
        raise ValueError("max_frames and worker_count must be positive")
    map_path = Path(map_archive)
    keyframes = tuple(Path(value) for value in keyframe_archives)
    videos = tuple(Path(value) for value in video_archives)
    producer = CloudArtifactProducer(
        map_archive=map_path,
        keyframe_archives=keyframes,
        video_archives=videos,
        output_dir=Path("."),
        batch_id=batch_id,
    )
    maps = producer._map_rows()
    images = producer._image_index(keyframes)
    video_index = producer._video_index(videos) if videos else {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for video_id in sorted(maps):
        rows = maps[video_id]
        seen_sources: set[int] = set()
        canonical_rows: list[dict[str, str]] = []
        archives: set[str] = set()
        for row in rows:
            source_frame_idx = int(row["frame_idx"])
            if source_frame_idx in seen_sources:
                continue
            location = images.get((video_id, int(row["n"])))
            if location is None:
                raise CloudArtifactError("keyframe_member_missing")
            seen_sources.add(source_frame_idx)
            canonical_rows.append(row)
            archives.add(location[0].name)
        if len(archives) != 1:
            raise CloudArtifactError("partition_video_crosses_archives")
        if stage == "asr":
            video_location = video_index.get(video_id)
            if video_location is None:
                raise CloudArtifactError("partition_video_member_missing")
            archive_name = video_location[0].name
            weight = max(
                1,
                round(max(float(row["pts_time"]) for row in rows) * 1000),
            )
        else:
            archive_name = next(iter(archives))
            weight = len(canonical_rows)
        grouped.setdefault(archive_name, []).append(
            {
                "video_id": video_id,
                "canonical_frames": len(canonical_rows),
                "estimated_duration_ms": round(max(float(row["pts_time"]) for row in rows) * 1000),
                "weight": weight,
            }
        )

    partitions: list[dict[str, Any]] = []
    for archive_name in sorted(grouped):
        current: list[dict[str, Any]] = []
        current_frames = 0
        sequence = 0
        for video in grouped[archive_name]:
            frames = int(video["canonical_frames"])
            if current and current_frames + frames > max_frames:
                partitions.append(_partition_descriptor(stage, archive_name, sequence, current))
                sequence += 1
                current = []
                current_frames = 0
            current.append(video)
            current_frames += frames
        if current:
            partitions.append(_partition_descriptor(stage, archive_name, sequence, current))

    loads = [0] * worker_count
    assignments: dict[str, int] = {}
    for partition in sorted(
        partitions,
        key=lambda value: (-int(value["weight"]), str(value["partition_id"])),
    ):
        slot = min(range(worker_count), key=lambda value: (loads[value], value))
        assignments[str(partition["partition_id"])] = slot
        loads[slot] += int(partition["weight"])
    for partition in partitions:
        partition["worker_slot"] = assignments[str(partition["partition_id"])]
        partition.pop("weight", None)
    manifest: dict[str, Any] = {
        "contract": "cloud-partition-plan-v1",
        "schema_version": 1,
        "batch_id": _validated_video_id(batch_id),
        "stage": stage,
        "max_frames": int(max_frames),
        "worker_count": int(worker_count),
        "worker_loads": loads,
        "expected_videos": sum(len(value["videos"]) for value in partitions),
        "expected_frames": sum(int(value["canonical_frames"]) for value in partitions),
        "map_archive": _zip_directory_fingerprint(map_path),
        "keyframe_archives": [_zip_directory_fingerprint(value) for value in sorted(keyframes)],
        "video_archives": [_zip_directory_fingerprint(value) for value in sorted(videos)],
        "partitions": sorted(partitions, key=lambda value: str(value["partition_id"])),
    }
    manifest["plan_fingerprint"] = _stable_json_hash(manifest)
    return manifest


def _partition_descriptor(
    stage: str,
    archive_name: str,
    sequence: int,
    videos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    safe_archive = re.sub(r"[^a-z0-9]+", "-", Path(archive_name).stem.casefold()).strip("-")
    canonical_frames = sum(int(value["canonical_frames"]) for value in videos)
    estimated_duration_ms = sum(int(value["estimated_duration_ms"]) for value in videos)
    return {
        "partition_id": f"{stage}-{safe_archive}-{sequence:03d}",
        "archive": archive_name,
        "videos": [str(value["video_id"]) for value in videos],
        "canonical_frames": canonical_frames,
        "estimated_duration_ms": estimated_duration_ms,
        "weight": estimated_duration_ms if stage == "asr" else canonical_frames,
    }


def load_cloud_partition(
    path: str | Path,
    *,
    batch_id: str,
    stage: str,
    partition_id: str,
    worker_slot: int,
    worker_count: int,
) -> dict[str, Any]:
    """Validate a worker assignment and return its immutable partition descriptor."""

    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CloudArtifactError("partition_manifest_invalid") from error
    fingerprint = manifest.pop("plan_fingerprint", None) if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("contract") != "cloud-partition-plan-v1"
        or manifest.get("schema_version") != 1
        or manifest.get("batch_id") != batch_id
        or manifest.get("stage") != stage
        or int(manifest.get("worker_count", -1)) != int(worker_count)
        or fingerprint != _stable_json_hash(manifest)
    ):
        raise CloudArtifactError("partition_manifest_contract_mismatch")
    matches = [
        value
        for value in manifest.get("partitions", ())
        if isinstance(value, Mapping) and value.get("partition_id") == partition_id
    ]
    if len(matches) != 1 or int(matches[0].get("worker_slot", -1)) != int(worker_slot):
        raise CloudArtifactError("partition_worker_assignment_mismatch")
    return dict(matches[0])


class CloudArtifactProducer:
    """Produce resumable B01 artifacts directly from one or more ZIP archives."""

    def __init__(
        self,
        *,
        map_archive: str | Path,
        keyframe_archives: Sequence[str | Path],
        video_archives: Sequence[str | Path] = (),
        output_dir: str | Path,
        batch_id: str = "aic25-b1-reference",
        resume: bool = True,
        expected_videos: int | None = None,
        expected_frames: int | None = None,
        scratch_dir: str | Path | None = None,
        progress_path: str | Path | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        model_contracts: Mapping[str, Mapping[str, Any]] | None = None,
        selected_video_ids: Sequence[str] | None = None,
        partition_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.map_archive = Path(map_archive)
        self.keyframe_archives = tuple(Path(path) for path in keyframe_archives)
        self.video_archives = tuple(Path(path) for path in video_archives)
        self.output_dir = Path(output_dir)
        self.batch_id = _validated_video_id(batch_id)
        self.resume = bool(resume)
        self.expected_videos = expected_videos
        self.expected_frames = expected_frames
        self._scratch_dir = None if scratch_dir is None else Path(scratch_dir)
        self._progress_path = None if progress_path is None else Path(progress_path)
        self.progress_callback = progress_callback
        self.selected_video_ids = (
            None
            if selected_video_ids is None
            else frozenset(_validated_video_id(value) for value in selected_video_ids)
        )
        self.partition_id = None if partition_id is None else _validated_video_id(partition_id)
        self.model_contracts = {}
        for stage, value in (model_contracts or {}).items():
            contract = dict(value)
            contract.setdefault("settings_fingerprint", _stable_json_hash(contract))
            self.model_contracts[str(stage)] = contract
        self.clock = clock
        if not self.keyframe_archives:
            raise ValueError("at least one keyframe ZIP is required")
        if expected_videos is not None and expected_videos < 1:
            raise ValueError("expected_videos must be positive")
        if expected_frames is not None and expected_frames < 1:
            raise ValueError("expected_frames must be positive")
        self._maps: dict[str, tuple[dict[str, str], ...]] | None = None
        self._map_digest: str | None = None

    @property
    def work_dir(self) -> Path:
        # Google DriveFS may retain files below dot-prefixed directories only in
        # the VM cache, so those shards can disappear when a Colab runtime is
        # reclaimed.  Keep the resumable workspace visible to Drive sync.
        return self.output_dir / "cloud_work"

    @property
    def scratch_dir(self) -> Path:
        """Return ephemeral ASR scratch outside Drive-backed output by default."""

        if self._scratch_dir is not None:
            return self._scratch_dir
        return Path(tempfile.gettempdir()) / "aic2026-cloud-asr" / self.batch_id

    @property
    def progress_path(self) -> Path:
        return self._progress_path or self.work_dir / "progress.json"

    def _report_progress(
        self,
        *,
        stage: str,
        completed: int,
        total: int,
        records: int,
        resumed: int,
        started: float,
        receipt: Path,
    ) -> None:
        elapsed = max(0.0, self.clock() - started)
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total - completed)
        event = {
            "contract": "cloud-progress-v1",
            "batch_id": self.batch_id,
            "stage": stage,
            "state": "completed" if completed == total else "running",
            "completed_videos": completed,
            "total_videos": total,
            "records": records,
            "resumed_videos": resumed,
            "elapsed_seconds": round(elapsed, 3),
            "videos_per_second": round(rate, 6),
            "eta_seconds": round(remaining / rate, 3) if rate > 0 else None,
            "last_receipt": receipt.relative_to(self.work_dir).as_posix(),
            "heartbeat_unix": time.time(),
        }
        path = self.progress_path
        _atomic_json(path, event)
        log_path = path.with_suffix(".jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.progress_callback is not None:
            self.progress_callback(event)

    def _map_rows(self) -> dict[str, tuple[dict[str, str], ...]]:
        if self._maps is not None:
            return self._maps
        if not self.map_archive.is_file():
            raise FileNotFoundError(self.map_archive)
        result: dict[str, tuple[dict[str, str], ...]] = {}
        with zipfile.ZipFile(self.map_archive) as archive:
            for info in _safe_members(archive, frozenset({".csv"})):
                video_id = _validated_video_id(PurePosixPath(info.filename).stem)
                if video_id in result:
                    raise CloudArtifactError("duplicate_map_video")
                with archive.open(info) as raw:
                    wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                    rows = tuple(dict(row) for row in csv.DictReader(wrapper))
                required = {"n", "pts_time", "fps", "frame_idx"}
                if not rows or not required.issubset(rows[0]):
                    raise CloudArtifactError("invalid_frame_map")
                try:
                    ordered = tuple(sorted(rows, key=lambda row: int(row["n"])))
                    ordinals = tuple(int(row["n"]) for row in ordered)
                except (TypeError, ValueError, KeyError) as error:
                    raise CloudArtifactError("invalid_frame_map") from error
                if ordinals != tuple(range(1, len(ordered) + 1)):
                    raise CloudArtifactError("non_contiguous_keyframe_ordinals")
                result[video_id] = ordered
        if not result:
            raise CloudArtifactError("frame_maps_empty")
        self._maps = result
        self._map_digest = sha256_file(self.map_archive)
        return result

    @staticmethod
    def _image_index(
        archives: Sequence[Path],
    ) -> dict[tuple[str, int], tuple[Path, str]]:
        index: dict[tuple[str, int], tuple[Path, str]] = {}
        for archive_path in archives:
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                for info in _safe_members(archive, frozenset({".jpg", ".jpeg", ".png", ".webp"})):
                    path = PurePosixPath(info.filename.replace("\\", "/"))
                    if len(path.parts) < 2 or not path.stem.isdigit():
                        continue
                    key = (_validated_video_id(path.parts[-2]), int(path.stem))
                    if key in index:
                        raise CloudArtifactError("duplicate_keyframe_member")
                    index[key] = (archive_path, info.filename)
        return index

    @staticmethod
    def _video_index(archives: Sequence[Path]) -> dict[str, tuple[Path, str]]:
        index: dict[str, tuple[Path, str]] = {}
        suffixes = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm"})
        for archive_path in archives:
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                for info in _safe_members(archive, suffixes):
                    video_id = _validated_video_id(PurePosixPath(info.filename).stem)
                    if video_id in index:
                        raise CloudArtifactError("duplicate_video_member")
                    index[video_id] = (archive_path, info.filename)
        return index

    def _frame_sources(self) -> dict[str, tuple[CloudFrameSource, ...]]:
        maps = self._map_rows()
        images = self._image_index(self.keyframe_archives)
        videos = self._video_index(self.video_archives) if self.video_archives else {}
        available_videos = sorted(set(maps) & {video_id for video_id, _ in images})
        if self.selected_video_ids is not None:
            missing = self.selected_video_ids - set(available_videos)
            if missing:
                raise CloudArtifactError("partition_video_missing")
            available_videos = [
                value for value in available_videos if value in self.selected_video_ids
            ]
        if not available_videos:
            raise CloudArtifactError("no_mapped_keyframes")
        digest = self._map_digest or sha256_file(self.map_archive)
        result: dict[str, tuple[CloudFrameSource, ...]] = {}
        for video_id in available_videos:
            canonical: list[CloudFrameSource] = []
            seen_sources: set[int] = set()
            for row in maps[video_id]:
                try:
                    ordinal = int(row["n"])
                    source_frame_idx = int(row["frame_idx"])
                    timestamp_ms = round(float(row["pts_time"]) * 1000)
                    fps = Fraction(str(row["fps"])).limit_denominator(100_000)
                except (TypeError, ValueError, KeyError, ZeroDivisionError) as error:
                    raise CloudArtifactError("invalid_frame_map") from error
                if source_frame_idx in seen_sources:
                    continue
                location = images.get((video_id, ordinal))
                if location is None:
                    raise CloudArtifactError("keyframe_member_missing")
                seen_sources.add(source_frame_idx)
                archive_path, member = location
                video_location = videos.get(video_id)
                video_uri = (
                    f"zip://{video_location[0].name}!/{video_location[1]}"
                    if video_location is not None
                    else f"reference://{self.batch_id}/{video_id}"
                )
                frame = FrameRecord(
                    batch_id=self.batch_id,
                    video_id=video_id,
                    keyframe_ordinal=ordinal,
                    source_frame_idx=source_frame_idx,
                    pts=timestamp_ms,
                    time_base="1/1000",
                    timestamp_ms=timestamp_ms,
                    fps_num=fps.numerator,
                    fps_den=fps.denominator,
                    shot_id=f"reference:{ordinal}",
                    sampling_reasons=("organizer_reference",),
                    image_uri=f"zip://{archive_path.name}!/{member}",
                    video_uri=video_uri,
                    source_sha256=digest,
                    extractor_version="cloud-zip-reference-v1",
                )
                canonical.append(CloudFrameSource(frame, archive_path, member))
            result[video_id] = tuple(canonical)
        return result

    def _flat_receipt_path(self, stage: str, video_id: str) -> Path:
        return self.work_dir / stage / f"{_validated_video_id(video_id)}.json"

    def _receipt_path(self, stage: str, video_id: str) -> Path:
        flat = self._flat_receipt_path(stage, video_id)
        if self.partition_id is None or flat.exists():
            return flat
        return self.work_dir / stage / "partitions" / self.partition_id / flat.name

    def _shard_path(self, stage: str, video_id: str) -> Path:
        suffix = ".npy" if stage == "siglip2" else ".jsonl.gz"
        name = f"{_validated_video_id(video_id)}{suffix}"
        flat_receipt = self._flat_receipt_path(stage, video_id)
        if self.partition_id is None or flat_receipt.exists():
            return self.work_dir / stage / name
        return self.work_dir / stage / "partitions" / self.partition_id / name

    def _valid_receipt(self, path: Path, fingerprint: str) -> bool:
        if not self.resume or not path.is_file():
            return False
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            shard = self.work_dir / str(receipt["artifact"])
            return (
                receipt.get("contract") == CLOUD_ARTIFACT_CONTRACT
                and receipt.get("batch_id") == self.batch_id
                and receipt.get("fingerprint") == fingerprint
                and (
                    receipt.get("stage") != "asr"
                    or receipt.get("extraction_contract") == ASR_AUDIO_EXTRACTION_CONTRACT
                )
                and shard.is_file()
                and int(receipt.get("size_bytes", -1)) == shard.stat().st_size
                and receipt.get("sha256") == sha256_file(shard)
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def _write_receipt(
        self,
        *,
        stage: str,
        video_id: str,
        fingerprint: str,
        shard: Path,
        count: int,
        model: Mapping[str, Any],
        frames: Sequence[FrameRecord] = (),
        scanned_frames: int | None = None,
        extraction_contract: str | None = None,
        audio_stream_present: bool | None = None,
        transcript_segment_count: int | None = None,
        zero_transcript_reason: str | None = None,
    ) -> None:
        receipt = {
            "schema_version": CLOUD_ARTIFACT_SCHEMA_VERSION,
            "contract": CLOUD_ARTIFACT_CONTRACT,
            "batch_id": self.batch_id,
            "stage": stage,
            "video_id": video_id,
            "fingerprint": fingerprint,
            "artifact": shard.relative_to(self.work_dir).as_posix(),
            "sha256": sha256_file(shard),
            "size_bytes": shard.stat().st_size,
            "count": int(count),
            "scanned_frames": scanned_frames,
            "model": dict(model),
            "frames": [frame.to_dict() for frame in frames],
        }
        if extraction_contract is not None:
            receipt["extraction_contract"] = str(extraction_contract)
        if stage == "asr":
            if not isinstance(audio_stream_present, bool):
                raise TypeError("ASR receipts require audio_stream_present")
            if (
                isinstance(transcript_segment_count, bool)
                or not isinstance(transcript_segment_count, int)
                or transcript_segment_count < 0
            ):
                raise ValueError(
                    "ASR receipts require a non-negative transcript segment count"
                )
            if zero_transcript_reason not in {
                None,
                "no_audio_stream",
                "no_speech_detected",
            }:
                raise ValueError("ASR zero transcript reason is unsupported")
            if transcript_segment_count > 0 and zero_transcript_reason is not None:
                raise ValueError(
                    "non-empty ASR transcripts cannot have a zero transcript reason"
                )
            if transcript_segment_count == 0 and zero_transcript_reason is None:
                raise ValueError("empty ASR transcripts require a zero transcript reason")
            receipt["audio_stream_present"] = audio_stream_present
            receipt["transcript_segment_count"] = transcript_segment_count
            receipt["zero_transcript_reason"] = zero_transcript_reason
        _atomic_json(self._receipt_path(stage, video_id), receipt)

    @staticmethod
    def _model_identity(model: Any, stage: str) -> dict[str, Any]:
        if stage == "siglip2":
            return {
                "name": str(getattr(model, "model_name", "siglip2")),
                "revision": getattr(model, "revision", None),
                "fingerprint": str(getattr(model, "fingerprint", "unknown")),
            }
        if stage == "ocr":
            return {
                "name": str(getattr(model, "model_name", "rapidocr")),
                "revision": getattr(model, "source_revision", None),
                "fingerprint": str(getattr(model, "fingerprint", "unknown")),
            }
        name = str(getattr(model, "model_size", "faster-whisper"))
        revision = getattr(model, "model_revision", None)
        profile = str(getattr(model, "profile_name", "reference"))
        fingerprint = getattr(model, "fingerprint", None)
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            fingerprint = _stable_json_hash(
                {
                    "contract": "asr-model-identity-v1",
                    "name": name,
                    "revision": revision,
                    "profile": profile,
                    "compute_type": getattr(model, "compute_type", None),
                    "vad_filter": getattr(model, "vad_filter", None),
                }
            )
        return {
            "name": name,
            "revision": revision,
            "profile": profile,
            "fingerprint": fingerprint,
        }

    def run_siglip2(self, encoder: Any, *, batch_size: int = 128) -> dict[str, int]:
        """Encode canonical keyframes into resumable per-video FP16 shards."""

        if batch_size not in {128, 256}:
            raise ValueError("SigLIP2 cloud batch size must be 128 or 256")
        sources = self._frame_sources()
        model = self._model_identity(encoder, "siglip2")
        completed = 0
        vectors_written = 0
        resumed = 0
        started = self.clock()
        total_videos = len(sources)
        with ExitStack() as stack:
            archives = {
                path: stack.enter_context(zipfile.ZipFile(path))
                for path in sorted(
                    {item.archive_path for rows in sources.values() for item in rows}
                )
            }
            for video_id, rows in sources.items():
                fingerprint = _stable_json_hash(
                    {
                        "contract": CLOUD_ARTIFACT_CONTRACT,
                        "stage": "siglip2",
                        "model": model,
                        "frames": [row.frame.to_dict() for row in rows],
                    }
                )
                receipt = self._receipt_path("siglip2", video_id)
                if self._valid_receipt(receipt, fingerprint):
                    completed += 1
                    vectors_written += len(rows)
                    resumed += 1
                    self._report_progress(
                        stage="siglip2",
                        completed=completed,
                        total=total_videos,
                        records=vectors_written,
                        resumed=resumed,
                        started=started,
                        receipt=receipt,
                    )
                    continue
                encoded_chunks: list[np.ndarray] = []
                for start in range(0, len(rows), batch_size):
                    chunk = rows[start : start + batch_size]
                    payloads = [archives[item.archive_path].read(item.member) for item in chunk]
                    encoded = np.asarray(encoder.encode_images(payloads), dtype=np.float32)
                    if encoded.ndim != 2 or encoded.shape[0] != len(chunk):
                        raise CloudArtifactError("siglip2_vector_shape_invalid")
                    if not np.all(np.isfinite(encoded)):
                        raise CloudArtifactError("siglip2_vector_nonfinite")
                    norms = np.linalg.norm(encoded, axis=1)
                    if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
                        raise CloudArtifactError("siglip2_vector_norm_invalid")
                    encoded_chunks.append(encoded.astype(np.float16))
                values = np.concatenate(encoded_chunks, axis=0)
                shard = self._shard_path("siglip2", video_id)
                _atomic_npy(shard, values)
                self._write_receipt(
                    stage="siglip2",
                    video_id=video_id,
                    fingerprint=fingerprint,
                    shard=shard,
                    count=len(rows),
                    model=model,
                    frames=[item.frame for item in rows],
                    scanned_frames=len(rows),
                )
                completed += 1
                vectors_written += len(rows)
                self._report_progress(
                    stage="siglip2",
                    completed=completed,
                    total=total_videos,
                    records=vectors_written,
                    resumed=resumed,
                    started=started,
                    receipt=receipt,
                )
        return {"completed_videos": completed, "vectors": vectors_written}

    def run_ocr(self, engine: Any) -> dict[str, int]:
        """Run RapidOCR per streamed canonical image and write text shards."""

        sources = self._frame_sources()
        model = self._model_identity(engine, "ocr")
        threshold = max(0.5, float(getattr(engine, "confidence_threshold", 0.5)))
        completed = 0
        documents_written = 0
        resumed = 0
        started = self.clock()
        total_videos = len(sources)
        with ExitStack() as stack:
            archives = {
                path: stack.enter_context(zipfile.ZipFile(path))
                for path in sorted(
                    {item.archive_path for rows in sources.values() for item in rows}
                )
            }
            for video_id, rows in sources.items():
                fingerprint = _stable_json_hash(
                    {
                        "contract": CLOUD_ARTIFACT_CONTRACT,
                        "stage": "ocr",
                        "model": model,
                        "threshold": threshold,
                        "frames": [row.frame.to_dict() for row in rows],
                    }
                )
                receipt = self._receipt_path("ocr", video_id)
                if self._valid_receipt(receipt, fingerprint):
                    cached = json.loads(receipt.read_text(encoding="utf-8"))
                    completed += 1
                    documents_written += int(cached["count"])
                    resumed += 1
                    self._report_progress(
                        stage="ocr",
                        completed=completed,
                        total=total_videos,
                        records=documents_written,
                        resumed=resumed,
                        started=started,
                        receipt=receipt,
                    )
                    continue
                documents: list[dict[str, Any]] = []
                for item in rows:
                    image = archives[item.archive_path].read(item.member)
                    for index, detection in enumerate(engine.extract_text(image)):
                        confidence = float(detection.get("confidence", 0.0))
                        if not math.isfinite(confidence) or confidence < threshold:
                            continue
                        polygon = tuple(
                            tuple(float(value) for value in point)
                            for point in detection.get("polygon", ())
                        )
                        segment = TextSegment(
                            segment_id=(
                                f"ocr:{self.batch_id}:{video_id}:"
                                f"{item.frame.source_frame_idx}:{index}"
                            ),
                            batch_id=self.batch_id,
                            video_id=video_id,
                            kind="ocr",
                            start_ms=item.frame.timestamp_ms,
                            end_ms=item.frame.timestamp_ms,
                            source_frame_idx=item.frame.source_frame_idx,
                            text_raw=str(detection["text_raw"]),
                            text_nfc=str(detection["text_nfc"]),
                            text_accent_folded=str(detection["text_accent_folded"]),
                            confidence=confidence,
                            model_version=str(model["name"]),
                            polygons=(polygon,),
                        )
                        documents.append(_text_document(segment))
                shard = self._shard_path("ocr", video_id)
                count = _atomic_jsonl_gzip(shard, documents)
                self._write_receipt(
                    stage="ocr",
                    video_id=video_id,
                    fingerprint=fingerprint,
                    shard=shard,
                    count=count,
                    model=model,
                    scanned_frames=len(rows),
                )
                completed += 1
                documents_written += count
                self._report_progress(
                    stage="ocr",
                    completed=completed,
                    total=total_videos,
                    records=documents_written,
                    resumed=resumed,
                    started=started,
                    receipt=receipt,
                )
        return {"completed_videos": completed, "documents": documents_written}

    @staticmethod
    def _nearest_frame(rows: Sequence[dict[str, str]], timestamp_ms: int) -> tuple[int, int]:
        values = sorted(
            (round(float(row["pts_time"]) * 1000), int(row["frame_idx"])) for row in rows
        )
        timestamps = [value[0] for value in values]
        position = bisect.bisect_left(timestamps, timestamp_ms)
        candidates = values[max(0, position - 1) : min(len(values), position + 1)]
        if not candidates:
            raise CloudArtifactError("asr_frame_map_empty")
        return min(candidates, key=lambda value: (abs(value[0] - timestamp_ms), value[0]))

    @staticmethod
    def _stream_audio_member(
        archive: zipfile.ZipFile,
        member: str,
        audio_path: Path,
        *,
        ffmpeg_path: str | Path,
    ) -> bool:
        """Copy one ZIP member to seekable scratch before decoding its audio.

        Ordinary MP4 files commonly store their ``moov`` atom after ``mdat``.
        FFmpeg cannot seek back to media packets when such a file is supplied on
        ``pipe:0``; its resulting "no streams" error must not be recorded as a
        genuine no-audio video.
        """

        stderr_path = audio_path.with_suffix(".ffmpeg.log")
        suffix = PurePosixPath(member).suffix.casefold() or ".media"
        media_path = audio_path.with_name(f"source{suffix}")
        command = [
            str(ffmpeg_path),
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(audio_path),
        ]
        try:
            with archive.open(member) as source, media_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            with stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    check=False,
                )
            error_text = stderr_path.read_text(encoding="utf-8", errors="replace").casefold()
            no_audio = any(
                marker in error_text
                for marker in ("matches no streams", "does not contain any stream")
            )
            input_failure = any(
                marker in error_text
                for marker in (
                    "invalid data found when processing input",
                    "moov atom not found",
                    "partial file",
                    "error reading header",
                    "could not find codec parameters",
                )
            )
            if completed.returncode != 0:
                if no_audio and not input_failure:
                    return False
                raise CloudArtifactError("ffmpeg_audio_stream_failed")
            if not audio_path.is_file() or audio_path.stat().st_size <= 44:
                raise CloudArtifactError("ffmpeg_audio_output_invalid")
            return True
        finally:
            media_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)

    def run_asr(self, engine: Any, *, ffmpeg_path: str | Path = "ffmpeg") -> dict[str, int]:
        """Stream zipped videos through ffmpeg and write Faster-Whisper shards."""

        if not self.video_archives:
            raise ValueError("at least one video ZIP is required for ASR")
        maps = self._map_rows()
        videos = self._video_index(self.video_archives)
        selected = sorted(set(maps) & set(videos))
        if self.selected_video_ids is not None:
            missing = self.selected_video_ids - set(selected)
            if missing:
                raise CloudArtifactError("partition_video_missing")
            selected = [value for value in selected if value in self.selected_video_ids]
        if not selected:
            raise CloudArtifactError("no_mapped_videos")
        model = self._model_identity(engine, "asr")
        completed = 0
        documents_written = 0
        resumed = 0
        started = self.clock()
        total_videos = len(selected)
        scratch_root = self.scratch_dir
        scratch_root.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            archives = {
                path: stack.enter_context(zipfile.ZipFile(path))
                for path in sorted({videos[video_id][0] for video_id in selected})
            }
            for video_id in selected:
                fingerprint = _stable_json_hash(
                    {
                        "contract": CLOUD_ARTIFACT_CONTRACT,
                        "stage": "asr",
                        "audio_extraction_contract": ASR_AUDIO_EXTRACTION_CONTRACT,
                        "model": model,
                        "video_member": videos[video_id][1],
                        "map_rows": maps[video_id],
                    }
                )
                receipt = self._receipt_path("asr", video_id)
                if self._valid_receipt(receipt, fingerprint):
                    cached = json.loads(receipt.read_text(encoding="utf-8"))
                    completed += 1
                    documents_written += int(cached["count"])
                    resumed += 1
                    self._report_progress(
                        stage="asr",
                        completed=completed,
                        total=total_videos,
                        records=documents_written,
                        resumed=resumed,
                        started=started,
                        receipt=receipt,
                    )
                    continue
                archive_path, member = videos[video_id]
                documents: list[dict[str, Any]] = []
                with tempfile.TemporaryDirectory(prefix=f"{video_id}-", dir=scratch_root) as tmp:
                    audio_path = Path(tmp) / "audio.wav"
                    has_audio = self._stream_audio_member(
                        archives[archive_path],
                        member,
                        audio_path,
                        ffmpeg_path=ffmpeg_path,
                    )
                    transcript = engine.transcribe(audio_path) if has_audio else {"segments": ()}
                for index, value in enumerate(transcript.get("segments", ())):
                    text_raw = str(value.get("text_raw", "")).strip()
                    if not text_raw:
                        continue
                    start_ms = int(value["start_ms"])
                    end_ms = int(value["end_ms"])
                    _, source_frame_idx = self._nearest_frame(
                        maps[video_id], start_ms + (end_ms - start_ms) // 2
                    )
                    confidence = max(
                        0.0,
                        min(1.0, math.exp(float(value.get("avg_logprob", 0.0)))),
                    )
                    segment = TextSegment(
                        segment_id=f"asr:{self.batch_id}:{video_id}:{index}",
                        batch_id=self.batch_id,
                        video_id=video_id,
                        kind="asr",
                        start_ms=start_ms,
                        end_ms=end_ms,
                        source_frame_idx=source_frame_idx,
                        text_raw=text_raw,
                        text_nfc=str(value["text_nfc"]),
                        text_accent_folded=str(value["text_accent_folded"]),
                        confidence=confidence,
                        model_version=str(model["name"]),
                    )
                    documents.append(_text_document(segment))
                shard = self._shard_path("asr", video_id)
                count = _atomic_jsonl_gzip(shard, documents)
                self._write_receipt(
                    stage="asr",
                    video_id=video_id,
                    fingerprint=fingerprint,
                    shard=shard,
                    count=count,
                    model=model,
                    extraction_contract=ASR_AUDIO_EXTRACTION_CONTRACT,
                    audio_stream_present=has_audio,
                    transcript_segment_count=count,
                    zero_transcript_reason=(
                        None
                        if count > 0
                        else "no_speech_detected"
                        if has_audio
                        else "no_audio_stream"
                    ),
                )
                completed += 1
                documents_written += count
                self._report_progress(
                    stage="asr",
                    completed=completed,
                    total=total_videos,
                    records=documents_written,
                    resumed=resumed,
                    started=started,
                    receipt=receipt,
                )
        try:
            scratch_root.rmdir()
        except OSError:
            pass
        return {"completed_videos": completed, "documents": documents_written}

    def _stage_receipts(self, stage: str) -> list[dict[str, Any]]:
        directory = self.work_dir / stage
        if not directory.is_dir():
            return []
        receipts: list[dict[str, Any]] = []
        for path in sorted(directory.rglob("*.json")):
            if path.name in {"progress.json", "b01_finalize_receipt.json"}:
                continue
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (
                receipt.get("contract") != CLOUD_ARTIFACT_CONTRACT
                or receipt.get("batch_id") != self.batch_id
                or receipt.get("stage") != stage
            ):
                raise CloudArtifactError("cloud_receipt_contract_invalid")
            if (
                stage == "asr"
                and receipt.get("extraction_contract") != ASR_AUDIO_EXTRACTION_CONTRACT
            ):
                raise CloudArtifactError("cloud_asr_extraction_contract_invalid")
            shard = self.work_dir / str(receipt["artifact"])
            if (
                not shard.is_file()
                or shard.stat().st_size != int(receipt["size_bytes"])
                or sha256_file(shard) != receipt["sha256"]
            ):
                raise CloudArtifactError("cloud_shard_hash_mismatch")
            receipts.append(receipt)
        video_ids = [str(receipt["video_id"]) for receipt in receipts]
        if len(video_ids) != len(set(video_ids)):
            raise CloudArtifactError("duplicate_cloud_receipt_video")
        return sorted(receipts, key=lambda value: str(value["video_id"]))

    @staticmethod
    def _ocr_rows_with_frame_aggregates(shard: Path) -> Iterator[dict[str, Any]]:
        """Yield OCR detections plus deterministic multi-line frame documents."""

        documents: list[dict[str, Any]] = []
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise CloudArtifactError("text_shard_row_invalid")
                    _segment_from_document(value)
                    documents.append(value)
                    yield value
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for document in documents:
            source_frame_idx = document.get("source_frame_idx")
            if isinstance(source_frame_idx, int):
                by_frame.setdefault(source_frame_idx, []).append(document)

        def order_key(document: Mapping[str, Any]) -> tuple[float, float, str]:
            polygons = document.get("polygons")
            points = polygons[0] if isinstance(polygons, (list, tuple)) and polygons else ()
            xs = [float(point[0]) for point in points if len(point) == 2]
            ys = [float(point[1]) for point in points if len(point) == 2]
            return (
                min(ys, default=float("inf")),
                min(xs, default=float("inf")),
                str(document.get("segment_id", "")),
            )

        for source_frame_idx in sorted(by_frame):
            components = sorted(by_frame[source_frame_idx], key=order_key)
            if len(components) < 2:
                continue
            first = components[0]
            raw_text = " ".join(str(value["text_raw"]).strip() for value in components)
            nfc_text = " ".join(str(value["text_nfc"]).strip() for value in components)
            folded_text = " ".join(str(value["text_accent_folded"]).strip() for value in components)
            polygons = tuple(
                tuple(tuple(float(coordinate) for coordinate in point) for point in polygon)
                for value in components
                for polygon in value.get("polygons", ())
            )
            segment = TextSegment(
                segment_id=(
                    f"ocr-frame:{first['batch_id']}:{first['video_id']}:{source_frame_idx}"
                ),
                batch_id=str(first["batch_id"]),
                video_id=str(first["video_id"]),
                kind="ocr",
                start_ms=int(first["start_ms"]),
                end_ms=int(first["end_ms"]),
                source_frame_idx=source_frame_idx,
                text_raw=raw_text,
                text_nfc=nfc_text,
                text_accent_folded=folded_text,
                confidence=sum(float(value["confidence"]) for value in components)
                / len(components),
                model_version=f"{first['model_version']}+frame-aggregate-v1",
                polygons=polygons,
            )
            aggregate = _text_document(segment)
            aggregate["aggregation"] = "frame"
            aggregate["constituent_segment_ids"] = [
                str(value["segment_id"]) for value in components
            ]
            yield aggregate

    def finalize(
        self,
        *,
        channels: Sequence[str] = _STAGES,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Atomically combine verified shards into a progressive B01 manifest."""

        selected = _normalized_channels(channels, require_siglip2=True)
        receipts = {stage: self._stage_receipts(stage) for stage in selected}
        if any(not receipts[stage] for stage in selected):
            raise CloudArtifactError("cloud_stage_shards_incomplete")
        for stage, stage_receipts in receipts.items():
            model_fingerprints = {_stable_json_hash(value.get("model")) for value in stage_receipts}
            if len(model_fingerprints) != 1:
                raise CloudArtifactError(f"cloud_{stage}_model_mismatch")
        video_sets = {
            stage: {str(value["video_id"]) for value in receipts[stage]} for stage in selected
        }
        siglip_videos = video_sets["siglip2"]
        if allow_partial:
            if any(
                not values or not values.issubset(siglip_videos)
                for stage, values in video_sets.items()
                if stage != "siglip2"
            ):
                raise CloudArtifactError("cloud_stage_video_mismatch")
        elif len({frozenset(values) for values in video_sets.values()}) != 1:
            raise CloudArtifactError("cloud_stage_video_mismatch")
        video_ids = sorted(video_sets["siglip2"])

        frames: list[dict[str, Any]] = []
        dimensions: set[int] = set()
        vector_count = 0
        for receipt in receipts["siglip2"]:
            rows = receipt.get("frames", [])
            if not isinstance(rows, list) or len(rows) != int(receipt["count"]):
                raise CloudArtifactError("siglip2_receipt_rows_invalid")
            shard = self.work_dir / str(receipt["artifact"])
            matrix = np.load(shard, allow_pickle=False, mmap_mode="r")
            if matrix.ndim != 2 or matrix.shape[0] != len(rows) or matrix.dtype != np.float16:
                raise CloudArtifactError("siglip2_shard_contract_invalid")
            dimensions.add(int(matrix.shape[1]))
            for start in range(0, len(matrix), 4096):
                chunk = np.asarray(matrix[start : start + 4096], dtype=np.float32)
                if not np.all(np.isfinite(chunk)):
                    raise CloudArtifactError("cloud_vector_nonfinite")
                norms = np.linalg.norm(chunk, axis=1)
                if not np.allclose(norms, 1.0, rtol=3e-3, atol=3e-3):
                    raise CloudArtifactError("cloud_vector_norm_invalid")
            vector_count += len(rows)
            frames.extend(rows)
        if len(dimensions) != 1:
            raise CloudArtifactError("siglip2_dimension_mismatch")
        dimension = next(iter(dimensions))
        expected_dimension = self.model_contracts.get("siglip2", {}).get("dimension")
        if expected_dimension is not None and dimension != int(expected_dimension):
            raise CloudArtifactError("cloud_vector_dimension_mismatch")
        identities = [(row.get("video_id"), row.get("source_frame_idx")) for row in frames]
        if len(identities) != len(set(identities)):
            raise CloudArtifactError("duplicate_canonical_vector_identity")
        if self.expected_videos is not None and len(video_ids) != self.expected_videos:
            raise CloudArtifactError("cloud_expected_video_count_mismatch")
        if self.expected_frames is not None and vector_count != self.expected_frames:
            raise CloudArtifactError("cloud_expected_frame_count_mismatch")
        if not allow_partial:
            if "ocr" in selected:
                ocr_scanned = sum(
                    int(value.get("scanned_frames") or 0) for value in receipts["ocr"]
                )
                if self.expected_frames is not None and ocr_scanned != self.expected_frames:
                    raise CloudArtifactError("cloud_expected_ocr_frame_count_mismatch")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        vector_target = self.output_dir / SIGLIP_FEATURES_NAME
        reuse_vector = False
        previous_manifest_path = self.output_dir / SIGLIP_MANIFEST_NAME
        if vector_target.is_file() and previous_manifest_path.is_file():
            try:
                previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
                previous_file = previous["files"][SIGLIP_FEATURES_NAME]
                previous_vector = previous["vector"]
                reuse_vector = (
                    previous_vector.get("rows") == frames
                    and int(previous_vector.get("count", -1)) == vector_count
                    and int(previous_vector.get("dimension", -1)) == dimension
                    and vector_target.stat().st_size == int(previous_file.get("size_bytes", -1))
                    and sha256_file(vector_target) == previous_file.get("sha256")
                )
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                reuse_vector = False
        if not reuse_vector:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{vector_target.stem}.", suffix=".npy", dir=self.output_dir
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                matrix = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.float16,
                    shape=(vector_count, dimension),
                )
                position = 0
                for receipt in receipts["siglip2"]:
                    shard = np.load(
                        self.work_dir / str(receipt["artifact"]),
                        allow_pickle=False,
                        mmap_mode="r",
                    )
                    for start in range(0, len(shard), 4096):
                        stop = min(start + 4096, len(shard))
                        matrix[position + start : position + stop] = shard[start:stop]
                    position += len(shard)
                matrix.flush()
                del matrix
                os.replace(temporary, vector_target)
            finally:
                temporary.unlink(missing_ok=True)

        text_metadata: dict[str, Any] = {}
        for stage, target_name in _TEXT_STAGE_FILES.items():
            if stage not in selected:
                continue
            target = self.output_dir / target_name
            video_counts: dict[str, int] = {}

            def rows() -> Iterator[Mapping[str, Any]]:
                for receipt in receipts[stage]:
                    shard = self.work_dir / str(receipt["artifact"])
                    emitted = 0
                    if stage == "ocr":
                        iterator: Iterable[Mapping[str, Any]] = (
                            self._ocr_rows_with_frame_aggregates(shard)
                        )
                    else:

                        def text_rows() -> Iterator[Mapping[str, Any]]:
                            with gzip.open(shard, "rt", encoding="utf-8") as handle:
                                for line in handle:
                                    if line.strip():
                                        value = json.loads(line)
                                        if not isinstance(value, Mapping):
                                            raise CloudArtifactError("text_shard_row_invalid")
                                        yield value

                        iterator = text_rows()
                    for value in iterator:
                        emitted += 1
                        yield value
                    video_counts[str(receipt["video_id"])] = emitted

            count = _atomic_jsonl_gzip(target, rows())
            if count != sum(video_counts.values()):
                raise CloudArtifactError("text_shard_count_mismatch")
            aggregation_contract = "frame-aggregate-v1" if stage == "ocr" else "segments-v1"
            video_fingerprints = {
                str(receipt["video_id"]): _stable_json_hash(
                    {
                        "stage": stage,
                        "artifact_sha256": receipt["sha256"],
                        "count": int(receipt["count"]),
                        "scanned_frames": receipt.get("scanned_frames"),
                        "model": receipt["model"],
                        "aggregation_contract": aggregation_contract,
                        "extraction_contract": receipt.get("extraction_contract"),
                    }
                )
                for receipt in receipts[stage]
            }
            text_metadata[stage] = {
                "file": target_name,
                "count": count,
                "model": receipts[stage][0]["model"],
                "completed_videos": len(receipts[stage]),
                "scanned_frames": sum(
                    int(value.get("scanned_frames") or 0) for value in receipts[stage]
                ),
                "videos": video_counts,
                "video_fingerprints": video_fingerprints,
                "aggregation_contract": aggregation_contract,
            }
            if stage == "asr":
                text_metadata[stage]["extraction_contract"] = ASR_AUDIO_EXTRACTION_CONTRACT

        data_files = tuple(self.output_dir / _channel_file_name(stage) for stage in selected)
        files = {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in data_files
        }
        vector_metadata = {
            "file": SIGLIP_FEATURES_NAME,
            "dtype": "float16",
            "dimension": dimension,
            "count": vector_count,
            "model": receipts["siglip2"][0]["model"],
            "rows": frames,
        }
        channel_fingerprints = {
            "siglip2": _stable_json_hash(
                {
                    "stage": "siglip2",
                    "file": files[SIGLIP_FEATURES_NAME],
                    "count": vector_count,
                    "dimension": dimension,
                    "model": vector_metadata["model"],
                    "rows_sha256": _stable_json_hash(frames),
                }
            )
        }
        for stage in ("ocr", "asr"):
            if stage in selected:
                channel_fingerprints[stage] = _stable_json_hash(
                    {
                        "stage": stage,
                        "file": files[_TEXT_STAGE_FILES[stage]],
                        "metadata": text_metadata[stage],
                    }
                )
        channel_coverage: dict[str, dict[str, Any]] = {
            "siglip2": {
                "coverage_complete": bool(
                    (self.expected_videos is None or len(video_ids) == self.expected_videos)
                    and (self.expected_frames is None or vector_count == self.expected_frames)
                ),
                "completed_videos": len(video_ids),
                "expected_videos": self.expected_videos,
                "records": vector_count,
                "expected_records": self.expected_frames,
            }
        }
        for stage in ("ocr", "asr"):
            if stage not in selected:
                continue
            metadata = text_metadata[stage]
            completed_videos = int(metadata["completed_videos"])
            scanned_frames = int(metadata["scanned_frames"])
            coverage_complete = bool(
                self.expected_videos is not None
                and completed_videos == self.expected_videos
                and (
                    stage != "ocr"
                    or self.expected_frames is None
                    or scanned_frames == self.expected_frames
                )
            )
            channel_coverage[stage] = {
                "coverage_complete": coverage_complete,
                "completed_videos": completed_videos,
                "expected_videos": self.expected_videos,
                "records": int(metadata["count"]),
                "scanned_frames": scanned_frames if stage == "ocr" else None,
                "expected_frames": self.expected_frames if stage == "ocr" else None,
            }
        complete = selected == _STAGES and not allow_partial
        manifest = {
            "schema_version": CLOUD_ARTIFACT_SCHEMA_VERSION,
            "contract": CLOUD_ARTIFACT_CONTRACT,
            "batch_id": self.batch_id,
            "created_at_unix": time.time(),
            "complete": complete,
            "completed_channels": list(selected),
            "channel_fingerprints": channel_fingerprints,
            "channel_coverage": channel_coverage,
            "videos": len(video_ids),
            "tool_versions": _tool_versions(),
            "input_archives": {
                "maps": _zip_directory_fingerprint(self.map_archive),
                "keyframes": [
                    _zip_directory_fingerprint(path)
                    for path in sorted(self.keyframe_archives, key=lambda value: value.name)
                ],
                "videos": [
                    _zip_directory_fingerprint(path)
                    for path in sorted(self.video_archives, key=lambda value: value.name)
                ],
            },
            "model_contracts": self.model_contracts,
            "vector": vector_metadata,
            "text": text_metadata,
            "files": files,
        }
        manifest_target = self.output_dir / SIGLIP_MANIFEST_NAME
        _atomic_json(manifest_target, manifest)
        receipt = {
            "contract": CLOUD_ARTIFACT_CONTRACT,
            "batch_id": self.batch_id,
            "manifest": SIGLIP_MANIFEST_NAME,
            "manifest_sha256": sha256_file(manifest_target),
            "completed_channels": list(selected),
            "complete": complete,
            "channel_coverage": channel_coverage,
            "videos": len(video_ids),
            "vectors": vector_count,
        }
        _atomic_json(self.work_dir / "b01_finalize_receipt.json", receipt)
        return manifest


class CloudArtifactBundle:
    """Validated, memory-mapped view over finalized cloud artifacts."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        expected_batch_id: str,
        expected_model_contracts: Mapping[str, Mapping[str, Any]] | None = None,
        expected_videos: int | None = None,
        expected_frames: int | None = None,
        allow_partial: bool = False,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.expected_batch_id = _validated_video_id(expected_batch_id)
        self.expected_model_contracts: dict[str, dict[str, Any]] = {}
        for stage, value in (expected_model_contracts or {}).items():
            contract = dict(value)
            contract.setdefault("settings_fingerprint", _stable_json_hash(contract))
            self.expected_model_contracts[str(stage)] = contract
        self.expected_videos = expected_videos
        self.expected_frames = expected_frames
        self.allow_partial = bool(allow_partial)
        self.manifest_path = self.artifact_dir / SIGLIP_MANIFEST_NAME
        self.manifest: dict[str, Any] | None = None
        self.vectors: np.ndarray | None = None

    def validate(self, channels: Sequence[str] = _STAGES) -> dict[str, int]:
        """Validate only the requested, already completed artifact channels."""

        selected = _normalized_channels(channels)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CloudArtifactError("cloud_manifest_invalid") from error
        if not isinstance(manifest, dict):
            raise CloudArtifactError("cloud_manifest_contract_mismatch")
        schema_version = int(manifest.get("schema_version", -1))
        if (
            schema_version
            not in {LEGACY_CLOUD_ARTIFACT_SCHEMA_VERSION, CLOUD_ARTIFACT_SCHEMA_VERSION}
            or manifest.get("contract") != CLOUD_ARTIFACT_CONTRACT
            or manifest.get("batch_id") != self.expected_batch_id
        ):
            raise CloudArtifactError("cloud_manifest_contract_mismatch")
        if schema_version == LEGACY_CLOUD_ARTIFACT_SCHEMA_VERSION:
            if manifest.get("complete") is not True:
                raise CloudArtifactError("cloud_manifest_contract_mismatch")
            completed_channels = _STAGES
        else:
            raw_completed = manifest.get("completed_channels")
            if not isinstance(raw_completed, list):
                raise CloudArtifactError("cloud_completed_channels_invalid")
            completed_channels = _normalized_channels(raw_completed, require_siglip2=True)
            manifest_complete = manifest.get("complete") is True
            raw_coverage = manifest.get("channel_coverage")
            if manifest_complete and completed_channels != _STAGES:
                raise CloudArtifactError("cloud_manifest_completion_invalid")
            if (
                completed_channels == _STAGES
                and not manifest_complete
                and not isinstance(raw_coverage, Mapping)
            ):
                raise CloudArtifactError("cloud_manifest_completion_invalid")
        if any(stage not in completed_channels for stage in selected):
            raise CloudArtifactError("cloud_requested_channel_incomplete")
        channel_coverage: Mapping[str, Any] = {}
        if schema_version == CLOUD_ARTIFACT_SCHEMA_VERSION:
            raw_coverage = manifest.get("channel_coverage")
            if raw_coverage is not None:
                if not isinstance(raw_coverage, Mapping) or set(raw_coverage) != set(
                    completed_channels
                ):
                    raise CloudArtifactError("cloud_channel_coverage_invalid")
                for stage in completed_channels:
                    value = raw_coverage.get(stage)
                    if not isinstance(value, Mapping) or not isinstance(
                        value.get("coverage_complete"), bool
                    ):
                        raise CloudArtifactError("cloud_channel_coverage_invalid")
                channel_coverage = raw_coverage
                if not self.allow_partial and any(
                    not bool(channel_coverage[stage]["coverage_complete"]) for stage in selected
                ):
                    raise CloudArtifactError("cloud_partial_bundle_requires_opt_in")
        if self.expected_videos is not None and int(manifest.get("videos", -1)) != int(
            self.expected_videos
        ):
            raise CloudArtifactError("cloud_expected_video_count_mismatch")
        versions = manifest.get("tool_versions")
        inputs = manifest.get("input_archives")
        if not isinstance(versions, Mapping) or not isinstance(inputs, Mapping):
            raise CloudArtifactError("cloud_manifest_provenance_invalid")
        if self.expected_model_contracts:
            configured = manifest.get("model_contracts")
            if not isinstance(configured, Mapping):
                raise CloudArtifactError("cloud_model_contract_missing")
            for stage in selected:
                expected = self.expected_model_contracts.get(stage)
                if expected is None:
                    continue
                actual = configured.get(stage)
                if not isinstance(actual, Mapping) or actual.get(
                    "settings_fingerprint"
                ) != expected.get("settings_fingerprint"):
                    raise CloudArtifactError("cloud_model_contract_mismatch")
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise CloudArtifactError("cloud_manifest_files_invalid")
        completed_names = {_channel_file_name(stage) for stage in completed_channels}
        if set(files) != completed_names:
            raise CloudArtifactError("cloud_manifest_file_set_invalid")
        for name in {_channel_file_name(stage) for stage in selected}:
            path = self.artifact_dir / name
            metadata = files[name]
            if (
                not path.is_file()
                or not isinstance(metadata, Mapping)
                or path.stat().st_size != int(metadata.get("size_bytes", -1))
                or sha256_file(path) != metadata.get("sha256")
            ):
                raise CloudArtifactError("cloud_artifact_hash_mismatch")

        channel_fingerprints = manifest.get("channel_fingerprints")
        if schema_version == CLOUD_ARTIFACT_SCHEMA_VERSION:
            if not isinstance(channel_fingerprints, Mapping) or set(channel_fingerprints) != set(
                completed_channels
            ):
                raise CloudArtifactError("cloud_channel_fingerprints_invalid")
            if any(
                not str(channel_fingerprints.get(stage, "")).strip() for stage in completed_channels
            ):
                raise CloudArtifactError("cloud_channel_fingerprints_invalid")

        vector = manifest.get("vector")
        if not isinstance(vector, Mapping) or vector.get("file") != SIGLIP_FEATURES_NAME:
            raise CloudArtifactError("cloud_vector_manifest_invalid")
        counts: dict[str, int] = {}
        matrix: np.ndarray | None = None
        if "siglip2" in selected:
            vector_model = vector.get("model")
            if (
                not isinstance(vector_model, Mapping)
                or not str(vector_model.get("fingerprint", "")).strip()
                or vector_model.get("fingerprint") == "unknown"
            ):
                raise CloudArtifactError("cloud_model_fingerprint_invalid")
            expected_visual = self.expected_model_contracts.get("siglip2", {})
            if expected_visual and (
                vector_model.get("name") != expected_visual.get("name")
                or vector_model.get("revision") != expected_visual.get("revision")
            ):
                raise CloudArtifactError("cloud_model_identity_mismatch")
            matrix = np.load(
                self.artifact_dir / SIGLIP_FEATURES_NAME,
                allow_pickle=False,
                mmap_mode="r",
            )
            expected_shape = (
                int(vector.get("count", -1)),
                int(vector.get("dimension", -1)),
            )
            if self.expected_frames is not None and expected_shape[0] != int(self.expected_frames):
                raise CloudArtifactError("cloud_expected_frame_count_mismatch")
            expected_dimension = expected_visual.get("dimension")
            if expected_dimension is not None and expected_shape[1] != int(expected_dimension):
                raise CloudArtifactError("cloud_vector_dimension_mismatch")
            if matrix.dtype != np.float16 or matrix.shape != expected_shape or matrix.ndim != 2:
                raise CloudArtifactError("cloud_vector_contract_invalid")
            rows = vector.get("rows")
            if not isinstance(rows, list) or len(rows) != len(matrix):
                raise CloudArtifactError("cloud_vector_rows_invalid")
            identities: set[tuple[str, int]] = set()
            for row in rows:
                frame = FrameRecord.from_dict(row)
                if frame.batch_id != self.expected_batch_id:
                    raise CloudArtifactError("cloud_frame_batch_mismatch")
                identity = (frame.video_id, frame.source_frame_idx)
                if identity in identities:
                    raise CloudArtifactError("duplicate_canonical_vector_identity")
                identities.add(identity)
            for start in range(0, len(matrix), 4096):
                chunk = np.asarray(matrix[start : start + 4096], dtype=np.float32)
                if not np.all(np.isfinite(chunk)):
                    raise CloudArtifactError("cloud_vector_nonfinite")
                norms = np.linalg.norm(chunk, axis=1)
                if not np.allclose(norms, 1.0, rtol=3e-3, atol=3e-3):
                    raise CloudArtifactError("cloud_vector_norm_invalid")
            counts["siglip2"] = len(matrix)

        text = manifest.get("text")
        if not isinstance(text, Mapping):
            raise CloudArtifactError("cloud_text_manifest_invalid")
        for stage, name in _TEXT_STAGE_FILES.items():
            if stage not in selected:
                continue
            metadata = text.get(stage)
            if not isinstance(metadata, Mapping) or metadata.get("file") != name:
                raise CloudArtifactError("cloud_text_manifest_invalid")
            if (
                stage == "asr"
                and schema_version == CLOUD_ARTIFACT_SCHEMA_VERSION
                and metadata.get("extraction_contract") != ASR_AUDIO_EXTRACTION_CONTRACT
            ):
                raise CloudArtifactError("cloud_asr_extraction_contract_invalid")
            text_model = metadata.get("model")
            expected_text_model = self.expected_model_contracts.get(stage, {})
            if (
                not isinstance(text_model, Mapping)
                or not str(text_model.get("fingerprint", "")).strip()
                or text_model.get("fingerprint") == "unknown"
                or (
                    expected_text_model
                    and text_model.get("name") != expected_text_model.get("name")
                )
            ):
                raise CloudArtifactError("cloud_model_identity_mismatch")
            count = 0
            previous_video = ""
            for document in self.iter_documents(stage, validate_contract=False):
                segment = _segment_from_document(document)
                if segment.batch_id != self.expected_batch_id or segment.kind != stage:
                    raise CloudArtifactError("cloud_text_contract_mismatch")
                if segment.video_id < previous_video:
                    raise CloudArtifactError("cloud_text_video_order_invalid")
                previous_video = segment.video_id
                count += 1
            if count != int(metadata.get("count", -1)):
                raise CloudArtifactError("cloud_text_count_mismatch")
            video_counts = metadata.get("videos")
            if not isinstance(video_counts, Mapping) or any(
                not _SAFE_VIDEO_ID.fullmatch(str(video_id)) or int(value) < 0
                for video_id, value in video_counts.items()
            ):
                raise CloudArtifactError("cloud_text_video_counts_invalid")
            if sum(int(value) for value in video_counts.values()) != count:
                raise CloudArtifactError("cloud_text_video_counts_mismatch")
            completed_videos = int(metadata.get("completed_videos", -1))
            if completed_videos != len(video_counts):
                raise CloudArtifactError("cloud_text_video_count_mismatch")
            coverage = channel_coverage.get(stage, {})
            coverage_complete = bool(coverage.get("coverage_complete", True))
            if (
                self.expected_videos is not None
                and completed_videos != int(self.expected_videos)
                and not (self.allow_partial and not coverage_complete)
            ):
                raise CloudArtifactError("cloud_text_video_count_mismatch")
            if self.expected_videos is not None and not 0 < completed_videos <= int(
                self.expected_videos
            ):
                raise CloudArtifactError("cloud_text_video_count_mismatch")
            video_fingerprints = metadata.get("video_fingerprints")
            if video_fingerprints is not None and (
                not isinstance(video_fingerprints, Mapping)
                or set(str(value) for value in video_fingerprints)
                != set(str(value) for value in video_counts)
                or any(not str(value).strip() for value in video_fingerprints.values())
            ):
                raise CloudArtifactError("cloud_text_video_fingerprints_invalid")
            if (
                self.allow_partial
                and not coverage_complete
                and not isinstance(video_fingerprints, Mapping)
            ):
                raise CloudArtifactError("cloud_text_video_fingerprints_invalid")
            vector_rows = vector.get("rows")
            if not isinstance(vector_rows, list):
                raise CloudArtifactError("cloud_vector_rows_invalid")
            vector_video_ids = {
                str(value.get("video_id")) for value in vector_rows if isinstance(value, Mapping)
            }
            if not set(str(value) for value in video_counts).issubset(vector_video_ids):
                raise CloudArtifactError("cloud_stage_video_mismatch")
            scanned_frames = int(metadata.get("scanned_frames", -1))
            if (
                stage == "ocr"
                and self.expected_frames is not None
                and scanned_frames != int(self.expected_frames)
                and not (self.allow_partial and not coverage_complete)
            ):
                raise CloudArtifactError("cloud_expected_ocr_frame_count_mismatch")
            if (
                stage == "ocr"
                and self.expected_frames is not None
                and not (0 < scanned_frames <= int(self.expected_frames))
            ):
                raise CloudArtifactError("cloud_expected_ocr_frame_count_mismatch")
            if coverage:
                manifest_expected_videos = int(manifest.get("videos", -1))
                manifest_expected_frames = int(vector.get("count", -1))
                expected_coverage = {
                    "completed_videos": completed_videos,
                    "expected_videos": manifest_expected_videos,
                    "records": count,
                }
                if any(coverage.get(key) != value for key, value in expected_coverage.items()):
                    raise CloudArtifactError("cloud_channel_coverage_invalid")
                if stage == "ocr" and (
                    coverage.get("scanned_frames") != scanned_frames
                    or coverage.get("expected_frames") != manifest_expected_frames
                ):
                    raise CloudArtifactError("cloud_channel_coverage_invalid")
            counts[stage] = count
        self.manifest = manifest
        self.vectors = matrix
        return counts

    def channel_fingerprint(self, stage: str) -> str:
        """Return a manifest-growth-stable fingerprint for one channel."""

        if self.manifest is None:
            raise CloudArtifactError("cloud_bundle_not_validated")
        if stage not in _STAGES:
            raise ValueError("stage must be siglip2, ocr, or asr")
        configured = self.manifest.get("channel_fingerprints")
        if isinstance(configured, Mapping) and str(configured.get(stage, "")).strip():
            return str(configured[stage])
        files = self.manifest.get("files", {})
        name = _channel_file_name(stage)
        descriptor: dict[str, Any] = {
            "stage": stage,
            "file": files.get(name),
        }
        descriptor["metadata"] = (
            self.manifest.get("vector")
            if stage == "siglip2"
            else self.manifest.get("text", {}).get(stage)
        )
        return _stable_json_hash(descriptor)

    def channel_coverage(self, stage: str) -> dict[str, Any]:
        """Return validated coverage metadata for one available channel."""

        if self.manifest is None:
            raise CloudArtifactError("cloud_bundle_not_validated")
        if stage not in _STAGES:
            raise ValueError("stage must be siglip2, ocr, or asr")
        configured = self.manifest.get("channel_coverage", {})
        value = configured.get(stage) if isinstance(configured, Mapping) else None
        if isinstance(value, Mapping):
            return dict(value)
        return {"coverage_complete": True}

    def video_fingerprint(self, stage: str, video_id: str) -> str:
        """Return the stable finalized fingerprint for one text video shard."""

        if self.manifest is None:
            raise CloudArtifactError("cloud_bundle_not_validated")
        if stage not in {"ocr", "asr"}:
            raise ValueError("stage must be ocr or asr")
        metadata = self.manifest.get("text", {}).get(stage, {})
        configured = metadata.get("video_fingerprints", {})
        value = configured.get(video_id) if isinstance(configured, Mapping) else None
        if str(value or "").strip():
            return str(value)
        return self.channel_fingerprint(stage)

    def iter_documents(
        self, stage: str, *, validate_contract: bool = True
    ) -> Iterator[dict[str, Any]]:
        if stage not in _TEXT_STAGE_FILES:
            raise ValueError("stage must be ocr or asr")
        path = self.artifact_dir / _TEXT_STAGE_FILES[stage]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CloudArtifactError("cloud_text_row_invalid")
                if validate_contract:
                    _segment_from_document(value)
                yield value

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.manifest_path)


class CloudArtifactImporter:
    """Import a validated bundle with bounded batches and resumable ledger stages."""

    def __init__(
        self,
        *,
        bundle: CloudArtifactBundle,
        vector_store: Any,
        ocr_store: Any,
        asr_store: Any,
        ledger: JobLedger | None = None,
        completion_path: str | Path | None = None,
        channel_receipt_dir: str | Path | None = None,
        vector_batch_size: int = 1000,
        ocr_batch_size: int = 2000,
        asr_batch_size: int = 1000,
        clock: Any = time.monotonic,
    ) -> None:
        self.bundle = bundle
        self.vector_store = vector_store
        self.text_stores = {"ocr": ocr_store, "asr": asr_store}
        self.ledger = ledger
        self.completion_path = None if completion_path is None else Path(completion_path)
        self.channel_receipt_dir = (
            None if channel_receipt_dir is None else Path(channel_receipt_dir)
        )
        self.batch_sizes = {
            "siglip2": int(vector_batch_size),
            "ocr": int(ocr_batch_size),
            "asr": int(asr_batch_size),
        }
        if self.batch_sizes != {"siglip2": 1000, "ocr": 2000, "asr": 1000}:
            raise ValueError("cloud import batch sizes must be 1000/2000/1000")
        self.clock = clock

    def _register_video(self, video_id: str, source_uri: str) -> None:
        if self.ledger is None:
            return
        self.ledger.register_video(
            self.bundle.expected_batch_id,
            video_id,
            source_uri,
        )

    def _start_stage(self, video_id: str, stage: str, *, resume: bool) -> tuple[bool, str]:
        source_fingerprint = (
            self.bundle.video_fingerprint(stage, video_id)
            if stage in {"ocr", "asr"}
            else self.bundle.channel_fingerprint(stage)
        )
        fingerprint = f"cloud-import-{stage}-v3:{source_fingerprint[:24]}"
        if self.ledger is None:
            return True, fingerprint
        try:
            transition = self.ledger.start_stage(
                self.bundle.expected_batch_id,
                video_id,
                f"reference_{stage}",
                fingerprint,
                allow_reprocess=not resume,
            )
        except StageFingerprintMismatch:
            # A completed legacy/reference stage cannot transition directly to
            # ``failed``.  Re-open it transactionally instead; ``start_stage``
            # removes the stale artifact rows and increments the attempt before
            # any external-store mutation occurs.
            transition = self.ledger.start_stage(
                self.bundle.expected_batch_id,
                video_id,
                f"reference_{stage}",
                fingerprint,
                allow_reprocess=True,
            )
        return transition.changed, fingerprint

    def _preflight_partial_text_state(
        self,
        stage: str,
        expected_by_video: Mapping[str, int],
    ) -> None:
        """Reject unverifiable live text state before a partial snapshot mutates it."""

        coverage = self.bundle.channel_coverage(stage)
        if bool(coverage.get("coverage_complete", True)):
            return
        if self.ledger is None:
            raise CloudArtifactError("cloud_partial_import_ledger_required")
        counter = getattr(self.text_stores[stage], "count_documents", None)
        if not callable(counter):
            raise CloudArtifactError("cloud_text_count_unavailable")
        try:
            actual = int(counter(filters=f'batch_id = "{self.bundle.expected_batch_id}"'))
        except Exception as error:
            raise CloudArtifactError("cloud_text_count_unavailable") from error
        records = {
            str(value["video_id"]): dict(value)
            for value in self.ledger.stage_records(
                self.bundle.expected_batch_id,
                f"reference_{stage}",
            )
        }
        if any(video_id not in expected_by_video for video_id in records):
            raise CloudArtifactError("cloud_text_existing_state_mismatch")
        completed: dict[str, str] = {}
        incomplete: dict[str, str] = {}
        for video_id, value in records.items():
            state = str(value.get("state") or "")
            fingerprint = str(value.get("fingerprint") or "")
            if state == "completed":
                completed[video_id] = fingerprint
            elif state == "failed":
                incomplete[video_id] = fingerprint
            elif state == "running":
                # A running stage may belong to another live importer.  It must
                # be recovered explicitly in the ledger before a resume can
                # mutate the same deterministic document IDs.
                raise CloudArtifactError("cloud_text_existing_state_running")
            else:
                raise CloudArtifactError("cloud_text_existing_state_mismatch")
        for video_id, fingerprint in {**completed, **incomplete}.items():
            expected = (
                f"cloud-import-{stage}-v3:{self.bundle.video_fingerprint(stage, video_id)[:24]}"
            )
            if fingerprint != expected:
                raise CloudArtifactError("cloud_text_existing_state_mismatch")

        incomplete_counts: dict[str, int] = {}
        expected_ids: dict[str, set[str]] = {video_id: set() for video_id in incomplete}
        if incomplete:
            document_iterator = getattr(self.text_stores[stage], "iter_documents", None)
            if not callable(document_iterator):
                raise CloudArtifactError("cloud_text_document_audit_unavailable")
            for document in self.bundle.iter_documents(stage):
                video_id = str(document.get("video_id") or "")
                if video_id not in expected_ids:
                    continue
                identifier = document.get("id")
                if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
                    raise CloudArtifactError("cloud_text_existing_state_mismatch")
                expected_ids[video_id].add(str(identifier))
            for video_id, identifiers in expected_ids.items():
                if len(identifiers) != int(expected_by_video[video_id]):
                    raise CloudArtifactError("cloud_text_existing_state_mismatch")
                filter_expression = (
                    f'batch_id = "{self.bundle.expected_batch_id}" AND video_id = "{video_id}"'
                )
                try:
                    video_count = int(counter(filters=filter_expression))
                    stored_documents = tuple(
                        document_iterator(filters=filter_expression, batch_size=1_000)
                    )
                except Exception as error:
                    raise CloudArtifactError("cloud_text_document_audit_unavailable") from error
                stored_ids: list[str] = []
                for document in stored_documents:
                    if not isinstance(document, Mapping):
                        raise CloudArtifactError("cloud_text_existing_state_mismatch")
                    identifier = document.get("external_id", document.get("id"))
                    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
                        raise CloudArtifactError("cloud_text_existing_state_mismatch")
                    stored_ids.append(str(identifier))
                if (
                    video_count != len(stored_ids)
                    or len(stored_ids) != len(set(stored_ids))
                    or not set(stored_ids).issubset(identifiers)
                ):
                    raise CloudArtifactError("cloud_text_existing_state_mismatch")
                incomplete_counts[video_id] = video_count

        expected_existing = sum(int(expected_by_video[value]) for value in completed)
        expected_existing += sum(incomplete_counts.values())
        if actual != expected_existing:
            raise CloudArtifactError("cloud_text_existing_state_mismatch")

    def _complete_stage(
        self,
        video_id: str,
        stage: str,
        fingerprint: str,
        count: int,
    ) -> None:
        if self.ledger is None:
            return
        kind = "asr_zero_transcript" if stage == "asr" and count == 0 else f"{stage}_records"
        self.ledger.complete_stage(
            self.bundle.expected_batch_id,
            video_id,
            f"reference_{stage}",
            artifacts=(
                ArtifactSpec(
                    kind,
                    f"cloud-artifact://{self.bundle.expected_batch_id}/{stage}/{video_id}",
                    fingerprint,
                    self.bundle.channel_fingerprint(stage),
                    int(count),
                ),
            ),
        )

    def _fail_stage(self, video_id: str, stage: str) -> None:
        if self.ledger is None:
            return
        self.ledger.fail_stage(
            self.bundle.expected_batch_id,
            video_id,
            f"reference_{stage}",
            "cloud_import_failed",
        )

    def _import_vectors(self, *, resume: bool) -> int:
        assert self.bundle.manifest is not None and self.bundle.vectors is not None
        rows = self.bundle.manifest["vector"]["rows"]
        self.vector_store.create_collection(recreate=False)
        total = 0
        position = 0
        while position < len(rows):
            video_id = str(rows[position]["video_id"])
            stop = position + 1
            while stop < len(rows) and rows[stop]["video_id"] == video_id:
                stop += 1
            source_uri = str(rows[position].get("video_uri", f"cloud-artifact://{video_id}"))
            self._register_video(video_id, source_uri)
            changed, fingerprint = self._start_stage(video_id, "siglip2", resume=resume)
            if not changed:
                total += stop - position
                position = stop
                continue
            try:
                for start in range(position, stop, self.batch_sizes["siglip2"]):
                    end = min(start + self.batch_sizes["siglip2"], stop)
                    payloads = [dict(value) for value in rows[start:end]]
                    for payload in payloads:
                        payload["model_fingerprint"] = self.bundle.manifest["vector"]["model"][
                            "fingerprint"
                        ]
                        payload["cloud_artifact_schema"] = CLOUD_ARTIFACT_SCHEMA_VERSION
                    identifiers = [
                        f"{row['batch_id']}:{row['video_id']}:{row['source_frame_idx']}"
                        for row in rows[start:end]
                    ]
                    self.vector_store.upsert(
                        identifiers,
                        self.bundle.vectors[start:end],
                        payloads,
                        batch_size=self.batch_sizes["siglip2"],
                    )
                counter = getattr(self.vector_store, "count", None)
                if callable(counter):
                    actual = int(
                        counter(
                            filters={
                                "batch_id": self.bundle.expected_batch_id,
                                "video_id": video_id,
                            }
                        )
                    )
                    if actual != stop - position:
                        raise CloudArtifactError("cloud_vector_external_count_mismatch")
                self._complete_stage(video_id, "siglip2", fingerprint, stop - position)
            except Exception:
                self._fail_stage(video_id, "siglip2")
                raise
            total += stop - position
            position = stop
        return total

    def _import_text(self, stage: str, *, resume: bool) -> int:
        assert self.bundle.manifest is not None
        store = self.text_stores[stage]
        expected_by_video = {
            str(key): int(value)
            for key, value in self.bundle.manifest["text"][stage]["videos"].items()
        }
        self._preflight_partial_text_state(stage, expected_by_video)
        store.create_index()
        iterator = self.bundle.iter_documents(stage)
        pending = next(iterator, None)
        total = 0
        for video_id in sorted(expected_by_video):
            documents: list[dict[str, Any]] = []
            while pending is not None and str(pending["video_id"]) == video_id:
                documents.append(pending)
                pending = next(iterator, None)
            if len(documents) != expected_by_video[video_id]:
                raise CloudArtifactError("cloud_text_video_count_mismatch")
            source_uri = f"cloud-artifact://{self.bundle.expected_batch_id}/{video_id}"
            self._register_video(video_id, source_uri)
            changed, fingerprint = self._start_stage(video_id, stage, resume=resume)
            if changed:
                try:
                    for start in range(0, len(documents), self.batch_sizes[stage]):
                        store.add_documents(
                            documents[start : start + self.batch_sizes[stage]],
                            batch_size=self.batch_sizes[stage],
                        )
                    self._complete_stage(video_id, stage, fingerprint, len(documents))
                except Exception:
                    self._fail_stage(video_id, stage)
                    raise
            total += len(documents)
        if pending is not None:
            raise CloudArtifactError("cloud_text_unexpected_video")
        return total

    def _validate_external_counts(
        self, counts: Mapping[str, int], selected: Sequence[str]
    ) -> dict[str, int]:
        """Read exact batch-scoped store counts after acknowledged imports."""

        actual: dict[str, int] = {}
        if "siglip2" in selected:
            counter = getattr(self.vector_store, "count", None)
            if not callable(counter):
                raise CloudArtifactError("cloud_vector_count_unavailable")
            actual["siglip2"] = int(counter(filters={"batch_id": self.bundle.expected_batch_id}))
        for stage in ("ocr", "asr"):
            if stage not in selected:
                continue
            counter = getattr(self.text_stores[stage], "count_documents", None)
            if not callable(counter):
                raise CloudArtifactError("cloud_text_count_unavailable")
            actual[stage] = int(counter(filters=f'batch_id = "{self.bundle.expected_batch_id}"'))
        for stage, value in actual.items():
            if value != int(counts[stage]):
                raise CloudArtifactError(f"cloud_{stage}_external_count_mismatch")
        return actual

    def run(
        self,
        *,
        channels: Sequence[str] = _STAGES,
        resume: bool = True,
        validate_only: bool = False,
    ) -> CloudImportReport:
        """Validate first, then import selected channels without destructive recreation."""

        selected = _normalized_channels(channels)
        started = self.clock()
        counts = self.bundle.validate(channels=selected)
        if self.ledger is not None and not validate_only:
            self.ledger.initialize()
            self.ledger.register_batch(
                self.bundle.expected_batch_id,
                "aic25",
                source_label=f"cloud-artifact://{self.bundle.manifest_sha256}",
            )
        imported = {"siglip2": 0, "ocr": 0, "asr": 0}
        if not validate_only:
            for stage in selected:
                imported[stage] = (
                    self._import_vectors(resume=resume)
                    if stage == "siglip2"
                    else self._import_text(stage, resume=resume)
                )
            external_counts = self._validate_external_counts(counts, selected)
            if self.channel_receipt_dir is not None:
                for stage in selected:
                    coverage = self.bundle.channel_coverage(stage)
                    _atomic_json(
                        self.channel_receipt_dir / f"{stage}.json",
                        {
                            "contract": "cloud-import-channel-v2",
                            "schema_version": CLOUD_ARTIFACT_SCHEMA_VERSION,
                            "batch_id": self.bundle.expected_batch_id,
                            "channel": stage,
                            "snapshot_import_complete": True,
                            "coverage_complete": bool(coverage.get("coverage_complete", True)),
                            "coverage": coverage,
                            "channel_fingerprint": self.bundle.channel_fingerprint(stage),
                            "manifest_sha256": self.bundle.manifest_sha256,
                            "manifest_path": str(self.bundle.manifest_path.resolve()),
                            "expected_count": int(counts[stage]),
                            "external_count": int(external_counts[stage]),
                        },
                    )
        else:
            for stage in selected:
                imported[stage] = counts[stage]
            external_counts = {}
        report = CloudImportReport(
            batch_id=self.bundle.expected_batch_id,
            channels=selected,
            vectors=imported["siglip2"],
            ocr_documents=imported["ocr"],
            asr_documents=imported["asr"],
            elapsed_ms=max(0.0, (self.clock() - started) * 1000.0),
            validate_only=validate_only,
        )
        if (
            self.completion_path is not None
            and not validate_only
            and selected == _STAGES
            and self.bundle.manifest is not None
            and self.bundle.manifest.get("complete") is True
        ):
            _atomic_json(
                self.completion_path,
                {
                    "contract": "cloud-import-completion-v2",
                    "schema_version": CLOUD_ARTIFACT_SCHEMA_VERSION,
                    "batch_id": self.bundle.expected_batch_id,
                    "complete": True,
                    "completed_channels": list(_STAGES),
                    "manifest_sha256": self.bundle.manifest_sha256,
                    "manifest_path": str(self.bundle.manifest_path.resolve()),
                    "expected_counts": {stage: int(counts[stage]) for stage in _STAGES},
                    "external_counts": external_counts,
                    **report.to_dict(),
                },
            )
        return report


__all__ = [
    "ASR_AUDIO_EXTRACTION_CONTRACT",
    "ASR_TRANSCRIPTS_NAME",
    "CLOUD_ARTIFACT_CONTRACT",
    "CLOUD_ARTIFACT_SCHEMA_VERSION",
    "CloudArtifactBundle",
    "CloudArtifactError",
    "CloudArtifactImporter",
    "CloudArtifactProducer",
    "CloudImportReport",
    "OCR_FEATURES_NAME",
    "SIGLIP_FEATURES_NAME",
    "SIGLIP_MANIFEST_NAME",
    "build_cloud_partition_manifest",
    "asr_profile_settings",
    "cloud_model_contracts",
    "load_cloud_partition",
    "sha256_file",
]
