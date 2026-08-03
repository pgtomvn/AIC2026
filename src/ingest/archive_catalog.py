"""Safe discovery, inspection, and relocation of incoming ZIP archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

PROTECTED_ARCHIVE_NAMES = frozenset({"keyframes_l21.zip", "videos_l21_a.zip"})
_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"})
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_MAP_COLUMNS = frozenset({"n", "pts_time", "fps", "frame_idx"})
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_JSON_INSPECTION_LIMIT = 8 * 1024 * 1024


class ArchiveKind(StrEnum):
    """Content families found in an organizer archive."""

    VIDEOS = "videos"
    KEYFRAMES = "keyframes"
    FRAME_MAPS = "frame_maps"
    FEATURES = "features"
    MEDIA_INFO = "media_info"
    OBJECTS = "objects"
    UNKNOWN = "unknown"


class ArchiveValidationError(ValueError):
    """Raised when a candidate is not a valid, complete ZIP archive."""


class ProtectedArchiveError(PermissionError):
    """Raised when relocation targets an immutable source archive."""


@dataclass(frozen=True, slots=True)
class ArchiveObservation:
    """A point-in-time file observation used for stable-file checks."""

    size_bytes: int
    mtime_ns: int
    observed_at: float


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    """Validated inventory and content classification for one ZIP file."""

    path: Path
    kinds: tuple[ArchiveKind, ...]
    size_bytes: int
    mtime_ns: int
    member_count: int
    file_count: int
    uncompressed_bytes: int
    crc_ok: bool | None = None
    sha256: str | None = None

    @property
    def primary_kind(self) -> ArchiveKind:
        """Return the only detected kind, or ``UNKNOWN`` for mixed content."""

        return self.kinds[0] if len(self.kinds) == 1 else ArchiveKind.UNKNOWN


def observe_archive(path: str | Path, *, now: float | None = None) -> ArchiveObservation:
    """Capture the size and modification time of an archive candidate."""

    archive_path = Path(path)
    stat = archive_path.stat()
    return ArchiveObservation(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        observed_at=time.time() if now is None else now,
    )


def is_stable_archive(
    path: str | Path,
    *,
    min_age_seconds: float = 120.0,
    now: float | None = None,
    previous: ArchiveObservation | None = None,
) -> bool:
    """Return whether a completed ZIP has remained unchanged long enough.

    With a prior observation, both snapshots must match and the observation
    interval must meet ``min_age_seconds``.  Without one, modification-time age
    is used, which is suitable for polling newly arrived files.
    """

    if min_age_seconds < 0:
        raise ValueError("min_age_seconds must be non-negative")
    archive_path = Path(path)
    if archive_path.name.casefold().endswith(".crdownload"):
        return False
    if archive_path.suffix.casefold() != ".zip" or not archive_path.is_file():
        return False

    current_time = time.time() if now is None else now
    stat = archive_path.stat()
    if previous is not None:
        unchanged = stat.st_size == previous.size_bytes and stat.st_mtime_ns == previous.mtime_ns
        return unchanged and current_time - previous.observed_at >= min_age_seconds
    return current_time - stat.st_mtime >= min_age_seconds


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a streaming SHA-256 digest without loading the archive into RAM."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip_crc(path: str | Path) -> None:
    """Read every compressed member and raise on the first CRC failure."""

    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            corrupt_member = archive.testzip()
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ArchiveValidationError(f"Invalid ZIP archive {archive_path}: {exc}") from exc
    if corrupt_member is not None:
        raise ArchiveValidationError(f"CRC validation failed for {archive_path}: {corrupt_member}")


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or ".." in member.parts:
        raise ArchiveValidationError(f"Unsafe ZIP member path: {name}")
    if member.parts and re.match(r"^[A-Za-z]:$", member.parts[0]):
        raise ArchiveValidationError(f"Unsafe ZIP member path: {name}")
    return member


def _read_member_prefix(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    limit: int,
) -> bytes:
    with archive.open(info) as member:
        return member.read(limit)


def _classify_csv(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> ArchiveKind | None:
    raw = _read_member_prefix(archive, info, 64 * 1024)
    try:
        text = raw.decode("utf-8-sig", errors="strict")
        header = next(csv.reader(io.StringIO(text)), [])
    except (UnicodeDecodeError, csv.Error):
        return None
    columns = {column.strip().casefold() for column in header}
    return ArchiveKind.FRAME_MAPS if _MAP_COLUMNS.issubset(columns) else None


def _walk_json_keys(value: Any, *, remaining: list[int]) -> set[str]:
    keys: set[str] = set()
    if remaining[0] <= 0:
        return keys
    remaining[0] -= 1
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_walk_json_keys(child, remaining=remaining))
            if remaining[0] <= 0:
                break
    elif isinstance(value, list):
        for child in value[:20]:
            keys.update(_walk_json_keys(child, remaining=remaining))
            if remaining[0] <= 0:
                break
    return keys


def _classify_json(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> ArchiveKind | None:
    member_name = info.filename.casefold().replace("\\", "/")
    if info.file_size > _JSON_INSPECTION_LIMIT:
        if "object" in member_name or "detect" in member_name:
            return ArchiveKind.OBJECTS
        if "media" in member_name or "metadata" in member_name:
            return ArchiveKind.MEDIA_INFO
        return None

    try:
        payload = json.loads(_read_member_prefix(archive, info, info.file_size + 1))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    keys = _walk_json_keys(payload, remaining=[500])
    object_keys = {
        "detection_boxes",
        "detection_class_entities",
        "detection_class_ids",
        "detection_scores",
        "detections",
        "objects",
        "bbox",
    }
    media_keys = {
        "codec_name",
        "codec_type",
        "duration",
        "format",
        "format_name",
        "r_frame_rate",
        "streams",
        "time_base",
    }
    if keys & object_keys:
        return ArchiveKind.OBJECTS
    if len(keys & media_keys) >= 2 or "streams" in keys or "format" in keys:
        return ArchiveKind.MEDIA_INFO
    return None


def classify_archive(archive: zipfile.ZipFile) -> tuple[ArchiveKind, ...]:
    """Classify an open ZIP from member types and sampled member contents."""

    kinds: set[ArchiveKind] = set()
    for info in archive.infolist():
        member = _safe_member_path(info.filename)
        if info.is_dir():
            continue
        suffix = member.suffix.casefold()
        if suffix in _VIDEO_SUFFIXES:
            kinds.add(ArchiveKind.VIDEOS)
        elif suffix in _IMAGE_SUFFIXES:
            kinds.add(ArchiveKind.KEYFRAMES)
        elif suffix == ".npy":
            kinds.add(ArchiveKind.FEATURES)
        elif suffix == ".csv" and ArchiveKind.FRAME_MAPS not in kinds:
            kind = _classify_csv(archive, info)
            if kind is not None:
                kinds.add(kind)
        elif suffix == ".json":
            kind = _classify_json(archive, info)
            if kind is not None:
                kinds.add(kind)
    if not kinds:
        return (ArchiveKind.UNKNOWN,)
    return tuple(sorted(kinds, key=lambda item: item.value))


def inspect_archive(
    path: str | Path,
    *,
    verify_crc: bool = False,
    compute_sha256: bool = False,
) -> ArchiveInspection:
    """Inventory and classify a ZIP, with optional full CRC and hash passes."""

    archive_path = Path(path).resolve()
    try:
        stat = archive_path.stat()
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            kinds = classify_archive(archive)
            if verify_crc:
                corrupt_member = archive.testzip()
                if corrupt_member is not None:
                    raise ArchiveValidationError(
                        f"CRC validation failed for {archive_path}: {corrupt_member}"
                    )
    except ArchiveValidationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ArchiveValidationError(f"Invalid ZIP archive {archive_path}: {exc}") from exc

    files = [info for info in infos if not info.is_dir()]
    return ArchiveInspection(
        path=archive_path,
        kinds=kinds,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        member_count=len(infos),
        file_count=len(files),
        uncompressed_bytes=sum(info.file_size for info in files),
        crc_ok=True if verify_crc else None,
        sha256=sha256_file(archive_path) if compute_sha256 else None,
    )


def discover_archives(
    roots: Iterable[str | Path],
    *,
    min_age_seconds: float = 120.0,
    now: float | None = None,
    previous_observations: Mapping[Path, ArchiveObservation] | None = None,
    recursive: bool = True,
    verify_crc: bool = False,
    compute_sha256: bool = False,
) -> list[ArchiveInspection]:
    """Discover stable ZIPs below roots and inspect their actual contents."""

    candidates: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file():
            candidates.add(root)
            continue
        if not root.is_dir():
            continue
        iterator = root.rglob("*") if recursive else root.iterdir()
        candidates.update(path for path in iterator if path.is_file())

    observations = previous_observations or {}
    inspections: list[ArchiveInspection] = []
    for path in sorted(candidates, key=lambda item: str(item).casefold()):
        previous = observations.get(path) or observations.get(path.resolve())
        if not is_stable_archive(
            path,
            min_age_seconds=min_age_seconds,
            now=now,
            previous=previous,
        ):
            continue
        inspections.append(
            inspect_archive(path, verify_crc=verify_crc, compute_sha256=compute_sha256)
        )
    return inspections


def _archive_bucket(kinds: Iterable[ArchiveKind]) -> str:
    kinds_set = set(kinds)
    if kinds_set == {ArchiveKind.VIDEOS}:
        return "videos"
    if kinds_set == {ArchiveKind.KEYFRAMES}:
        return "keyframes"
    return "reference"


def _safe_directory_component(value: str, field_name: str) -> str:
    component = value.strip()
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or ":" in component
    ):
        raise ValueError(f"{field_name} must be one safe directory name")
    return component


def _same_volume(source: Path, destination_parent: Path) -> bool:
    source_drive = os.path.splitdrive(str(source))[0].casefold()
    destination_drive = os.path.splitdrive(str(destination_parent))[0].casefold()
    if source_drive or destination_drive:
        return source_drive == destination_drive
    return source.stat().st_dev == destination_parent.stat().st_dev


def relocate_archive(
    source: str | Path,
    archive_root: str | Path,
    *,
    competition: str,
    batch_id: str,
    kinds: Iterable[ArchiveKind],
    expected_sha256: str | None = None,
    protected_names: Iterable[str] = PROTECTED_ARCHIVE_NAMES,
) -> Path:
    """Atomically move a verified archive into its canonical same-volume folder.

    Existing destinations are never overwritten.  The protected development
    archives are rejected by basename regardless of case.
    """

    source_path = Path(source).resolve(strict=True)
    protected = {name.casefold() for name in protected_names}
    if source_path.name.casefold() in protected:
        raise ProtectedArchiveError(f"Protected archive cannot be relocated: {source_path.name}")
    if source_path.suffix.casefold() != ".zip" or not source_path.is_file():
        raise ArchiveValidationError(f"Not a ZIP file: {source_path}")
    competition_name = _safe_directory_component(competition, "competition")
    batch_name = _safe_directory_component(batch_id, "batch_id")

    if expected_sha256 is not None:
        normalized_digest = expected_sha256.casefold()
        if not _DIGEST_PATTERN.fullmatch(normalized_digest):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
        actual_digest = sha256_file(source_path)
        if actual_digest != normalized_digest:
            raise ArchiveValidationError(
                f"SHA-256 mismatch for {source_path}: expected {normalized_digest}, got {actual_digest}"
            )

    bucket = _archive_bucket(kinds)
    destination_parent = Path(archive_root).resolve() / competition_name / batch_name / bucket
    existing_parent = destination_parent
    while not existing_parent.exists() and existing_parent.parent != existing_parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir() or not _same_volume(source_path, existing_parent):
        raise OSError("Archive relocation must remain on the same filesystem volume")
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / source_path.name
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    # os.rename is atomic on one volume and, on Windows, refuses an existing target.
    # The explicit existence check preserves that no-overwrite contract on all hosts.
    os.rename(source_path, destination)
    return destination


def _manifest_path(path: str | Path, root: Path) -> str:
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(root.resolve())
    except ValueError:
        relative = candidate
    return relative.as_posix()


def update_checksum_manifest(
    manifest_path: str | Path,
    archive_path: str | Path,
    digest: str,
    *,
    previous_path: str | Path | None = None,
    root: str | Path | None = None,
) -> None:
    """Atomically add or update one GNU-style SHA-256 manifest entry."""

    normalized_digest = digest.casefold()
    if not _DIGEST_PATTERN.fullmatch(normalized_digest):
        raise ValueError("digest must be a 64-character hexadecimal SHA-256")
    manifest = Path(manifest_path)
    manifest_root = Path(root) if root is not None else manifest.parent.parent
    new_name = _manifest_path(archive_path, manifest_root)
    old_name = (
        _manifest_path(previous_path, manifest_root) if previous_path is not None else new_name
    )

    lines = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
    replacement = f"{normalized_digest}  {new_name}"
    updated: list[str] = []
    found = False
    for line in lines:
        match = re.match(r"^([0-9A-Fa-f]{64})\s+[* ]?(.+?)\s*$", line)
        if match and match.group(2).replace("\\", "/") in {old_name, new_name}:
            if not found:
                updated.append(replacement)
                found = True
            continue
        updated.append(line)
    if not found:
        updated.append(replacement)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(updated) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
