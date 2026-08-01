"""Safely inspect or extract the project data archives.

Inspection is the default. Extraction must be explicitly limited to one sample
or enabled for the complete corpus::

    python scripts/extract_data.py
    python scripts/extract_data.py --sample L21_V001
    python scripts/extract_data.py --full

Archive members are validated before the destination directory is created.
This module deliberately does not use :meth:`zipfile.ZipFile.extractall`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

# Allow the script to be run directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import load_config
from src.utils.logger import logger, setup_logger_from_config

_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_MAX_ARCHIVE_FILES = 100_000
_MAX_UNCOMPRESSED_BYTES = 100 * 1024**3
_MAX_COMPRESSION_RATIO = 200


class UnsafeArchiveError(ValueError):
    """Raised when a ZIP member cannot be extracted safely and portably."""


@dataclass(frozen=True)
class ArchiveSpec:
    """An archive and its normalized extraction destination."""

    archive_path: Path
    output_dir: Path
    archive_root: str


@dataclass(frozen=True)
class ArchiveReport:
    """Read-only inventory returned by :func:`inspect_archive`."""

    archive_path: Path
    file_count: int
    selected_file_count: int
    uncompressed_bytes: int
    selected_uncompressed_bytes: int


@dataclass(frozen=True)
class ExtractionReport:
    """Outcome of an idempotent archive extraction."""

    archive_path: Path
    selected_file_count: int
    extracted_file_count: int
    skipped_file_count: int
    bytes_written: int


@dataclass(frozen=True)
class _PlannedMember:
    info: zipfile.ZipInfo
    relative_path: PurePosixPath


def _validate_sample_id(sample: str | None) -> None:
    if sample is None:
        return
    if not _SAMPLE_ID.fullmatch(sample) or sample in {".", ".."}:
        raise ValueError(
            "sample must be a single identifier containing only letters, "
            "digits, dots, underscores, or hyphens"
        )


def _validate_component(component: str, member_name: str) -> None:
    if component in {"", ".", ".."}:
        raise UnsafeArchiveError(
            f"Unsafe path component in ZIP member: {member_name!r}"
        )
    if any(character in _WINDOWS_INVALID_CHARS for character in component):
        raise UnsafeArchiveError(f"Non-portable ZIP member path: {member_name!r}")
    if any(ord(character) < 32 for character in component):
        raise UnsafeArchiveError(f"Control character in ZIP member: {member_name!r}")
    if component.endswith((" ", ".")):
        raise UnsafeArchiveError(f"Non-portable ZIP member path: {member_name!r}")
    if component.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise UnsafeArchiveError(f"Reserved path in ZIP member: {member_name!r}")


def _normalized_member_path(
    info: zipfile.ZipInfo,
    archive_root: str,
) -> PurePosixPath | None:
    """Validate a member name and remove a redundant archive root."""
    name = info.filename
    if not name or "\x00" in name or "\\" in name:
        raise UnsafeArchiveError(f"Unsafe ZIP member name: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise UnsafeArchiveError(f"Absolute ZIP member path: {name!r}")

    trimmed_name = name[:-1] if name.endswith("/") else name
    if not trimmed_name:
        raise UnsafeArchiveError(f"Unsafe ZIP member name: {name!r}")

    parts = trimmed_name.split("/")
    for component in parts:
        _validate_component(component, name)

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        raise UnsafeArchiveError(f"Symbolic-link ZIP member is not allowed: {name!r}")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise UnsafeArchiveError(f"Special-file ZIP member is not allowed: {name!r}")
    if info.flag_bits & 0x1:
        raise UnsafeArchiveError(f"Encrypted ZIP member is not supported: {name!r}")

    if parts[0].casefold() != archive_root.casefold():
        raise UnsafeArchiveError(
            f"ZIP member is outside the expected {archive_root!r} root: {name!r}"
        )
    parts = parts[1:]
    if not parts:
        return None
    return PurePosixPath(*parts)


def _matches_sample(path: PurePosixPath, sample: str | None) -> bool:
    if sample is None:
        return True
    expected = sample.casefold()
    return any(part.casefold() == expected for part in path.parts[:-1]) or (
        path.stem.casefold() == expected
    )


def _plan_archive(
    archive: zipfile.ZipFile,
    *,
    archive_root: str,
    sample: str | None,
) -> tuple[list[_PlannedMember], list[_PlannedMember]]:
    """Validate every member, then return all files and the selected files."""
    _validate_sample_id(sample)
    files: list[_PlannedMember] = []
    selected: list[_PlannedMember] = []
    destinations: dict[str, str] = {}

    for info in archive.infolist():
        relative_path = _normalized_member_path(info, archive_root)
        if relative_path is None or info.is_dir():
            continue

        destination_key = relative_path.as_posix().casefold()
        previous = destinations.get(destination_key)
        if previous is not None:
            raise UnsafeArchiveError(
                "ZIP members normalize to the same destination: "
                f"{previous!r} and {info.filename!r}"
            )
        destinations[destination_key] = info.filename

        planned = _PlannedMember(info=info, relative_path=relative_path)
        files.append(planned)
        if len(files) > _MAX_ARCHIVE_FILES:
            raise UnsafeArchiveError(
                f"Archive contains more than {_MAX_ARCHIVE_FILES:,} files"
            )
        if info.file_size > 0 and info.compress_size == 0:
            raise UnsafeArchiveError(
                f"ZIP member has an invalid compressed size: {info.filename!r}"
            )
        if (
            info.file_size > 1024**2
            and info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO
        ):
            raise UnsafeArchiveError(
                f"ZIP member exceeds the compression-ratio limit: {info.filename!r}"
            )
        if _matches_sample(relative_path, sample):
            selected.append(planned)

    # Reject file/directory prefix conflicts such as both ``a`` and ``a/b``.
    destination_keys = set(destinations)
    for destination_key, member_name in destinations.items():
        parts = destination_key.split("/")
        for end in range(1, len(parts)):
            if "/".join(parts[:end]) in destination_keys:
                raise UnsafeArchiveError(
                    f"ZIP member conflicts with a parent file: {member_name!r}"
                )

    total_size = sum(member.info.file_size for member in files)
    if total_size > _MAX_UNCOMPRESSED_BYTES:
        raise UnsafeArchiveError(
            "Archive exceeds the 100 GiB uncompressed-size safety limit"
        )

    return files, selected


def verify_archive(archive_path: str | Path) -> None:
    """Read every member and fail if the ZIP CRC or stream is corrupt."""
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise zipfile.BadZipFile(
            f"CRC validation failed for {bad_member!r} in {archive_path}"
        )


def inspect_archive(
    archive_path: str | Path,
    *,
    archive_root: str,
    sample: str | None = None,
) -> ArchiveReport:
    """Validate and inventory an archive without writing to the filesystem."""
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        files, selected = _plan_archive(
            archive,
            archive_root=archive_root,
            sample=sample,
        )
    return ArchiveReport(
        archive_path=archive_path,
        file_count=len(files),
        selected_file_count=len(selected),
        uncompressed_bytes=sum(member.info.file_size for member in files),
        selected_uncompressed_bytes=sum(member.info.file_size for member in selected),
    )


def _destination_for(output_root: Path, member: _PlannedMember) -> Path:
    target = output_root
    for component in member.relative_path.parts:
        target /= component
        if target.is_symlink():
            raise UnsafeArchiveError(
                f"ZIP destination traverses a symbolic link: {target}"
            )

    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(output_root)
    except ValueError as error:
        raise UnsafeArchiveError(
            f"ZIP member escapes the output directory: {member.info.filename!r}"
        ) from error
    return target


def _preflight_destinations(
    output_dir: Path,
    selected: list[_PlannedMember],
) -> tuple[Path, list[tuple[_PlannedMember, Path]]]:
    current = output_dir
    while True:
        if current.is_symlink():
            raise UnsafeArchiveError(
                f"Extraction destination traverses a symbolic link: {current}"
            )
        if current.exists():
            attributes = getattr(os.lstat(current), "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if attributes & reparse_flag:
                raise UnsafeArchiveError(
                    f"Extraction destination traverses a reparse point: {current}"
                )
        if current == current.parent:
            break
        current = current.parent

    output_root = output_dir.resolve()
    planned_targets: list[tuple[_PlannedMember, Path]] = []

    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(
            f"Extraction destination is not a directory: {output_dir}"
        )

    for member in selected:
        target = _destination_for(output_root, member)
        if target.exists() and target.is_dir():
            raise FileExistsError(f"File destination is already a directory: {target}")

        parent = target.parent
        while parent != output_root:
            if parent.exists() and not parent.is_dir():
                raise FileExistsError(f"Extraction parent is not a directory: {parent}")
            parent = parent.parent
        planned_targets.append((member, target))

    return output_root, planned_targets


def _space_requirement(
    output_root: Path,
    planned_targets: list[tuple[_PlannedMember, Path]],
    *,
    force: bool,
) -> tuple[int, int, int]:
    """Return destination device, free bytes, and bytes needed for staging."""
    required_bytes = sum(
        member.info.file_size
        for member, target in planned_targets
        if force or not target.exists()
    )
    disk_probe = output_root
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    return os.stat(disk_probe).st_dev, shutil.disk_usage(disk_probe).free, required_bytes


def extract_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    archive_root: str,
    sample: str | None = None,
    force: bool = False,
    verify_integrity: bool = True,
) -> ExtractionReport:
    """Safely extract selected files, skipping existing files by default.

    Every archive member and destination is validated before any output is
    created. A repeated call is therefore idempotent unless ``force`` is true.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    with zipfile.ZipFile(archive_path, "r") as archive:
        _, selected = _plan_archive(
            archive,
            archive_root=archive_root,
            sample=sample,
        )
        if verify_integrity:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise zipfile.BadZipFile(
                    f"CRC validation failed for {bad_member!r} in {archive_path}"
                )
        output_root, planned_targets = _preflight_destinations(output_dir, selected)

        _, free_bytes, required_bytes = _space_requirement(
            output_root,
            planned_targets,
            force=force,
        )
        if required_bytes > free_bytes:
            raise OSError(
                f"Extraction requires {required_bytes} bytes but only "
                f"{free_bytes} bytes are free on the destination drive"
            )

        extracted = 0
        skipped = 0
        bytes_written = 0
        for member, target in planned_targets:
            if target.exists() and not force:
                skipped += 1
                continue

            output_root.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    dir=target.parent,
                )
                temporary_path = Path(temporary_name)
                with os.fdopen(descriptor, "wb") as destination:
                    with archive.open(member.info, "r") as source:
                        shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())

                if temporary_path.stat().st_size != member.info.file_size:
                    raise OSError(
                        f"Extracted size mismatch for {member.info.filename!r}"
                    )

                if force:
                    os.replace(temporary_path, target)
                else:
                    # A hard link provides an atomic create-if-absent operation on
                    # both NTFS and POSIX filesystems. Removing the temporary name
                    # leaves the completed file at its final destination.
                    os.link(temporary_path, target)
                    temporary_path.unlink()
            except FileExistsError:
                # Preserve no-overwrite behavior if another process wins a race.
                skipped += 1
                continue
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
            extracted += 1
            bytes_written += member.info.file_size

    return ExtractionReport(
        archive_path=archive_path,
        selected_file_count=len(selected),
        extracted_file_count=extracted,
        skipped_file_count=skipped,
        bytes_written=bytes_written,
    )


def _archive_specs(config: dict) -> tuple[ArchiveSpec, ...]:
    data_root = (PROJECT_ROOT / "data").resolve()

    def safe_output(configured_path: str) -> Path:
        configured = Path(configured_path)
        if (
            configured.is_absolute()
            or PureWindowsPath(configured_path).is_absolute()
            or ".." in configured.parts
        ):
            raise ValueError(
                f"Configured data path must be project-relative: {configured_path!r}"
            )
        candidate = PROJECT_ROOT / configured
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(data_root)
        except ValueError as error:
            raise ValueError(
                f"Configured data path escapes {data_root}: {configured_path!r}"
            ) from error
        if not relative.parts:
            raise ValueError("An extraction destination cannot be the data root itself")

        current = candidate
        while True:
            if current.is_symlink():
                raise ValueError(f"Configured data path traverses a link: {current}")
            if current.exists():
                attributes = getattr(os.lstat(current), "st_file_attributes", 0)
                reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                if attributes & reparse_flag:
                    raise ValueError(
                        f"Configured data path traverses a reparse point: {current}"
                    )
            if current == PROJECT_ROOT:
                break
            if current == current.parent:
                raise ValueError(
                    f"Configured data path is not below the project: {configured_path!r}"
                )
            current = current.parent
        return candidate

    return (
        ArchiveSpec(
            archive_path=PROJECT_ROOT / "Keyframes_L21.zip",
            output_dir=safe_output(config["data"]["raw_keyframes"]),
            archive_root="keyframes",
        ),
        ArchiveSpec(
            archive_path=PROJECT_ROOT / "Videos_L21_a.zip",
            output_dir=safe_output(config["data"]["raw_videos"]),
            archive_root="video",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect data archives by default; extraction requires an explicit mode."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--inspect",
        action="store_true",
        help="validate and inventory archives without extracting (default)",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="inspect and fully CRC-read both archives without extracting",
    )
    mode.add_argument(
        "--sample",
        metavar="SAMPLE_ID",
        help="extract only one sample, for example L21_V001",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="extract the complete corpus",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing files (valid only with --sample or --full)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only inventory or an explicitly requested extraction."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.force and not (args.sample or args.full):
        parser.error("--force requires --sample or --full")

    config = load_config()
    setup_logger_from_config(config)
    specs = _archive_specs(config)
    missing = [spec.archive_path for spec in specs if not spec.archive_path.is_file()]
    if missing:
        for archive_path in missing:
            logger.error("Archive not found: {}", archive_path)
        return 1

    # Validate and inventory every archive before performing the first write.
    reports = [
        inspect_archive(
            spec.archive_path,
            archive_root=spec.archive_root,
            sample=args.sample,
        )
        for spec in specs
    ]
    if args.sample:
        missing_sample = [
            spec.archive_path.name
            for spec, report in zip(specs, reports, strict=True)
            if report.selected_file_count == 0
        ]
        if missing_sample:
            logger.error(
                "Sample {} is missing from required archives: {}",
                args.sample,
                ", ".join(missing_sample),
            )
            return 2

    if not (args.sample or args.full):
        if args.verify:
            for spec in specs:
                verify_archive(spec.archive_path)
        for spec, report in zip(specs, reports, strict=True):
            logger.info(
                "{}: {} files, {} uncompressed bytes; destination {}",
                spec.archive_path.name,
                report.file_count,
                report.uncompressed_bytes,
                spec.output_dir,
            )
        return 0

    # Preflight every destination and aggregate required space by filesystem
    # before committing a file from either member of the paired dataset.
    volume_requirements: dict[int, tuple[int, int]] = {}
    try:
        for spec in specs:
            with zipfile.ZipFile(spec.archive_path, "r") as archive:
                _, selected = _plan_archive(
                    archive,
                    archive_root=spec.archive_root,
                    sample=args.sample,
                )
            output_root, targets = _preflight_destinations(spec.output_dir, selected)
            device, free_bytes, required_bytes = _space_requirement(
                output_root,
                targets,
                force=args.force,
            )
            previous_required, previous_free = volume_requirements.get(
                device,
                (0, free_bytes),
            )
            volume_requirements[device] = (
                previous_required + required_bytes,
                min(previous_free, free_bytes),
            )

        for required_bytes, free_bytes in volume_requirements.values():
            if required_bytes > free_bytes:
                raise OSError(
                    f"Paired extraction requires {required_bytes} bytes but only "
                    f"{free_bytes} bytes are free on the destination filesystem"
                )

        for spec in specs:
            verify_archive(spec.archive_path)
    except (OSError, UnsafeArchiveError, zipfile.BadZipFile) as error:
        logger.error("Extraction preflight failed: {}", error)
        return 3

    for spec in specs:
        report = extract_archive(
            spec.archive_path,
            spec.output_dir,
            archive_root=spec.archive_root,
            sample=args.sample,
            force=args.force,
            verify_integrity=False,
        )
        logger.info(
            "{}: selected {}, extracted {}, skipped {}",
            spec.archive_path.name,
            report.selected_file_count,
            report.extracted_file_count,
            report.skipped_file_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
