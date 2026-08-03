"""Checksum-gated installation of project-local FFmpeg executables."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_REQUIRED_TOOLS = ("ffmpeg", "ffprobe")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name.replace("\\", "/"))
    if (
        member.is_absolute()
        or not member.parts
        or ".." in member.parts
        or re.fullmatch(r"[A-Za-z]:", member.parts[0])
    ):
        raise ValueError(f"unsafe FFmpeg archive member: {name!r}")
    return member


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class FFmpegInstallation:
    """Verified paths for one immutable, checksum-addressed tool installation."""

    version: str
    archive_sha256: str
    install_dir: Path
    ffmpeg: Path
    ffprobe: Path
    manifest: Path
    active_manifest: Path


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid FFmpeg tool manifest: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"unsupported FFmpeg tool manifest: {path}")
    return value


def _resolve_manifest_entry(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"invalid executable entry in {manifest_path}")
    relative = _safe_member(value["path"])
    target = (manifest_path.parent / Path(*relative.parts)).resolve()
    root = manifest_path.parent.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"executable escapes manifest directory: {value['path']!r}")
    expected = str(value.get("sha256", "")).casefold()
    if not _SHA256_PATTERN.fullmatch(expected):
        raise ValueError(f"invalid executable checksum in {manifest_path}")
    if not target.is_file():
        raise FileNotFoundError(f"FFmpeg executable is missing: {target}")
    actual = _sha256(target)
    if actual != expected:
        raise ValueError(f"FFmpeg executable checksum mismatch: {target}")
    return target


def verify_ffmpeg_manifest(manifest_path: str | Path) -> FFmpegInstallation:
    """Verify a version or active manifest and every referenced executable."""

    path = Path(manifest_path).resolve()
    value = _read_manifest(path)
    archive_sha256 = str(value.get("archive_sha256", "")).casefold()
    if not _SHA256_PATTERN.fullmatch(archive_sha256):
        raise ValueError(f"invalid archive checksum in {path}")
    version = str(value.get("version", ""))
    if version != archive_sha256[:16]:
        raise ValueError(f"manifest version does not match archive checksum: {path}")
    executables = value.get("executables")
    if not isinstance(executables, dict) or set(executables) != set(_REQUIRED_TOOLS):
        raise ValueError(f"manifest must contain exactly {_REQUIRED_TOOLS}: {path}")
    ffmpeg = _resolve_manifest_entry(path, executables["ffmpeg"])
    ffprobe = _resolve_manifest_entry(path, executables["ffprobe"])
    install_dir = ffmpeg.parent.parent
    version_manifest = install_dir / "tool-manifest.json"
    active_manifest = path if path.name == "active.json" else install_dir.parent / "active.json"
    return FFmpegInstallation(
        version=version,
        archive_sha256=archive_sha256,
        install_dir=install_dir,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        manifest=version_manifest,
        active_manifest=active_manifest,
    )


def resolve_ffmpeg_executable(manifest_path: str | Path, tool: str) -> Path:
    """Resolve and checksum-verify ``ffmpeg`` or ``ffprobe`` from a manifest."""

    if tool not in _REQUIRED_TOOLS:
        raise ValueError(f"unsupported FFmpeg tool: {tool!r}")
    installation = verify_ffmpeg_manifest(manifest_path)
    return getattr(installation, tool)


def _find_executable_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    seen_members: set[str] = set()
    selected: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        member = _safe_member(info.filename)
        normalized = "/".join(part.casefold() for part in member.parts)
        if normalized in seen_members:
            raise ValueError(f"duplicate FFmpeg archive member: {info.filename!r}")
        seen_members.add(normalized)
        if info.is_dir():
            continue
        filename = member.name.casefold()
        for tool in _REQUIRED_TOOLS:
            accepted = {tool, f"{tool}.exe"}
            if filename in accepted:
                if tool in selected:
                    raise ValueError(f"multiple {tool} executables in FFmpeg archive")
                selected[tool] = info
    missing = sorted(set(_REQUIRED_TOOLS) - set(selected))
    if missing:
        raise ValueError(f"FFmpeg archive is missing executables: {', '.join(missing)}")
    return selected


def install_ffmpeg_archive(
    archive_path: str | Path,
    expected_sha256: str,
    *,
    install_root: str | Path = ".tools/ffmpeg",
) -> FFmpegInstallation:
    """Install only FFmpeg executables after complete ZIP and SHA-256 validation.

    Installations are immutable and addressed by the first 16 checksum characters.
    The version directory is published atomically, followed by an atomic active
    manifest update. Existing matching installations are verified, not rewritten.
    """

    archive_path = Path(archive_path).resolve()
    install_root = Path(install_root).resolve()
    expected = expected_sha256.strip().casefold()
    if not _SHA256_PATTERN.fullmatch(expected):
        raise ValueError("expected SHA-256 must be exactly 64 hexadecimal characters")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    actual = _sha256(archive_path)
    if actual != expected:
        raise ValueError(f"FFmpeg archive SHA-256 mismatch: expected {expected}, got {actual}")

    version = expected[:16]
    target = install_root / version
    version_manifest = target / "tool-manifest.json"
    active_manifest = install_root / "active.json"
    if target.exists():
        existing = verify_ffmpeg_manifest(version_manifest)
        if existing.archive_sha256 != expected:
            raise ValueError(f"existing FFmpeg version directory conflicts with {expected}")
        active_value = _manifest_value(existing, relative_to=install_root)
        _atomic_json(active_manifest, active_value)
        return verify_ffmpeg_manifest(active_manifest)

    install_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=install_root))
    try:
        bin_dir = temporary / "bin"
        bin_dir.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("FFmpeg archive failed CRC validation")
            members = _find_executable_members(archive)
            executable_paths: dict[str, Path] = {}
            for tool, info in members.items():
                suffix = ".exe" if info.filename.casefold().endswith(".exe") else ""
                output = bin_dir / f"{tool}{suffix}"
                with archive.open(info) as source, output.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                    destination.flush()
                    os.fsync(destination.fileno())
                output.chmod(0o755)
                executable_paths[tool] = output
        staged = FFmpegInstallation(
            version=version,
            archive_sha256=expected,
            install_dir=temporary,
            ffmpeg=executable_paths["ffmpeg"],
            ffprobe=executable_paths["ffprobe"],
            manifest=temporary / "tool-manifest.json",
            active_manifest=active_manifest,
        )
        _atomic_json(staged.manifest, _manifest_value(staged, relative_to=temporary))
        verify_ffmpeg_manifest(staged.manifest)
        try:
            os.replace(temporary, target)
        except OSError:
            if not target.exists():
                raise
            verify_ffmpeg_manifest(version_manifest)
        installed = verify_ffmpeg_manifest(version_manifest)
        _atomic_json(active_manifest, _manifest_value(installed, relative_to=install_root))
        return verify_ffmpeg_manifest(active_manifest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _manifest_value(installation: FFmpegInstallation, *, relative_to: Path) -> dict[str, Any]:
    def entry(path: Path) -> dict[str, str]:
        return {
            "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
            "sha256": _sha256(path),
        }

    return {
        "schema_version": 1,
        "version": installation.version,
        "archive_sha256": installation.archive_sha256,
        "executables": {
            "ffmpeg": entry(installation.ffmpeg),
            "ffprobe": entry(installation.ffprobe),
        },
    }
