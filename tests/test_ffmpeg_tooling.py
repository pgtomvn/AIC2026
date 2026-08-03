"""CPU/offline checks for the checksum-addressed FFmpeg installer."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from src.tooling.ffmpeg import (
    install_ffmpeg_archive,
    resolve_ffmpeg_executable,
    verify_ffmpeg_manifest,
)


def _archive(path: Path, members: dict[str, bytes]) -> tuple[Path, str]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_installer_extracts_only_required_tools_and_is_idempotent(tmp_path: Path) -> None:
    archive, digest = _archive(
        tmp_path / "ffmpeg.zip",
        {
            "ffmpeg-build/bin/ffmpeg.exe": b"ffmpeg-binary",
            "ffmpeg-build/bin/ffprobe.exe": b"ffprobe-binary",
            "ffmpeg-build/doc/readme.txt": b"must not be extracted",
        },
    )
    root = tmp_path / ".tools" / "ffmpeg"

    first = install_ffmpeg_archive(archive, digest, install_root=root)
    second = install_ffmpeg_archive(archive, digest, install_root=root)

    assert first.version == digest[:16]
    assert first.ffmpeg.read_bytes() == b"ffmpeg-binary"
    assert first.ffprobe.read_bytes() == b"ffprobe-binary"
    assert first.install_dir == root / digest[:16]
    assert first.active_manifest == root / "active.json"
    assert second == verify_ffmpeg_manifest(root / "active.json")
    assert resolve_ffmpeg_executable(root / "active.json", "ffprobe") == first.ffprobe
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert not any("readme" in value for value in files)


def test_installer_rejects_checksum_mismatch_without_creating_root(tmp_path: Path) -> None:
    archive, _ = _archive(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"a", "bin/ffprobe.exe": b"b"},
    )
    root = tmp_path / "tools"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install_ffmpeg_archive(archive, "0" * 64, install_root=root)

    assert not root.exists()


@pytest.mark.parametrize(
    ("members", "message"),
    [
        (
            {"../ffmpeg.exe": b"a", "bin/ffprobe.exe": b"b"},
            "unsafe FFmpeg archive member",
        ),
        (
            {
                "one/ffmpeg.exe": b"a",
                "two/FFMPEG.EXE": b"b",
                "bin/ffprobe.exe": b"c",
            },
            "multiple ffmpeg executables",
        ),
        ({"bin/ffmpeg.exe": b"a"}, "missing executables"),
    ],
)
def test_installer_rejects_unsafe_duplicate_or_incomplete_archives(
    tmp_path: Path,
    members: dict[str, bytes],
    message: str,
) -> None:
    archive, digest = _archive(tmp_path / "bad.zip", members)

    with pytest.raises(ValueError, match=message):
        install_ffmpeg_archive(archive, digest, install_root=tmp_path / "tools")


def test_manifest_verification_detects_executable_tampering(tmp_path: Path) -> None:
    archive, digest = _archive(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"a", "bin/ffprobe.exe": b"b"},
    )
    installed = install_ffmpeg_archive(archive, digest, install_root=tmp_path / "tools")
    installed.ffmpeg.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_ffmpeg_manifest(installed.active_manifest)
