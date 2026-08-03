"""Project-local external tool installation and verification helpers."""

from src.tooling.ffmpeg import (
    FFmpegInstallation,
    install_ffmpeg_archive,
    resolve_ffmpeg_executable,
    verify_ffmpeg_manifest,
)

__all__ = [
    "FFmpegInstallation",
    "install_ffmpeg_archive",
    "resolve_ffmpeg_executable",
    "verify_ffmpeg_manifest",
]
