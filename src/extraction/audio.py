"""Explicit FFmpeg boundary for normalized ASR audio extraction."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class AudioExtractor:
    """Extract 16 kHz mono PCM audio without performing constructor I/O."""

    sample_rate = 16_000
    channels = 1
    sample_width_bytes = 2

    def __init__(
        self,
        ffmpeg_path: str | Path,
        *,
        tool_fingerprint: str = "unversioned",
        timeout_seconds: float = 3600,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.ffmpeg_path = Path(ffmpeg_path)
        self.tool_fingerprint = tool_fingerprint
        self.timeout_seconds = timeout_seconds

    @property
    def fingerprint(self) -> str:
        """Identify the tool and normalized-audio preprocessing contract."""

        return f"ffmpeg:{self.tool_fingerprint}:pcm_s16le:{self.sample_rate}:mono"

    @classmethod
    def estimated_size_bytes(cls, duration_ms: int) -> int:
        """Return a conservative PCM WAV scratch-size estimate."""

        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        pcm_bytes = (duration_ms * cls.sample_rate * cls.channels * cls.sample_width_bytes) // 1000
        return pcm_bytes + 1024 * 1024

    def extract_16k_mono(
        self,
        media_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Extract the first audio stream and atomically publish a normalized WAV."""

        media = Path(media_path)
        output = Path(output_path)
        if not self.ffmpeg_path.is_file():
            raise FileNotFoundError(f"configured ffmpeg executable is missing: {self.ffmpeg_path}")
        if not media.is_file():
            raise FileNotFoundError(media)
        if output.exists() and not overwrite:
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.stem}.", suffix=".wav", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        command = [
            str(self.ffmpeg_path),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
                raise RuntimeError(f"FFmpeg audio extraction failed: {detail}")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("FFmpeg audio extraction produced no audio")
            os.replace(temporary, output)
            return output
        finally:
            temporary.unlink(missing_ok=True)
