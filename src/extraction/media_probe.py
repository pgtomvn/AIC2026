"""Read-only media probing with ffprobe, PyAV, and OpenCV fallbacks."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2


def _fraction(value: str | int | float | None) -> Fraction | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        parsed = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if parsed > 0 else None


@dataclass(frozen=True)
class MediaProbe:
    """Canonical media properties needed before an ingestion stage starts."""

    path: str
    duration_ms: int
    video_codec: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    time_base_num: int
    time_base_den: int
    variable_frame_rate: bool
    has_audio: bool
    audio_codec: str | None
    probe_backend: str
    frame_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_ffprobe(path: Path, executable: str) -> MediaProbe:
    command = [
        executable,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"media has no video stream: {path}")
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    average_rate = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    real_rate = _fraction(video.get("r_frame_rate"))
    if average_rate is None:
        raise ValueError(f"media reports invalid FPS: {path}")
    time_base = _fraction(video.get("time_base")) or Fraction(
        average_rate.denominator, average_rate.numerator
    )
    duration = video.get("duration", payload.get("format", {}).get("duration", 0))
    frame_count = video.get("nb_frames")
    return MediaProbe(
        path=str(path),
        duration_ms=round(float(duration or 0) * 1000),
        video_codec=str(video.get("codec_name", "unknown")),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps_num=average_rate.numerator,
        fps_den=average_rate.denominator,
        time_base_num=time_base.numerator,
        time_base_den=time_base.denominator,
        variable_frame_rate=real_rate is not None and real_rate != average_rate,
        has_audio=audio is not None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        probe_backend="ffprobe",
        frame_count=int(frame_count) if frame_count not in (None, "N/A") else None,
    )


def _probe_pyav(path: Path) -> MediaProbe:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"media has no video stream: {path}")
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate) if stream.average_rate else None
        if not rate:
            raise ValueError(f"media reports invalid FPS: {path}")
        time_base = Fraction(stream.time_base)
        duration = stream.duration * time_base if stream.duration is not None else 0
        audio = container.streams.audio[0] if container.streams.audio else None
        guessed_rate = Fraction(stream.guessed_rate) if stream.guessed_rate else rate
        return MediaProbe(
            path=str(path),
            duration_ms=round(float(duration) * 1000),
            video_codec=str(stream.codec_context.name or "unknown"),
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            fps_num=rate.numerator,
            fps_den=rate.denominator,
            time_base_num=time_base.numerator,
            time_base_den=time_base.denominator,
            variable_frame_rate=guessed_rate != rate,
            has_audio=audio is not None,
            audio_codec=str(audio.codec_context.name) if audio else None,
            probe_backend="pyav",
            frame_count=int(stream.frames) if stream.frames else None,
        )


def _probe_opencv(path: Path) -> MediaProbe:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError(f"media reports invalid FPS: {path}")
        rate = Fraction(fps).limit_denominator(100_000)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return MediaProbe(
        path=str(path),
        duration_ms=round(frame_count / fps * 1000),
        video_codec="unknown",
        width=width,
        height=height,
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        time_base_num=rate.denominator,
        time_base_den=rate.numerator,
        variable_frame_rate=False,
        has_audio=False,
        audio_codec=None,
        probe_backend="opencv",
        frame_count=frame_count,
    )


def probe_media(path: str | Path, *, ffprobe_path: str | Path | None = None) -> MediaProbe:
    """Probe without writing files, preferring a project-local ffprobe binary."""

    media_path = Path(path)
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    configured = str(ffprobe_path) if ffprobe_path is not None else None
    executable = (
        configured if configured and Path(configured).is_file() else shutil.which("ffprobe")
    )
    if executable:
        return _probe_ffprobe(media_path, executable)
    try:
        return _probe_pyav(media_path)
    except ImportError:
        return _probe_opencv(media_path)
