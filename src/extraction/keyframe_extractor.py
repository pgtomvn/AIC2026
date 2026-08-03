"""Adaptive keyframe extraction with exact decoded-frame provenance."""

from __future__ import annotations

import hashlib
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"})


class KeyframeExtractor:
    """Select scene transitions plus anchors that enforce a maximum gap.

    PyAV is preferred because it exposes presentation timestamps and time bases.
    OpenCV remains a functional constant-frame-rate fallback for the lightweight
    environment.  In either case ``source_frame_idx`` is the zero-based decoded
    ordinal, never a JPEG/keyframe ordinal.
    """

    def __init__(
        self,
        fps_sample: int | None = None,
        min_scene_change: float = 0.30,
        resize_max_dim: int | None = 512,
        output_dir: str | Path = "data/keyframes/native",
        *,
        max_gap_seconds: float = 2.0,
        min_separation_seconds: float = 0.5,
        batch_id: str = "native",
        extractor_version: str = "adaptive-v2",
        prefer_pyav: bool = True,
        webp_quality: int = 85,
    ) -> None:
        if not 0 <= min_scene_change <= 1:
            # Preserve compatibility with the old 0..100 histogram threshold.
            if 1 < min_scene_change <= 100:
                min_scene_change /= 100
            else:
                raise ValueError("min_scene_change must be between 0 and 1")
        if max_gap_seconds <= 0 or min_separation_seconds < 0:
            raise ValueError("invalid temporal extraction thresholds")
        self.fps_sample = fps_sample
        self.min_scene_change = float(min_scene_change)
        self.resize_max_dim = resize_max_dim
        self.output_dir = Path(output_dir)
        self.max_gap_seconds = float(max_gap_seconds)
        self.min_separation_seconds = float(min_separation_seconds)
        self.batch_id = batch_id
        self.extractor_version = extractor_version
        self.prefer_pyav = prefer_pyav
        self.webp_quality = webp_quality

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _scene_signature(image: np.ndarray) -> np.ndarray:
        small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _scene_score(previous: np.ndarray | None, current: np.ndarray) -> float:
        if previous is None:
            return 1.0
        return float(np.mean(cv2.absdiff(previous, current)) / 255.0)

    def _decoded_pyav(
        self, video_path: Path
    ) -> Iterator[tuple[int, int | None, Fraction, int, Fraction, np.ndarray]]:
        try:
            import av
        except ImportError as error:
            raise RuntimeError("PyAV is not installed") from error
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            time_base = Fraction(stream.time_base)
            stream_rate = next(
                (
                    Fraction(value)
                    for value in (
                        getattr(stream, "average_rate", None),
                        getattr(stream, "guessed_rate", None),
                        getattr(stream, "base_rate", None),
                    )
                    if value
                ),
                None,
            )
            if stream_rate is None or stream_rate <= 0:
                raise ValueError(f"video reports invalid FPS: {video_path}")
            for source_frame_idx, frame in enumerate(container.decode(stream)):
                pts = int(frame.pts) if frame.pts is not None else None
                if pts is not None:
                    timestamp_ms = int(round(float(pts * time_base) * 1000))
                else:
                    timestamp_ms = int(round(source_frame_idx / float(stream_rate) * 1000))
                yield (
                    source_frame_idx,
                    pts,
                    time_base,
                    timestamp_ms,
                    stream_rate,
                    frame.to_ndarray(format="bgr24"),
                )

    def _decoded_opencv(
        self, video_path: Path
    ) -> Iterator[tuple[int, int, Fraction, int, Fraction, np.ndarray]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError(f"video reports invalid FPS: {video_path}")
        fps_fraction = Fraction(fps).limit_denominator(100_000)
        time_base = Fraction(fps_fraction.denominator, fps_fraction.numerator)
        try:
            source_frame_idx = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                timestamp_ms = int(round(source_frame_idx / fps * 1000))
                yield (
                    source_frame_idx,
                    source_frame_idx,
                    time_base,
                    timestamp_ms,
                    fps_fraction,
                    image,
                )
                source_frame_idx += 1
        finally:
            capture.release()

    def _decoded(
        self, video_path: Path
    ) -> Iterator[tuple[int, int | None, Fraction, int, Fraction, np.ndarray]]:
        if self.prefer_pyav:
            try:
                yield from self._decoded_pyav(video_path)
                return
            except RuntimeError as error:
                if "not installed" not in str(error):
                    raise
        yield from self._decoded_opencv(video_path)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if self.resize_max_dim is None:
            return image
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.resize_max_dim:
            return image
        scale = self.resize_max_dim / longest
        return cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _write_webp(self, image: np.ndarray, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(
            ".webp", self._resize(image), [cv2.IMWRITE_WEBP_QUALITY, self.webp_quality]
        )
        if not ok:
            raise OSError(f"failed to encode keyframe: {target}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded.tobytes())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def extract(
        self, video_path: str | Path, *, video_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Extract adaptive frames and return canonical provenance dictionaries."""

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved_video_id = video_id or path.stem
        if any(
            value in {"", ".", ".."} or any(separator in value for separator in ("/", "\\", ":"))
            for value in (self.batch_id, resolved_video_id)
        ):
            raise ValueError("batch_id and video_id must be safe path components")
        source_hash = self._sha256(path)
        previous_signature: np.ndarray | None = None
        last_selected_ms: int | None = None
        shot_id = 0
        records: list[dict[str, Any]] = []

        for source_idx, pts, time_base, timestamp_ms, fps, image in self._decoded(path):
            signature = self._scene_signature(image)
            scene_score = self._scene_score(previous_signature, signature)
            is_scene = previous_signature is not None and scene_score >= self.min_scene_change
            if is_scene:
                shot_id += 1
            reasons: list[str] = []
            if source_idx == 0:
                reasons.append("time_zero")
            elapsed = (
                float("inf")
                if last_selected_ms is None
                else (timestamp_ms - last_selected_ms) / 1000
            )
            if is_scene and elapsed >= self.min_separation_seconds:
                reasons.append("scene")
            if last_selected_ms is not None and elapsed >= self.max_gap_seconds:
                reasons.append("max_gap")
            if self.fps_sample and last_selected_ms is not None:
                if elapsed >= 1 / self.fps_sample:
                    reasons.append("uniform")

            if reasons:
                persisted_pts = pts
                if persisted_pts is None:
                    # The required integer PTS field remains explicitly marked
                    # as estimated so it can never be used as an exact VFR
                    # refinement anchor.
                    persisted_pts = round(Fraction(timestamp_ms, 1000) / time_base)
                    reasons.append("estimated_pts")
                target = (
                    self.output_dir / self.batch_id / resolved_video_id / f"{source_idx:09d}.webp"
                )
                self._write_webp(image, target)
                records.append(
                    {
                        "batch_id": self.batch_id,
                        "video_id": resolved_video_id,
                        "keyframe_ordinal": len(records) + 1,
                        "source_frame_idx": source_idx,
                        "pts": persisted_pts,
                        "time_base": f"{time_base.numerator}/{time_base.denominator}",
                        "timestamp_ms": timestamp_ms,
                        "fps_num": fps.numerator,
                        "fps_den": fps.denominator,
                        "shot_id": str(shot_id),
                        "sampling_reasons": sorted(set(reasons)),
                        "scene_score": scene_score,
                        "image_uri": str(target),
                        "video_uri": str(path),
                        "source_sha256": source_hash,
                        "extractor_version": self.extractor_version,
                    }
                )
                last_selected_ms = timestamp_ms
            previous_signature = signature
        if not records:
            raise ValueError(f"video contains no decodable frames: {path}")
        return records

    def extract_batch(self, video_dir: str | Path) -> list[dict[str, Any]]:
        """Extract every supported loose video below a directory."""

        root = Path(video_dir)
        if not root.is_dir():
            raise NotADirectoryError(root)
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.casefold() in _VIDEO_EXTENSIONS:
                records.extend(self.extract(path))
        return records
