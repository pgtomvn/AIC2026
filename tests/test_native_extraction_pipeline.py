"""CPU-only tests for adaptive native-video extraction and probing."""

from __future__ import annotations

import hashlib
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.extraction.media_probe as media_probe_module
from src.extraction.calibration import select_calibration, temporal_coverage
from src.extraction.keyframe_extractor import KeyframeExtractor


def _write_tiny_cut_video(path: Path, *, fps: float = 10.0) -> Path:
    """Write a deterministic 2.1-second MJPEG video with one hard cut."""

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot create a local MJPEG fixture")
    try:
        for index in range(21):
            level = 0 if index < 10 else 255
            writer.write(np.full((24, 32, 3), level, dtype=np.uint8))
    finally:
        writer.release()
    assert path.is_file() and path.stat().st_size > 0
    return path


@pytest.fixture
def tiny_cut_video(tmp_path: Path) -> Path:
    return _write_tiny_cut_video(tmp_path / "L99_V001.avi")


def test_temporal_coverage_and_smallest_qualifying_calibration() -> None:
    references = (0, 1_000, 2_000, 3_000)

    assert temporal_coverage(references, (3_000, 2_000, 1_000, 0), tolerance_ms=0) == 1.0
    assert temporal_coverage((), (), tolerance_ms=0) == 1.0
    assert temporal_coverage(references, (), tolerance_ms=1_000) == 0.0

    result = select_calibration(
        {
            (0.25, 2.0): (0, 1_000, 3_000),
            (0.30, 3.0): (0, 2_000, 3_000),
            (0.35, 4.0): references,
        },
        references,
        minimum_coverage=0.75,
        tolerance_ms=0,
    )

    # The two three-frame runs tie on count and coverage; the smaller max gap wins.
    assert (result.scene_threshold, result.max_gap_seconds) == (0.25, 2.0)
    assert result.coverage == 0.75
    assert result.frame_count == 3
    assert result.used_fallback is False


def test_calibration_uses_documented_fallback_when_grid_misses_target() -> None:
    result = select_calibration(
        {
            (0.25, 3.0): (0,),
            (0.30, 2.0): (10_000,),
        },
        (0, 1_000, 2_000),
        minimum_coverage=1.0,
        tolerance_ms=0,
    )

    assert (result.scene_threshold, result.max_gap_seconds) == (0.30, 2.0)
    assert result.coverage == 0.0
    assert result.frame_count == 1
    assert result.used_fallback is True


def test_probe_media_falls_back_to_opencv_without_ffprobe_or_pyav(
    tiny_cut_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_probe_module.shutil, "which", lambda _name: None)

    def no_pyav(_path: Path):
        raise ImportError("PyAV intentionally unavailable in this offline fixture")

    monkeypatch.setattr(media_probe_module, "_probe_pyav", no_pyav)

    probe = media_probe_module.probe_media(
        tiny_cut_video,
        ffprobe_path=tmp_path / "missing-ffprobe.exe",
    )

    assert probe.probe_backend == "opencv"
    assert probe.path == str(tiny_cut_video)
    assert probe.frame_count == 21
    assert probe.duration_ms == 2_100
    assert (probe.fps_num, probe.fps_den) == (10, 1)
    assert (probe.time_base_num, probe.time_base_den) == (1, 10)
    assert (probe.width, probe.height) == (32, 24)
    assert probe.has_audio is False
    assert probe.variable_frame_rate is False


def test_adaptive_extractor_preserves_frame_identity_timing_and_shots(
    tiny_cut_video: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generated" / "keyframes"
    extractor = KeyframeExtractor(
        output_dir=output_dir,
        batch_id="batch-test",
        min_scene_change=0.80,
        max_gap_seconds=0.50,
        min_separation_seconds=0.20,
        resize_max_dim=16,
        prefer_pyav=False,
        extractor_version="adaptive-test-v1",
    )

    # Merely configuring an extractor must not create generated-state directories.
    assert not output_dir.exists()

    records = extractor.extract(tiny_cut_video, video_id="L99_V001")

    assert [record["keyframe_ordinal"] for record in records] == [1, 2, 3, 4, 5]
    assert [record["source_frame_idx"] for record in records] == [0, 5, 10, 15, 20]
    assert [record["timestamp_ms"] for record in records] == [0, 500, 1_000, 1_500, 2_000]
    assert [record["pts"] for record in records] == [0, 5, 10, 15, 20]
    assert {(record["fps_num"], record["fps_den"]) for record in records} == {(10, 1)}
    assert {record["time_base"] for record in records} == {"1/10"}
    assert [record["shot_id"] for record in records] == ["0", "0", "1", "1", "1"]
    assert records[0]["sampling_reasons"] == ["time_zero"]
    assert records[1]["sampling_reasons"] == ["max_gap"]
    assert records[2]["sampling_reasons"] == ["max_gap", "scene"]
    assert records[3]["sampling_reasons"] == ["max_gap"]
    assert records[4]["sampling_reasons"] == ["max_gap"]

    expected_hash = hashlib.sha256(tiny_cut_video.read_bytes()).hexdigest()
    assert {record["source_sha256"] for record in records} == {expected_hash}
    assert {record["batch_id"] for record in records} == {"batch-test"}
    assert {record["video_id"] for record in records} == {"L99_V001"}
    assert {record["extractor_version"] for record in records} == {"adaptive-test-v1"}

    for record in records:
        image_path = Path(record["image_uri"])
        assert image_path.suffix == ".webp"
        assert image_path.stem == f"{record['source_frame_idx']:09d}"
        assert image_path.parent == output_dir / "batch-test" / "L99_V001"
        decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert max(decoded.shape[:2]) == 16


def test_pyav_decoder_preserves_stream_rate_instead_of_inverting_time_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFrame:
        pts = 90_000

        @staticmethod
        def to_ndarray(*, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeContainer:
        def __init__(self) -> None:
            self.stream = SimpleNamespace(
                time_base=Fraction(1, 90_000),
                average_rate=Fraction(30_000, 1001),
                guessed_rate=Fraction(30, 1),
                base_rate=Fraction(30, 1),
            )
            self.streams = SimpleNamespace(video=[self.stream])

        def __enter__(self) -> "FakeContainer":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def decode(self, stream: object):
            assert stream is self.stream
            yield FakeFrame()

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: FakeContainer()))
    media_path = tmp_path / "fixture.mp4"
    media_path.write_bytes(b"fake media opened by the PyAV fixture")
    extractor = KeyframeExtractor(output_dir=tmp_path / "frames", batch_id="b2")

    decoded = list(extractor._decoded_pyav(media_path))

    assert len(decoded) == 1
    source_idx, pts, time_base, timestamp_ms, fps, _ = decoded[0]
    assert (source_idx, pts, timestamp_ms) == (0, 90_000, 1_000)
    assert time_base == Fraction(1, 90_000)
    assert fps == Fraction(30_000, 1001)
    assert fps != Fraction(time_base.denominator, time_base.numerator)
    records = extractor.extract(media_path, video_id="L99_V001")
    assert (records[0]["fps_num"], records[0]["fps_den"]) == (30_000, 1001)
    assert records[0]["time_base"] == "1/90000"


def test_missing_pyav_pts_is_explicitly_estimated_and_versioned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFrame:
        pts = None

        @staticmethod
        def to_ndarray(*, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeContainer:
        def __init__(self) -> None:
            self.stream = SimpleNamespace(
                time_base=Fraction(1, 90_000),
                average_rate=Fraction(25, 1),
                guessed_rate=Fraction(25, 1),
                base_rate=Fraction(25, 1),
            )
            self.streams = SimpleNamespace(video=[self.stream])

        def __enter__(self) -> "FakeContainer":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def decode(self, _stream: object):
            yield FakeFrame()

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: FakeContainer()))
    media_path = tmp_path / "missing-pts.mp4"
    media_path.write_bytes(b"fixture")

    record = KeyframeExtractor(output_dir=tmp_path / "frames", batch_id="b2").extract(
        media_path, video_id="L99_V001"
    )[0]

    assert record["pts"] == 0
    assert "estimated_pts" in record["sampling_reasons"]
    assert record["extractor_version"] == "adaptive-v2"


def test_thumbnail_paths_are_isolated_by_batch(
    tiny_cut_video: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "keyframes"
    common = {
        "output_dir": output_dir,
        "min_scene_change": 1.0,
        "max_gap_seconds": 10.0,
        "prefer_pyav": False,
    }

    first = KeyframeExtractor(batch_id="batch-one", **common).extract(
        tiny_cut_video, video_id="L99_V001"
    )
    second = KeyframeExtractor(batch_id="batch-two", **common).extract(
        tiny_cut_video, video_id="L99_V001"
    )

    first_path = Path(first[0]["image_uri"])
    second_path = Path(second[0]["image_uri"])
    assert first_path != second_path
    assert first_path == output_dir / "batch-one" / "L99_V001" / "000000000.webp"
    assert second_path == output_dir / "batch-two" / "L99_V001" / "000000000.webp"
    assert first_path.is_file() and second_path.is_file()
