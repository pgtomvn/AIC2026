"""CPU/offline tests for normalized, bounded ASR audio staging."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.extraction.audio as audio_module
from src.contracts import FrameRecord
from src.extraction.audio import AudioExtractor
from src.extraction.media_probe import MediaProbe
from src.ingest.native_pipeline import NativeIngestor, PreparedVideo, ScratchManager, VideoSource


def test_audio_extractor_publishes_atomic_16khz_mono_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"fixture")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fixture video")
    output = tmp_path / "audio" / "normalized.wav"
    observed: list[str] = []

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        observed.extend(command)
        Path(command[-1]).write_bytes(b"RIFF-fixture")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    extractor = AudioExtractor(executable)

    assert extractor.extract_16k_mono(media, output) == output
    assert output.read_bytes() == b"RIFF-fixture"
    assert observed[observed.index("-ac") + 1] == "1"
    assert observed[observed.index("-ar") + 1] == "16000"
    assert observed[observed.index("-c:a") + 1] == "pcm_s16le"
    assert not list(output.parent.glob(".*.wav"))


def test_audio_extractor_removes_partial_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"fixture")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fixture video")
    output = tmp_path / "normalized.wav"

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        Path(command[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stdout="", stderr="bad audio stream")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="bad audio stream"):
        AudioExtractor(executable).extract_16k_mono(media, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".*.wav"))


class _FakeAudioExtractor:
    sample_rate = 16_000
    fingerprint = "fixture-audio"

    def __init__(self) -> None:
        self.calls = 0
        self.audio_path: Path | None = None

    @staticmethod
    def estimated_size_bytes(duration_ms: int) -> int:
        return duration_ms * 32 + 128

    def extract_16k_mono(self, media: Path, output: Path) -> Path:
        assert media.is_file()
        self.calls += 1
        output.write_bytes(b"normalized audio")
        self.audio_path = output
        return output


class _FakeASR:
    model_size = "fixture-asr"

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def transcribe(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        assert path.is_file()
        self.paths.append(path)
        return {
            "segments": [
                {
                    "start_ms": 100,
                    "end_ms": 300,
                    "avg_logprob": -0.1,
                    "text_raw": "xin chào",
                    "text_nfc": "xin chào",
                    "text_accent_folded": "xin chao",
                }
            ]
        }


class _Artifacts:
    @staticmethod
    def nearest_frame(*_: Any) -> None:
        return None


def test_native_asr_extracts_audio_once_and_cleans_bounded_workspace(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    scratch = ScratchManager(tmp_path / "scratch", limit_bytes=100_000, reserve_bytes=0)
    audio = _FakeAudioExtractor()
    asr = _FakeASR()
    ingestor = NativeIngestor(
        batch_id="b2",
        competition="aic26",
        ledger=object(),
        artifacts=_Artifacts(),
        extractor=object(),
        visual_encoder=object(),
        visual_store=object(),
        scratch=scratch,
        asr_engine=asr,  # type: ignore[arg-type]
        audio_extractor=audio,  # type: ignore[arg-type]
    )
    frame = FrameRecord(
        batch_id="b2",
        video_id="L99_V001",
        keyframe_ordinal=1,
        source_frame_idx=0,
        pts=0,
        time_base="1/25",
        timestamp_ms=0,
        fps_num=25,
        fps_den=1,
        shot_id="0",
        sampling_reasons=("time_zero",),
        image_uri="frame.webp",
        video_uri=str(video),
        source_sha256="0" * 64,
        extractor_version="fixture",
    )
    probe = MediaProbe(
        path=str(video),
        duration_ms=1000,
        video_codec="fixture",
        width=16,
        height=16,
        fps_num=25,
        fps_den=1,
        time_base_num=1,
        time_base_den=25,
        variable_frame_rate=False,
        has_audio=True,
        audio_codec="pcm",
        probe_backend="fixture",
        frame_count=25,
    )

    segments = ingestor._asr_segments(
        PreparedVideo(VideoSource("L99_V001", video), probe, (frame,))
    )

    assert audio.calls == 1
    assert asr.paths == [audio.audio_path]
    assert segments[0].text_nfc == "xin chào"
    assert not any(path.is_file() for path in scratch.root.rglob("*"))


def test_scratch_workspace_rejects_estimate_over_limit_without_creating_root(
    tmp_path: Path,
) -> None:
    scratch = ScratchManager(tmp_path / "scratch", limit_bytes=10, reserve_bytes=0)

    with pytest.raises(OSError, match="scratch limit"):
        with scratch.workspace("video", "asr", required_bytes=11):
            pytest.fail("over-budget workspace must not be yielded")

    assert not scratch.root.exists()
