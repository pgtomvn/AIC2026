"""CPU-only contracts for the independent pre-keyframe ASR staging worker."""

import gzip
import json
import wave
import zipfile

import pytest

from scripts.colab_video_asr import (
    WorkerLease,
    execution_settings,
    freeze_plan,
    inventory,
    run_worker,
    transcript_documents,
)
from src.ingest.cloud_artifacts import CloudArtifactProducer


def archive(path, entries):
    with zipfile.ZipFile(path, "w") as handle:
        for name, content in entries.items():
            handle.writestr(name, content)


def test_inventory_no_maps_stable_balanced_and_excludes_only_known_empty(tmp_path):
    archive(tmp_path / "Videos_M01.zip", {"M01/M01_V001.mp4": b"123", "M01_V002.mp4": b"45"})
    (tmp_path / "Videos_M010.zip").touch()
    plan = inventory(tmp_path, "aic26-b2", 2, ("Videos_M010.zip",))
    assert plan == inventory(tmp_path, "aic26-b2", 2, ("Videos_M010.zip",))
    assert plan["expected_videos"] == 2
    assert {row["worker_id"] for row in plan["videos"]} == {0, 1}
    freeze_plan(tmp_path / "plan.json", plan)
    freeze_plan(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="Frozen plan"):
        freeze_plan(tmp_path / "plan.json", dict(plan, worker_count=1))
    (tmp_path / "Videos_M010.zip").write_bytes(b"not empty now")
    with pytest.raises(ValueError, match="now non-empty"):
        inventory(tmp_path, "aic26-b2", 2, ("Videos_M010.zip",))


@pytest.mark.parametrize("names,match", [
    (["a/M01_V001.mp4", "b/M01_V001.mp4"], "Duplicate"),
    (["../M01_V001.mp4"], "Unsafe"),
])
def test_rejects_duplicate_and_unsafe_members(tmp_path, names, match):
    archive(tmp_path / "video.zip", dict.fromkeys(names, b"video"))
    with pytest.raises(ValueError, match=match):
        inventory(tmp_path, "aic26-b2", 2)


def test_no_silent_skip_of_truncated_zip(tmp_path):
    (tmp_path / "video.zip").write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        inventory(tmp_path, "aic26-b2", 1)


def test_worker_resume_and_tamper_fail_closed(tmp_path, monkeypatch):
    raw, out, scratch = tmp_path / "raw", tmp_path / "out", tmp_path / "scratch"
    raw.mkdir()
    archive(raw / "videos.zip", {"M01_V001.mp4": b"video"})
    plan = inventory(raw, "aic26-b2", 1)
    calls = []

    def decode(handle, member, audio, **kwargs):
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 16000)
        return True

    class Engine:
        model_size = "turbo"
        fingerprint = "fixed"
        profile_name = "production"
        model_revision = "revision"

        def transcribe(self, audio):
            calls.append(audio)
            return {"segments": [{"start_ms": 400, "end_ms": 950, "text_raw": "hello",
                                  "text_nfc": "hello", "text_accent_folded": "hello"}]}

    monkeypatch.setattr(CloudArtifactProducer, "_stream_audio_member", decode)
    report = run_worker(raw, out, scratch, plan, 0, Engine(), "a" * 64)
    assert report["state"] == "worker_complete"
    assert report["documents"] == 1 and not report["import_ready"]
    with gzip.open(out / "shards/M01_V001.jsonl.gz", "rt") as stream:
        doc = json.loads(stream.readline())
        assert doc["source_frame_idx"] is None and doc["start_ms"] == 400
    again = run_worker(raw, out, scratch, plan, 0, Engine(), "a" * 64)
    assert again["resumed_videos"] == 1 and len(calls) == 1
    with pytest.raises(ValueError, match="Checkpoint mismatch"):
        run_worker(raw, out, scratch, plan, 0, Engine(), "b" * 64)
    assert not (out / "leases/worker-0.lock").exists()


@pytest.mark.parametrize("audio_present,reason", [(False, "no_audio_stream"),
                                                  (True, "no_speech_detected")])
def test_zero_transcript_reasons(tmp_path, monkeypatch, audio_present, reason):
    raw = tmp_path / "raw"
    raw.mkdir()
    archive(raw / "videos.zip", {"N001.mp4": b"video"})
    plan = inventory(raw, "aic26-b2", 1)
    def decode(handle, member, audio, **kwargs):
        if audio_present:
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 16000)
        return audio_present

    monkeypatch.setattr(CloudArtifactProducer, "_stream_audio_member", decode)

    class Engine:
        fingerprint = "fixed"

        def transcribe(self, path):
            assert audio_present
            return {"segments": []}

    out = tmp_path / "out"
    run_worker(raw, out, tmp_path / "scratch", plan, 0, Engine(), "a" * 64)
    receipt = json.loads((out / "receipts/N001.json").read_text())
    assert receipt["zero_transcript_reason"] == reason
    assert receipt["audio_stream_present"] is audio_present


def test_lease_prevents_duplicate_worker(tmp_path):
    lock = tmp_path / "worker.lock"
    with WorkerLease(lock):
        with pytest.raises(RuntimeError, match="already leased"):
            with WorkerLease(lock):
                pytest.fail("duplicate worker")
    assert not lock.exists()


def test_invalid_timestamp_is_not_committed():
    with pytest.raises(ValueError):
        transcript_documents("batch", "video", "model", [{
            "start_ms": 1000, "end_ms": 50, "text_raw": "hello",
            "text_nfc": "hello", "text_accent_folded": "hello",
        }])


def test_cpu_keeps_model_but_has_explicit_quantization():
    config = {"models": {"whisper": {"hub_model_id": "turbo", "revision": "pinned"}}}
    cpu = execution_settings(config, "cpu")
    gpu = execution_settings(config, "cuda")
    assert cpu["hub_model_id"] == gpu["hub_model_id"] == "turbo"
    assert cpu["revision"] == gpu["revision"] == "pinned"
    assert cpu["compute_type"] == "int8"
    assert gpu["compute_type"] == "int8_float16"
    assert "device" not in config["models"]["whisper"]
    with pytest.raises(ValueError):
        execution_settings(config, "auto")
