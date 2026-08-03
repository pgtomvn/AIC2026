"""CPU-only tests for the resumable SQLite ingestion ledger."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest

from src.ingest.archive_catalog import inspect_archive
from src.ingest.ledger import (
    ArtifactSpec,
    JobLedger,
    LedgerError,
    StageFingerprintMismatch,
    StageState,
)


def _registered_ledger(tmp_path: Path, *, videos: tuple[str, ...] = ("L99_V001",)) -> JobLedger:
    ledger = JobLedger(tmp_path / "state" / "ingest.sqlite3")
    ledger.initialize()
    ledger.register_batch("b2", "aic26", now=1.0)
    for video_id in videos:
        ledger.register_video("b2", video_id, f"archive://{video_id}.mp4", now=2.0)
    return ledger


def test_constructor_is_side_effect_free_and_initialize_enables_wal(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "ledger.sqlite3"
    ledger = JobLedger(database)

    assert not database.parent.exists()

    ledger.initialize()
    with sqlite3.connect(database) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert mode == "wal"
    assert {
        "archives",
        "batches",
        "videos",
        "stages",
        "artifacts",
        "timings",
        "retrieval_generations",
    } <= tables


def test_read_only_batch_registration_probe_never_creates_storage(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "ledger.sqlite3"
    unopened = JobLedger(database)

    assert unopened.is_batch_registered("b2") is False
    assert not database.parent.exists()

    ledger = _registered_ledger(tmp_path)
    probe = JobLedger(ledger.path)
    assert probe.is_batch_registered("b2") is True
    assert probe.is_batch_registered("missing") is False


def test_retrieval_generation_is_monotonic_and_persists(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path)

    assert ledger.retrieval_generation("b2") == 0
    assert ledger.bump_retrieval_generation("b2", now=10.0) == 1
    assert ledger.bump_retrieval_generation("b2", now=11.0) == 2
    assert JobLedger(ledger.path).retrieval_generation("b2") == 2
    with pytest.raises(LedgerError, match="Unknown batch"):
        ledger.bump_retrieval_generation("missing")


def test_archive_ingestion_requires_verified_relocation_with_protected_exception(
    tmp_path: Path,
) -> None:
    ledger = JobLedger(tmp_path / "jobs.sqlite3")
    ledger.initialize()
    archive = tmp_path / "incoming.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("videos/L99_V001.mp4", b"video")
    inspection = inspect_archive(archive, verify_crc=True, compute_sha256=True)
    ledger.record_archive(inspection)

    with pytest.raises(LedgerError, match="canonical relocation"):
        ledger.assert_archive_ready(archive)

    ledger.record_archive(inspection, relocated_to=archive)
    assert ledger.assert_archive_ready(archive)["sha256"] == inspection.sha256

    protected = tmp_path / "Videos_L21_a.zip"
    with zipfile.ZipFile(protected, "w") as handle:
        handle.writestr("videos/L21_V001.mp4", b"protected")
    protected_inspection = inspect_archive(protected, verify_crc=True, compute_sha256=True)
    ledger.record_archive(protected_inspection)
    assert ledger.assert_archive_ready(protected)["state"] == "cataloged"


def test_stage_transitions_are_idempotent_and_retries_are_counted(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path)

    first = ledger.start_stage("b2", "L99_V001", "visual", "siglip:v1", now=10.0)
    duplicate = ledger.start_stage("b2", "L99_V001", "visual", "siglip:v1", now=11.0)
    failed = ledger.fail_stage("b2", "L99_V001", "visual", "CUDA OOM", now=14.0)
    duplicate_failure = ledger.fail_stage("b2", "L99_V001", "visual", "CUDA OOM", now=15.0)
    retry = ledger.start_stage("b2", "L99_V001", "visual", "siglip:v1", now=20.0)
    completed = ledger.complete_stage(
        "b2",
        "L99_V001",
        "visual",
        artifacts=(
            ArtifactSpec("thumbnail", "frames://L99_V001/1.webp", "siglip:v1", size_bytes=9),
            ArtifactSpec("vector", "qdrant://native/1", "siglip:v1"),
        ),
        now=26.0,
    )
    duplicate_completion = ledger.complete_stage("b2", "L99_V001", "visual", now=27.0)

    assert (first.changed, first.attempts) == (True, 1)
    assert (duplicate.changed, duplicate.attempts) == (False, 1)
    assert failed.changed is True
    assert duplicate_failure.changed is False
    assert (retry.changed, retry.attempts) == (True, 2)
    assert (completed.state, completed.attempts) == (StageState.COMPLETED, 2)
    assert duplicate_completion.changed is False


def test_completed_fingerprint_requires_explicit_reprocessing(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path)
    ledger.start_stage("b2", "L99_V001", "probe", "ffprobe:v1", now=1.0)
    ledger.complete_stage("b2", "L99_V001", "probe", now=2.0)

    with pytest.raises(StageFingerprintMismatch):
        ledger.start_stage("b2", "L99_V001", "probe", "ffprobe:v2", now=3.0)

    transition = ledger.start_stage(
        "b2",
        "L99_V001",
        "probe",
        "ffprobe:v2",
        now=3.0,
        allow_reprocess=True,
    )
    assert transition.changed is True
    assert transition.attempts == 2


def test_stage_completion_rolls_back_mismatched_artifact(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path)
    ledger.start_stage("b2", "L99_V001", "visual", "siglip:v1", now=1.0)

    with pytest.raises(StageFingerprintMismatch):
        ledger.complete_stage(
            "b2",
            "L99_V001",
            "visual",
            artifacts=(ArtifactSpec("vector", "qdrant://native/1", "other:v1"),),
            now=2.0,
        )

    stage = ledger.get_stage("b2", "L99_V001", "visual")
    assert stage is not None
    assert stage["state"] == "running"


def test_video_activation_checks_all_requirements_atomically(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path)
    ledger.start_stage("b2", "L99_V001", "visual", "siglip:v1", now=10.0)
    ledger.complete_stage(
        "b2",
        "L99_V001",
        "visual",
        artifacts=(ArtifactSpec("vector", "qdrant://native/1", "siglip:v1"),),
        now=14.0,
    )

    with pytest.raises(LedgerError, match="thumbnail"):
        ledger.activate_video(
            "b2",
            "L99_V001",
            required_stages=("visual",),
            required_artifacts={"vector": 1, "thumbnail": 1},
            now=15.0,
        )
    status = ledger.status_summary("b2")
    assert status.active_videos == 0

    ledger.start_stage("b2", "L99_V001", "frames", "extract:v1", now=16.0)
    ledger.complete_stage(
        "b2",
        "L99_V001",
        "frames",
        artifacts=(ArtifactSpec("thumbnail", "frames://L99_V001/1.webp", "extract:v1"),),
        now=18.0,
    )
    assert ledger.activate_video(
        "b2",
        "L99_V001",
        required_stages=("visual", "frames"),
        required_artifacts={"vector": 1, "thumbnail": 1},
        now=19.0,
    )
    assert not ledger.activate_video("b2", "L99_V001", now=20.0)
    assert ledger.activate_batch("b2", now=21.0)
    assert ledger.is_batch_active("b2")
    assert not ledger.activate_batch("b2", now=22.0)


def test_batch_activates_only_when_every_registered_video_is_ready(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path, videos=("L99_V001", "L99_V002"))
    ledger.activate_video("b2", "L99_V001")

    assert not ledger.activate_batch_if_ready("b2")
    assert not ledger.is_batch_active("b2")

    ledger.activate_video("b2", "L99_V002")
    assert ledger.activate_batch_if_ready("b2")
    assert ledger.is_batch_active("b2")


def test_registering_new_video_revokes_active_batch_until_ready(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path)
    ledger.activate_video("b2", "L99_V001")
    assert ledger.activate_batch_if_ready("b2")

    ledger.register_video("b2", "L99_V002", "fixture://L99_V002")

    assert not ledger.is_batch_active("b2")
    assert ledger.status_summary("b2").total_videos == 2
    ledger.activate_video("b2", "L99_V002")
    assert ledger.activate_batch_if_ready("b2")
    assert ledger.is_batch_active("b2")


def test_status_summary_survives_restart_and_estimates_remaining_work(tmp_path: Path) -> None:
    ledger = _registered_ledger(tmp_path, videos=("L99_V001", "L99_V002"))
    ledger.start_stage("b2", "L99_V001", "probe", "ffprobe:v1", now=10.0)
    ledger.complete_stage("b2", "L99_V001", "probe", now=14.0)
    ledger.activate_video("b2", "L99_V001", required_stages=("probe",), now=15.0)

    restarted = JobLedger(ledger.path)
    status = restarted.status_summary("b2")

    assert status.total_videos == 2
    assert status.active_videos == 1
    assert status.pending_videos == 1
    assert status.readiness == 0.5
    assert status.completed_stage_timings == 1
    assert status.average_seconds_per_active_video == 4.0
    assert status.eta_seconds == 4.0
    assert status.stage_counts == {"probe": {"completed": 1}}


def test_core_projection_waits_for_ten_videos_and_compares_preload_budget(
    tmp_path: Path,
) -> None:
    videos = tuple(f"L99_V{index:03d}" for index in range(1, 12))
    ledger = _registered_ledger(tmp_path, videos=videos)
    for index, video_id in enumerate(videos[:10]):
        started = 100.0 + index * 20.0
        for offset, stage in enumerate(("probe", "frames", "visual")):
            ledger.start_stage("b2", video_id, stage, f"{stage}:v1", now=started + offset * 2)
            ledger.complete_stage("b2", video_id, stage, now=started + offset * 2 + 2)
        ledger.activate_video("b2", video_id, now=started + 10)

    projection = ledger.core_ingestion_projection(
        "b2",
        total_videos=11,
        preload_window_seconds=80.0,
    )

    assert projection.sampled_videos == 10
    assert projection.projection_ready
    assert projection.average_core_seconds_per_video == 6.0
    assert projection.projected_core_seconds == 66.0
    assert projection.preload_budget_seconds == 64.0
    assert projection.fits_preload_budget is False
    assert projection.accelerated_profile_mandatory is True

    incomplete = ledger.core_ingestion_projection(
        "b2", total_videos=11, sample_videos=11, preload_window_seconds=100.0
    )
    assert incomplete.projection_ready is False
    assert incomplete.projected_core_seconds is None
    assert incomplete.accelerated_profile_mandatory is None
