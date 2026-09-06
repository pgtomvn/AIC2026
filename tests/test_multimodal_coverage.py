from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from src.readiness import (
    ChannelApplicability,
    ChannelReadiness,
    ChannelState,
    multimodal_coverage_snapshot,
)


def _config(tmp_path: Path, *, expected_videos: int = 1) -> dict:
    return {
        "data": {
            "catalog_db": str(tmp_path / "jobs.sqlite3"),
            "artifact_db": str(tmp_path / "artifacts.sqlite3"),
        },
        "readiness": {
            "batches": {
                "aic25-b1-reference": {
                    "expected_videos": expected_videos,
                    "expected_canonical_frames": expected_videos,
                    "expected_mapping_rows": expected_videos + 1,
                }
            }
        },
    }


def _channel(name: str, count: int) -> ChannelReadiness:
    return ChannelReadiness(
        name,
        ChannelApplicability.OPTIONAL,
        ChannelState.READY,
        count=count,
    )


def _databases(tmp_path: Path, *, completed_asr: bool = True) -> None:
    with sqlite3.connect(tmp_path / "artifacts.sqlite3") as connection:
        connection.execute("CREATE TABLE canonical_frames(batch_id TEXT, video_id TEXT)")
        connection.execute("INSERT INTO canonical_frames VALUES ('aic25-b1-reference', 'L21_V001')")
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        # Source catalog and stage progress are independent contracts.
        connection.execute("CREATE TABLE videos(batch_id TEXT, video_id TEXT, source_uri TEXT)")
        connection.execute(
            "INSERT INTO videos VALUES (?, ?, ?)",
            ("aic25-b1-reference", "L21_V001", "zip://videos/L21_V001.mp4"),
        )
        connection.execute(
            """
            CREATE TABLE stages(
                batch_id TEXT, video_id TEXT, stage_name TEXT,
                state TEXT, error TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE artifacts(
                batch_id TEXT, video_id TEXT, stage_name TEXT, kind TEXT
            )
            """
        )
        for stage in ("reference_ocr", "reference_siglip2"):
            connection.execute(
                "INSERT INTO stages VALUES (?, ?, ?, 'completed', NULL)",
                ("aic25-b1-reference", "L21_V001", stage),
            )
        if completed_asr:
            connection.execute(
                "INSERT INTO stages VALUES (?, ?, 'reference_asr', 'completed', NULL)",
                ("aic25-b1-reference", "L21_V001"),
            )
            connection.execute(
                "INSERT INTO artifacts VALUES (?, ?, 'reference_asr', 'asr_no_audio')",
                ("aic25-b1-reference", "L21_V001"),
            )


def _completion_receipt(tmp_path: Path, *, ocr: int, asr: int) -> None:
    manifest = tmp_path / "bundle" / "b01_siglip2_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "data/checkpoints/cloud-import/b01-import-complete.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "contract": "cloud-import-completion-v2",
                "batch_id": "aic25-b1-reference",
                "complete": True,
                "completed_channels": ["siglip2", "ocr", "asr"],
                "manifest_path": str(manifest),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "external_counts": {"siglip2": 1, "ocr": ocr, "asr": asr},
            }
        ),
        encoding="utf-8",
    )
def test_multimodal_health_requires_full_external_and_ledger_consistency(
    tmp_path: Path,
) -> None:
    _databases(tmp_path)
    _completion_receipt(tmp_path, ocr=3, asr=0)
    snapshot = multimodal_coverage_snapshot(
        _config(tmp_path),
        "aic25-b1-reference",
        (
            _channel("organizer_clip", 2),
            _channel("visual", 1),
            _channel("ocr", 3),
            _channel("asr", 0),
        ),
        project_root=tmp_path,
    )
    assert snapshot["status"] == "healthy"
    assert snapshot["ocr"]["scanned_canonical_frames"] == 1
    assert snapshot["asr"]["completed_videos"] == 1
    assert snapshot["asr"]["zero_transcript_videos"] == 1
    assert snapshot["asr"]["no_audio_videos"] == 1


def test_multimodal_health_accepts_new_zero_transcript_artifact_kind(
    tmp_path: Path,
) -> None:
    _databases(tmp_path)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute(
            "UPDATE artifacts SET kind = 'asr_zero_transcript' "
            "WHERE stage_name = 'reference_asr'"
        )
    _completion_receipt(tmp_path, ocr=3, asr=0)

    snapshot = multimodal_coverage_snapshot(
        _config(tmp_path),
        "aic25-b1-reference",
        (
            _channel("organizer_clip", 2),
            _channel("visual", 1),
            _channel("ocr", 3),
            _channel("asr", 0),
        ),
        project_root=tmp_path,
    )

    assert snapshot["asr"]["zero_transcript_videos"] == 1
    assert snapshot["asr"]["no_audio_videos"] == 1


def test_partial_asr_never_reports_healthy(tmp_path: Path) -> None:
    _databases(tmp_path, completed_asr=False)
    snapshot = multimodal_coverage_snapshot(
        _config(tmp_path),
        "aic25-b1-reference",
        (
            _channel("organizer_clip", 2),
            _channel("visual", 1),
            _channel("ocr", 1),
            _channel("asr", 0),
        ),
        project_root=tmp_path,
    )
    assert snapshot["status"] == "partial"
    assert snapshot["asr"]["missing_source_videos"] == 0
    assert snapshot["asr"]["consistent"] is False


def test_single_index_record_cannot_fake_multimodal_health(tmp_path: Path) -> None:
    snapshot = multimodal_coverage_snapshot(
        _config(tmp_path, expected_videos=873),
        "aic25-b1-reference",
        (
            _channel("organizer_clip", 1),
            _channel("visual", 1),
            _channel("ocr", 1),
            _channel("asr", 1),
        ),
        project_root=tmp_path,
    )
    assert snapshot["status"] != "healthy"
