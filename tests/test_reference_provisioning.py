from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.contracts import FrameRecord
from src.extraction.rapid_ocr_engine import RapidOCREngine
from src.ingest.artifacts import ArtifactRepository
from src.ingest.checkpoint_mgr import CheckpointMirror
from src.ingest.ledger import JobLedger
from src.ingest.reference_provisioner import (
    REFERENCE_BATCH_ID,
    ReferenceProvisioner,
    normalize_reference_batch,
)
from src.ingest.resource_watchdog import ResourceBudgetExceeded, ResourceBudgetGuard


class _Store:
    index_name = "aic_ocr_v1"

    def __init__(self) -> None:
        self.documents: list[dict] = []

    def create_index(self) -> None:
        return None

    def add_documents(self, documents: list[dict], batch_size: int = 1000) -> None:
        assert batch_size == 500
        self.documents.extend(documents)


class _OCR:
    fingerprint = "ocr-fingerprint"
    model_name = "rapidocr-ppocrv6-small-multilingual"

    def __init__(self) -> None:
        self.calls = 0
        self.unloaded = False

    def extract_text(self, source: bytes) -> list[dict]:
        assert source == b"jpeg"
        self.calls += 1
        return [
            {
                "text_raw": "Dầu mazut",
                "text_nfc": "Dầu mazut",
                "text_accent_folded": "dau mazut",
                "confidence": 0.9,
                "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            }
        ]

    def unload(self) -> None:
        self.unloaded = True


def _frame(video_id: str = "L21_V001") -> FrameRecord:
    return FrameRecord(
        batch_id=REFERENCE_BATCH_ID,
        video_id=video_id,
        keyframe_ordinal=1,
        source_frame_idx=10,
        pts=10,
        time_base="1/25",
        timestamp_ms=400,
        fps_num=25,
        fps_den=1,
        shot_id="shot-1",
        sampling_reasons=("reference",),
        image_uri="zip://keyframes.zip!/L21_V001/001.jpg",
        video_uri="zip://videos.zip!/L21_V001.mp4",
        source_sha256="0" * 64,
        extractor_version="reference-v1",
    )


def _repositories(tmp_path: Path) -> tuple[JobLedger, ArtifactRepository]:
    ledger = JobLedger(tmp_path / "jobs.sqlite3")
    ledger.initialize()
    ledger.register_batch(REFERENCE_BATCH_ID, "aic2026")
    ledger.register_video(
        REFERENCE_BATCH_ID,
        "L21_V001",
        "zip://videos.zip!/L21_V001.mp4",
    )
    artifacts = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    artifacts.initialize()
    artifacts.upsert_frame(_frame(), fingerprint="frames-v1")
    return ledger, artifacts


def test_batch_alias_is_strict() -> None:
    assert normalize_reference_batch("B01") == REFERENCE_BATCH_ID
    assert normalize_reference_batch(REFERENCE_BATCH_ID) == REFERENCE_BATCH_ID
    with pytest.raises(ValueError, match="reference_batch_unsupported"):
        normalize_reference_batch("B02")


def test_ocr_stage_scans_and_persists_with_ledger_resume(tmp_path: Path) -> None:
    ledger, artifacts = _repositories(tmp_path)
    engine = _OCR()
    store = _Store()
    provisioner = ReferenceProvisioner(
        ledger=ledger,
        artifacts=artifacts,
        image_resolver=SimpleNamespace(
            resolve=lambda _frame: SimpleNamespace(data=b"jpeg"),
            close=lambda: None,
        ),
        resource_guard=ResourceBudgetGuard(
            rss_limit_bytes=1024,
            gpu_limit_bytes=1024,
            snapshot=lambda: {
                "process_rss_bytes": 1,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
            },
        ),
        checkpoint_mirror=CheckpointMirror(tmp_path / "checkpoints"),
        ocr_engine=engine,
        ocr_store=store,
        expected_videos=1,
        expected_frames=1,
        ocr_upsert_batch_size=500,
    )

    report = provisioner.run_stage("ocr")

    assert report.state == "complete"
    assert report.canonical_frames == 1
    assert report.records_written == 1
    assert engine.calls == 1
    assert engine.unloaded is True
    segment = artifacts.get_text_segment(
        REFERENCE_BATCH_ID,
        "L21_V001",
        "ocr:aic25-b1-reference:L21_V001:10:0",
    )
    assert segment is not None
    assert segment.source_frame_idx == 10
    assert store.documents[0]["text"] == "Dầu mazut"
    checkpoint = json.loads(
        (tmp_path / "checkpoints" / "b01_ocr_progress.json").read_text(encoding="utf-8")
    )
    assert checkpoint["state"] == "complete"
    assert "zip://" not in json.dumps(checkpoint)

    second = provisioner.run_stage("ocr")
    assert second.completed_videos == 1
    assert engine.calls == 1


def test_scoped_text_reprocess_does_not_touch_frames(tmp_path: Path) -> None:
    _ledger, artifacts = _repositories(tmp_path)
    assert artifacts.count_frames(REFERENCE_BATCH_ID) == 1
    artifacts.replace_text_segments_for_video(
        REFERENCE_BATCH_ID,
        "L21_V001",
        "ocr",
        (),
        fingerprint="ocr-v2",
    )
    assert artifacts.count_frames(REFERENCE_BATCH_ID) == 1
    assert artifacts.count_text_segments(REFERENCE_BATCH_ID, kind="ocr") == 0


def test_resource_guard_fails_at_configured_threshold() -> None:
    guard = ResourceBudgetGuard(
        rss_limit_bytes=100,
        gpu_limit_bytes=100,
        snapshot=lambda: {
            "process_rss_bytes": 100,
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
        },
    )
    with pytest.raises(ResourceBudgetExceeded) as caught:
        guard.check()
    assert caught.value.code == "provisioning_rss_budget_exceeded"


def test_rapidocr_filtering_rejects_low_confidence_and_malformed_rows(
    tmp_path: Path,
) -> None:
    engine = RapidOCREngine(
        model_dir=tmp_path,
        confidence_threshold=0.5,
        min_text_characters=2,
    )
    engine._model = lambda _image: SimpleNamespace(
        boxes=[
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[0, 0], [1, 0]],
        ],
        txts=["Dầu mazut", "x", "valid but weak", "broken"],
        scores=[0.91, 0.99, 0.49, 0.99],
    )

    rows = engine.extract_text(Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)))

    assert len(rows) == 1
    assert rows[0]["text_nfc"] == "Dầu mazut"
    assert rows[0]["text_accent_folded"] == "dau mazut"
    assert rows[0]["bbox"] == [0.0, 0.0, 10.0, 10.0]


def test_stage_aborts_and_marks_ledger_when_resource_budget_is_reached(
    tmp_path: Path,
) -> None:
    ledger, artifacts = _repositories(tmp_path)
    engine = _OCR()
    provisioner = ReferenceProvisioner(
        ledger=ledger,
        artifacts=artifacts,
        image_resolver=SimpleNamespace(
            resolve=lambda _frame: SimpleNamespace(data=b"jpeg"),
            close=lambda: None,
        ),
        resource_guard=ResourceBudgetGuard(
            rss_limit_bytes=1,
            gpu_limit_bytes=1024,
            snapshot=lambda: {
                "process_rss_bytes": 1,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
            },
        ),
        checkpoint_mirror=CheckpointMirror(tmp_path / "checkpoints"),
        ocr_engine=engine,
        ocr_store=_Store(),
        expected_videos=1,
        expected_frames=1,
        ocr_upsert_batch_size=500,
    )

    with pytest.raises(ResourceBudgetExceeded) as caught:
        provisioner.run_stage("ocr")

    assert caught.value.code == "provisioning_rss_budget_exceeded"
    stage = ledger.get_stage(
        REFERENCE_BATCH_ID,
        "L21_V001",
        "reference_ocr",
    )
    assert stage is not None
    assert stage["state"] == "failed"
    assert stage["error"] == "provisioning_rss_budget_exceeded"
    assert engine.unloaded is True
    assert not (tmp_path / "checkpoints" / "b01_ocr_progress.json").exists()


def test_asr_segment_midpoint_maps_to_nearest_canonical_frame(tmp_path: Path) -> None:
    ledger, artifacts = _repositories(tmp_path)
    artifacts.upsert_frame(
        replace(
            _frame(),
            keyframe_ordinal=2,
            source_frame_idx=60,
            pts=60,
            timestamp_ms=2_400,
        ),
        fingerprint="frames-v1",
    )
    provisioner = ReferenceProvisioner(
        ledger=ledger,
        artifacts=artifacts,
        image_resolver=SimpleNamespace(close=lambda: None),
        resource_guard=ResourceBudgetGuard(
            rss_limit_bytes=1024,
            gpu_limit_bytes=1024,
            snapshot=lambda: {},
        ),
        checkpoint_mirror=CheckpointMirror(tmp_path / "checkpoints"),
        asr_engine=SimpleNamespace(model_size="small"),
        expected_videos=1,
        expected_frames=2,
    )

    segments = provisioner._asr_segments(
        "L21_V001",
        {
            "segments": [
                {
                    "start_ms": 1_800,
                    "end_ms": 2_200,
                    "text_raw": "giá dầu mazut",
                    "text_nfc": "giá dầu mazut",
                    "text_accent_folded": "gia dau mazut",
                    "avg_logprob": -0.2,
                }
            ]
        },
    )

    assert len(segments) == 1
    assert segments[0].source_frame_idx == 60
    assert 0.0 < segments[0].confidence < 1.0
