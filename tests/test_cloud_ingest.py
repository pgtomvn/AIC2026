from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.build_colab_source_bundle import build_colab_source_bundle
from src.embedding.siglip_encoder import SigLIP2Encoder
from src.ingest.cloud_artifacts import (
    ASR_AUDIO_EXTRACTION_CONTRACT,
    ASR_TRANSCRIPTS_NAME,
    OCR_FEATURES_NAME,
    SIGLIP_FEATURES_NAME,
    SIGLIP_MANIFEST_NAME,
    CloudArtifactBundle,
    CloudArtifactError,
    CloudArtifactImporter,
    CloudArtifactProducer,
    build_cloud_partition_manifest,
    load_cloud_partition,
)
from src.ingest.ledger import JobLedger
from src.readiness import (
    ChannelApplicability,
    ChannelReadiness,
    ChannelState,
    multimodal_coverage_snapshot,
)
from src.warmup import safe_resource_snapshot

BATCH_ID = "aic25-b1-reference"


def _jpeg(color: tuple[int, int, int]) -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(payload, format="JPEG")
    return payload.getvalue()


def _zip_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    maps = tmp_path / "maps.zip"
    keyframes = tmp_path / "keyframes.zip"
    videos = tmp_path / "videos.zip"
    with zipfile.ZipFile(maps, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "map-keyframes/L21_V001.csv",
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n2,0.04,25.0,0\n3,1.0,25.0,25\n",
        )
    with zipfile.ZipFile(keyframes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("keyframes/L21_V001/001.jpg", _jpeg((255, 0, 0)))
        archive.writestr("keyframes/L21_V001/002.jpg", _jpeg((0, 255, 0)))
        archive.writestr("keyframes/L21_V001/003.jpg", _jpeg((0, 0, 255)))
    with zipfile.ZipFile(videos, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("video/L21_V001.mp4", b"synthetic-video-stream")
    return maps, keyframes, videos


def _zip_two_video_corpus(tmp_path: Path) -> tuple[Path, Path]:
    maps = tmp_path / "maps-two.zip"
    keyframes = tmp_path / "keyframes-two.zip"
    with zipfile.ZipFile(maps, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for ordinal in (1, 2):
            video_id = f"L21_V{ordinal:03d}"
            archive.writestr(
                f"map-keyframes/{video_id}.csv",
                "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n",
            )
    with zipfile.ZipFile(keyframes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("keyframes/L21_V001/001.jpg", _jpeg((255, 0, 0)))
        archive.writestr("keyframes/L21_V002/001.jpg", _jpeg((0, 0, 255)))
    return maps, keyframes


class _Encoder:
    model_name = "google/siglip2-base-patch16-224"
    revision = "revision"
    fingerprint = "siglip-fingerprint"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode_images(self, images: list[bytes]) -> np.ndarray:
        self.batch_sizes.append(len(images))
        result = np.zeros((len(images), 4), dtype=np.float32)
        result[:, 0] = 1.0
        return result


class _OCR:
    model_name = "rapidocr-ppocrv6-small-multilingual"
    fingerprint = "rapidocr-fingerprint"
    confidence_threshold = 0.5

    def __init__(self) -> None:
        self.calls = 0

    def extract_text(self, image: bytes) -> list[dict]:
        assert image.startswith(b"\xff\xd8")
        self.calls += 1
        polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        return [
            {
                "text_raw": "loại bỏ",
                "text_nfc": "loại bỏ",
                "text_accent_folded": "loai bo",
                "confidence": 0.49,
                "polygon": polygon,
            },
            {
                "text_raw": "Trao kinh phí",
                "text_nfc": "Trao kinh phí",
                "text_accent_folded": "trao kinh phi",
                "confidence": 0.50,
                "polygon": polygon,
            },
        ]


class _ASR:
    model_size = "Systran/faster-whisper-small"
    model_revision = "revision"
    compute_type = "float16"
    vad_filter = True

    def transcribe(self, audio_path: Path) -> dict:
        assert audio_path.stat().st_size > 44
        return {
            "segments": [
                {
                    "start_ms": 800,
                    "end_ms": 1_000,
                    "text_raw": "chương trình từ thiện",
                    "text_nfc": "chương trình từ thiện",
                    "text_accent_folded": "chuong trinh tu thien",
                    "avg_logprob": math.log(0.8),
                }
            ]
        }


def _finalized_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_contracts: dict[str, dict] | None = None,
):
    maps, keyframes, videos = _zip_corpus(tmp_path)
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        video_archives=(videos,),
        output_dir=tmp_path / "artifacts",
        batch_id=BATCH_ID,
        expected_videos=1,
        expected_frames=2,
        model_contracts=model_contracts,
    )
    encoder = _Encoder()
    ocr = _OCR()
    producer.run_siglip2(encoder, batch_size=128)
    producer.run_ocr(ocr)

    def fake_audio(_archive, _member, audio_path, *, ffmpeg_path):
        assert str(ffmpeg_path) == "ffmpeg"
        audio_path.write_bytes(b"R" * 45)
        return True

    monkeypatch.setattr(CloudArtifactProducer, "_stream_audio_member", staticmethod(fake_audio))
    producer.run_asr(_ASR())
    manifest = producer.finalize()
    return producer, encoder, ocr, manifest


def test_zip_streaming_deduplicates_canonical_frames_and_never_extracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        zipfile.ZipFile,
        "extract",
        lambda *_args, **_kwargs: pytest.fail("ZIP extraction is forbidden"),
    )
    monkeypatch.setattr(
        zipfile.ZipFile,
        "extractall",
        lambda *_args, **_kwargs: pytest.fail("ZIP extraction is forbidden"),
    )

    producer, encoder, ocr, manifest = _finalized_bundle(tmp_path, monkeypatch)

    assert encoder.batch_sizes == [2]
    assert ocr.calls == 2
    assert manifest["videos"] == 1
    assert manifest["vector"]["count"] == 2
    assert manifest["text"]["ocr"]["count"] == 2
    assert manifest["schema_version"] == 3
    assert manifest["complete"] is True
    assert manifest["completed_channels"] == ["siglip2", "ocr", "asr"]
    assert manifest["text"]["ocr"]["scanned_frames"] == 2
    assert manifest["input_archives"]["keyframes"][0]["members"] == 3
    assert manifest["tool_versions"]["python"]
    assert manifest["text"]["asr"]["count"] == 1
    assert {row["source_frame_idx"] for row in manifest["vector"]["rows"]} == {0, 25}
    assert (producer.output_dir / SIGLIP_FEATURES_NAME).is_file()
    assert (producer.output_dir / SIGLIP_MANIFEST_NAME).is_file()
    assert (producer.output_dir / OCR_FEATURES_NAME).is_file()
    assert (producer.output_dir / ASR_TRANSCRIPTS_NAME).is_file()
    progress = json.loads(producer.progress_path.read_text(encoding="utf-8"))
    assert progress["stage"] == "asr"
    assert progress["state"] == "completed"
    assert progress["completed_videos"] == 1
    assert producer.progress_path.with_suffix(".jsonl").is_file()
    assert producer.output_dir not in producer.scratch_dir.parents
    assert not producer.scratch_dir.exists()


def test_ocr_finalizer_adds_deterministic_multiline_frame_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MultiLineOCR(_OCR):
        def extract_text(self, image: bytes) -> list[dict]:
            assert image.startswith(b"\xff\xd8")
            self.calls += 1
            return [
                {
                    "text_raw": "Trao kinh phí",
                    "text_nfc": "Trao kinh phí",
                    "text_accent_folded": "trao kinh phi",
                    "confidence": 0.9,
                    "polygon": [[0, 20], [10, 20], [10, 30], [0, 30]],
                },
                {
                    "text_raw": "Chương trình",
                    "text_nfc": "Chương trình",
                    "text_accent_folded": "chuong trinh",
                    "confidence": 0.8,
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                },
            ]

    maps, keyframes, videos = _zip_corpus(tmp_path)
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        video_archives=(videos,),
        output_dir=tmp_path / "artifacts",
        batch_id=BATCH_ID,
        expected_videos=1,
        expected_frames=2,
    )
    producer.run_siglip2(_Encoder())
    producer.run_ocr(MultiLineOCR())

    def fake_audio(_archive, _member, audio_path, *, ffmpeg_path):
        del ffmpeg_path
        audio_path.write_bytes(b"R" * 45)
        return True

    monkeypatch.setattr(CloudArtifactProducer, "_stream_audio_member", staticmethod(fake_audio))
    producer.run_asr(_ASR())
    manifest = producer.finalize()
    assert manifest["text"]["ocr"]["count"] == 6
    with gzip.open(producer.output_dir / OCR_FEATURES_NAME, "rt", encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    aggregates = [value for value in documents if value.get("aggregation") == "frame"]
    assert len(aggregates) == 2
    assert aggregates[0]["text_nfc"] == "Chương trình Trao kinh phí"
    assert aggregates[0]["id"].startswith("ocr-frame:")


def test_resume_skips_valid_video_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, encoder, ocr, _manifest = _finalized_bundle(tmp_path, monkeypatch)
    assert producer.work_dir.name == "cloud_work"
    assert not producer.work_dir.name.startswith(".")
    producer.run_siglip2(encoder, batch_size=128)
    producer.run_ocr(ocr)
    assert encoder.batch_sizes == [2]
    assert ocr.calls == 2


def test_bundle_hash_validation_and_memmap_memory_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _encoder, _ocr, _manifest = _finalized_bundle(tmp_path, monkeypatch)
    before = int(safe_resource_snapshot().get("process_rss_bytes") or 0)
    bundle = CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID)
    counts = bundle.validate()
    after = int(safe_resource_snapshot().get("process_rss_bytes") or 0)

    assert counts == {"siglip2": 2, "ocr": 2, "asr": 1}
    assert isinstance(bundle.vectors, np.memmap)
    assert max(before, after) < 2 * 1024**3

    with (producer.output_dir / OCR_FEATURES_NAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(CloudArtifactError, match="cloud_artifact_hash_mismatch"):
        CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID).validate()


class _VectorStore:
    vector_size = 4

    def __init__(self) -> None:
        self.created: list[bool] = []
        self.calls: list[tuple[list[str], np.ndarray, list[dict], int]] = []
        self.by_video: dict[str, int] = {}

    def create_collection(self, recreate: bool = False) -> None:
        self.created.append(recreate)

    def upsert(self, ids, vectors, payloads, batch_size=256) -> None:
        self.calls.append((list(ids), vectors, list(payloads), batch_size))
        for payload in payloads:
            self.by_video[str(payload["video_id"])] = (
                self.by_video.get(str(payload["video_id"]), 0) + 1
            )

    def count(self, filters=None) -> int:
        video_id = (filters or {}).get("video_id")
        return self.by_video.get(str(video_id), 0) if video_id else sum(self.by_video.values())


class _TextStore:
    def __init__(self) -> None:
        self.created = 0
        self.calls: list[tuple[list[dict], int]] = []
        self.documents: dict[str, dict] = {}

    def create_index(self) -> None:
        self.created += 1

    def add_documents(self, documents, batch_size=1000) -> None:
        self.calls.append((list(documents), batch_size))
        self.documents.update({str(value["id"]): dict(value) for value in documents})

    def count_documents(self, filters=None) -> int:
        if not filters:
            return len(self.documents)
        batch_match = re.search(r'batch_id = "([^"]+)"', filters)
        video_match = re.search(r'video_id = "([^"]+)"', filters)
        return sum(
            1
            for value in self.documents.values()
            if (batch_match is None or value.get("batch_id") == batch_match.group(1))
            and (video_match is None or value.get("video_id") == video_match.group(1))
        )

    def iter_documents(self, *, filters=None, batch_size=1000):
        del batch_size
        batch_match = re.search(r'batch_id = "([^"]+)"', filters or "")
        video_match = re.search(r'video_id = "([^"]+)"', filters or "")
        for value in self.documents.values():
            if batch_match is not None and value.get("batch_id") != batch_match.group(1):
                continue
            if video_match is not None and value.get("video_id") != video_match.group(1):
                continue
            yield dict(value)


def test_importer_uses_runtime_batches_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _encoder, _ocr, _manifest = _finalized_bundle(tmp_path, monkeypatch)
    bundle = CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID)
    vector = _VectorStore()
    ocr = _TextStore()
    asr = _TextStore()
    ledger = JobLedger(tmp_path / "jobs.sqlite3")
    importer = CloudArtifactImporter(
        bundle=bundle,
        vector_store=vector,
        ocr_store=ocr,
        asr_store=asr,
        ledger=ledger,
        completion_path=tmp_path / "import-complete.json",
    )

    report = importer.run()

    assert report.vectors == 2
    assert report.ocr_documents == 2
    assert report.asr_documents == 1
    assert vector.created == [False]
    assert [call[3] for call in vector.calls] == [1000]
    assert [call[1] for call in ocr.calls] == [2000]
    assert [call[1] for call in asr.calls] == [1000]
    assert all(call[1].shape[0] <= 1000 for call in vector.calls)
    assert (tmp_path / "import-complete.json").is_file()

    previous = (len(vector.calls), len(ocr.calls), len(asr.calls))
    second = importer.run(resume=True)
    assert second.to_dict()["channels"] == ["siglip2", "ocr", "asr"]
    assert (len(vector.calls), len(ocr.calls), len(asr.calls)) == previous


def test_partial_import_reopens_completed_legacy_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _encoder, _ocr, _manifest = _finalized_bundle(tmp_path, monkeypatch)
    bundle = CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID)
    ledger = JobLedger(tmp_path / "jobs.sqlite3")
    ledger.initialize()
    ledger.register_batch(BATCH_ID, "aic25")
    ledger.register_video(BATCH_ID, "L21_V001", "legacy://L21_V001")
    ledger.start_stage(
        BATCH_ID,
        "L21_V001",
        "reference_siglip2",
        "be1a58f08d133dcbf24342d4",
    )
    ledger.complete_stage(BATCH_ID, "L21_V001", "reference_siglip2")
    vector = _VectorStore()

    report = CloudArtifactImporter(
        bundle=bundle,
        vector_store=vector,
        ocr_store=_TextStore(),
        asr_store=_TextStore(),
        ledger=ledger,
    ).run(channels=("siglip2",), resume=True)

    assert report.vectors == 2
    assert len(vector.calls) == 1
    assert ledger.get_stage(BATCH_ID, "L21_V001", "reference_siglip2")["state"] == "completed"


def test_validate_only_never_contacts_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _encoder, _ocr, _manifest = _finalized_bundle(tmp_path, monkeypatch)
    bundle = CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID)
    vector = _VectorStore()
    ocr = _TextStore()
    asr = _TextStore()
    report = CloudArtifactImporter(
        bundle=bundle,
        vector_store=vector,
        ocr_store=ocr,
        asr_store=asr,
    ).run(validate_only=True)
    assert report.vectors == 2
    assert vector.created == []
    assert ocr.created == asr.created == 0


def test_model_contract_mismatch_fails_before_store_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = {
        "siglip2": {
            "name": _Encoder.model_name,
            "revision": _Encoder.revision,
            "dimension": 4,
            "dtype": "float16",
        },
        "ocr": {"name": _OCR.model_name, "revision": None},
        "asr": {"name": _ASR.model_size, "revision": _ASR.model_revision},
    }
    producer, _encoder, _ocr, _manifest = _finalized_bundle(
        tmp_path,
        monkeypatch,
        model_contracts=contracts,
    )
    mismatched = {stage: dict(value) for stage, value in contracts.items()}
    mismatched["siglip2"]["revision"] = "wrong-revision"
    vector, ocr, asr = _VectorStore(), _TextStore(), _TextStore()
    importer = CloudArtifactImporter(
        bundle=CloudArtifactBundle(
            producer.output_dir,
            expected_batch_id=BATCH_ID,
            expected_model_contracts=mismatched,
            expected_videos=1,
            expected_frames=2,
        ),
        vector_store=vector,
        ocr_store=ocr,
        asr_store=asr,
    )

    with pytest.raises(CloudArtifactError, match="cloud_model_contract_mismatch"):
        importer.run()
    assert vector.created == []
    assert ocr.created == asr.created == 0


def test_partial_import_does_not_write_full_completion_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _encoder, _ocr, _manifest = _finalized_bundle(tmp_path, monkeypatch)
    completion = tmp_path / "import-complete.json"
    importer = CloudArtifactImporter(
        bundle=CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID),
        vector_store=_VectorStore(),
        ocr_store=_TextStore(),
        asr_store=_TextStore(),
        completion_path=completion,
    )
    report = importer.run(channels=("siglip2",))
    assert report.channels == ("siglip2",)
    assert not completion.exists()


def test_vector_only_bundle_validates_imports_and_rejects_missing_ocr(
    tmp_path: Path,
) -> None:
    maps, keyframes, _videos = _zip_corpus(tmp_path)
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=tmp_path / "artifacts",
        batch_id=BATCH_ID,
        expected_videos=1,
        expected_frames=2,
    )
    producer.run_siglip2(_Encoder())
    manifest = producer.finalize(channels=("siglip2",))

    assert manifest["complete"] is False
    assert manifest["completed_channels"] == ["siglip2"]
    assert set(manifest["files"]) == {SIGLIP_FEATURES_NAME}
    assert not (producer.output_dir / OCR_FEATURES_NAME).exists()
    bundle = CloudArtifactBundle(
        producer.output_dir,
        expected_batch_id=BATCH_ID,
        expected_videos=1,
        expected_frames=2,
    )
    assert bundle.validate(channels=("siglip2",)) == {"siglip2": 2}

    vector, ocr, asr = _VectorStore(), _TextStore(), _TextStore()
    receipts = tmp_path / "channel-receipts"
    report = CloudArtifactImporter(
        bundle=bundle,
        vector_store=vector,
        ocr_store=ocr,
        asr_store=asr,
        channel_receipt_dir=receipts,
        completion_path=tmp_path / "full.json",
    ).run(channels=("siglip2",))
    assert report.vectors == 2
    assert (receipts / "siglip2.json").is_file()
    assert not (tmp_path / "full.json").exists()

    untouched_vector, untouched_ocr = _VectorStore(), _TextStore()
    with pytest.raises(CloudArtifactError, match="cloud_requested_channel_incomplete"):
        CloudArtifactImporter(
            bundle=CloudArtifactBundle(
                producer.output_dir,
                expected_batch_id=BATCH_ID,
            ),
            vector_store=untouched_vector,
            ocr_store=untouched_ocr,
            asr_store=_TextStore(),
        ).run(channels=("ocr",))
    assert untouched_vector.created == []
    assert untouched_ocr.created == 0


def test_partial_ocr_snapshot_requires_opt_in_and_expands_without_reimport(
    tmp_path: Path,
) -> None:
    maps, keyframes = _zip_two_video_corpus(tmp_path)
    output = tmp_path / "artifacts"
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=output,
        batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
    )
    producer.run_siglip2(_Encoder())
    first_ocr = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=output,
        batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
        selected_video_ids=("L21_V001",),
    )
    first_ocr.run_ocr(_OCR())

    with pytest.raises(CloudArtifactError, match="cloud_stage_video_mismatch"):
        producer.finalize(channels=("siglip2", "ocr"))
    manifest = producer.finalize(
        channels=("siglip2", "ocr"),
        allow_partial=True,
    )
    assert manifest["complete"] is False
    assert manifest["channel_coverage"]["siglip2"]["coverage_complete"] is True
    assert manifest["channel_coverage"]["ocr"] == {
        "coverage_complete": False,
        "completed_videos": 1,
        "expected_videos": 2,
        "records": 1,
        "scanned_frames": 1,
        "expected_frames": 2,
    }
    assert set(manifest["text"]["ocr"]["video_fingerprints"]) == {"L21_V001"}

    with pytest.raises(CloudArtifactError, match="cloud_partial_bundle_requires_opt_in"):
        CloudArtifactBundle(
            output,
            expected_batch_id=BATCH_ID,
            expected_videos=2,
            expected_frames=2,
        ).validate(channels=("ocr",))

    ledger = JobLedger(tmp_path / "partial-jobs.sqlite3")
    store = _TextStore()
    channel_receipts = tmp_path / "partial-channel-receipts"
    partial_bundle = CloudArtifactBundle(
        output,
        expected_batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
        allow_partial=True,
    )
    first_report = CloudArtifactImporter(
        bundle=partial_bundle,
        vector_store=_VectorStore(),
        ocr_store=store,
        asr_store=_TextStore(),
        ledger=ledger,
        channel_receipt_dir=channel_receipts,
        completion_path=tmp_path / "full-completion.json",
    ).run(channels=("ocr",))
    assert first_report.ocr_documents == 1
    first_call_count = len(store.calls)
    receipt = json.loads((channel_receipts / "ocr.json").read_text(encoding="utf-8"))
    assert receipt["contract"] == "cloud-import-channel-v2"
    assert receipt["snapshot_import_complete"] is True
    assert receipt["coverage_complete"] is False
    assert not (tmp_path / "full-completion.json").exists()

    second_ocr = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=output,
        batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
        selected_video_ids=("L21_V002",),
    )
    second_ocr.run_ocr(_OCR())
    expanded_manifest = producer.finalize(channels=("siglip2", "ocr"))
    assert expanded_manifest["channel_coverage"]["ocr"]["coverage_complete"] is True
    expanded_bundle = CloudArtifactBundle(
        output,
        expected_batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
    )
    second_report = CloudArtifactImporter(
        bundle=expanded_bundle,
        vector_store=_VectorStore(),
        ocr_store=store,
        asr_store=_TextStore(),
        ledger=ledger,
    ).run(channels=("ocr",), resume=True)
    assert second_report.ocr_documents == 2
    assert len(store.calls) == first_call_count + 1
    assert len(store.documents) == 2


def test_partial_ocr_resume_audits_and_replays_acknowledged_failed_batch(
    tmp_path: Path,
) -> None:
    maps, keyframes = _zip_two_video_corpus(tmp_path)
    output = tmp_path / "artifacts"
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=output,
        batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
    )
    producer.run_siglip2(_Encoder())
    partial_ocr = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=output,
        batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
        selected_video_ids=("L21_V001",),
    )
    partial_ocr.run_ocr(_OCR())
    producer.finalize(channels=("siglip2", "ocr"), allow_partial=True)
    bundle = CloudArtifactBundle(
        output,
        expected_batch_id=BATCH_ID,
        expected_videos=2,
        expected_frames=2,
        allow_partial=True,
    )
    bundle.validate(channels=("ocr",))
    documents = list(bundle.iter_documents("ocr"))
    assert len(documents) == 1

    ledger = JobLedger(tmp_path / "jobs.sqlite3")
    ledger.initialize()
    ledger.register_batch(BATCH_ID, "aic25")
    ledger.register_video(BATCH_ID, "L21_V001", "cloud-artifact://L21_V001")
    fingerprint = f"cloud-import-ocr-v3:{bundle.video_fingerprint('ocr', 'L21_V001')[:24]}"
    ledger.start_stage(BATCH_ID, "L21_V001", "reference_ocr", fingerprint)
    ledger.fail_stage(
        BATCH_ID,
        "L21_V001",
        "reference_ocr",
        "simulated_poll_failure",
    )
    store = _TextStore()
    store.add_documents(documents)

    report = CloudArtifactImporter(
        bundle=bundle,
        vector_store=_VectorStore(),
        ocr_store=store,
        asr_store=_TextStore(),
        ledger=ledger,
    ).run(channels=("ocr",), resume=True)

    assert report.ocr_documents == 1
    assert len(store.documents) == 1
    assert ledger.get_stage(BATCH_ID, "L21_V001", "reference_ocr")["state"] == "completed"


def test_channel_fingerprint_survives_manifest_expansion_and_v2_is_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps, keyframes, videos = _zip_corpus(tmp_path)
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        video_archives=(videos,),
        output_dir=tmp_path / "artifacts",
        batch_id=BATCH_ID,
        expected_videos=1,
        expected_frames=2,
    )
    producer.run_siglip2(_Encoder())
    producer.finalize(channels=("siglip2",))
    partial = CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID)
    partial.validate(channels=("siglip2",))
    siglip_fingerprint = partial.channel_fingerprint("siglip2")
    producer.run_ocr(_OCR())

    def fake_audio(_archive, _member, audio_path, *, ffmpeg_path):
        del ffmpeg_path
        audio_path.write_bytes(b"R" * 45)
        return True

    monkeypatch.setattr(CloudArtifactProducer, "_stream_audio_member", staticmethod(fake_audio))
    producer.run_asr(_ASR())
    producer.finalize()
    expanded = CloudArtifactBundle(producer.output_dir, expected_batch_id=BATCH_ID)
    expanded.validate()
    assert expanded.channel_fingerprint("siglip2") == siglip_fingerprint

    legacy_dir = tmp_path / "legacy-v2"
    legacy_dir.mkdir()
    for name in (
        SIGLIP_FEATURES_NAME,
        SIGLIP_MANIFEST_NAME,
        OCR_FEATURES_NAME,
        ASR_TRANSCRIPTS_NAME,
    ):
        (legacy_dir / name).write_bytes((producer.output_dir / name).read_bytes())
    legacy_manifest = json.loads((legacy_dir / SIGLIP_MANIFEST_NAME).read_text("utf-8"))
    legacy_manifest["schema_version"] = 2
    legacy_manifest.pop("completed_channels")
    legacy_manifest.pop("channel_fingerprints")
    (legacy_dir / SIGLIP_MANIFEST_NAME).write_text(json.dumps(legacy_manifest), encoding="utf-8")
    assert CloudArtifactBundle(legacy_dir, expected_batch_id=BATCH_ID).validate() == {
        "siglip2": 2,
        "ocr": 2,
        "asr": 1,
    }


def test_partition_plan_is_deterministic_archive_local_and_worker_disjoint(
    tmp_path: Path,
) -> None:
    maps = tmp_path / "maps.zip"
    keyframes = tmp_path / "keyframes-pack.zip"
    with zipfile.ZipFile(maps, "w") as archive:
        for ordinal in range(1, 6):
            video_id = f"L21_V{ordinal:03d}"
            archive.writestr(
                f"maps/{video_id}.csv",
                "n,pts_time,fps,frame_idx\n1,0.0,25,0\n2,1.0,25,25\n",
            )
    with zipfile.ZipFile(keyframes, "w") as archive:
        for ordinal in range(1, 6):
            video_id = f"L21_V{ordinal:03d}"
            archive.writestr(f"keyframes/{video_id}/001.jpg", _jpeg((1, 2, ordinal)))
            archive.writestr(f"keyframes/{video_id}/002.jpg", _jpeg((2, 3, ordinal)))

    first = build_cloud_partition_manifest(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        batch_id=BATCH_ID,
        stage="ocr",
        max_frames=4,
        worker_count=2,
    )
    second = build_cloud_partition_manifest(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        batch_id=BATCH_ID,
        stage="ocr",
        max_frames=4,
        worker_count=2,
    )
    assert first == second
    assert first["expected_videos"] == 5
    assert first["expected_frames"] == 10
    assert all(value["canonical_frames"] <= 4 for value in first["partitions"])
    assert {value["archive"] for value in first["partitions"]} == {keyframes.name}
    assert {value["worker_slot"] for value in first["partitions"]} == {0, 1}
    videos = [item for value in first["partitions"] for item in value["videos"]]
    assert len(videos) == len(set(videos)) == 5

    path = tmp_path / "partitions.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    selected = first["partitions"][0]
    loaded = load_cloud_partition(
        path,
        batch_id=BATCH_ID,
        stage="ocr",
        partition_id=selected["partition_id"],
        worker_slot=selected["worker_slot"],
        worker_count=2,
    )
    assert loaded["videos"] == selected["videos"]
    with pytest.raises(CloudArtifactError, match="partition_worker_assignment_mismatch"):
        load_cloud_partition(
            path,
            batch_id=BATCH_ID,
            stage="ocr",
            partition_id=selected["partition_id"],
            worker_slot=1 - selected["worker_slot"],
            worker_count=2,
        )


def test_partition_worker_resumes_existing_flat_receipt(tmp_path: Path) -> None:
    maps, keyframes, _videos = _zip_corpus(tmp_path)
    output = tmp_path / "artifacts"
    original = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=output,
        batch_id=BATCH_ID,
    )
    ocr = _OCR()
    original.run_ocr(ocr)
    partitioned = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        output_dir=output,
        batch_id=BATCH_ID,
        partition_id="ocr-keyframes-000",
        selected_video_ids=("L21_V001",),
    )
    partitioned.run_ocr(ocr)
    assert ocr.calls == 2
    assert not (output / "cloud_work" / "ocr" / "partitions").exists()


def test_asr_failure_cleans_ephemeral_audio_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingASR(_ASR):
        def transcribe(self, audio_path: Path) -> dict:
            assert audio_path.is_file()
            raise RuntimeError("synthetic ASR failure")

    maps, keyframes, videos = _zip_corpus(tmp_path)
    scratch = tmp_path / "ephemeral-scratch"
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        video_archives=(videos,),
        output_dir=tmp_path / "artifacts",
        scratch_dir=scratch,
    )

    def fake_audio(_archive, _member, audio_path, *, ffmpeg_path):
        del ffmpeg_path
        audio_path.write_bytes(b"R" * 45)
        return True

    monkeypatch.setattr(CloudArtifactProducer, "_stream_audio_member", staticmethod(fake_audio))
    with pytest.raises(RuntimeError, match="synthetic ASR failure"):
        producer.run_asr(FailingASR())
    assert not tuple(scratch.rglob("*.wav"))
    assert not tuple(scratch.rglob("*.log"))


def test_asr_audio_member_uses_seekable_scratch_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "videos.zip"
    payload = b"ordinary-mp4-with-moov-at-end"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("video/L21_V001.mp4", payload)

    audio_path = tmp_path / "audio.wav"
    observed_media_path: Path | None = None

    def fake_run(command, *, stdin, stdout, stderr, check):
        nonlocal observed_media_path
        del stdin, stdout, check
        media_path = Path(command[command.index("-i") + 1])
        observed_media_path = media_path
        assert media_path.name == "source.mp4"
        assert media_path.read_bytes() == payload
        assert "pipe:0" not in command
        assert command[command.index("-map") + 1] == "0:a:0"
        Path(command[-1]).write_bytes(b"R" * 45)
        stderr.write(b"")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("src.ingest.cloud_artifacts.subprocess.run", fake_run)
    with zipfile.ZipFile(archive_path) as archive:
        assert CloudArtifactProducer._stream_audio_member(
            archive,
            "video/L21_V001.mp4",
            audio_path,
            ffmpeg_path="ffmpeg",
        )

    assert observed_media_path is not None
    assert not observed_media_path.exists()
    assert audio_path.stat().st_size == 45
    assert not audio_path.with_suffix(".ffmpeg.log").exists()


def test_asr_audio_member_does_not_hide_container_failure_as_no_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "videos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("video/L21_V001.mp4", b"broken-media")

    def fake_run(_command, *, stdin, stdout, stderr, check):
        del stdin, stdout, check
        stderr.write(b"moov atom not found\nStream map '0:a:0' matches no streams\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("src.ingest.cloud_artifacts.subprocess.run", fake_run)
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(CloudArtifactError, match="ffmpeg_audio_stream_failed"):
            CloudArtifactProducer._stream_audio_member(
                archive,
                "video/L21_V001.mp4",
                tmp_path / "audio.wav",
                ffmpeg_path="ffmpeg",
            )


def test_asr_audio_member_accepts_verified_missing_audio_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "videos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("video/L21_V001.mp4", b"silent-media")

    def fake_run(_command, *, stdin, stdout, stderr, check):
        del stdin, stdout, check
        stderr.write(b"Stream map '0:a:0' matches no streams\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("src.ingest.cloud_artifacts.subprocess.run", fake_run)
    with zipfile.ZipFile(archive_path) as archive:
        assert not CloudArtifactProducer._stream_audio_member(
            archive,
            "video/L21_V001.mp4",
            tmp_path / "audio.wav",
            ffmpeg_path="ffmpeg",
        )


def test_asr_receipt_records_seekable_extraction_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps, keyframes, videos = _zip_corpus(tmp_path)
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        video_archives=(videos,),
        output_dir=tmp_path / "artifacts",
    )
    calls = 0

    def fake_audio(_archive, _member, audio_path, *, ffmpeg_path):
        nonlocal calls
        del ffmpeg_path
        calls += 1
        audio_path.write_bytes(b"R" * 45)
        return True

    monkeypatch.setattr(CloudArtifactProducer, "_stream_audio_member", staticmethod(fake_audio))
    producer.run_asr(_ASR())
    receipt_path = next((producer.work_dir / "asr").rglob("*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["extraction_contract"] == ASR_AUDIO_EXTRACTION_CONTRACT
    assert receipt["audio_stream_present"] is True
    assert receipt["transcript_segment_count"] == 1
    assert receipt["zero_transcript_reason"] is None

    receipt.pop("extraction_contract")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    producer.run_asr(_ASR())
    assert calls == 2


@pytest.mark.parametrize(
    ("has_audio", "expected_reason"),
    ((False, "no_audio_stream"), (True, "no_speech_detected")),
)
def test_asr_zero_transcript_receipt_distinguishes_stream_from_speech(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_audio: bool,
    expected_reason: str,
) -> None:
    class EmptyASR(_ASR):
        def transcribe(self, audio_path: Path) -> dict:
            assert has_audio
            assert audio_path.is_file()
            return {"segments": ()}

    maps, keyframes, videos = _zip_corpus(tmp_path)
    producer = CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=(keyframes,),
        video_archives=(videos,),
        output_dir=tmp_path / f"artifacts-{expected_reason}",
    )

    def fake_audio(_archive, _member, audio_path, *, ffmpeg_path):
        del ffmpeg_path
        if has_audio:
            audio_path.write_bytes(b"R" * 45)
        return has_audio

    monkeypatch.setattr(
        CloudArtifactProducer,
        "_stream_audio_member",
        staticmethod(fake_audio),
    )
    producer.run_asr(EmptyASR())
    receipt_path = next((producer.work_dir / "asr").rglob("*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["audio_stream_present"] is has_audio
    assert receipt["transcript_segment_count"] == 0
    assert receipt["zero_transcript_reason"] == expected_reason


def test_siglip_cuda_oom_backoff_keeps_every_image(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = SigLIP2Encoder(device="cuda", batch_size=4, max_batch_size=4)
    encoder._model = object()
    encoder._processor = object()
    encoder._torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setattr(encoder, "_load_model", lambda: None)
    monkeypatch.setattr(encoder, "_open_image", lambda _value: Image.new("RGB", (2, 2)))
    attempted: list[int] = []

    def encode(images):
        attempted.append(len(images))
        if len(images) > 1:
            raise RuntimeError("CUDA out of memory")
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(encoder, "_encode_image_batch", encode)
    output = encoder._encode_images_unlocked([b"x"] * 4)
    assert output.shape == (4, 4)
    assert attempted[:3] == [4, 2, 1]


def _channel(name: str, count: int) -> ChannelReadiness:
    return ChannelReadiness(
        name,
        ChannelApplicability.OPTIONAL,
        ChannelState.READY,
        count=count,
    )


def _write_completion_receipt(tmp_path: Path, counts: dict[str, int]) -> None:
    manifest = tmp_path / "bundle" / SIGLIP_MANIFEST_NAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt = tmp_path / "data/checkpoints/cloud-import/b01-import-complete.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "contract": "cloud-import-completion-v2",
                "batch_id": BATCH_ID,
                "complete": True,
                "completed_channels": ["siglip2", "ocr", "asr"],
                "manifest_path": str(manifest),
                "manifest_sha256": digest,
                "external_counts": counts,
            }
        ),
        encoding="utf-8",
    )


def test_full_multimodal_health_requires_positive_channel_counts(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "artifacts.sqlite3") as connection:
        connection.execute("CREATE TABLE canonical_frames(batch_id TEXT, video_id TEXT)")
        connection.execute("INSERT INTO canonical_frames VALUES (?, ?)", (BATCH_ID, "L21_V001"))
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute("CREATE TABLE videos(batch_id TEXT, video_id TEXT, source_uri TEXT)")
        connection.execute(
            "INSERT INTO videos VALUES (?, ?, ?)",
            (BATCH_ID, "L21_V001", "cloud-artifact://L21_V001"),
        )
        connection.execute(
            "CREATE TABLE stages(batch_id TEXT, video_id TEXT, stage_name TEXT, state TEXT, error TEXT)"
        )
        connection.execute(
            "CREATE TABLE artifacts(batch_id TEXT, video_id TEXT, stage_name TEXT, kind TEXT)"
        )
        for stage in ("reference_ocr", "reference_siglip2", "reference_asr"):
            connection.execute(
                "INSERT INTO stages VALUES (?, ?, ?, 'completed', NULL)",
                (BATCH_ID, "L21_V001", stage),
            )
    config = {
        "data": {
            "catalog_db": str(tmp_path / "jobs.sqlite3"),
            "artifact_db": str(tmp_path / "artifacts.sqlite3"),
        },
        "readiness": {
            "batches": {
                BATCH_ID: {
                    "expected_videos": 1,
                    "expected_canonical_frames": 1,
                    "expected_mapping_rows": 1,
                }
            }
        },
    }
    _write_completion_receipt(tmp_path, {"siglip2": 1, "ocr": 1, "asr": 1})
    healthy = multimodal_coverage_snapshot(
        config,
        BATCH_ID,
        (
            _channel("organizer_clip", 1),
            _channel("visual", 1),
            _channel("ocr", 1),
            _channel("asr", 1),
        ),
        project_root=tmp_path,
    )
    assert healthy["capability_mode"] == "full_multimodal"
    assert healthy["status_label"] == "HEALTHY · Full Multi-Modal"
    assert healthy["full_multimodal"]["healthy"] is True

    degraded = multimodal_coverage_snapshot(
        config,
        BATCH_ID,
        (
            _channel("organizer_clip", 1),
            _channel("visual", 1),
            _channel("ocr", 1),
            _channel("asr", 0),
        ),
        project_root=tmp_path,
    )
    assert degraded["capability_mode"] == "multimodal_partial"
    assert degraded["full_multimodal"]["unavailable_channels"] == ("asr",)


def test_progressive_health_lists_channels_missing_after_partial_import(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "artifacts.sqlite3") as connection:
        connection.execute("CREATE TABLE canonical_frames(batch_id TEXT, video_id TEXT)")
        connection.execute("INSERT INTO canonical_frames VALUES (?, ?)", (BATCH_ID, "L21_V001"))
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute("CREATE TABLE videos(batch_id TEXT, video_id TEXT, source_uri TEXT)")
        connection.execute(
            "INSERT INTO videos VALUES (?, ?, ?)",
            (BATCH_ID, "L21_V001", "cloud-artifact://L21_V001"),
        )
        connection.execute(
            "CREATE TABLE stages(batch_id TEXT, video_id TEXT, stage_name TEXT, state TEXT, error TEXT)"
        )
        connection.execute(
            "CREATE TABLE artifacts(batch_id TEXT, video_id TEXT, stage_name TEXT, kind TEXT)"
        )
        connection.execute(
            "INSERT INTO stages VALUES (?, ?, 'reference_siglip2', 'completed', NULL)",
            (BATCH_ID, "L21_V001"),
        )
    config = {
        "data": {
            "catalog_db": str(tmp_path / "jobs.sqlite3"),
            "artifact_db": str(tmp_path / "artifacts.sqlite3"),
        },
        "readiness": {
            "batches": {
                BATCH_ID: {
                    "expected_videos": 1,
                    "expected_canonical_frames": 1,
                    "expected_mapping_rows": 1,
                }
            }
        },
    }
    siglip_only = multimodal_coverage_snapshot(
        config,
        BATCH_ID,
        (
            _channel("organizer_clip", 1),
            _channel("visual", 1),
            _channel("ocr", 0),
            _channel("asr", 0),
        ),
        project_root=tmp_path,
    )
    assert siglip_only["siglip2"]["consistent"] is True
    assert siglip_only["status_label"] == "DEGRADED · Multi-Modal Incomplete"
    assert siglip_only["full_multimodal"]["unavailable_channels"] == ("ocr", "asr")

    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute(
            "INSERT INTO stages VALUES (?, ?, 'reference_ocr', 'completed', NULL)",
            (BATCH_ID, "L21_V001"),
        )
    after_ocr = multimodal_coverage_snapshot(
        config,
        BATCH_ID,
        (
            _channel("organizer_clip", 1),
            _channel("visual", 1),
            _channel("ocr", 3),
            _channel("asr", 0),
        ),
        project_root=tmp_path,
    )
    assert after_ocr["ocr"]["consistent"] is True
    assert after_ocr["full_multimodal"]["unavailable_channels"] == ("asr",)


def test_partial_ocr_is_usable_while_full_multimodal_remains_degraded(
    tmp_path: Path,
) -> None:
    with sqlite3.connect(tmp_path / "artifacts.sqlite3") as connection:
        connection.execute("CREATE TABLE canonical_frames(batch_id TEXT, video_id TEXT)")
        connection.executemany(
            "INSERT INTO canonical_frames VALUES (?, ?)",
            ((BATCH_ID, "L21_V001"), (BATCH_ID, "L21_V002")),
        )
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute("CREATE TABLE videos(batch_id TEXT, video_id TEXT, source_uri TEXT)")
        connection.executemany(
            "INSERT INTO videos VALUES (?, ?, ?)",
            (
                (BATCH_ID, "L21_V001", "cloud-artifact://L21_V001"),
                (BATCH_ID, "L21_V002", "cloud-artifact://L21_V002"),
            ),
        )
        connection.execute(
            "CREATE TABLE stages(batch_id TEXT, video_id TEXT, stage_name TEXT, state TEXT, error TEXT)"
        )
        connection.execute(
            "CREATE TABLE artifacts(batch_id TEXT, video_id TEXT, stage_name TEXT, kind TEXT)"
        )
        connection.executemany(
            "INSERT INTO stages VALUES (?, ?, ?, 'completed', NULL)",
            (
                (BATCH_ID, "L21_V001", "reference_siglip2"),
                (BATCH_ID, "L21_V002", "reference_siglip2"),
                (BATCH_ID, "L21_V001", "reference_ocr"),
            ),
        )
    config = {
        "data": {
            "catalog_db": str(tmp_path / "jobs.sqlite3"),
            "artifact_db": str(tmp_path / "artifacts.sqlite3"),
        },
        "readiness": {
            "batches": {
                BATCH_ID: {
                    "expected_videos": 2,
                    "expected_canonical_frames": 2,
                    "expected_mapping_rows": 2,
                }
            }
        },
    }
    snapshot = multimodal_coverage_snapshot(
        config,
        BATCH_ID,
        (
            _channel("organizer_clip", 2),
            _channel("visual", 2),
            _channel("ocr", 3),
            _channel("asr", 0),
        ),
        project_root=tmp_path,
    )
    assert snapshot["capability_mode"] == "multimodal_partial"
    assert snapshot["ocr"]["usable"] is True
    assert snapshot["ocr"]["consistent"] is False
    assert snapshot["ocr"]["coverage_mode"] == "partial"
    assert snapshot["ocr"]["coverage_ratio"] == 0.5
    assert snapshot["full_multimodal"]["unavailable_channels"] == ("ocr", "asr")
    assert "ocr" in snapshot["full_multimodal"]["usable_channels"]


def test_colab_source_bundle_is_allowlisted_and_hashed(tmp_path: Path) -> None:
    for directory in ("src", "scripts", "config", "checksums", "notebooks"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "runner.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "config" / "settings.yaml").write_text("profile: test\n", encoding="utf-8")
    (tmp_path / "checksums" / "assets.sha256").write_text("abc  asset\n", encoding="ascii")
    (tmp_path / "notebooks" / "launcher.ipynb").write_text("{}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "config" / ".env").write_text("SECRET=never\n", encoding="utf-8")
    (tmp_path / "scripts" / "__pycache__").mkdir()
    (tmp_path / "scripts" / "__pycache__" / "runner.pyc").write_bytes(b"compiled")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "corpus.zip").write_bytes(b"corpus")

    output = tmp_path / "publish" / "aic2026-colab-source.zip"
    report = build_colab_source_bundle(tmp_path, output)

    assert report["bundle"] == str(output.resolve())
    assert len(str(report["sha256"])) == 64
    assert (
        output.with_suffix(".zip.sha256")
        .read_text(encoding="ascii")
        .startswith(str(report["sha256"]))
    )
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "src/module.py" in names
    assert "scripts/runner.py" in names
    assert "config/.env" not in names
    assert not any("__pycache__" in name for name in names)
    assert not any(name.startswith("data/") for name in names)


def test_colab_notebook_enforces_streaming_full_b01_contract() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "aic2026_cloud_feature_extraction.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook.get("cells", ())
        if cell.get("cell_type") == "code"
    )
    required = (
        "SOURCE_BUNDLE",
        "T4/A100 required",
        "EXPECTED_VIDEOS = 873",
        "EXPECTED_MAPPING_ROWS = 177_321",
        "EXPECTED_CANONICAL_FRAMES = 176_707",
        "--expected-videos",
        "--expected-frames",
        "--resume",
        "--progress-file",
        "--scratch-dir",
        "SCRATCH_ROOT = Path('/content/aic2026-scratch')",
        "--finalize-only",
        "ZIP_STORED",
        "bundle_sha256",
    )
    for token in required:
        assert token in source
    forbidden = (
        "extractall(",
        ".extract(",
        "v_stream.read(",
        "np.vstack(",
        "rapidocr_onnxruntime",
        "ViT-B-16-SigLIP",
        'WhisperModel("base"',
        "CloudArtifactProducer(",
    )
    for token in forbidden:
        assert token not in source


def test_partition_worker_notebook_is_resumable_and_never_finalizes() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "aic2026_cloud_partition_worker.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook.get("cells", ())
        if cell.get("cell_type") == "code"
    )
    for token in (
        "PHYSICAL_WORKER_ID = 0",
        "PHYSICAL_WORKER_COUNT = 4",
        "WORKER_COUNT = 2",
        "WORKER_SLOT = PHYSICAL_WORKER_ID % WORKER_COUNT",
        "assigned = logical_assigned[EXECUTOR_SLOT::EXECUTOR_COUNT]",
        "--partition-manifest",
        "--partition-id",
        "--worker-slot",
        "--worker-count",
        "--resume",
        "SCRATCH_ROOT = Path('/content/aic2026-scratch')",
        "Source fingerprint mismatch",
        "schema == '3'",
    ):
        assert token in source
    for token in ("--finalize", "extractall(", "np.vstack(", "v_stream.read("):
        assert token not in source
