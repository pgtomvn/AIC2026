"""Offline tests for native source discovery, scratch use, and resumable ingestion."""

from __future__ import annotations

import hashlib
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import src.ingest.native_pipeline as native_pipeline_module
from src.extraction.media_probe import MediaProbe
from src.ingest.artifacts import ArtifactRepository
from src.ingest.batch_lock import BatchIngestLock, BatchIngestLockError
from src.ingest.ledger import JobLedger
from src.ingest.native_pipeline import (
    NativeIngestor,
    ScratchManager,
    VideoSource,
    discover_video_sources,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_video_source_discovery_handles_loose_archived_and_partial_inputs(
    tmp_path: Path,
) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    loose = incoming / "L99_V002.MP4"
    loose.write_bytes(b"loose-video")
    archive = _write_zip(
        incoming / "batch.zip",
        {
            "nested/L99_V001.mp4": b"archived-video",
            "nested/readme.txt": b"ignored",
        },
    )
    _write_zip(
        incoming / "unfinished.zip.crdownload",
        {"L99_V999.mp4": b"must-not-be-discovered"},
    )

    sources = discover_video_sources((incoming,))

    assert [source.video_id for source in sources] == ["L99_V001", "L99_V002"]
    archived, discovered_loose = sources
    assert archived.path == archive
    assert archived.member == "nested/L99_V001.mp4"
    assert archived.size_bytes == len(b"archived-video")
    assert archived.uri == f"zip://{archive.resolve()}!/nested/L99_V001.mp4"
    assert discovered_loose.path == loose
    assert discovered_loose.member is None
    assert discovered_loose.uri == str(loose.resolve())


@pytest.mark.parametrize("member", ("../escape.mp4", "C:/escape.mp4"))
def test_video_source_discovery_rejects_unsafe_archive_members(
    tmp_path: Path,
    member: str,
) -> None:
    archive = _write_zip(tmp_path / "unsafe.zip", {member: b"payload"})

    with pytest.raises(ValueError, match="unsafe archive member"):
        discover_video_sources((archive,))

    assert not (tmp_path / "escape.mp4").exists()


def test_scratch_manager_materializes_and_cleans_archive_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"bounded-video-payload"
    archive = _write_zip(tmp_path / "videos.zip", {"nested/L99_V001.mp4": payload})
    source = discover_video_sources((archive,))[0]
    scratch_root = tmp_path / "scratch"
    manager = ScratchManager(
        scratch_root,
        limit_bytes=1_024,
        reserve_bytes=2_048,
    )
    monkeypatch.setattr(manager, "_available", lambda: 8_192)

    assert not scratch_root.exists()
    with manager.materialize(source) as materialized:
        assert materialized == scratch_root / "L99_V001" / "source.mp4"
        assert materialized.read_bytes() == payload
        assert materialized.resolve().is_relative_to(scratch_root.resolve())

    assert not (scratch_root / "L99_V001").exists()


def test_scratch_manager_cleans_after_consumer_failure_and_enforces_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"video"
    archive = _write_zip(tmp_path / "videos.zip", {"L99_V001.mp4": payload})
    source = discover_video_sources((archive,))[0]
    scratch_root = tmp_path / "scratch"
    manager = ScratchManager(scratch_root, limit_bytes=100, reserve_bytes=50)
    monkeypatch.setattr(manager, "_available", lambda: 1_000)

    with pytest.raises(RuntimeError, match="simulated consumer failure"):
        with manager.materialize(source) as materialized:
            assert materialized.exists()
            raise RuntimeError("simulated consumer failure")
    assert not (scratch_root / "L99_V001").exists()

    constrained_root = tmp_path / "constrained"
    constrained = ScratchManager(constrained_root, limit_bytes=100, reserve_bytes=50)
    monkeypatch.setattr(constrained, "_available", lambda: len(payload) + 49)
    with pytest.raises(OSError, match="reserve"):
        with constrained.materialize(source):
            pytest.fail("reserve violation must fail before yielding a file")
    assert not constrained_root.exists()


def test_scratch_manager_revalidates_member_safety_before_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_zip(tmp_path / "unsafe.zip", {"C:/escape.mp4": b"payload"})
    source = VideoSource("escape", archive, "C:/escape.mp4", len(b"payload"))
    scratch_root = tmp_path / "scratch"
    manager = ScratchManager(scratch_root, limit_bytes=100, reserve_bytes=0)
    monkeypatch.setattr(manager, "_available", lambda: 1_000)

    with pytest.raises(ValueError, match="unsafe archive member"):
        with manager.materialize(source):
            pytest.fail("unsafe member must never be materialized")

    assert not (scratch_root / "escape").exists()
    assert not (tmp_path / "escape.mp4").exists()


class _FakeExtractor:
    extractor_version = "fake-extractor-v1"
    min_scene_change = 0.30
    max_gap_seconds = 2.0
    min_separation_seconds = 0.5
    resize_max_dim = 512

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.calls = 0

    def extract(
        self, video_path: str | Path, *, video_id: str | None = None
    ) -> list[dict[str, Any]]:
        self.calls += 1
        path = Path(video_path)
        resolved_video_id = video_id or path.stem
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records: list[dict[str, Any]] = []
        for ordinal, (source_idx, timestamp_ms, shot_id, reasons) in enumerate(
            (
                (0, 0, "0", ["time_zero"]),
                (10, 1_000, "1", ["scene"]),
            ),
            start=1,
        ):
            image_path = self.output_root / resolved_video_id / f"{source_idx:09d}.webp"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(f"frame-{source_idx}".encode())
            records.append(
                {
                    "batch_id": "ignored-by-ingestor",
                    "video_id": resolved_video_id,
                    "keyframe_ordinal": ordinal,
                    "source_frame_idx": source_idx,
                    "pts": source_idx,
                    "time_base": "1/10",
                    "timestamp_ms": timestamp_ms,
                    "fps_num": 10,
                    "fps_den": 1,
                    "shot_id": shot_id,
                    "sampling_reasons": reasons,
                    "image_uri": str(image_path),
                    "video_uri": str(path),
                    "source_sha256": digest,
                    "extractor_version": self.extractor_version,
                }
            )
        return records


class _FakeVisualEncoder:
    fingerprint = "fake-visual-v1"

    def __init__(self) -> None:
        self.calls = 0

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        self.calls += 1
        assert all(path.is_file() for path in paths)
        return np.asarray([[float(index), 1.0] for index in range(len(paths))], dtype=np.float32)


class _FakeVisualStore:
    collection_name = "native_test_fake_visual_v1_2d"

    def __init__(self, *, fail_upserts: int = 0) -> None:
        self.create_calls: list[bool] = []
        self.upsert_calls = 0
        self.fail_upserts = fail_upserts
        self.batch_sizes: list[int] = []
        self.points: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    def create_collection(self, *, recreate: bool = False) -> None:
        self.create_calls.append(recreate)

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        assert batch_size > 0
        self.batch_sizes.append(batch_size)
        self.upsert_calls += 1
        for identifier, vector, payload in zip(ids, vectors, payloads, strict=True):
            self.points[identifier] = (vector.copy(), dict(payload))
        if self.fail_upserts:
            self.fail_upserts -= 1
            raise RuntimeError("simulated vector-store failure")

    def count(self, filters: dict[str, Any] | None = None) -> int:
        if not filters:
            return len(self.points)
        return sum(
            1
            for _, payload in self.points.values()
            if all(payload.get(key) == value for key, value in filters.items())
        )


def _fake_probe(path: str | Path) -> MediaProbe:
    return MediaProbe(
        path=str(path),
        duration_ms=2_000,
        video_codec="fixture",
        width=32,
        height=24,
        fps_num=10,
        fps_den=1,
        time_base_num=1,
        time_base_den=10,
        variable_frame_rate=False,
        has_audio=False,
        audio_codec=None,
        probe_backend="fixture",
        frame_count=20,
    )


def _native_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_upserts: int = 0,
) -> tuple[
    NativeIngestor,
    VideoSource,
    JobLedger,
    ArtifactRepository,
    _FakeExtractor,
    _FakeVisualEncoder,
    _FakeVisualStore,
]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video_path = input_dir / "L99_V001.mp4"
    video_path.write_bytes(b"deterministic fake video source")
    source = VideoSource("L99_V001", video_path)
    ledger = JobLedger(tmp_path / "state" / "ledger.sqlite3")
    artifacts = ArtifactRepository(tmp_path / "state" / "artifacts.sqlite3")
    extractor = _FakeExtractor(tmp_path / "generated" / "frames")
    encoder = _FakeVisualEncoder()
    store = _FakeVisualStore(fail_upserts=fail_upserts)
    scratch = ScratchManager(tmp_path / "scratch", limit_bytes=1_024, reserve_bytes=0)
    manifests = tmp_path / "generated" / "manifests"
    monkeypatch.setattr(native_pipeline_module, "probe_media", _fake_probe)
    ingestor = NativeIngestor(
        batch_id="b2",
        competition="aic26",
        ledger=ledger,
        artifacts=artifacts,
        extractor=extractor,
        visual_encoder=encoder,
        visual_store=store,
        scratch=scratch,
        state_root=manifests,
    )

    # Every durable boundary remains explicit until initialize/ingest is requested.
    assert not ledger.path.exists()
    assert not artifacts.path.exists()
    assert not manifests.exists()
    assert not extractor.output_root.exists()
    assert store.create_calls == []
    return ingestor, source, ledger, artifacts, extractor, encoder, store


def test_native_ingestor_is_idempotent_across_completed_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor, source, ledger, artifacts, extractor, encoder, store = _native_fixture(
        tmp_path,
        monkeypatch,
    )
    ingestor.initialize()

    first = ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)
    frames_after_first = artifacts.list_frames("b2", "L99_V001")
    second = ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)

    assert first == {"L99_V001": "active"}
    assert second == first
    assert [frame.source_frame_idx for frame in frames_after_first] == [0, 10]
    assert artifacts.list_frames("b2", "L99_V001") == frames_after_first
    assert extractor.calls == 1
    assert encoder.calls == 1
    assert store.upsert_calls == 1
    assert store.batch_sizes == [256]
    assert len(store.points) == 2
    assert store.create_calls == [False]
    assert ledger.status_summary("b2").active_videos == 1
    assert ledger.get_stage("b2", "L99_V001", "probe")["attempts"] == 1
    assert ledger.get_stage("b2", "L99_V001", "frames")["attempts"] == 1
    assert ledger.get_stage("b2", "L99_V001", "visual")["attempts"] == 1
    assert ledger.retrieval_generation("b2") == 1
    assert (ingestor.state_root / "b2" / "L99_V001.probe.json").is_file()
    assert (ingestor.state_root / "b2" / "L99_V001.frames.json").is_file()


def test_index_batch_size_is_wired_to_vector_upsert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor, source, _ledger, _artifacts, _extractor, _encoder, store = _native_fixture(
        tmp_path,
        monkeypatch,
    )
    ingestor.index_batch_size = 1
    ingestor.initialize()

    assert ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False) == {
        "L99_V001": "active"
    }
    assert store.batch_sizes == [1]


def test_scoped_reprocess_reruns_only_requested_video_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor, source, ledger, _artifacts, extractor, encoder, _store = _native_fixture(
        tmp_path,
        monkeypatch,
    )
    ingestor.initialize()
    ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)
    ingestor.reprocess_stages = frozenset({"visual"})
    ingestor.reprocess_videos = frozenset({source.video_id})

    assert ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False) == {
        source.video_id: "active"
    }
    assert extractor.calls == 1
    assert encoder.calls == 2
    assert ledger.get_stage("b2", source.video_id, "visual")["attempts"] == 2
    assert ledger.retrieval_generation("b2") == 2


def test_archive_slot_is_released_before_consumer_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_zip(
        tmp_path / "videos.zip",
        {"L99_V001.mp4": b"one", "L99_V002.mp4": b"two"},
    )
    sources = discover_video_sources((archive,))
    manager = ScratchManager(
        tmp_path / "scratch",
        limit_bytes=10_000,
        reserve_bytes=0,
        archive_materializers=1,
    )
    monkeypatch.setattr(manager, "_available", lambda: 100_000)
    second_materialized = threading.Event()

    def consume_second() -> None:
        with manager.materialize(sources[1]) as path:
            assert path.read_bytes() == b"two"
            second_materialized.set()

    with manager.materialize(sources[0]) as first:
        thread = threading.Thread(target=consume_second)
        thread.start()
        assert first.read_bytes() == b"one"
        assert second_materialized.wait(timeout=2)
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_batch_ingest_lock_is_cross_instance_single_writer(tmp_path: Path) -> None:
    first = BatchIngestLock(tmp_path, "AIC26-B2")
    second = BatchIngestLock(tmp_path, "AIC26-B2")

    with first:
        with pytest.raises(BatchIngestLockError, match="batch_ingest_lock_unavailable"):
            with second:
                pytest.fail("a second writer must not enter")
    with second:
        assert second.path.is_file()


def test_native_ingestor_retries_failed_visual_stage_without_reextracting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor, source, ledger, artifacts, extractor, encoder, store = _native_fixture(
        tmp_path,
        monkeypatch,
        fail_upserts=1,
    )
    ingestor.initialize()

    failed = ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)

    assert failed["L99_V001"].startswith("failed: RuntimeError: simulated vector-store failure")
    assert ledger.status_summary("b2").active_videos == 0
    assert ledger.get_stage("b2", "L99_V001", "visual")["state"] == "failed"
    assert [frame.source_frame_idx for frame in artifacts.list_frames("b2", "L99_V001")] == [
        0,
        10,
    ]

    recovered = ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)

    assert recovered == {"L99_V001": "active"}
    assert extractor.calls == 1
    assert encoder.calls == 2
    assert store.upsert_calls == 2
    assert len(store.points) == 2
    visual_stage = ledger.get_stage("b2", "L99_V001", "visual")
    assert visual_stage["state"] == "completed"
    assert visual_stage["attempts"] == 2
    assert ledger.status_summary("b2").active_videos == 1


def test_completed_visual_stage_repairs_lost_points_before_reactivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor, source, ledger, _artifacts, extractor, encoder, store = _native_fixture(
        tmp_path, monkeypatch
    )
    ingestor.initialize()
    assert ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False) == {
        "L99_V001": "active"
    }
    store.points.clear()

    repaired = ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)

    assert repaired == {"L99_V001": "active"}
    assert extractor.calls == 1
    assert encoder.calls == 2
    assert len(store.points) == 2
    assert ledger.get_stage("b2", "L99_V001", "visual")["attempts"] == 2
    assert ledger.is_batch_active("b2")


def test_missing_completed_thumbnail_revokes_video_and_batch_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor, source, ledger, artifacts, _extractor, _encoder, _store = _native_fixture(
        tmp_path, monkeypatch
    )
    ingestor.initialize()
    ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)
    Path(artifacts.list_frames("b2", "L99_V001")[0].image_uri).unlink()

    result = ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=False)

    assert result["L99_V001"].startswith("failed: FileNotFoundError")
    assert ledger.status_summary("b2").active_videos == 0
    assert not ledger.is_batch_active("b2")


def test_ingestion_explicitly_hands_gpu_from_visual_models_to_asr(tmp_path: Path) -> None:
    events: list[str] = []

    class Model:
        def __init__(self, name: str) -> None:
            self.name = name

        def unload(self) -> None:
            events.append(f"{self.name}_unload")

    class Ledger:
        @staticmethod
        def activate_batch_if_ready(_batch_id: str) -> bool:
            return True

    source = VideoSource("L99_V001", tmp_path / "fixture.mp4")
    visual = Model("visual")
    accelerated = Model("accelerated")
    asr = Model("asr")
    ingestor = NativeIngestor(
        batch_id="b2",
        competition="aic26",
        ledger=Ledger(),  # type: ignore[arg-type]
        artifacts=object(),
        extractor=object(),  # type: ignore[arg-type]
        visual_encoder=visual,
        visual_store=object(),
        scratch=ScratchManager(tmp_path / "scratch", limit_bytes=100, reserve_bytes=0),
        asr_engine=asr,  # type: ignore[arg-type]
        additional_visual_channels=(("accelerated_visual", accelerated, object()),),
    )
    prepared = SimpleNamespace(source=source)
    ingestor._prepare = lambda _source: prepared  # type: ignore[method-assign]
    ingestor._index_visual = lambda _prepared: events.append("visual")  # type: ignore[method-assign]
    ingestor._index_additional_visuals = lambda _prepared: events.append(  # type: ignore[method-assign]
        "accelerated"
    )
    ingestor.enrich = lambda _prepared: events.append("enrich")  # type: ignore[method-assign]

    assert ingestor.ingest_many((source,), cpu_workers=1, queue_size=1, enrich=True) == {
        "L99_V001": "active"
    }
    assert events == [
        "asr_unload",
        "visual",
        "accelerated",
        "visual_unload",
        "accelerated_unload",
        "enrich",
        "asr_unload",
    ]


def test_enrichment_stages_are_isolated_and_index_successful_segments() -> None:
    """An OCR failure must not prevent ASR or semantic text indexing."""

    ingestor = object.__new__(NativeIngestor)
    events: list[object] = []
    asr_segment = object()

    def fail_ocr(_prepared: object) -> tuple[object, ...]:
        events.append("ocr")
        raise RuntimeError("ocr unavailable")

    def run_asr(_prepared: object) -> tuple[object, ...]:
        events.append("asr")
        return (asr_segment,)

    def persisted(_prepared: object, kind: str) -> tuple[object, ...]:
        events.append(f"persisted:{kind}")
        return ()

    def index_text(_prepared: object, segments: list[object]) -> None:
        events.append(("semantic", tuple(segments)))

    ingestor._enrich_ocr = fail_ocr  # type: ignore[method-assign]
    ingestor._enrich_asr = run_asr  # type: ignore[method-assign]
    ingestor._persisted_segments = persisted  # type: ignore[method-assign]
    ingestor._enrich_text = index_text  # type: ignore[method-assign]

    failures = NativeIngestor.enrich(ingestor, object())  # type: ignore[arg-type]

    assert failures == ("ocr",)
    assert events == [
        "ocr",
        "persisted:ocr",
        "asr",
        ("semantic", (asr_segment,)),
    ]
