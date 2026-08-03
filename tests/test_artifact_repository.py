"""CPU-only tests for canonical frame and text artifact storage."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from src.contracts import FrameRecord, RefinedFrameRecord, TextSegment
from src.ingest.artifacts import (
    ArtifactConflictError,
    ArtifactRepository,
    RepositoryNotInitialized,
)
from src.submission import FrameIndexEvidence, export_submission


def _frame(
    source_frame_idx: int,
    timestamp_ms: int,
    *,
    batch_id: str = "b2",
    video_id: str = "L99_V001",
    keyframe_ordinal: int | None = None,
    extractor_version: str = "extract:v1",
) -> FrameRecord:
    return FrameRecord(
        batch_id=batch_id,
        video_id=video_id,
        keyframe_ordinal=(source_frame_idx + 1 if keyframe_ordinal is None else keyframe_ordinal),
        source_frame_idx=source_frame_idx,
        pts=timestamp_ms * 90,
        time_base="1/90000",
        timestamp_ms=timestamp_ms,
        fps_num=25,
        fps_den=1,
        shot_id=f"shot-{timestamp_ms // 2_000}",
        sampling_reasons=("scene",),
        image_uri=f"frames://{video_id}/{source_frame_idx}.webp",
        video_uri=f"videos://{video_id}.mp4",
        source_sha256="a" * 64,
        extractor_version=extractor_version,
    )


def _segment(
    segment_id: str = "asr-1",
    *,
    batch_id: str = "b2",
    video_id: str = "L99_V001",
    kind: str = "asr",
    start_ms: int = 1_300,
    end_ms: int = 1_700,
    model_version: str = "whisper:v1",
) -> TextSegment:
    return TextSegment(
        segment_id=segment_id,
        batch_id=batch_id,
        video_id=video_id,
        kind=kind,
        start_ms=start_ms,
        end_ms=end_ms,
        source_frame_idx=None,
        text_raw="Cửa hàng màu đỏ",
        text_nfc="Cửa hàng màu đỏ",
        text_accent_folded="cua hang mau do",
        confidence=0.91,
        model_version=model_version,
        polygons=(((1, 2), (3, 2), (3, 4), (1, 4)),) if kind == "ocr" else (),
    )


def _refined(
    source_frame_idx: int = 26,
    *,
    coarse_source_frame_idx: int = 25,
    batch_id: str = "b2",
    video_id: str = "L99_V001",
    source_sha256: str = "a" * 64,
) -> RefinedFrameRecord:
    return RefinedFrameRecord(
        batch_id=batch_id,
        video_id=video_id,
        source_frame_idx=source_frame_idx,
        coarse_source_frame_idx=coarse_source_frame_idx,
        pts=93_600,
        time_base="1/90000",
        timestamp_ms=1_040,
        fps_num=25,
        fps_den=1,
        shot_id="shot-0",
        image_uri=f"refined/{video_id}/{source_frame_idx}.webp",
        video_uri=f"videos://{video_id}.mp4",
        source_sha256=source_sha256,
        image_sha256="b" * 64,
        image_width=512,
        image_height=288,
        image_bytes=12_345,
        refiner_fingerprint="raw-refiner:v2",
    )


def _repository(tmp_path: Path) -> ArtifactRepository:
    repository = ArtifactRepository(tmp_path / "state" / "artifacts.sqlite3")
    repository.initialize()
    return repository


def test_constructor_is_side_effect_free_and_initialize_enables_wal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "nested" / "artifacts.sqlite3"
    repository = ArtifactRepository(database)

    assert not database.parent.exists()
    with pytest.raises(RepositoryNotInitialized):
        repository.get_frame("b2", "L99_V001", 0)

    repository.initialize()
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
        "canonical_frames",
        "canonical_frame_mappings",
        "canonical_text_segments",
        "raw_refined_frames",
    } <= tables


def test_frame_upsert_is_idempotent_and_rejects_identity_conflicts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    frame = _frame(25, 1_000, keyframe_ordinal=1)

    first = repository.upsert_frame(frame, fingerprint="pipeline:v1", index_ready=False)
    duplicate = repository.upsert_frame(frame, fingerprint="pipeline:v1", index_ready=False)
    promoted = repository.upsert_frame(frame, fingerprint="pipeline:v1", index_ready=True)

    assert first.inserted == 1
    assert duplicate.unchanged == 1
    assert promoted.readiness_changed == 1
    assert repository.get_frame("b2", "L99_V001", 25) == frame

    with pytest.raises(ArtifactConflictError, match="processing fingerprint"):
        repository.upsert_frame(frame, fingerprint="pipeline:v2")
    with pytest.raises(ArtifactConflictError, match="canonical content"):
        repository.upsert_frame(replace(frame, shot_id="different-shot"), fingerprint="pipeline:v1")
    with pytest.raises(ArtifactConflictError, match="ordinal"):
        repository.upsert_frame(_frame(30, 1_200, keyframe_ordinal=1), fingerprint="pipeline:v1")


def test_schema_v1_migrates_additively_without_changing_canonical_counts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.upsert_frame(_frame(25, 1_000, keyframe_ordinal=1))
    before = repository.batch_counts("b2")
    with sqlite3.connect(repository.path) as connection:
        connection.execute("DROP TABLE raw_refined_frames")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    migrated = ArtifactRepository(repository.path)
    migrated.initialize()

    assert migrated.batch_counts("b2") == before
    assert migrated.get_frame("b2", "L99_V001", 25) is not None
    assert migrated.refined_frame_count("b2") == 0
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_refined_frames_are_durable_but_excluded_from_canonical_counts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    anchor = _frame(25, 1_000, keyframe_ordinal=1)
    repository.upsert_frame(anchor)
    before = repository.batch_counts("b2")
    record = _refined()

    inserted = repository.upsert_refined_frame(record)
    duplicate = repository.upsert_refined_frame(record)

    assert inserted.inserted == 1
    assert duplicate.unchanged == 1
    assert repository.get_frame("b2", "L99_V001", 26) is None
    assert repository.get_refined_frame("b2", "L99_V001", 26) == record
    assert repository.get_source_frame("b2", "L99_V001", 26) == record
    assert repository.list_frames("b2", "L99_V001") == [anchor]
    assert repository.batch_counts("b2") == before
    assert repository.refined_frame_count("b2") == 1
    assert repository.refined_frame_usage() == (1, record.image_bytes)
    assert repository.frame_index_mappings("b2") == {"L99_V001": {1: 25}}

    with pytest.raises(ArtifactConflictError, match="different content"):
        repository.upsert_refined_frame(replace(record, timestamp_ms=1_041))
    with pytest.raises(ArtifactConflictError, match="missing canonical"):
        repository.upsert_refined_frame(_refined(source_frame_idx=50, coarse_source_frame_idx=49))
    with pytest.raises(ArtifactConflictError, match="source digest"):
        repository.upsert_refined_frame(_refined(source_frame_idx=50, source_sha256="c" * 64))


def test_refined_lookup_is_candidate_scoped_and_anchor_delete_cascades(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    anchor = _frame(25, 1_000, keyframe_ordinal=1)
    repository.upsert_frame(anchor)
    repository.upsert_refined_frame(_refined())

    identities = [("L99_V001", value) for value in range(500)]
    assert repository.existing_source_frames("b2", identities) == frozenset(
        {("L99_V001", 25), ("L99_V001", 26)}
    )
    assert repository.existing_source_frames("b2", identities, include_refined=False) == frozenset(
        {("L99_V001", 25)}
    )

    repository.replace_video("b2", "L99_V001", frames=())

    assert repository.refined_frame_count("b2") == 0
    assert repository.get_source_frame("b2", "L99_V001", 26) is None


def test_refined_frame_membership_survives_restart_and_is_exportable(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.upsert_frame(_frame(25, 1_000, keyframe_ordinal=1))
    repository.upsert_refined_frame(_refined())
    restarted = ArtifactRepository(repository.path)
    restarted.initialize()
    identities = [("L99_V001", 26)]
    existing = restarted.existing_source_frames("b2", identities)
    evidence = restarted.frame_index_mappings_for_sources("b2", identities)
    destination = tmp_path / "refined-submission.csv"

    export_submission(
        [{"rank": 1, "video_id": "L99_V001", "source_frame_idx": 26}],
        destination,
        batch_id="b2",
        is_batch_active=lambda batch_id: batch_id == "b2",
        is_canonical_source_frame=lambda batch, video, frame: (
            batch == "b2" and (video, frame) in existing
        ),
        frame_index_evidence=FrameIndexEvidence(evidence),
    )

    assert destination.read_bytes() == b"L99_V001,26\n"


def test_refined_identity_reuses_an_existing_retained_frame(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    anchor = _frame(25, 1_000, keyframe_ordinal=1)
    exact = _frame(26, 1_040, keyframe_ordinal=2)
    repository.upsert_frames((anchor, exact))

    result = repository.upsert_refined_frame(_refined())

    assert result.unchanged == 1
    assert repository.refined_frame_count("b2") == 0
    assert repository.get_source_frame("b2", "L99_V001", 26) == exact


def test_frame_mapping_evidence_preserves_duplicate_organizer_ordinals(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.upsert_frames(
        (
            _frame(25, 1_000, keyframe_ordinal=1),
            _frame(50, 2_000, keyframe_ordinal=3),
        ),
        fingerprint="reference:v1",
    )

    assert repository.upsert_frame_mappings("b2", "L99_V001", {1: 25, 2: 25, 3: 50}) == 3
    assert repository.frame_index_mappings("b2") == {"L99_V001": {1: 25, 2: 25, 3: 50}}
    with pytest.raises(ArtifactConflictError, match="missing source frames"):
        repository.upsert_frame_mappings("b2", "L99_V001", {4: 999})

    replacements = (
        _frame(25, 1_000, keyframe_ordinal=1, extractor_version="extract:v2"),
        _frame(50, 2_000, keyframe_ordinal=3, extractor_version="extract:v2"),
    )
    with pytest.raises(ArtifactConflictError, match="ordinal aliases"):
        repository.replace_video("b2", "L99_V001", frames=replacements)
    repository.replace_video(
        "b2",
        "L99_V001",
        frames=replacements,
        ordinal_to_source={1: 25, 2: 25, 3: 50},
    )
    assert repository.frame_index_mappings("b2") == {"L99_V001": {1: 25, 2: 25, 3: 50}}


def test_multi_frame_upsert_rolls_back_every_insert_on_conflict(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    existing = _frame(25, 1_000, keyframe_ordinal=1)
    repository.upsert_frame(existing, fingerprint="pipeline:v1")

    with pytest.raises(ArtifactConflictError):
        repository.upsert_frames(
            (
                _frame(50, 2_000, keyframe_ordinal=2),
                replace(existing, image_uri="frames://changed.webp"),
            ),
            fingerprint="pipeline:v1",
        )

    assert repository.get_frame("b2", "L99_V001", 50) is None
    assert repository.list_frames("b2", "L99_V001") == [existing]


def test_text_upsert_and_asr_midpoint_resolve_to_same_video_frame(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    earlier = _frame(25, 1_000, keyframe_ordinal=1)
    later = _frame(50, 2_000, keyframe_ordinal=2)
    other_video = _frame(
        38,
        1_500,
        video_id="L99_V002",
        keyframe_ordinal=1,
    )
    repository.upsert_frames((later, other_video, earlier))
    segment = _segment()

    first = repository.upsert_text_segment(segment, fingerprint="asr:v1")
    duplicate = repository.upsert_text_segment(segment, fingerprint="asr:v1")

    assert first.inserted == 1
    assert duplicate.unchanged == 1
    assert repository.get_text_segment("b2", "L99_V001", "asr-1") == segment
    assert repository.nearest_frame_for_segment("b2", "L99_V001", "asr-1") == earlier
    assert repository.nearest_frame_for_segment("b2", "L99_V001", "missing") is None

    ocr = _segment("ocr-1", kind="ocr")
    repository.upsert_text_segment(ocr)
    with pytest.raises(ValueError, match="not an ASR"):
        repository.nearest_frame_for_segment("b2", "L99_V001", "ocr-1")
    assert (
        repository.nearest_frame_for_segment("b2", "L99_V001", "ocr-1", require_asr=False)
        == earlier
    )

    with pytest.raises(ArtifactConflictError, match="processing fingerprint"):
        repository.upsert_text_segment(segment, fingerprint="asr:v2")
    with pytest.raises(ArtifactConflictError, match="canonical content"):
        repository.upsert_text_segment(replace(segment, confidence=0.5), fingerprint="asr:v1")


def test_neighbors_lists_and_index_ready_iteration_are_stable(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    frames = (
        _frame(75, 3_000, keyframe_ordinal=3),
        _frame(25, 1_000, keyframe_ordinal=1),
        _frame(50, 2_000, keyframe_ordinal=2),
    )
    repository.upsert_frames(frames, index_ready=False)
    repository.upsert_frame(frames[1], index_ready=True)
    repository.upsert_frame(_frame(50, 2_000, video_id="L99_V002", keyframe_ordinal=1))

    assert [frame.source_frame_idx for frame in repository.list_frames("b2", "L99_V001")] == [
        25,
        50,
        75,
    ]
    assert [
        frame.source_frame_idx
        for frame in repository.neighbors("b2", "L99_V001", 2_000, radius_ms=1_000)
    ] == [25, 50, 75]
    assert [
        frame.source_frame_idx
        for frame in repository.neighbors("b2", "L99_V001", 2_500, radius_ms=2_000, limit=2)
    ] == [50, 75]
    assert list(repository.iter_index_ready_frames(batch_id="b2", video_id="L99_V001")) == [
        frames[1]
    ]


def test_batch_counts_and_readiness_include_text_only_videos(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.upsert_frames(
        (
            _frame(25, 1_000, keyframe_ordinal=1),
            _frame(50, 2_000, keyframe_ordinal=2),
        ),
        index_ready=False,
    )
    repository.upsert_text_segments(
        (
            _segment("asr-1"),
            _segment("ocr-1", kind="ocr"),
            _segment("asr-other", video_id="L99_V003"),
        ),
        index_ready=False,
    )

    assert repository.set_video_index_ready("b2", "L99_V001", True) == 4
    assert [
        segment.segment_id
        for segment in repository.iter_index_ready_text_segments(batch_id="b2", video_id="L99_V001")
    ] == ["asr-1", "ocr-1"]

    counts = repository.batch_counts("b2")
    assert counts.video_count == 2
    assert counts.frame_count == 2
    assert counts.text_segment_count == 3
    assert counts.index_ready_frame_count == 2
    assert counts.index_ready_text_segment_count == 2
    assert counts.text_segments_by_kind == {"asr": 2, "ocr": 1}


def test_replace_video_is_scoped_and_transactional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    old_frame = _frame(25, 1_000, keyframe_ordinal=1)
    old_segment = _segment("asr-old")
    repository.replace_video("b2", "L99_V001", frames=(old_frame,), text_segments=(old_segment,))

    new_frame = _frame(50, 2_000, keyframe_ordinal=1, extractor_version="extract:v2")
    new_segment = _segment("asr-new", model_version="whisper:v2")
    original_insert = ArtifactRepository._insert_text_segment

    def fail_insert(*args, **kwargs) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(ArtifactRepository, "_insert_text_segment", fail_insert)
    with pytest.raises(RuntimeError, match="simulated"):
        repository.replace_video(
            "b2", "L99_V001", frames=(new_frame,), text_segments=(new_segment,)
        )
    monkeypatch.setattr(ArtifactRepository, "_insert_text_segment", staticmethod(original_insert))

    assert repository.list_frames("b2", "L99_V001") == [old_frame]
    assert repository.get_text_segment("b2", "L99_V001", "asr-old") == old_segment

    replacement = repository.replace_video(
        "b2", "L99_V001", frames=(new_frame,), text_segments=(new_segment,)
    )
    assert (replacement.frame_count, replacement.text_segment_count) == (1, 1)
    assert repository.list_frames("b2", "L99_V001") == [new_frame]
    assert repository.get_text_segment("b2", "L99_V001", "asr-old") is None
    assert repository.get_text_segment("b2", "L99_V001", "asr-new") == new_segment

    with pytest.raises(ValueError, match="requested video"):
        repository.replace_video(
            "b2",
            "L99_V001",
            frames=(_frame(1, 0, video_id="L99_V002", keyframe_ordinal=1),),
        )
    assert repository.list_frames("b2", "L99_V001") == [new_frame]
