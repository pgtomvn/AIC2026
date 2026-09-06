from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.export.preflight import (
    BundleEntry,
    BundleValidationContext,
    BundleValidationError,
    validate_bundle_entries,
)
from src.export.staging_store import (
    StagedEntry,
    StagingRevisionConflict,
    StagingStore,
    StagingStoreError,
)
from src.official_submission import parse_submission_bytes


def _clock() -> datetime:
    return datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _entry(
    task: str,
    data: bytes,
    *,
    query_id: str = "p1-15",
    stage_count: int | None = None,
) -> BundleEntry:
    return BundleEntry(
        task=task,  # type: ignore[arg-type]
        query_id=query_id,
        csv_bytes=data,
        batch_id="batch",
        source_fingerprint=hashlib.sha256(data).hexdigest(),
        stage_count=stage_count,
    )


def _strict_validator(entry: BundleEntry) -> int:
    parsed = parse_submission_bytes(
        entry.csv_bytes,
        filename=entry.filename,
        task=entry.task,
        spec_id="codabench_10187_2025",
        require_qa_answer_quotes=entry.task == "qa",
    )
    canonical = {
        ("L01_V028", 3450),
        ("L01_V028", 3460),
        ("L01_V029", 10),
        ("L01_V029", 20),
        ("L01_V030", 30),
        ("L01_V030", 40),
        ("L01_V031", 50),
        ("L01_V031", 60),
    }
    exact_qa_rows: set[tuple[str, int, str]] = set()
    trake_videos: set[str] = set()
    for row in parsed.rows:
        if entry.task == "qa":
            identity = (row[0], int(row[1]), row[2])
            if (row[0], int(row[1])) not in canonical or identity in exact_qa_rows:
                raise ValueError("invalid QA row")
            exact_qa_rows.add(identity)
            continue
        frames = tuple(int(value) for value in row[1:])
        if row[0] in trake_videos or any((row[0], frame) not in canonical for frame in frames):
            raise ValueError("invalid TRAKE row")
        if any(left >= right for left, right in zip(frames, frames[1:])):
            raise ValueError("invalid TRAKE chronology")
        trake_videos.add(row[0])
    return len(parsed.rows)


def _seed(store: StagingStore, entry: BundleEntry) -> int:
    return store.upsert(
        StagedEntry.create(entry, row_count=1, clock=_clock),
        expected_revision=0,
    ).revision


def test_qa_append_update_delete_preserves_rank_order_and_exact_csv(tmp_path: Path) -> None:
    path = tmp_path / "active_staging.json"
    store = StagingStore(path, clock=_clock)
    filename = "query-p1-15-qa.csv"
    first = 'L01_V028,3450,"xã cam bắc"\n'.encode()
    revision = _seed(store, _entry("qa", first))

    second = 'L01_V028,3450,"cam bắc"\n'.encode()
    appended = store.append_row(
        filename,
        second,
        expected_revision=revision,
        validator=_strict_validator,
    )
    third = 'L01_V028,3450,"cam bắc xã"\n'.encode()
    appended = store.append_row(
        filename,
        third,
        expected_revision=appended.revision,
        validator=_strict_validator,
    )
    entry = appended.entries[0]
    assert entry.row_count == 3
    assert entry.bundle_entry.csv_bytes == first + second + third
    assert entry.sha256 == hashlib.sha256(first + second + third).hexdigest()
    assert entry.bundle_entry.source_fingerprint == entry.sha256

    replacement = 'L01_V028,3460,"cam hiệp bắc"\n'.encode()
    updated = store.update_row(
        filename,
        2,
        replacement,
        expected_revision=appended.revision,
        validator=_strict_validator,
    )
    assert updated.entries[0].bundle_entry.csv_bytes == first + replacement + third

    deleted = store.delete_row(
        filename,
        2,
        expected_revision=updated.revision,
        validator=_strict_validator,
    )
    assert deleted.entries[0].row_count == 2
    assert deleted.entries[0].bundle_entry.csv_bytes == first + third
    assert StagingStore(path).load().entries[0].bundle_entry.csv_bytes == first + third

    with pytest.raises(StagingRevisionConflict):
        store.append_row(
            filename,
            second,
            expected_revision=updated.revision,
            validator=_strict_validator,
        )


def test_qa_append_uses_full_preflight_and_allows_alternate_answer_same_frame(
    tmp_path: Path,
) -> None:
    store = StagingStore(tmp_path / "active_staging.json", clock=_clock)
    filename = "query-p1-15-qa.csv"
    first = 'L01_V028,3450,"xã cam bắc"\n'.encode()
    revision = _seed(store, _entry("qa", first))
    context = BundleValidationContext(
        is_batch_active=lambda batch_id: batch_id == "batch",
        is_canonical_source_frame=lambda batch_id, video_id, frame_idx: (
            batch_id == "batch" and (video_id, frame_idx) == ("L01_V028", 3450)
        ),
        get_canonical_timestamp_ms=lambda _batch, _video, _frame: 1_000,
    )

    def validate(entry: BundleEntry) -> int:
        result = validate_bundle_entries((entry,), context=context, mode="current")
        return len(result[0].parsed.rows)

    alternate = 'L01_V028,3450,"cam bắc"\n'.encode()
    snapshot = store.append_row(
        filename,
        alternate,
        expected_revision=revision,
        validator=validate,
    )
    assert snapshot.entries[0].bundle_entry.csv_bytes == first + alternate

    with pytest.raises(BundleValidationError) as duplicate:
        store.append_row(
            filename,
            alternate,
            expected_revision=snapshot.revision,
            validator=validate,
        )
    assert duplicate.value.issues[0].code == "submission_duplicate_answer"
    assert store.load().revision == snapshot.revision


def test_trake_move_update_delete_and_last_row_removes_entry(tmp_path: Path) -> None:
    store = StagingStore(tmp_path / "active_staging.json", clock=_clock)
    filename = "query-p1-4-trake.csv"
    first = b"L01_V029,10,20\n"
    revision = _seed(
        store,
        _entry("trake", first, query_id="p1-4", stage_count=2),
    )
    second = b"L01_V030,30,40\n"
    snapshot = store.append_row(
        filename,
        second,
        expected_revision=revision,
        validator=_strict_validator,
    )
    third = b"L01_V031,50,60\n"
    snapshot = store.append_row(
        filename,
        third,
        expected_revision=snapshot.revision,
        validator=_strict_validator,
    )

    moved = store.move_row(
        filename,
        3,
        1,
        expected_revision=snapshot.revision,
        validator=_strict_validator,
    )
    assert moved.entries[0].bundle_entry.csv_bytes == third + first + second

    no_op = store.move_row(
        filename,
        1,
        1,
        expected_revision=moved.revision,
        validator=_strict_validator,
    )
    assert no_op.revision == moved.revision

    deleted = store.delete_row(
        filename,
        2,
        expected_revision=moved.revision,
        validator=_strict_validator,
    )
    assert deleted.entries[0].bundle_entry.csv_bytes == third + second
    deleted = store.delete_row(
        filename,
        2,
        expected_revision=deleted.revision,
        validator=_strict_validator,
    )
    empty = store.delete_row(
        filename,
        1,
        expected_revision=deleted.revision,
        validator=_strict_validator,
    )
    assert empty.entries == ()
    assert empty.revision == deleted.revision + 1


def test_mutation_validation_failure_is_atomic_and_revision_is_unchanged(
    tmp_path: Path,
) -> None:
    store = StagingStore(tmp_path / "active_staging.json", clock=_clock)
    filename = "query-p1-15-qa.csv"
    first = 'L01_V028,3450,"xã cam bắc"\n'.encode()
    revision = _seed(store, _entry("qa", first))

    with pytest.raises(ValueError, match="invalid QA row"):
        store.append_row(
            filename,
            'L99_V999,1,"không canonical"\n'.encode(),
            expected_revision=revision,
            validator=_strict_validator,
        )
    snapshot = store.load()
    assert snapshot.revision == revision
    assert snapshot.entries[0].bundle_entry.csv_bytes == first

    with pytest.raises(StagingStoreError) as multiple:
        store.append_row(
            filename,
            first + first,
            expected_revision=revision,
            validator=_strict_validator,
        )
    assert multiple.value.code == "staging_row_invalid"
    assert store.load().revision == revision


def test_mutation_rejects_invalid_rank_kis_and_more_than_one_hundred_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active_staging.json"
    store = StagingStore(path, clock=_clock)
    qa_rows = b"".join(f'L01_V028,3450,"answer {rank}"\n'.encode() for rank in range(1, 101))
    entry = _entry("qa", qa_rows)
    revision = store.upsert(
        StagedEntry.create(entry, row_count=100, clock=_clock),
        expected_revision=0,
    ).revision
    with pytest.raises(StagingStoreError) as capped:
        store.append_row(
            entry.filename,
            'L01_V028,3450,"overflow"\n'.encode(),
            expected_revision=revision,
            validator=lambda _entry: 101,
        )
    assert capped.value.code == "staging_row_limit_exceeded"
    assert store.load().revision == revision

    with pytest.raises(StagingStoreError) as rank_error:
        store.delete_row(
            entry.filename,
            0,
            expected_revision=revision,
            validator=_strict_validator,
        )
    assert rank_error.value.code == "staging_rank_invalid"

    kis_store = StagingStore(tmp_path / "kis.json", clock=_clock)
    kis = _entry("kis", b"L01_V028,3450\n", query_id="p1-1")
    kis_revision = _seed(kis_store, kis)
    with pytest.raises(StagingStoreError) as unsupported:
        kis_store.mutate_rows(
            kis.filename,
            action="delete",
            expected_revision=kis_revision,
            validator=lambda _entry: 0,
            rank=1,
        )
    assert unsupported.value.code == "staging_row_mutation_unsupported"

    inconsistent = StagedEntry(
        bundle_entry=kis,
        row_count=2,
        sha256=hashlib.sha256(kis.csv_bytes).hexdigest(),
        updated_at_utc="2026-08-20T12:00:00Z",
    )
    with pytest.raises(StagingStoreError) as corrupt_entry:
        StagingStore(tmp_path / "invalid.json").upsert(
            inconsistent,
            expected_revision=0,
        )
    assert corrupt_entry.value.code == "staging_entry_invalid"
