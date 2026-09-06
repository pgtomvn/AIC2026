from __future__ import annotations

import hashlib

import pytest

from src.codabench_query_id import (
    codabench_query_filename,
    codabench_query_id_sort_key,
    normalize_codabench_query_id,
    parse_codabench_query_filename,
)
from src.export.preflight import (
    BundleEntry,
    BundleValidationContext,
    BundleValidationError,
    validate_bundle_entries,
)
from src.official_submission import parse_submission_bytes


def _entry(task: str, query_id: str, data: bytes, *, stage_count: int | None = None):
    return BundleEntry(
        task=task,  # type: ignore[arg-type]
        query_id=query_id,
        csv_bytes=data,
        batch_id="batch",
        source_fingerprint=hashlib.sha256(data).hexdigest(),
        stage_count=stage_count,
    )


@pytest.fixture
def context() -> BundleValidationContext:
    frames = {
        ("L21_V001", 10): 100,
        ("L21_V001", 20): 200,
        ("L21_V001", 30): 300,
    }
    return BundleValidationContext(
        is_batch_active=lambda batch: batch == "batch",
        is_canonical_source_frame=lambda batch, video, frame: (
            batch == "batch" and (video, frame) in frames
        ),
        get_canonical_timestamp_ms=lambda batch, video, frame: (
            frames.get((video, frame)) if batch == "batch" else None
        ),
    )


@pytest.mark.parametrize(
    ("filename", "query_id", "task"),
    (
        ("query-p1-1-kis.csv", "p1-1", "kis"),
        ("query-p1-15-qa.csv", "p1-15", "qa"),
        ("query-p1-4-trake.csv", "p1-4", "trake"),
        ("query-round_2-kis.csv", "round_2", "kis"),
        ("query-p1-4-trake-trake.csv", "p1-4-trake", "trake"),
    ),
)
def test_alphanumeric_query_filenames_round_trip(filename, query_id, task) -> None:
    assert codabench_query_filename(query_id, task) == filename
    assert parse_codabench_query_filename(filename) == (query_id, task)


@pytest.mark.parametrize(
    "query_id",
    ("", "contains space", "p1@1", "p1!1", "p1#1", "p1.1", "p1/1", "đợt-1", "x" * 33),
)
def test_query_id_rejects_unsafe_values(query_id: str) -> None:
    with pytest.raises(ValueError):
        normalize_codabench_query_id(query_id)


@pytest.mark.parametrize("value", (None, 1, True, {}, []))
def test_query_id_requires_a_real_string(value) -> None:
    with pytest.raises(TypeError):
        normalize_codabench_query_id(value)


def test_query_id_trims_outer_space_and_preserves_case_and_leading_zero() -> None:
    assert normalize_codabench_query_id("  P1-01  ") == "P1-01"
    assert _entry("kis", "  01  ", b"L21_V001,10\n").query_id == "01"


def test_alphanumeric_entries_validate_and_sort_naturally(context) -> None:
    entries = (
        _entry("trake", "p1-15", b"L21_V001,10,20\n", stage_count=2),
        _entry("qa", "p1-2", b'L21_V001,20,"yes"\n'),
        _entry("kis", "round1_01", b"L21_V001,30\n"),
    )
    validated = validate_bundle_entries(entries, context=context, mode="all")
    assert [item.entry.query_id for item in validated] == ["p1-2", "p1-15", "round1_01"]
    assert sorted(("p1-15", "p1-2"), key=codabench_query_id_sort_key) == [
        "p1-2",
        "p1-15",
    ]


def test_case_only_query_id_collision_is_rejected(context) -> None:
    with pytest.raises(BundleValidationError) as caught:
        validate_bundle_entries(
            (
                _entry("kis", "P1-1", b"L21_V001,10\n"),
                _entry("qa", "p1-1", b'L21_V001,20,"yes"\n'),
            ),
            context=context,
            mode="all",
        )
    assert caught.value.issues[0].code == "bundle_duplicate_query_id"


def test_official_parser_accepts_alphanumeric_codabench_filename() -> None:
    parsed = parse_submission_bytes(
        b"L21_V001,10,yes\n",
        filename="query-p1-15-qa.csv",
        task="qa",
        spec_id="codabench_10187_2025",
    )
    assert parsed.rows == (("L21_V001", "10", "yes"),)
