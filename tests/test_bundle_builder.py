from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone

import pytest

from src.export.bundle_builder import BundleBuildError, build_codabench_bundle
from src.export.preflight import (
    BundleEntry,
    BundleValidationContext,
    BundleValidationError,
    validate_bundle_entries,
)


def _entry(task: str, query_id: str, data: bytes, *, stages: int | None = None) -> BundleEntry:
    return BundleEntry(
        task=task,  # type: ignore[arg-type]
        query_id=query_id,
        csv_bytes=data,
        batch_id="batch",
        source_fingerprint=hashlib.sha256(data).hexdigest(),
        stage_count=stages,
        source_ranks=(
            tuple(range(1, len(data.decode("utf-8").splitlines()) + 1))
            if task == "kis"
            else None
        ),
    )


@pytest.fixture
def context() -> BundleValidationContext:
    frames = {
        ("L21_V001", 10): 100,
        ("L21_V001", 20): 200,
        ("L21_V001", 30): 300,
        ("L21_V002", 40): 400,
    }
    return BundleValidationContext(
        is_batch_active=lambda value: value == "batch",
        is_canonical_source_frame=lambda batch, video, frame: (
            batch == "batch" and (video, frame) in frames
        ),
        get_canonical_timestamp_ms=lambda batch, video, frame: (
            frames.get((video, frame)) if batch == "batch" else None
        ),
    )


def test_build_valid_bundle_has_exact_submission_root(tmp_path, context) -> None:
    entries = (
        _entry("kis", "1", b"L21_V001,10\n"),
        _entry("kis", "02", b"L21_V002,40\r\n"),
        _entry("qa", "3", 'L21_V001,20,"Màu đỏ, rất đẹp"\n'.encode()),
        _entry("trake", "4", b"L21_V001,10,20,30\n", stages=3),
    )
    artifact = build_codabench_bundle(
        entries,
        tmp_path,
        validation_context=context,
        now=lambda: datetime(2026, 8, 17, 12, 34, 56, 123456, tzinfo=timezone.utc),
    )

    assert artifact.path.name == "AIC2026_submission_20260817T123456123456Z.zip"
    assert artifact.manifest["file_count"] == 4
    assert artifact.manifest["validation_status"] == "valid"
    assert artifact.manifest["sha256"] == hashlib.sha256(artifact.path.read_bytes()).hexdigest()
    with zipfile.ZipFile(artifact.path) as archive:
        assert archive.namelist() == [
            "submission/",
            "submission/query-1-kis.csv",
            "submission/query-02-kis.csv",
            "submission/query-3-qa.csv",
            "submission/query-4-trake.csv",
        ]
        assert archive.testzip() is None
        assert archive.read("submission/query-02-kis.csv").endswith(b"\r\n")
    root_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["bundles"][-1]["run_id"] == artifact.manifest["run_id"]
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        (_entry("kis", "1", b"L21_V001.mp4,10\n"), "submission_video_extension_forbidden"),
        (
            _entry("qa", "2", f'L21_V001,10,"{"đ" * 101}"\n'.encode()),
            "submission_answer_too_long",
        ),
        (_entry("trake", "3", b"L21_V001,20,10\n", stages=2), "submission_frame_order_invalid"),
    ],
)
def test_invalid_entry_aborts_before_bundle(tmp_path, context, entry, code) -> None:
    with pytest.raises(BundleValidationError) as caught:
        build_codabench_bundle(
            (entry,),
            tmp_path,
            validation_context=context,
            mode="current",
        )
    assert caught.value.issues[0].code == code
    assert caught.value.issues[0].row == 1
    assert not list(tmp_path.glob("*.zip"))
    assert not (tmp_path / "manifest.json").exists()


def test_full_bundle_requires_all_workflows(context) -> None:
    with pytest.raises(BundleValidationError) as caught:
        validate_bundle_entries(
            (_entry("kis", "1", b"L21_V001,10\n"),),
            context=context,
            mode="all",
        )
    assert caught.value.issues[0].code == "bundle_workflow_missing"


def test_current_bundle_requires_exactly_one_entry(context) -> None:
    with pytest.raises(BundleValidationError) as caught:
        validate_bundle_entries(
            (
                _entry("kis", "1", b"L21_V001,10\n"),
                _entry("qa", "2", b'L21_V001,20,"yes"\n'),
            ),
            context=context,
            mode="current",
        )
    assert caught.value.issues[0].code == "bundle_current_entry_count_invalid"


def test_bundle_rejects_duplicate_query_ids_across_tasks(context) -> None:
    with pytest.raises(BundleValidationError) as caught:
        validate_bundle_entries(
            (
                _entry("kis", "1", b"L21_V001,10\n"),
                _entry("qa", "1", b'L21_V001,20,"yes"\n'),
            ),
            context=context,
            mode="all",
        )
    assert caught.value.issues[0].code == "bundle_duplicate_query_id"


def test_trake_uses_canonical_timestamp_order(context) -> None:
    reversed_time = BundleValidationContext(
        is_batch_active=context.is_batch_active,
        is_canonical_source_frame=context.is_canonical_source_frame,
        get_canonical_timestamp_ms=lambda _batch, _video, frame: {10: 200, 20: 100}[frame],
    )
    with pytest.raises(BundleValidationError) as caught:
        validate_bundle_entries(
            (_entry("trake", "4", b"L21_V001,10,20\n", stages=2),),
            context=reversed_time,
            mode="current",
        )
    assert caught.value.issues[0].code == "submission_timestamp_order_invalid"


def test_kis_rejects_noncontiguous_source_rank_snapshot(context) -> None:
    data = b"L21_V001,10\nL21_V002,40\n"
    entry = BundleEntry(
        task="kis",
        query_id="7",
        csv_bytes=data,
        batch_id="batch",
        source_fingerprint=hashlib.sha256(data).hexdigest(),
        source_ranks=(1, 3),
    )
    with pytest.raises(BundleValidationError) as caught:
        validate_bundle_entries((entry,), context=context, mode="current")
    assert caught.value.issues[0].code == "submission_rank_sequence_invalid"


def test_same_timestamp_never_overwrites_published_bundle(tmp_path, context) -> None:
    def fixed() -> datetime:
        return datetime(2026, 8, 17, 13, 0, 0, 1, tzinfo=timezone.utc)

    entry = _entry("kis", "8", b"L21_V001,10\n")
    first = build_codabench_bundle(
        (entry,), tmp_path, validation_context=context, mode="current", now=fixed
    )
    original = first.path.read_bytes()

    with pytest.raises(BundleBuildError, match="bundle_run_id_collision"):
        build_codabench_bundle(
            (entry,), tmp_path, validation_context=context, mode="current", now=fixed
        )

    assert first.path.read_bytes() == original
    assert not list(tmp_path.rglob("*.tmp"))


def test_preflight_reports_one_sanitized_issue_per_invalid_file(context) -> None:
    with pytest.raises(BundleValidationError) as caught:
        validate_bundle_entries(
            (
                _entry("kis", "1", b"L21_V001.mp4,10\n"),
                _entry("qa", "2", f'L21_V001,20,"{"x" * 101}"\n'.encode()),
                _entry("trake", "3", b"L21_V001,10,20\n", stages=2),
            ),
            context=context,
            mode="all",
        )
    assert [(issue.filename, issue.code) for issue in caught.value.issues] == [
        ("query-1-kis.csv", "submission_video_extension_forbidden"),
        ("query-2-qa.csv", "submission_answer_too_long"),
    ]
