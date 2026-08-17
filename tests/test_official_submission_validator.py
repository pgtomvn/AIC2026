from __future__ import annotations

import json

import pytest

from scripts.validate_official_submission import _score, main
from src.official_submission import (
    OFFICIAL_2026_UNAVAILABLE,
    OfficialSubmissionValidationError,
    parse_submission_csv,
    validate_parsed_submission,
)


def _write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    ("name", "data", "code"),
    [
        ("q.csv", b"\xef\xbb\xbfL21_V001,10\n", "submission_bom_forbidden"),
        ("q.csv", b"L21_V001,10\r\n", "submission_newline_invalid"),
        ("q.csv", b"video_name,frame_idx\n", "submission_header_forbidden"),
        ("q.csv", b"L21_V001;10\n", "submission_row_width_invalid"),
        ("q.csv", b'"L21_V001",10\n', "submission_csv_noncanonical"),
    ],
)
def test_format_failures_have_stable_codes(tmp_path, name, data, code) -> None:
    path = _write(tmp_path, name, data)
    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_csv(path, task="kis", spec_id="historical_2025")
    assert caught.value.code == code


def test_official_2026_fails_closed(tmp_path) -> None:
    path = _write(tmp_path, "q.csv", b"L21_V001,10\n")
    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_csv(path, task="kis", spec_id="official_2026")
    assert caught.value.code == OFFICIAL_2026_UNAVAILABLE


def test_kis_and_qa_validate_canonical_membership(tmp_path) -> None:
    kis = parse_submission_csv(
        _write(tmp_path, "kis.csv", b"L21_V001,10\n"),
        task="kis",
        spec_id="historical_2025",
    )
    report = validate_parsed_submission(
        kis,
        batch_id="batch",
        is_batch_active=lambda value: value == "batch",
        is_canonical_source_frame=lambda _batch, _video, frame: frame == 10,
        get_canonical_timestamp_ms=lambda _batch, _video, frame: frame * 40,
    )
    assert report.row_count == 1
    qa = parse_submission_csv(
        _write(tmp_path, "qa.csv", b"L21_V001,10,2\n"),
        task="qa",
        spec_id="historical_qa_numeric",
    )
    assert validate_parsed_submission(
        qa,
        batch_id="batch",
        is_batch_active=lambda _value: True,
        is_canonical_source_frame=lambda *_args: True,
        get_canonical_timestamp_ms=lambda *_args: 400,
    ).task == "qa"


def test_noncanonical_frame_is_rejected(tmp_path) -> None:
    parsed = parse_submission_csv(
        _write(tmp_path, "kis.csv", b"L21_V001,999999\n"),
        task="kis",
        spec_id="historical_2025",
    )
    with pytest.raises(OfficialSubmissionValidationError) as caught:
        validate_parsed_submission(
            parsed,
            batch_id="batch",
            is_batch_active=lambda _value: True,
            is_canonical_source_frame=lambda *_args: False,
            get_canonical_timestamp_ms=lambda *_args: None,
        )
    assert caught.value.code == "submission_contract_invalid"


def test_trake_uses_artifact_timestamps(tmp_path) -> None:
    parsed = parse_submission_csv(
        _write(tmp_path, "q-trake-submission.csv", b"L21_V001,10,20\n"),
        task="trake",
        spec_id="historical_2025_provisional_trake",
    )
    report = validate_parsed_submission(
        parsed,
        batch_id="batch",
        is_batch_active=lambda _value: True,
        is_canonical_source_frame=lambda *_args: True,
        get_canonical_timestamp_ms=lambda _batch, _video, frame: {10: 100, 20: 200}[frame],
    )
    assert report.stage_count == 2
    with pytest.raises(OfficialSubmissionValidationError):
        validate_parsed_submission(
            parsed,
            batch_id="batch",
            is_batch_active=lambda _value: True,
            is_canonical_source_frame=lambda *_args: True,
            get_canonical_timestamp_ms=lambda *_args: 100,
        )


def test_cli_score_requires_ground_truth(capsys) -> None:
    assert main(
        [
            "--submission",
            "missing.csv",
            "--task",
            "kis",
            "--spec",
            "historical_2025",
            "--batch-id",
            "batch",
            "--score",
        ]
    ) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "ground_truth_required_for_scoring"


def test_historical_kis_scoring_requires_explicit_ground_truth(tmp_path) -> None:
    parsed = parse_submission_csv(
        _write(tmp_path, "kis.csv", b"L21_V001,10\n"),
        task="kis",
        spec_id="historical_2025",
    )
    ground_truth = tmp_path / "ground-truth.json"
    ground_truth.write_text(
        json.dumps(
            [
                {
                    "query_id": "q-1",
                    "video_id": "L21_V001",
                    "start_frame_idx": 5,
                    "end_frame_idx": 15,
                }
            ]
        ),
        encoding="utf-8",
    )

    metrics = _score(parsed, ground_truth, "q-1")

    assert metrics["mean_top_k_score"] == 1.0
