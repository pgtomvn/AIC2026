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


def test_codabench_10187_accepts_lf_or_crlf_and_official_qa_quoting(tmp_path) -> None:
    lf = parse_submission_csv(
        _write(tmp_path, "query-1-kis.csv", b"L21_V001,10\n"),
        task="kis",
        spec_id="codabench_10187_2025",
    )
    crlf = parse_submission_csv(
        _write(tmp_path, "query-2-kis.csv", b"L21_V001,10\r\n"),
        task="kis",
        spec_id="codabench_10187_2025",
    )
    qa = parse_submission_csv(
        _write(
            tmp_path,
            "query-3-qa.csv",
            'L21_V001,10,"Có 3 người, gồm nam và nữ"\n'.encode("utf-8"),
        ),
        task="qa",
        spec_id="codabench_10187_2025",
    )
    assert lf.rows == crlf.rows == (("L21_V001", "10"),)
    assert qa.rows[0][2] == "Có 3 người, gồm nam và nữ"


def test_codabench_10187_rejects_wrong_filename_extension_and_long_answer(tmp_path) -> None:
    with pytest.raises(OfficialSubmissionValidationError) as wrong_name:
        parse_submission_csv(
            _write(tmp_path, "kis.csv", b"L21_V001,10\n"),
            task="kis",
            spec_id="codabench_10187_2025",
        )
    assert wrong_name.value.code == "submission_filename_invalid"

    with pytest.raises(OfficialSubmissionValidationError) as extension:
        parse_submission_csv(
            _write(tmp_path, "query-1-kis.csv", b"L21_V001.mp4,10\n"),
            task="kis",
            spec_id="codabench_10187_2025",
        )
    assert extension.value.code == "submission_video_extension_forbidden"
    assert extension.value.row == 1

    with pytest.raises(OfficialSubmissionValidationError) as long_answer:
        parse_submission_csv(
            _write(tmp_path, "query-2-qa.csv", f"L21_V001,10,{'a' * 101}\n".encode()),
            task="qa",
            spec_id="codabench_10187_2025",
        )
    assert long_answer.value.code == "submission_answer_too_long"


@pytest.mark.parametrize(
    ("name", "task", "data", "code"),
    [
        ("query-1-kis.csv", "kis", b"\xef\xbb\xbfL21_V001,10\n", "submission_bom_forbidden"),
        ("query-1-kis.csv", "kis", b"video_id,frame_idx\n", "submission_header_forbidden"),
        ("query-1-kis.csv", "kis", b"L21_V001;10\n", "submission_row_width_invalid"),
        ("query-1-kis.csv", "kis", b" L21_V001,10\n", "submission_video_id_invalid"),
        ("query-1-kis.csv", "kis", b"L21_V001, 10\n", "submission_integer_invalid"),
        ("query-1-qa.csv", "qa", b"L21_V001,10,   \n", "submission_answer_empty"),
        (
            "query-1-kis.csv",
            "kis",
            b"L21_V001,10\r\nL21_V001,20\n",
            "submission_newline_invalid",
        ),
    ],
)
def test_codabench_10187_strict_format_gate(tmp_path, name, task, data, code) -> None:
    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_csv(
            _write(tmp_path, name, data),
            task=task,
            spec_id="codabench_10187_2025",
        )
    assert caught.value.code == code


def test_codabench_10187_limits_rows_and_accepts_100_unicode_codepoints(tmp_path) -> None:
    valid_answer = "đ" * 100
    parsed = parse_submission_csv(
        _write(tmp_path, "query-7-qa.csv", f"L21_V001,10,{valid_answer}\n".encode()),
        task="qa",
        spec_id="codabench_10187_2025",
    )
    assert len(parsed.rows[0][2]) == 100

    too_many = b"".join(f"L21_V001,{index}\n".encode() for index in range(101))
    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_csv(
            _write(tmp_path, "query-8-kis.csv", too_many),
            task="kis",
            spec_id="codabench_10187_2025",
        )
    assert caught.value.code == "submission_row_limit_exceeded"


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
