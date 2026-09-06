from __future__ import annotations

import csv
import io

import pytest

from src.contracts import QACompetitionSpec
from src.official_submission import (
    OfficialSubmissionValidationError,
    parse_submission_bytes,
)
from src.qa_submission import serialize_qa_submission
from src.submission import (
    export_qa_csv,
    format_qa_submission_row,
    serialize_submission,
)


def test_format_qa_row_uses_minimal_quoting_and_preserves_answer_whitespace() -> None:
    assert format_qa_submission_row(" L01_V028 ", " 3450 ", "5") == ("L01_V028,3450,5")
    assert format_qa_submission_row("L01_V028", 3450, "  Năm người  ") == (
        "L01_V028,3450,  Năm người  "
    )
    assert (
        format_qa_submission_row("L01_V028", 3450, "Có 3 người, bao gồm nam và nữ")
        == 'L01_V028,3450,"Có 3 người, bao gồm nam và nữ"'
    )
    assert (
        format_qa_submission_row("L01_V028", 3450, 'Anh ấy nói "Xin chào"')
        == 'L01_V028,3450,"Anh ấy nói ""Xin chào"""'
    )
    assert format_qa_submission_row("L01_V028", 3450, "Dòng 1\nDòng 2") == (
        'L01_V028,3450,"Dòng 1\nDòng 2"'
    )
    assert format_qa_submission_row("L01_V028", 3450, "Dòng 1\rDòng 2") == (
        'L01_V028,3450,"Dòng 1\rDòng 2"'
    )


def test_export_qa_csv_uses_lf_and_round_trips_logical_records() -> None:
    rows = [
        ("L01_V028", 3450, "5"),
        ("L02_V011", "1200", "Năm người, gồm nam và nữ"),
        ("L03_V005", 2800, 'Dòng 1\nAnh ấy nói "xin chào"'),
    ]
    text = export_qa_csv(rows)

    assert text.endswith("\n")
    assert not text.startswith("\ufeff")
    assert list(csv.reader(io.StringIO(text, newline=""), strict=True)) == [
        [str(value) for value in row] for row in rows
    ]


def test_codabench_parser_accepts_minimal_and_multiline_qa_records() -> None:
    data = export_qa_csv(
        [
            ("L01_V028", 3450, "5"),
            ("L02_V011", 1200, "Dòng 1\nDòng 2"),
            ("L03_V005", 2800, "Dòng 1\rDòng 2"),
        ]
    ).encode("utf-8")

    parsed = parse_submission_bytes(
        data,
        filename="query-p1-15-qa.csv",
        task="qa",
        spec_id="codabench_10187_2025",
    )
    assert parsed.rows[0] == ("L01_V028", "3450", "5")
    assert parsed.rows[1][2] == "Dòng 1\nDòng 2"
    assert parsed.rows[2][2] == "Dòng 1\rDòng 2"

    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_bytes(
            data,
            filename="query-p1-15-qa.csv",
            task="qa",
            spec_id="codabench_10187_2025",
            require_qa_answer_quotes=True,
        )
    assert caught.value.code == "submission_qa_answer_quotes_required"


def test_codabench_parser_rejects_unescaped_quote_in_unquoted_answer() -> None:
    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_bytes(
            b'L01_V028,3450,bad"quote\n',
            filename="query-p1-15-qa.csv",
            task="qa",
            spec_id="codabench_10187_2025",
        )
    assert caught.value.code == "submission_csv_invalid"


def test_canonical_qa_serializer_normalizes_then_uses_minimal_quoting() -> None:
    data = serialize_qa_submission(
        (
            ("L01_V028", 3450, "  Năm người  "),
            ("L02_V011", 1200, "Có nam, nữ"),
        ),
        spec=QACompetitionSpec.conservative_general_text(),
        batch_id="batch",
        is_batch_active=lambda value: value == "batch",
        is_canonical_source_frame=lambda *_identity: True,
    )
    assert data == ('L01_V028,3450,năm người\nL02_V011,1200,"có nam, nữ"\n').encode("utf-8")


def test_submission_package_preserves_legacy_public_imports() -> None:
    assert callable(serialize_submission)
