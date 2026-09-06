"""Evidence-safe CSV validation for versioned competition adapters.

The organizer Colab does not define an AIC 2026 submission contract.  This
module therefore exposes historical adapters by explicit names and refuses the
reserved ``official_2026`` identifier until an immutable rule pack exists.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from src.codabench_query_id import parse_codabench_query_filename
from src.contracts import CompetitionSpec, QACompetitionSpec, TRAKECompetitionSpec
from src.qa_submission import validate_qa_submission
from src.submission import validate_submission
from src.trake_submission import validate_trake_submission

SubmissionTask = Literal["kis", "qa", "trake"]

OFFICIAL_2026_UNAVAILABLE = "official_2026_spec_unavailable"
SUPPORTED_SPECS = frozenset(
    {
        "codabench_10187_2025",
        "historical_2025",
        "historical_qa_numeric",
        "historical_2025_provisional_trake",
    }
)


class OfficialSubmissionValidationError(ValueError):
    """A stable-code validation failure suitable for CLI/API boundaries."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        line: int | None = None,
        row: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.line = line
        self.row = row
        self.suggestion = suggestion


@dataclass(frozen=True, slots=True)
class ParsedSubmission:
    """Canonical CSV rows plus byte-format evidence."""

    task: SubmissionTask
    spec_id: str
    rows: tuple[tuple[str, ...], ...]
    byte_count: int
    line_count: int
    stage_count: int | None = None
    physical_lines: tuple[int, ...] = ()
    raw_records: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SubmissionValidationReport:
    """Machine-readable result of format and artifact-membership validation."""

    task: SubmissionTask
    spec_id: str
    provenance: str
    row_count: int
    byte_count: int
    batch_id: str
    canonical_membership_checked: bool
    ground_truth_supplied: bool
    stage_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "valid",
            "task": self.task,
            "spec_id": self.spec_id,
            "provenance": self.provenance,
            "row_count": self.row_count,
            "byte_count": self.byte_count,
            "batch_id": self.batch_id,
            "canonical_membership_checked": self.canonical_membership_checked,
            "ground_truth_supplied": self.ground_truth_supplied,
            "stage_count": self.stage_count,
        }


def _spec_for(task: SubmissionTask, spec_id: str) -> object:
    if spec_id == "official_2026":
        raise OfficialSubmissionValidationError(
            OFFICIAL_2026_UNAVAILABLE,
            "the organizer has not published an AIC 2026 submission rule pack",
        )
    if spec_id == "codabench_10187_2025":
        if task == "kis":
            return CompetitionSpec.historical_2025()
        if task == "qa":
            return QACompetitionSpec.conservative_general_text()
        return TRAKECompetitionSpec.historical_2025_provisional()
    expected = {
        "kis": "historical_2025",
        "qa": "historical_qa_numeric",
        "trake": "historical_2025_provisional_trake",
    }[task]
    if spec_id != expected:
        raise OfficialSubmissionValidationError(
            "submission_spec_task_mismatch",
            f"spec {spec_id!r} is not valid for task {task!r}",
        )
    if task == "kis":
        return CompetitionSpec.historical_2025()
    if task == "qa":
        return QACompetitionSpec.historical_numeric()
    return TRAKECompetitionSpec.historical_2025_provisional()


_VIDEO_ID = re.compile(r"^L[0-9]{2}_V[0-9]{3}$")
_FRAME_INDEX = re.compile(r"^[0-9]+$")


def _raise_csv_lexical_error(
    code: str,
    message: str,
    *,
    line: int,
) -> None:
    raise OfficialSubmissionValidationError(code, message, line=max(1, line))


def _validate_codabench_csv_lexemes(text: str) -> None:
    """Validate RFC 4180 quotes and record separators."""

    in_quotes = False
    after_quote = False
    at_field_start = True
    record_newlines: set[str] = set()
    line = 1
    index = 0
    while index < len(text):
        character = text[index]
        if in_quotes:
            if character == '"':
                if index + 1 < len(text) and text[index + 1] == '"':
                    index += 2
                    continue
                in_quotes = False
                after_quote = True
            elif character == "\n":
                line += 1
            index += 1
            continue

        if after_quote:
            if character == ",":
                after_quote = False
                at_field_start = True
                index += 1
                continue
            if character == "\r":
                if index + 1 >= len(text) or text[index + 1] != "\n":
                    _raise_csv_lexical_error(
                        "submission_newline_invalid",
                        "submission contains a bare record-separator carriage return",
                        line=line,
                    )
                record_newlines.add("crlf")
                after_quote = False
                at_field_start = True
                line += 1
                index += 2
                continue
            if character == "\n":
                record_newlines.add("lf")
                after_quote = False
                at_field_start = True
                line += 1
                index += 1
                continue
            _raise_csv_lexical_error(
                "submission_csv_invalid",
                "characters after a closing quote are not valid CSV",
                line=line,
            )

        if character == '"':
            if not at_field_start:
                _raise_csv_lexical_error(
                    "submission_csv_invalid",
                    "quotes inside an unquoted field must be escaped",
                    line=line,
                )
            in_quotes = True
            at_field_start = False
            index += 1
            continue
        if character == ",":
            at_field_start = True
            index += 1
            continue
        if character == "\r":
            if index + 1 >= len(text) or text[index + 1] != "\n":
                _raise_csv_lexical_error(
                    "submission_newline_invalid",
                    "submission contains a bare record-separator carriage return",
                    line=line,
                )
            record_newlines.add("crlf")
            at_field_start = True
            line += 1
            index += 2
            continue
        if character == "\n":
            record_newlines.add("lf")
            at_field_start = True
            line += 1
            index += 1
            continue
        at_field_start = False
        index += 1

    if in_quotes:
        _raise_csv_lexical_error(
            "submission_csv_invalid",
            "submission contains an unterminated quoted field",
            line=line,
        )
    if len(record_newlines) > 1:
        _raise_csv_lexical_error(
            "submission_newline_invalid",
            "submission mixes LF and CRLF record terminators",
            line=line,
        )


def _qa_answer_was_quoted(record: str) -> bool:
    field_starts = [0]
    in_quotes = False
    index = 0
    while index < len(record):
        character = record[index]
        if character == '"':
            if in_quotes and index + 1 < len(record) and record[index + 1] == '"':
                index += 2
                continue
            in_quotes = not in_quotes
        elif character == "," and not in_quotes:
            field_starts.append(index + 1)
        index += 1
    return len(field_starts) == 3 and record[field_starts[2] :].startswith('"')


def parse_submission_bytes(
    data: bytes,
    *,
    filename: str,
    task: SubmissionTask,
    spec_id: str,
    require_qa_answer_quotes: bool = False,
) -> ParsedSubmission:
    """Parse one submission payload under an explicit versioned rule pack."""

    if task not in {"kis", "qa", "trake"}:
        raise OfficialSubmissionValidationError("submission_task_invalid", "unknown task")
    _spec_for(task, spec_id)
    source = Path(filename)
    if source.name != filename or source.suffix.casefold() != ".csv" or filename != filename.strip():
        raise OfficialSubmissionValidationError(
            "submission_filename_invalid", "submission filename must be a safe .csv name"
        )
    codabench = spec_id == "codabench_10187_2025"
    if codabench:
        try:
            _, filename_task = parse_codabench_query_filename(filename)
        except (TypeError, ValueError):
            filename_task = None
        if filename_task != task:
            raise OfficialSubmissionValidationError(
                "submission_filename_invalid",
                "filename must match query-<id>-<task>.csv",
                suggestion=f"Use query-<id>-{task}.csv",
            )
    elif task == "trake" and not source.name.endswith("-trake-submission.csv"):
        raise OfficialSubmissionValidationError(
            "submission_filename_invalid",
            "TRAKE filename must end with -trake-submission.csv",
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise OfficialSubmissionValidationError(
            "submission_bom_forbidden", "UTF-8 BOM is not allowed"
        )
    if not codabench and b"\r" in data:
        raise OfficialSubmissionValidationError(
            "submission_newline_invalid", "submission must use LF newlines"
        )
    if data and not data.endswith(b"\n"):
        raise OfficialSubmissionValidationError(
            "submission_newline_invalid", "submission must end with LF"
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OfficialSubmissionValidationError(
            "submission_encoding_invalid", "submission must be valid UTF-8"
        ) from exc
    if codabench:
        _validate_codabench_csv_lexemes(text)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        rows_list: list[tuple[str, ...]] = []
        physical_lines: list[int] = []
        for row in reader:
            rows_list.append(tuple(row))
            physical_lines.append(reader.line_num)
        rows = tuple(rows_list)
    except csv.Error as exc:
        raise OfficialSubmissionValidationError(
            "submission_csv_invalid",
            "submission is not valid CSV",
            line=max(1, reader.line_num),
        ) from exc
    if not rows or any(not row for row in rows):
        raise OfficialSubmissionValidationError(
            "submission_empty", "submission must contain non-empty rows"
        )
    header_tokens = {value.strip().casefold() for value in rows[0]}
    if header_tokens & {"video", "video_id", "video_name", "frame", "frame_idx", "answer"}:
        raise OfficialSubmissionValidationError(
            "submission_header_forbidden", "historical adapters require no header"
        )
    expected_width = 2 if task == "kis" else 3 if task == "qa" else len(rows[0])
    if task == "trake" and not 3 <= expected_width <= 21:
        raise OfficialSubmissionValidationError(
            "submission_row_width_invalid", "TRAKE rows require one video and 2-20 frames"
        )
    if any(len(row) != expected_width for row in rows):
        bad_index = next(index for index, row in enumerate(rows) if len(row) != expected_width)
        raise OfficialSubmissionValidationError(
            "submission_row_width_invalid",
            "submission rows have inconsistent width",
            row=bad_index + 1,
            line=physical_lines[bad_index],
        )
    text_lines = text.splitlines(keepends=True)
    raw_records: list[str] = []
    previous_line = 0
    for line_number in physical_lines:
        raw_records.append("".join(text_lines[previous_line:line_number]))
        previous_line = line_number
    if require_qa_answer_quotes and task == "qa":
        for index, record in enumerate(raw_records):
            if not _qa_answer_was_quoted(record):
                raise OfficialSubmissionValidationError(
                    "submission_qa_answer_quotes_required",
                    "QA answer must be enclosed in double quotes",
                    row=index + 1,
                    line=physical_lines[index],
                    suggestion='Serialize the row as video_id,frame_idx,"answer"',
                )
    if codabench:
        if len(rows) > 100:
            raise OfficialSubmissionValidationError(
                "submission_row_limit_exceeded", "submission may contain at most 100 rows"
            )
        for index, row in enumerate(rows):
            line = physical_lines[index]
            video_id = row[0]
            if video_id.casefold().endswith(".mp4"):
                raise OfficialSubmissionValidationError(
                    "submission_video_extension_forbidden",
                    "video ID must not contain the .mp4 extension",
                    row=index + 1,
                    line=line,
                    suggestion="Remove the .mp4 suffix from the video ID",
                )
            if _VIDEO_ID.fullmatch(video_id) is None:
                raise OfficialSubmissionValidationError(
                    "submission_video_id_invalid",
                    "video ID must match L00_V000 without surrounding whitespace",
                    row=index + 1,
                    line=line,
                )
            frame_values = row[1:-1] if task == "qa" else row[1:]
            if any(_FRAME_INDEX.fullmatch(value) is None for value in frame_values):
                raise OfficialSubmissionValidationError(
                    "submission_integer_invalid",
                    "frame fields must be base-10 integers without whitespace",
                    row=index + 1,
                    line=line,
                )
            if task == "qa":
                answer = row[2]
                if not answer.strip():
                    raise OfficialSubmissionValidationError(
                        "submission_answer_empty",
                        "QA answer must not be empty",
                        row=index + 1,
                        line=line,
                    )
                if len(answer) > 100:
                    raise OfficialSubmissionValidationError(
                        "submission_answer_too_long",
                        "QA answer must contain at most 100 characters",
                        row=index + 1,
                        line=line,
                        suggestion="Shorten the answer to 100 characters or fewer",
                    )
    else:
        canonical = io.StringIO(newline="")
        writer = csv.writer(canonical, lineterminator="\n")
        writer.writerows(rows)
        if canonical.getvalue().encode("utf-8") != data:
            raise OfficialSubmissionValidationError(
                "submission_csv_noncanonical",
                "delimiter, quoting, or whitespace does not match canonical CSV",
            )
    return ParsedSubmission(
        task=task,
        spec_id=spec_id,
        rows=rows,
        byte_count=len(data),
        line_count=len(rows),
        stage_count=expected_width - 1 if task == "trake" else None,
        physical_lines=tuple(physical_lines),
        raw_records=tuple(raw_records),
    )


def parse_submission_csv(
    path: str | Path,
    *,
    task: SubmissionTask,
    spec_id: str,
    require_qa_answer_quotes: bool = False,
) -> ParsedSubmission:
    """Read and parse a submission file without changing its bytes."""

    source = Path(path)
    return parse_submission_bytes(
        source.read_bytes(),
        filename=source.name,
        task=task,
        spec_id=spec_id,
        require_qa_answer_quotes=require_qa_answer_quotes,
    )


def validate_parsed_submission(
    parsed: ParsedSubmission,
    *,
    batch_id: str,
    is_batch_active: Callable[[str], bool],
    is_canonical_source_frame: Callable[[str, str, int], bool],
    get_canonical_timestamp_ms: Callable[[str, str, int], int | None],
    ground_truth_supplied: bool = False,
) -> SubmissionValidationReport:
    """Apply the existing strict exporters' trust-boundary validation."""

    active_spec = _spec_for(parsed.task, parsed.spec_id)
    items: list[dict[str, object]] = []
    try:
        for rank, row in enumerate(parsed.rows, start=1):
            if parsed.task == "kis":
                items.append(
                    {
                        "rank": rank,
                        "batch_id": batch_id,
                        "video_name": row[0],
                        "frame_idx": int(row[1]),
                    }
                )
            elif parsed.task == "qa":
                items.append(
                    {
                        "rank": rank,
                        "batch_id": batch_id,
                        "video_name": row[0],
                        "frame_idx": int(row[1]),
                        "answer": row[2],
                    }
                )
            else:
                items.append(
                    {
                        "rank": rank,
                        "batch_id": batch_id,
                        "video_name": row[0],
                        "frame_ids": [int(value) for value in row[1:]],
                    }
                )
    except ValueError as exc:
        raise OfficialSubmissionValidationError(
            "submission_integer_invalid", "frame fields must be base-10 integers"
        ) from exc

    try:
        if parsed.task == "kis":
            validate_submission(
                items,
                spec=active_spec,  # type: ignore[arg-type]
                batch_id=batch_id,
                is_batch_active=is_batch_active,
                is_canonical_source_frame=is_canonical_source_frame,
            )
        elif parsed.task == "qa":
            validate_qa_submission(
                items,
                spec=active_spec,  # type: ignore[arg-type]
                batch_id=batch_id,
                is_batch_active=is_batch_active,
                is_canonical_source_frame=is_canonical_source_frame,
            )
        else:
            validate_trake_submission(
                items,
                stage_count=parsed.stage_count,
                spec=active_spec,  # type: ignore[arg-type]
                batch_id=batch_id,
                is_batch_active=is_batch_active,
                is_canonical_source_frame=is_canonical_source_frame,
                get_canonical_timestamp_ms=get_canonical_timestamp_ms,
            )
    except (TypeError, ValueError) as exc:
        raise OfficialSubmissionValidationError(
            "submission_contract_invalid", str(exc)
        ) from exc
    return SubmissionValidationReport(
        task=parsed.task,
        spec_id=parsed.spec_id,
        provenance="historical_adapter_not_official_2026",
        row_count=len(parsed.rows),
        byte_count=parsed.byte_count,
        batch_id=batch_id,
        canonical_membership_checked=True,
        ground_truth_supplied=ground_truth_supplied,
        stage_count=parsed.stage_count,
    )


def prediction_rows(parsed: ParsedSubmission) -> Sequence[dict[str, object]]:
    """Return evaluator-friendly rows after format validation."""

    if parsed.task == "kis":
        return tuple(
            {"video_name": row[0], "frame_idx": int(row[1])} for row in parsed.rows
        )
    if parsed.task == "qa":
        return tuple(
            {"video_name": row[0], "frame_idx": int(row[1]), "answer": row[2]}
            for row in parsed.rows
        )
    return tuple(
        {"video_name": row[0], "frame_ids": [int(value) for value in row[1:]]}
        for row in parsed.rows
    )


__all__ = [
    "OFFICIAL_2026_UNAVAILABLE",
    "OfficialSubmissionValidationError",
    "ParsedSubmission",
    "SUPPORTED_SPECS",
    "SubmissionValidationReport",
    "parse_submission_csv",
    "parse_submission_bytes",
    "prediction_rows",
    "validate_parsed_submission",
]
