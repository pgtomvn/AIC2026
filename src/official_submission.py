"""Evidence-safe CSV validation for versioned competition adapters.

The organizer Colab does not define an AIC 2026 submission contract.  This
module therefore exposes historical adapters by explicit names and refuses the
reserved ``official_2026`` identifier until an immutable rule pack exists.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from src.contracts import CompetitionSpec, QACompetitionSpec, TRAKECompetitionSpec
from src.qa_submission import validate_qa_submission
from src.submission import validate_submission
from src.trake_submission import validate_trake_submission

SubmissionTask = Literal["kis", "qa", "trake"]

OFFICIAL_2026_UNAVAILABLE = "official_2026_spec_unavailable"
SUPPORTED_SPECS = frozenset(
    {
        "historical_2025",
        "historical_qa_numeric",
        "historical_2025_provisional_trake",
    }
)


class OfficialSubmissionValidationError(ValueError):
    """A stable-code validation failure suitable for CLI/API boundaries."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ParsedSubmission:
    """Canonical CSV rows plus byte-format evidence."""

    task: SubmissionTask
    spec_id: str
    rows: tuple[tuple[str, ...], ...]
    byte_count: int
    line_count: int
    stage_count: int | None = None


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


def parse_submission_csv(
    path: str | Path,
    *,
    task: SubmissionTask,
    spec_id: str,
) -> ParsedSubmission:
    """Read a canonical UTF-8/LF/no-header historical submission."""

    if task not in {"kis", "qa", "trake"}:
        raise OfficialSubmissionValidationError("submission_task_invalid", "unknown task")
    _spec_for(task, spec_id)
    source = Path(path)
    if source.suffix.casefold() != ".csv" or source.name != source.name.strip():
        raise OfficialSubmissionValidationError(
            "submission_filename_invalid", "submission filename must be a safe .csv name"
        )
    if task == "trake" and not source.name.endswith("-trake-submission.csv"):
        raise OfficialSubmissionValidationError(
            "submission_filename_invalid",
            "TRAKE filename must end with -trake-submission.csv",
        )
    data = source.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise OfficialSubmissionValidationError(
            "submission_bom_forbidden", "UTF-8 BOM is not allowed"
        )
    if b"\r" in data:
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
    try:
        rows = tuple(tuple(row) for row in csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise OfficialSubmissionValidationError(
            "submission_csv_invalid", "submission is not canonical CSV"
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
        raise OfficialSubmissionValidationError(
            "submission_row_width_invalid", "submission rows have inconsistent width"
        )
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
    "prediction_rows",
    "validate_parsed_submission",
]
