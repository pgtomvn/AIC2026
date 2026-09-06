"""Codabench #10187 pre-submission validation with canonical frame checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

from src.codabench_query_id import (
    codabench_query_filename,
    codabench_query_id_collision_key,
    codabench_query_id_sort_key,
    normalize_codabench_query_id,
)
from src.official_submission import (
    OfficialSubmissionValidationError,
    ParsedSubmission,
    SubmissionTask,
    parse_submission_bytes,
    validate_parsed_submission,
)

CODABENCH_SPEC_ID = "codabench_10187_2025"
BundleMode = Literal["all", "current", "selection"]
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BundleEntry:
    """One immutable, already-serialized query result staged for a bundle."""

    task: SubmissionTask
    query_id: str
    csv_bytes: bytes
    batch_id: str
    source_fingerprint: str
    stage_count: int | None = None
    source_ranks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.task not in {"kis", "qa", "trake"}:
            raise ValueError("task must be kis, qa, or trake")
        normalized_query_id = normalize_codabench_query_id(self.query_id)
        object.__setattr__(self, "query_id", normalized_query_id)
        if not isinstance(self.csv_bytes, bytes):
            raise TypeError("csv_bytes must be bytes")
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        fingerprint = str(self.source_fingerprint).casefold()
        if _FINGERPRINT.fullmatch(fingerprint) is None:
            raise ValueError("source_fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "source_fingerprint", fingerprint)
        if self.task == "trake":
            if (
                isinstance(self.stage_count, bool)
                or not isinstance(self.stage_count, int)
                or not 2 <= self.stage_count <= 20
            ):
                raise ValueError("TRAKE stage_count must be between 2 and 20")
        elif self.stage_count is not None:
            raise ValueError("stage_count is valid only for TRAKE")
        if self.source_ranks is not None:
            if self.task != "kis":
                raise ValueError("source_ranks is valid only for KIS")
            if not isinstance(self.source_ranks, (list, tuple)):
                raise ValueError("source_ranks must be a sequence of integers")
            ranks = tuple(self.source_ranks)
            if any(
                isinstance(rank, bool) or not isinstance(rank, int)
                for rank in ranks
            ):
                raise ValueError("source_ranks must contain integers")
            object.__setattr__(self, "source_ranks", ranks)

    @property
    def filename(self) -> str:
        return codabench_query_filename(self.query_id, self.task)


@dataclass(frozen=True, slots=True)
class BundleValidationContext:
    """Injected trust-boundary callbacks owned by the FastAPI runtime."""

    is_batch_active: Callable[[str], bool]
    is_canonical_source_frame: Callable[[str, str, int], bool]
    get_canonical_timestamp_ms: Callable[[str, str, int], int | None]


@dataclass(frozen=True, slots=True)
class BundleValidationIssue:
    """Sanitized, actionable validation failure for one staged CSV."""

    filename: str | None
    task: str | None
    code: str
    message: str
    row: int | None = None
    line: int | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "filename": self.filename,
                "task": self.task,
                "code": self.code,
                "message": self.message,
                "row": self.row,
                "line": self.line,
                "suggestion": self.suggestion,
            }.items()
            if value is not None
        }


class BundleValidationError(ValueError):
    """Raised before any ZIP is published when one or more entries are invalid."""

    def __init__(self, issues: Sequence[BundleValidationIssue]) -> None:
        normalized = tuple(issues)
        if not normalized:
            raise ValueError("BundleValidationError requires at least one issue")
        super().__init__(normalized[0].code)
        self.issues = normalized


@dataclass(frozen=True, slots=True)
class ValidatedBundleEntry:
    entry: BundleEntry
    parsed: ParsedSubmission

    def to_manifest(self) -> dict[str, object]:
        return {
            "filename": self.entry.filename,
            "task": self.entry.task,
            "query_id": self.entry.query_id,
            "row_count": len(self.parsed.rows),
            "byte_count": len(self.entry.csv_bytes),
            "source_fingerprint": self.entry.source_fingerprint,
            "stage_count": self.entry.stage_count,
        }


def _line(parsed: ParsedSubmission, row_index: int) -> int:
    if row_index < len(parsed.physical_lines):
        return parsed.physical_lines[row_index]
    return row_index + 1


def _issue(
    entry: BundleEntry | None,
    code: str,
    message: str,
    *,
    row: int | None = None,
    line: int | None = None,
    suggestion: str | None = None,
) -> BundleValidationIssue:
    return BundleValidationIssue(
        filename=None if entry is None else entry.filename,
        task=None if entry is None else entry.task,
        code=code,
        message=message,
        row=row,
        line=line,
        suggestion=suggestion,
    )


def _precheck_rows(
    entry: BundleEntry,
    parsed: ParsedSubmission,
    context: BundleValidationContext,
) -> None:
    if not context.is_batch_active(entry.batch_id):
        raise BundleValidationError(
            (_issue(entry, "submission_batch_inactive", "selected batch is not active"),)
        )
    if entry.task == "trake" and parsed.stage_count != entry.stage_count:
        raise BundleValidationError(
            (
                _issue(
                    entry,
                    "submission_stage_count_mismatch",
                    "TRAKE frame count does not match the requested event count",
                    row=1,
                    line=_line(parsed, 0),
                    suggestion=f"Provide exactly {entry.stage_count} frame IDs per row",
                ),
            )
        )
    if entry.task == "kis" and entry.source_ranks is not None:
        expected_ranks = tuple(range(1, len(parsed.rows) + 1))
        if entry.source_ranks != expected_ranks:
            raise BundleValidationError(
                (
                    _issue(
                        entry,
                        "submission_rank_sequence_invalid",
                        "KIS source ranks must be contiguous from 1 in CSV row order",
                        suggestion="Rebuild the immutable export snapshot before staging",
                    ),
                )
            )

    seen_kis: set[tuple[str, int]] = set()
    seen_qa: set[tuple[str, int, str]] = set()
    seen_trake_videos: set[str] = set()
    for index, values in enumerate(parsed.rows):
        row_number = index + 1
        line_number = _line(parsed, index)
        video_id = values[0]
        frames = tuple(int(value) for value in (values[1:-1] if entry.task == "qa" else values[1:]))
        if entry.task == "kis":
            identity = (video_id, frames[0])
            if identity in seen_kis:
                raise BundleValidationError(
                    (
                        _issue(
                            entry,
                            "submission_duplicate_frame",
                            "canonical frame is duplicated",
                            row=row_number,
                            line=line_number,
                            suggestion="Remove the duplicate result row",
                        ),
                    )
                )
            seen_kis.add(identity)
        elif entry.task == "qa":
            identity = (video_id, frames[0], values[2])
            if identity in seen_qa:
                raise BundleValidationError(
                    (
                        _issue(
                            entry,
                            "submission_duplicate_answer",
                            "the same canonical frame and answer are duplicated",
                            row=row_number,
                            line=line_number,
                            suggestion="Remove the exact duplicate answer row",
                        ),
                    )
                )
            seen_qa.add(identity)
        elif video_id in seen_trake_videos:
            raise BundleValidationError(
                (
                    _issue(
                        entry,
                        "submission_duplicate_video",
                        "TRAKE permits only one alignment per video",
                        row=row_number,
                        line=line_number,
                    ),
                )
            )
        else:
            seen_trake_videos.add(video_id)

        for frame in frames:
            if not context.is_canonical_source_frame(entry.batch_id, video_id, frame):
                raise BundleValidationError(
                    (
                        _issue(
                            entry,
                            "submission_frame_not_canonical",
                            f"{video_id}:{frame} is not a canonical source frame",
                            row=row_number,
                            line=line_number,
                            suggestion="Select a persisted canonical source frame",
                        ),
                    )
                )
        if entry.task != "trake":
            continue
        if any(left >= right for left, right in zip(frames, frames[1:])):
            raise BundleValidationError(
                (
                    _issue(
                        entry,
                        "submission_frame_order_invalid",
                        "TRAKE frame IDs must be strictly increasing",
                        row=row_number,
                        line=line_number,
                    ),
                )
            )
        timestamps: list[int] = []
        for frame in frames:
            try:
                timestamp = context.get_canonical_timestamp_ms(entry.batch_id, video_id, frame)
            except Exception:
                timestamp = None
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                raise BundleValidationError(
                    (
                        _issue(
                            entry,
                            "submission_timestamp_unavailable",
                            "canonical timestamp could not be verified",
                            row=row_number,
                            line=line_number,
                        ),
                    )
                )
            timestamps.append(timestamp)
        if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
            raise BundleValidationError(
                (
                    _issue(
                        entry,
                        "submission_timestamp_order_invalid",
                        "TRAKE canonical timestamps must be strictly increasing",
                        row=row_number,
                        line=line_number,
                    ),
                )
            )


def validate_bundle_entries(
    entries: Sequence[BundleEntry],
    *,
    context: BundleValidationContext,
    mode: BundleMode = "all",
    require_qa_answer_quotes: bool = False,
) -> tuple[ValidatedBundleEntry, ...]:
    """Validate all staged entries before filesystem publication begins."""

    if mode not in {"all", "current", "selection"}:
        raise BundleValidationError(
            (
                _issue(
                    None,
                    "bundle_mode_invalid",
                    "mode must be all, current, or selection",
                ),
            )
        )
    staged = tuple(entries)
    if not staged:
        raise BundleValidationError(
            (_issue(None, "bundle_empty", "bundle requires at least one CSV"),)
        )
    if mode == "current" and len(staged) != 1:
        raise BundleValidationError(
            (
                _issue(
                    None,
                    "bundle_current_entry_count_invalid",
                    "current-workflow bundle requires exactly one CSV",
                ),
            )
        )
    batch_ids = {entry.batch_id for entry in staged}
    if len(batch_ids) != 1:
        raise BundleValidationError(
            (_issue(None, "bundle_mixed_batch", "bundle entries cannot mix batches"),)
        )
    filenames = [entry.filename.casefold() for entry in staged]
    if len(filenames) != len(set(filenames)):
        raise BundleValidationError(
            (_issue(None, "bundle_duplicate_filename", "bundle filenames must be unique"),)
        )
    query_ids = [codabench_query_id_collision_key(entry.query_id) for entry in staged]
    if len(query_ids) != len(set(query_ids)):
        raise BundleValidationError(
            (_issue(None, "bundle_duplicate_query_id", "query IDs must be unique"),)
        )
    tasks = {entry.task for entry in staged}
    if mode == "all" and tasks != {"kis", "qa", "trake"}:
        missing = ", ".join(sorted({"kis", "qa", "trake"} - tasks))
        raise BundleValidationError(
            (
                _issue(
                    None,
                    "bundle_workflow_missing",
                    f"full bundle is missing required workflows: {missing}",
                    suggestion="Stage at least one KIS, QA, and TRAKE CSV",
                ),
            )
        )

    validated: list[ValidatedBundleEntry] = []
    issues: list[BundleValidationIssue] = []
    for entry in sorted(
        staged,
        key=lambda item: (codabench_query_id_sort_key(item.query_id), item.task),
    ):
        try:
            parsed = parse_submission_bytes(
                entry.csv_bytes,
                filename=entry.filename,
                task=entry.task,
                spec_id=CODABENCH_SPEC_ID,
                require_qa_answer_quotes=(
                    require_qa_answer_quotes and entry.task == "qa"
                ),
            )
            _precheck_rows(entry, parsed, context)
            validate_parsed_submission(
                parsed,
                batch_id=entry.batch_id,
                is_batch_active=context.is_batch_active,
                is_canonical_source_frame=context.is_canonical_source_frame,
                get_canonical_timestamp_ms=context.get_canonical_timestamp_ms,
            )
        except BundleValidationError as error:
            issues.extend(error.issues)
            continue
        except OfficialSubmissionValidationError as exc:
            issues.append(
                _issue(
                    entry,
                    exc.code,
                    str(exc),
                    row=exc.row,
                    line=exc.line,
                    suggestion=exc.suggestion,
                )
            )
            continue
        except (TypeError, ValueError):
            issues.append(
                _issue(
                    entry,
                    "submission_contract_invalid",
                    "submission violates its task contract",
                )
            )
            continue
        validated.append(ValidatedBundleEntry(entry=entry, parsed=parsed))
    if issues:
        raise BundleValidationError(issues)
    return tuple(validated)


def entry_from_payload(value: Mapping[str, object]) -> BundleEntry:
    """Build a strict entry from an already-decoded API mapping."""

    return BundleEntry(
        task=str(value.get("task", "")),  # type: ignore[arg-type]
        query_id=value.get("query_id", ""),  # type: ignore[arg-type]
        csv_bytes=value.get("csv_bytes", b""),  # type: ignore[arg-type]
        batch_id=str(value.get("batch_id", "")),
        source_fingerprint=str(value.get("source_fingerprint", "")),
        stage_count=value.get("stage_count"),  # type: ignore[arg-type]
        source_ranks=value.get("source_ranks"),  # type: ignore[arg-type]
    )


__all__ = [
    "BundleEntry",
    "BundleMode",
    "BundleValidationContext",
    "BundleValidationError",
    "BundleValidationIssue",
    "CODABENCH_SPEC_ID",
    "ValidatedBundleEntry",
    "entry_from_payload",
    "validate_bundle_entries",
]
