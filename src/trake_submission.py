"""Validation and atomic CSV export for provisional TRAKE submissions."""

from __future__ import annotations

import csv
import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.contracts import TRAKECompetitionSpec, TRAKEResponseItem


class TRAKESubmissionValidationError(ValueError):
    """Raised before a malformed or ambiguous TRAKE submission is written."""


@dataclass(frozen=True, slots=True)
class TRAKEFrameIndexEvidence:
    """Known organizer ordinal-to-canonical-source mappings by video."""

    ordinal_to_source: Mapping[str, Mapping[int, int]]


@dataclass(frozen=True, slots=True)
class TRAKESubmissionRow:
    """One normalized no-header CSV row."""

    rank: int
    batch_id: str
    video_id: str
    source_frame_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ParsedAlignment:
    source_frame_indices: tuple[int, ...]
    timestamps_ms: tuple[int, ...] | None


TRAKESubmissionItem = TRAKEResponseItem | Mapping[str, object]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TRAKESubmissionValidationError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+").isdigit():
        return int(value.strip())
    raise TRAKESubmissionValidationError(f"{name} must be an integer")


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TRAKESubmissionValidationError(f"{name} must be a strict integer")
    return value


def _mapping_frames(value: Mapping[str, object]) -> _ParsedAlignment:
    if "keyframe_ordinal" in value or "keyframe_ordinals" in value:
        raise TRAKESubmissionValidationError(
            "keyframe_ordinal cannot be submitted; provide canonical source frame indices"
        )
    stage_matches = value.get("stage_matches")
    if stage_matches is not None:
        if not isinstance(stage_matches, (list, tuple)):
            raise TRAKESubmissionValidationError("stage_matches must be a sequence")
        frames: list[int] = []
        timestamps_ms: list[int] = []
        for expected_stage_index, match in enumerate(stage_matches, start=1):
            if not isinstance(match, Mapping):
                raise TRAKESubmissionValidationError("stage_matches must contain mappings")
            if "frame_id" in match or "keyframe_ordinal" in match:
                raise TRAKESubmissionValidationError(
                    "ambiguous frame identifiers are forbidden; use source_frame_idx"
                )
            if "source_frame_idx" not in match:
                raise TRAKESubmissionValidationError(
                    "each stage match requires source_frame_idx"
                )
            if "stage_index" not in match:
                raise TRAKESubmissionValidationError(
                    "each stage match requires stage_index"
                )
            if "timestamp_ms" not in match:
                raise TRAKESubmissionValidationError(
                    "each stage match requires timestamp_ms"
                )
            stage_index = _strict_integer(match["stage_index"], "stage_index")
            if stage_index != expected_stage_index:
                raise TRAKESubmissionValidationError(
                    "stage_index values must be contiguous from 1 in input order"
                )
            timestamp_ms = _strict_integer(match["timestamp_ms"], "timestamp_ms")
            if timestamp_ms < 0:
                raise TRAKESubmissionValidationError(
                    "timestamp_ms values must be non-negative"
                )
            frames.append(_integer(match["source_frame_idx"], "source_frame_idx"))
            timestamps_ms.append(timestamp_ms)
        return _ParsedAlignment(tuple(frames), tuple(timestamps_ms))
    frame_ids = value.get("frame_ids", value.get("source_frame_indices"))
    if not isinstance(frame_ids, (list, tuple)):
        raise TRAKESubmissionValidationError(
            "each result requires stage_matches or canonical frame_ids"
        )
    return _ParsedAlignment(
        tuple(_integer(item, "source_frame_idx") for item in frame_ids),
        None,
    )


def _entry(
    value: TRAKESubmissionItem,
) -> tuple[int, str, str, _ParsedAlignment]:
    if isinstance(value, TRAKEResponseItem):
        return (
            value.rank,
            value.batch_id,
            value.video_id,
            _ParsedAlignment(
                value.source_frame_indices,
                tuple(match.timestamp_ms for match in value.stage_matches),
            ),
        )
    if not isinstance(value, Mapping):
        raise TRAKESubmissionValidationError(
            f"unsupported submission item: {type(value).__name__}"
        )
    rank = _integer(value.get("rank"), "rank")
    batch_id = value.get("batch_id")
    video_id = value.get("video_id", value.get("video_name"))
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise TRAKESubmissionValidationError("batch_id must be a non-empty string")
    if not isinstance(video_id, str) or not video_id.strip():
        raise TRAKESubmissionValidationError("video_id must be a non-empty string")
    return rank, batch_id.strip(), video_id.strip(), _mapping_frames(value)


def validate_trake_submission(
    items: Sequence[TRAKESubmissionItem],
    *,
    stage_count: int | None = None,
    spec: TRAKECompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: Callable[[str, str, int], bool] | None = None,
    get_canonical_timestamp_ms: Callable[[str, str, int], int | None] | None = None,
    frame_index_evidence: TRAKEFrameIndexEvidence | None = None,
) -> tuple[TRAKESubmissionRow, ...]:
    """Validate ranking, batch identity, chronology, and canonical frame semantics.

    Mapping-based ``stage_matches`` carry their timestamps directly. Lightweight
    ``frame_ids`` inputs require ``get_canonical_timestamp_ms`` so their temporal
    order is validated before any competition row is emitted.
    """

    active_spec = spec or TRAKECompetitionSpec.historical_2025_provisional()
    if not items:
        raise TRAKESubmissionValidationError("submission must contain at least one row")
    if len(items) > active_spec.max_results:
        raise TRAKESubmissionValidationError(
            f"submission has {len(items)} rows; maximum is {active_spec.max_results}"
        )
    parsed = [_entry(item) for item in items]
    parsed.sort(key=lambda row: row[0])
    if [row[0] for row in parsed] != list(range(1, len(parsed) + 1)):
        raise TRAKESubmissionValidationError("result ranks must be contiguous from 1")
    inferred_stage_count = (
        len(parsed[0][3].source_frame_indices) if stage_count is None else stage_count
    )
    if (
        isinstance(inferred_stage_count, bool)
        or not isinstance(inferred_stage_count, int)
        or not 2 <= inferred_stage_count <= 20
    ):
        raise TRAKESubmissionValidationError("stage_count must be between 2 and 20")

    requested_batch = batch_id.strip() if isinstance(batch_id, str) else None
    row_batches = {row[1] for row in parsed}
    if len(row_batches) != 1:
        raise TRAKESubmissionValidationError("TRAKE submission rows cannot mix batches")
    sole_batch = next(iter(row_batches))
    if requested_batch is None:
        requested_batch = sole_batch
    if is_batch_active is not None:
        if not requested_batch:
            raise TRAKESubmissionValidationError(
                "batch_id is required when an active-batch callback is provided"
            )
        if not is_batch_active(requested_batch):
            raise TRAKESubmissionValidationError(f"batch {requested_batch!r} is not active")

    rows: list[TRAKESubmissionRow] = []
    seen_videos: set[str] = set()
    for rank, row_batch, raw_video_id, alignment in parsed:
        frames = alignment.source_frame_indices
        if requested_batch is not None and row_batch != requested_batch:
            raise TRAKESubmissionValidationError(
                "prediction batch_id does not match export batch_id"
            )
        try:
            video_id = active_spec.normalize_video_id(raw_video_id)
        except (TypeError, ValueError) as exc:
            raise TRAKESubmissionValidationError(str(exc)) from exc
        if video_id in seen_videos:
            raise TRAKESubmissionValidationError("submission may contain only one row per video")
        seen_videos.add(video_id)
        if len(frames) != inferred_stage_count:
            raise TRAKESubmissionValidationError(
                f"each row must contain exactly {inferred_stage_count} source frames"
            )
        minimum = 0 if active_spec.source_frame_zero_based else 1
        if any(frame < minimum for frame in frames):
            raise TRAKESubmissionValidationError(
                f"source frame indices must be at least {minimum}"
            )
        if any(left >= right for left, right in zip(frames, frames[1:])):
            raise TRAKESubmissionValidationError(
                "source frame indices must be strictly increasing"
            )
        if is_canonical_source_frame is not None:
            for frame in frames:
                if not is_canonical_source_frame(row_batch, video_id, frame):
                    raise TRAKESubmissionValidationError(
                        f"{video_id}:{frame} is not a canonical source frame"
                    )
        canonical_timestamps: tuple[int, ...] | None = None
        if alignment.timestamps_ms is None and get_canonical_timestamp_ms is not None:
            resolved_timestamps: list[int] = []
            for frame in frames:
                try:
                    timestamp_ms = get_canonical_timestamp_ms(
                        row_batch,
                        video_id,
                        frame,
                    )
                except Exception:
                    raise TRAKESubmissionValidationError(
                        "canonical frame timestamp lookup failed"
                    ) from None
                if (
                    isinstance(timestamp_ms, bool)
                    or not isinstance(timestamp_ms, int)
                    or timestamp_ms < 0
                ):
                    raise TRAKESubmissionValidationError(
                        f"{video_id}:{frame} has no valid canonical timestamp"
                    )
                resolved_timestamps.append(timestamp_ms)
            canonical_timestamps = tuple(resolved_timestamps)
        timestamps_ms = alignment.timestamps_ms or canonical_timestamps
        if timestamps_ms is None:
            raise TRAKESubmissionValidationError(
                "canonical timestamp lookup is required when exporting frame_ids"
            )
        if any(
            left >= right
            for left, right in zip(timestamps_ms, timestamps_ms[1:])
        ):
            raise TRAKESubmissionValidationError(
                "timestamp_ms values must be strictly increasing"
            )
        if frame_index_evidence is not None:
            ordinal_map: Mapping[int, int] = {}
            for evidence_video, mapping in frame_index_evidence.ordinal_to_source.items():
                try:
                    normalized_evidence_video = active_spec.normalize_video_id(evidence_video)
                except (TypeError, ValueError):
                    continue
                if normalized_evidence_video == video_id:
                    ordinal_map = mapping
                    break
            ordinal_only = set(ordinal_map) - set(ordinal_map.values())
            ambiguous = ordinal_only.intersection(frames)
            if ambiguous:
                raise TRAKESubmissionValidationError(
                    f"{video_id} contains a keyframe_ordinal instead of source_frame_idx"
                )
        rows.append(TRAKESubmissionRow(rank, row_batch, video_id, frames))
    return tuple(rows)


def trake_submission_filename(query_id: str) -> str:
    """Return the safe required filename for one TRAKE query."""

    if not isinstance(query_id, str) or not query_id.strip():
        raise TRAKESubmissionValidationError("query_id must be a non-empty string")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", query_id.strip()).strip(".-")
    if not safe:
        raise TRAKESubmissionValidationError("query_id cannot form a safe filename")
    return f"{safe}-trake-submission.csv"


def export_trake_submission(
    items: Sequence[TRAKESubmissionItem],
    destination: str | Path,
    *,
    query_id: str,
    stage_count: int | None = None,
    spec: TRAKECompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: Callable[[str, str, int], bool] | None = None,
    get_canonical_timestamp_ms: Callable[[str, str, int], int | None] | None = None,
    frame_index_evidence: TRAKEFrameIndexEvidence | None = None,
) -> Path:
    """Validate and atomically write the provisional no-header TRAKE CSV."""

    data = serialize_trake_submission(
        items,
        stage_count=stage_count,
        spec=spec,
        batch_id=batch_id,
        is_batch_active=is_batch_active,
        is_canonical_source_frame=is_canonical_source_frame,
        get_canonical_timestamp_ms=get_canonical_timestamp_ms,
        frame_index_evidence=frame_index_evidence,
    )
    destination_path = Path(destination)
    target = (
        destination_path / trake_submission_filename(query_id)
        if destination_path.suffix.casefold() != ".csv"
        else destination_path
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return target


def serialize_trake_submission(
    items: Sequence[TRAKESubmissionItem],
    *,
    stage_count: int | None = None,
    spec: TRAKECompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: Callable[[str, str, int], bool] | None = None,
    get_canonical_timestamp_ms: Callable[[str, str, int], int | None] | None = None,
    frame_index_evidence: TRAKEFrameIndexEvidence | None = None,
) -> bytes:
    """Validate TRAKE rows and return UTF-8/LF competition bytes."""

    rows = validate_trake_submission(
        items,
        stage_count=stage_count,
        spec=spec,
        batch_id=batch_id,
        is_batch_active=is_batch_active,
        is_canonical_source_frame=is_canonical_source_frame,
        get_canonical_timestamp_ms=get_canonical_timestamp_ms,
        frame_index_evidence=frame_index_evidence,
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow((row.video_id, *row.source_frame_indices))
    return buffer.getvalue().encode("utf-8")


__all__ = [
    "TRAKEFrameIndexEvidence",
    "TRAKESubmissionItem",
    "TRAKESubmissionRow",
    "TRAKESubmissionValidationError",
    "export_trake_submission",
    "serialize_trake_submission",
    "trake_submission_filename",
    "validate_trake_submission",
]
