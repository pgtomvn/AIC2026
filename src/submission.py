"""Validation and atomic CSV export for Textual KIS submissions."""

from __future__ import annotations

import csv
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.contracts import CompetitionSpec, SearchHit
from src.evaluation.textual_kis import Prediction


class SubmissionValidationError(ValueError):
    """Raised before writing a submission that violates its competition spec."""


@dataclass(frozen=True, slots=True)
class FrameIndexEvidence:
    """Known organizer ordinal-to-source mappings, grouped by video."""

    ordinal_to_source: Mapping[str, Mapping[int, int]]

    def normalized(self, spec: CompetitionSpec) -> FrameIndexEvidence:
        """Normalize evidence keys with the same adapter used for submissions."""

        result: dict[str, dict[int, int]] = {}
        for raw_video_id, mapping in self.ordinal_to_source.items():
            video_id = spec.normalize_video_id(raw_video_id)
            normalized_mapping = {int(key): int(value) for key, value in mapping.items()}
            previous = result.get(video_id)
            if previous is not None and previous != normalized_mapping:
                raise SubmissionValidationError(
                    f"conflicting frame-map evidence normalizes to {video_id}"
                )
            result[video_id] = normalized_mapping
        return FrameIndexEvidence(result)

    def assert_not_ordinal(
        self,
        video_id: str,
        submitted_frame_idx: int,
        *,
        keyframe_ordinal: int | None = None,
        coarse_source_frame_idx: int | None = None,
    ) -> None:
        """Validate explicit frame provenance without guessing from an integer alone."""

        if keyframe_ordinal is None:
            return
        mapping = self.ordinal_to_source.get(video_id)
        if mapping is None:
            return
        mapped_source = mapping.get(keyframe_ordinal)
        if mapped_source is None:
            raise SubmissionValidationError(
                f"{video_id} has no mapping for keyframe_ordinal {keyframe_ordinal}"
            )
        if coarse_source_frame_idx is not None:
            if coarse_source_frame_idx != mapped_source:
                raise SubmissionValidationError(
                    f"{video_id} coarse frame provenance does not match keyframe_ordinal "
                    f"{keyframe_ordinal}"
                )
            return
        if submitted_frame_idx == keyframe_ordinal and mapped_source != submitted_frame_idx:
            raise SubmissionValidationError(
                f"{video_id} frame {submitted_frame_idx} is a keyframe_ordinal; "
                f"submit source_frame_idx {mapped_source} instead"
            )
        if submitted_frame_idx != mapped_source:
            raise SubmissionValidationError(
                f"{video_id} source frame {submitted_frame_idx} does not match "
                f"keyframe_ordinal {keyframe_ordinal} -> {mapped_source}"
            )


SubmissionItem = SearchHit | Prediction | tuple[str, int] | Mapping[str, object]
CanonicalSourceFrameCheck = Callable[[str, str, int], bool]


def _entry(
    value: SubmissionItem,
) -> tuple[int | None, str | None, str, int, int | None, int | None]:
    if isinstance(value, SearchHit):
        return value.rank, None, value.video_id, value.source_frame_idx, None, None
    if isinstance(value, Prediction):
        return None, None, value.video_id, value.source_frame_idx, None, None
    if isinstance(value, tuple) and len(value) == 2:
        video_id, source_frame_idx = value
        return None, None, str(video_id), _frame_int(source_frame_idx), None, None
    if isinstance(value, Mapping):
        if "keyframe_ordinal" in value and not (
            "source_frame_idx" in value or "frame_idx" in value
        ):
            raise SubmissionValidationError(
                "keyframe_ordinal cannot be submitted; provide source_frame_idx"
            )
        video_id = value.get("video_id", value.get("video_name"))
        source_frame_idx = value.get("source_frame_idx", value.get("frame_idx"))
        if video_id is None or source_frame_idx is None:
            raise SubmissionValidationError("each result requires video_id and source_frame_idx")
        rank_value = value.get("rank")
        rank = None if rank_value is None else _frame_int(rank_value)
        batch_value = value.get("batch_id")
        if batch_value is not None and (
            not isinstance(batch_value, str) or not batch_value.strip()
        ):
            raise SubmissionValidationError("batch_id must be a non-empty string or null")
        ordinal_value = value.get("keyframe_ordinal")
        ordinal = None if ordinal_value is None else _frame_int(ordinal_value)
        coarse_value = value.get("coarse_source_frame_idx")
        coarse = None if coarse_value is None else _frame_int(coarse_value)
        return (
            rank,
            batch_value.strip() if isinstance(batch_value, str) else None,
            str(video_id),
            _frame_int(source_frame_idx),
            ordinal,
            coarse,
        )
    raise SubmissionValidationError(f"unsupported submission item: {type(value).__name__}")


def _frame_int(value: object) -> int:
    if isinstance(value, bool):
        raise SubmissionValidationError("frame indices and ranks must be integers")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+").isdigit():
        return int(value.strip())
    raise SubmissionValidationError("frame indices and ranks must be integers")


def validate_submission(
    items: Sequence[SubmissionItem],
    *,
    spec: CompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: CanonicalSourceFrameCheck | None = None,
    frame_index_evidence: FrameIndexEvidence | None = None,
) -> tuple[Prediction, ...]:
    """Validate and normalize a ranked Textual KIS result list.

    Batch activation and canonical source-frame membership are mandatory trust
    boundaries. Ordinal misuse detection is additionally enabled when explicit
    organizer frame-map evidence is supplied.
    """
    active_spec = spec or CompetitionSpec.historical_2025()
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise SubmissionValidationError("batch_id is required for submission validation")
    batch_id = batch_id.strip()
    if is_batch_active is None:
        raise SubmissionValidationError("active-batch validation is required")
    if not is_batch_active(batch_id):
        raise SubmissionValidationError(f"batch {batch_id!r} is not active")
    if is_canonical_source_frame is None:
        raise SubmissionValidationError("canonical source-frame validation is required")
    if len(items) > active_spec.max_results:
        raise SubmissionValidationError(
            f"submission has {len(items)} rows; maximum is {active_spec.max_results}"
        )
    if not items:
        raise SubmissionValidationError("submission must contain at least one row")

    parsed = [_entry(item) for item in items]
    explicit_ranks = [rank for rank, *_ in parsed if rank is not None]
    if explicit_ranks:
        if len(explicit_ranks) != len(parsed):
            raise SubmissionValidationError("results cannot mix ranked and unranked rows")
        if len(explicit_ranks) != len(set(explicit_ranks)):
            raise SubmissionValidationError("result ranks must be unique")
        parsed.sort(key=lambda item: item[0] or 0)
        expected = list(range(1, len(parsed) + 1))
        if [rank for rank, *_ in parsed] != expected:
            raise SubmissionValidationError("result ranks must be contiguous from 1")

    normalized: list[Prediction] = []
    seen: set[tuple[str, int]] = set()
    normalized_evidence = (
        frame_index_evidence.normalized(active_spec) if frame_index_evidence is not None else None
    )
    supplied_batches = {item_batch for _, item_batch, *_ in parsed if item_batch is not None}
    if len(supplied_batches) > 1 or (supplied_batches and supplied_batches != {batch_id}):
        raise SubmissionValidationError("results cannot mix batches")
    for _, _, raw_video_id, source_frame_idx, keyframe_ordinal, coarse_source in parsed:
        try:
            video_id = active_spec.normalize_video_id(raw_video_id)
        except (TypeError, ValueError) as exc:
            raise SubmissionValidationError(str(exc)) from exc
        minimum = 0 if active_spec.source_frame_zero_based else 1
        if source_frame_idx < minimum:
            raise SubmissionValidationError(
                f"source_frame_idx must be at least {minimum} for {active_spec.spec_id}"
            )
        if normalized_evidence is not None:
            normalized_evidence.assert_not_ordinal(
                video_id,
                source_frame_idx,
                keyframe_ordinal=keyframe_ordinal,
                coarse_source_frame_idx=coarse_source,
            )
        identity = (video_id, source_frame_idx)
        if identity in seen:
            raise SubmissionValidationError(f"duplicate result: {video_id},{source_frame_idx}")
        seen.add(identity)
        normalized.append(Prediction(video_id, source_frame_idx))
    for prediction in normalized:
        if not is_canonical_source_frame(
            batch_id,
            prediction.video_id,
            prediction.source_frame_idx,
        ):
            raise SubmissionValidationError(
                f"{prediction.video_id}:{prediction.source_frame_idx} "
                "is not a canonical source frame"
            )
    return tuple(normalized)


def serialize_submission(
    items: Sequence[SubmissionItem],
    *,
    spec: CompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: CanonicalSourceFrameCheck | None = None,
    frame_index_evidence: FrameIndexEvidence | None = None,
) -> bytes:
    """Validate and return competition-format UTF-8 bytes without filesystem I/O."""

    active_spec = spec or CompetitionSpec.historical_2025()
    predictions = validate_submission(
        items,
        spec=active_spec,
        batch_id=batch_id,
        is_batch_active=is_batch_active,
        is_canonical_source_frame=is_canonical_source_frame,
        frame_index_evidence=frame_index_evidence,
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    if active_spec.include_header:
        writer.writerow(active_spec.output_columns)
    for prediction in predictions:
        values = {
            "video_name": prediction.video_id,
            "frame_idx": prediction.source_frame_idx,
        }
        writer.writerow([values[column] for column in active_spec.output_columns])
    return buffer.getvalue().encode("utf-8")


def export_submission(
    items: Sequence[SubmissionItem],
    destination: str | Path,
    *,
    spec: CompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: CanonicalSourceFrameCheck | None = None,
    frame_index_evidence: FrameIndexEvidence | None = None,
) -> Path:
    """Validate then atomically write a competition-format UTF-8 CSV."""
    data = serialize_submission(
        items,
        spec=spec,
        batch_id=batch_id,
        is_batch_active=is_batch_active,
        is_canonical_source_frame=is_canonical_source_frame,
        frame_index_evidence=frame_index_evidence,
    )
    target = Path(destination)
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


__all__ = [
    "FrameIndexEvidence",
    "CanonicalSourceFrameCheck",
    "SubmissionValidationError",
    "export_submission",
    "serialize_submission",
    "validate_submission",
]
