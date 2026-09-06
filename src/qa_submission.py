"""Validation and atomic CSV export for Visual Q&A predictions."""

from __future__ import annotations

import csv
import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.contracts import QACompetitionSpec, QAPrediction
from src.submission import (
    CanonicalSourceFrameCheck,
    FrameIndexEvidence,
    export_qa_csv,
)


class QASubmissionValidationError(ValueError):
    """Raised before writing a malformed Visual Q&A submission."""


QASubmissionItem = QAPrediction | tuple[str, int, str] | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Entry:
    rank: int | None
    video_id: str
    source_frame_idx: int
    answer: str
    confidence_score: float
    answer_source: str
    window_id: str
    batch_id: str | None = None
    timestamp_ms: int | None = None
    thumbnail_uri: str | None = None
    raw_answer: str | None = None
    model_name: str | None = None
    evidence: Mapping[str, Any] | None = None
    keyframe_ordinal: int | None = None
    coarse_source_frame_idx: int | None = None


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise QASubmissionValidationError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+").isdigit():
        return int(value.strip())
    raise QASubmissionValidationError(f"{name} must be an integer")


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QASubmissionValidationError("confidence_score must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise QASubmissionValidationError("confidence_score must be between 0 and 1")
    return result


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise QASubmissionValidationError(f"{name} must be a non-empty string or null")
    return value


def _entry(value: QASubmissionItem) -> _Entry:
    if isinstance(value, QAPrediction):
        return _Entry(
            rank=value.rank,
            video_id=value.video_id,
            source_frame_idx=value.source_frame_idx,
            answer=value.answer,
            confidence_score=value.confidence_score,
            answer_source=value.answer_source,
            window_id=value.window_id,
            batch_id=value.batch_id,
            timestamp_ms=value.timestamp_ms,
            thumbnail_uri=value.thumbnail_uri,
            raw_answer=value.raw_answer,
            model_name=value.model_name,
            evidence=value.evidence,
        )
    if isinstance(value, tuple) and len(value) == 3:
        video_id, source_frame_idx, answer = value
        if not isinstance(answer, str):
            raise QASubmissionValidationError("answer must be a string")
        return _Entry(
            rank=None,
            video_id=str(video_id),
            source_frame_idx=_integer(source_frame_idx, "source_frame_idx"),
            answer=answer,
            confidence_score=0.0,
            answer_source="submission",
            window_id=f"frame:{source_frame_idx}",
            raw_answer=answer,
        )
    if not isinstance(value, Mapping):
        raise QASubmissionValidationError(f"unsupported submission item: {type(value).__name__}")
    if "frame_id" in value:
        raise QASubmissionValidationError("ambiguous frame_id is forbidden; use source_frame_idx")
    if "keyframe_ordinal" in value and not ("source_frame_idx" in value or "frame_idx" in value):
        raise QASubmissionValidationError(
            "keyframe_ordinal cannot be submitted; provide source_frame_idx"
        )
    video_id = value.get("video_id", value.get("video_name"))
    source_frame_idx = value.get("source_frame_idx", value.get("frame_idx"))
    answer = value.get("answer")
    if video_id is None or source_frame_idx is None or answer is None:
        raise QASubmissionValidationError(
            "each result requires video_id, source_frame_idx, and answer"
        )
    if not isinstance(answer, str):
        raise QASubmissionValidationError("answer must be a string")
    rank_value = value.get("rank")
    timestamp_value = value.get("timestamp_ms")
    ordinal_value = value.get("keyframe_ordinal")
    coarse_value = value.get("coarse_source_frame_idx")
    evidence = value.get("evidence", {})
    if not isinstance(evidence, Mapping):
        raise QASubmissionValidationError("evidence must be a mapping")
    return _Entry(
        rank=None if rank_value is None else _integer(rank_value, "rank"),
        video_id=str(video_id),
        source_frame_idx=_integer(source_frame_idx, "source_frame_idx"),
        answer=answer,
        confidence_score=_confidence(value.get("confidence_score", 0.0)),
        answer_source=_optional_text(value.get("answer_source", "submission"), "answer_source")
        or "submission",
        window_id=_optional_text(value.get("window_id", f"frame:{source_frame_idx}"), "window_id")
        or f"frame:{source_frame_idx}",
        batch_id=_optional_text(value.get("batch_id"), "batch_id"),
        timestamp_ms=(
            None if timestamp_value is None else _integer(timestamp_value, "timestamp_ms")
        ),
        thumbnail_uri=_optional_text(value.get("thumbnail_uri"), "thumbnail_uri"),
        raw_answer=_optional_text(value.get("raw_answer", answer), "raw_answer"),
        model_name=_optional_text(value.get("model_name"), "model_name"),
        evidence=dict(evidence),
        keyframe_ordinal=(
            None if ordinal_value is None else _integer(ordinal_value, "keyframe_ordinal")
        ),
        coarse_source_frame_idx=(
            None if coarse_value is None else _integer(coarse_value, "coarse_source_frame_idx")
        ),
    )


def validate_qa_submission(
    items: Sequence[QASubmissionItem],
    *,
    spec: QACompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: CanonicalSourceFrameCheck | None = None,
    frame_index_evidence: FrameIndexEvidence | None = None,
) -> tuple[QAPrediction, ...]:
    """Validate, normalize, rank, and deduplicate Visual Q&A predictions.

    Alternate normalized answers may intentionally target the same canonical
    frame at different ranks.  Only an exact ``(video, frame, answer)`` row is
    rejected because it cannot add submission coverage.
    """

    active_spec = spec or QACompetitionSpec.historical_numeric()
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise QASubmissionValidationError("batch_id is required for submission validation")
    batch_id = batch_id.strip()
    if is_batch_active is None:
        raise QASubmissionValidationError("active-batch validation is required")
    if not is_batch_active(batch_id):
        raise QASubmissionValidationError(f"batch {batch_id!r} is not active")
    if is_canonical_source_frame is None:
        raise QASubmissionValidationError("canonical source-frame validation is required")
    if not items:
        raise QASubmissionValidationError("submission must contain at least one row")
    if len(items) > active_spec.max_results:
        raise QASubmissionValidationError(
            f"submission has {len(items)} rows; maximum is {active_spec.max_results}"
        )

    parsed = [_entry(item) for item in items]
    explicit_ranks = [item.rank for item in parsed if item.rank is not None]
    if explicit_ranks:
        if len(explicit_ranks) != len(parsed):
            raise QASubmissionValidationError("results cannot mix ranked and unranked rows")
        if len(explicit_ranks) != len(set(explicit_ranks)):
            raise QASubmissionValidationError("result ranks must be unique")
        parsed.sort(key=lambda item: item.rank or 0)
        if [item.rank for item in parsed] != list(range(1, len(parsed) + 1)):
            raise QASubmissionValidationError("result ranks must be contiguous from 1")

    try:
        normalized_evidence = (
            frame_index_evidence.normalized(active_spec)
            if frame_index_evidence is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        raise QASubmissionValidationError(str(exc)) from exc
    seen_rows: set[tuple[str, int, str]] = set()
    seen_batches = {item.batch_id for item in parsed if item.batch_id is not None}
    if len(seen_batches) > 1:
        raise QASubmissionValidationError("results cannot mix batches")
    if batch_id is not None and seen_batches and seen_batches != {batch_id}:
        raise QASubmissionValidationError("prediction batch_id does not match export batch_id")
    canonical_batch_id = batch_id or (next(iter(seen_batches)) if seen_batches else None)
    if canonical_batch_id is None:
        raise QASubmissionValidationError(
            "batch_id is required to validate canonical Visual Q&A frames"
        )

    normalized: list[QAPrediction] = []
    for rank, item in enumerate(parsed, start=1):
        try:
            video_id = active_spec.normalize_video_id(item.video_id)
            answer = active_spec.normalize_answer(item.answer)
        except (TypeError, ValueError) as exc:
            raise QASubmissionValidationError(str(exc)) from exc
        minimum = 0 if active_spec.source_frame_zero_based else 1
        if item.source_frame_idx < minimum:
            raise QASubmissionValidationError(
                f"source_frame_idx must be at least {minimum} for {active_spec.spec_id}"
            )
        if item.timestamp_ms is not None and item.timestamp_ms < 0:
            raise QASubmissionValidationError("timestamp_ms cannot be negative")
        if normalized_evidence is not None:
            try:
                normalized_evidence.assert_not_ordinal(
                    video_id,
                    item.source_frame_idx,
                    keyframe_ordinal=item.keyframe_ordinal,
                    coarse_source_frame_idx=item.coarse_source_frame_idx,
                )
            except (TypeError, ValueError) as exc:
                raise QASubmissionValidationError(str(exc)) from exc
        if not is_canonical_source_frame(
            canonical_batch_id, video_id, item.source_frame_idx
        ):
            raise QASubmissionValidationError(
                f"{video_id}:{item.source_frame_idx} is not a canonical source frame"
            )
        identity = (video_id, item.source_frame_idx, answer)
        if identity in seen_rows:
            raise QASubmissionValidationError(
                f"duplicate QA row: {video_id},{item.source_frame_idx},{answer}"
            )
        seen_rows.add(identity)
        normalized.append(
            QAPrediction(
                rank=rank,
                batch_id=canonical_batch_id,
                video_id=video_id,
                source_frame_idx=item.source_frame_idx,
                timestamp_ms=item.timestamp_ms or 0,
                answer=answer,
                confidence_score=item.confidence_score,
                answer_source=item.answer_source,
                window_id=item.window_id,
                evidence=dict(item.evidence or {}),
                thumbnail_uri=item.thumbnail_uri,
                raw_answer=item.raw_answer or item.answer,
                model_name=item.model_name,
            )
        )
    return tuple(normalized)


def serialize_qa_submission(
    items: Sequence[QASubmissionItem],
    *,
    spec: QACompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: CanonicalSourceFrameCheck | None = None,
    frame_index_evidence: FrameIndexEvidence | None = None,
) -> bytes:
    """Validate and return UTF-8 Visual Q&A CSV bytes without server writes."""

    active_spec = spec or QACompetitionSpec.historical_numeric()
    predictions = validate_qa_submission(
        items,
        spec=active_spec,
        batch_id=batch_id,
        is_batch_active=is_batch_active,
        is_canonical_source_frame=is_canonical_source_frame,
        frame_index_evidence=frame_index_evidence,
    )
    canonical_columns = ("video_name", "frame_idx", "answer")
    if active_spec.output_columns == canonical_columns:
        body = export_qa_csv(
            (prediction.video_id, prediction.source_frame_idx, prediction.answer)
            for prediction in predictions
        )
        if not active_spec.include_header:
            return body.encode("utf-8")
        header = io.StringIO(newline="")
        csv.writer(
            header,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True,
        ).writerow(active_spec.output_columns)
        return (header.getvalue() + body).encode("utf-8")

    # Preserve the configurable column-order compatibility surface while the
    # competition's canonical three-column path uses the shared exporter.
    buffer = io.StringIO(newline="")
    writer = csv.writer(
        buffer,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
    )
    if active_spec.include_header:
        writer.writerow(active_spec.output_columns)
    for prediction in predictions:
        values: dict[str, str | int] = {
            "video_name": prediction.video_id,
            "frame_idx": prediction.source_frame_idx,
            "answer": prediction.answer,
        }
        writer.writerow([values[column] for column in active_spec.output_columns])
    return buffer.getvalue().encode("utf-8")


def export_qa_submission(
    items: Sequence[QASubmissionItem],
    destination: str | Path,
    *,
    spec: QACompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: CanonicalSourceFrameCheck | None = None,
    frame_index_evidence: FrameIndexEvidence | None = None,
) -> Path:
    """Validate then atomically write a UTF-8 Visual Q&A CSV."""

    data = serialize_qa_submission(
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
    "QASubmissionItem",
    "QASubmissionValidationError",
    "export_qa_submission",
    "serialize_qa_submission",
    "validate_qa_submission",
]
