"""Offline evaluator for the provisional historical TRAKE rule."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.contracts import TRAKECompetitionSpec, TRAKEResponseItem


@dataclass(frozen=True, slots=True)
class TRAKEStageInterval:
    """Inclusive source-frame target interval for one ordered stage."""

    stage_index: int
    start_frame_idx: int
    end_frame_idx: int

    def __post_init__(self) -> None:
        for name in ("stage_index", "start_frame_idx", "end_frame_idx"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.stage_index < 1:
            raise ValueError("stage_index must be positive")
        if self.start_frame_idx < 0 or self.end_frame_idx < self.start_frame_idx:
            raise ValueError("stage interval must contain non-negative ordered frames")


@dataclass(frozen=True, slots=True)
class TRAKEGroundTruthQuery:
    """One target video and its ordered stage intervals."""

    query_id: str
    video_id: str
    stage_intervals: tuple[TRAKEStageInterval, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        intervals = tuple(self.stage_intervals)
        if not 2 <= len(intervals) <= 20:
            raise ValueError("ground truth must contain between 2 and 20 stages")
        if not all(isinstance(item, TRAKEStageInterval) for item in intervals):
            raise TypeError("stage_intervals must contain TRAKEStageInterval values")
        if tuple(item.stage_index for item in intervals) != tuple(range(1, len(intervals) + 1)):
            raise ValueError("stage intervals must be contiguous from 1")
        object.__setattr__(self, "stage_intervals", intervals)


@dataclass(frozen=True, slots=True)
class TRAKEEvaluationPrediction:
    """Minimal ranked alignment identity used by the evaluator."""

    video_id: str
    source_frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        frames = tuple(self.source_frame_indices)
        if not frames:
            raise ValueError("source_frame_indices must not be empty")
        for frame in frames:
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise ValueError("source frame indices must be non-negative integers")
        if any(left >= right for left, right in zip(frames, frames[1:])):
            raise ValueError("source frame indices must be strictly increasing")
        object.__setattr__(self, "source_frame_indices", frames)


@dataclass(frozen=True, slots=True)
class TRAKEQueryMetrics:
    """Provisional metrics for one query."""

    query_id: str
    best_score_at: Mapping[int, float]
    mean_top_k_r_score: float
    video_recall: float
    stage_accuracy: float
    exact_alignment_recall: float
    first_video_rank: int | None
    reciprocal_rank: float
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly metric record."""

        result = asdict(self)
        result["best_score_at"] = {
            str(cutoff): score for cutoff, score in self.best_score_at.items()
        }
        return result


@dataclass(frozen=True, slots=True)
class TRAKEEvaluationReport:
    """Aggregate provisional TRAKE metrics and per-query details."""

    query_count: int
    cutoffs: tuple[int, ...]
    mean_top_k_r_score: float
    best_score_at: Mapping[int, float]
    video_recall: float
    stage_accuracy: float
    exact_alignment_recall: float
    mrr: float
    latency_ms: Mapping[str, float | None]
    per_query: tuple[TRAKEQueryMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report."""

        return {
            "query_count": self.query_count,
            "cutoffs": list(self.cutoffs),
            "mean_top_k_r_score": self.mean_top_k_r_score,
            "best_score_at": {
                str(cutoff): score for cutoff, score in self.best_score_at.items()
            },
            "video_recall": self.video_recall,
            "stage_accuracy": self.stage_accuracy,
            "exact_alignment_recall": self.exact_alignment_recall,
            "mrr": self.mrr,
            "latency_ms": dict(self.latency_ms),
            "per_query": [item.to_dict() for item in self.per_query],
        }


TRAKEPredictionLike = TRAKEEvaluationPrediction | TRAKEResponseItem | Mapping[str, Any]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+").isdigit():
        return int(value.strip())
    raise ValueError(f"{name} must be an integer")


def _prediction(value: TRAKEPredictionLike) -> TRAKEEvaluationPrediction:
    if isinstance(value, TRAKEEvaluationPrediction):
        return value
    if isinstance(value, TRAKEResponseItem):
        return TRAKEEvaluationPrediction(value.video_id, value.source_frame_indices)
    if not isinstance(value, Mapping):
        raise TypeError(f"unsupported TRAKE prediction type: {type(value).__name__}")
    if "keyframe_ordinal" in value or "keyframe_ordinals" in value:
        raise ValueError("keyframe_ordinal cannot be evaluated as a source frame")
    video_id = value.get("video_id", value.get("video_name"))
    if not isinstance(video_id, str):
        raise ValueError("prediction requires video_id")
    stage_matches = value.get("stage_matches")
    if stage_matches is not None:
        if not isinstance(stage_matches, (list, tuple)):
            raise ValueError("stage_matches must be a sequence")
        frames = []
        for match in stage_matches:
            if not isinstance(match, Mapping) or "source_frame_idx" not in match:
                raise ValueError("each stage match requires source_frame_idx")
            frames.append(_integer(match["source_frame_idx"], "source_frame_idx"))
    else:
        raw_frames = value.get("frame_ids", value.get("source_frame_indices"))
        if not isinstance(raw_frames, (list, tuple)):
            raise ValueError("prediction requires frame_ids or stage_matches")
        frames = [_integer(item, "source_frame_idx") for item in raw_frames]
    return TRAKEEvaluationPrediction(video_id, tuple(frames))


def trake_alignment_score(
    prediction: TRAKEPredictionLike,
    ground_truth: TRAKEGroundTruthQuery,
    spec: TRAKECompetitionSpec | None = None,
) -> float:
    """Return the fraction of correctly localized stages, or zero for a wrong video."""

    active_spec = spec or TRAKECompetitionSpec.historical_2025_provisional()
    item = _prediction(prediction)
    try:
        predicted_video = active_spec.normalize_video_id(item.video_id)
        target_video = active_spec.normalize_video_id(ground_truth.video_id)
    except (TypeError, ValueError):
        return 0.0
    if predicted_video != target_video:
        return 0.0
    if len(item.source_frame_indices) != len(ground_truth.stage_intervals):
        return 0.0
    correct = sum(
        active_spec.contains_frame(frame, interval.start_frame_idx, interval.end_frame_idx)
        for frame, interval in zip(item.source_frame_indices, ground_truth.stage_intervals)
    )
    return correct / len(ground_truth.stage_intervals)


def evaluate_trake_query(
    predictions: Sequence[TRAKEPredictionLike],
    ground_truth: TRAKEGroundTruthQuery,
    *,
    spec: TRAKECompetitionSpec | None = None,
    latency_ms: float | None = None,
) -> TRAKEQueryMetrics:
    """Evaluate one ranked list using best partial alignment at each cutoff."""

    active_spec = spec or TRAKECompetitionSpec.historical_2025_provisional()
    if latency_ms is not None and (not math.isfinite(latency_ms) or latency_ms < 0):
        raise ValueError("latency_ms must be a finite non-negative number")
    retained = tuple(predictions[: active_spec.max_results])
    parsed = tuple(_prediction(item) for item in retained)
    scores = tuple(
        trake_alignment_score(item, ground_truth, active_spec) for item in parsed
    )
    best_at = {
        cutoff: max(scores[:cutoff], default=0.0) for cutoff in active_spec.cutoffs
    }
    target_video = active_spec.normalize_video_id(ground_truth.video_id)
    first_video_rank: int | None = None
    for rank, item in enumerate(parsed, start=1):
        try:
            matches = active_spec.normalize_video_id(item.video_id) == target_video
        except (TypeError, ValueError):
            matches = False
        if matches:
            first_video_rank = rank
            break
    best_score = max(scores, default=0.0)
    return TRAKEQueryMetrics(
        query_id=ground_truth.query_id,
        best_score_at=best_at,
        mean_top_k_r_score=statistics.fmean(best_at.values()),
        video_recall=float(first_video_rank is not None),
        stage_accuracy=best_score,
        exact_alignment_recall=float(best_score == 1.0),
        first_video_rank=first_video_rank,
        reciprocal_rank=0.0 if first_video_rank is None else 1.0 / first_video_rank,
        latency_ms=None if latency_ms is None else float(latency_ms),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def evaluate_trake_queries(
    predictions: Mapping[str, Sequence[TRAKEPredictionLike]],
    ground_truth: Sequence[TRAKEGroundTruthQuery],
    *,
    spec: TRAKECompetitionSpec | None = None,
    latencies_ms: Mapping[str, float] | None = None,
) -> TRAKEEvaluationReport:
    """Evaluate a query set and aggregate retrieval, alignment, and latency metrics."""

    active_spec = spec or TRAKECompetitionSpec.historical_2025_provisional()
    queries = tuple(ground_truth)
    if not queries:
        raise ValueError("ground_truth must contain at least one query")
    query_ids = tuple(item.query_id for item in queries)
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("ground_truth query IDs must be unique")
    unknown = set(predictions) - set(query_ids)
    if unknown:
        raise ValueError(f"predictions contain unknown query IDs: {sorted(unknown)}")
    metrics = tuple(
        evaluate_trake_query(
            predictions.get(query.query_id, ()),
            query,
            spec=active_spec,
            latency_ms=None if latencies_ms is None else latencies_ms.get(query.query_id),
        )
        for query in queries
    )
    latency_values = [item.latency_ms for item in metrics if item.latency_ms is not None]
    latency_summary: dict[str, float | None] = {
        "mean": statistics.fmean(latency_values) if latency_values else None,
        "median": statistics.median(latency_values) if latency_values else None,
        "p95": _percentile(latency_values, 0.95) if latency_values else None,
    }
    return TRAKEEvaluationReport(
        query_count=len(metrics),
        cutoffs=active_spec.cutoffs,
        mean_top_k_r_score=statistics.fmean(
            item.mean_top_k_r_score for item in metrics
        ),
        best_score_at={
            cutoff: statistics.fmean(item.best_score_at[cutoff] for item in metrics)
            for cutoff in active_spec.cutoffs
        },
        video_recall=statistics.fmean(item.video_recall for item in metrics),
        stage_accuracy=statistics.fmean(item.stage_accuracy for item in metrics),
        exact_alignment_recall=statistics.fmean(
            item.exact_alignment_recall for item in metrics
        ),
        mrr=statistics.fmean(item.reciprocal_rank for item in metrics),
        latency_ms=latency_summary,
        per_query=metrics,
    )


__all__ = [
    "TRAKEEvaluationPrediction",
    "TRAKEEvaluationReport",
    "TRAKEGroundTruthQuery",
    "TRAKEPredictionLike",
    "TRAKEQueryMetrics",
    "TRAKEStageInterval",
    "evaluate_trake_queries",
    "evaluate_trake_query",
    "trake_alignment_score",
]
