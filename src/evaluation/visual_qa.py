"""Exact interval-and-answer evaluation for Visual Question Answering."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.contracts import QACompetitionSpec, QAPrediction
from src.evaluation.textual_kis import GroundTruthInterval


@dataclass(frozen=True, slots=True)
class QAGroundTruthQuery:
    """Acceptable source-frame intervals and answers for one Visual Q&A query."""

    query_id: str
    intervals: tuple[GroundTruthInterval, ...]
    answers: tuple[str, ...]
    event_description: str | None = None
    question: str | None = None
    intent_class: str = "unspecified"

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        intervals = tuple(self.intervals)
        if not intervals or not all(
            isinstance(interval, GroundTruthInterval) for interval in intervals
        ):
            raise ValueError("intervals must contain at least one GroundTruthInterval")
        object.__setattr__(self, "intervals", intervals)
        answers = tuple(self.answers)
        if not answers or any(
            not isinstance(answer, str) or not answer.strip() for answer in answers
        ):
            raise ValueError("answers must contain at least one non-empty string")
        object.__setattr__(self, "answers", answers)
        for name in ("event_description", "question"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if not isinstance(self.intent_class, str) or not self.intent_class.strip():
            raise ValueError("intent_class must be a non-empty string")


GroundTruthQAQuery = QAGroundTruthQuery


@dataclass(frozen=True, slots=True)
class QAEvaluationPrediction:
    """Minimal prediction triple consumed by the evaluator."""

    video_id: str
    source_frame_idx: int
    answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if (
            isinstance(self.source_frame_idx, bool)
            or not isinstance(self.source_frame_idx, int)
            or self.source_frame_idx < 0
        ):
            raise ValueError("source_frame_idx must be a non-negative integer")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string")


@dataclass(frozen=True, slots=True)
class QAQueryMetrics:
    """End-to-end interval-and-answer metrics for one query."""

    query_id: str
    intent_class: str
    first_localized_rank: int | None
    first_relevant_rank: int | None
    retrieval_recall_at: Mapping[int, float]
    recall_at: Mapping[int, float]
    answer_accuracy_given_retrieval_at: Mapping[int, float | None]
    mean_top_k_score: float
    reciprocal_rank: float
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["retrieval_recall_at"] = {
            str(key): value for key, value in self.retrieval_recall_at.items()
        }
        result["recall_at"] = {str(key): value for key, value in self.recall_at.items()}
        result["answer_accuracy_given_retrieval_at"] = {
            str(key): value for key, value in self.answer_accuracy_given_retrieval_at.items()
        }
        return result


@dataclass(frozen=True, slots=True)
class QAEvaluationReport:
    """Aggregate and per-query Visual Q&A metrics."""

    query_count: int
    cutoffs: tuple[int, ...]
    retrieval_recall_at: Mapping[int, float]
    recall_at: Mapping[int, float]
    answer_accuracy_given_retrieval_at: Mapping[int, float | None]
    answer_accuracy_given_retrieval: float | None
    mean_top_k_score: float
    mrr: float
    median_rank: float
    median_rank_miss_value: int
    latency_ms: Mapping[str, float | None]
    cache_hit_rate: float | None
    cache_observation_count: int
    provider_usage: Mapping[str, int]
    by_intent_class: Mapping[str, Mapping[str, float | int]]
    per_query: tuple[QAQueryMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "cutoffs": list(self.cutoffs),
            "retrieval_recall_at": {
                str(key): value for key, value in self.retrieval_recall_at.items()
            },
            "recall_at": {str(key): value for key, value in self.recall_at.items()},
            "answer_accuracy_given_retrieval_at": {
                str(key): value for key, value in self.answer_accuracy_given_retrieval_at.items()
            },
            "answer_accuracy_given_retrieval": self.answer_accuracy_given_retrieval,
            "mean_top_k_score": self.mean_top_k_score,
            "mrr": self.mrr,
            "median_rank": self.median_rank,
            "median_rank_miss_value": self.median_rank_miss_value,
            "latency_ms": dict(self.latency_ms),
            "cache_hit_rate": self.cache_hit_rate,
            "cache_observation_count": self.cache_observation_count,
            "provider_usage": dict(self.provider_usage),
            "by_intent_class": {
                name: dict(metrics) for name, metrics in self.by_intent_class.items()
            },
            "per_query": [metrics.to_dict() for metrics in self.per_query],
        }


QAPredictionLike = QAEvaluationPrediction | QAPrediction | tuple[str, int, str] | Mapping[str, Any]


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+").isdigit():
        return int(value.strip())
    raise ValueError(f"{name} must be an integer")


def _prediction(value: QAPredictionLike) -> QAEvaluationPrediction:
    if isinstance(value, QAEvaluationPrediction):
        return value
    if isinstance(value, QAPrediction):
        return QAEvaluationPrediction(value.video_id, value.source_frame_idx, value.answer)
    if isinstance(value, tuple) and len(value) == 3:
        if not isinstance(value[2], str):
            raise ValueError("answer must be a string")
        return QAEvaluationPrediction(
            str(value[0]), _parse_int(value[1], "source_frame_idx"), value[2]
        )
    if isinstance(value, Mapping):
        if "frame_id" in value:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        if "keyframe_ordinal" in value and not (
            "source_frame_idx" in value or "frame_idx" in value
        ):
            raise ValueError("keyframe_ordinal cannot be evaluated as a submission frame")
        video_id = value.get("video_id", value.get("video_name"))
        source_frame_idx = value.get("source_frame_idx", value.get("frame_idx"))
        answer = value.get("answer")
        if video_id is None or source_frame_idx is None or answer is None:
            raise ValueError("prediction requires video_id, source_frame_idx, and answer")
        if not isinstance(answer, str):
            raise ValueError("answer must be a string")
        return QAEvaluationPrediction(
            str(video_id), _parse_int(source_frame_idx, "source_frame_idx"), answer
        )
    raise TypeError(f"unsupported prediction type: {type(value).__name__}")


def _is_relevant(
    prediction: QAEvaluationPrediction,
    ground_truth: QAGroundTruthQuery,
    spec: QACompetitionSpec,
) -> bool:
    try:
        predicted_answer = spec.normalize_answer(prediction.answer, ground_truth.question)
    except (TypeError, ValueError):
        return False
    accepted_answers = {
        spec.normalize_answer(answer, ground_truth.question) for answer in ground_truth.answers
    }
    if predicted_answer not in accepted_answers:
        return False
    return _is_localized(prediction, ground_truth, spec)


def _is_localized(
    prediction: QAEvaluationPrediction,
    ground_truth: QAGroundTruthQuery,
    spec: QACompetitionSpec,
) -> bool:
    try:
        predicted_video = spec.normalize_video_id(prediction.video_id)
    except (TypeError, ValueError):
        return False
    for interval in ground_truth.intervals:
        target_video = spec.normalize_video_id(interval.video_id)
        if predicted_video == target_video and spec.contains_frame(
            prediction.source_frame_idx,
            interval.start_frame_idx,
            interval.end_frame_idx,
        ):
            return True
    return False


def first_localized_qa_rank(
    predictions: Sequence[QAPredictionLike],
    ground_truth: QAGroundTruthQuery,
    spec: QACompetitionSpec | None = None,
) -> int | None:
    """Return the first correct video/interval rank while ignoring the answer."""

    active_spec = spec or QACompetitionSpec.historical_numeric()
    for rank, item in enumerate(predictions[: active_spec.max_results], start=1):
        if _is_localized(_prediction(item), ground_truth, active_spec):
            return rank
    return None


def first_relevant_qa_rank(
    predictions: Sequence[QAPredictionLike],
    ground_truth: QAGroundTruthQuery,
    spec: QACompetitionSpec | None = None,
) -> int | None:
    """Return the first rank matching video, source interval, and normalized answer."""

    active_spec = spec or QACompetitionSpec.historical_numeric()
    for rank, item in enumerate(predictions[: active_spec.max_results], start=1):
        if _is_relevant(_prediction(item), ground_truth, active_spec):
            return rank
    return None


def qa_recall_at_k(first_rank: int | None, k: int) -> float:
    """Return binary end-to-end Visual Q&A recall at cutoff ``k``."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return float(first_rank is not None and first_rank <= k)


def evaluate_qa_query(
    predictions: Sequence[QAPredictionLike],
    ground_truth: QAGroundTruthQuery,
    *,
    spec: QACompetitionSpec | None = None,
    latency_ms: float | None = None,
) -> QAQueryMetrics:
    """Evaluate one ranked list under exact video, interval, and answer rules."""

    active_spec = spec or QACompetitionSpec.historical_numeric()
    if latency_ms is not None and (not math.isfinite(latency_ms) or latency_ms < 0):
        raise ValueError("latency_ms must be a finite non-negative number")
    first_localized = first_localized_qa_rank(predictions, ground_truth, active_spec)
    first_rank = first_relevant_qa_rank(predictions, ground_truth, active_spec)
    retrieval_recalls = {
        cutoff: qa_recall_at_k(first_localized, cutoff) for cutoff in active_spec.cutoffs
    }
    recalls = {cutoff: qa_recall_at_k(first_rank, cutoff) for cutoff in active_spec.cutoffs}
    conditional = {
        cutoff: (None if retrieval_recalls[cutoff] == 0 else recalls[cutoff])
        for cutoff in active_spec.cutoffs
    }
    return QAQueryMetrics(
        query_id=ground_truth.query_id,
        intent_class=ground_truth.intent_class,
        first_localized_rank=first_localized,
        first_relevant_rank=first_rank,
        retrieval_recall_at=retrieval_recalls,
        recall_at=recalls,
        answer_accuracy_given_retrieval_at=conditional,
        mean_top_k_score=statistics.fmean(recalls.values()),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
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


def _prediction_provenance(value: QAPredictionLike) -> dict[str, Any]:
    if isinstance(value, QAPrediction):
        evidence = dict(value.evidence)
        result = {
            "window_id": value.window_id,
            "provider": evidence.get("provider") or value.answer_source.split("/", 1)[0],
            "cache_status": evidence.get("cache_status"),
        }
        reasoning = evidence.get("reasoning")
    elif isinstance(value, Mapping):
        evidence_value = value.get("evidence", {})
        evidence = dict(evidence_value) if isinstance(evidence_value, Mapping) else {}
        result = {
            "window_id": value.get("window_id"),
            "provider": value.get("provider") or evidence.get("provider"),
            "cache_status": value.get("cache_status") or evidence.get("cache_status"),
        }
        reasoning = value.get("reasoning", evidence.get("reasoning"))
    else:
        return {}
    if isinstance(reasoning, Mapping):
        result["provider"] = result.get("provider") or reasoning.get("provider")
        result["cache_status"] = result.get("cache_status") or reasoning.get("cache_status")
    return {key: item for key, item in result.items() if item is not None}


def _provenance_summary(
    predictions: Mapping[str, Sequence[QAPredictionLike]],
    provenance: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> tuple[float | None, int, dict[str, int]]:
    providers: dict[str, int] = defaultdict(int)
    cache_hits = 0
    cache_observations = 0
    for query_id, query_predictions in predictions.items():
        if provenance is not None and query_id in provenance:
            supplied = provenance[query_id]
            if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
                raise TypeError("provenance values must be sequences of mappings")
            rows: list[dict[str, Any]] = []
            for row in supplied:
                if not isinstance(row, Mapping):
                    raise TypeError("each provenance record must be a mapping")
                rows.append(dict(row))
        else:
            rows = [_prediction_provenance(item) for item in query_predictions]

        windows: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            window_value = row.get("window_id")
            window_id = (
                str(window_value)
                if isinstance(window_value, (str, int)) and str(window_value)
                else f"prediction:{index}"
            )
            merged = windows.setdefault(window_id, {})
            for name in ("provider", "cache_status"):
                value = row.get(name)
                if isinstance(value, str) and value.strip():
                    merged[name] = value.strip()
        for metadata in windows.values():
            provider = metadata.get("provider")
            if isinstance(provider, str):
                providers[provider] += 1
            cache_status = metadata.get("cache_status")
            if isinstance(cache_status, str):
                cache_observations += 1
                if cache_status.casefold() in {"hit", "cache_hit", "cached"}:
                    cache_hits += 1
    cache_rate = cache_hits / cache_observations if cache_observations else None
    return cache_rate, cache_observations, dict(sorted(providers.items()))


def evaluate_qa_queries(
    predictions: Mapping[str, Sequence[QAPredictionLike]],
    ground_truth: Sequence[QAGroundTruthQuery],
    *,
    spec: QACompetitionSpec | None = None,
    latencies_ms: Mapping[str, float] | None = None,
    provenance: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> QAEvaluationReport:
    """Evaluate a query set and aggregate the historical top-k metric."""

    active_spec = spec or QACompetitionSpec.historical_numeric()
    queries = tuple(ground_truth)
    if not queries:
        raise ValueError("ground_truth must contain at least one query")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("ground_truth query IDs must be unique")
    unknown_prediction_ids = set(predictions) - set(query_ids)
    if unknown_prediction_ids:
        raise ValueError(f"predictions contain unknown query IDs: {sorted(unknown_prediction_ids)}")
    unknown_provenance_ids = set(provenance or {}) - set(query_ids)
    if unknown_provenance_ids:
        raise ValueError(f"provenance contains unknown query IDs: {sorted(unknown_provenance_ids)}")

    metrics = tuple(
        evaluate_qa_query(
            predictions.get(query.query_id, ()),
            query,
            spec=active_spec,
            latency_ms=None if latencies_ms is None else latencies_ms.get(query.query_id),
        )
        for query in queries
    )
    retrieval_recalls = {
        cutoff: statistics.fmean(item.retrieval_recall_at[cutoff] for item in metrics)
        for cutoff in active_spec.cutoffs
    }
    recalls = {
        cutoff: statistics.fmean(item.recall_at[cutoff] for item in metrics)
        for cutoff in active_spec.cutoffs
    }
    conditional_accuracy: dict[int, float | None] = {}
    for cutoff in active_spec.cutoffs:
        localized = [item for item in metrics if item.retrieval_recall_at[cutoff] == 1.0]
        conditional_accuracy[cutoff] = (
            statistics.fmean(item.recall_at[cutoff] for item in localized) if localized else None
        )
    miss_rank = active_spec.cutoffs[-1] + 1
    median_rank = float(
        statistics.median(
            item.first_relevant_rank if item.first_relevant_rank is not None else miss_rank
            for item in metrics
        )
    )

    intent_groups: dict[str, list[QAQueryMetrics]] = defaultdict(list)
    for item in metrics:
        intent_groups[item.intent_class].append(item)
    by_intent: dict[str, dict[str, float | int]] = {}
    for intent, group in sorted(intent_groups.items()):
        by_intent[intent] = {
            "query_count": len(group),
            "mean_top_k_score": statistics.fmean(item.mean_top_k_score for item in group),
            "mrr": statistics.fmean(item.reciprocal_rank for item in group),
            **{
                f"r_at_{cutoff}": statistics.fmean(item.recall_at[cutoff] for item in group)
                for cutoff in active_spec.cutoffs
            },
            **{
                f"retrieval_r_at_{cutoff}": statistics.fmean(
                    item.retrieval_recall_at[cutoff] for item in group
                )
                for cutoff in active_spec.cutoffs
            },
        }

    latency_values = [item.latency_ms for item in metrics if item.latency_ms is not None]
    latency_summary: dict[str, float | None] = {
        "mean": statistics.fmean(latency_values) if latency_values else None,
        "median": statistics.median(latency_values) if latency_values else None,
        "p95": _percentile(latency_values, 0.95) if latency_values else None,
    }
    cache_hit_rate, cache_observation_count, provider_usage = _provenance_summary(
        predictions,
        provenance,
    )
    return QAEvaluationReport(
        query_count=len(metrics),
        cutoffs=active_spec.cutoffs,
        retrieval_recall_at=retrieval_recalls,
        recall_at=recalls,
        answer_accuracy_given_retrieval_at=conditional_accuracy,
        answer_accuracy_given_retrieval=conditional_accuracy[active_spec.cutoffs[-1]],
        mean_top_k_score=statistics.fmean(item.mean_top_k_score for item in metrics),
        mrr=statistics.fmean(item.reciprocal_rank for item in metrics),
        median_rank=median_rank,
        median_rank_miss_value=miss_rank,
        latency_ms=latency_summary,
        cache_hit_rate=cache_hit_rate,
        cache_observation_count=cache_observation_count,
        provider_usage=provider_usage,
        by_intent_class=by_intent,
        per_query=metrics,
    )


__all__ = [
    "GroundTruthQAQuery",
    "QAEvaluationPrediction",
    "QAEvaluationReport",
    "QAGroundTruthQuery",
    "QAPredictionLike",
    "QAQueryMetrics",
    "evaluate_qa_queries",
    "evaluate_qa_query",
    "first_localized_qa_rank",
    "first_relevant_qa_rank",
    "qa_recall_at_k",
]
