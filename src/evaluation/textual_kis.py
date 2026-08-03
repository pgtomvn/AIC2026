"""Historical Textual KIS interval evaluation and grouped dataset splits."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.contracts import CompetitionSpec, SearchHit


@dataclass(frozen=True, slots=True)
class GroundTruthInterval:
    """One acceptable source-frame interval for a target video."""

    video_id: str
    start_frame_idx: int
    end_frame_idx: int

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        for name in ("start_frame_idx", "end_frame_idx"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.end_frame_idx < self.start_frame_idx:
            raise ValueError("end_frame_idx must be at least start_frame_idx")


@dataclass(frozen=True, slots=True)
class GroundTruthQuery:
    """All acceptable target intervals for one query."""

    query_id: str
    intervals: tuple[GroundTruthInterval, ...]
    text: str | None = None
    intent_class: str = "unspecified"

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        intervals = tuple(self.intervals)
        if not intervals:
            raise ValueError("a ground-truth query must have at least one interval")
        if not all(isinstance(interval, GroundTruthInterval) for interval in intervals):
            raise TypeError("intervals must contain GroundTruthInterval values")
        object.__setattr__(self, "intervals", intervals)
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or None")
        if not isinstance(self.intent_class, str) or not self.intent_class.strip():
            raise ValueError("intent_class must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Prediction:
    """Minimal prediction identity used by the evaluator."""

    video_id: str
    source_frame_idx: int

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if (
            isinstance(self.source_frame_idx, bool)
            or not isinstance(self.source_frame_idx, int)
            or self.source_frame_idx < 0
        ):
            raise ValueError("source_frame_idx must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Historical interval metrics for a single query."""

    query_id: str
    intent_class: str
    first_relevant_rank: int | None
    recall_at: Mapping[int, float]
    mean_top_k_score: float
    reciprocal_rank: float
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly metric record."""
        result = asdict(self)
        result["recall_at"] = {str(key): value for key, value in self.recall_at.items()}
        return result


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate and per-query Textual KIS metrics."""

    query_count: int
    cutoffs: tuple[int, ...]
    recall_at: Mapping[int, float]
    mean_top_k_score: float
    mrr: float
    median_rank: float
    median_rank_miss_value: int
    latency_ms: Mapping[str, float | None]
    by_intent_class: Mapping[str, Mapping[str, float | int]]
    per_query: tuple[QueryMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report."""
        return {
            "query_count": self.query_count,
            "cutoffs": list(self.cutoffs),
            "recall_at": {str(key): value for key, value in self.recall_at.items()},
            "mean_top_k_score": self.mean_top_k_score,
            "mrr": self.mrr,
            "median_rank": self.median_rank,
            "median_rank_miss_value": self.median_rank_miss_value,
            "latency_ms": dict(self.latency_ms),
            "by_intent_class": {
                name: dict(metrics) for name, metrics in self.by_intent_class.items()
            },
            "per_query": [metrics.to_dict() for metrics in self.per_query],
        }


@dataclass(frozen=True, slots=True)
class HoldoutSplit:
    """Video-grouped train, validation, and locked-test partitions."""

    train: tuple[GroundTruthQuery, ...]
    validation: tuple[GroundTruthQuery, ...]
    test: tuple[GroundTruthQuery, ...]


@dataclass(frozen=True, slots=True)
class CrossValidationFold:
    """One video-grouped cross-validation fold."""

    fold_index: int
    train: tuple[GroundTruthQuery, ...]
    validation: tuple[GroundTruthQuery, ...]


PredictionLike = Prediction | SearchHit | tuple[str, int] | Mapping[str, Any]


def _prediction(value: PredictionLike) -> Prediction:
    if isinstance(value, Prediction):
        return value
    if isinstance(value, SearchHit):
        return Prediction(value.video_id, value.source_frame_idx)
    if isinstance(value, tuple) and len(value) == 2:
        return Prediction(str(value[0]), _parse_int(value[1], "source_frame_idx"))
    if isinstance(value, Mapping):
        if "keyframe_ordinal" in value and not (
            "source_frame_idx" in value or "frame_idx" in value
        ):
            raise ValueError(
                "keyframe_ordinal cannot be evaluated as a submission frame; "
                "provide source_frame_idx"
            )
        video = _first(value, "video_id", "video_name", "video")
        frame = _first(value, "source_frame_idx", "frame_idx")
        return Prediction(str(video), _parse_int(frame, "source_frame_idx"))
    raise TypeError(f"unsupported prediction type: {type(value).__name__}")


def _is_relevant(
    prediction: Prediction,
    ground_truth: GroundTruthQuery,
    spec: CompetitionSpec,
) -> bool:
    try:
        predicted_video = spec.normalize_video_id(prediction.video_id)
    except ValueError:
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


def first_relevant_rank(
    predictions: Sequence[PredictionLike],
    ground_truth: GroundTruthQuery,
    spec: CompetitionSpec | None = None,
) -> int | None:
    """Return the one-based rank of the first prediction inside any target interval."""
    active_spec = spec or CompetitionSpec.historical_2025()
    for rank, item in enumerate(predictions[: active_spec.max_results], start=1):
        if _is_relevant(_prediction(item), ground_truth, active_spec):
            return rank
    return None


def recall_at_k(first_rank: int | None, k: int) -> float:
    """Return binary historical interval recall at cutoff ``k``."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return float(first_rank is not None and first_rank <= k)


def evaluate_query(
    predictions: Sequence[PredictionLike],
    ground_truth: GroundTruthQuery,
    *,
    spec: CompetitionSpec | None = None,
    latency_ms: float | None = None,
) -> QueryMetrics:
    """Evaluate one ranked result list with the historical interval rule."""
    active_spec = spec or CompetitionSpec.historical_2025()
    if latency_ms is not None and (not math.isfinite(latency_ms) or latency_ms < 0):
        raise ValueError("latency_ms must be a finite non-negative number")
    first_rank = first_relevant_rank(predictions, ground_truth, active_spec)
    recalls = {k: recall_at_k(first_rank, k) for k in active_spec.cutoffs}
    return QueryMetrics(
        query_id=ground_truth.query_id,
        intent_class=ground_truth.intent_class,
        first_relevant_rank=first_rank,
        recall_at=recalls,
        mean_top_k_score=statistics.fmean(recalls.values()),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        latency_ms=None if latency_ms is None else float(latency_ms),
    )


def evaluate_queries(
    predictions: Mapping[str, Sequence[PredictionLike]],
    ground_truth: Sequence[GroundTruthQuery],
    *,
    spec: CompetitionSpec | None = None,
    latencies_ms: Mapping[str, float] | None = None,
) -> EvaluationReport:
    """Evaluate multiple queries and return aggregate, intent, and per-query metrics."""
    active_spec = spec or CompetitionSpec.historical_2025()
    queries = tuple(ground_truth)
    if not queries:
        raise ValueError("ground_truth must contain at least one query")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("ground_truth query IDs must be unique")
    unknown_prediction_ids = set(predictions) - set(query_ids)
    if unknown_prediction_ids:
        raise ValueError(f"predictions contain unknown query IDs: {sorted(unknown_prediction_ids)}")

    metrics = tuple(
        evaluate_query(
            predictions.get(query.query_id, ()),
            query,
            spec=active_spec,
            latency_ms=None if latencies_ms is None else latencies_ms.get(query.query_id),
        )
        for query in queries
    )
    recalls = {
        cutoff: statistics.fmean(item.recall_at[cutoff] for item in metrics)
        for cutoff in active_spec.cutoffs
    }
    miss_rank = active_spec.cutoffs[-1] + 1
    median_rank = float(
        statistics.median(
            item.first_relevant_rank if item.first_relevant_rank is not None else miss_rank
            for item in metrics
        )
    )

    by_intent: dict[str, dict[str, float | int]] = {}
    intent_groups: dict[str, list[QueryMetrics]] = defaultdict(list)
    for item in metrics:
        intent_groups[item.intent_class].append(item)
    for intent, group in sorted(intent_groups.items()):
        by_intent[intent] = {
            "query_count": len(group),
            "mean_top_k_score": statistics.fmean(item.mean_top_k_score for item in group),
            "mrr": statistics.fmean(item.reciprocal_rank for item in group),
            **{
                f"r_at_{cutoff}": statistics.fmean(item.recall_at[cutoff] for item in group)
                for cutoff in active_spec.cutoffs
            },
        }

    latency_values = [item.latency_ms for item in metrics if item.latency_ms is not None]
    latency_summary: dict[str, float | None] = {
        "mean": statistics.fmean(latency_values) if latency_values else None,
        "median": statistics.median(latency_values) if latency_values else None,
        "p95": _percentile(latency_values, 0.95) if latency_values else None,
    }
    return EvaluationReport(
        query_count=len(metrics),
        cutoffs=active_spec.cutoffs,
        recall_at=recalls,
        mean_top_k_score=statistics.fmean(item.mean_top_k_score for item in metrics),
        mrr=statistics.fmean(item.reciprocal_rank for item in metrics),
        median_rank=median_rank,
        median_rank_miss_value=miss_rank,
        latency_ms=latency_summary,
        by_intent_class=by_intent,
        per_query=metrics,
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


def _query_components(
    queries: Sequence[GroundTruthQuery],
) -> list[tuple[GroundTruthQuery, ...]]:
    """Build connected query groups so a target video never crosses a partition."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for query in queries:
        videos = sorted({interval.video_id.strip().casefold() for interval in query.intervals})
        for video in videos:
            find(video)
        for video in videos[1:]:
            union(videos[0], video)

    grouped: dict[str, list[GroundTruthQuery]] = defaultdict(list)
    for query in queries:
        root = find(query.intervals[0].video_id.strip().casefold())
        grouped[root].append(query)
    return [tuple(group) for group in grouped.values()]


def _assign_groups(
    groups: Sequence[tuple[GroundTruthQuery, ...]],
    proportions: Sequence[float],
    seed: int,
) -> list[list[GroundTruthQuery]]:
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    total = sum(len(group) for group in shuffled)
    targets = [total * proportion for proportion in proportions]
    partitions: list[list[GroundTruthQuery]] = [[] for _ in proportions]

    for group in shuffled:
        scores = [
            (targets[index] - len(partitions[index])) / targets[index]
            if targets[index] > 0
            else float("-inf")
            for index in range(len(partitions))
        ]
        target_index = max(range(len(partitions)), key=lambda index: scores[index])
        partitions[target_index].extend(group)
    return partitions


def _assert_no_video_leakage(partitions: Sequence[Sequence[GroundTruthQuery]]) -> None:
    video_partitions: dict[str, int] = {}
    for partition_index, partition in enumerate(partitions):
        for query in partition:
            for interval in query.intervals:
                video_key = interval.video_id.strip().casefold()
                previous = video_partitions.setdefault(video_key, partition_index)
                if previous != partition_index:
                    raise AssertionError("target video leaked across grouped partitions")


def grouped_train_validation_test_split(
    queries: Sequence[GroundTruthQuery],
    *,
    seed: int = 2026,
) -> HoldoutSplit:
    """Create a deterministic 60/20/20 split without target-video leakage."""
    query_values = tuple(queries)
    if not query_values:
        raise ValueError("queries must not be empty")
    _validate_split_queries(query_values)
    groups = _query_components(query_values)
    if len(groups) < 3:
        raise ValueError("at least three independent video groups are required")
    train, validation, test = _assign_groups(groups, (0.6, 0.2, 0.2), seed)
    _assert_no_video_leakage((train, validation, test))
    return HoldoutSplit(tuple(train), tuple(validation), tuple(test))


def grouped_k_fold(
    queries: Sequence[GroundTruthQuery],
    *,
    folds: int = 5,
    seed: int = 2026,
) -> tuple[CrossValidationFold, ...]:
    """Create deterministic grouped cross-validation folds."""
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    query_values = tuple(queries)
    if not query_values:
        raise ValueError("queries must not be empty")
    _validate_split_queries(query_values)
    groups = _query_components(query_values)
    if len(groups) < folds:
        raise ValueError(f"at least {folds} independent video groups are required")
    validation_partitions = _assign_groups(groups, (1 / folds,) * folds, seed)
    _assert_no_video_leakage(validation_partitions)
    result: list[CrossValidationFold] = []
    for fold_index, validation in enumerate(validation_partitions):
        train = [
            query
            for index, partition in enumerate(validation_partitions)
            if index != fold_index
            for query in partition
        ]
        result.append(CrossValidationFold(fold_index, tuple(train), tuple(validation)))
    return tuple(result)


def build_grouped_splits(
    queries: Sequence[GroundTruthQuery],
    *,
    holdout_threshold: int = 50,
    seed: int = 2026,
) -> HoldoutSplit | tuple[CrossValidationFold, ...]:
    """Use 60/20/20 holdout for 50+ queries, otherwise five-fold CV."""
    query_values = tuple(queries)
    if len(query_values) >= holdout_threshold:
        return grouped_train_validation_test_split(query_values, seed=seed)
    return grouped_k_fold(query_values, folds=5, seed=seed)


def _validate_split_queries(queries: Sequence[GroundTruthQuery]) -> None:
    if not all(isinstance(query, GroundTruthQuery) for query in queries):
        raise TypeError("queries must contain GroundTruthQuery values")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique before splitting")


def load_ground_truth(path: str | Path) -> tuple[GroundTruthQuery, ...]:
    """Load grouped target intervals from CSV or common JSON layouts."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8-sig") as handle:
            rows = json.load(handle)
    else:
        raise ValueError("ground truth must be a .csv or .json file")
    return _ground_truth_rows(rows)


def _ground_truth_rows(value: Any) -> tuple[GroundTruthQuery, ...]:
    if isinstance(value, Mapping) and "queries" in value:
        value = value["queries"]
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for query_id, query_value in value.items():
            if isinstance(query_value, list):
                for interval in query_value:
                    rows.append({"query_id": query_id, **dict(interval)})
            elif isinstance(query_value, Mapping) and "intervals" in query_value:
                for interval in query_value["intervals"]:
                    rows.append(
                        {
                            "query_id": query_id,
                            "text": query_value.get("text"),
                            "intent_class": query_value.get("intent_class"),
                            **dict(interval),
                        }
                    )
            elif isinstance(query_value, Mapping):
                rows.append({"query_id": query_id, **dict(query_value)})
            else:
                raise ValueError(f"invalid ground-truth entry for {query_id!r}")
        value = rows
    if not isinstance(value, list):
        raise ValueError("ground-truth JSON must contain a list or query mapping")

    expanded_rows: list[Any] = []
    for raw_row in value:
        if isinstance(raw_row, Mapping) and "intervals" in raw_row:
            intervals = raw_row["intervals"]
            if not isinstance(intervals, list) or not intervals:
                raise ValueError("ground-truth intervals must be a non-empty list")
            common = {key: item for key, item in raw_row.items() if key != "intervals"}
            for interval in intervals:
                if not isinstance(interval, Mapping):
                    raise ValueError("each ground-truth interval must be an object")
                expanded_rows.append({**common, **dict(interval)})
        else:
            expanded_rows.append(raw_row)
    value = expanded_rows
    if not value:
        raise ValueError("ground truth must contain at least one interval")

    grouped: dict[str, list[GroundTruthInterval]] = defaultdict(list)
    metadata: dict[str, tuple[str | None, str]] = {}
    for index, raw_row in enumerate(value, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"ground-truth row {index} must be an object")
        row = dict(raw_row)
        query_id = str(_first(row, "query_id", "qid", "id", "query")).strip()
        video_id = str(_first(row, "video_id", "video_name", "video")).strip()
        if "keyframe_ordinal" in row and not any(
            name in row
            for name in (
                "source_frame_idx",
                "frame_idx",
                "frame_id",
                "start_frame_idx",
                "start",
                "s",
            )
        ):
            raise ValueError("ground truth must use source-frame intervals, not keyframe_ordinal")
        exact = _optional_first(row, "source_frame_idx", "frame_idx", "frame_id")
        start = (
            exact
            if exact is not None
            else _first(row, "start_frame_idx", "start_frame", "frame_start", "start", "s")
        )
        end = (
            exact
            if exact is not None
            else _first(row, "end_frame_idx", "end_frame", "frame_end", "end", "e")
        )
        interval = GroundTruthInterval(
            video_id=video_id,
            start_frame_idx=_parse_int(start, "start_frame_idx"),
            end_frame_idx=_parse_int(end, "end_frame_idx"),
        )
        grouped[query_id].append(interval)
        text_value = row.get("text")
        intent_value = row.get("intent_class") or row.get("intent") or "unspecified"
        candidate_metadata = (
            None if text_value in (None, "") else str(text_value),
            str(intent_value),
        )
        existing_metadata = metadata.get(query_id)
        if existing_metadata is None:
            metadata[query_id] = candidate_metadata
        else:
            old_text, old_intent = existing_metadata
            new_text, new_intent = candidate_metadata
            if old_text is not None and new_text is not None and old_text != new_text:
                raise ValueError(f"query {query_id!r} contains inconsistent text")
            if (
                old_intent != "unspecified"
                and new_intent != "unspecified"
                and old_intent != new_intent
            ):
                raise ValueError(f"query {query_id!r} contains inconsistent intent classes")
            metadata[query_id] = (
                old_text or new_text,
                old_intent if old_intent != "unspecified" else new_intent,
            )
    return tuple(
        GroundTruthQuery(query_id, tuple(intervals), *metadata[query_id])
        for query_id, intervals in grouped.items()
    )


def load_predictions(path: str | Path) -> dict[str, list[Prediction]]:
    """Load ranked predictions from CSV or common JSON layouts.

    A headerless two-column competition CSV is accepted for a single query and
    receives the query ID ``default``. Headerless multi-query CSV uses
    ``query_id,video_name,frame_idx``.
    """
    source = Path(path)
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return _prediction_rows(value)
    if source.suffix.lower() != ".csv":
        raise ValueError("predictions must be a .csv or .json file")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    if not raw_rows:
        return {}
    normalized_header = [cell.strip().lower() for cell in raw_rows[0]]
    known_headers = {"query_id", "qid", "video_id", "video_name", "source_frame_idx", "frame_idx"}
    if set(normalized_header) & known_headers:
        if len(normalized_header) != len(set(normalized_header)):
            raise ValueError("prediction CSV header contains duplicate columns")
        if any(len(row) != len(normalized_header) for row in raw_rows[1:]):
            raise ValueError("prediction CSV rows must match the header width")
        rows = [dict(zip(normalized_header, row, strict=True)) for row in raw_rows[1:]]
    else:
        width = len(raw_rows[0])
        if width not in (2, 3):
            raise ValueError("headerless prediction CSV must have two or three columns")
        names = (
            ("video_name", "frame_idx")
            if width == 2
            else (
                "query_id",
                "video_name",
                "frame_idx",
            )
        )
        if any(len(row) != width for row in raw_rows):
            raise ValueError("headerless prediction CSV rows must have a consistent width")
        rows = [dict(zip(names, row, strict=True)) for row in raw_rows]
    return _prediction_rows(rows)


def _prediction_rows(value: Any) -> dict[str, list[Prediction]]:
    if isinstance(value, Mapping) and "predictions" in value:
        value = value["predictions"]
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for query_id, predictions in value.items():
            if not isinstance(predictions, list):
                raise ValueError(f"predictions for {query_id!r} must be a list")
            for prediction in predictions:
                if isinstance(prediction, Mapping):
                    rows.append({"query_id": query_id, **dict(prediction)})
                elif isinstance(prediction, (list, tuple)) and len(prediction) == 2:
                    rows.append(
                        {
                            "query_id": query_id,
                            "video_id": prediction[0],
                            "source_frame_idx": prediction[1],
                        }
                    )
                else:
                    raise ValueError(f"invalid prediction for {query_id!r}")
        value = rows
    if not isinstance(value, list):
        raise ValueError("prediction JSON must contain a list or query mapping")

    grouped: dict[str, list[tuple[int | None, int, Prediction]]] = defaultdict(list)
    for order, raw_row in enumerate(value):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"prediction row {order + 1} must be an object")
        row = dict(raw_row)
        query_id = str(_optional_first(row, "query_id", "qid") or "default").strip()
        prediction = _prediction(row)
        rank_value = _optional_first(row, "rank")
        rank = None if rank_value in (None, "") else _parse_int(rank_value, "rank")
        if rank is not None and rank < 1:
            raise ValueError("rank must be positive")
        grouped[query_id].append((rank, order, prediction))

    result: dict[str, list[Prediction]] = {}
    for query_id, values in grouped.items():
        has_rank = [rank is not None for rank, _, _ in values]
        if any(has_rank) and not all(has_rank):
            raise ValueError(f"query {query_id!r} mixes ranked and unranked rows")
        if all(has_rank):
            ranks = [rank for rank, _, _ in values]
            if len(ranks) != len(set(ranks)):
                raise ValueError(f"query {query_id!r} contains duplicate ranks")
            values.sort(key=lambda item: (item[0], item[1]))
            if [rank for rank, _, _ in values] != list(range(1, len(values) + 1)):
                raise ValueError(f"query {query_id!r} ranks must be contiguous from 1")
        result[query_id] = [prediction for _, _, prediction in values]
    return result


def _first(value: Mapping[str, Any], *names: str) -> Any:
    result = _optional_first(value, *names)
    if result in (None, ""):
        raise ValueError(f"missing required field; expected one of {names}")
    return result


def _optional_first(value: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in value:
            return value[name]
    return None


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+").isdigit():
            return int(stripped)
    raise ValueError(f"{name} must be an integer")
