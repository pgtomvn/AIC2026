"""Offline RRF tuning and cumulative ablations over cached channel rankings."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.evaluation.textual_kis import GroundTruthQuery, evaluate_queries
from src.retrieval.fusion import ChannelRanking, weighted_rrf


def _stable_fingerprint(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedChannelRanking:
    """One query/channel ranking and its measured retrieval latency."""

    results: tuple[dict[str, Any], ...]
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class TuningCandidate:
    """Parameters and validation objective for one candidate."""

    rrf_k: int
    depths: dict[str, int]
    weights: dict[str, float]
    mean_top_k_score: float
    r_at_1: float
    p95_latency_ms: float


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Cumulative evaluation after adding one retrieval channel."""

    channels: tuple[str, ...]
    mean_top_k_score: float
    r_at_1: float
    p95_latency_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalTuningReport:
    """Persistable tuning result with explicit default/no-label behavior."""

    selected_rrf_k: int
    selected_depths: dict[str, int]
    selected_weights: dict[str, float]
    used_defaults: bool
    reason: str
    candidates_evaluated: int
    objective: dict[str, str]
    best_metrics: dict[str, float] | None
    ablations: tuple[AblationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        selected = {
            "rrf_k": self.selected_rrf_k,
            "candidate_depths": self.selected_depths,
            "channel_weights": self.selected_weights,
        }
        fingerprint_content = {
            "selected": selected,
            "used_defaults": self.used_defaults,
            "candidates_evaluated": self.candidates_evaluated,
            "best_metrics": self.best_metrics,
        }
        return {
            "schema_version": 1,
            "selected": selected,
            "used_defaults": self.used_defaults,
            "reason": self.reason,
            "candidates_evaluated": self.candidates_evaluated,
            "objective": self.objective,
            "best_metrics": self.best_metrics,
            "ablations": [asdict(value) for value in self.ablations],
            "selection_fingerprint": _stable_fingerprint(fingerprint_content),
        }


@dataclass(frozen=True, slots=True)
class RetrievalTuningOverride:
    """Validated operational subset consumed by the retrieval composition root."""

    rrf_k: int
    depths: dict[str, int]
    weights: dict[str, float]


CachedRankings = Mapping[str, Mapping[str, CachedChannelRanking]]


def load_cached_rankings(path: str | Path) -> dict[str, dict[str, CachedChannelRanking]]:
    """Load the versioned cached-ranking JSON format used by the tuning CLI."""

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("cached rankings require schema_version 1")
    rows = payload.get("queries") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("cached rankings JSON must contain a queries list")
    output: dict[str, dict[str, CachedChannelRanking]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("cached ranking query rows must be objects")
        query_id = row.get("query_id")
        channels = row.get("channels")
        if not isinstance(query_id, str) or not query_id or not isinstance(channels, Mapping):
            raise ValueError("each cached query requires query_id and channels")
        if query_id in output:
            raise ValueError(f"duplicate cached query ID: {query_id}")
        output[query_id] = {}
        for channel, raw in channels.items():
            if not isinstance(channel, str) or not channel:
                raise ValueError("cached channel names must be non-empty strings")
            if isinstance(raw, list):
                results, latency = raw, 0.0
            elif isinstance(raw, Mapping):
                results, latency = raw.get("results"), raw.get("latency_ms", 0.0)
            else:
                raise ValueError(f"invalid cached channel payload: {query_id}/{channel}")
            if not isinstance(results, list):
                raise ValueError(f"cached channel results must be a list: {query_id}/{channel}")
            latency_value = float(latency)
            if not math.isfinite(latency_value) or latency_value < 0:
                raise ValueError("cached channel latency must be finite and non-negative")
            if not all(isinstance(result, dict) for result in results):
                raise ValueError("cached channel results must contain objects")
            output[query_id][channel] = CachedChannelRanking(
                tuple(dict(result) for result in results), latency_value
            )
    return output


def load_tuning_override(
    path: str | Path, *, expected_channels: Sequence[str]
) -> RetrievalTuningOverride | None:
    """Load a labeled tuning result; return ``None`` for a no-label defaults report."""

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("retrieval tuning record requires schema_version 1")
    used_defaults = payload.get("used_defaults")
    if not isinstance(used_defaults, bool):
        raise ValueError("retrieval tuning record requires a boolean used_defaults")
    if used_defaults:
        return None
    if not isinstance(payload.get("best_metrics"), Mapping):
        raise ValueError("labeled retrieval tuning record requires best_metrics")
    if not isinstance(payload.get("candidates_evaluated"), int) or payload["candidates_evaluated"] < 1:
        raise ValueError("labeled retrieval tuning record requires evaluated candidates")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("retrieval tuning record is missing selected parameters")
    fingerprint_content = {
        "selected": dict(selected),
        "used_defaults": used_defaults,
        "candidates_evaluated": payload.get("candidates_evaluated"),
        "best_metrics": payload.get("best_metrics"),
    }
    if payload.get("selection_fingerprint") != _stable_fingerprint(fingerprint_content):
        raise ValueError("retrieval tuning record selection fingerprint mismatch")
    raw_depths, raw_weights = selected.get("candidate_depths"), selected.get("channel_weights")
    if not isinstance(raw_depths, Mapping) or not isinstance(raw_weights, Mapping):
        raise ValueError("retrieval tuning record contains invalid channel parameters")
    channels = set(expected_channels)
    if set(raw_depths) != channels or set(raw_weights) != channels:
        raise ValueError("retrieval tuning record channels do not match configured channels")
    rrf_k = int(selected["rrf_k"])
    depths = {str(key): int(value) for key, value in raw_depths.items()}
    weights = {str(key): float(value) for key, value in raw_weights.items()}
    if rrf_k < 0 or any(value < 1 for value in depths.values()):
        raise ValueError("retrieval tuning record contains invalid k/depths")
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("retrieval tuning record contains invalid weights")
    return RetrievalTuningOverride(rrf_k, depths, weights)


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _evaluate_configuration(
    cached: CachedRankings,
    ground_truth: Sequence[GroundTruthQuery],
    *,
    rrf_k: int,
    depths: Mapping[str, int],
    weights: Mapping[str, float],
    channels: Sequence[str] | None = None,
) -> tuple[float, float, float]:
    enabled = tuple(channels or weights.keys())
    predictions: dict[str, list[dict[str, Any]]] = {}
    latencies: dict[str, float] = {}
    for query in ground_truth:
        query_cache = cached[query.query_id]
        rankings = []
        latency = 0.0
        for channel in enabled:
            cached_channel = query_cache.get(channel)
            if cached_channel is None or weights[channel] <= 0:
                continue
            rankings.append(
                ChannelRanking(
                    channel,
                    cached_channel.results[: depths[channel]],
                    weights[channel],
                )
            )
            # Cached channel timings are conservatively additive. They are used
            # only as the final tie-breaker after both retrieval metrics.
            latency += cached_channel.latency_ms
        predictions[query.query_id] = weighted_rrf(rankings, k=rrf_k)[:100]
        latencies[query.query_id] = latency
    report = evaluate_queries(predictions, ground_truth, latencies_ms=latencies)
    return (
        report.mean_top_k_score,
        report.recall_at[1],
        float(report.latency_ms["p95"] or 0.0),
    )


def tune_cached_rankings(
    cached: CachedRankings | None,
    ground_truth: Sequence[GroundTruthQuery] | None,
    *,
    default_rrf_k: int,
    default_depths: Mapping[str, int],
    default_weights: Mapping[str, float],
    rrf_k_values: Sequence[int] = (20, 40, 60, 80),
    depth_values: Mapping[str, Sequence[int]] | None = None,
    weight_values: Mapping[str, Sequence[float]] | None = None,
    max_candidates: int = 50_000,
) -> RetrievalTuningReport:
    """Tune cached rankings; preserve documented defaults when labels are absent."""

    depths = {str(key): int(value) for key, value in default_depths.items()}
    weights = {str(key): float(value) for key, value in default_weights.items()}
    if set(depths) != set(weights):
        raise ValueError("default depths and weights must declare identical channels")
    if not ground_truth:
        return RetrievalTuningReport(
            default_rrf_k,
            depths,
            weights,
            True,
            "ground-truth labels unavailable; documented defaults preserved",
            0,
            {
                "primary": "mean_top_k_score",
                "tie_1": "r_at_1",
                "tie_2": "p95_latency_ms",
            },
            None,
            (),
        )
    if cached is None:
        raise ValueError("cached rankings are required when ground-truth labels are supplied")
    query_ids = {query.query_id for query in ground_truth}
    missing = sorted(query_ids - set(cached))
    if missing:
        raise ValueError(f"cached rankings are missing labeled queries: {missing}")
    k_grid = tuple(dict.fromkeys(int(value) for value in rrf_k_values))
    if not k_grid or any(value < 0 for value in k_grid):
        raise ValueError("RRF k values must be non-negative")
    depth_grid = {
        channel: tuple(
            dict.fromkeys(
                int(value)
                for value in (depth_values or {}).get(channel, (depths[channel],))
            )
        )
        for channel in depths
    }
    weight_grid = {
        channel: tuple(
            dict.fromkeys(
                float(value)
                for value in (weight_values or {}).get(channel, (weights[channel],))
            )
        )
        for channel in weights
    }
    if any(not values or any(value < 1 for value in values) for values in depth_grid.values()):
        raise ValueError("candidate depths must be positive")
    if any(not values or any(value < 0 for value in values) for values in weight_grid.values()):
        raise ValueError("channel weights must be non-negative")
    channels = tuple(depths)
    count = len(k_grid)
    for values in depth_grid.values():
        count *= len(values)
    for values in weight_grid.values():
        count *= len(values)
    if count > max_candidates:
        raise ValueError(f"tuning grid has {count} candidates; maximum is {max_candidates}")

    candidates: list[TuningCandidate] = []
    depth_products = itertools.product(*(depth_grid[channel] for channel in channels))
    for depth_tuple in depth_products:
        candidate_depths = dict(zip(channels, depth_tuple, strict=True))
        for weight_tuple in itertools.product(*(weight_grid[channel] for channel in channels)):
            candidate_weights = dict(zip(channels, weight_tuple, strict=True))
            for rrf_k in k_grid:
                score, r_at_1, p95 = _evaluate_configuration(
                    cached,
                    ground_truth,
                    rrf_k=rrf_k,
                    depths=candidate_depths,
                    weights=candidate_weights,
                )
                candidates.append(
                    TuningCandidate(
                        rrf_k,
                        candidate_depths,
                        candidate_weights,
                        score,
                        r_at_1,
                        p95,
                    )
                )
    best = min(
        candidates,
        key=lambda value: (
            -value.mean_top_k_score,
            -value.r_at_1,
            value.p95_latency_ms,
            value.rrf_k,
            tuple(value.depths.items()),
            tuple(value.weights.items()),
        ),
    )
    ablations: list[AblationResult] = []
    active: list[str] = []
    for channel in channels:
        active.append(channel)
        score, r_at_1, p95 = _evaluate_configuration(
            cached,
            ground_truth,
            rrf_k=best.rrf_k,
            depths=best.depths,
            weights=best.weights,
            channels=active,
        )
        ablations.append(AblationResult(tuple(active), score, r_at_1, p95))
    return RetrievalTuningReport(
        best.rrf_k,
        best.depths,
        best.weights,
        False,
        "selected on labeled cached rankings",
        len(candidates),
        {"primary": "mean_top_k_score", "tie_1": "r_at_1", "tie_2": "p95_latency_ms"},
        {
            "mean_top_k_score": best.mean_top_k_score,
            "r_at_1": best.r_at_1,
            "p95_latency_ms": best.p95_latency_ms,
        },
        tuple(ablations),
    )


def persist_tuning_report(report: RetrievalTuningReport, destination: str | Path) -> Path:
    """Atomically persist selected parameters and cumulative ablations."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return path


__all__ = [
    "AblationResult",
    "CachedChannelRanking",
    "RetrievalTuningReport",
    "RetrievalTuningOverride",
    "TuningCandidate",
    "load_cached_rankings",
    "load_tuning_override",
    "persist_tuning_report",
    "tune_cached_rankings",
]
