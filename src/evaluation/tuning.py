"""Offline RRF tuning and cumulative ablations over cached channel rankings."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.textual_kis import GroundTruthQuery, evaluate_queries
from src.retrieval.fusion import ChannelRanking, canonical_key, weighted_rrf


def _stable_fingerprint(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedChannelRanking:
    """One query/channel ranking and its measured retrieval latency."""

    results: tuple[dict[str, Any], ...]
    latency_ms: float = 0.0
    request_count: int = 1


@dataclass(frozen=True, slots=True)
class TuningCandidate:
    """Parameters and validation objective for one candidate."""

    rrf_k: int
    depths: dict[str, int]
    weights: dict[str, float]
    r_at_3: float
    mrr: float
    mean_top_k_score: float
    r_at_1: float
    p95_latency_ms: float


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Cumulative evaluation after adding one retrieval channel."""

    channels: tuple[str, ...]
    r_at_3: float
    mrr: float
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


@dataclass(frozen=True, slots=True)
class IntentPolicyTuningReport:
    """Qrels-derived RRF policies for runtime intent buckets."""

    policies: dict[str, dict[str, float | int]]
    bucket_reports: dict[str, dict[str, Any]]
    qrels_provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        content = {
            "intent_policies": self.policies,
            "bucket_reports": self.bucket_reports,
            "qrels_provenance": dict(self.qrels_provenance),
        }
        return {
            "schema_version": 1,
            **content,
            "selection_fingerprint": _stable_fingerprint(content),
        }


@dataclass(frozen=True, slots=True)
class SemanticViewPolicy:
    """Fingerprint-gated semantic late-fusion policy selected on locked qrels."""

    fusion: str
    weights: dict[str, float]
    consensus_alpha: float


@dataclass(frozen=True, slots=True)
class SemanticViewTuningCandidate:
    """One empirically evaluated semantic-view fusion hypothesis."""

    policy: str
    runtime_fusion: str
    rrf_k: int
    candidate_depth: int
    weights: dict[str, float]
    consensus_alpha: float
    r_at_3: float
    mrr: float
    r_at_1: float
    mean_top_k_score: float
    p95_latency_ms: float


@dataclass(frozen=True, slots=True)
class SemanticViewTuningReport:
    """Train-derived weights and validation-selected semantic-view policy."""

    selected: SemanticViewTuningCandidate
    standalone_train_mrr: dict[str, float]
    candidates_evaluated: int
    qrels_provenance: Mapping[str, Any]
    fingerprints: Mapping[str, str]
    activation_gate: Mapping[str, Any] = field(
        default_factory=lambda: {
            "passed": False,
            "reason": "locked test has not been evaluated",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a record that runtime rejects until locked-test activation passes."""

        selected = {
            "fusion": self.selected.runtime_fusion,
            "weights": dict(sorted(self.selected.weights.items())),
            "consensus_alpha": self.selected.consensus_alpha,
            "rrf_k": self.selected.rrf_k,
            "candidate_depth": self.selected.candidate_depth,
        }
        content = {
            "selected": selected,
            "qrels_provenance": dict(self.qrels_provenance),
            "fingerprints": dict(self.fingerprints),
            "activation_gate": dict(self.activation_gate),
        }
        return {
            "schema_version": 1,
            **content,
            "selection": {
                "policy": self.selected.policy,
                "objective": [
                    "r_at_3",
                    "mrr",
                    "r_at_1",
                    "mean_top_k_score",
                    "p95_latency_ms",
                    "candidate_depth",
                ],
                "metrics": {
                    "r_at_3": self.selected.r_at_3,
                    "mrr": self.selected.mrr,
                    "r_at_1": self.selected.r_at_1,
                    "mean_top_k_score": self.selected.mean_top_k_score,
                    "p95_latency_ms": self.selected.p95_latency_ms,
                },
                "standalone_train_mrr": dict(
                    sorted(self.standalone_train_mrr.items())
                ),
                "candidates_evaluated": self.candidates_evaluated,
            },
            "selection_fingerprint": _stable_fingerprint(content),
        }


def load_semantic_view_policy(path: str | Path) -> SemanticViewPolicy:
    """Load only an activation-gated semantic-view policy with immutable provenance."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    selected = payload.get("selected") if isinstance(payload, Mapping) else None
    provenance = payload.get("qrels_provenance") if isinstance(payload, Mapping) else None
    fingerprints = payload.get("fingerprints") if isinstance(payload, Mapping) else None
    activation = payload.get("activation_gate") if isinstance(payload, Mapping) else None
    content = {
        "selected": selected,
        "qrels_provenance": provenance,
        "fingerprints": fingerprints,
        "activation_gate": activation,
    }
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or not isinstance(selected, Mapping)
        or not isinstance(provenance, Mapping)
        or not provenance.get("locked_test_sha256")
        or not isinstance(fingerprints, Mapping)
        or not all(
            isinstance(fingerprints.get(name), str) and fingerprints.get(name)
            for name in ("model", "ranking_cache", "query_adapter")
        )
        or not isinstance(activation, Mapping)
        or activation.get("passed") is not True
        or payload.get("selection_fingerprint") != _stable_fingerprint(content)
    ):
        raise ValueError("semantic-view tuning record is not activation-compatible")
    fusion = str(selected.get("fusion"))
    if fusion not in {"global_only", "weighted_rrf", "max_rrf", "rrf_consensus"}:
        raise ValueError("semantic-view tuning record has an unsupported fusion")
    raw_weights = selected.get("weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("semantic-view tuning weights must be a mapping")
    weights = {str(name): float(value) for name, value in raw_weights.items()}
    allowed_roles = {
        "global",
        "action",
        "scene_action",
        "foreground_subject",
        "distinctive_attributes",
        "instance_attributes",
        "relation_relaxed",
    }
    if set(weights) - allowed_roles or any(
        not math.isfinite(value) or value < 0.0 for value in weights.values()
    ):
        raise ValueError("semantic-view tuning weights are invalid")
    consensus_alpha = float(selected.get("consensus_alpha", 0.0))
    if not math.isfinite(consensus_alpha) or not 0.0 <= consensus_alpha <= 1.0:
        raise ValueError("semantic-view consensus alpha is invalid")
    return SemanticViewPolicy(fusion, weights, consensus_alpha)


CachedRankings = Mapping[str, Mapping[str, CachedChannelRanking]]


def observed_threshold_midpoints(
    positive_values: Sequence[float],
    negative_values: Sequence[float],
) -> tuple[float, ...]:
    """Return deterministic class-boundary midpoints from observed train evidence."""

    labeled: list[tuple[float, bool]] = []
    for is_positive, values in ((True, positive_values), (False, negative_values)):
        for raw in values:
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("threshold observations must be finite ratios in [0, 1]")
            labeled.append((value, is_positive))
    if not labeled:
        raise ValueError("threshold tuning requires observed train evidence")
    ordered = sorted(set(labeled))
    midpoints = {
        (left_value + right_value) / 2.0
        for (left_value, left_label), (right_value, right_label) in zip(
            ordered,
            ordered[1:],
            strict=False,
        )
        if left_label != right_label and left_value != right_value
    }
    return tuple(sorted(midpoints))


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
            request_count = int(raw.get("request_count", 1)) if isinstance(raw, Mapping) else 1
            if request_count < 0:
                raise ValueError("cached channel request_count must be non-negative")
            output[query_id][channel] = CachedChannelRanking(
                tuple(dict(result) for result in results), latency_value, request_count
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


def load_intent_policy_override(path: str | Path) -> dict[str, dict[str, float | int]]:
    """Load a fingerprinted intent-policy record produced from locked qrels."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("intent tuning record requires schema_version 1")
    policies = payload.get("intent_policies")
    reports = payload.get("bucket_reports")
    provenance = payload.get("qrels_provenance")
    content = {
        "intent_policies": policies,
        "bucket_reports": reports,
        "qrels_provenance": provenance,
    }
    if (
        not isinstance(policies, Mapping)
        or not isinstance(reports, Mapping)
        or not isinstance(provenance, Mapping)
        or payload.get("selection_fingerprint") != _stable_fingerprint(content)
    ):
        raise ValueError("intent tuning record fingerprint mismatch")
    output: dict[str, dict[str, float | int]] = {}
    for name, raw in policies.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("intent tuning policies must be mappings")
        policy = {
            "rrf_k": int(raw["rrf_k"]),
            "dense_share": float(raw["dense_share"]),
            "text_share": float(raw["text_share"]),
        }
        if policy["rrf_k"] < 0 or not math.isclose(
            float(policy["dense_share"]) + float(policy["text_share"]),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("invalid intent tuning policy")
        output[name] = policy
    return output


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
) -> tuple[float, float, float, float, float]:
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
    r_at_3 = statistics.fmean(
        float(
            metric.first_relevant_rank is not None
            and metric.first_relevant_rank <= 3
        )
        for metric in report.per_query
    )
    return (
        r_at_3,
        report.mrr,
        report.mean_top_k_score,
        report.recall_at[1],
        float(report.latency_ms["p95"] or 0.0),
    )


def _semantic_view_weights_from_train_mrr(
    cached: CachedRankings,
    ground_truth: Sequence[GroundTruthQuery],
    *,
    roles: Sequence[str],
    candidate_depth: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Derive relative view weights only from standalone train-set MRR."""

    standalone: dict[str, float] = {}
    for role in roles:
        _r3, mrr, _mean, _r1, _latency = _evaluate_configuration(
            cached,
            ground_truth,
            rrf_k=1,
            depths={role: candidate_depth},
            weights={role: 1.0},
        )
        standalone[role] = float(mrr)
    total = sum(standalone.values())
    if total <= 0.0:
        equal = 1.0 / len(roles)
        return ({role: equal for role in roles}, standalone)
    return (
        {role: standalone[role] / total for role in roles},
        standalone,
    )


def _semantic_view_predictions(
    cached: CachedRankings,
    ground_truth: Sequence[GroundTruthQuery],
    *,
    roles: Sequence[str],
    runtime_fusion: str,
    rrf_k: int,
    candidate_depth: int,
    weights: Mapping[str, float],
    consensus_alpha: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    latencies: dict[str, float] = {}
    for query in ground_truth:
        query_cache = cached[query.query_id]
        rankings: list[ChannelRanking] = []
        latency = 0.0
        for role in roles:
            value = query_cache.get(role)
            if value is None or weights.get(role, 0.0) <= 0.0:
                continue
            rankings.append(
                ChannelRanking(
                    role,
                    value.results[:candidate_depth],
                    float(weights[role]),
                )
            )
            latency += value.latency_ms
        if runtime_fusion in {"weighted_rrf", "rrf_consensus"}:
            fused = weighted_rrf(rankings, k=rrf_k)
            if runtime_fusion == "rrf_consensus" and consensus_alpha > 0.0:
                denominator = max(1, len(rankings) - 1)
                for item in fused:
                    consensus = max(0, len(item.get("channel_ranks", {})) - 1) / denominator
                    item["score"] = float(item.get("score", 0.0)) * (
                        1.0 + consensus_alpha * consensus
                    )
                fused.sort(
                    key=lambda item: (
                        -float(item.get("score", 0.0)),
                        canonical_key(item),
                    )
                )
        else:
            by_key: dict[tuple[str, int], dict[str, Any]] = {}
            for ranking in rankings:
                for item in weighted_rrf((ranking,), k=rrf_k):
                    key = canonical_key(item)
                    previous = by_key.get(key)
                    if previous is None or float(item.get("score", 0.0)) > float(
                        previous.get("score", 0.0)
                    ):
                        by_key[key] = dict(item)
            fused = sorted(
                by_key.values(),
                key=lambda item: (-float(item.get("score", 0.0)), canonical_key(item)),
            )
        predictions[query.query_id] = fused[:100]
        latencies[query.query_id] = latency
    return predictions, latencies


def tune_semantic_view_policy(
    cached: CachedRankings,
    training_ground_truth: Sequence[GroundTruthQuery],
    validation_ground_truth: Sequence[GroundTruthQuery],
    *,
    roles: Sequence[str],
    candidate_depth: int,
    qrels_provenance: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    consensus_alpha_values: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    max_candidates: int = 50_000,
) -> SemanticViewTuningReport:
    """Tune semantic late fusion without consulting the locked test partition.

    Standalone train MRR determines the only performance-derived weight vector.
    Validation then evaluates every RRF ``k`` from one through the current
    candidate depth for global-only, max-RRF, unweighted/weighted RRF, and
    consensus RRF.  The smallest candidate depth is the final deterministic
    tie-breaker after the declared retrieval and latency objectives.
    """

    selected_roles = tuple(dict.fromkeys(str(role) for role in roles))
    if not selected_roles or "global" not in selected_roles:
        raise ValueError("semantic view tuning requires a global role")
    if candidate_depth < 1:
        raise ValueError("semantic view candidate depth must be positive")
    if not training_ground_truth or not validation_ground_truth:
        raise ValueError("semantic view tuning requires train and validation labels")
    required_ids = {
        query.query_id for query in (*training_ground_truth, *validation_ground_truth)
    }
    missing_queries = sorted(required_ids - set(cached))
    if missing_queries:
        raise ValueError(f"semantic view cache is missing queries: {missing_queries}")
    if any("global" not in cached[query_id] for query_id in required_ids):
        raise ValueError("every semantic view query requires a global ranking")
    required_fingerprints = {"model", "ranking_cache", "query_adapter"}
    if any(
        not isinstance(fingerprints.get(name), str) or not fingerprints.get(name)
        for name in required_fingerprints
    ):
        raise ValueError("semantic view tuning requires model/cache/adapter fingerprints")
    alphas = tuple(dict.fromkeys(float(value) for value in consensus_alpha_values))
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("semantic consensus alpha candidates must be in (0, 1]")

    derived_weights, standalone = _semantic_view_weights_from_train_mrr(
        cached,
        training_ground_truth,
        roles=selected_roles,
        candidate_depth=candidate_depth,
    )
    hypotheses: tuple[tuple[str, str, Mapping[str, float], float], ...] = (
        (
            "global_only",
            "max_rrf",
            {role: float(role == "global") for role in selected_roles},
            0.0,
        ),
        ("max_rrf", "max_rrf", derived_weights, 0.0),
        (
            "unweighted_rrf",
            "weighted_rrf",
            {role: 1.0 / len(selected_roles) for role in selected_roles},
            0.0,
        ),
        ("performance_weighted_rrf", "weighted_rrf", derived_weights, 0.0),
        *tuple(
            ("consensus_rrf", "rrf_consensus", derived_weights, alpha)
            for alpha in alphas
        ),
    )
    candidate_count = len(hypotheses) * sum(range(1, candidate_depth + 1))
    if candidate_count > max_candidates:
        raise ValueError(
            f"semantic view grid has {candidate_count} candidates; maximum is {max_candidates}"
        )

    candidates: list[SemanticViewTuningCandidate] = []
    for depth in range(1, candidate_depth + 1):
        for rrf_k in range(1, depth + 1):
            for policy, runtime_fusion, weights, alpha in hypotheses:
                predictions, latencies = _semantic_view_predictions(
                    cached,
                    validation_ground_truth,
                    roles=selected_roles,
                    runtime_fusion=runtime_fusion,
                    rrf_k=rrf_k,
                    candidate_depth=depth,
                    weights=weights,
                    consensus_alpha=alpha,
                )
                report = evaluate_queries(
                    predictions,
                    validation_ground_truth,
                    latencies_ms=latencies,
                )
                r_at_3 = statistics.fmean(
                    float(
                        metric.first_relevant_rank is not None
                        and metric.first_relevant_rank <= 3
                    )
                    for metric in report.per_query
                )
                candidates.append(
                    SemanticViewTuningCandidate(
                        policy=policy,
                        runtime_fusion=runtime_fusion,
                        rrf_k=rrf_k,
                        candidate_depth=depth,
                        weights=dict(weights),
                        consensus_alpha=alpha,
                        r_at_3=r_at_3,
                        mrr=report.mrr,
                        r_at_1=report.recall_at[1],
                        mean_top_k_score=report.mean_top_k_score,
                        p95_latency_ms=float(report.latency_ms["p95"] or 0.0),
                    )
                )
    selected = min(
        candidates,
        key=lambda value: (
            -value.r_at_3,
            -value.mrr,
            -value.r_at_1,
            -value.mean_top_k_score,
            value.p95_latency_ms,
            value.candidate_depth,
            value.rrf_k,
            value.policy,
            value.consensus_alpha,
        ),
    )
    return SemanticViewTuningReport(
        selected=selected,
        standalone_train_mrr=standalone,
        candidates_evaluated=len(candidates),
        qrels_provenance=dict(qrels_provenance),
        fingerprints={str(name): str(value) for name, value in fingerprints.items()},
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
                "primary": "r_at_3",
                "tie_1": "mrr",
                "tie_2": "r_at_1",
                "tie_3": "mean_top_k_score",
                "tie_4": "p95_latency_ms",
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
                r_at_3, mrr, score, r_at_1, p95 = _evaluate_configuration(
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
                        r_at_3,
                        mrr,
                        score,
                        r_at_1,
                        p95,
                    )
                )
    best = min(
        candidates,
        key=lambda value: (
            -value.r_at_3,
            -value.mrr,
            -value.r_at_1,
            -value.mean_top_k_score,
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
        r_at_3, mrr, score, r_at_1, p95 = _evaluate_configuration(
            cached,
            ground_truth,
            rrf_k=best.rrf_k,
            depths=best.depths,
            weights=best.weights,
            channels=active,
        )
        ablations.append(
            AblationResult(tuple(active), r_at_3, mrr, score, r_at_1, p95)
        )
    return RetrievalTuningReport(
        best.rrf_k,
        best.depths,
        best.weights,
        False,
        "selected on labeled cached rankings",
        len(candidates),
        {
            "primary": "r_at_3",
            "tie_1": "mrr",
            "tie_2": "r_at_1",
            "tie_3": "mean_top_k_score",
            "tie_4": "p95_latency_ms",
        },
        {
            "r_at_3": best.r_at_3,
            "mrr": best.mrr,
            "mean_top_k_score": best.mean_top_k_score,
            "r_at_1": best.r_at_1,
            "p95_latency_ms": best.p95_latency_ms,
        },
        tuple(ablations),
    )


def tune_intent_policies(
    cached: CachedRankings,
    ground_truth: Sequence[GroundTruthQuery],
    *,
    dense_channels: Sequence[str],
    default_rrf_k: int,
    default_depths: Mapping[str, int],
    default_weights: Mapping[str, float],
    rrf_k_values: Sequence[int] = (20, 40, 60, 80),
    depth_values: Mapping[str, Sequence[int]] | None = None,
    weight_values: Mapping[str, Sequence[float]] | None = None,
    max_candidates: int = 50_000,
    qrels_provenance: Mapping[str, Any] | None = None,
) -> IntentPolicyTuningReport:
    """Tune three runtime intent buckets from immutable cached rankings."""

    routing = {
        "visual_dense": "visual_dense",
        "lexical_dominant": "lexical_dominant",
        "absolute_spatial": "spatial_color",
        "relative_spatial": "spatial_color",
        "color": "spatial_color",
        "negation": "spatial_color",
    }
    grouped: dict[str, list[GroundTruthQuery]] = {}
    for query in ground_truth:
        grouped.setdefault(routing.get(query.intent_class or "", "visual_dense"), []).append(
            query
        )
    dense = set(dense_channels)
    policies: dict[str, dict[str, float | int]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for bucket in ("visual_dense", "lexical_dominant", "spatial_color"):
        queries = grouped.get(bucket, [])
        report = tune_cached_rankings(
            cached,
            queries,
            default_rrf_k=default_rrf_k,
            default_depths=default_depths,
            default_weights=default_weights,
            rrf_k_values=rrf_k_values,
            depth_values=depth_values,
            weight_values=weight_values,
            max_candidates=max_candidates,
        )
        dense_weight = sum(
            value for name, value in report.selected_weights.items() if name in dense
        )
        text_weight = sum(
            value for name, value in report.selected_weights.items() if name not in dense
        )
        total = dense_weight + text_weight
        dense_share = dense_weight / total if total > 0 else 0.5
        policies[bucket] = {
            "rrf_k": report.selected_rrf_k,
            "dense_share": round(dense_share, 9),
            "text_share": round(1.0 - dense_share, 9),
        }
        reports[bucket] = report.to_dict()
    return IntentPolicyTuningReport(policies, reports, dict(qrels_provenance or {}))


def persist_tuning_report(
    report: (
        RetrievalTuningReport
        | IntentPolicyTuningReport
        | SemanticViewTuningReport
    ),
    destination: str | Path,
) -> Path:
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
    "IntentPolicyTuningReport",
    "RetrievalTuningReport",
    "RetrievalTuningOverride",
    "SemanticViewPolicy",
    "SemanticViewTuningCandidate",
    "SemanticViewTuningReport",
    "TuningCandidate",
    "load_cached_rankings",
    "load_intent_policy_override",
    "load_semantic_view_policy",
    "load_tuning_override",
    "observed_threshold_midpoints",
    "persist_tuning_report",
    "tune_cached_rankings",
    "tune_intent_policies",
    "tune_semantic_view_policy",
]
