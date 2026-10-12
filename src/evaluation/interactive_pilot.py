"""Separate local pilot gate. Does not change the locked 120-qrels validator.

Positive anchors are not exhaustive judgments. This gate cannot substitute KIS
queries for actual multi-turn/feedback labels or claim an independent test.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def anchor_metrics(rows: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    expected = {q["query_id"]: q for q in queries}
    hits = {k: [] for k in (1, 3, 5)}
    videos = {k: [] for k in (1, 3, 5)}
    reciprocal: list[float] = []
    seen: set[str] = set()
    for row in rows:
        qid = row["query_id"]
        if qid in seen or qid not in expected:
            raise ValueError("rows must contain one ranking per selected query")
        seen.add(qid)
        intervals = expected[qid]["intervals"]
        positive_videos = {i["video_id"] for i in intervals}
        ranking = row.get("hits", ())
        correct = [any(h["video_id"] == i["video_id"] and i["start_frame_idx"] <= h["source_frame_idx"] <= i["end_frame_idx"]
                       and (not expected[qid].get("batch_id") or h.get("batch_id") == expected[qid]["batch_id"])
                       for i in intervals) for h in ranking]
        reciprocal.append(next((1 / (i + 1) for i, ok in enumerate(correct) if ok), 0.))
        for k in hits:
            hits[k].append(float(any(correct[:k])))
            videos[k].append(float(any(h["video_id"] in positive_videos for h in ranking[:k])))
    if seen != set(expected) or not seen:
        raise ValueError("incomplete pilot ranking")
    result = {f"hit_at_{k}": statistics.fmean(v) for k, v in hits.items()}
    result.update({f"video_hit_at_{k}": statistics.fmean(v) for k, v in videos.items()})
    result["mrr"] = statistics.fmean(reciprocal)
    result["mean_top_k"] = statistics.fmean(result[f"hit_at_{k}"] for k in hits)
    return result


def local_release_gate(
    baseline: Mapping[str, Any], trial: Mapping[str, Any], *, feature: str,
    data_kind: str, parameters_locked: bool, outside_scope_invariant: bool,
) -> dict[str, Any]:
    """Fail closed on missing/failed measurements; only governs automatic use."""
    required_kind = {"feedback": "human_multiturn", "temporal_sequence": "ordered_events"}.get(feature)
    checks: dict[str, bool] = {
        "appropriate_feature_data": required_kind is not None and data_kind == required_kind,
        "parameters_locked_before_test": parameters_locked is True,
        "outside_scope_rank_invariant": outside_scope_invariant is True,
        "healthy_baseline_and_trial": baseline.get("health_ok") is True and trial.get("health_ok") is True,
        "no_errors": all(type(v.get("error_count")) is int and v["error_count"] == 0 for v in (baseline, trial)),
        "same_query_fingerprint": bool(baseline.get("query_sha256")) and baseline.get("query_sha256") == trial.get("query_sha256"),
        "same_split_fingerprint": bool(baseline.get("split_sha256")) and baseline.get("split_sha256") == trial.get("split_sha256"),
    }
    for name in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_top_k", "warm_p95_ms", "rss_bytes"):
        a, b = baseline.get(name), trial.get(name)
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in (a, b))
        if name in {"warm_p95_ms", "rss_bytes"}:
            checks[name] = bool(valid and a > 0 and 0 < b <= a * 1.2)
        elif name == "mean_top_k":
            checks[name] = bool(valid and 0 <= a <= b <= 1 and b >= a + .01)
        else:
            checks[name] = bool(valid and 0 <= a <= b <= 1)
    return {"schema_version": 1, "evaluation_kind": "local_regression_pilot",
            "not_independent_test": True, "automatic_activation_allowed": all(checks.values()),
            "feature": feature, "checks": checks}
