"""Benchmark spatial geometry separately from optional live Tier-2 I/O.

Synthetic geometry is a performance benchmark, not a retrieval-accuracy claim.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

from src.query.query_decomposer import RelativeSpatialConstraint
from src.retrieval.spatial_graph import SpatialGraphEvaluator
from src.retrieval.spatial_rerank import ObjectDetection

_SYNTHETIC_COUNT_MIN = 1_000
_LIVE_QUERIES = (
    "a yellow cyclist left of a blue car",
    "a person near a car",
    "a person between a car and a bus",
    "a sign overlapping a building",
    "a cyclist in front of a car",
    "một người đứng phía sau xe ô tô",
)


@dataclass(frozen=True, slots=True)
class SyntheticGraph:
    constraints: tuple[RelativeSpatialConstraint, ...]
    detections: tuple[ObjectDetection, ...]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _graph(index: int) -> SyntheticGraph:
    shift = (index % 97) / 10_000.0
    detections = (
        ObjectDetection("person", 0.95, (0.35, 0.10 + shift, 0.80, 0.30 + shift)),
        ObjectDetection("car", 0.92, (0.45, 0.68 - shift, 0.72, 0.90 - shift)),
        ObjectDetection("bus", 0.90, (0.15, 0.40, 0.35, 0.62)),
        ObjectDetection("person", 0.80, (0.10, 0.72, 0.28, 0.90)),
        ObjectDetection("car", 0.78, (0.70, 0.05, 0.92, 0.28)),
    )
    relation = ("left_of", "near", "far_from", "in_front_of")[index % 4]
    constraints = (
        RelativeSpatialConstraint("person", relation, "car", relation, "p", "c"),
        RelativeSpatialConstraint("person", "above", "bus", "above", "p", "b"),
    )
    return SyntheticGraph(constraints, detections)


def benchmark_geometry(count: int) -> dict[str, Any]:
    if count < _SYNTHETIC_COUNT_MIN:
        raise ValueError(f"count must be at least {_SYNTHETIC_COUNT_MIN}")
    evaluator = SpatialGraphEvaluator()
    evaluator.evaluate(_graph(0).constraints, _graph(0).detections, detections_are_nms=True)
    latencies: list[float] = []
    failures = 0
    for index in range(count):
        graph = _graph(index)
        started = time.perf_counter()
        match = evaluator.evaluate(
            graph.constraints,
            graph.detections,
            detections_are_nms=True,
        )
        latencies.append((time.perf_counter() - started) * 1_000)
        if match.state != "verified" or match.score is None or not math.isfinite(match.score):
            failures += 1
    p95 = _percentile(latencies, 0.95)
    return {
        "label": "synthetic_geometry_performance_not_accuracy",
        "candidate_graphs": count,
        "failures": failures,
        "p50_ms_per_candidate": round(statistics.median(latencies), 6),
        "p95_ms_per_candidate": round(p95, 6),
        "max_ms_per_candidate": round(max(latencies), 6),
        "hard_limit_ms_per_candidate": 1.0,
        "hard_gate_passed": failures == 0 and p95 < 1.0,
    }


def benchmark_live(api_url: str, timeout: float) -> dict[str, Any]:
    import requests

    latencies: list[float] = []
    tier: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, query in enumerate(_LIVE_QUERIES, start=1):
        started = time.perf_counter()
        try:
            response = requests.post(
                api_url.rstrip("/") + "/v1/kis/search",
                json={
                    "query_id": f"spatial-live-{index}",
                    "original_text": query,
                    "batch_filters": ["aic25-b1-reference"],
                    "limit": 100,
                    "refine_raw": False,
                    "precision_profile": "advanced",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            processing = body.get("query_processing", {})
            advanced = processing.get("advanced_rerank", {}) if isinstance(processing, dict) else {}
            tier.append(
                {
                    key: advanced.get(key)
                    for key in (
                        "spatial_latency_ms", "relation_latency_ms", "geometry_latency_ms",
                        "negation_latency_ms", "color_latency_ms", "total_latency_ms",
                        "object_cache", "color_cache",
                    )
                    if key in advanced
                }
            )
        except (requests.RequestException, ValueError, TypeError) as error:
            failures.append(type(error).__name__)
        latencies.append((time.perf_counter() - started) * 1_000)
    return {
        "label": "live_unlabeled_latency_not_accuracy",
        "request_count": len(_LIVE_QUERIES),
        "failure_count": len(failures),
        "failure_types": sorted(set(failures)),
        "http_p50_ms": round(statistics.median(latencies), 3),
        "http_p95_ms": round(_percentile(latencies, 0.95), 3),
        "tier_diagnostics": tier,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--hard-gate", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        output: dict[str, Any] = {"geometry": benchmark_geometry(args.count)}
    except ValueError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    if args.live:
        output["live"] = benchmark_live(args.api_url, args.timeout)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if args.hard_gate and not output["geometry"]["hard_gate_passed"]:
        return 1
    if args.live and output["live"]["failure_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())