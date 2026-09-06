#!/usr/bin/env python3
"""Run the reviewed multi-instance query through a non-persistent shadow policy."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

from src.contracts import QueryRequest
from src.retrieval.fusion import canonical_key
from src.runtime import build_pipeline
from src.warmup import safe_resource_snapshot

QUERY = (
    "Footage capturing vehicles wading through flood water, where a yellow car, "
    "a red car, and a black car prepare one after another to cross a bridge."
)
TARGET = ("L21_V003", 26220)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main(argv: list[str] | None = None) -> int:
    """Evaluate a candidate semantic policy without changing config or stores."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--fusion",
        choices=("max_rrf", "weighted_rrf", "rrf_consensus"),
        default="max_rrf",
    )
    args = parser.parse_args(argv)
    if args.iterations < 3:
        parser.error("--iterations must be at least three for stability gating")

    pipeline = build_pipeline()
    configured_fusion = pipeline.semantic_view_fusion
    pipeline.semantic_view_fusion = args.fusion
    request = QueryRequest(
        query_id="reviewed-multi-vehicle-shadow",
        original_text=QUERY,
        query_language="en",
        mode="automatic",
        batch_filters=(args.batch_id,),
    )
    rows: list[dict[str, Any]] = []
    for iteration in range(1, args.iterations + 1):
        started = time.perf_counter()
        hits = pipeline.search(request, limit=10, refine_raw=False)
        latency_ms = (time.perf_counter() - started) * 1_000
        top_ten = tuple(canonical_key(hit) for hit in hits)
        target_rank = next(
            (rank for rank, key in enumerate(top_ten, start=1) if key == TARGET),
            None,
        )
        rows.append(
            {
                "iteration": iteration,
                "target_rank": target_rank,
                "latency_ms": round(latency_ms, 3),
                "rss_bytes": safe_resource_snapshot()["process_rss_bytes"],
                "top_ten": [list(key) for key in top_ten],
                "tier2_state": pipeline.last_channel_statuses.get(
                    "advanced_rerank", {}
                ).get("state"),
                "channel_statuses": pipeline.last_channel_statuses,
                "channel_errors": pipeline.last_channel_errors,
                "semantic_view_rankings": pipeline.last_query_processing.get(
                    "semantic_view_rankings", {}
                ),
                "rank_trace": pipeline.last_query_processing.get("rank_trace", {}),
            }
        )
    latencies = [float(row["latency_ms"]) for row in rows]
    stable_top_ten = len(
        {
            json.dumps(row["top_ten"], separators=(",", ":"))
            for row in rows
        }
    ) == 1
    stable_tier2 = len({row["tier2_state"] for row in rows}) == 1
    passed = (
        all(
            isinstance(row["target_rank"], int) and int(row["target_rank"]) <= 3
            for row in rows
        )
        and stable_top_ten
        and stable_tier2
    )
    report = {
        "schema_version": 1,
        "shadow_only": True,
        "persistent_state_changed": False,
        "configured_fusion": configured_fusion,
        "candidate_fusion": args.fusion,
        "target": {"video_id": TARGET[0], "source_frame_idx": TARGET[1]},
        "passed": passed,
        "stable_top_ten": stable_top_ten,
        "stable_tier2_state": stable_tier2,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "peak_rss_bytes": max(
            (
                int(row["rss_bytes"])
                for row in rows
                if isinstance(row["rss_bytes"], int)
            ),
            default=None,
        ),
        "runs": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
