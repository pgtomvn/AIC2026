#!/usr/bin/env python3
"""Evaluate single-channel and weighted-RRF baselines from one ranking cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.qrels import (  # noqa: E402
    build_locked_qrels_split,
    load_qrels_pack,
    qrels_coverage_report,
    validate_production_qrels,
)
from src.evaluation.textual_kis import evaluate_queries  # noqa: E402
from src.evaluation.tuning import CachedChannelRanking, load_cached_rankings  # noqa: E402
from src.retrieval.candidate_rescue import candidate_concentration  # noqa: E402
from src.retrieval.fusion import ChannelRanking, canonical_key, weighted_rrf  # noqa: E402
from src.utils.io_helpers import load_config  # noqa: E402


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proxy_query_ids(path: Path) -> tuple[str, ...]:
    values: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            if not query_id or query_id in values:
                raise ValueError("proxy query file has an invalid or duplicate query_id")
            values.append(query_id)
    if not values:
        raise ValueError("proxy query file is empty")
    return tuple(values)


def _diagnostics(
    predictions: dict[str, list[dict]],
    latencies: dict[str, float],
    request_counts: dict[str, int],
) -> dict[str, float]:
    distinct = []
    top_video_shares = []
    hhi_values = []
    for results in predictions.values():
        videos = set()
        for result in results[:50]:
            try:
                video_id, _frame_idx = canonical_key(result)
            except ValueError:
                continue
            videos.add(video_id)
        distinct.append(len(videos))
        concentration = candidate_concentration(results)
        top_video_shares.append(concentration.top_video_share)
        hhi_values.append(concentration.video_hhi)
    latency_values = list(latencies.values())
    request_values = list(request_counts.values())
    return {
        "mean_distinct_videos_top_50": statistics.fmean(distinct) if distinct else 0.0,
        "mean_top_video_share_top_50": (
            statistics.fmean(top_video_shares) if top_video_shares else 0.0
        ),
        "mean_video_hhi_top_50": statistics.fmean(hhi_values) if hhi_values else 0.0,
        "warm_latency_p50_ms": _percentile(latency_values, 0.50),
        "warm_latency_p95_ms": _percentile(latency_values, 0.95),
        "mean_external_requests": (
            statistics.fmean(request_values) if request_values else 0.0
        ),
    }


def _evaluate_policy(
    *,
    query_ids: tuple[str, ...],
    ground_truth_by_id: dict | None,
    cached: dict[str, dict[str, CachedChannelRanking]],
    selected_channels: tuple[str, ...],
    weights: dict[str, float],
    rrf_k: int,
) -> dict:
    predictions: dict[str, list[dict]] = {}
    latencies: dict[str, float] = {}
    requests: dict[str, int] = {}
    for query_id in query_ids:
        available = cached.get(query_id, {})
        rankings = [
            ChannelRanking(
                channel,
                available[channel].results,
                float(weights.get(channel, 1.0)),
            )
            for channel in selected_channels
            if channel in available and available[channel].results
        ]
        predictions[query_id] = weighted_rrf(rankings, k=rrf_k) if rankings else []
        latencies[query_id] = sum(
            available[channel].latency_ms
            for channel in selected_channels
            if channel in available
        )
        requests[query_id] = sum(
            available[channel].request_count
            for channel in selected_channels
            if channel in available
        )
    if ground_truth_by_id is None:
        result_counts = [len(values) for values in predictions.values()]
        report = {
            "evaluation_kind": "proxy_unjudged",
            "judged_metrics_available": False,
            "query_count": len(query_ids),
            "mean_returned_results": (
                statistics.fmean(result_counts) if result_counts else 0.0
            ),
        }
    else:
        report = evaluate_queries(
            predictions,
            tuple(ground_truth_by_id[query_id] for query_id in query_ids),
            latencies_ms=latencies,
        ).to_dict()
    report["operational_diagnostics"] = _diagnostics(predictions, latencies, requests)
    report["channels"] = list(selected_channels)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    query_source = parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--qrels-manifest", type=Path)
    query_source.add_argument("--proxy-queries", type=Path)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/kis-baselines.json"),
    )
    args = parser.parse_args(argv)
    ranking_payload = json.loads(args.rankings.read_text(encoding="utf-8-sig"))
    if args.qrels_manifest is not None:
        pack = load_qrels_pack(args.qrels_manifest)
        validate_production_qrels(pack)
        locked = build_locked_qrels_split(
            pack, seed=int(pack.manifest.split_metadata["seed"])
        )
        query_ids_by_split = {
            "train": locked.train_query_ids,
            "validation": locked.validation_query_ids,
            "test": locked.test_query_ids,
            "all": tuple(sorted(query.query_id for query in pack.queries)),
        }
        query_ids = query_ids_by_split[args.split]
        ground_truth: dict | None = {
            query.query_id: query for query in pack.queries
        }
        coverage: dict = qrels_coverage_report(pack, split=locked)
        qrels_content_sha256: str | None = pack.manifest.content_sha256
        locked_test_sha256: str | None = locked.test_sha256
        evaluation_kind = "locked_qrels"
    else:
        query_ids = _proxy_query_ids(args.proxy_queries)
        ground_truth = None
        coverage = {
            "query_count": len(query_ids),
            "judged": False,
            "activation_eligible": False,
        }
        qrels_content_sha256 = None
        locked_test_sha256 = None
        evaluation_kind = "proxy_unjudged"
        if ranking_payload.get("evaluation_kind") != evaluation_kind:
            raise ValueError("ranking cache is not an unjudged proxy cache")
        if ranking_payload.get("query_source_sha256") != _sha256(args.proxy_queries):
            raise ValueError("ranking cache proxy source fingerprint does not match")
    cached = load_cached_rankings(args.rankings)
    missing = [query_id for query_id in query_ids if query_id not in cached]
    if missing:
        raise ValueError("ranking cache does not cover the selected qrels split")
    channels = tuple(sorted(set().union(*(set(cached[value]) for value in query_ids))))
    retrieval = load_config()["retrieval"]
    configured_weights = {
        str(name): float(weight)
        for name, weight in retrieval.get("channel_weights", {}).items()
    }
    policies = {
        f"single:{channel}": _evaluate_policy(
            query_ids=query_ids,
            ground_truth_by_id=ground_truth,
            cached=cached,
            selected_channels=(channel,),
            weights={channel: 1.0},
            rrf_k=int(retrieval["rrf_k"]),
        )
        for channel in channels
    }
    policies["fusion:configured"] = _evaluate_policy(
        query_ids=query_ids,
        ground_truth_by_id=ground_truth,
        cached=cached,
        selected_channels=channels,
        weights=configured_weights,
        rrf_k=int(retrieval["rrf_k"]),
    )
    payload = {
        "schema_version": 1,
        "evaluation_kind": evaluation_kind,
        "activation_eligible": evaluation_kind == "locked_qrels",
        "split": args.split if evaluation_kind == "locked_qrels" else "all",
        "qrels_content_sha256": qrels_content_sha256,
        "locked_test_sha256": locked_test_sha256,
        "qrels_coverage": coverage,
        "policies": policies,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "policies": len(policies)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
