#!/usr/bin/env python3
"""Run one rank-invariant KIS shadow feature over frozen unjudged proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QueryRequest  # noqa: E402
from src.retrieval.candidate_rescue import candidate_concentration  # noqa: E402
from src.retrieval.fusion import canonical_key  # noqa: E402
from src.runtime import build_runtime  # noqa: E402
from src.utils.io_helpers import load_config  # noqa: E402

FEATURES = ("off", "candidate_rescue", "video_local", "ocr_quality")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _queries(path: Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            query_id = str(value.get("query_id", "")).strip()
            text = str(value.get("text", "")).strip()
            if not query_id or not text or query_id in seen:
                raise ValueError(f"proxy query row {line_number} is invalid")
            seen.add(query_id)
            rows.append({"query_id": query_id, "text": text})
    return tuple(rows)


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-queries", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--feature", choices=FEATURES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config()
    retrieval = config["retrieval"]
    for name in ("candidate_rescue", "video_local", "ocr_quality"):
        retrieval[name]["mode"] = "shadow" if args.feature == name else "off"
    retrieval["semantic_views"]["fusion"] = "global_only"
    retrieval["interaction_log"]["mode"] = "off"
    runtime = build_runtime(config)
    pipeline = runtime.pipeline
    trajectory_report: Mapping[str, Any] | None = None
    if args.feature == "video_local":
        trajectories = pipeline.video_local_trajectories
        if trajectories is None:
            raise RuntimeError("video-local trajectory cache is unavailable")
        trajectory_report = trajectories.warm(args.batch_id)

    query_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, query in enumerate(_queries(args.proxy_queries), start=1):
        request = QueryRequest(
            query_id=query["query_id"],
            original_text=query["text"],
            mode="automatic",
            batch_filters=(args.batch_id,),
        )
        started = time.monotonic()
        results = pipeline.search(
            request,
            limit=100,
            refine_raw=False,
            advanced_rerank=False,
        )
        latency_ms = (time.monotonic() - started) * 1_000
        latencies.append(latency_ms)
        concentration = candidate_concentration(results)
        identities = []
        for result in results:
            try:
                video_id, frame_idx = canonical_key(result)
            except ValueError:
                continue
            identities.append(f"{video_id}:{frame_idx}")
        diagnostics = dict(pipeline.last_query_processing)
        if args.feature == "ocr_quality":
            diagnostics["ocr_quality"] = dict(
                pipeline.last_channel_statuses.get("ocr", {}).get("ocr_quality", {})
            )
        query_rows.append(
            {
                "query_id": query["query_id"],
                "latency_ms": round(latency_ms, 3),
                "result_ids": identities,
                "concentration": concentration.to_dict(),
                "feature_diagnostics": _json_safe(diagnostics.get(args.feature, {})),
                "channel_statuses": _json_safe(pipeline.last_channel_statuses),
            }
        )
        print(
            json.dumps(
                {
                    "feature": args.feature,
                    "query_id": query["query_id"],
                    "completed": index,
                    "total": 27,
                    "latency_ms": round(latency_ms, 3),
                }
            ),
            flush=True,
        )

    summary: dict[str, Any] = {
        "query_count": len(query_rows),
        "warm_latency_p50_ms": round(_percentile(latencies[1:], 0.50), 3),
        "warm_latency_p95_ms": round(_percentile(latencies[1:], 0.95), 3),
        "mean_distinct_videos_top_50": round(
            statistics.fmean(
                row["concentration"]["unique_videos"] for row in query_rows
            ),
            3,
        ),
        "mean_top_video_share_top_50": round(
            statistics.fmean(
                row["concentration"]["top_video_share"] for row in query_rows
            ),
            6,
        ),
        "mean_video_hhi_top_50": round(
            statistics.fmean(row["concentration"]["video_hhi"] for row in query_rows),
            6,
        ),
    }
    if args.feature == "candidate_rescue":
        summary["triggered_queries"] = sum(
            bool(row["feature_diagnostics"].get("triggered")) for row in query_rows
        )
    elif args.feature == "video_local":
        summary["scored_queries"] = sum(
            int(row["feature_diagnostics"].get("scored_video_count", 0)) > 0
            for row in query_rows
        )
    elif args.feature == "ocr_quality":
        eligible = [
            row["feature_diagnostics"]
            for row in query_rows
            if int(row["feature_diagnostics"].get("evaluated_candidate_count", 0)) > 0
        ]
        evaluated_count = sum(
            int(row.get("evaluated_candidate_count", 0)) for row in eligible
        )
        summary["conditional_feature_coverage"] = round(
            sum(int(row.get("feature_count", 0)) for row in eligible)
            / max(1, evaluated_count),
            6,
        )
    payload = {
        "schema_version": 1,
        "evaluation_kind": "proxy_unjudged_shadow",
        "activation_eligible": False,
        "rank_mutation_allowed": False,
        "feature": args.feature,
        "batch_id": args.batch_id,
        "query_source_sha256": _sha256(args.proxy_queries),
        "pipeline_fingerprint": pipeline.fingerprint,
        "trajectory_cache": _json_safe(trajectory_report),
        "summary": summary,
        "queries": query_rows,
    }
    output = args.output.resolve()
    _atomic_json(output, payload)
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
