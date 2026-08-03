"""Measure genuine cold and post-warm Textual KIS latency in fresh processes."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QueryRequest
from src.runtime import build_runtime
from src.utils.io_helpers import load_config
from src.warmup import safe_resource_snapshot


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency values are empty")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _initialize_runtime() -> tuple[Any, float]:
    started = time.perf_counter()
    context = build_runtime(load_config())
    context.ledger.initialize()
    if context.artifact_repository is not None:
        context.artifact_repository.initialize()
    return context, (time.perf_counter() - started) * 1_000


def _run_query(
    context: Any,
    *,
    text: str,
    batch_id: str,
    query_id: str,
    refine_raw: bool,
) -> tuple[float, int]:
    request = QueryRequest(
        query_id=query_id,
        original_text=text,
        mode="automatic",
        batch_filters=context.validate_batch_filters((batch_id,)),
        limit=100,
    )
    started = time.perf_counter()
    hits = context.pipeline.search(request, limit=100, refine_raw=refine_raw)
    return (time.perf_counter() - started) * 1_000, len(hits)


def _worker(payload: dict[str, Any], output: Any) -> None:
    try:
        context, build_ms = _initialize_runtime()
        mode = str(payload["mode"])
        queries = list(payload["queries"])
        batch_id = str(payload["batch_id"])
        refine_raw = bool(payload["refine_raw"])
        pipeline_fingerprint = str(context.pipeline.fingerprint)
        if mode == "cold":
            latency, hit_count = _run_query(
                context,
                text=queries[0],
                batch_id=batch_id,
                query_id="benchmark-cold-0",
                refine_raw=refine_raw,
            )
            output.put(
                {
                    "runtime_build_ms": round(build_ms, 3),
                    "cold_latency_ms": round(latency, 3),
                    "cold_hit_count": hit_count,
                    "error_count": 0,
                    "pipeline_fingerprint": pipeline_fingerprint,
                    "resources": safe_resource_snapshot(),
                }
            )
            return

        warmup = getattr(context, "warmup", None)
        if warmup is None:
            raise RuntimeError("warm-up service is unavailable")
        if not refine_raw:
            warmup.include_raw_refiner = False
        warm_started = time.perf_counter()
        warmup.start()
        warm_snapshot = warmup.wait(
            float(payload["warmup_timeout_seconds"]), workflow="kis"
        )
        warmup_ms = (time.perf_counter() - warm_started) * 1_000
        kis = warm_snapshot.get("workflows", {}).get("kis", {})
        if not bool(kis.get("usable", warm_snapshot.get("search_usable", False))):
            raise RuntimeError("warm-up did not reach a usable state")

        first_latency, first_hits = _run_query(
            context,
            text=queries[0],
            batch_id=batch_id,
            query_id="benchmark-first-post-warm",
            refine_raw=refine_raw,
        )
        samples: list[dict[str, Any]] = []
        latencies: list[float] = []
        iterations = int(payload["iterations"])
        for index in range(iterations):
            text = queries[index % len(queries)]
            try:
                latency, hits = _run_query(
                    context,
                    text=text,
                    batch_id=batch_id,
                    query_id=f"benchmark-warm-{index}",
                    refine_raw=refine_raw,
                )
                latencies.append(latency)
                samples.append(
                    {"iteration": index + 1, "latency_ms": round(latency, 3), "hits": hits}
                )
            except Exception as error:
                samples.append(
                    {
                        "iteration": index + 1,
                        "error": type(error).__name__,
                        "hits": 0,
                    }
                )
        if not latencies:
            raise RuntimeError("every warm benchmark query failed")
        error_count = sum("error" in sample for sample in samples)
        output.put(
            {
                "runtime_build_ms": round(build_ms, 3),
                "pipeline_fingerprint": pipeline_fingerprint,
                "warmup_ms": round(warmup_ms, 3),
                "warmup": warm_snapshot,
                "first_post_warm_ms": round(first_latency, 3),
                "first_post_warm_hits": first_hits,
                "successful_iterations": len(latencies),
                "error_count": error_count,
                "total_hit_count": sum(int(sample.get("hits", 0)) for sample in samples),
                "warm_latency_ms": {
                    "mean": round(statistics.fmean(latencies), 3),
                    "p50": round(_percentile(latencies, 0.50), 3),
                    "p90": round(_percentile(latencies, 0.90), 3),
                    "p95": round(_percentile(latencies, 0.95), 3),
                    "min": round(min(latencies), 3),
                    "max": round(max(latencies), 3),
                },
                "samples": samples,
                "resources": safe_resource_snapshot(),
            }
        )
    except Exception as error:
        output.put({"error": type(error).__name__})


def _spawn_worker(payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(payload, output))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        return {"error": "WorkerTimeout"}
    try:
        result = output.get(timeout=2)
    except Empty:
        return {"error": "WorkerNoResult", "exit_code": process.exitcode}
    return dict(result)


def _load_queries(path: Path | None, direct: list[str]) -> list[str]:
    values = [item.strip() for item in direct if item.strip()]
    if path is not None:
        if path.suffix.casefold() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("JSON query file must contain an array")
            for item in payload:
                value = item.get("text") if isinstance(item, dict) else item
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
        else:
            values.extend(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    if not values:
        raise ValueError("at least one --query or --query-file entry is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--refine-raw", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    queries = _load_queries(args.query_file, args.query)
    config = load_config()
    warmup_timeout = float(config.get("warmup", {}).get("startup_budget_seconds", 90))
    common = {
        "batch_id": args.batch_id,
        "queries": queries,
        "refine_raw": args.refine_raw,
        "iterations": args.iterations,
        "warmup_timeout_seconds": warmup_timeout,
    }
    cold = _spawn_worker(
        {**common, "mode": "cold"}, timeout_seconds=warmup_timeout + 120
    )
    warm = _spawn_worker(
        {**common, "mode": "warm"}, timeout_seconds=warmup_timeout + 300
    )
    result = {
        "profile": str(config.get("profile", "laptop")),
        "batch_id": args.batch_id,
        "pipeline_fingerprint": warm.get(
            "pipeline_fingerprint", cold.get("pipeline_fingerprint")
        ),
        "iterations": args.iterations,
        "refine_raw": args.refine_raw,
        "cold": cold,
        "warm": warm,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if "error" in cold or "error" in warm else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
