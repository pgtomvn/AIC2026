"""Measure cold and warm TRAKE latency in isolated local processes."""

from __future__ import annotations

import argparse
import hashlib
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

from src.contracts import TRAKEQueryRequest, TRAKEStage
from src.retrieval.trake_alignment import TRAKE_ALIGNMENT_VERSION
from src.retrieval.trake_trajectory_cache import TRAKE_TRAJECTORY_CACHE_VERSION
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


def _pipeline_fingerprint(config: dict[str, Any], batch_id: str) -> str:
    payload = {
        "version": "trake-benchmark-v2",
        "alignment_version": TRAKE_ALIGNMENT_VERSION,
        "trajectory_cache_version": TRAKE_TRAJECTORY_CACHE_VERSION,
        "profile": config.get("profile"),
        "batch_id": batch_id,
        "trake": config.get("retrieval", {}).get("trake", {}),
        "visual": {
            name: {
                key: model.get(key)
                for key in ("model_name", "pretrained", "revision", "embedding_dim")
            }
            for name in ("visual", "organizer_clip")
            if isinstance((model := config.get("models", {}).get(name)), dict)
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    batch_id: str,
    global_description: str,
    stages: list[str],
    query_id: str,
    top_k: int,
) -> tuple[float, int]:
    request = TRAKEQueryRequest(
        query_id=query_id,
        global_description=global_description,
        stages=tuple(
            TRAKEStage(stage_index=index, description=description)
            for index, description in enumerate(stages, start=1)
        ),
        batch_filters=context.validate_batch_filters((batch_id,)),
        top_k=top_k,
    )
    started = time.perf_counter()
    response = context.trake_pipeline.search(request, default_batch_id=batch_id)
    return (time.perf_counter() - started) * 1_000, len(response.results)


def _summary(latencies: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(latencies), 3),
        "p50": round(_percentile(latencies, 0.50), 3),
        "p90": round(_percentile(latencies, 0.90), 3),
        "p95": round(_percentile(latencies, 0.95), 3),
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
    }


def _worker(payload: dict[str, Any], output: Any) -> None:
    try:
        context, build_ms = _initialize_runtime()
        batch_id = str(payload["batch_id"])
        global_description = str(payload["global_description"])
        stages = [str(value) for value in payload["stages"]]
        top_k = int(payload["top_k"])
        common = {
            "context": context,
            "batch_id": batch_id,
            "global_description": global_description,
            "stages": stages,
            "top_k": top_k,
        }
        warmup = getattr(context, "warmup", None)
        if warmup is None:
            raise RuntimeError("warm-up service is unavailable")
        if payload["mode"] == "cold":
            warm_started = time.perf_counter()
            warmup.start()
            snapshot = warmup.wait(
                float(payload["warmup_timeout_seconds"]), workflow="trake"
            )
            warmup_ms = (time.perf_counter() - warm_started) * 1_000
            trake = snapshot.get("workflows", {}).get("trake", {})
            if not bool(trake.get("usable", False)):
                raise RuntimeError("TRAKE warm-up did not reach a usable state")
            latency, hits = _run_query(**common, query_id="trake-benchmark-cold")
            output.put(
                {
                    "runtime_build_ms": round(build_ms, 3),
                    "warmup_ms": round(warmup_ms, 3),
                    "warmup": snapshot,
                    "latency_ms": round(latency, 3),
                    "hit_count": hits,
                    "resources": safe_resource_snapshot(),
                }
            )
            return

        warm_started = time.perf_counter()
        warmup.start()
        snapshot = warmup.wait(
            float(payload["warmup_timeout_seconds"]), workflow="trake"
        )
        warmup_ms = (time.perf_counter() - warm_started) * 1_000
        trake = snapshot.get("workflows", {}).get("trake", {})
        if not bool(trake.get("usable", False)):
            raise RuntimeError("TRAKE warm-up did not reach a usable state")
        first_latency, first_hits = _run_query(
            **common, query_id="trake-benchmark-first-post-warm"
        )
        latencies: list[float] = []
        samples: list[dict[str, Any]] = []
        for index in range(int(payload["iterations"])):
            try:
                latency, hits = _run_query(
                    **common,
                    query_id=f"trake-benchmark-warm-{index}",
                )
                latencies.append(latency)
                samples.append(
                    {
                        "iteration": index + 1,
                        "latency_ms": round(latency, 3),
                        "hits": hits,
                    }
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
            raise RuntimeError("every warm TRAKE query failed")
        output.put(
            {
                "runtime_build_ms": round(build_ms, 3),
                "warmup_ms": round(warmup_ms, 3),
                "warmup": snapshot,
                "first_post_warm_ms": round(first_latency, 3),
                "first_post_warm_hits": first_hits,
                "successful_iterations": len(latencies),
                "error_count": sum("error" in sample for sample in samples),
                "total_hit_count": sum(int(sample.get("hits", 0)) for sample in samples),
                "warm_latency_ms": _summary(latencies),
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
        return dict(output.get(timeout=2))
    except Empty:
        return {"error": "WorkerNoResult", "exit_code": process.exitcode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--global-description", required=True)
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--warmup-timeout-seconds",
        type=float,
        help="bounded warm-up wait; defaults to the configured startup budget",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= len(args.stage) <= 20:
        parser.error("repeat --stage between 2 and 20 times")
    if not 1 <= args.top_k <= 100:
        parser.error("--top-k must be between 1 and 100")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.warmup_timeout_seconds is not None and args.warmup_timeout_seconds <= 0:
        parser.error("--warmup-timeout-seconds must be positive")
    config = load_config()
    warmup_timeout = (
        float(args.warmup_timeout_seconds)
        if args.warmup_timeout_seconds is not None
        else float(config.get("warmup", {}).get("startup_budget_seconds", 90))
    )
    common = {
        "batch_id": args.batch_id,
        "global_description": args.global_description,
        "stages": list(args.stage),
        "top_k": args.top_k,
        "iterations": args.iterations,
        "warmup_timeout_seconds": warmup_timeout,
    }
    cold = (
        {"skipped": True}
        if args.skip_cold
        else _spawn_worker(
            {**common, "mode": "cold"},
            timeout_seconds=warmup_timeout + 180,
        )
    )
    warm = _spawn_worker(
        {**common, "mode": "warm"},
        timeout_seconds=warmup_timeout + max(300, args.iterations * 15),
    )
    result = {
        "profile": str(config.get("profile", "laptop")),
        "batch_id": args.batch_id,
        "stage_count": len(args.stage),
        "top_k": args.top_k,
        "iterations": args.iterations,
        "pipeline_fingerprint": _pipeline_fingerprint(config, args.batch_id),
        "cold": cold,
        "warm": warm,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int("error" in cold or "error" in warm)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
