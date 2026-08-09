"""Benchmark decoupled KIS retrieval and single-frame Visual Q&A locally."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACandidateRequest, QACompetitionSpec
from src.runtime import build_runtime
from src.utils.io_helpers import load_config
from src.vlm.qa_reasoner import QAReasonerError
from src.warmup import safe_resource_snapshot

DEFAULT_BATCH = "aic25-b1-reference"
DEFAULT_EVENT = "A man is broadcasting TV news."
DEFAULT_QUESTION = "Does the man wear glasses?"


def _api_is_running(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5).status_code == 200
    except requests.RequestException:
        return False


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(_percentile(values, 0.50), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def _resource_maximum(samples: list[dict[str, int | None]]) -> dict[str, int | None]:
    names = set().union(*(sample.keys() for sample in samples)) if samples else set()
    return {
        name: max(
            (int(sample[name]) for sample in samples if isinstance(sample.get(name), int)),
            default=None,
        )
        for name in sorted(names)
    }


def _host_memory() -> dict[str, int | None]:
    try:
        import psutil
    except ImportError:
        return {"system_ram_used_bytes": None, "ollama_process_rss_bytes": None}
    try:
        ollama_rss = 0
        for process in psutil.process_iter(("name", "memory_info")):
            name = str(process.info.get("name") or "").casefold()
            if "ollama" in name or "llama" in name:
                memory = process.info.get("memory_info")
                ollama_rss += int(getattr(memory, "rss", 0) or 0)
        return {
            "system_ram_used_bytes": int(psutil.virtual_memory().used),
            "ollama_process_rss_bytes": ollama_rss or None,
        }
    except (OSError, psutil.Error):
        return {"system_ram_used_bytes": None, "ollama_process_rss_bytes": None}


def _failure_bucket(error: Exception) -> str:
    if isinstance(error, QAReasonerError):
        category = str(error.safe_metadata().get("category") or "invalid_response")
        if "timeout" in category:
            return "timeout"
        if category == "resource_exhausted":
            return "oom"
        return "invalid_response"
    return "invalid_response"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH)
    parser.add_argument("--event-description", default=DEFAULT_EVENT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--api-port", type=int, default=8080)
    parser.add_argument("--include-florence", action="store_true")
    parser.add_argument("--florence-target-label", default="person")
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 100:
        parser.error("--iterations must be between 1 and 100")
    if _api_is_running(args.api_port):
        print(json.dumps({"status": "blocked", "code": "api_runtime_already_running"}))
        return 2

    context = None
    result: dict[str, Any] = {
        "status": "failed",
        "case": {
            "batch_id": args.batch_id,
            "event_description": args.event_description,
            "question": args.question,
        },
    }
    try:
        context = build_runtime(load_config())
        if context.qa_pipeline is None or context.artifact_repository is None:
            raise RuntimeError("qa_runtime_unavailable")
        context.ledger.initialize()
        context.artifact_repository.initialize()
        if context.qa_session_store is not None:
            context.qa_session_store.initialize()
        batch = context.validate_batch_filters((args.batch_id,))

        retrieval_samples: list[dict[str, Any]] = []
        candidates = ()
        for sequence in ("initial", "repeat"):
            request = QACandidateRequest(
                query_id=f"benchmark-{sequence}-{uuid4().hex}",
                event_description=args.event_description,
                batch_filters=batch,
                top_k=5,
            )
            session_id = f"benchmark-{sequence}-{uuid4().hex}"
            started = time.perf_counter()
            candidates = context.qa_pipeline.start_candidate_search(
                request, session_id=session_id
            )
            elapsed_ms = (time.perf_counter() - started) * 1_000
            snapshot = context.qa_pipeline.get_search(session_id) or {}
            retrieval_samples.append(
                {
                    "pass": sequence,
                    "cache_status": snapshot.get("state", {}).get(
                        "candidate_cache_status"
                    ),
                    "latency_ms": round(elapsed_ms, 3),
                    "candidate_count": len(candidates),
                }
            )
        if not candidates:
            raise LookupError("no_candidates")
        candidate = candidates[0]
        frame = context.artifact_repository.get_frame(
            candidate.batch_id, candidate.video_id, candidate.source_frame_idx
        )
        if frame is None:
            raise LookupError("candidate_not_canonical")

        router = context.qa_reasoner
        reasoner = getattr(router, "local", router)
        latency_values: list[float] = []
        frame_prepare_values: list[float] = []
        handoff_values: list[float] = []
        answers: list[str] = []
        answerable_count = 0
        provider_metrics: list[dict[str, Any]] = []
        resources: list[dict[str, int | None]] = []
        failures = {"oom": 0, "timeout": 0, "invalid_response": 0}
        successes = 0
        for index in range(args.iterations):
            try:
                reasoning, timings = context.qa_pipeline.answer_vqa_single_frame(
                    frame,
                    args.question,
                    event_description=args.event_description,
                    answer_spec_id=QACompetitionSpec.conservative_general_text().spec_id,
                    reasoning_policy="local_only",
                    query_id=f"benchmark-answer-{index}",
                    reasoner=reasoner,
                )
                successes += 1
                answerable_count += int(bool(reasoning.get("answerable")))
                answers.append(str(reasoning.get("answer") or ""))
                latency_values.append(float(timings["vlm_inference_ms"]))
                frame_prepare_values.append(float(timings["frame_prepare_ms"]))
                handoff_values.append(float(timings.get("gpu_handoff_ms", 0.0)))
                metrics = getattr(reasoner, "last_metrics", {})
                provider_metrics.append(
                    {
                        name: value
                        for name, value in dict(metrics).items()
                        if name
                        in {
                            "wall_latency_ms",
                            "load_duration_ms",
                            "prompt_eval_duration_ms",
                            "eval_duration_ms",
                            "prompt_eval_count",
                            "eval_count",
                            "handoff_ms",
                        }
                    }
                )
            except Exception as error:
                failures[_failure_bucket(error)] += 1
            resources.append(safe_resource_snapshot())

        warm_values = latency_values[1:] if len(latency_values) > 1 else latency_values
        cold_ms = latency_values[0] if latency_values else None
        warm_summary = _summary(warm_values) if warm_values else {}
        handoff_summary = _summary(handoff_values) if handoff_values else {}
        retrieval_hit_ms = float(retrieval_samples[-1]["latency_ms"])
        warm_p95 = float(warm_summary.get("p95_ms", 0.0))
        warnings: list[str] = []
        if retrieval_hit_ms >= 800:
            warnings.append("retrieval_cache_hit_above_800ms")
        if warm_p95 >= 1_200:
            warnings.append("single_frame_warm_p95_above_1200ms")
        if retrieval_hit_ms + warm_p95 >= 2_000:
            warnings.append("warm_end_to_end_above_2000ms")
        ollama_status = getattr(reasoner, "runtime_status", lambda: {})()
        result.update(
            {
                "status": "passed" if successes == args.iterations else "failed",
                "candidate": candidate.to_dict(),
                "retrieval": retrieval_samples,
                "qwen": {
                    "iterations": args.iterations,
                    "successes": successes,
                    "answerable_count": answerable_count,
                    "failures": failures,
                    "answers": answers,
                    "cold_inference_ms": None if cold_ms is None else round(cold_ms, 3),
                    "warm_inference": warm_summary,
                    "frame_prepare": _summary(frame_prepare_values)
                    if frame_prepare_values
                    else {},
                    "gpu_handoff": handoff_summary,
                    "model_swap": handoff_summary,
                    "provider_metrics": provider_metrics,
                    "runtime": ollama_status,
                },
                "warm_total_target_ms": round(retrieval_hit_ms + warm_p95, 3),
                "resources_peak": _resource_maximum(resources),
                "host_memory": _host_memory(),
                "warnings": warnings,
            }
        )

        if args.include_florence:
            florence = context.florence_reasoner
            if florence is None:
                result["florence"] = {
                    "status": "not_provisioned",
                    "task": "object_count",
                }
            else:
                started = time.perf_counter()
                reasoning, timings = context.qa_pipeline.answer_vqa_single_frame(
                    frame,
                    f"How many {args.florence_target_label} objects are visible?",
                    event_description=args.event_description,
                    answer_spec_id=QACompetitionSpec.historical_numeric().spec_id,
                    query_id="benchmark-florence-object-count",
                    reasoner=florence,
                    florence_task="object_count",
                    florence_target_label=args.florence_target_label,
                )
                result["florence"] = {
                    "status": "passed" if reasoning.get("answerable") else "unanswerable",
                    "task": "object_count",
                    "answer": reasoning.get("answer"),
                    "wall_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "stage_timings": timings,
                }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if result["status"] == "passed" else 1
    except Exception as error:
        result["code"] = (
            error.safe_metadata().get("category")
            if isinstance(error, QAReasonerError)
            else type(error).__name__
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 1
    finally:
        if context is not None:
            close = getattr(context.qa_image_resolver, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
