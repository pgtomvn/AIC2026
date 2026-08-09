"""Explicit live comparison of Gemini Flash and local Qwen on one QA frame."""

from __future__ import annotations

import argparse
import json
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

DEFAULT_EVENT = "A man is broadcasting TV news."
DEFAULT_QUESTION = "Does the man wear glasses?"


def _api_running(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5).status_code == 200
    except requests.RequestException:
        return False


def _safe_failure(error: BaseException) -> str:
    if isinstance(error, QAReasonerError):
        return str(error.category or "qa_provider_failed")
    return type(error).__name__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--event-description", default=DEFAULT_EVENT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--api-port", type=int, default=8080)
    parser.add_argument("--allow-external-frame-transfer", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_external_frame_transfer:
        parser.error("--allow-external-frame-transfer is required for the live Gemini check")
    if _api_running(args.api_port):
        print(json.dumps({"status": "blocked", "code": "api_runtime_already_running"}))
        return 2

    config = load_config()
    gemini_config = config.get("services", {}).get("gemini_vlm", {})
    if not isinstance(gemini_config.get("api_key"), str) or not gemini_config["api_key"].strip():
        print(json.dumps({"status": "blocked", "code": "gemini_key_not_configured"}))
        return 2

    context = None
    report: dict[str, Any] = {
        "status": "failed",
        "case": {
            "batch_id": args.batch_id,
            "event_description": args.event_description,
            "question": args.question,
        },
    }
    try:
        context = build_runtime(config)
        if context.qa_pipeline is None or context.artifact_repository is None:
            raise RuntimeError("qa_runtime_unavailable")
        context.ledger.initialize()
        context.artifact_repository.initialize()
        if context.qa_session_store is not None:
            context.qa_session_store.initialize()

        batch = context.validate_batch_filters((args.batch_id,))
        retrieval_started = time.perf_counter()
        candidates = context.qa_pipeline.start_candidate_search(
            QACandidateRequest(
                query_id=f"gemini-live-{uuid4().hex}",
                event_description=args.event_description,
                batch_filters=batch,
                top_k=5,
            ),
            session_id=f"gemini-live-{uuid4().hex}",
        )
        kis_ms = (time.perf_counter() - retrieval_started) * 1_000
        if not candidates:
            raise LookupError("no_candidates")
        candidate = candidates[0]
        frame = context.artifact_repository.get_frame(
            candidate.batch_id, candidate.video_id, candidate.source_frame_idx
        )
        if frame is None:
            raise LookupError("candidate_not_canonical")

        spec_id = QACompetitionSpec.conservative_general_text().spec_id
        gemini_reasoning: dict[str, Any] | None = None
        gemini_timings: dict[str, float] = {}
        gemini_failure: str | None = None
        gemini_provider = getattr(context.qa_reasoner, "gemini", None)
        if gemini_provider is None:
            raise RuntimeError("gemini_reasoner_unavailable")
        try:
            gemini_reasoning, gemini_timings = context.qa_pipeline.answer_vqa_single_frame(
                frame,
                args.question,
                event_description=args.event_description,
                answer_spec_id=spec_id,
                reasoning_policy="gemini_then_local",
                allow_external_frame_transfer=True,
                query_id="gemini-live-single-frame",
                reasoner=gemini_provider,
            )
        except QAReasonerError as error:
            gemini_failure = str(error.category or "qa_provider_failed")
        local = getattr(context.qa_reasoner, "local", None)
        if local is None:
            raise RuntimeError("local_reasoner_unavailable")
        qwen_runs: list[dict[str, Any]] = []
        for phase in ("cold", "warm"):
            try:
                reasoning, timings = context.qa_pipeline.answer_vqa_single_frame(
                    frame,
                    args.question,
                    event_description=args.event_description,
                    answer_spec_id=spec_id,
                    reasoning_policy="local_only",
                    query_id=f"qwen-{phase}-single-frame",
                    reasoner=local,
                )
                qwen_runs.append(
                    {
                        "phase": phase,
                        "status": "passed",
                        "answer": reasoning.get("answer"),
                        "provider_ms": timings.get("vlm_inference_ms"),
                        "total_ms": timings.get("answer_call_ms"),
                        "image_prepare_ms": timings.get("frame_prepare_ms"),
                    }
                )
            except QAReasonerError as error:
                qwen_runs.append(
                    {"phase": phase, "status": "failed", "code": _safe_failure(error)}
                )

        actual = str((gemini_reasoning or {}).get("provider") or "unknown")
        gemini_is_actual = actual == "gemini" and gemini_failure is None
        qwen_available = any(value.get("status") == "passed" for value in qwen_runs)
        warnings: list[str] = []
        if kis_ms >= 100:
            warnings.append("stage1_above_100ms")
        if gemini_is_actual and float(gemini_timings["vlm_inference_ms"]) >= 1_000:
            warnings.append("gemini_above_1000ms")
        if gemini_is_actual and kis_ms + float(gemini_timings["answer_call_ms"]) >= 1_200:
            warnings.append("end_to_end_above_1200ms")
        report.update(
            {
                "status": (
                    "passed"
                    if gemini_is_actual and qwen_available
                    else "partial"
                    if gemini_is_actual
                    else "fallback"
                    if qwen_available
                    else "failed"
                ),
                "candidate": {
                    "batch_id": candidate.batch_id,
                    "video_id": candidate.video_id,
                    "source_frame_idx": candidate.source_frame_idx,
                    "timestamp_ms": candidate.timestamp_ms,
                },
                "kis_retrieval_ms": round(kis_ms, 3),
                "gemini": {
                    "status": "passed" if gemini_is_actual else "failed",
                    "answer": (gemini_reasoning or {}).get("answer"),
                    "actual_provider": actual,
                    "fallback_used": not gemini_is_actual and qwen_available,
                    "failure_code": gemini_failure,
                    "image_prepare_ms": gemini_timings.get("frame_prepare_ms"),
                    "provider_ms": gemini_timings.get("vlm_inference_ms"),
                    "total_ms": gemini_timings.get("answer_call_ms"),
                },
                "qwen": qwen_runs,
                "resources": safe_resource_snapshot(),
                "ollama": getattr(local, "runtime_status", lambda: {})(),
                "warnings": warnings,
            }
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if gemini_is_actual and qwen_available else 1
    except Exception as error:
        report["code"] = _safe_failure(error)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 1
    finally:
        if context is not None:
            resolver = getattr(context, "qa_image_resolver", None)
            close = getattr(resolver, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
