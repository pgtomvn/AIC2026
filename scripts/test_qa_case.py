"""Reproduce the reported Visual Q&A case without Streamlit or session persistence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACandidateRequest, QACompetitionSpec
from src.retrieval.qa_media import CanonicalImageError
from src.runtime import build_runtime
from src.utils.io_helpers import load_config
from src.vlm.qa_reasoner import QAReasonerError

DEFAULT_EVENT = "A man is broadcasting TV news."
DEFAULT_QUESTION = "Does the man wear glasses?"
DEFAULT_BATCH = "aic25-b1-reference"


def _api_is_running(port: int) -> bool:
    try:
        response = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _safe_output(outcome: dict[str, Any]) -> dict[str, Any]:
    frames = [
        {
            "position": int(frame["position"]),
            "batch_id": str(frame["batch_id"]),
            "video_id": str(frame["video_id"]),
            "source_frame_idx": int(frame["source_frame_idx"]),
            "timestamp_ms": int(frame["timestamp_ms"]),
        }
        for frame in outcome.get("frames", ())
        if isinstance(frame, dict)
        and all(
            name in frame
            for name in (
                "position",
                "batch_id",
                "video_id",
                "source_frame_idx",
                "timestamp_ms",
            )
        )
    ]
    evidence_positions = {
        int(value)
        for value in outcome.get("evidence_frame_indices", ())
        if isinstance(value, int) and not isinstance(value, bool)
    }
    timings = dict(outcome.get("stage_timings", {}))
    metrics = dict(outcome.get("ollama_metrics", {}))
    return {
        "status": str(outcome.get("status", "unknown")),
        "answer": str(outcome.get("normalized_answer") or outcome.get("answer") or ""),
        "provider": outcome.get("provider"),
        "model": outcome.get("model_name"),
        "evidence": [frame for frame in frames if frame["position"] in evidence_positions],
        "candidate_frames": frames,
        "candidate_retrieval_ms": timings.get("candidate_retrieval_ms"),
        "frame_prepare_ms": timings.get("frame_prepare_ms"),
        "ollama_inference_ms": timings.get("ollama_inference_ms"),
        "total_ms": timings.get("total_ms"),
        "ollama_metrics": {
            name: metrics[name]
            for name in (
                "wall_latency_ms",
                "load_duration_ms",
                "prompt_eval_duration_ms",
                "eval_duration_ms",
                "prompt_eval_count",
                "eval_count",
            )
            if name in metrics
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH)
    parser.add_argument("--event-description", default=DEFAULT_EVENT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--api-port", type=int, default=8080)
    args = parser.parse_args(argv)

    if _api_is_running(args.api_port):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "code": "api_runtime_already_running",
                    "message": "Stop FastAPI before running this direct-runtime diagnostic.",
                },
                sort_keys=True,
            )
        )
        return 2

    context = None
    try:
        context = build_runtime(load_config())
        if context.qa_pipeline is None or context.artifact_repository is None:
            raise RuntimeError("Visual Q&A runtime is unavailable")
        context.ledger.initialize()
        context.artifact_repository.initialize()
        batch_filters = context.validate_batch_filters((args.batch_id,))
        spec = QACompetitionSpec.conservative_general_text()
        request = QACandidateRequest(
            query_id=f"qa-case-{uuid4().hex}",
            event_description=args.event_description,
            batch_filters=batch_filters,
            top_k=5,
        )
        retrieval_started = __import__("time").perf_counter()
        candidates = context.qa_pipeline.retrieve_vqa_candidates(
            request.event_description,
            request.top_k,
            batch_filters=batch_filters,
        )
        retrieval_ms = (__import__("time").perf_counter() - retrieval_started) * 1_000
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
        reasoning, timings = context.qa_pipeline.answer_vqa_single_frame(
            frame,
            args.question,
            event_description=args.event_description,
            answer_spec_id=spec.spec_id,
            reasoning_policy="local_only",
            query_id=request.query_id,
            reasoner=reasoner,
        )
        metrics = getattr(reasoner, "last_metrics", {})
        result = {
            "status": "answered" if reasoning.get("answerable") else "unanswerable",
            "answer": str(reasoning.get("answer") or ""),
            "provider": reasoning.get("provider"),
            "model": reasoning.get("model_name"),
            "evidence": {
                "batch_id": frame.batch_id,
                "video_id": frame.video_id,
                "source_frame_idx": frame.source_frame_idx,
                "timestamp_ms": frame.timestamp_ms,
            },
            "candidate_count": len(candidates),
            "candidate_retrieval_ms": round(retrieval_ms, 3),
            "frame_prepare_ms": timings.get("frame_prepare_ms"),
            "gpu_handoff_ms": timings.get("gpu_handoff_ms"),
            "ollama_inference_ms": timings.get("vlm_inference_ms"),
            "ollama_metrics": {
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
                }
            },
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        answer = str(result["answer"]).strip().casefold()
        return 0 if answer in {"yes", "no"} else 1
    except QAReasonerError as error:
        router = getattr(locals().get("context"), "qa_reasoner", None)
        local = getattr(router, "local", router)
        metrics = getattr(local, "last_metrics", {})
        safe_shape = {
            name: value
            for name, value in dict(metrics).items()
            if name.startswith("response_") and isinstance(value, (int, float, bool))
        }
        print(
            json.dumps(
                {"status": "failed", **error.safe_metadata(), "response_shape": safe_shape},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1
    except CanonicalImageError:
        print('{"code":"media_unavailable","status":"failed"}')
        return 1
    except LookupError:
        print('{"code":"no_candidates","status":"failed"}')
        return 1
    except Exception:
        print('{"code":"qa_case_failed","status":"failed"}')
        return 1
    finally:
        if context is not None:
            close = getattr(context.qa_image_resolver, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
