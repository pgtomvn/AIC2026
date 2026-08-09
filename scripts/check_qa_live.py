"""Explicit one- or five-candidate single-frame QA check against indexed media."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACandidateRequest, QACompetitionSpec
from src.retrieval.qa_media import CanonicalImageError
from src.runtime import build_runtime
from src.utils.io_helpers import load_config
from src.vlm.qa_reasoner import QAReasonerError


def _emit(status: str, **metadata: object) -> None:
    payload = {"check": "qa_live", "status": status}
    payload.update({key: value for key, value in metadata.items() if value is not None})
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--event-description", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer-spec", choices=("numeric", "general"), default="general")
    parser.add_argument(
        "--reasoning-policy",
        choices=("local_only", "gemini_then_local"),
        default="local_only",
    )
    parser.add_argument("--allow-external-frame-transfer", action="store_true")
    parser.add_argument("--window-count", type=int, choices=(1, 5), default=1)
    args = parser.parse_args(argv)
    if (
        args.reasoning_policy == "gemini_then_local"
        and not args.allow_external_frame_transfer
    ):
        _emit("blocked", category="external_transfer_consent_required")
        return 2

    try:
        context = build_runtime(load_config())
        if context.qa_pipeline is None or context.artifact_repository is None:
            _emit("failed", category="runtime_unavailable")
            return 1
        context.ledger.initialize()
        context.artifact_repository.initialize()
        batch_filters = context.validate_batch_filters((args.batch_id,))
        spec = (
            QACompetitionSpec.historical_numeric()
            if args.answer_spec == "numeric"
            else QACompetitionSpec.conservative_general_text()
        )
        request = QACandidateRequest(
            query_id=f"qa-live-check-{uuid4().hex}",
            event_description=args.event_description,
            batch_filters=batch_filters,
            top_k=args.window_count,
        )
        candidates = context.qa_pipeline.retrieve_vqa_candidates(
            request.event_description,
            request.top_k,
            batch_filters=batch_filters,
        )
        if len(candidates) < args.window_count:
            _emit("failed", category="insufficient_candidates", count=len(candidates))
            return 1
        outcomes = []
        for index, candidate in enumerate(candidates[: args.window_count], 1):
            frame = context.artifact_repository.get_frame(
                candidate.batch_id, candidate.video_id, candidate.source_frame_idx
            )
            if frame is None:
                raise LookupError("candidate_not_canonical")
            reasoning, timings = context.qa_pipeline.answer_vqa_single_frame(
                frame,
                args.question,
                event_description=args.event_description,
                answer_spec_id=spec.spec_id,
                reasoning_policy=args.reasoning_policy,
                allow_external_frame_transfer=args.allow_external_frame_transfer,
                query_id=f"{request.query_id}-{index}",
                reasoner=context.qa_reasoner,
            )
            outcomes.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": "answered" if reasoning.get("answerable") else "unanswerable",
                    "provider": reasoning.get("provider"),
                    "video_id": candidate.video_id,
                    "source_frame_idx": candidate.source_frame_idx,
                    "answer": reasoning.get("answer"),
                    "vlm_inference_ms": timings.get("vlm_inference_ms"),
                }
            )
        if args.window_count == 1:
            outcome = outcomes[0]
            _emit(
                "passed",
                window_count=1,
                outcome=outcome["status"],
                provider=outcome.get("provider"),
                video_id=outcome.get("video_id"),
                source_frame_idx=outcome.get("source_frame_idx"),
                frame_count=1,
                answer=outcome.get("answer"),
                vlm_inference_ms=outcome.get("vlm_inference_ms"),
            )
            return 0
        failed = len(outcomes) != 5
        _emit("failed" if failed else "passed", window_count=5, outcomes=outcomes)
        return 1 if failed else 0
    except QAReasonerError as error:
        _emit("failed", **error.safe_metadata())
        return 1
    except CanonicalImageError:
        _emit("failed", category="media_unavailable")
        return 1
    except LookupError:
        _emit("failed", category="no_candidates")
        return 1
    except (TypeError, ValueError):
        _emit("failed", category="invalid_request")
        return 1
    except Exception:
        _emit("failed", category="internal_check_error")
        return 1
    finally:
        reasoner = getattr(locals().get("context"), "qa_reasoner", None)
        unload = getattr(reasoner, "unload", None)
        if callable(unload):
            unload()


if __name__ == "__main__":
    raise SystemExit(main())
