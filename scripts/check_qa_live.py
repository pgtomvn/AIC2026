"""Explicit one- or five-window Visual Q&A checks against real indexed media.

The default performs one retrieval and at most one Gemini provider call without
creating a session.  Five-window mode is a separate, explicit production-path
check that requires an additional multi-call acknowledgement.  Neither mode is
part of bootstrap or the default offline checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.retrieval.qa_media import CanonicalImageError
from src.retrieval.visual_qa import VisualQATotalFailureError
from src.runtime import build_runtime
from src.utils.io_helpers import load_config
from src.vlm.qa_reasoner import GeminiInteractionsReasoner, QAReasonerError


def _emit(status: str, **metadata: object) -> None:
    payload = {"check": "qa_live", "status": status}
    payload.update({key: value for key, value in metadata.items() if value is not None})
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--event-description", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--answer-spec",
        choices=("numeric", "general"),
        default="numeric",
    )
    parser.add_argument(
        "--live-gemini",
        action="store_true",
        help="required acknowledgement that one real Gemini request will be sent",
    )
    parser.add_argument(
        "--allow-external-frame-transfer",
        action="store_true",
        help="required acknowledgement that selected competition frames leave the machine",
    )
    parser.add_argument("--window-count", type=int, choices=(1, 5), default=1)
    parser.add_argument(
        "--allow-multiple-provider-calls",
        action="store_true",
        help="required acknowledgement for the five-window production-path check",
    )
    args = parser.parse_args(argv)
    if not args.live_gemini or not args.allow_external_frame_transfer:
        _emit("blocked", category="confirmation_required")
        return 2
    if args.window_count == 5 and not args.allow_multiple_provider_calls:
        _emit("blocked", category="multiple_call_confirmation_required")
        return 2

    reasoner: GeminiInteractionsReasoner | None = None
    try:
        config = load_config()
        provider = config.get("services", {}).get("gemini", {})
        api_key = provider.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            _emit("failed", category="not_configured")
            return 1

        context = build_runtime(config)
        if context.qa_pipeline is None or context.artifact_repository is None:
            _emit("failed", category="runtime_unavailable")
            return 1
        # These are the public, explicit read-model initialization boundaries.
        # Deliberately do not initialize ``qa_session_store``.
        context.ledger.initialize()
        try:
            batch_filters = context.validate_batch_filters((args.batch_id,))
        except (TypeError, ValueError):
            _emit("failed", category="invalid_batch")
            return 1
        context.artifact_repository.initialize()

        spec = (
            QACompetitionSpec.historical_numeric()
            if args.answer_spec == "numeric"
            else QACompetitionSpec.conservative_general_text()
        )
        request = QAQueryRequest(
            query_id=f"qa-live-check-{uuid4().hex}",
            event_description=args.event_description,
            question=args.question,
            mode="interactive",
            batch_filters=batch_filters,
            answer_spec_id=spec.spec_id,
            limit=5,
        )
        if args.window_count == 1:
            reasoner = GeminiInteractionsReasoner(
                api_key=api_key,
                model_name=str(provider.get("model", "gemini-3.6-flash")),
                timeout_seconds=float(provider.get("timeout_seconds", 30)),
                max_attempts=1,
                requests_per_minute=min(
                    13, max(1, int(provider.get("requests_per_minute", 13)))
                ),
                max_concurrency=1,
                min_request_interval_seconds=max(
                    4.5,
                    float(provider.get("min_request_interval_seconds", 4.5)),
                ),
            )
            reasoner.initialize()
            outcome = context.qa_pipeline.check_one_window_live(
                request,
                reasoner=reasoner,
                batch_filters=batch_filters,
            )
            _emit(
                "passed",
                window_count=1,
                outcome=outcome.get("status"),
                provider=outcome.get("provider"),
                model=outcome.get("model_name"),
                attempts=outcome.get("attempts"),
                batch_id=outcome.get("batch_id"),
                video_id=outcome.get("video_id"),
                source_frame_idx=outcome.get("source_frame_idx"),
                frame_count=len(outcome.get("frames", ())),
                answer=outcome.get("normalized_answer"),
            )
            return 0

        if context.qa_session_store is None:
            _emit("failed", category="session_store_unavailable")
            return 1
        context.qa_session_store.initialize()
        search_id = uuid4().hex
        total_failure = False
        try:
            context.qa_pipeline.search(request, session_id=search_id)
        except VisualQATotalFailureError:
            total_failure = True
        snapshot = context.qa_pipeline.get_search(search_id) or {}
        raw_outcomes = snapshot.get("window_outcomes", ())
        outcomes = [
            {
                key: item.get(key)
                for key in (
                    "window_id",
                    "status",
                    "provider",
                    "attempts",
                    "video_id",
                    "source_frame_idx",
                    "normalized_answer",
                    "failure_code",
                )
                if item.get(key) is not None
            }
            for item in raw_outcomes
            if isinstance(item, dict)
        ][:5]
        failed = total_failure or len(outcomes) != 5 or any(
            str(item.get("status", "")).endswith("_failed") for item in outcomes
        )
        _emit(
            "failed" if failed else "passed",
            window_count=5,
            search_id=search_id,
            outcomes=outcomes,
        )
        return 1 if failed else 0
    except QAReasonerError as exc:
        _emit("failed", **exc.safe_metadata())
        return 1
    except CanonicalImageError:
        _emit("failed", category="media_unavailable")
        return 1
    except LookupError:
        _emit("failed", category="no_candidates")
        return 1
    except Exception:
        _emit("failed", category="internal_check_error")
        return 1
    finally:
        if reasoner is not None:
            reasoner.unload()
        context_value = locals().get("context")
        pipeline = getattr(context_value, "qa_pipeline", None)
        pipeline_reasoner = getattr(pipeline, "reasoner", None)
        if args.window_count == 5 and callable(getattr(pipeline_reasoner, "unload", None)):
            pipeline_reasoner.unload()


if __name__ == "__main__":
    raise SystemExit(main())
