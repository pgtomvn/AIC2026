"""Explicit live Gemini check using only tiny synthetic images.

This command is never invoked by bootstrap or the default test suite. Configure
the rotated replacement credential as ``GEMINI_API_KEY`` in untracked ``.env``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.utils.io_helpers import load_config
from src.vlm.qa_reasoner import (
    GeminiInteractionsReasoner,
    GeminiRequestGovernor,
    QAImage,
    QAReasonerError,
    retry_delay_for_exception,
    sanitize_provider_exception,
)

_TRANSIENT_CATEGORIES = frozenset(
    {"network_unreachable", "rate_limited", "server_unavailable"}
)


def _image(index: int, color: str) -> QAImage:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(output, format="PNG")
    data = output.getvalue()
    return QAImage(
        data=data,
        mime_type="image/png",
        index=index,
        source_frame_idx=index - 1,
        timestamp_ms=(index - 1) * 100,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _emit(stage: str, status: str, **metadata: object) -> None:
    try:
        sdk_version = version("google-genai")
    except PackageNotFoundError:
        sdk_version = "unavailable"
    allowed = {"provider", "model", "attempts", "category", "status_code"}
    payload = {
        "check": "gemini_synthetic",
        "stage": stage,
        "status": status,
        "sdk_version": sdk_version,
    }
    payload.update(
        {key: value for key, value in metadata.items() if key in allowed and value is not None}
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _diagnostic_max_attempts(provider: Mapping[str, Any]) -> int:
    """Use the production retry budget while retaining a hard five-attempt cap."""

    try:
        retries = int(provider.get("max_retries", 2))
    except (TypeError, ValueError):
        retries = 2
    return min(5, max(1, retries + 1))


def _new_direct_client(api_key: str, timeout_seconds: float) -> Any:
    from google import genai

    return genai.Client(
        api_key=api_key,
        http_options={
            "timeout": int(timeout_seconds * 1_000),
            # Retry classification remains under this checker's control so
            # the emitted attempt count is exact and permanent errors stop.
            "retry_options": {"attempts": 1},
        },
    )


def _close_client(client: object | None) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _run_direct_stage(
    operation: Callable[[Any], Any],
    *,
    api_key: str,
    model_name: str,
    timeout_seconds: float,
    max_attempts: int,
    request_governor: GeminiRequestGovernor | None = None,
    retry_base_seconds: float = 5.0,
    retry_max_seconds: float = 60.0,
) -> tuple[Any | None, int, QAReasonerError | None]:
    """Run one text stage with fresh clients and transient-only retries."""

    last_error: QAReasonerError | None = None
    for attempt in range(1, max_attempts + 1):
        client: object | None = None
        try:
            if request_governor is not None:
                request_governor.acquire()
            client = _new_direct_client(api_key, timeout_seconds)
            return operation(client), attempt, None
        except Exception as exc:
            last_error = sanitize_provider_exception(
                exc, provider="gemini", model=model_name, attempts=attempt
            )
            if (
                last_error.category not in _TRANSIENT_CATEGORIES
                or attempt == max_attempts
            ):
                return None, attempt, last_error
            time.sleep(
                retry_delay_for_exception(
                    exc,
                    attempt,
                    base_seconds=retry_base_seconds,
                    max_seconds=retry_max_seconds,
                )
            )
        finally:
            _close_client(client)
    return None, max_attempts, last_error


def _reasoner_settings(
    provider: Mapping[str, Any],
    *,
    max_attempts: int,
    request_governor: GeminiRequestGovernor,
) -> dict[str, Any]:
    return {
        "api_key": provider["api_key"],
        "model_name": str(provider.get("model", "gemini-3.6-flash")),
        "timeout_seconds": float(provider.get("timeout_seconds", 30)),
        "max_attempts": max_attempts,
        "requests_per_minute": min(
            13, max(1, int(provider.get("requests_per_minute", 13)))
        ),
        "max_concurrency": 1,
        "retry_base_seconds": float(provider.get("retry_base_seconds", 5)),
        "retry_max_seconds": float(provider.get("retry_max_seconds", 60)),
        "rate_limit_cooldown_seconds": float(
            provider.get("rate_limit_cooldown_seconds", 60)
        ),
        "min_request_interval_seconds": float(
            provider.get("min_request_interval_seconds", 4.5)
        ),
        "request_governor": request_governor,
    }


def _one_image_request(spec: QACompetitionSpec) -> QAQueryRequest:
    return QAQueryRequest(
        query_id="live-gemini-one-image-check",
        event_description="One simple test frame contains a red square.",
        question="How many provided frames are red?",
        mode="automatic",
        answer_spec_id=spec.spec_id,
        limit=1,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-multiple-provider-calls",
        action="store_true",
        help="run the extended four-stage diagnostic using the production governor",
    )
    args = parser.parse_args(argv)
    stage = "configuration"
    reasoner: GeminiInteractionsReasoner | None = None
    try:
        config = load_config()
        provider = config.get("services", {}).get("gemini", {})
        api_key = provider.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            _emit(stage, "failed", category="not_configured")
            return 1
        model_name = str(provider.get("model", "gemini-3.6-flash"))
        _emit(stage, "passed", provider="gemini", model=model_name)

        stage = "client_initialization"
        timeout_seconds = float(provider.get("timeout_seconds", 30))
        requests_per_minute = min(
            13, max(1, int(provider.get("requests_per_minute", 13)))
        )
        min_request_interval_seconds = float(
            provider.get("min_request_interval_seconds", 4.5)
        )
        governor = GeminiRequestGovernor(
            requests_per_minute,
            clock=time.monotonic,
            sleep=time.sleep,
            min_interval_seconds=min_request_interval_seconds,
        )
        max_attempts = (
            _diagnostic_max_attempts(provider)
            if args.allow_multiple_provider_calls
            else 1
        )
        reasoner = GeminiInteractionsReasoner(
            **_reasoner_settings(
                {**provider, "api_key": api_key},
                max_attempts=max_attempts,
                request_governor=governor,
            )
        )
        reasoner.initialize()
        _emit(stage, "passed", provider="gemini", model=model_name)

        images = tuple(_image(index, "red" if index <= 3 else "blue") for index in range(1, 6))
        spec = QACompetitionSpec.historical_numeric()

        if args.allow_multiple_provider_calls:
            retry_base_seconds = float(provider.get("retry_base_seconds", 5))
            retry_max_seconds = float(provider.get("retry_max_seconds", 60))
            stage = "minimal_text"
            interaction, attempts, clean = _run_direct_stage(
                lambda client: client.interactions.create(
                    model=model_name,
                    input="Reply with the single word OK.",
                    store=False,
                ),
                api_key=api_key,
                model_name=model_name,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                request_governor=governor,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            if clean is not None:
                _emit(stage, "failed", **clean.safe_metadata())
                return 1
            if not isinstance(getattr(interaction, "output_text", None), str):
                _emit(stage, "failed", category="response_invalid")
                return 1
            _emit(stage, "passed", provider="gemini", model=model_name, attempts=attempts)

            stage = "structured_text"
            interaction, attempts, clean = _run_direct_stage(
                lambda client: client.interactions.create(
                    model=model_name,
                    input="Return the answer OK using the required JSON schema.",
                    response_format=[
                        {
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                            },
                        }
                    ],
                    store=False,
                ),
                api_key=api_key,
                model_name=model_name,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                request_governor=governor,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            if clean is not None:
                _emit(stage, "failed", **clean.safe_metadata())
                return 1
            try:
                structured = json.loads(str(getattr(interaction, "output_text", "")))
            except (TypeError, ValueError):
                structured = None
            if not isinstance(structured, dict) or not isinstance(
                structured.get("answer"), str
            ):
                _emit(stage, "failed", category="response_invalid")
                return 1
            _emit(stage, "passed", provider="gemini", model=model_name, attempts=attempts)

        stage = "one_image"
        one_result = reasoner.reason(_one_image_request(spec), images[:1], spec)
        if not one_result.answerable or one_result.normalized_answer != "1":
            _emit(stage, "failed", category="response_invalid")
            return 1
        _emit(
            stage,
            "passed",
            provider=one_result.provider,
            model=one_result.model_name,
            attempts=one_result.attempts,
            latency_ms=round(one_result.latency_ms, 1),
        )

        if not args.allow_multiple_provider_calls:
            return 0

        stage = "five_images"
        request = QAQueryRequest(
            query_id="live-gemini-five-image-check",
            event_description="Five simple test frames contain red and blue squares.",
            question="How many provided frames are red?",
            mode="automatic",
            answer_spec_id=spec.spec_id,
            limit=5,
        )
        result = reasoner.reason(
            request,
            images,
            spec,
        )
        if not result.answerable or result.normalized_answer != "3":
            _emit(
                stage,
                "failed",
                provider=result.provider,
                model=result.model_name,
                attempts=result.attempts,
                category="response_invalid",
            )
            return 1
        _emit(
            stage,
            "passed",
            provider=result.provider,
            model=result.model_name,
            attempts=result.attempts,
            latency_ms=round(result.latency_ms, 1),
        )
        return 0
    except QAReasonerError as exc:
        _emit(stage, "failed", **exc.safe_metadata())
        return 1
    except Exception:
        # Never render an unexpected exception, traceback, path, provider body,
        # request content, or chained SDK error from this credentialed command.
        _emit(stage, "failed", category="provider_request_failed")
        return 1
    finally:
        if reasoner is not None:
            reasoner.unload()


def check_five_image_once() -> int:
    """Run only the five-image structured gate for one-call readiness checks."""

    stage = "configuration"
    reasoner: GeminiInteractionsReasoner | None = None
    try:
        config = load_config()
        provider = config.get("services", {}).get("gemini", {})
        api_key = provider.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            _emit(stage, "failed", category="not_configured")
            return 1
        model_name = str(provider.get("model", "gemini-3.6-flash"))
        reasoner = GeminiInteractionsReasoner(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=float(provider.get("timeout_seconds", 30)),
            max_attempts=1,
            requests_per_minute=min(
                13, max(1, int(provider.get("requests_per_minute", 13)))
            ),
            max_concurrency=1,
            min_request_interval_seconds=float(
                provider.get("min_request_interval_seconds", 4.5)
            ),
        )
        reasoner.initialize()
        stage = "five_images"
        images = tuple(
            _image(index, "red" if index <= 3 else "blue") for index in range(1, 6)
        )
        spec = QACompetitionSpec.historical_numeric()
        request = QAQueryRequest(
            query_id="readiness-five-image-check",
            event_description="Five simple test frames contain red and blue squares.",
            question="How many provided frames are red?",
            mode="automatic",
            answer_spec_id=spec.spec_id,
            limit=5,
        )
        result = reasoner.reason(request, images, spec)
        if not result.answerable or result.normalized_answer != "3":
            _emit(
                stage,
                "failed",
                provider=result.provider,
                model=result.model_name,
                attempts=result.attempts,
                category="response_invalid",
            )
            return 1
        _emit(
            stage,
            "passed",
            provider=result.provider,
            model=result.model_name,
            attempts=result.attempts,
        )
        return 0
    except QAReasonerError as exc:
        _emit(stage, "failed", **exc.safe_metadata())
        return 1
    except Exception:
        _emit(stage, "failed", category="provider_request_failed")
        return 1
    finally:
        if reasoner is not None:
            reasoner.unload()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
