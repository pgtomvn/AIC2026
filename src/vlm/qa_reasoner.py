"""Lazy, auditable Visual Q&A reasoning providers.

The module deliberately performs no client construction, model loading, network
access, or filesystem writes at import or object-construction time.  Both
providers consume already canonicalized in-memory images and return the shared
``QAReasoningResult`` contract.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import importlib
import io
import json
import random
import sys
import threading
import time
from collections import deque
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, NoReturn, Protocol, Sequence, runtime_checkable

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.utils.offline import enforce_huggingface_offline

PROMPT_VERSION = "visual-qa-judge-v1"
NORMALIZER_VERSION = "qa-answer-normalizer-v1"
RESPONSE_SCHEMA_VERSION = "qa-structured-response-v1"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

_SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_RETRYABLE_STATUS_CODES = frozenset({408, 429})
_RESPONSE_KEYS = frozenset({"answer", "answerable", "evidence_frame_indices"})
_SAFE_ERROR_CATEGORIES = frozenset(
    {
        "auth_or_permission",
        "client_initialization",
        "client_unavailable",
        "inference_failed",
        "invalid_request",
        "model_unavailable",
        "not_configured",
        "provider_chain_exhausted",
        "provider_request_failed",
        "network_unreachable",
        "rate_limited",
        "request_rejected",
        "response_invalid",
        "server_unavailable",
    }
)


def _computed_retry_delay(
    completed_attempts: int,
    *,
    base_seconds: float,
    max_seconds: float,
    server_hint_seconds: float | None,
    jitter: Callable[[float, float], float],
) -> float:
    jitter_seconds = max(0.0, float(jitter(0.0, 1.0)))
    exponential = float(base_seconds) * (2 ** (completed_attempts - 1)) + jitter_seconds
    hinted = max(0.0, float(server_hint_seconds or 0.0))
    return min(float(max_seconds), max(exponential, hinted))


def _safe_error_label(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate or len(candidate) > 128:
        return None
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._:/")
    return candidate if all(character in allowed for character in candidate) else None


class QAReasonerError(RuntimeError):
    """Base class for sanitized provider failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        attempts: int | None = None,
        category: str | None = None,
        status_code: int | None = None,
    ) -> None:
        # ``message`` is always authored locally.  Raw SDK messages, response
        # bodies, headers, request content, and exception chains must never be
        # copied into this public boundary.
        super().__init__(message)
        self.provider = _safe_error_label(provider)
        self.model = _safe_error_label(model)
        self.model_name = self.model
        self.attempts = (
            int(attempts)
            if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 0
            else None
        )
        self.category = category if category in _SAFE_ERROR_CATEGORIES else None
        self.status_code = (
            int(status_code)
            if isinstance(status_code, int)
            and not isinstance(status_code, bool)
            and 100 <= status_code <= 599
            else None
        )

    def safe_metadata(self) -> dict[str, str | int | None]:
        """Return the allowlisted operational metadata safe for a live checker."""

        return {
            "provider": self.provider,
            "model": self.model,
            "attempts": self.attempts,
            "category": self.category,
            "status_code": self.status_code,
        }


class QAProviderUnavailableError(QAReasonerError):
    """Raised when a configured reasoning backend cannot be used."""


class QAProviderRequestError(QAReasonerError):
    """Raised when a bounded provider request cannot be completed."""


class QAProviderResponseError(QAReasonerError):
    """Raised when a provider returns invalid structured output."""


@dataclass(frozen=True, slots=True)
class QAImage:
    """One canonical image supplied to a reasoning provider.

    ``index`` is the 1-based position exposed to the VLM.  It is intentionally
    distinct from the zero-based source-video frame ordinal.
    """

    data: bytes
    mime_type: str
    index: int
    source_frame_idx: int
    timestamp_ms: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("data must be non-empty bytes")
        if self.mime_type not in _SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"unsupported image MIME type: {self.mime_type!r}")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 1:
            raise ValueError("index must be a positive integer")
        for name in ("source_frame_idx", "timestamp_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.sha256)
        ):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        if not _constant_time_equal(hashlib.sha256(self.data).hexdigest(), self.sha256.lower()):
            raise ValueError("sha256 does not match the canonical image bytes")


@runtime_checkable
class QAReasoner(Protocol):
    """Common interface implemented by Visual Q&A reasoning providers."""

    @property
    def fingerprint(self) -> str:
        """Return a deterministic, secret-free reasoning-cache fingerprint."""

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        """Answer one question using a chronological candidate-image window."""

    def unload(self) -> None:
        """Release provider resources, if any."""


def _constant_time_equal(left: str, right: str) -> bool:
    return (
        hashlib.sha256(left.encode("ascii")).digest()
        == hashlib.sha256(right.encode("ascii")).digest()
    )


def _fingerprint(provider: str, settings: Mapping[str, Any]) -> str:
    payload = {
        "provider": provider,
        "prompt_version": PROMPT_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "settings": dict(settings),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{provider}:v1:{hashlib.sha256(encoded).hexdigest()}"


def _raise_clean(error: QAReasonerError) -> NoReturn:
    """Raise a public error without retaining an SDK exception chain."""

    try:
        raise error from None
    except QAReasonerError as clean:
        clean.__cause__ = None
        clean.__context__ = None
        clean.__suppress_context__ = True
        raise


def _with_error_context(
    error: QAReasonerError,
    *,
    provider: str,
    model: str,
    attempts: int | None = None,
    category: str | None = None,
    status_code: int | None = None,
) -> QAReasonerError:
    """Copy an already-sanitized error while filling safe provider context."""

    message = (
        "the configured provider is unavailable"
        if isinstance(error, QAProviderUnavailableError)
        else "the provider returned an invalid structured response"
        if isinstance(error, QAProviderResponseError)
        else "the provider request failed"
    )
    return type(error)(
        message,
        provider=error.provider or provider,
        model=error.model or model,
        attempts=error.attempts if error.attempts is not None else attempts,
        category=error.category or category,
        status_code=(error.status_code if error.status_code is not None else status_code),
    )


def _validate_images(
    images: Sequence[QAImage],
    *,
    max_images: int,
    max_total_image_bytes: int,
) -> tuple[QAImage, ...]:
    values = tuple(images)
    if not values:
        raise ValueError("at least one canonical image is required")
    if len(values) > max_images:
        raise ValueError(f"image count exceeds the configured maximum of {max_images}")
    indices = tuple(image.index for image in values)
    expected = tuple(range(1, len(values) + 1))
    if indices != expected:
        raise ValueError("image indices must be unique, chronological, and contiguous from 1")
    if sum(len(image.data) for image in values) > max_total_image_bytes:
        raise ValueError("canonical image payload exceeds the configured byte limit")
    return values


def _response_schema(image_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "answerable": {"type": "boolean"},
            "evidence_frame_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": image_count},
                "uniqueItems": True,
            },
        },
        "required": ["answer", "answerable", "evidence_frame_indices"],
    }


def _system_instruction() -> str:
    return (
        "You are a Visual Question Answering judge. Use only the supplied chronological "
        "frames and event description. Answer in the question's language. Use Arabic numerals "
        "for quantities and one to three lowercase words for colors or objects. Do not explain "
        "the answer. Evidence indices must refer to the displayed Frame labels. If the frames "
        "do not establish the answer, set answerable to false, answer to an empty string, and "
        "evidence_frame_indices to an empty array."
    )


def _question_text(request: QAQueryRequest, *, include_json_instruction: bool = False) -> str:
    suffix = ""
    if include_json_instruction:
        suffix = (
            "\nReturn exactly one JSON object with only answer, answerable, and "
            "evidence_frame_indices. Do not use Markdown."
        )
    return f"Event description: {request.event_description}\nQuestion: {request.question}{suffix}"


def _parse_reasoning_output(
    text: str,
    *,
    request: QAQueryRequest,
    spec: QACompetitionSpec,
    image_count: int,
    provider: str,
    model_name: str,
    latency_ms: float,
    attempts: int,
) -> QAReasoningResult:
    if not isinstance(text, str) or not text.strip():
        raise QAProviderResponseError("provider returned no structured response")
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        _raise_clean(QAProviderResponseError("provider returned malformed JSON"))
    if not isinstance(value, dict) or frozenset(value) != _RESPONSE_KEYS:
        raise QAProviderResponseError("provider response does not match the required fields")

    answer = value["answer"]
    answerable = value["answerable"]
    evidence = value["evidence_frame_indices"]
    if not isinstance(answer, str) or len(answer) > 256:
        raise QAProviderResponseError("provider answer must be a bounded string")
    if type(answerable) is not bool:
        raise QAProviderResponseError("provider answerable field must be a boolean")
    if not isinstance(evidence, list):
        raise QAProviderResponseError("provider evidence field must be an array")
    if any(type(index) is not int for index in evidence):
        raise QAProviderResponseError("provider evidence indices must be integers")
    if len(evidence) != len(set(evidence)):
        raise QAProviderResponseError("provider evidence indices must be unique")
    if any(index < 1 or index > image_count for index in evidence):
        raise QAProviderResponseError("provider evidence index is outside the supplied window")

    raw_answer = answer.strip()
    if answerable:
        if not raw_answer or not evidence:
            raise QAProviderResponseError("answerable output requires an answer and evidence")
        try:
            normalized_answer = spec.normalize_answer(raw_answer, question=request.question)
        except (TypeError, ValueError):
            _raise_clean(
                QAProviderResponseError(
                    "provider answer violates the competition normalization policy"
                )
            )
    else:
        if raw_answer or evidence:
            raise QAProviderResponseError("unanswerable output must not include answer evidence")
        normalized_answer = ""

    return QAReasoningResult(
        raw_answer=raw_answer,
        normalized_answer=normalized_answer,
        answerable=answerable,
        evidence_frame_indices=tuple(evidence),
        provider=provider,
        model_name=model_name,
        prompt_version=PROMPT_VERSION,
        latency_ms=max(0.0, latency_ms),
        attempts=attempts,
        cache_status="miss",
        failure=None,
    )


def _default_gemini_client_factory(*, api_key: str, timeout_seconds: float) -> Any:
    try:
        from google import genai
    except (ImportError, ModuleNotFoundError):
        _raise_clean(
            QAProviderUnavailableError(
                "the Gemini client package is unavailable",
                provider="gemini",
                attempts=0,
                category="client_unavailable",
            )
        )
    return genai.Client(
        api_key=api_key,
        http_options={
            "timeout": int(timeout_seconds * 1_000),
            "retry_options": {"attempts": 1},
        },
    )


class GeminiRequestGovernor:
    """Thread-safe rolling request limiter shared by diagnostics and production."""

    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        min_interval_seconds: float = 0.0,
    ) -> None:
        self._limit = requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._timestamps: deque[float] = deque()
        self._last_request_at: float | None = None
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                boundary = now - 60.0
                while self._timestamps and self._timestamps[0] <= boundary:
                    self._timestamps.popleft()
                rpm_wait = (
                    max(0.001, self._timestamps[0] + 60.0 - now)
                    if len(self._timestamps) >= self._limit
                    else 0.0
                )
                interval_wait = (
                    max(0.001, self._last_request_at + self._min_interval_seconds - now)
                    if self._last_request_at is not None
                    and now < self._last_request_at + self._min_interval_seconds
                    else 0.0
                )
                wait_seconds = max(rpm_wait, interval_wait)
                if wait_seconds <= 0:
                    self._timestamps.append(now)
                    self._last_request_at = now
                    return
            self._sleep(wait_seconds)


class _GeminiSharedControls:
    """Process-local quota, concurrency, and cooldown controls for one credential."""

    def __init__(
        self,
        *,
        governor: GeminiRequestGovernor,
        max_concurrency: int,
        clock: Callable[[], float],
    ) -> None:
        self.governor = governor
        self.concurrency = threading.BoundedSemaphore(max_concurrency)
        self.clock = clock
        self.cooldown_lock = threading.Lock()
        self.cooldown_until = 0.0


_GEMINI_SHARED_CONTROLS: dict[tuple[bytes, int, int], _GeminiSharedControls] = {}
_GEMINI_SHARED_CONTROLS_LOCK = threading.Lock()


def _gemini_shared_controls(
    *,
    api_key: str | None,
    requests_per_minute: int,
    max_concurrency: int,
    min_request_interval_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> _GeminiSharedControls:
    """Return shared controls without retaining or exposing the raw credential."""

    credential_digest = hashlib.sha256((api_key or "").encode("utf-8")).digest()
    # Custom clocks/sleep functions are isolated for deterministic tests. Normal
    # production instances use the module-level functions and therefore share one
    # governor, semaphore, and cooldown for the configured Gemini credential.
    key = (credential_digest, id(clock), id(sleep))
    with _GEMINI_SHARED_CONTROLS_LOCK:
        controls = _GEMINI_SHARED_CONTROLS.get(key)
        if controls is None:
            controls = _GeminiSharedControls(
                governor=GeminiRequestGovernor(
                    requests_per_minute,
                    clock=clock,
                    sleep=sleep,
                    min_interval_seconds=min_request_interval_seconds,
                ),
                max_concurrency=max_concurrency,
                clock=clock,
            )
            _GEMINI_SHARED_CONTROLS[key] = controls
        return controls


def _reset_gemini_shared_controls_for_tests() -> None:
    """Clear process controls for isolated offline tests only."""

    with _GEMINI_SHARED_CONTROLS_LOCK:
        _GEMINI_SHARED_CONTROLS.clear()


class GeminiInteractionsReasoner:
    """Google Gemini Interactions provider with bounded local controls."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        client_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        requests_per_minute: int = 13,
        max_concurrency: int = 1,
        retry_base_seconds: float = 5.0,
        retry_max_seconds: float = 60.0,
        rate_limit_cooldown_seconds: float = 60.0,
        cooldown_max_seconds: float = 300.0,
        min_request_interval_seconds: float = 4.5,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        request_governor: GeminiRequestGovernor | None = None,
        max_images: int = 5,
        max_total_image_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must be non-empty when provided")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name, value in (
            ("max_attempts", max_attempts),
            ("requests_per_minute", requests_per_minute),
            ("max_concurrency", max_concurrency),
            ("max_images", max_images),
            ("max_total_image_bytes", max_total_image_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_attempts > 5:
            raise ValueError("max_attempts cannot exceed 5")
        if requests_per_minute > 13:
            raise ValueError("requests_per_minute cannot exceed the Gemini Free Tier cap of 13")
        if max_concurrency > 1:
            raise ValueError("Gemini Free Tier max_concurrency must be 1")
        if retry_base_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError("retry_max_seconds cannot be below retry_base_seconds")
        if rate_limit_cooldown_seconds < 0 or cooldown_max_seconds < 0:
            raise ValueError("rate-limit cooldowns cannot be negative")
        if cooldown_max_seconds < rate_limit_cooldown_seconds:
            raise ValueError("cooldown_max_seconds cannot be below the default cooldown")
        if min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds cannot be negative")

        self._api_key = api_key.strip() if api_key is not None else None
        self.model_name = model_name.strip()
        self._client_factory = client_factory or _default_gemini_client_factory
        self._timeout_seconds = float(timeout_seconds)
        self._max_attempts = max_attempts
        self._retry_base_seconds = float(retry_base_seconds)
        self._retry_max_seconds = float(retry_max_seconds)
        self._rate_limit_cooldown_seconds = float(rate_limit_cooldown_seconds)
        self._cooldown_max_seconds = float(cooldown_max_seconds)
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._jitter = jitter
        self._max_images = max_images
        self._max_total_image_bytes = max_total_image_bytes
        self._client: Any | None = None
        self._client_lock = threading.Lock()
        if request_governor is None:
            self._shared_controls = _gemini_shared_controls(
                api_key=self._api_key,
                requests_per_minute=requests_per_minute,
                max_concurrency=max_concurrency,
                min_request_interval_seconds=min_request_interval_seconds,
                clock=clock,
                sleep=sleep,
            )
        else:
            # An explicitly injected governor owns its test/diagnostic scope.
            self._shared_controls = _GeminiSharedControls(
                governor=request_governor,
                max_concurrency=max_concurrency,
                clock=clock,
            )
        self._concurrency = self._shared_controls.concurrency
        self._rate_limiter = self._shared_controls.governor

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_name={self.model_name!r})"

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.provider_name,
            {
                "model_name": self.model_name,
                "max_images": self._max_images,
                "max_total_image_bytes": self._max_total_image_bytes,
                "structured_output": True,
            },
        )

    @property
    def client_initialized(self) -> bool:
        """Return whether the API client has been constructed."""

        return self._client is not None

    def initialize(self) -> None:
        """Explicitly construct the configured client without issuing a request."""

        self._get_client()

    @property
    def max_attempts(self) -> int:
        """Return the hard provider-call attempt cap."""

        return self._max_attempts

    def _get_client(self) -> Any:
        if self._api_key is None:
            raise QAProviderUnavailableError(
                "a Gemini API credential is not configured",
                provider=self.provider_name,
                model=self.model_name,
                attempts=0,
                category="not_configured",
            )
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    self._client = self._client_factory(
                        api_key=self._api_key,
                        timeout_seconds=self._timeout_seconds,
                    )
                except QAReasonerError as exc:
                    _raise_clean(
                        _with_error_context(
                            exc,
                            provider=self.provider_name,
                            model=self.model_name,
                            attempts=0,
                            category="client_unavailable",
                        )
                    )
                except Exception:
                    _raise_clean(
                        QAProviderUnavailableError(
                            "the Gemini client could not be initialized",
                            provider=self.provider_name,
                            model=self.model_name,
                            attempts=0,
                            category="client_initialization",
                        )
                    )
        return self._client

    def _create_interaction(
        self,
        request: QAQueryRequest,
        images: tuple[QAImage, ...],
    ) -> Any:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Frame {image.index} (source_frame_idx={image.source_frame_idx}, "
                        f"timestamp_ms={image.timestamp_ms})"
                    ),
                }
            )
            content.append(
                {
                    "type": "image",
                    "mime_type": image.mime_type,
                    "data": base64.b64encode(image.data).decode("ascii"),
                }
            )
        content.append({"type": "text", "text": _question_text(request)})
        client = self._get_client()
        return client.interactions.create(
            model=self.model_name,
            input=content,
            response_format=[
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _response_schema(len(images)),
                }
            ],
            system_instruction=_system_instruction(),
            store=False,
        )

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        values = _validate_images(
            images,
            max_images=self._max_images,
            max_total_image_bytes=self._max_total_image_bytes,
        )
        if not isinstance(request, QAQueryRequest):
            raise TypeError("request must be a QAQueryRequest")
        if not isinstance(spec, QACompetitionSpec):
            raise TypeError("spec must be a QACompetitionSpec")

        started = self._clock()
        last_category = "provider_request_failed"
        last_status_code: int | None = None
        last_retry_after: float | None = None
        with self._concurrency:
            self._raise_if_cooling_down()
            for attempt in range(1, self._max_attempts + 1):
                self._rate_limiter.acquire()
                retryable = False
                retry_after: float | None = None
                try:
                    interaction = self._create_interaction(request, values)
                    output_text = getattr(interaction, "output_text", None)
                    if output_text is None and isinstance(interaction, Mapping):
                        output_text = interaction.get("output_text")
                    return _parse_reasoning_output(
                        output_text,
                        request=request,
                        spec=spec,
                        image_count=len(values),
                        provider=self.provider_name,
                        model_name=self.model_name,
                        latency_ms=(self._clock() - started) * 1_000,
                        attempts=attempt,
                    )
                except QAProviderResponseError as exc:
                    _raise_clean(
                        _with_error_context(
                            exc,
                            provider=self.provider_name,
                            model=self.model_name,
                            attempts=attempt,
                            category="response_invalid",
                        )
                    )
                except QAProviderUnavailableError as exc:
                    _raise_clean(
                        _with_error_context(
                            exc,
                            provider=self.provider_name,
                            model=self.model_name,
                            attempts=attempt,
                            category="client_unavailable",
                        )
                    )
                except Exception as exc:
                    details = _provider_exception_details(exc, wall_clock=self._wall_clock)
                    retryable = details.retryable
                    last_category = details.category
                    last_status_code = details.status_code
                    retry_after = details.retry_after_seconds
                    last_retry_after = retry_after
                    if not retryable:
                        _raise_clean(
                            QAProviderRequestError(
                                "the Gemini request failed with a non-retryable provider error",
                                provider=self.provider_name,
                                model=self.model_name,
                                attempts=attempt,
                                category=details.category,
                                status_code=details.status_code,
                            )
                        )
                if attempt == self._max_attempts:
                    break
                if retryable:
                    self._sleep(self._retry_delay(attempt, retry_after))
            # Publish the terminal 429 cooldown before releasing the shared
            # semaphore, so another waiting window cannot slip through.
            if last_status_code == 429:
                self._begin_rate_limit_cooldown(last_retry_after)

        _raise_clean(
            QAProviderRequestError(
                "the Gemini request failed after bounded retries",
                provider=self.provider_name,
                model=self.model_name,
                attempts=self._max_attempts,
                category=last_category,
                status_code=last_status_code,
            )
        )

    def _retry_delay(
        self,
        completed_attempts: int,
        server_hint_seconds: float | None = None,
    ) -> float:
        return _computed_retry_delay(
            completed_attempts,
            base_seconds=self._retry_base_seconds,
            max_seconds=self._retry_max_seconds,
            server_hint_seconds=server_hint_seconds,
            jitter=self._jitter,
        )

    def _raise_if_cooling_down(self) -> None:
        controls = self._shared_controls
        with controls.cooldown_lock:
            cooling_down = controls.clock() < controls.cooldown_until
        if cooling_down:
            _raise_clean(
                QAProviderRequestError(
                    "the Gemini provider is in a bounded rate-limit cooldown",
                    provider=self.provider_name,
                    model=self.model_name,
                    attempts=0,
                    category="rate_limited",
                    status_code=429,
                )
            )

    def _begin_rate_limit_cooldown(self, server_hint_seconds: float | None) -> None:
        hinted = max(0.0, float(server_hint_seconds or 0.0))
        duration = min(
            self._cooldown_max_seconds,
            max(self._rate_limit_cooldown_seconds, hinted),
        )
        controls = self._shared_controls
        with controls.cooldown_lock:
            controls.cooldown_until = max(
                controls.cooldown_until,
                controls.clock() + duration,
            )

    def unload(self) -> None:
        """Close and forget the lazily constructed API client."""

        with self._client_lock:
            client = self._client
            self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                return


@dataclass(frozen=True, slots=True)
class _ProviderExceptionDetails:
    retryable: bool
    category: str
    status_code: int | None
    retry_after_seconds: float | None = None


def _status_code(exc: BaseException) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return int(value)
        if isinstance(value, str) and value.isdecimal():
            parsed = int(value)
            if 100 <= parsed <= 599:
                return parsed
    return None


def _optional_exception_types(module_name: str, names: Sequence[str]) -> tuple[type, ...]:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return ()
    return tuple(
        value for name in names if isinstance((value := getattr(module, name, None)), type)
    )


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    values: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(values) < 4:
        values.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(values)


def _provider_header(exc: BaseException, name: str) -> str | None:
    """Read one retry header without retaining or exposing provider content."""

    target = name.casefold()
    for candidate in _exception_chain(exc):
        for headers in (
            getattr(candidate, "headers", None),
            getattr(getattr(candidate, "response", None), "headers", None),
        ):
            if headers is None:
                continue
            if isinstance(headers, Mapping):
                for raw_name, raw_value in headers.items():
                    if str(raw_name).casefold() == target:
                        value = str(raw_value).strip()
                        return value or None
                continue
            getter = getattr(headers, "get", None)
            if callable(getter):
                try:
                    value = getter(name)
                except Exception:
                    continue
                if value is not None and str(value).strip():
                    return str(value).strip()
    return None


def _bounded_retry_seconds(value: float) -> float | None:
    if value < 0 or value != value or value == float("inf"):
        return None
    return min(300.0, value)


def _retry_after_hint(exc: BaseException, *, wall_clock: Callable[[], float]) -> float | None:
    value = _provider_header(exc, "Retry-After")
    if value is not None:
        try:
            return _bounded_retry_seconds(float(value))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                return _bounded_retry_seconds(parsed.timestamp() - wall_clock())
            except (TypeError, ValueError, OverflowError):
                pass

    value = _provider_header(exc, "x-ratelimit-reset")
    if value is None:
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    if numeric >= 1_000_000_000_000:
        numeric /= 1_000.0
    # Reset headers are seen both as delta seconds and Unix timestamps.
    if numeric >= 1_000_000_000:
        numeric -= wall_clock()
    return _bounded_retry_seconds(numeric)


def _provider_exception_details(
    exc: Exception,
    *,
    wall_clock: Callable[[], float] = time.time,
) -> _ProviderExceptionDetails:
    """Classify an SDK error without copying any provider-controlled content."""

    status_code = _status_code(exc)
    if status_code == 400:
        return _ProviderExceptionDetails(False, "invalid_request", status_code)
    if status_code in {401, 403}:
        return _ProviderExceptionDetails(False, "auth_or_permission", status_code)
    if status_code == 404:
        return _ProviderExceptionDetails(False, "model_unavailable", status_code)
    if status_code == 408:
        return _ProviderExceptionDetails(True, "network_unreachable", status_code)
    if status_code == 429:
        return _ProviderExceptionDetails(
            True,
            "rate_limited",
            status_code,
            _retry_after_hint(exc, wall_clock=wall_clock),
        )
    if status_code is not None and 500 <= status_code <= 599:
        return _ProviderExceptionDetails(True, "server_unavailable", status_code)
    if status_code is not None:
        return _ProviderExceptionDetails(False, "request_rejected", status_code)

    timeout_types = (
        TimeoutError,
        *_optional_exception_types("httpx", ("TimeoutException",)),
        *_optional_exception_types("requests.exceptions", ("Timeout",)),
        *_optional_exception_types(
            "google.api_core.exceptions",
            ("DeadlineExceeded", "GatewayTimeout"),
        ),
    )
    connection_types = (
        ConnectionError,
        *_optional_exception_types("httpx", ("TransportError",)),
        *_optional_exception_types("requests.exceptions", ("ConnectionError",)),
        *_optional_exception_types("google.auth.exceptions", ("TransportError",)),
        *_optional_exception_types("google.genai.errors", ("APIConnectionError",)),
        *_optional_exception_types(
            "google.api_core.exceptions",
            ("ServiceUnavailable",),
        ),
    )
    google_server_types = _optional_exception_types("google.genai.errors", ("ServerError",))
    for candidate in _exception_chain(exc):
        if timeout_types and isinstance(candidate, timeout_types):
            return _ProviderExceptionDetails(True, "network_unreachable", None)
        if connection_types and isinstance(candidate, connection_types):
            return _ProviderExceptionDetails(True, "network_unreachable", None)
        if google_server_types and isinstance(candidate, google_server_types):
            return _ProviderExceptionDetails(True, "server_unavailable", None)
    return _ProviderExceptionDetails(False, "provider_request_failed", None)


def retry_delay_for_exception(
    exc: Exception,
    completed_attempts: int,
    *,
    base_seconds: float = 5.0,
    max_seconds: float = 60.0,
    jitter: Callable[[float, float], float] = random.uniform,
    wall_clock: Callable[[], float] = time.time,
) -> float:
    """Return the production retry delay for an SDK exception."""

    if completed_attempts < 1:
        raise ValueError("completed_attempts must be positive")
    if base_seconds < 0 or max_seconds < base_seconds:
        raise ValueError("invalid retry delay bounds")
    details = _provider_exception_details(exc, wall_clock=wall_clock)
    return _computed_retry_delay(
        completed_attempts,
        base_seconds=base_seconds,
        max_seconds=max_seconds,
        server_hint_seconds=details.retry_after_seconds,
        jitter=jitter,
    )


def _is_retryable_exception(exc: Exception) -> bool:
    return _provider_exception_details(exc).retryable


def sanitize_provider_exception(
    exc: Exception,
    *,
    provider: str,
    model: str,
    attempts: int,
) -> QAProviderRequestError:
    """Convert an SDK exception to a metadata-only public diagnostic error."""

    details = _provider_exception_details(exc)
    return QAProviderRequestError(
        "the provider diagnostic request failed",
        provider=provider,
        model=model,
        attempts=attempts,
        category=details.category,
        status_code=details.status_code,
    )


class Qwen3VLReasoner:
    """Cache-only Qwen3-VL provider with explicit GPU lifecycle hooks."""

    provider_name = "qwen3-vl"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_MODEL,
        revision: str | None = None,
        cache_dir: str | None = None,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 128,
        local_files_only: bool = True,
        max_images: int = 5,
        max_total_image_bytes: int = 20 * 1024 * 1024,
        max_pixels: int = 1024 * 1024,
        gpu_lease: Callable[[], AbstractContextManager[Any]] | None = None,
        before_load: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be non-empty when provided")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be non-empty")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("dtype must be non-empty")
        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("max_images", max_images),
            ("max_total_image_bytes", max_total_image_bytes),
            ("max_pixels", max_pixels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if local_files_only is not True:
            raise ValueError("Qwen3VLReasoner requires local_files_only=True")

        self.model_name = model_name.strip()
        self.revision = revision.strip() if revision else None
        self.cache_dir = cache_dir
        self.device = device.strip()
        self.dtype = dtype.strip()
        self.max_new_tokens = max_new_tokens
        self.local_files_only = True
        self._max_images = max_images
        self._max_total_image_bytes = max_total_image_bytes
        self.max_pixels = max_pixels
        self._gpu_lease = gpu_lease or nullcontext
        self._before_load = before_load
        self._clock = clock
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._lease_context: AbstractContextManager[Any] | None = None
        self._lifecycle_lock = threading.RLock()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_name={self.model_name!r}, "
            f"device={self.device!r}, local_files_only=True)"
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.provider_name,
            {
                "model_name": self.model_name,
                "revision": self.revision,
                "device": self.device,
                "dtype": self.dtype,
                "max_new_tokens": self.max_new_tokens,
                "max_images": self._max_images,
                "max_total_image_bytes": self._max_total_image_bytes,
                "max_pixels": self.max_pixels,
                "local_files_only": self.local_files_only,
                "do_sample": False,
            },
        )

    @property
    def loaded(self) -> bool:
        """Return whether processor and model are currently resident."""

        return self._model is not None and self._processor is not None

    def load(self) -> None:
        """Load a local cached model while holding the injected GPU lease."""

        with self._lifecycle_lock:
            if self.loaded:
                return
            try:
                lease = self._gpu_lease()
                lease.__enter__()
            except Exception:
                _raise_clean(QAProviderUnavailableError("the Qwen GPU lease is unavailable"))
            self._lease_context = lease
            if self._before_load is not None:
                try:
                    self._before_load()
                except Exception:
                    self._release_lease_locked()
                    _raise_clean(
                        QAProviderUnavailableError("the pre-load GPU handoff callback failed")
                    )
            try:
                self._load_locked()
            except Exception:
                self._release_lease_locked()
                raise

    def _load_locked(self) -> None:
        enforce_huggingface_offline()
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except (ImportError, ModuleNotFoundError):
            _raise_clean(QAProviderUnavailableError("the local Qwen runtime is unavailable"))

        processor_class = getattr(transformers, "AutoProcessor", None)
        model_class = None
        for class_name in (
            "AutoModelForMultimodalLM",
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
        ):
            model_class = getattr(transformers, class_name, None)
            if model_class is not None:
                break
        if processor_class is None or model_class is None:
            raise QAProviderUnavailableError(
                "the installed Transformers build does not support Qwen3-VL"
            )

        common_kwargs: dict[str, Any] = {"local_files_only": True}
        if self.revision is not None:
            common_kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            common_kwargs["cache_dir"] = self.cache_dir
        try:
            processor = processor_class.from_pretrained(
                self.model_name,
                max_pixels=self.max_pixels,
                **common_kwargs,
            )
            model_kwargs = dict(common_kwargs)
            torch_dtype = getattr(torch, self.dtype, None)
            if torch_dtype is None:
                raise QAProviderUnavailableError("the configured Qwen dtype is unsupported")
            model_kwargs["torch_dtype"] = torch_dtype
            model = model_class.from_pretrained(self.model_name, **model_kwargs)
            if hasattr(model, "to"):
                model = model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
        except QAReasonerError:
            raise
        except Exception:
            _raise_clean(
                QAProviderUnavailableError("cached Qwen3-VL artifacts could not be loaded")
            )
        self._processor = processor
        self._model = model
        self._torch = torch

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        values = _validate_images(
            images,
            max_images=self._max_images,
            max_total_image_bytes=self._max_total_image_bytes,
        )
        if not isinstance(request, QAQueryRequest):
            raise TypeError("request must be a QAQueryRequest")
        if not isinstance(spec, QACompetitionSpec):
            raise TypeError("spec must be a QACompetitionSpec")

        started = self._clock()
        with self._lifecycle_lock:
            self.load()
            try:
                output_text = self._generate_locked(request, values)
            except QAReasonerError:
                raise
            except Exception:
                _raise_clean(QAProviderRequestError("local Qwen inference failed"))
        return _parse_reasoning_output(
            output_text,
            request=request,
            spec=spec,
            image_count=len(values),
            provider=self.provider_name,
            model_name=self.model_name,
            latency_ms=(self._clock() - started) * 1_000,
            attempts=1,
        )

    def _generate_locked(
        self,
        request: QAQueryRequest,
        images: tuple[QAImage, ...],
    ) -> str:
        if self._processor is None or self._model is None or self._torch is None:
            raise QAProviderUnavailableError("the local Qwen model is not loaded")
        try:
            from PIL import Image

            user_content: list[dict[str, Any]] = []
            opened_images = []
            for image in images:
                opened = Image.open(io.BytesIO(image.data)).convert("RGB")
                if opened.width * opened.height > self.max_pixels:
                    scale = (self.max_pixels / (opened.width * opened.height)) ** 0.5
                    target = (
                        max(1, int(opened.width * scale)),
                        max(1, int(opened.height * scale)),
                    )
                    opened.thumbnail(target, Image.Resampling.LANCZOS)
                opened_images.append(opened)
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Frame {image.index} (source_frame_idx={image.source_frame_idx}, "
                            f"timestamp_ms={image.timestamp_ms})"
                        ),
                    }
                )
                user_content.append({"type": "image", "image": opened})
            user_content.append(
                {
                    "type": "text",
                    "text": _question_text(request, include_json_instruction=True),
                }
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": _system_instruction()}]},
                {"role": "user", "content": user_content},
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_inputs(inputs, self.device)
            inference_mode = getattr(self._torch, "inference_mode", nullcontext)
            with inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            if isinstance(generated, str):
                return generated
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise QAProviderResponseError("Qwen processor omitted input token IDs")
            trimmed = [
                output_ids[len(prompt_ids) :]
                for prompt_ids, output_ids in zip(input_ids, generated, strict=True)
            ]
            decoded = self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not decoded or not isinstance(decoded[0], str):
                raise QAProviderResponseError("Qwen returned no decodable response")
            return decoded[0]
        except QAReasonerError:
            raise
        except Exception:
            _raise_clean(QAProviderRequestError("local Qwen preprocessing or generation failed"))
        finally:
            for opened in locals().get("opened_images", []):
                opened.close()

    def unload(self) -> None:
        """Release the local model, cached tensors, and held GPU lease."""

        with self._lifecycle_lock:
            self._model = None
            self._processor = None
            torch = self._torch
            self._torch = None
            gc.collect()
            if torch is None:
                torch = sys.modules.get("torch")
            cuda = getattr(torch, "cuda", None) if torch is not None else None
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                try:
                    empty_cache()
                except Exception:
                    pass
            self._release_lease_locked()

    def _release_lease_locked(self) -> None:
        lease = self._lease_context
        self._lease_context = None
        if lease is not None:
            try:
                lease.__exit__(None, None, None)
            except Exception:
                return


def _move_inputs(inputs: Any, device: str) -> Any:
    move = getattr(inputs, "to", None)
    if callable(move):
        return move(device)
    if not isinstance(inputs, Mapping):
        raise QAProviderRequestError("Qwen processor returned unsupported model inputs")
    return {
        name: value.to(device) if callable(getattr(value, "to", None)) else value
        for name, value in inputs.items()
    }


_FALLBACK_ERROR_CATEGORIES = frozenset(
    {
        "network_unreachable",
        "rate_limited",
        "server_unavailable",
    }
)


def _allows_provider_fallback(error: QAReasonerError) -> bool:
    """Allow fallback only after Gemini reports a transient request failure.

    Provider setup, missing credentials, model lookup, authentication, request
    validation and response-schema failures are operational/configuration bugs;
    silently replacing them with a local model would hide the real fault.
    ``GeminiInteractionsReasoner`` owns retries and emits the three categories
    below only after its bounded retry loop (or its shared 429 cooldown) has
    denied the request.
    """

    return (
        isinstance(error, QAProviderRequestError)
        and error.category in _FALLBACK_ERROR_CATEGORIES
    )


class ProviderChainReasoner:
    """Try bounded providers in order without exposing sensitive failures."""

    provider_name = "provider-chain"

    def __init__(self, providers: Sequence[QAReasoner]) -> None:
        self.providers = tuple(providers)
        if not self.providers:
            raise ValueError("providers must not be empty")

    @property
    def fingerprint(self) -> str:
        ordered = []
        for provider in self.providers:
            value = getattr(provider, "fingerprint", None)
            if callable(value):
                value = value()
            if not isinstance(value, str) or not value:
                value = f"{type(provider).__module__}.{type(provider).__qualname__}"
            ordered.append(value)
        return _fingerprint(self.provider_name, {"ordered_providers": ordered})

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        last_error: QAReasonerError | None = None
        for provider in self.providers:
            provider_name = str(getattr(provider, "provider_name", type(provider).__name__))
            model_name = getattr(provider, "model_name", None)
            try:
                return provider.reason(request, images, spec)
            except QAReasonerError as exc:
                message = (
                    "the Visual Q&A provider is unavailable"
                    if isinstance(exc, QAProviderUnavailableError)
                    else "the Visual Q&A provider returned an invalid response"
                    if isinstance(exc, QAProviderResponseError)
                    else "the Visual Q&A provider request failed"
                )
                last_error = type(exc)(
                    message,
                    provider=exc.provider or provider_name,
                    model=exc.model or str(model_name or "unknown"),
                    attempts=exc.attempts,
                    category=exc.category,
                    status_code=exc.status_code,
                )
                if _allows_provider_fallback(last_error):
                    continue
                _raise_clean(last_error)
            except Exception:
                last_error = QAProviderRequestError(
                    "a Visual Q&A provider failed unexpectedly",
                    provider=provider_name,
                    model=str(model_name or "unknown"),
                    category="provider_request_failed",
                )
                _raise_clean(last_error)
        if last_error is not None:
            _raise_clean(last_error)
        raise QAProviderUnavailableError(
            "all configured Visual Q&A providers failed",
            provider=self.provider_name,
            attempts=len(self.providers),
            category="provider_chain_exhausted",
        )

    def unload(self) -> None:
        for provider in self.providers:
            try:
                provider.unload()
            except Exception:
                continue


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_QWEN_MODEL",
    "PROMPT_VERSION",
    "GeminiInteractionsReasoner",
    "GeminiRequestGovernor",
    "ProviderChainReasoner",
    "QAImage",
    "QAProviderRequestError",
    "QAProviderResponseError",
    "QAProviderUnavailableError",
    "QAReasoner",
    "QAReasonerError",
    "QAReasoningResult",
    "Qwen3VLReasoner",
    "retry_delay_for_exception",
    "sanitize_provider_exception",
]
