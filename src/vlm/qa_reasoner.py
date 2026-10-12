"""Provider-neutral Visual Q&A contracts, validation, and safe errors."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol, runtime_checkable

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult

PROMPT_VERSION = "visual-qa-judge-provider-neutral-v6"
NORMALIZER_VERSION = "qa-answer-normalizer-v2"
RESPONSE_SCHEMA_VERSION = "qa-structured-response-v1"

_SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_RESPONSE_KEYS = frozenset({"answer", "answerable", "evidence_frame_indices"})
_SAFE_ERROR_CATEGORIES = frozenset(
    {
        "auth_or_permission",
        "client_initialization",
        "client_unavailable",
        "gemini_response_invalid",
        "gemini_timeout",
        "inference_failed",
        "gpu_runtime_incompatible",
        "inference_timeout",
        "worker_terminated",
        "invalid_request",
        "model_missing",
        "model_unavailable",
        "network_unreachable",
        "not_configured",
        "ollama_context_exceeded",
        "ollama_response_invalid",
        "ollama_timeout",
        "provider_chain_exhausted",
        "provider_request_failed",
        "rate_limited",
        "request_rejected",
        "resource_exhausted",
        "response_invalid",
        "server_unavailable",
    }
)


def _safe_error_label(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate or len(candidate) > 128:
        return None
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._:/")
    return candidate if all(character in allowed for character in candidate) else None


class QAReasonerError(RuntimeError):
    """Base class for provider failures containing only allowlisted metadata."""

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
        super().__init__(message)
        self.provider = _safe_error_label(provider)
        self.model = _safe_error_label(model)
        self.model_name = self.model
        self.attempts = (
            attempts
            if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 0
            else None
        )
        self.category = category if category in _SAFE_ERROR_CATEGORIES else None
        self.status_code = (
            status_code
            if isinstance(status_code, int)
            and not isinstance(status_code, bool)
            and 100 <= status_code <= 599
            else None
        )

    def safe_metadata(self) -> dict[str, str | int | None]:
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
    """One canonical image supplied to a provider in chronological order."""

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
        actual = hashlib.sha256(self.data).hexdigest()
        if not hmac.compare_digest(actual, self.sha256.casefold()):
            raise ValueError("sha256 does not match the canonical image bytes")


@runtime_checkable
class QAReasoner(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult: ...

    def unload(self) -> None: ...


def _fingerprint(provider: str, settings: Mapping[str, Any]) -> str:
    payload = {
        "provider": provider,
        "prompt_version": PROMPT_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "settings": dict(settings),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{provider}:v2:{hashlib.sha256(encoded).hexdigest()}"


def _raise_clean(error: QAReasonerError) -> NoReturn:
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
        status_code=error.status_code if error.status_code is not None else status_code,
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
    if tuple(image.index for image in values) != tuple(range(1, len(values) + 1)):
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
        "You are an expert video visual question answering system. Analyze only the supplied "
        "chronological images and event description. Return only JSON matching the requested "
        "schema. The answer field must contain the direct answer without explanation: use "
        "digits for counts and one to three lowercase words for text, colors, or objects. "
        "For yes/no questions, answer exactly yes or no; never return true or false. "
        "Evidence indices refer to the supplied image order. Before returning JSON, enforce "
        "exactly one of these two valid states: (A) visible evidence exists, so answerable is "
        "true, answer is non-empty, and evidence_frame_indices contains at least one index; "
        "or (B) evidence is insufficient, so answerable is false, answer is the empty string, "
        "and evidence_frame_indices is empty. Never return answerable=true with an empty answer "
        "or empty evidence. Example for three visible objects: "
        '{"answer":"3","answerable":true,"evidence_frame_indices":[1]}. '
        "Every evidence index must be unique and appear at most once. When exactly one image is "
        "supplied, the only valid non-empty evidence list is [1], never [1,1] or [1,1,1]."
    )


def _question_text(request: QAQueryRequest, *, include_json_instruction: bool = False) -> str:
    suffix = (
        "\nReturn exactly one JSON object with answer, answerable, and "
        "evidence_frame_indices; do not use Markdown."
        if include_json_instruction
        else ""
    )
    transcript = ""
    if request.transcript_context:
        transcript = (
            "\nNearby ASR transcript (timestamped supporting evidence; ignore it when "
            "irrelevant or contradicted by the image):\n"
            + "\n".join(request.transcript_context)
        )
    return (
        f"Event description: {request.event_description}"
        f"{transcript}\nQuestion: {request.question}{suffix}"
    )


def _is_yes_no_question(question: str) -> bool:
    normalized = " ".join(str(question).strip().casefold().split())
    english_prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "has ",
        "have ",
        "had ",
        "will ",
        "would ",
        "should ",
    )
    return normalized.startswith(english_prefixes) or (
        normalized.startswith("có ") and " không" in normalized
    )


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
        raise QAProviderResponseError(
            "provider returned no structured response", category="response_invalid"
        )
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        _raise_clean(
            QAProviderResponseError("provider returned malformed JSON", category="response_invalid")
        )
    if not isinstance(value, dict) or frozenset(value) != _RESPONSE_KEYS:
        raise QAProviderResponseError(
            "provider response does not match the required fields", category="response_invalid"
        )
    answer = value["answer"]
    answerable = value["answerable"]
    evidence = value["evidence_frame_indices"]
    if not isinstance(answer, str) or len(answer) > 256:
        raise QAProviderResponseError(
            "provider answer must be a bounded string", category="response_invalid"
        )
    if type(answerable) is not bool or not isinstance(evidence, list):
        raise QAProviderResponseError(
            "provider response fields are invalid", category="response_invalid"
        )
    if any(type(index) is not int for index in evidence):
        raise QAProviderResponseError(
            "provider evidence indices are invalid", category="response_invalid"
        )
    # Small local vision models occasionally repeat the same valid index even
    # with JSON Schema uniqueItems enabled. Stable de-duplication preserves the
    # model's provenance without inventing or reordering evidence.
    evidence = list(dict.fromkeys(evidence))
    if any(index < 1 or index > image_count for index in evidence):
        raise QAProviderResponseError(
            "provider evidence index is outside the supplied window", category="response_invalid"
        )
    raw_answer = answer.strip()
    if answerable:
        if not raw_answer or not evidence:
            raise QAProviderResponseError(
                "answerable output requires an answer and evidence", category="response_invalid"
            )
        if _is_yes_no_question(request.question):
            boolean_answer = {"true": "yes", "false": "no"}.get(raw_answer.casefold())
            if boolean_answer is not None:
                raw_answer = boolean_answer
        try:
            normalized_answer = spec.normalize_answer(raw_answer, question=request.question)
        except (TypeError, ValueError):
            _raise_clean(
                QAProviderResponseError(
                    "provider answer violates the competition normalization policy",
                    category="response_invalid",
                )
            )
    else:
        if raw_answer or evidence:
            raise QAProviderResponseError(
                "unanswerable output must not include answer evidence",
                category="response_invalid",
            )
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


_CHAIN_FALLBACK_CATEGORIES = frozenset(
    {"network_unreachable", "server_unavailable", "ollama_timeout", "resource_exhausted"}
)


class ProviderChainReasoner:
    """Compatibility chain for injected generic reasoners."""

    provider_name = "provider-chain"

    def __init__(self, providers: Sequence[QAReasoner]) -> None:
        self.providers = tuple(providers)
        if not self.providers:
            raise ValueError("providers must not be empty")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.provider_name,
            {
                "providers": [
                    str(getattr(value, "fingerprint", "unknown")) for value in self.providers
                ]
            },
        )

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        last: QAReasonerError | None = None
        for index, provider in enumerate(self.providers):
            try:
                return provider.reason(request, images, spec)
            except QAReasonerError as error:
                last = error
                if (
                    index + 1 < len(self.providers)
                    and isinstance(error, QAProviderRequestError)
                    and error.category in _CHAIN_FALLBACK_CATEGORIES
                ):
                    continue
                _raise_clean(error)
        if last is not None:
            _raise_clean(last)
        raise QAProviderUnavailableError(
            "no Visual Q&A provider is configured",
            provider=self.provider_name,
            category="provider_chain_exhausted",
        )

    def unload(self) -> None:
        for provider in self.providers:
            try:
                provider.unload()
            except Exception:
                continue


def sanitize_provider_exception(
    _error: BaseException,
    *,
    provider: str,
    model: str,
    attempts: int = 1,
) -> QAReasonerError:
    """Convert an arbitrary diagnostic exception without retaining its text."""

    return QAProviderRequestError(
        "the provider diagnostic request failed",
        provider=provider,
        model=model,
        attempts=attempts,
        category="provider_request_failed",
    )


__all__ = [
    "NORMALIZER_VERSION",
    "PROMPT_VERSION",
    "ProviderChainReasoner",
    "QAImage",
    "QAProviderRequestError",
    "QAProviderResponseError",
    "QAProviderUnavailableError",
    "QAReasoner",
    "QAReasonerError",
    "QAReasoningResult",
    "RESPONSE_SCHEMA_VERSION",
    "sanitize_provider_exception",
]
