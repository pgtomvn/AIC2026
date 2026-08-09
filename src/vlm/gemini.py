"""Single-frame Gemini Flash provider for Visual Q&A."""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.vlm.qa_reasoner import (
    QAImage,
    QAProviderRequestError,
    QAProviderResponseError,
    QAProviderUnavailableError,
    _fingerprint,
    _parse_reasoning_output,
    _question_text,
    _response_schema,
    _system_instruction,
    _validate_images,
)

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def _interaction_text(value: Any) -> str:
    text = getattr(value, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    outputs = getattr(value, "outputs", None)
    if not isinstance(outputs, Sequence):
        return ""
    parts: list[str] = []
    for output in outputs:
        candidate = output.get("text") if isinstance(output, Mapping) else getattr(output, "text", None)
        if isinstance(candidate, str):
            parts.append(candidate)
    return "".join(parts)


def _status_code(error: BaseException) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) and 100 <= value <= 599 else None


def _provider_error(error: BaseException, *, model: str) -> QAProviderRequestError:
    status = _status_code(error)
    name = type(error).__name__.casefold()
    if status in {401, 403}:
        category = "auth_or_permission"
    elif status in {400, 404}:
        category = "invalid_request" if status == 400 else "model_missing"
    elif status == 408 or "timeout" in name:
        category = "gemini_timeout"
    elif status == 429:
        category = "rate_limited"
    elif status is not None and status >= 500:
        category = "server_unavailable"
    elif isinstance(error, (ConnectionError, OSError)) or "connection" in name or "transport" in name:
        category = "network_unreachable"
    else:
        category = "provider_request_failed"
    return QAProviderRequestError(
        "the Gemini request failed",
        provider="gemini",
        model=model,
        attempts=1,
        category=category,
        status_code=status,
    )


class GeminiFlashReasoner:
    """Reason over exactly one canonical frame through Gemini Interactions."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float = 5.0,
        max_total_image_bytes: int = 20 * 1024 * 1024,
        client_factory: Callable[..., Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self.model_name = str(model_name).strip()
        if not self.model_name or len(self.model_name) > 128:
            raise ValueError("Gemini model_name must be a bounded non-empty string")
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("Gemini timeout must be positive")
        self.max_total_image_bytes = int(max_total_image_bytes)
        self._client_factory = client_factory
        self._client: Any | None = None
        self._monotonic = monotonic
        self._lock = threading.Lock()

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.provider_name,
            {
                "model": self.model_name,
                "api": "interactions-v1",
                "images": 1,
                "thinking_level": "minimal",
                "sdk_attempts": 1,
            },
        )

    @property
    def max_attempts(self) -> int:
        return 1

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise QAProviderUnavailableError(
                "Gemini is not configured",
                provider=self.provider_name,
                model=self.model_name,
                attempts=0,
                category="not_configured",
            )
        factory = self._client_factory
        if factory is None:
            try:
                from google import genai
            except (ImportError, ModuleNotFoundError):
                raise QAProviderUnavailableError(
                    "the Gemini client package is unavailable",
                    provider=self.provider_name,
                    model=self.model_name,
                    attempts=0,
                    category="client_unavailable",
                ) from None
            factory = genai.Client
        try:
            self._client = factory(
                api_key=self._api_key,
                http_options={
                    "timeout": int(self.timeout_seconds * 1_000),
                    "retry_options": {"attempts": 1},
                },
            )
        except Exception as error:
            raise _provider_error(error, model=self.model_name) from None
        return self._client

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        if not request.allow_external_frame_transfer:
            raise ValueError("Gemini requires explicit external frame transfer consent")
        values = _validate_images(
            images,
            max_images=1,
            max_total_image_bytes=self.max_total_image_bytes,
        )
        if len(values) != 1:
            raise ValueError("Gemini single-frame QA requires exactly one image")
        image = values[0]
        payload = [
            {"type": "text", "text": _question_text(request)},
            {
                "type": "image",
                "mime_type": image.mime_type,
                "data": base64.b64encode(image.data).decode("ascii"),
            },
        ]
        started = self._monotonic()
        with self._lock:
            try:
                interaction = self._get_client().interactions.create(
                    model=self.model_name,
                    input=payload,
                    response_format=[
                        {
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": _response_schema(1),
                        }
                    ],
                    system_instruction=_system_instruction(),
                    generation_config={"thinking_level": "minimal", "max_output_tokens": 64},
                    store=False,
                )
            except (QAProviderRequestError, QAProviderUnavailableError):
                raise
            except Exception as error:
                raise _provider_error(error, model=self.model_name) from None
        latency_ms = (self._monotonic() - started) * 1_000
        try:
            return _parse_reasoning_output(
                _interaction_text(interaction),
                request=request,
                spec=spec,
                image_count=1,
                provider=self.provider_name,
                model_name=self.model_name,
                latency_ms=latency_ms,
                attempts=1,
            )
        except QAProviderResponseError:
            raise QAProviderResponseError(
                "Gemini returned an invalid structured response",
                provider=self.provider_name,
                model=self.model_name,
                attempts=1,
                category="gemini_response_invalid",
            ) from None

    def unload(self) -> None:
        """Gemini has no local model state to release."""


__all__ = ["DEFAULT_GEMINI_MODEL", "GeminiFlashReasoner"]
