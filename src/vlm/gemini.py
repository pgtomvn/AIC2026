"""Single-frame Gemini provider for Visual Q&A (REST generateContent, model fallback)."""

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


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
HttpPost = Callable[[str, Mapping[str, str], Mapping[str, Any], float], tuple[int, Any]]


def _retry_seconds(body: Any, default: float = 60.0) -> float:
    """Seconds until a quota resets, from the API's RetryInfo detail."""
    try:
        for detail in body["error"]["details"]:
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                return min(max(float(str(detail["retryDelay"]).rstrip("s")), 1.0), 86_400.0)
    except (KeyError, TypeError, ValueError):
        pass
    return default


def _candidate_text(body: Any) -> str:
    try:
        parts = body["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    return "".join(p.get("text", "") for p in parts if isinstance(p, Mapping) and not p.get("thought"))


class GeminiFlashReasoner:
    """Reason over exactly one canonical frame with Gemini ``generateContent``.

    Models are tried in order. A model whose daily or per-minute quota is used up (HTTP 429)
    is skipped until the reset time the API reports, so free-tier quotas of several models add
    up. Errors return immediately; every call also has a hard deadline.
    """

    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        fallback_models: Sequence[str] = (),
        timeout_seconds: float = 5.0,
        max_total_image_bytes: int = 20 * 1024 * 1024,
        http_post: HttpPost | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        ip_family: str = "auto",
        thinking_level: str | None = "minimal",
    ) -> None:
        self._api_key = str(api_key or "").strip()
        models = [str(m).strip() for m in (model_name, *fallback_models) if str(m).strip()]
        self.models = tuple(dict.fromkeys(models))
        if not self.models or any(len(m) > 128 or "/" in m for m in self.models):
            raise ValueError("Gemini model names must be bounded non-empty strings")
        self.model_name = self.models[0]
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("Gemini timeout must be positive")
        if ip_family not in {"auto", "ipv4"}:
            raise ValueError("Gemini ip_family must be auto or ipv4")
        # Some networks resolve Google to IPv6 but cannot route it; "ipv4" binds the HTTP
        # client to IPv4 so a connection never waits on unreachable IPv6 addresses.
        self.ip_family = ip_family
        self.thinking_level = thinking_level or None
        self.max_total_image_bytes = int(max_total_image_bytes)
        self._http_post = http_post
        self._http: Any | None = None
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._exhausted_until: dict[str, float] = {}
        self._no_thinking: set[str] = set()

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.provider_name,
            {"models": list(self.models), "api": "generate-content-v1beta", "images": 1,
             "thinking_level": self.thinking_level or "default", "sdk_attempts": 1},
        )

    @property
    def max_attempts(self) -> int:
        return 1

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            transport = httpx.HTTPTransport(
                retries=0, **({"local_address": "0.0.0.0"} if self.ip_family == "ipv4" else {}))
            self._http = httpx.Client(transport=transport,
                                      timeout=httpx.Timeout(self.timeout_seconds, connect=min(5.0, self.timeout_seconds)))
        return self._http

    def _post(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float) -> tuple[int, Any]:
        if self._http_post is not None:
            return self._http_post(url, headers, body, timeout)
        response = self._client().post(url, headers=dict(headers), json=dict(body))
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, None

    def _call_with_deadline(self, call: Callable[[], Any]) -> Any:
        """Run one HTTP call with a hard deadline; a stalled call drops the connection pool."""

        outcome: dict[str, Any] = {}

        def run() -> None:
            try:
                outcome["value"] = call()
            except BaseException as error:  # noqa: BLE001 - re-raised in the caller
                outcome["error"] = error

        worker = threading.Thread(target=run, name="gemini-call", daemon=True)
        worker.start()
        worker.join(self.timeout_seconds + 2.0)
        if worker.is_alive():
            http, self._http = self._http, None
            if http is not None:
                try:
                    http.close()
                except Exception:  # noqa: BLE001 - the stalled call owns the connection
                    pass
            raise QAProviderRequestError("Gemini did not answer in time", provider=self.provider_name,
                                         model=self.model_name, attempts=1, category="gemini_timeout")
        error = outcome.get("error")
        if isinstance(error, (QAProviderRequestError, QAProviderUnavailableError)):
            raise error
        if error is not None:
            raise _provider_error(error, model=self.model_name) from None
        return outcome.get("value")

    def available_models(self) -> tuple[str, ...]:
        now = self._monotonic()
        return tuple(m for m in self.models if self._exhausted_until.get(m, 0.0) <= now)

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        if not request.allow_external_frame_transfer:
            raise ValueError("Gemini requires explicit external frame transfer consent")
        values = _validate_images(images, max_images=1, max_total_image_bytes=self.max_total_image_bytes)
        if len(values) != 1:
            raise ValueError("Gemini single-frame QA requires exactly one image")
        if not self._api_key:
            raise QAProviderUnavailableError("Gemini is not configured", provider=self.provider_name,
                                             model=self.model_name, attempts=0, category="not_configured")
        image = values[0]
        contents = [{"role": "user", "parts": [
            {"text": _question_text(request)},
            {"inline_data": {"mime_type": image.mime_type, "data": base64.b64encode(image.data).decode("ascii")}},
        ]}]
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        last: QAProviderRequestError | None = None
        with self._lock:
            for model in self.available_models():
                started = self._monotonic()
                while True:
                    config: dict[str, Any] = {"responseMimeType": "application/json",
                                              "responseJsonSchema": _response_schema(1), "maxOutputTokens": 64}
                    thinking = self.thinking_level is not None and model not in self._no_thinking
                    if thinking:
                        config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
                    body = {"systemInstruction": {"parts": [{"text": _system_instruction()}]},
                            "contents": contents, "generationConfig": config}
                    try:
                        status, data = self._call_with_deadline(
                            lambda m=model, b=body: self._post(GEMINI_ENDPOINT.format(model=m), headers, b, self.timeout_seconds))
                    except QAProviderRequestError as error:
                        status, data, last = None, None, error
                        break
                    message = str((data or {}).get("error", {}).get("message", "")).casefold() if isinstance(data, Mapping) else ""
                    if status == 400 and thinking and "think" in message:
                        self._no_thinking.add(model)  # this model rejects a thinking level
                        continue
                    break
                if status == 200:
                    latency_ms = (self._monotonic() - started) * 1_000
                    try:
                        return _parse_reasoning_output(_candidate_text(data), request=request, spec=spec, image_count=1,
                                                       provider=self.provider_name, model_name=model,
                                                       latency_ms=latency_ms, attempts=1)
                    except QAProviderResponseError:
                        last = QAProviderRequestError("Gemini returned an invalid structured response",
                                                      provider=self.provider_name, model=model, attempts=1,
                                                      category="gemini_response_invalid")
                        continue
                if status is None:
                    continue  # timed out: the next model may still answer
                if status == 429:
                    self._exhausted_until[model] = self._monotonic() + _retry_seconds(data)
                    last = QAProviderRequestError("Gemini quota exhausted", provider=self.provider_name, model=model,
                                                  attempts=1, category="rate_limited", status_code=429)
                    continue
                error = QAProviderRequestError(
                    "the Gemini request failed", provider=self.provider_name, model=model, attempts=1,
                    category=("auth_or_permission" if status in {401, 403} else "model_missing" if status == 404
                              else "invalid_request" if status == 400 else "server_unavailable" if status >= 500
                              else "provider_request_failed"),
                    status_code=status)
                if status >= 500 or status == 404:
                    last = error
                    continue
                raise error
        if last is not None:
            raise last
        raise QAProviderRequestError("every Gemini model has used its quota", provider=self.provider_name,
                                     model=self.model_name, attempts=0, category="rate_limited", status_code=429)

    def unload(self) -> None:
        """Gemini has no local model state to release."""


__all__ = ["DEFAULT_GEMINI_MODEL", "GeminiFlashReasoner"]
