"""Local Qwen2.5-VL provider backed by Ollama's loopback REST API."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError

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

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b-q4_K_M"


def _ollama_image_bytes(image: QAImage) -> bytes:
    """Normalize WebP frames because Ollama's Windows vision runner rejects them."""

    if image.mime_type in {"image/png", "image/jpeg"}:
        return image.data
    if image.mime_type != "image/webp":
        raise QAProviderRequestError(
            "the local VLM image type is unsupported",
            provider="ollama-qwen",
            category="request_rejected",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as source:
            source.load()
            converted = source.convert("RGB")
            output = io.BytesIO()
            converted.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue()
    except (OSError, UnidentifiedImageError):
        raise QAProviderRequestError(
            "the local VLM image payload is invalid",
            provider="ollama-qwen",
            category="request_rejected",
        ) from None


def _loopback_base_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama URL must use HTTP on the local loopback interface")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must not contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("Ollama URL must not contain a path")
    port = parsed.port or 11434
    host = "[::1]" if parsed.hostname == "::1" else parsed.hostname
    return f"http://{host}:{port}"


def _json_mapping(response: Any) -> Mapping[str, Any]:
    try:
        value = response.json()
    except Exception:
        raise QAProviderResponseError(
            "the local VLM returned an invalid response",
            provider="ollama-qwen",
            category="response_invalid",
        ) from None
    if not isinstance(value, Mapping):
        raise QAProviderResponseError(
            "the local VLM returned an invalid response",
            provider="ollama-qwen",
            category="response_invalid",
        )
    return value


def _safe_response_shape(value: Any) -> dict[str, int | bool]:
    """Describe structured output without retaining answer or provider text."""

    if not isinstance(value, str):
        return {"response_is_json_object": False}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"response_is_json_object": False}
    if not isinstance(parsed, Mapping):
        return {"response_is_json_object": False}
    answer = parsed.get("answer")
    answerable = parsed.get("answerable")
    evidence = parsed.get("evidence_frame_indices")
    return {
        "response_is_json_object": True,
        "response_keys_valid": frozenset(parsed)
        == {"answer", "answerable", "evidence_frame_indices"},
        "response_answer_is_string": isinstance(answer, str),
        "response_answer_chars": len(answer) if isinstance(answer, str) else 0,
        "response_answer_words": len(answer.split()) if isinstance(answer, str) else 0,
        "response_answerable_is_bool": type(answerable) is bool,
        "response_answerable": answerable if type(answerable) is bool else False,
        "response_evidence_is_list": isinstance(evidence, list),
        "response_evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        "response_evidence_all_integer": isinstance(evidence, list)
        and all(type(index) is int for index in evidence),
    }


def _downgrade_empty_local_answer(
    value: Any, *, image_count: int
) -> tuple[Any, bool]:
    """Turn one known contradictory local-model state into unanswerable.

    This never invents answer text or evidence. It only removes unsupported
    positive provenance when the model supplied no answer at all.
    """

    if not isinstance(value, str):
        return value, False
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value, False
    evidence = parsed.get("evidence_frame_indices") if isinstance(parsed, Mapping) else None
    if not (
        isinstance(parsed, Mapping)
        and frozenset(parsed) == {"answer", "answerable", "evidence_frame_indices"}
        and isinstance(parsed.get("answer"), str)
        and not parsed["answer"].strip()
        and parsed.get("answerable") is True
        and isinstance(evidence, list)
        and evidence
        and all(type(index) is int and 1 <= index <= image_count for index in evidence)
    ):
        return value, False
    return json.dumps(
        {"answer": "", "answerable": False, "evidence_frame_indices": []},
        separators=(",", ":"),
    ), True


class OllamaQwenReasoner:
    """Reason over chronological images with a locally hosted Ollama model."""

    provider_name = "ollama-qwen"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        expected_digest_prefix: str | None = None,
        connect_timeout_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
        keep_alive: str | int = "15m",
        num_ctx: int = 4096,
        max_new_tokens: int = 64,
        max_images: int = 1,
        max_total_image_bytes: int = 20 * 1024 * 1024,
        http: Any | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        before_inference: Callable[[str | None], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self.base_url = _loopback_base_url(base_url)
        self.model_name = str(model_name).strip()
        if not self.model_name or len(self.model_name) > 128:
            raise ValueError("Ollama model_name must be a bounded non-empty string")
        digest = None if expected_digest_prefix is None else str(expected_digest_prefix).strip()
        if digest and (len(digest) < 8 or any(c not in "0123456789abcdefABCDEF" for c in digest)):
            raise ValueError(
                "expected_digest_prefix must contain at least 8 hexadecimal characters"
            )
        self.expected_digest_prefix = digest.casefold() if digest else None
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._timeout_seconds = float(timeout_seconds)
        self.keep_alive = keep_alive
        self.num_ctx = int(num_ctx)
        self.max_new_tokens = int(max_new_tokens)
        self.max_images = int(max_images)
        self.max_total_image_bytes = int(max_total_image_bytes)
        if min(self._connect_timeout_seconds, self._timeout_seconds) <= 0:
            raise ValueError("Ollama timeouts must be positive")
        if self.num_ctx < 256 or self.max_new_tokens < 1:
            raise ValueError("Ollama context and output limits must be positive")
        self._http = http
        self._monotonic = monotonic
        self._before_inference = before_inference
        self._lock = threading.RLock()
        self._max_attempts = 1
        self._retry_max_seconds = 0.0
        self._validated_model_digest: str | None = None
        self._last_metrics: dict[str, int | float] = {}

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.provider_name,
            {
                "model": self.model_name,
                "digest": self.expected_digest_prefix or "unrestricted",
                "num_ctx": self.num_ctx,
                "max_new_tokens": self.max_new_tokens,
                "image_payload": "single-jpeg-webp-v2",
            },
        )

    @property
    def last_metrics(self) -> dict[str, int | float]:
        """Return allowlisted timing counters from the latest Ollama response."""

        with self._lock:
            return dict(self._last_metrics)

    @property
    def max_attempts(self) -> int:
        """Expose the physical-attempt cap used by live diagnostic guards."""

        return self._max_attempts

    def _session(self) -> Any:
        if self._http is None:
            self._http = requests.Session()
        return self._http

    @staticmethod
    def _status_error(
        status_code: int,
        *,
        model_name: str,
        response: Any | None = None,
    ) -> QAProviderRequestError:
        safe_error = ""
        if response is not None:
            try:
                body = response.json()
                raw_error = body.get("error") if isinstance(body, Mapping) else None
                if isinstance(raw_error, str):
                    safe_error = raw_error.casefold()
            except Exception:
                safe_error = ""
        if status_code == 404:
            category = "model_missing"
        elif status_code in {408, 504}:
            category = "ollama_timeout"
        elif status_code in {429, 507} or any(
            marker in safe_error
            for marker in ("out of memory", "resource exhausted", "allocation failed")
        ):
            category = "resource_exhausted"
        elif status_code == 400 and "context" in safe_error and any(
            marker in safe_error for marker in ("exceed", "too long", "too large")
        ):
            category = "ollama_context_exceeded"
        elif status_code >= 500:
            category = "server_unavailable"
        else:
            category = "request_rejected"
        return QAProviderRequestError(
            "the local VLM request failed",
            provider="ollama-qwen",
            model=model_name,
            attempts=1,
            category=category,
            status_code=status_code,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._session().request(
                method,
                f"{self.base_url}{path}",
                timeout=(self._connect_timeout_seconds, self._timeout_seconds),
                **kwargs,
            )
        except requests.Timeout:
            raise QAProviderRequestError(
                "the local VLM request timed out",
                provider=self.provider_name,
                model=self.model_name,
                attempts=1,
                category="ollama_timeout",
            ) from None
        except requests.ConnectionError:
            raise QAProviderUnavailableError(
                "the local VLM is unavailable",
                provider=self.provider_name,
                model=self.model_name,
                attempts=1,
                category="network_unreachable",
            ) from None
        except requests.RequestException:
            raise QAProviderRequestError(
                "the local VLM request failed",
                provider=self.provider_name,
                model=self.model_name,
                attempts=1,
                category="provider_request_failed",
            ) from None
        status = int(getattr(response, "status_code", 0))
        if not 200 <= status < 300:
            raise self._status_error(
                status,
                model_name=self.model_name,
                response=response,
            ) from None
        return response

    def probe(self) -> dict[str, Any]:
        """Validate availability, exact model identity, and vision capability."""

        with self._lock:
            tags = _json_mapping(self._request("GET", "/api/tags"))
            models = tags.get("models")
            if not isinstance(models, list):
                raise QAProviderResponseError(
                    "the local VLM model catalog is invalid",
                    provider=self.provider_name,
                    model=self.model_name,
                    category="response_invalid",
                )
            record = next(
                (
                    item
                    for item in models
                    if isinstance(item, Mapping)
                    and str(item.get("name") or item.get("model") or "") == self.model_name
                ),
                None,
            )
            if record is None:
                raise QAProviderUnavailableError(
                    "the configured local VLM model is not installed",
                    provider=self.provider_name,
                    model=self.model_name,
                    category="model_missing",
                )
            digest = str(record.get("digest") or "").casefold()
            if self.expected_digest_prefix and not digest.startswith(self.expected_digest_prefix):
                raise QAProviderUnavailableError(
                    "the installed local VLM model fingerprint is incompatible",
                    provider=self.provider_name,
                    model=self.model_name,
                    category="model_unavailable",
                )
            details = _json_mapping(
                self._request("POST", "/api/show", json={"model": self.model_name})
            )
            capabilities = details.get("capabilities", ())
            if not isinstance(capabilities, list) or "vision" not in capabilities:
                raise QAProviderUnavailableError(
                    "the configured local model has no vision capability",
                    provider=self.provider_name,
                    model=self.model_name,
                    category="model_unavailable",
                )
            self._validated_model_digest = digest or None
            return {
                "provider": self.provider_name,
                "model": self.model_name,
                "digest": digest or None,
                "vision": True,
            }

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        values = _validate_images(
            images,
            max_images=self.max_images,
            max_total_image_bytes=self.max_total_image_bytes,
        )
        if len(values) != 1:
            raise ValueError("Ollama Qwen Visual Q&A requires exactly one image")
        with self._lock:
            handoff_metrics: dict[str, int | float | bool] = {}
            if self._before_inference is not None:
                batch_id = request.batch_filters[0] if request.batch_filters else None
                returned = self._before_inference(batch_id)
                if isinstance(returned, Mapping):
                    handoff_metrics = {
                        str(name): value
                        for name, value in returned.items()
                        if isinstance(value, (int, float, bool))
                    }
            started = self._monotonic()
            payload = {
                "model": self.model_name,
                "stream": False,
                "keep_alive": self.keep_alive,
                "format": _response_schema(len(values)),
                "options": {
                    "temperature": 0,
                    "seed": 0,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.max_new_tokens,
                },
                "messages": [
                    {"role": "system", "content": _system_instruction()},
                    {
                        "role": "user",
                        "content": _question_text(request, include_json_instruction=True),
                        "images": [
                            base64.b64encode(_ollama_image_bytes(image)).decode("ascii")
                            for image in values
                        ],
                    },
                ],
            }
            response = self._request("POST", "/api/chat", json=payload)
            body = _json_mapping(response)
            wall_latency_ms = (self._monotonic() - started) * 1_000
            metrics: dict[str, int | float] = {}
            for name in (
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            ):
                value = body.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                    metrics[f"{name}_ms"] = round(float(value) / 1_000_000, 3)
            for name in ("prompt_eval_count", "eval_count"):
                value = body.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    metrics[name] = value
            metrics["wall_latency_ms"] = round(max(0.0, wall_latency_ms), 3)
            metrics.update(handoff_metrics)
            self._last_metrics = metrics
            message = body.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            metrics.update(_safe_response_shape(content))
            content, downgraded = _downgrade_empty_local_answer(
                content, image_count=len(values)
            )
            metrics["response_downgraded_unanswerable"] = downgraded
            self._last_metrics = metrics
            try:
                return _parse_reasoning_output(
                    content,
                    request=request,
                    spec=spec,
                    image_count=len(values),
                    provider=self.provider_name,
                    model_name=self.model_name,
                    latency_ms=wall_latency_ms,
                    attempts=1,
                )
            except QAProviderResponseError:
                raise QAProviderResponseError(
                    "the local VLM returned an invalid structured response",
                    provider=self.provider_name,
                    model=self.model_name,
                    attempts=1,
                    category="ollama_response_invalid",
                ) from None

    def runtime_status(self) -> dict[str, int | float | str | None]:
        """Return sanitized resident-model memory and approximate offload data."""

        try:
            with self._lock:
                body = _json_mapping(self._request("GET", "/api/ps"))
                models = body.get("models")
                record = (
                    next(
                        (
                            item
                            for item in models
                            if isinstance(item, Mapping)
                            and str(item.get("name") or item.get("model") or "") == self.model_name
                        ),
                        None,
                    )
                    if isinstance(models, list)
                    else None
                )
                if record is None:
                    return {"state": "not_resident", "model": self.model_name}
                size = record.get("size")
                size_vram = record.get("size_vram")
                size_bytes = (
                    int(size)
                    if isinstance(size, int) and not isinstance(size, bool) and size >= 0
                    else 0
                )
                vram_bytes = (
                    int(size_vram)
                    if isinstance(size_vram, int)
                    and not isinstance(size_vram, bool)
                    and size_vram >= 0
                    else 0
                )
                gpu_percent = (
                    round(100 * min(vram_bytes, size_bytes) / size_bytes, 2) if size_bytes else None
                )
                return {
                    "state": "resident",
                    "model": self.model_name,
                    "size_bytes": size_bytes,
                    "size_vram_bytes": vram_bytes,
                    "gpu_offload_percent": gpu_percent,
                    "cpu_offload_percent": round(100 - gpu_percent, 2)
                    if gpu_percent is not None
                    else None,
                }
        except (QAProviderRequestError, QAProviderResponseError, QAProviderUnavailableError):
            return {"state": "unavailable", "model": self.model_name}

    def unload(self) -> None:
        """Ask Ollama to release the local model without surfacing shutdown errors."""

        with self._lock:
            if self._http is None:
                return
            try:
                self._request(
                    "POST",
                    "/api/generate",
                    json={"model": self.model_name, "keep_alive": 0},
                )
            except Exception:
                return


__all__ = ["DEFAULT_OLLAMA_MODEL", "DEFAULT_OLLAMA_URL", "OllamaQwenReasoner"]
