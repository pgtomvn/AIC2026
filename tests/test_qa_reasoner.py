from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from src.contracts import QACompetitionSpec, QAQueryRequest, QASingleFrameAnswerRequest
from src.vlm.gemini import GeminiFlashReasoner
from src.vlm.local_qwen import OllamaQwenReasoner
from src.vlm.qa_reasoner import (
    QAImage,
    QAProviderRequestError,
    QAProviderResponseError,
    QAProviderUnavailableError,
)
from src.vlm.qa_router import QAReasonerRouter


class _Response:
    def __init__(self, value, status_code: int = 200) -> None:
        self.value = value
        self.status_code = status_code

    def json(self):
        return self.value


class _OllamaHTTP:
    def __init__(self, chat=None) -> None:
        self.calls = []
        self.chat = chat or {
            "message": {
                "content": json.dumps(
                    {"answer": "3", "answerable": True, "evidence_frame_indices": [1]}
                )
            }
        }

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/api/tags"):
            return _Response(
                {
                    "models": [
                        {
                            "name": "qwen2.5vl:3b-q4_K_M",
                            "digest": "fb90415cde1e1234",
                        }
                    ]
                }
            )
        if url.endswith("/api/show"):
            return _Response({"capabilities": ["vision", "completion"]})
        if url.endswith("/api/generate"):
            return _Response({"done": True})
        if url.endswith("/api/ps"):
            return _Response(
                {
                    "models": [
                        {
                            "name": "qwen2.5vl:3b-q4_K_M",
                            "size": 100,
                            "size_vram": 75,
                        }
                    ]
                }
            )
        return _Response(self.chat)


class _OpenRouterHTTP:
    def __init__(self, *, status_code: int = 200) -> None:
        self.calls = []
        self.status_code = status_code

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "đỏ",
                                    "answerable": True,
                                    "evidence_frame_indices": [1],
                                }
                            )
                        }
                    }
                ]
            },
            self.status_code,
        )


def _image(index: int = 1) -> QAImage:
    output = io.BytesIO()
    Image.new("RGB", (24, 16), "red").save(output, format="PNG")
    data = output.getvalue()
    return QAImage(
        data=data,
        mime_type="image/png",
        index=index,
        source_frame_idx=index * 10,
        timestamp_ms=index * 1_000,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _webp_image(index: int = 1) -> QAImage:
    output = io.BytesIO()
    Image.new("RGB", (24, 16), "red").save(output, format="WEBP")
    data = output.getvalue()
    return QAImage(
        data=data,
        mime_type="image/webp",
        index=index,
        source_frame_idx=index * 10,
        timestamp_ms=index * 1_000,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _request(**updates) -> QAQueryRequest:
    value = QAQueryRequest(
        query_id="qa-test",
        event_description="ba hình tròn màu đỏ",
        question="Có bao nhiêu hình tròn?",
        answer_spec_id=QACompetitionSpec.conservative_general_text().spec_id,
        batch_filters=("aic25-b1-reference",),
        limit=5,
        reasoning_policy="local_only",
    )
    return replace(value, **updates)


def test_ollama_payload_uses_raw_base64_schema_and_deterministic_options() -> None:
    http = _OllamaHTTP()
    reasoner = OllamaQwenReasoner(http=http, expected_digest_prefix="fb90415cde1e")
    assert reasoner.probe()["vision"] is True
    result = reasoner.reason(_request(), (_image(),), QACompetitionSpec.conservative_general_text())
    assert result.answer == "3"
    method, url, kwargs = http.calls[-1]
    assert (method, url.rsplit("/", 1)[-1]) == ("POST", "chat")
    payload = kwargs["json"]
    assert payload["stream"] is False
    assert payload["options"] == {
        "temperature": 0,
        "seed": 0,
        "num_ctx": 4096,
        "num_predict": 64,
    }
    assert payload["keep_alive"] == "15m"
    assert kwargs["timeout"] == (2.0, 60.0)
    encoded = payload["messages"][1]["images"][0]
    assert not encoded.startswith("data:")
    assert base64.b64decode(encoded) == _image().data
    assert payload["format"]["additionalProperties"] is False


def test_ollama_normalizes_webp_frames_to_supported_jpeg_payloads() -> None:
    http = _OllamaHTTP()
    reasoner = OllamaQwenReasoner(http=http)

    reasoner.reason(_request(), (_webp_image(),), QACompetitionSpec.conservative_general_text())

    encoded = http.calls[-1][2]["json"]["messages"][1]["images"][0]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as normalized:
        assert normalized.format == "JPEG"
        assert normalized.size == (24, 16)


def test_ollama_reports_allowlisted_response_timings_and_residency() -> None:
    http = _OllamaHTTP(
        chat={
            "message": {
                "content": json.dumps(
                    {"answer": "3", "answerable": True, "evidence_frame_indices": [1]}
                )
            },
            "load_duration": 2_000_000,
            "prompt_eval_duration": 3_000_000,
            "eval_duration": 4_000_000,
            "total_duration": 9_000_000,
            "prompt_eval_count": 12,
            "eval_count": 3,
        }
    )
    reasoner = OllamaQwenReasoner(http=http)
    reasoner.reason(_request(), (_image(),), QACompetitionSpec.conservative_general_text())

    metrics = reasoner.last_metrics
    assert {
        name: metrics[name]
        for name in (
            "total_duration_ms",
            "load_duration_ms",
            "prompt_eval_duration_ms",
            "eval_duration_ms",
            "prompt_eval_count",
            "eval_count",
        )
    } == {
        "total_duration_ms": 9.0,
        "load_duration_ms": 2.0,
        "prompt_eval_duration_ms": 3.0,
        "eval_duration_ms": 4.0,
        "prompt_eval_count": 12,
        "eval_count": 3,
    }
    assert metrics["wall_latency_ms"] >= 0.0
    assert metrics["response_keys_valid"] is True
    assert reasoner.last_metrics["wall_latency_ms"] >= 0
    assert reasoner.runtime_status()["gpu_offload_percent"] == 75.0


@pytest.mark.parametrize("url", ["https://localhost:11434", "http://example.com:11434"])
def test_ollama_rejects_non_loopback_or_non_http_endpoint(url: str) -> None:
    with pytest.raises(ValueError):
        OllamaQwenReasoner(base_url=url)


def test_ollama_missing_model_is_sanitized() -> None:
    http = _OllamaHTTP()
    http.request = lambda *_args, **_kwargs: _Response({"models": []})
    with pytest.raises(QAProviderUnavailableError) as captured:
        OllamaQwenReasoner(http=http).probe()
    assert captured.value.category == "model_missing"
    assert "models" not in str(captured.value)


def test_ollama_timeout_is_one_attempt_and_not_rate_limited() -> None:
    class Timeout:
        def request(self, *_args, **_kwargs):
            raise requests.Timeout("secret body")

    with pytest.raises(QAProviderRequestError) as captured:
        OllamaQwenReasoner(http=Timeout()).reason(
            _request(), (_image(),), QACompetitionSpec.conservative_general_text()
        )
    assert captured.value.attempts == 1
    assert captured.value.category == "ollama_timeout"
    assert "secret" not in str(captured.value)


def test_ollama_invalid_json_is_response_error() -> None:
    http = _OllamaHTTP(chat={"message": {"content": "not-json"}})
    with pytest.raises(QAProviderResponseError) as captured:
        OllamaQwenReasoner(http=http).reason(
            _request(), (_image(),), QACompetitionSpec.conservative_general_text()
        )
    assert captured.value.category == "ollama_response_invalid"


def test_ollama_downgrades_empty_answer_with_false_positive_evidence() -> None:
    http = _OllamaHTTP(
        chat={
            "message": {
                "content": json.dumps(
                    {
                        "answer": "",
                        "answerable": True,
                        "evidence_frame_indices": [1],
                    }
                )
            }
        }
    )
    reasoner = OllamaQwenReasoner(http=http)

    result = reasoner.reason(
        _request(), (_image(),), QACompetitionSpec.conservative_general_text()
    )

    assert result.answerable is False
    assert result.answer == ""
    assert result.evidence_frame_indices == ()
    assert reasoner.last_metrics["response_downgraded_unanswerable"] is True


def test_ollama_context_exceeded_is_classified_without_body_leak() -> None:
    class ContextExceeded(_OllamaHTTP):
        def request(self, method, url, **kwargs):
            if url.endswith("/api/chat"):
                return _Response(
                    {"error": "input context exceeds configured limit; private detail"},
                    400,
                )
            return super().request(method, url, **kwargs)

    with pytest.raises(QAProviderRequestError) as captured:
        OllamaQwenReasoner(http=ContextExceeded()).reason(
            _request(), (_image(),), QACompetitionSpec.conservative_general_text()
        )
    assert captured.value.category == "ollama_context_exceeded"
    assert "private" not in str(captured.value)


def test_ollama_accepts_exactly_one_image_and_rejects_multiple() -> None:
    http = _OllamaHTTP()
    reasoner = OllamaQwenReasoner(http=http)
    image = _image(1)

    reasoner.reason(_request(), (image,), QACompetitionSpec.conservative_general_text())

    payload = http.calls[-1][2]["json"]
    assert len(payload["messages"][1]["images"]) == 1
    with pytest.raises(ValueError):
        reasoner.reason(
            _request(), (image, _image(2)), QACompetitionSpec.conservative_general_text()
        )


def test_ollama_stably_deduplicates_repeated_valid_evidence_indices() -> None:
    http = _OllamaHTTP(
        chat={
            "message": {
                "content": json.dumps(
                    {
                        "answer": "3",
                        "answerable": True,
                        "evidence_frame_indices": [1, 1, 1],
                    }
                )
            }
        }
    )

    result = OllamaQwenReasoner(http=http).reason(
        _request(), (_image(),), QACompetitionSpec.conservative_general_text()
    )

    assert result.answer == "3"
    assert result.evidence_frame_indices == (1,)


@pytest.mark.parametrize(("raw_answer", "expected"), (("true", "yes"), ("false", "no")))
def test_ollama_normalizes_boolean_literals_for_yes_no_questions(
    raw_answer: str,
    expected: str,
) -> None:
    http = _OllamaHTTP(
        chat={
            "message": {
                "content": json.dumps(
                    {
                        "answer": raw_answer,
                        "answerable": True,
                        "evidence_frame_indices": [1],
                    }
                )
            }
        }
    )

    result = OllamaQwenReasoner(http=http).reason(
        _request(question="Does the man wear glasses?"),
        (_image(),),
        QACompetitionSpec.conservative_general_text(),
    )

    assert result.answer == expected


def test_ollama_serializes_concurrent_inference_and_unloads_explicitly() -> None:
    class ConcurrentHTTP(_OllamaHTTP):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def request(self, method, url, **kwargs):
            if url.endswith("/api/chat"):
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                try:
                    return super().request(method, url, **kwargs)
                finally:
                    with self.guard:
                        self.active -= 1
            return super().request(method, url, **kwargs)

    http = ConcurrentHTTP()
    reasoner = OllamaQwenReasoner(http=http)
    spec = QACompetitionSpec.conservative_general_text()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _index: reasoner.reason(_request(), (_image(),), spec), range(4))
        )
    reasoner.unload()

    assert [result.answer for result in results] == ["3"] * 4
    assert http.max_active == 1
    method, url, kwargs = http.calls[-1]
    assert (method, url.rsplit("/", 1)[-1]) == ("POST", "generate")
    assert kwargs["json"] == {"model": "qwen2.5vl:3b-q4_K_M", "keep_alive": 0}


def test_ollama_resource_exhaustion_is_sanitized() -> None:
    class Exhausted(_OllamaHTTP):
        def request(self, method, url, **kwargs):
            if url.endswith("/api/chat"):
                return _Response({"error": "private allocation details"}, 507)
            return super().request(method, url, **kwargs)

    with pytest.raises(QAProviderRequestError) as captured:
        OllamaQwenReasoner(http=Exhausted()).reason(
            _request(), (_image(),), QACompetitionSpec.conservative_general_text()
        )
    assert captured.value.category == "resource_exhausted"
    assert "private" not in str(captured.value)


def test_gemini_interactions_payload_is_single_image_structured_and_bounded() -> None:
    calls: list[dict] = []
    client_options: list[dict] = []

    class Interactions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {"answer": "3", "answerable": True, "evidence_frame_indices": [1]}
                )
            )

    def factory(**kwargs):
        client_options.append(kwargs)
        return SimpleNamespace(interactions=Interactions())

    provider = GeminiFlashReasoner(api_key="secret", client_factory=factory)
    request = _request(
        reasoning_policy="gemini_then_local", allow_external_frame_transfer=True
    )
    result = provider.reason(request, (_image(),), QACompetitionSpec.conservative_general_text())

    assert result.provider == "gemini"
    assert client_options == [
        {"api_key": "secret", "http_options": {"timeout": 5000, "retry_options": {"attempts": 1}}}
    ]
    assert calls[0]["input"][0]["type"] == "text"
    assert calls[0]["input"][1]["type"] == "image"
    assert base64.b64decode(calls[0]["input"][1]["data"]) == _image().data
    assert calls[0]["generation_config"]["thinking_level"] == "minimal"
    assert calls[0]["store"] is False
    with pytest.raises(ValueError):
        provider.reason(request, (_image(1), _image(2)), QACompetitionSpec.conservative_general_text())


def test_router_local_only_never_calls_gemini() -> None:
    class ForbiddenGemini:
        provider_name = "gemini"
        model_name = "gemini-test"
        fingerprint = "gemini"

        def reason(self, *_args, **_kwargs):
            raise AssertionError("Gemini must not be called")

        def unload(self):
            return None

    local_http = _OllamaHTTP()
    router = QAReasonerRouter(ForbiddenGemini(), OllamaQwenReasoner(http=local_http))
    result = router.reason(_request(), (_image(),), QACompetitionSpec.conservative_general_text())
    assert result.provider == "ollama-qwen"
    assert result.fallback_used is False


def test_router_falls_back_from_missing_gemini_key_once() -> None:
    local_http = _OllamaHTTP()
    router = QAReasonerRouter(
        GeminiFlashReasoner(api_key=None), OllamaQwenReasoner(http=local_http)
    )
    request = _request(
        reasoning_policy="gemini_then_local", allow_external_frame_transfer=True
    )
    result = router.reason(request, (_image(),), QACompetitionSpec.conservative_general_text())
    assert result.provider == "ollama-qwen"
    assert result.fallback_used is True
    assert result.fallback_reason_code == "not_configured"
    assert result.attempts == 2


def test_router_does_not_fallback_after_permanent_gemini_error() -> None:
    class RejectedGemini:
        provider_name = "gemini"
        model_name = "gemini-test"
        fingerprint = "gemini"

        def reason(self, *_args, **_kwargs):
            raise QAProviderRequestError(
                "safe", provider=self.provider_name, category="invalid_request", status_code=400
            )

        def unload(self):
            return None

    class ForbiddenLocal:
        provider_name = "ollama-qwen"
        model_name = "local-test"
        fingerprint = "local"

        def reason(self, *_args, **_kwargs):
            raise AssertionError("local fallback must not run")

        def unload(self):
            return None

    router = QAReasonerRouter(RejectedGemini(), ForbiddenLocal())
    request = _request(
        reasoning_policy="gemini_then_local", allow_external_frame_transfer=True
    )
    with pytest.raises(QAProviderRequestError):
        router.reason(request, (_image(),), QACompetitionSpec.conservative_general_text())


def test_gemini_policy_without_consent_is_rejected_by_contract() -> None:
    with pytest.raises(ValueError):
        QASingleFrameAnswerRequest(candidate_id="b1:V001:10", question="question")
