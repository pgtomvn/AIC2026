"""Gemini REST reasoner: deadlines, IPv4 transport, and quota-aware model fallback."""

from __future__ import annotations

import base64
import json
import threading
import time

import pytest

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.vlm.gemini import GeminiFlashReasoner
from src.vlm.qa_reasoner import QAImage, QAProviderRequestError

ANSWER = {"candidates": [{"content": {"parts": [{"text": json.dumps(
    {"answer": "3", "answerable": True, "evidence_frame_indices": [1]})}]}}]}
QUOTA = {"error": {"code": 429, "message": "quota", "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "120s"}]}}


def _request() -> QAQueryRequest:
    return QAQueryRequest(
        query_id="qa-test", event_description="scene", question="How many?",
        answer_spec_id=QACompetitionSpec.conservative_general_text().spec_id,
        batch_filters=("aic25-b1-reference",), limit=5,
        reasoning_policy="gemini_then_local", allow_external_frame_transfer=True)


def _image() -> QAImage:
    import hashlib
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    data = buffer.getvalue()
    return QAImage(data=data, mime_type="image/png", index=1, source_frame_idx=10, timestamp_ms=1000,
                   sha256=hashlib.sha256(data).hexdigest())


def _reason(reasoner):
    return reasoner.reason(_request(), (_image(),), QACompetitionSpec.conservative_general_text())


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_payload_is_one_structured_image_request():
    calls = []

    def post(url, headers, body, timeout):
        calls.append((url, headers, body))
        return 200, ANSWER

    result = _reason(GeminiFlashReasoner(api_key="secret", model_name="m1", http_post=post))
    url, headers, body = calls[0]
    assert url.endswith("/models/m1:generateContent") and headers["x-goog-api-key"] == "secret"
    parts = body["contents"][0]["parts"]
    assert parts[0]["text"] and base64.b64decode(parts[1]["inline_data"]["data"]) == _image().data
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json" and config["maxOutputTokens"] == 64
    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert result.answer == "3" and result.model_name == "m1" and result.provider == "gemini"


def test_quota_exhaustion_falls_back_and_skips_until_reset():
    clock, calls = Clock(), []

    def post(url, headers, body, timeout):
        calls.append(url.split("/models/")[1].split(":")[0])
        return (429, QUOTA) if "m1" in url else (200, ANSWER)

    reasoner = GeminiFlashReasoner(api_key="k", model_name="m1", fallback_models=["m2"], http_post=post, monotonic=clock)
    assert _reason(reasoner).model_name == "m2"
    assert _reason(reasoner).model_name == "m2"
    assert calls == ["m1", "m2", "m2"]          # m1 is not retried before its reset
    clock.now += 121
    _reason(reasoner)
    assert calls[-2:] == ["m1", "m2"]


def test_all_models_exhausted_is_rate_limited_without_hanging():
    reasoner = GeminiFlashReasoner(api_key="k", model_name="m1", fallback_models=["m2"],
                                   http_post=lambda *a: (429, QUOTA))
    with pytest.raises(QAProviderRequestError) as error:
        _reason(reasoner)
    assert error.value.category == "rate_limited"
    with pytest.raises(QAProviderRequestError) as again:
        _reason(reasoner)
    assert again.value.category == "rate_limited"


def test_model_rejecting_a_thinking_level_is_retried_without_it():
    bodies = []

    def post(url, headers, body, timeout):
        bodies.append(body)
        if "thinkingConfig" in body["generationConfig"]:
            return 400, {"error": {"message": "Thinking level is not supported for this model."}}
        return 200, ANSWER

    reasoner = GeminiFlashReasoner(api_key="k", model_name="lite", http_post=post)
    assert _reason(reasoner).answer == "3"
    _reason(reasoner)
    assert [("thinkingConfig" in b["generationConfig"]) for b in bodies] == [True, False, False]


def test_auth_errors_are_not_hidden_by_fallback():
    reasoner = GeminiFlashReasoner(api_key="k", model_name="m1", fallback_models=["m2"],
                                   http_post=lambda *a: (403, {"error": {"message": "denied"}}))
    with pytest.raises(QAProviderRequestError) as error:
        _reason(reasoner)
    assert error.value.category == "auth_or_permission"


def test_stalled_call_times_out_and_drops_the_connection_pool():
    reasoner = GeminiFlashReasoner(api_key="k", timeout_seconds=0.1)

    class Pool:
        closed = False

        def close(self):
            Pool.closed = True

    reasoner._http = Pool()
    release = threading.Event()
    started = time.monotonic()
    with pytest.raises(QAProviderRequestError) as error:
        reasoner._call_with_deadline(lambda: release.wait(30))
    release.set()
    assert error.value.category == "gemini_timeout"
    assert time.monotonic() - started < 5
    assert reasoner._http is None and Pool.closed


def test_ipv4_family_binds_the_http_client_to_ipv4():
    client = GeminiFlashReasoner(api_key="k", ip_family="ipv4")._client()
    assert client._transport._pool._local_address == "0.0.0.0"
    assert GeminiFlashReasoner(api_key="k")._client()._transport._pool._local_address is None
    with pytest.raises(ValueError):
        GeminiFlashReasoner(api_key="k", ip_family="ipv6")
