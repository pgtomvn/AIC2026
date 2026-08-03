"""CPU-only, offline tests for Visual Q&A provider boundaries."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.vlm.qa_reasoner import (
    GeminiInteractionsReasoner,
    GeminiRequestGovernor,
    ProviderChainReasoner,
    QAImage,
    QAProviderRequestError,
    QAProviderResponseError,
    QAProviderUnavailableError,
    QAReasonerError,
    Qwen3VLReasoner,
    _reset_gemini_shared_controls_for_tests,
    retry_delay_for_exception,
)
from src.vlm.vlm_engine import VLMEngine


@pytest.fixture(autouse=True)
def _isolate_gemini_process_controls():
    _reset_gemini_shared_controls_for_tests()
    yield
    _reset_gemini_shared_controls_for_tests()


def _image_bytes(*, size: tuple[int, int] = (12, 8)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "red").save(output, "PNG")
    return output.getvalue()


def _image(index: int = 1, *, size: tuple[int, int] = (12, 8)) -> QAImage:
    data = _image_bytes(size=size)
    return QAImage(
        data=data,
        mime_type="image/png",
        index=index,
        source_frame_idx=(index - 1) * 10,
        timestamp_ms=(index - 1) * 1_000,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _request(*, question: str = "How many people are visible?") -> QAQueryRequest:
    return QAQueryRequest(
        query_id="qa-1",
        event_description="Two people enter a room.",
        question=question,
        limit=2,
    )


def _response(answer: str = "two", *, evidence: list[int] | None = None) -> str:
    return json.dumps(
        {
            "answer": answer,
            "answerable": True,
            "evidence_frame_indices": evidence or [1],
        }
    )


class _Interactions:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(output_text=outcome)


class _Client:
    def __init__(self, outcomes) -> None:
        self.interactions = _Interactions(outcomes)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ProviderHTTPError(RuntimeError):
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__("provider detail must remain private")
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


def _gemini(client: _Client, **kwargs) -> GeminiInteractionsReasoner:
    # Individual unit tests opt into pacing explicitly; production defaults are
    # covered separately without adding wall-clock delays to the offline suite.
    kwargs.setdefault("min_request_interval_seconds", 0)
    return GeminiInteractionsReasoner(
        api_key="dummy-test-key",
        client_factory=lambda **_settings: client,
        retry_base_seconds=0,
        retry_max_seconds=0,
        **kwargs,
    )


def test_qa_image_checks_content_hash_and_chronological_window() -> None:
    image = _image()
    assert image.index == 1

    with pytest.raises(ValueError, match="does not match"):
        QAImage(
            data=image.data,
            mime_type=image.mime_type,
            index=1,
            source_frame_idx=0,
            timestamp_ms=0,
            sha256="0" * 64,
        )

    reasoner = _gemini(_Client([_response()]))
    with pytest.raises(ValueError, match="contiguous"):
        reasoner.reason(_request(), (_image(2),), QACompetitionSpec.historical_numeric())
    assert not reasoner.client_initialized


def test_gemini_interactions_sends_inline_ordered_images_and_strict_schema() -> None:
    client = _Client([_response("two", evidence=[1, 2])])
    factory_calls: list[dict] = []
    reasoner = GeminiInteractionsReasoner(
        api_key="dummy-test-key",
        client_factory=lambda **settings: factory_calls.append(settings) or client,
    )
    assert not reasoner.client_initialized
    assert reasoner.max_attempts == 3
    reasoner.initialize()
    assert reasoner.client_initialized
    assert client.interactions.calls == []

    result = reasoner.reason(
        _request(),
        (_image(1), _image(2)),
        QACompetitionSpec.historical_numeric(),
    )

    assert result.answer == "2"
    assert result.raw_answer == "two"
    assert result.evidence_frame_indices == (1, 2)
    assert result.provider == "gemini"
    assert result.attempts == 1
    assert factory_calls[0]["timeout_seconds"] == 30.0
    call = client.interactions.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert call["store"] is False
    assert call["response_format"][0]["mime_type"] == "application/json"
    schema = call["response_format"][0]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["evidence_frame_indices"]["items"]["maximum"] == 2
    assert [part["type"] for part in call["input"]] == [
        "text",
        "image",
        "text",
        "image",
        "text",
    ]
    assert "dummy-test-key" not in repr(reasoner)

    reasoner.unload()
    assert client.closed


class _StatusError(RuntimeError):
    def __init__(self, status_code: int, detail: str = "provider failed") -> None:
        super().__init__(detail)
        self.status_code = status_code


@pytest.mark.parametrize("status_code", [408, 429, 500, 503, 599])
def test_gemini_retries_only_transient_statuses(status_code: int) -> None:
    client = _Client([_StatusError(status_code), _response()])
    reasoner = _gemini(client)

    result = reasoner.reason(
        _request(),
        (_image(),),
        QACompetitionSpec.historical_numeric(),
    )

    assert result.attempts == 2
    assert len(client.interactions.calls) == 2


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (400, "invalid_request"),
        (401, "auth_or_permission"),
        (403, "auth_or_permission"),
        (404, "model_unavailable"),
        (409, "request_rejected"),
    ],
)
def test_gemini_does_not_retry_permanent_http_errors(status_code: int, category: str) -> None:
    client = _Client([_StatusError(status_code, "secret provider body"), _response()])
    reasoner = _gemini(client)

    with pytest.raises(QAProviderRequestError) as captured:
        reasoner.reason(
            _request(),
            (_image(),),
            QACompetitionSpec.historical_numeric(),
        )

    assert len(client.interactions.calls) == 1
    assert captured.value.safe_metadata() == {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "attempts": 1,
        "category": category,
        "status_code": status_code,
    }
    assert "secret provider body" not in str(captured.value)


@pytest.mark.parametrize("outcome", ["not json", ""])
def test_gemini_does_not_retry_invalid_structured_output(outcome) -> None:
    client = _Client([outcome, _response()])
    reasoner = _gemini(client)

    with pytest.raises(QAProviderResponseError) as captured:
        reasoner.reason(
            _request(),
            (_image(),),
            QACompetitionSpec.historical_numeric(),
        )
    assert len(client.interactions.calls) == 1
    assert captured.value.provider == "gemini"
    assert captured.value.model == "gemini-3.6-flash"
    assert captured.value.attempts == 1
    assert captured.value.category == "response_invalid"


@pytest.mark.parametrize(
    "outcome",
    [
        TimeoutError("secret timeout"),
        ConnectionError("secret connection"),
        httpx.ReadTimeout(
            "secret read timeout",
            request=httpx.Request("POST", "https://example.invalid"),
        ),
        httpx.ConnectError(
            "secret blocked connection",
            request=httpx.Request("POST", "https://example.invalid"),
        ),
    ],
)
def test_gemini_retries_builtin_and_httpx_transport_errors(outcome: Exception) -> None:
    client = _Client([outcome, _response()])
    reasoner = _gemini(client)

    result = reasoner.reason(
        _request(),
        (_image(),),
        QACompetitionSpec.historical_numeric(),
    )

    assert result.attempts == 2
    assert len(client.interactions.calls) == 2


def test_gemini_exhausted_blocked_connection_is_sanitized_network_failure() -> None:
    secret = "blocked-host-with-secret-token"
    failures = [
        httpx.ConnectError(
            secret,
            request=httpx.Request("POST", "https://example.invalid"),
        )
        for _ in range(3)
    ]
    reasoner = _gemini(_Client(failures))

    with pytest.raises(QAProviderRequestError) as captured:
        reasoner.reason(_request(), (_image(),), QACompetitionSpec.historical_numeric())

    assert captured.value.category == "network_unreachable"
    assert captured.value.attempts == 3
    assert captured.value.status_code is None
    assert secret not in str(captured.value)


def test_gemini_bounds_attempts_and_scrubs_sdk_exception_chain() -> None:
    secret = "fake-secret-value-that-must-not-survive"
    client = _Client([RuntimeError(f"header credential={secret}")])
    reasoner = _gemini(client)

    with pytest.raises(QAProviderRequestError) as captured:
        reasoner.reason(
            _request(question=f"prompt containing {secret}"),
            (_image(),),
            QACompetitionSpec.historical_numeric(),
        )

    rendered = "".join(
        traceback.format_exception(
            type(captured.value),
            captured.value,
            captured.value.__traceback__,
        )
    )
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert secret not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None

    with pytest.raises(ValueError, match="cannot exceed 5"):
        GeminiInteractionsReasoner(api_key="dummy", max_attempts=6)


def test_gemini_free_tier_governor_limits_thirteenth_request_window() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = _Client([_response() for _ in range(14)])
    reasoner = _gemini(
        client,
        requests_per_minute=13,
        clock=clock,
        sleep=sleep,
    )
    spec = QACompetitionSpec.historical_numeric()

    for _ in range(14):
        reasoner.reason(_request(), (_image(),), spec)

    assert sleeps == [60.0]


def test_gemini_uses_injected_governor_for_every_physical_attempt() -> None:
    acquisitions: list[str] = []

    class Governor:
        def acquire(self) -> None:
            acquisitions.append("acquire")

    client = _Client([_StatusError(500), _response()])
    reasoner = _gemini(client, request_governor=Governor())  # type: ignore[arg-type]

    result = reasoner.reason(
        _request(),
        (_image(),),
        QACompetitionSpec.historical_numeric(),
    )

    assert result.attempts == 2
    assert acquisitions == ["acquire", "acquire"]


def test_gemini_governor_preserves_minimum_spacing_across_threads() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    governor = GeminiRequestGovernor(
        13,
        clock=lambda: now[0],
        sleep=sleep,
        min_interval_seconds=4.5,
    )
    governor.acquire()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(governor.acquire).result(timeout=1)

    assert sleeps == [4.5]


def test_gemini_reasoners_share_pacing_for_one_process_credential() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    settings = {
        "api_key": "one-process-key",
        "clock": clock,
        "sleep": sleep,
        "min_request_interval_seconds": 4.5,
        "retry_base_seconds": 0,
        "retry_max_seconds": 0,
    }
    first = GeminiInteractionsReasoner(
        client_factory=lambda **_settings: _Client([_response()]),
        **settings,
    )
    second = GeminiInteractionsReasoner(
        client_factory=lambda **_settings: _Client([_response()]),
        **settings,
    )

    spec = QACompetitionSpec.historical_numeric()
    first.reason(_request(), (_image(),), spec)
    second.reason(_request(), (_image(),), spec)

    assert sleeps == [4.5]


def test_gemini_reasoners_share_terminal_429_cooldown() -> None:
    now = [0.0]

    def clock() -> float:
        return now[0]

    settings = {
        "api_key": "one-process-key",
        "clock": clock,
        "sleep": lambda seconds: now.__setitem__(0, now[0] + seconds),
        "min_request_interval_seconds": 0,
        "max_attempts": 1,
    }
    limited_client = _Client([_ProviderHTTPError(429)])
    waiting_client = _Client([_response()])
    limited = GeminiInteractionsReasoner(
        client_factory=lambda **_settings: limited_client,
        **settings,
    )
    waiting = GeminiInteractionsReasoner(
        client_factory=lambda **_settings: waiting_client,
        **settings,
    )

    spec = QACompetitionSpec.historical_numeric()
    with pytest.raises(QAProviderRequestError) as exhausted:
        limited.reason(_request(), (_image(),), spec)
    with pytest.raises(QAProviderRequestError) as cooling:
        waiting.reason(_request(), (_image(),), spec)

    assert exhausted.value.attempts == 1
    assert cooling.value.attempts == 0
    assert cooling.value.category == "rate_limited"
    assert not waiting_client.interactions.calls


def test_gemini_retry_headers_jitter_and_terminal_cooldown() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = _Client(
        [_ProviderHTTPError(429, {"Retry-After": "7"}) for _ in range(5)]
    )
    reasoner = GeminiInteractionsReasoner(
        api_key="dummy-test-key",
        client_factory=lambda **_settings: client,
        max_attempts=5,
        retry_base_seconds=5,
        retry_max_seconds=60,
        rate_limit_cooldown_seconds=60,
        clock=lambda: now[0],
        sleep=sleep,
        jitter=lambda _low, _high: 0.25,
    )

    with pytest.raises(QAProviderRequestError) as captured:
        reasoner.reason(_request(), (_image(),), QACompetitionSpec.historical_numeric())
    assert captured.value.attempts == 5
    assert captured.value.category == "rate_limited"
    assert len(client.interactions.calls) == 5
    assert sleeps == [7.0, 10.25, 20.25, 40.25]

    with pytest.raises(QAProviderRequestError) as cooling:
        reasoner.reason(_request(), (_image(),), QACompetitionSpec.historical_numeric())
    assert cooling.value.attempts == 0
    assert cooling.value.status_code == 429
    assert len(client.interactions.calls) == 5


def test_retry_header_parser_supports_http_date_epoch_and_malformed_values() -> None:
    from email.utils import formatdate

    wall = 1_700_000_000.0
    retry_date = _ProviderHTTPError(
        429, {"Retry-After": formatdate(wall + 12, usegmt=True)}
    )
    assert retry_delay_for_exception(
        retry_date,
        1,
        jitter=lambda _low, _high: 0,
        wall_clock=lambda: wall,
    ) == pytest.approx(12.0)

    reset = _ProviderHTTPError(429, {"x-ratelimit-reset": str(wall + 9)})
    assert retry_delay_for_exception(
        reset,
        1,
        jitter=lambda _low, _high: 0,
        wall_clock=lambda: wall,
    ) == pytest.approx(9.0)

    oversized = _ProviderHTTPError(429, {"Retry-After": "90"})
    assert retry_delay_for_exception(
        oversized,
        1,
        jitter=lambda _low, _high: 0,
        wall_clock=lambda: wall,
    ) == pytest.approx(60.0)

    malformed = _ProviderHTTPError(429, {"Retry-After": "not-a-date"})
    assert retry_delay_for_exception(
        malformed,
        1,
        jitter=lambda _low, _high: 0,
        wall_clock=lambda: wall,
    ) == pytest.approx(5.0)


def test_gemini_free_tier_backoff_sequence_is_deterministic() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    reasoner = GeminiInteractionsReasoner(
        api_key="dummy-test-key",
        client_factory=lambda **_settings: _Client(
            [_ProviderHTTPError(500) for _ in range(5)]
        ),
        max_attempts=5,
        retry_base_seconds=5,
        retry_max_seconds=60,
        clock=lambda: now[0],
        sleep=sleep,
        jitter=lambda _low, _high: 0,
    )

    with pytest.raises(QAProviderRequestError):
        reasoner.reason(_request(), (_image(),), QACompetitionSpec.historical_numeric())

    assert sleeps == [5.0, 10.0, 20.0, 40.0]


def test_gemini_concurrency_bound_serializes_provider_calls() -> None:
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    state = {"active": 0, "maximum": 0, "calls": 0}

    class BlockingInteractions:
        def create(self, **_kwargs):
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
                state["calls"] += 1
                entered.set()
            release.wait(timeout=2)
            with lock:
                state["active"] -= 1
            return SimpleNamespace(output_text=_response())

    client = SimpleNamespace(interactions=BlockingInteractions())
    reasoner = GeminiInteractionsReasoner(
        api_key="dummy-test-key",
        client_factory=lambda **_settings: client,
        max_concurrency=1,
        requests_per_minute=10,
        # This test isolates semaphore behavior; pacing has its own deterministic test.
        min_request_interval_seconds=0,
    )
    spec = QACompetitionSpec.historical_numeric()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(reasoner.reason, _request(), (_image(),), spec)
        assert entered.wait(timeout=1)
        second = executor.submit(reasoner.reason, _request(), (_image(),), spec)
        time.sleep(0.05)
        assert state["calls"] == 1
        release.set()
        assert first.result(timeout=2).answer == "2"
        assert second.result(timeout=2).answer == "2"
    assert state["maximum"] == 1


def test_gemini_free_tier_rejects_unsafe_rate_or_concurrency_configuration() -> None:
    default = GeminiInteractionsReasoner(api_key="dummy-test-key")
    assert default._rate_limiter._min_interval_seconds == 4.5
    with pytest.raises(ValueError, match="cannot exceed.*13"):
        GeminiInteractionsReasoner(api_key="dummy-test-key", requests_per_minute=14)
    with pytest.raises(ValueError, match="max_concurrency must be 1"):
        GeminiInteractionsReasoner(api_key="dummy-test-key", max_concurrency=2)


def test_provider_fingerprints_are_deterministic_secret_free_and_ordered() -> None:
    first = GeminiInteractionsReasoner(api_key="secret-a")
    equivalent = GeminiInteractionsReasoner(api_key="secret-b")
    other = GeminiInteractionsReasoner(api_key="secret-a", model_name="gemini-other")

    assert first.fingerprint == equivalent.fingerprint
    assert first.fingerprint != other.fingerprint
    assert "secret" not in first.fingerprint
    assert (
        ProviderChainReasoner((first, other)).fingerprint
        != ProviderChainReasoner((other, first)).fingerprint
    )


class _FailingReasoner:
    provider_name = "failing"
    fingerprint = "failing-v1"

    def reason(self, *_args):
        raise QAProviderRequestError(
            "temporary provider failure",
            attempts=5,
            category="server_unavailable",
        )

    def unload(self) -> None:
        return None


class _SuccessfulReasoner:
    provider_name = "successful"
    fingerprint = "successful-v1"

    def reason(self, *_args):
        return QAReasoningResult(
            raw_answer="two",
            normalized_answer="2",
            answerable=True,
            evidence_frame_indices=(1,),
            provider=self.provider_name,
            model_name="fake",
            prompt_version="fake-v1",
            latency_ms=0,
        )

    def unload(self) -> None:
        return None


def test_provider_chain_falls_back_in_order() -> None:
    result = ProviderChainReasoner((_FailingReasoner(), _SuccessfulReasoner())).reason(
        _request(),
        (_image(),),
        QACompetitionSpec.historical_numeric(),
    )
    assert result.provider == "successful"


@pytest.mark.parametrize(
    "category",
    ["client_initialization", "client_unavailable", "model_unavailable", "not_configured"],
)
def test_provider_chain_does_not_fallback_for_provider_setup_errors(category: str) -> None:
    fallback = _SuccessfulReasoner()

    class First(_FailingReasoner):
        def reason(self, *_args):
            raise QAProviderUnavailableError("provider unavailable", category=category)

    with pytest.raises(QAProviderUnavailableError) as captured:
        ProviderChainReasoner((First(), fallback)).reason(
            _request(),
            (_image(),),
            QACompetitionSpec.historical_numeric(),
        )

    assert captured.value.category == category


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (
            QAProviderRequestError(
                "private",
                category="auth_or_permission",
                status_code=401,
            ),
            "auth_or_permission",
        ),
        (
            QAProviderRequestError(
                "private",
                category="invalid_request",
                status_code=400,
            ),
            "invalid_request",
        ),
        (QAProviderResponseError("private", category="response_invalid"), "response_invalid"),
    ],
)
def test_provider_chain_does_not_fallback_for_permanent_or_schema_errors(
    error: Exception,
    category: str,
) -> None:
    fallback = _SuccessfulReasoner()

    class First(_FailingReasoner):
        def reason(self, *_args):
            raise error

    with pytest.raises(QAReasonerError) as captured:
        ProviderChainReasoner((First(), fallback)).reason(
            _request(),
            (_image(),),
            QACompetitionSpec.historical_numeric(),
        )

    assert captured.value.category == category


@pytest.mark.parametrize("category", ["network_unreachable", "rate_limited", "server_unavailable"])
def test_provider_chain_falls_back_after_exhausted_transient_error(category: str) -> None:
    class First(_FailingReasoner):
        def reason(self, *_args):
            raise QAProviderRequestError(
                "private",
                attempts=5,
                category=category,
            )

    result = ProviderChainReasoner((First(), _SuccessfulReasoner())).reason(
        _request(),
        (_image(),),
        QACompetitionSpec.historical_numeric(),
    )

    assert result.provider == "successful"


def test_provider_chain_preserves_only_safe_error_metadata() -> None:
    secret = "raw-sdk-body-with-secret"

    class LeakyReasoner(_FailingReasoner):
        provider_name = "gemini"
        model_name = "gemini-test"

        def reason(self, *_args):
            raise QAProviderRequestError(
                secret,
                provider=self.provider_name,
                model=self.model_name,
                attempts=1,
                category="invalid_request",
                status_code=400,
            )

    with pytest.raises(QAProviderRequestError) as captured:
        ProviderChainReasoner((LeakyReasoner(),)).reason(
            _request(),
            (_image(),),
            QACompetitionSpec.historical_numeric(),
        )

    assert secret not in str(captured.value)
    assert captured.value.safe_metadata()["category"] == "invalid_request"
    assert captured.value.safe_metadata()["status_code"] == 400


def test_qwen_is_lazy_cache_only_pixel_bounded_and_releases_gpu(monkeypatch) -> None:
    events: list[str] = []
    processor_settings: dict = {}
    observed_sizes: list[tuple[int, int]] = []

    class Lease:
        def __enter__(self):
            events.append("lease-enter")
            return self

        def __exit__(self, *_args):
            events.append("lease-exit")

    class Batch(dict):
        def to(self, device):
            events.append(f"inputs-to-{device}")
            return self

    class Processor:
        def apply_chat_template(self, messages, **settings):
            events.append("process")
            assert settings["tokenize"] is True
            for part in messages[1]["content"]:
                if part["type"] == "image":
                    observed_sizes.append(part["image"].size)
            return Batch(input_ids=[[1, 2]])

        def batch_decode(self, *_args, **_kwargs):
            raise AssertionError("string generation should bypass decoding")

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(model_name, **settings):
            events.append("processor-load")
            processor_settings.update({"model_name": model_name, **settings})
            return Processor()

    class Model:
        def to(self, device):
            events.append(f"model-to-{device}")
            return self

        def eval(self):
            events.append("model-eval")

        def generate(self, **settings):
            events.append("generate")
            assert settings["do_sample"] is False
            assert settings["max_new_tokens"] == 32
            return _response("Red")

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_name, **settings):
            events.append("model-load")
            assert model_name == "Qwen/Qwen3-VL-4B-Instruct"
            assert settings["local_files_only"] is True
            return Model()

    class Cuda:
        @staticmethod
        def empty_cache():
            events.append("empty-cache")

    fake_torch = SimpleNamespace(
        float16=object(),
        inference_mode=nullcontext,
        cuda=Cuda(),
    )
    fake_transformers = SimpleNamespace(
        AutoProcessor=ProcessorFactory,
        AutoModelForMultimodalLM=ModelFactory,
    )

    def import_module(name: str):
        return {"torch": fake_torch, "transformers": fake_transformers}[name]

    monkeypatch.setattr("src.vlm.qa_reasoner.importlib.import_module", import_module)
    reasoner = Qwen3VLReasoner(
        max_new_tokens=32,
        max_pixels=25,
        gpu_lease=Lease,
        before_load=lambda: events.append("before-load"),
    )
    assert not reasoner.loaded
    assert events == []

    result = reasoner.reason(
        _request(question="What color is the wall?"),
        (_image(size=(20, 10)),),
        QACompetitionSpec.conservative_general_text(),
    )

    assert result.answer == "red"
    assert reasoner.loaded
    assert events.index("lease-enter") < events.index("before-load")
    assert events.index("before-load") < events.index("processor-load")
    assert processor_settings["local_files_only"] is True
    assert processor_settings["max_pixels"] == 25
    assert observed_sizes and observed_sizes[0][0] * observed_sizes[0][1] <= 25

    reasoner.unload()
    assert not reasoner.loaded
    assert events[-2:] == ["empty-cache", "lease-exit"]


def test_vlm_engine_facade_preserves_legacy_shape_without_eager_calls(tmp_path) -> None:
    class Reasoner:
        fingerprint = "facade-fake-v1"

        def __init__(self) -> None:
            self.calls = []
            self.unloaded = False

        def reason(self, request, images, spec):
            self.calls.append((request, images, spec))
            return QAReasoningResult(
                raw_answer="Red",
                normalized_answer="red",
                answerable=True,
                evidence_frame_indices=(1,),
                provider="fake",
                model_name="fake-model",
                prompt_version="fake-v1",
                latency_ms=0,
            )

        def unload(self):
            self.unloaded = True

    path = tmp_path / "frame.png"
    path.write_bytes(_image_bytes())
    provider = Reasoner()
    engine = VLMEngine(reasoner=provider)
    assert provider.calls == []

    result = engine.ask(path, "What color is the object?", system_prompt="A person enters.")

    assert result == {
        "answer": "red",
        "confidence": 1.0,
        "answerable": True,
        "evidence_frame_indices": [1],
        "provider": "fake",
        "model_name": "fake-model",
    }
    assert provider.calls[0][0].event_description == "A person enters."
    engine.unload()
    assert provider.unloaded
