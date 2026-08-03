"""Offline checks for explicit, bounded Visual Q&A diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import check_gemini, check_qa_live
from src.contracts import QAReasoningResult


def _config() -> dict:
    return {
        "services": {
            "gemini": {
                "api_key": "dummy-test-key",
                "model": "gemini-test",
                "timeout_seconds": 1,
                "requests_per_minute": 30,
                "min_request_interval_seconds": 0,
                "max_retries": 2,
            }
        }
    }


def _install_synthetic_checker_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict], list[int], list[dict], list[object]]:
    direct_calls: list[dict] = []
    reason_calls: list[int] = []
    constructor_settings: list[dict] = []
    governors: list[object] = []

    class Governor:
        def __init__(self, *_args, **_kwargs) -> None:
            self.acquisitions = 0
            governors.append(self)

        def acquire(self) -> None:
            self.acquisitions += 1

    class DirectInteractions:
        def create(self, **kwargs):
            direct_calls.append(kwargs)
            output = "OK" if len(direct_calls) == 1 else '{"answer":"OK"}'
            return SimpleNamespace(output_text=output)

    class DirectClient:
        def __init__(self, **_kwargs) -> None:
            self.interactions = DirectInteractions()

        def close(self) -> None:
            return None

    class Reasoner:
        def __init__(self, **settings) -> None:
            constructor_settings.append(settings)
            self.settings = settings

        def initialize(self) -> None:
            return None

        def reason(self, _request, images, _spec):
            self.settings["request_governor"].acquire()
            reason_calls.append(len(images))
            answer = str(sum(image.source_frame_idx < 3 for image in images))
            return QAReasoningResult(
                raw_answer=answer,
                normalized_answer=answer,
                answerable=True,
                evidence_frame_indices=(1,),
                provider="gemini",
                model_name="gemini-test",
                prompt_version="test-v1",
                latency_ms=1.0,
                attempts=1,
            )

        def unload(self) -> None:
            return None

    from google import genai

    monkeypatch.setattr(check_gemini, "load_config", _config)
    monkeypatch.setattr(check_gemini, "GeminiInteractionsReasoner", Reasoner)
    monkeypatch.setattr(check_gemini, "GeminiRequestGovernor", Governor)
    monkeypatch.setattr(genai, "Client", DirectClient)
    return direct_calls, reason_calls, constructor_settings, governors


def test_synthetic_checker_defaults_to_one_physical_multimodal_attempt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    direct_calls, reason_calls, constructor_settings, governors = (
        _install_synthetic_checker_fakes(monkeypatch)
    )

    assert check_gemini.main([]) == 0

    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert payloads[-1]["stage"] == "one_image"
    assert all(payload["sdk_version"] for payload in payloads)
    assert direct_calls == []
    assert reason_calls == [1]
    assert constructor_settings[0]["max_attempts"] == 1
    assert constructor_settings[0]["requests_per_minute"] == 13
    assert governors[0].acquisitions == 1


def test_synthetic_checker_extended_mode_is_explicit_and_uses_one_governor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    direct_calls, reason_calls, constructor_settings, governors = (
        _install_synthetic_checker_fakes(monkeypatch)
    )

    assert check_gemini.main(["--allow-multiple-provider-calls"]) == 0

    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    stages = [payload["stage"] for payload in payloads]
    assert stages[-4:] == ["minimal_text", "structured_text", "one_image", "five_images"]
    assert len(direct_calls) == 2
    assert reason_calls == [1, 5]
    assert constructor_settings[0]["max_attempts"] == 3
    assert constructor_settings[0]["request_governor"] is governors[0]
    assert governors[0].acquisitions == 4


def test_synthetic_direct_stage_retries_timeout_with_a_fresh_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[object] = []
    closed: list[object] = []
    calls = 0

    def new_client(_api_key: str, _timeout_seconds: float) -> object:
        client = object()
        clients.append(client)
        return client

    def operation(_client: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("transient fixture")
        return SimpleNamespace(output_text="OK")

    monkeypatch.setattr(check_gemini, "_new_direct_client", new_client)
    monkeypatch.setattr(check_gemini, "_close_client", closed.append)
    monkeypatch.setattr(check_gemini.time, "sleep", lambda _seconds: None)

    result, attempts, error = check_gemini._run_direct_stage(
        operation,
        api_key="dummy",
        model_name="gemini-test",
        timeout_seconds=1,
        max_attempts=3,
    )

    assert result.output_text == "OK"
    assert attempts == 2
    assert error is None
    assert len(clients) == 2
    assert closed == clients


@pytest.mark.parametrize(
    "flags",
    [(), ("--live-gemini",), ("--allow-external-frame-transfer",)],
)
def test_real_checker_requires_both_explicit_consent_flags(
    flags: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_qa_live,
        "build_runtime",
        lambda _config: (_ for _ in ()).throw(AssertionError("runtime must not start")),
    )
    arguments = [
        "--batch-id",
        "b1",
        "--event-description",
        "event",
        "--question",
        "question",
        *flags,
    ]

    assert check_qa_live.main(arguments) == 2


def test_real_checker_uses_public_initialization_and_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized: list[str] = []
    provider_settings: list[dict] = []
    live_calls: list[object] = []

    class Initializable:
        def __init__(self, name: str) -> None:
            self.name = name

        def initialize(self) -> None:
            initialized.append(self.name)

    class Pipeline:
        def check_one_window_live(self, request, *, reasoner, batch_filters):
            live_calls.append((request, reasoner, batch_filters))
            return {
                "status": "unanswerable",
                "provider": "gemini",
                "model_name": "gemini-test",
                "attempts": 1,
                "batch_id": "b1",
                "video_id": "video-1",
                "source_frame_idx": 10,
                "frames": [{"position": 1}],
                "normalized_answer": "",
            }

    class Reasoner:
        max_attempts = 1

        def __init__(self, **settings) -> None:
            provider_settings.append(settings)

        def initialize(self) -> None:
            initialized.append("provider")

        def unload(self) -> None:
            return None

    context = SimpleNamespace(
        qa_pipeline=Pipeline(),
        artifact_repository=Initializable("artifacts"),
        ledger=Initializable("ledger"),
        validate_batch_filters=lambda values: tuple(values),
    )
    monkeypatch.setattr(check_qa_live, "load_config", _config)
    monkeypatch.setattr(check_qa_live, "build_runtime", lambda _config: context)
    monkeypatch.setattr(check_qa_live, "GeminiInteractionsReasoner", Reasoner)

    result = check_qa_live.main(
        [
            "--batch-id",
            "b1",
            "--event-description",
            "event",
            "--question",
            "question",
            "--live-gemini",
            "--allow-external-frame-transfer",
        ]
    )

    assert result == 0
    assert initialized == ["ledger", "artifacts", "provider"]
    assert provider_settings[0]["max_attempts"] == 1
    assert len(live_calls) == 1
