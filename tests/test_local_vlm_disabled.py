"""A disabled local VLM must never touch the network (Gemini-only deployments)."""

from __future__ import annotations

import pytest

from src.readiness import ChannelApplicability, ChannelState
from src.utils.io_helpers import service_enabled
from src.vlm.local_qwen import OllamaQwenReasoner
from src.vlm.qa_reasoner import QAProviderUnavailableError


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), ("off", False), ("0", False), ("No", False), ("on", True), (None, False),
])
def test_service_enabled_reads_environment_strings(value, expected):
    assert service_enabled({"enabled": value}) is expected
    assert service_enabled({}) is True


def test_disabled_reasoner_fails_fast_without_a_connection(monkeypatch):
    reasoner = OllamaQwenReasoner(enabled=False)
    monkeypatch.setattr(reasoner, "_session", lambda: pytest.fail("no HTTP session for a disabled VLM"))
    reasoner.unload()
    with pytest.raises(QAProviderUnavailableError) as error:
        reasoner._request("GET", "/api/tags")
    assert error.value.category == "not_configured"


def test_readiness_reports_disabled_local_vlm_without_probing(monkeypatch):
    from src import readiness

    service = object.__new__(readiness.ReadinessService)
    service.config = {"services": {"ollama_vlm": {"enabled": "off"}}}
    monkeypatch.setattr(service, "_request_json", lambda *a, **k: pytest.fail("probed a disabled VLM"), raising=False)
    channel = service._probe_ollama_vlm(ChannelApplicability.OPTIONAL)
    assert channel.state == ChannelState.DISABLED_BY_PROFILE
    assert channel.reason_code == "local_vlm_disabled"


def test_router_reports_gemini_errors_when_the_local_model_is_disabled():
    from types import SimpleNamespace

    from src.vlm.qa_router import QAReasonerRouter

    request = SimpleNamespace(reasoning_policy="gemini_then_local", allow_external_frame_transfer=True)
    gemini = SimpleNamespace(provider_name="gemini")
    assert QAReasonerRouter(gemini, OllamaQwenReasoner(enabled=False)).providers_for(request) == (gemini,)
    enabled = OllamaQwenReasoner()
    assert QAReasonerRouter(gemini, enabled).providers_for(request) == (gemini, enabled)
