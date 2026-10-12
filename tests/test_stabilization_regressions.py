"""Regression cases discovered by the September live system audit."""

import pytest

from app.streamlit_app import _hud_state
from src.query.translator import QueryTextAdapter
from src.vlm.local_qwen import OllamaQwenReasoner


@pytest.mark.parametrize("query,expected", [
    ("Một người mặc áo đỏ đi qua cây cầu.", {("áo", "red")}),
    ("Một chiếc xe trắng đi qua cây cầu.", {("chiếc xe", "white")}),
    ("A red car.", {("car", "red")}),
    ("A person in red stands to the left of a white car.", {("person", "red"), ("car", "white")}),
])
def test_color_bindings_exclude_actions(query, expected):
    decomposition = QueryTextAdapter().prepare(query).decomposition
    assert decomposition is not None
    actual = {(value.subject, value.color) for value in decomposition.color_constraints}
    assert actual == expected


@pytest.mark.parametrize("state", ["configured", "configured_unverified"])
def test_configured_provider_is_not_presented_as_verified(state):
    assert _hud_state(state) == "UNVERIFIED"


def test_ollama_failed_allocation_is_resource_error():
    class Response:
        def json(self):
            return {"error": "failed to allocate CPU_REPACK buffer"}

    error = OllamaQwenReasoner._status_error(500, model_name="qwen", response=Response())
    assert error.category == "resource_exhausted"
    assert "CPU_REPACK" not in str(error)


def test_ollama_incompatible_gpu_is_distinct_from_service_down():
    class Response:
        def json(self):
            return {"error": "CUDA error: device kernel image is invalid"}

    error = OllamaQwenReasoner._status_error(500, model_name="qwen", response=Response())
    assert error.category == "gpu_runtime_incompatible"
