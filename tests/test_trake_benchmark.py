"""CPU-only tests for the TRAKE latency benchmark contract."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.benchmark_trake_latency import (
    _pipeline_fingerprint,
    _run_query,
    _summary,
)


def test_trake_benchmark_summary_and_fingerprint_are_deterministic() -> None:
    assert _summary([1.0, 2.0, 3.0, 4.0]) == {
        "mean": 2.5,
        "p50": 2.5,
        "p90": 3.7,
        "p95": 3.85,
        "min": 1.0,
        "max": 4.0,
    }
    config = {
        "profile": "laptop",
        "retrieval": {"trake": {"candidate_depth": 300}},
        "models": {"organizer_clip": {"model_name": "clip", "embedding_dim": 512}},
        "services": {"qdrant": {"api_key": "must-not-be-fingerprinted"}},
    }
    first = _pipeline_fingerprint(config, "b1")
    config["services"]["qdrant"]["api_key"] = "another-secret"
    assert _pipeline_fingerprint(config, "b1") == first
    assert "secret" not in first


def test_trake_benchmark_constructs_one_batch_ordered_request() -> None:
    calls = []

    class _Pipeline:
        @staticmethod
        def search(request, *, default_batch_id):
            calls.append((request, default_batch_id))
            return SimpleNamespace(results=(1, 2))

    context = SimpleNamespace(
        trake_pipeline=_Pipeline(),
        validate_batch_filters=lambda values: tuple(values),
    )

    latency, hits = _run_query(
        context,
        batch_id="b1",
        global_description="toàn cảnh",
        stages=["đầu", "cuối"],
        query_id="bench",
        top_k=100,
    )

    request, default_batch = calls[0]
    assert latency >= 0
    assert hits == 2
    assert default_batch == "b1"
    assert request.batch_filters == ("b1",)
    assert [stage.stage_index for stage in request.stages] == [1, 2]
