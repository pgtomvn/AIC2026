"""Side-effect-free runtime composition tests for Visual Q&A profiles."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src import CONFIG_DIR
from src.runtime import build_runtime
from src.vlm.gemini import GeminiFlashReasoner
from src.vlm.local_qwen import OllamaQwenReasoner
from src.vlm.qa_router import QAReasonerRouter


def _config(tmp_path: Path, profile: str) -> dict:
    config = deepcopy(
        yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8"))
    )
    config["profile"] = profile
    config["data"]["catalog_db"] = str(tmp_path / "jobs.sqlite3")
    config["data"]["artifact_db"] = str(tmp_path / "artifacts.sqlite3")
    config["data"]["qa_state_db"] = str(tmp_path / "qa.sqlite3")
    config["models"]["organizer_clip"]["enabled"] = False
    config["retrieval"]["raw_refinement"]["enabled"] = False
    config["services"]["gemini_vlm"]["api_key"] = "fake-secret-runtime-key"
    return config


def test_laptop_runtime_is_lazy_local_first_and_secret_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in tuple(os.environ):
        if name.startswith("AIC_LAUNCH_"):
            monkeypatch.delenv(name, raising=False)
    context = build_runtime(_config(tmp_path, "laptop"))
    reasoner = context.qa_pipeline.reasoner

    assert isinstance(reasoner, QAReasonerRouter)
    assert isinstance(reasoner.local, OllamaQwenReasoner)
    assert isinstance(reasoner.gemini, GeminiFlashReasoner)
    assert reasoner.local._http is None
    assert reasoner.gemini._client is None
    assert not (tmp_path / "jobs.sqlite3").exists()
    assert not (tmp_path / "artifacts.sqlite3").exists()
    assert not (tmp_path / "qa.sqlite3").exists()
    assert "fake-secret-runtime-key" not in repr(context)
    assert "accelerated_visual" not in {
        channel.name for channel in context.pipeline.dense_channels
    }
    assert context.selected_batch_id == "aic25-b1-reference"
    assert context.launch_identity["canonical"] is False
    assert context.validate_batch_filters(None) == ("aic25-b1-reference", "aic26-b2")
    context.config["retrieval"]["all_batches"]["enabled"] = False
    assert context.validate_batch_filters(None) == ("aic25-b1-reference",)


def test_runtime_batch_validation_accepts_registered_native_batch(tmp_path: Path) -> None:
    context = build_runtime(_config(tmp_path, "laptop"))
    context.ledger.initialize()
    context.ledger.register_batch("aic26-b2", "aic2026")

    assert context.validate_batch_filters(["aic26-b2", "aic26-b2"]) == ("aic26-b2",)
    with pytest.raises(ValueError, match="not configured or registered"):
        context.validate_batch_filters(["unknown-batch"])
    with pytest.raises(TypeError, match="iterable"):
        context.validate_batch_filters("aic26-b2")


def test_runtime_propagates_qdrant_grpc_without_connecting(tmp_path: Path) -> None:
    config = _config(tmp_path, "laptop")
    config["services"]["qdrant"]["prefer_grpc"] = True
    config["services"]["qdrant"]["grpc_port"] = 7444

    context = build_runtime(config)

    stores = [channel.store for channel in context.pipeline.dense_channels]
    assert stores
    assert all(store.prefer_grpc is True for store in stores)
    assert all(store.grpc_port == 7444 for store in stores)
    assert all(store._client is None for store in stores)

    trake_cores = context.trake_pipeline.pipelines
    assert trake_cores
    assert all(core.trajectory_cache is not None for core in trake_cores.values())
    assert all(
        core.trajectory_cache.snapshot()["state"] == "idle"
        for core in trake_cores.values()
    )
    assert context.warmup.trake_pipeline is context.trake_pipeline
    assert context.trake_pipeline.readiness_cache_seconds == 300


def test_accelerated_runtime_keeps_same_qa_router_and_adds_visual_channel(
    tmp_path: Path,
) -> None:
    context = build_runtime(_config(tmp_path, "accelerated"))

    assert isinstance(context.qa_reasoner, QAReasonerRouter)
    assert isinstance(context.qa_reasoner.local, OllamaQwenReasoner)
    assert "accelerated_visual" in {
        channel.name for channel in context.pipeline.dense_channels
    }
    assert not (tmp_path / "qa.sqlite3").exists()
