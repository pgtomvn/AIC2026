"""Focused tests for environment-aware configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils import io_helpers


def _write_project_files(project_root: Path, config_text: str, env_text: str = "") -> None:
    config_dir = project_root / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(config_text, encoding="utf-8")
    if env_text:
        (project_root / ".env").write_text(env_text, encoding="utf-8")


def _point_loader_at(monkeypatch, project_root: Path) -> None:
    monkeypatch.setattr(io_helpers, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(io_helpers, "CONFIG_DIR", project_root / "config")


def test_load_config_resolves_service_metadata_with_process_precedence(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    _write_project_files(
        tmp_path,
        """
services:
  qdrant:
    url: http://yaml-qdrant:6333
    url_env: TEST_QDRANT_URL
    api_key_env: TEST_QDRANT_API_KEY
  meilisearch:
    url: http://yaml-meilisearch:7700
    url_env: TEST_MEILISEARCH_URL
    api_key_env: TEST_MEILISEARCH_API_KEY
""",
        """
TEST_QDRANT_URL=http://dotenv-qdrant:6333
TEST_QDRANT_API_KEY=dotenv-qdrant-secret
TEST_MEILISEARCH_URL=http://dotenv-meilisearch:7700
TEST_MEILISEARCH_API_KEY=dotenv-meilisearch-secret
""",
    )
    _point_loader_at(monkeypatch, tmp_path)
    monkeypatch.setenv("TEST_QDRANT_URL", "http://process-qdrant:6333")
    monkeypatch.setenv("TEST_QDRANT_API_KEY", "process-qdrant-secret")
    monkeypatch.delenv("TEST_MEILISEARCH_URL", raising=False)
    monkeypatch.delenv("TEST_MEILISEARCH_API_KEY", raising=False)
    environment_before = os.environ.copy()

    config = io_helpers.load_config()

    assert config["services"]["qdrant"]["url"] == "http://process-qdrant:6333"
    assert config["services"]["qdrant"]["api_key"] == "process-qdrant-secret"
    assert (
        config["services"]["meilisearch"]["url"]
        == "http://dotenv-meilisearch:7700"
    )
    assert (
        config["services"]["meilisearch"]["api_key"]
        == "dotenv-meilisearch-secret"
    )
    assert os.environ == environment_before
    captured = capfd.readouterr()
    assert "process-qdrant-secret" not in captured.out + captured.err
    assert "dotenv-meilisearch-secret" not in captured.out + captured.err


def test_load_config_keeps_defaults_for_unset_or_empty_values(
    tmp_path: Path, monkeypatch
) -> None:
    _write_project_files(
        tmp_path,
        """
services:
  qdrant:
    url: http://yaml-qdrant:6333
    api_key: yaml-qdrant-key
    url_env: TEST_EMPTY_QDRANT_URL
    api_key_env: TEST_MISSING_QDRANT_API_KEY
  meilisearch:
    url: http://yaml-meilisearch:7700
    api_key: yaml-meilisearch-key
    url_env: TEST_MISSING_MEILISEARCH_URL
    api_key_env: TEST_EMPTY_MEILISEARCH_API_KEY
""",
        """
TEST_EMPTY_QDRANT_URL=
TEST_EMPTY_MEILISEARCH_API_KEY=
""",
    )
    _point_loader_at(monkeypatch, tmp_path)
    for env_name in (
        "TEST_EMPTY_QDRANT_URL",
        "TEST_MISSING_QDRANT_API_KEY",
        "TEST_MISSING_MEILISEARCH_URL",
        "TEST_EMPTY_MEILISEARCH_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    files_before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    config = io_helpers.load_config()

    assert config["services"]["qdrant"]["url"] == "http://yaml-qdrant:6333"
    assert config["services"]["qdrant"]["api_key"] == "yaml-qdrant-key"
    assert (
        config["services"]["meilisearch"]["url"]
        == "http://yaml-meilisearch:7700"
    )
    assert config["services"]["meilisearch"]["api_key"] == "yaml-meilisearch-key"
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == files_before


def test_load_config_resolves_profile_with_process_then_dotenv_precedence(
    tmp_path: Path, monkeypatch
) -> None:
    _write_project_files(tmp_path, "profile: laptop\n", "AIC_PROFILE=accelerated\n")
    _point_loader_at(monkeypatch, tmp_path)
    monkeypatch.delenv("AIC_PROFILE", raising=False)

    assert io_helpers.load_config()["profile"] == "accelerated"

    monkeypatch.setenv("AIC_PROFILE", "laptop")
    assert io_helpers.load_config()["profile"] == "laptop"


@pytest.mark.parametrize("source", ["yaml", "dotenv", "process"])
def test_load_config_rejects_unknown_profiles(
    tmp_path: Path, monkeypatch, source: str
) -> None:
    yaml_profile = "unsupported" if source == "yaml" else "laptop"
    env_text = "AIC_PROFILE=unsupported\n" if source == "dotenv" else ""
    _write_project_files(tmp_path, f"profile: {yaml_profile}\n", env_text)
    _point_loader_at(monkeypatch, tmp_path)
    monkeypatch.delenv("AIC_PROFILE", raising=False)
    if source == "process":
        monkeypatch.setenv("AIC_PROFILE", "unsupported")

    with pytest.raises(ValueError, match="AIC_PROFILE must be one of"):
        io_helpers.load_config()


def test_load_config_resolves_batch_with_process_then_dotenv_precedence(
    tmp_path: Path, monkeypatch
) -> None:
    _write_project_files(
        tmp_path,
        "profile: laptop\nruntime:\n  default_batch_id: yaml-batch\n",
        "AIC_BATCH_ID=dotenv-batch\n",
    )
    _point_loader_at(monkeypatch, tmp_path)
    monkeypatch.delenv("AIC_BATCH_ID", raising=False)

    assert io_helpers.load_config()["runtime"]["default_batch_id"] == "dotenv-batch"

    monkeypatch.setenv("AIC_BATCH_ID", "process-batch")
    assert io_helpers.load_config()["runtime"]["default_batch_id"] == "process-batch"


def test_load_config_rejects_unsafe_selected_batch(tmp_path: Path, monkeypatch) -> None:
    _write_project_files(
        tmp_path,
        "profile: laptop\nruntime:\n  default_batch_id: safe-batch\n",
    )
    _point_loader_at(monkeypatch, tmp_path)
    monkeypatch.setenv("AIC_BATCH_ID", "../unsafe")

    with pytest.raises(ValueError, match="batch_id"):
        io_helpers.load_config()


def test_launch_identity_is_canonical_only_for_complete_matching_metadata() -> None:
    config = {
        "profile": "laptop",
        "runtime": {"default_batch_id": "b2"},
    }
    environment = {
        "AIC_LAUNCH_CONTRACT": "run_app-v2",
        "AIC_LAUNCH_ID": "08b8fdc7-2a15-49d3-887c-884299e42473",
        "AIC_LAUNCH_PORT": "8504",
        "AIC_LAUNCH_BATCH": "b2",
        "AIC_LAUNCH_PROFILE": "laptop",
        "AIC_LAUNCHED_AT_UTC": "2026-08-02T05:00:00Z",
        "AIC_LAUNCH_LOCK_SHA256": "b" * 64,
        "AIC_RUNTIME_STAMP_SHA256": "a" * 64,
        "OPENROUTER_API_KEY": "must-not-appear",
    }

    identity = io_helpers.load_launch_identity(config, environ=environment)

    assert identity["canonical"] is True
    assert identity["port"] == 8504
    assert identity["batch_id"] == "b2"
    assert identity["lock_sha256"] == "b" * 64
    assert "must-not-appear" not in str(identity)

    environment["AIC_LAUNCH_BATCH"] = "another-batch"
    invalid = io_helpers.load_launch_identity(config, environ=environment)
    assert invalid == {
        "canonical": False,
        "contract": "run_app-v2",
        "batch_id": "b2",
        "profile": "laptop",
        "reason_code": "invalid_launch_identity",
    }
