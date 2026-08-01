"""Focused tests for environment-aware configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

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
