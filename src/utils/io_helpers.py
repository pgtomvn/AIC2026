"""
I/O Helpers
===========
File I/O, path management, config loading, and data serialization
utilities shared across the project.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from src import CONFIG_DIR, PROJECT_ROOT


def load_config(config_name: str = "settings.yaml") -> dict[str, Any]:
    """Load a YAML configuration file and apply service environment values.

    Values in the process environment take precedence over values read from the
    project ``.env`` file. Neither source is mutated. Service settings declare
    their variable names through ``url_env`` and ``api_key_env`` metadata; an
    unset or empty variable leaves the corresponding YAML default unchanged.

    Args:
        config_name: Filename of the config in config/.

    Returns:
        Parsed config dict.
    """
    config_path = CONFIG_DIR / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        return config

    dotenv_path = PROJECT_ROOT / ".env"
    dotenv_data = dotenv_values(dotenv_path) if dotenv_path.is_file() else {}
    services = config.get("services")
    if not isinstance(services, dict):
        return config

    for service in services.values():
        if not isinstance(service, dict):
            continue

        for setting_name in ("url", "api_key"):
            env_name = service.get(f"{setting_name}_env")
            if not isinstance(env_name, str) or not env_name:
                continue

            value = os.environ.get(env_name)
            if value in (None, ""):
                value = dotenv_data.get(env_name)
            if value not in (None, ""):
                service[setting_name] = value

    return config


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root.

    Args:
        relative_path: Path string relative to PROJECT_ROOT.

    Returns:
        Absolute Path object.
    """
    return (PROJECT_ROOT / relative_path).resolve()


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as a formatted JSON file.

    Args:
        data: Serializable data.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Args:
        path: Input file path.

    Returns:
        Parsed data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they don't exist.

    Args:
        *dirs: Directory paths to create.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
