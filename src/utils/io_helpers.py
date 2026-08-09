"""
I/O Helpers
===========
File I/O, path management, config loading, and data serialization
utilities shared across the project.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import yaml
from dotenv import dotenv_values

from src import CONFIG_DIR, PROJECT_ROOT

_VALID_PROFILES = frozenset({"laptop", "accelerated"})
_SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_LAUNCH_CONTRACT = "run_app-v2"
_SAFE_LAUNCH_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _configured_value(
    name: str,
    dotenv_data: dict[str, Any],
) -> str | None:
    """Return a non-empty process/.env value without mutating either source."""

    value = os.environ.get(name)
    if value in (None, ""):
        value = dotenv_data.get(name)
    if value in (None, ""):
        return None
    return str(value).strip()


def validate_batch_identifier(value: str) -> str:
    """Return a normalized batch identifier or reject unsafe input."""

    if not isinstance(value, str) or not _SAFE_BATCH_ID.fullmatch(value.strip()):
        raise ValueError("batch_id must contain only letters, digits, '.', '_' or '-'")
    return value.strip()


def configured_batch_id(config: Mapping[str, Any]) -> str:
    """Return the selected runtime batch from an already-resolved config."""

    runtime = config.get("runtime", {})
    value = runtime.get("default_batch_id") if isinstance(runtime, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        readiness = config.get("readiness", {})
        batches = readiness.get("batches", {}) if isinstance(readiness, Mapping) else {}
        if isinstance(batches, Mapping) and len(batches) == 1:
            value = str(next(iter(batches)))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("runtime.default_batch_id is required")
    return validate_batch_identifier(value)


def sanitize_launch_identity(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Allowlist the operational launch fields safe for UI and API responses."""

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    canonical = value.get("canonical")
    if isinstance(canonical, bool):
        result["canonical"] = canonical
    for name in ("contract", "profile", "reason_code"):
        item = value.get(name)
        if isinstance(item, str) and _SAFE_LAUNCH_LABEL.fullmatch(item):
            result[name] = item
    batch_id = value.get("batch_id")
    if isinstance(batch_id, str):
        try:
            result["batch_id"] = validate_batch_identifier(batch_id)
        except ValueError:
            pass
    launch_id = value.get("launch_id")
    if isinstance(launch_id, str):
        try:
            result["launch_id"] = str(UUID(launch_id))
        except ValueError:
            pass
    port = value.get("port")
    if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65_535:
        result["port"] = port
    started = value.get("started_at_utc")
    if isinstance(started, str) and len(started) <= 64:
        try:
            datetime.fromisoformat(started.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            result["started_at_utc"] = started
    for name in ("lock_sha256", "runtime_stamp_sha256"):
        item = value.get(name)
        if isinstance(item, str) and _SHA256.fullmatch(item.casefold()):
            result[name] = item.casefold()
    return result


def load_launch_identity(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read and sanitize canonical-launch metadata from the process environment."""

    source = os.environ if environ is None else environ
    selected_batch = configured_batch_id(config)
    profile = str(config.get("profile", "laptop"))
    contract = source.get("AIC_LAUNCH_CONTRACT")
    if contract != _CANONICAL_LAUNCH_CONTRACT:
        return sanitize_launch_identity({
            "canonical": False,
            "contract": "direct",
            "batch_id": selected_batch,
            "profile": profile,
            "reason_code": "noncanonical_launch",
        })

    try:
        launch_id = str(UUID(str(source["AIC_LAUNCH_ID"])))
        port = int(source["AIC_LAUNCH_PORT"])
        batch_id = validate_batch_identifier(str(source["AIC_LAUNCH_BATCH"]))
        launch_profile = str(source["AIC_LAUNCH_PROFILE"])
        started = str(source["AIC_LAUNCHED_AT_UTC"])
        datetime.fromisoformat(started.replace("Z", "+00:00"))
        stamp_hash = str(source["AIC_RUNTIME_STAMP_SHA256"]).casefold()
        lock_hash = str(source["AIC_LAUNCH_LOCK_SHA256"]).casefold()
        if not 1 <= port <= 65_535:
            raise ValueError("invalid port")
        if (
            launch_profile not in _VALID_PROFILES
            or not _SHA256.fullmatch(stamp_hash)
            or not _SHA256.fullmatch(lock_hash)
        ):
            raise ValueError("invalid profile or stamp")
        if batch_id != selected_batch or launch_profile != profile:
            raise ValueError("launch/config mismatch")
    except (KeyError, TypeError, ValueError):
        return sanitize_launch_identity({
            "canonical": False,
            "contract": _CANONICAL_LAUNCH_CONTRACT,
            "batch_id": selected_batch,
            "profile": profile,
            "reason_code": "invalid_launch_identity",
        })
    return sanitize_launch_identity({
        "canonical": True,
        "contract": _CANONICAL_LAUNCH_CONTRACT,
        "launch_id": launch_id,
        "port": port,
        "batch_id": batch_id,
        "profile": launch_profile,
        "started_at_utc": started,
        "lock_sha256": lock_hash,
        "runtime_stamp_sha256": stamp_hash,
    })


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
    configured_profile = _configured_value("AIC_PROFILE", dotenv_data)
    profile = configured_profile or str(config.get("profile", "laptop")).strip()
    if profile not in _VALID_PROFILES:
        allowed = ", ".join(sorted(_VALID_PROFILES))
        raise ValueError(f"AIC_PROFILE must be one of: {allowed}")
    config["profile"] = profile

    runtime = config.get("runtime")
    if runtime is None:
        runtime = {}
        config["runtime"] = runtime
    if not isinstance(runtime, dict):
        raise ValueError("runtime configuration must be a mapping")
    configured_batch = _configured_value("AIC_BATCH_ID", dotenv_data)
    if configured_batch is not None:
        runtime["default_batch_id"] = validate_batch_identifier(configured_batch)
    elif "default_batch_id" in runtime:
        runtime["default_batch_id"] = validate_batch_identifier(
            str(runtime["default_batch_id"])
        )

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

            value = _configured_value(env_name, dotenv_data)
            if value is not None:
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


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write bytes through a same-directory temporary file and atomic replace."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return target
