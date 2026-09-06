"""Fingerprint gates for off -> shadow -> active retrieval rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.retrieval.candidate_rescue import validate_rollout_mode


def rollout_config_fingerprint(value: Mapping[str, Any]) -> str:
    """Fingerprint behavior settings while excluding deployment metadata."""

    selected = {
        str(key): item
        for key, item in value.items()
        if key not in {"mode", "tuning_record"}
    }
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_rollout_activation(
    feature: str,
    value: Mapping[str, Any] | None,
    *,
    project_root: str | Path,
) -> None:
    """Reject an active feature unless its locked tuning record matches."""

    config = value or {}
    mode = validate_rollout_mode(config.get("mode", "off"), name=feature)
    if mode != "active":
        return
    record_value = config.get("tuning_record")
    if not isinstance(record_value, str) or not record_value.strip():
        raise ValueError(f"active {feature} requires a tuning_record")
    record_path = Path(record_value)
    if not record_path.is_absolute():
        record_path = Path(project_root) / record_path
    payload = json.loads(record_path.read_text(encoding="utf-8-sig"))
    activation = payload.get("activation_gate") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("feature") != feature
        or payload.get("config_fingerprint") != rollout_config_fingerprint(config)
        or not isinstance(activation, Mapping)
        or activation.get("passed") is not True
        or not isinstance(payload.get("qrels_locked_test_sha256"), str)
        or len(payload["qrels_locked_test_sha256"]) != 64
    ):
        raise ValueError(f"active {feature} tuning record is incompatible")


__all__ = ["rollout_config_fingerprint", "validate_rollout_activation"]
