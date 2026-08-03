"""Persist and validate the organizer encoder compatibility gate."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


def _path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def persist_organizer_verification(
    model: Mapping[str, Any],
    *,
    encoder_fingerprint: str,
    median_cosine: float,
    sample_size: int,
    project_root: Path,
) -> Path:
    """Atomically persist a successful, model-bound compatibility result."""

    minimum = float(model["verification_median_cosine"])
    if sample_size < 1:
        raise ValueError("verification sample_size must be positive")
    if median_cosine < minimum:
        raise ValueError("cannot persist a failed organizer encoder verification")
    target = _path(model["verification_record"], project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verified": True,
        "model_name": str(model["model_name"]),
        "pretrained": str(model["pretrained"]),
        "encoder_fingerprint": encoder_fingerprint,
        "sample_size": sample_size,
        "median_cosine": float(median_cosine),
        "minimum_median_cosine": minimum,
        "verified_at_unix": time.time(),
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def organizer_verification_passed(
    model: Mapping[str, Any],
    *,
    expected_fingerprint: str,
    project_root: Path,
) -> bool:
    """Return true only for a persisted result matching current configuration."""

    if not model.get("enabled", False) or not model.get("verification_record"):
        return False
    target = _path(model["verification_record"], project_root)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return bool(payload.get("verified")) and all(
            (
                payload.get("model_name") == model["model_name"],
                payload.get("pretrained") == model["pretrained"],
                payload.get("encoder_fingerprint") == expected_fingerprint,
                int(payload.get("sample_size", 0))
                >= int(model.get("verification_sample_size", 1)),
                float(payload.get("median_cosine", -1.0))
                >= float(model["verification_median_cosine"]),
                float(payload.get("minimum_median_cosine", -1.0))
                == float(model["verification_median_cosine"]),
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
