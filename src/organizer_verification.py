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


def organizer_batch_verification_passed(
    model: Mapping[str, Any],
    *,
    batch_id: str,
    expected_fingerprint: str,
    project_root: Path,
) -> bool:
    """Gate reference-batch vectors on their own sampled encoder receipt.

    The historical B01 collection retains its original verification record.
    A later batch must explicitly bind a receipt and source-artifact digests;
    an unrelated B01 pass never licenses a new vector collection.
    """

    if batch_id == "aic25-b1-reference":
        return organizer_verification_passed(
            model,
            expected_fingerprint=expected_fingerprint,
            project_root=project_root,
        )
    configured = model.get("batch_verifications", {})
    entry = configured.get(batch_id) if isinstance(configured, Mapping) else None
    if not isinstance(entry, Mapping):
        return False
    record = entry.get("record")
    if not isinstance(record, str) or not record:
        return False
    try:
        payload = json.loads(_path(record, project_root).read_text(encoding="utf-8"))
        return bool(payload.get("verified")) and all(
            (
                payload.get("batch_id") == batch_id,
                payload.get("schema_version") == "b02-clip-encoder-verification-v1",
                payload.get("model_name") == model["model_name"],
                payload.get("pretrained") == model["pretrained"],
                payload.get("encoder_fingerprint") == expected_fingerprint,
                payload.get("manifest_sha256") == entry["manifest_sha256"],
                payload.get("feature_archive_sha256") == entry["feature_archive_sha256"],
                int(payload.get("dimension", 0)) == int(model["embedding_dim"]),
                int(payload.get("sample_count", 0)) >= int(entry["sample_count"]),
                float(payload.get("median_cosine", -1.0))
                >= float(entry["minimum_median_cosine"]),
                float(payload.get("minimum_sample_cosine", -1.0))
                >= float(entry["minimum_sample_cosine"]),
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
