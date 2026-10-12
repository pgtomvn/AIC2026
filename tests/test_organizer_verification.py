"""Offline tests for the persisted organizer encoder compatibility gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.runtime as runtime_module
from src.organizer_verification import (
    organizer_batch_verification_passed,
    organizer_verification_passed,
    persist_organizer_verification,
)
from src.utils.io_helpers import load_config


def _model(tmp_path: Path) -> dict:
    return {
        "enabled": True,
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "verification_sample_size": 16,
        "verification_median_cosine": 0.999,
        "verification_record": str(tmp_path / "verification.json"),
    }


def test_persisted_verification_is_bound_to_model_threshold_and_sample(tmp_path: Path) -> None:
    model = _model(tmp_path)
    assert not organizer_verification_passed(
        model, expected_fingerprint="fp-1", project_root=tmp_path
    )

    target = persist_organizer_verification(
        model,
        encoder_fingerprint="fp-1",
        median_cosine=0.9994,
        sample_size=16,
        project_root=tmp_path,
    )

    assert target.is_file()
    assert organizer_verification_passed(
        model, expected_fingerprint="fp-1", project_root=tmp_path
    )
    assert not organizer_verification_passed(
        {**model, "model_name": "other"},
        expected_fingerprint="fp-1",
        project_root=tmp_path,
    )
    assert not organizer_verification_passed(
        {**model, "verification_median_cosine": 0.9995},
        expected_fingerprint="fp-1",
        project_root=tmp_path,
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["verified"] is True
    assert payload["sample_size"] == 16


def test_failed_verification_cannot_be_persisted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed"):
        persist_organizer_verification(
            _model(tmp_path),
            encoder_fingerprint="fp",
            median_cosine=0.998,
            sample_size=16,
            project_root=tmp_path,
        )


def test_b02_verification_requires_matching_batch_and_artifact_hashes(tmp_path: Path) -> None:
    record = tmp_path / "b02-verification.json"
    payload = {
        "schema_version": "b02-clip-encoder-verification-v1",
        "verified": True,
        "batch_id": "aic26-b2",
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "encoder_fingerprint": "fp-1",
        "manifest_sha256": "manifest-digest",
        "feature_archive_sha256": "feature-digest",
        "dimension": 512,
        "sample_count": 6,
        "median_cosine": 0.9999,
        "minimum_sample_cosine": 0.9998,
    }
    record.write_text(json.dumps(payload), encoding="utf-8")
    model = {
        **_model(tmp_path),
        "embedding_dim": 512,
        "batch_verifications": {
            "aic26-b2": {
                "record": str(record),
                "manifest_sha256": "manifest-digest",
                "feature_archive_sha256": "feature-digest",
                "sample_count": 6,
                "minimum_median_cosine": 0.999,
                "minimum_sample_cosine": 0.999,
            }
        },
    }

    assert organizer_batch_verification_passed(
        model, batch_id="aic26-b2", expected_fingerprint="fp-1", project_root=tmp_path
    )
    assert not organizer_batch_verification_passed(
        model, batch_id="other", expected_fingerprint="fp-1", project_root=tmp_path
    )
    assert not organizer_batch_verification_passed(
        model, batch_id="aic26-b2", expected_fingerprint="fp-2", project_root=tmp_path
    )
    payload["feature_archive_sha256"] = "wrong-digest"
    record.write_text(json.dumps(payload), encoding="utf-8")
    assert not organizer_batch_verification_passed(
        model, batch_id="aic26-b2", expected_fingerprint="fp-1", project_root=tmp_path
    )


def test_runtime_uses_cpu_for_verified_organizer_query_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    config["models"]["organizer_clip"]["query_device"] = "cpu"
    monkeypatch.setattr(
        runtime_module, "organizer_batch_verification_passed", lambda *a, **k: True
    )

    pipeline = runtime_module.build_pipeline(config)
    organizer = next(channel for channel in pipeline.dense_channels if channel.name == "organizer_clip")

    assert organizer.encoder.device == "cpu"
    accelerated = [
        channel for channel in pipeline.dense_channels if channel.name == "accelerated_visual"
    ]
    assert not accelerated
