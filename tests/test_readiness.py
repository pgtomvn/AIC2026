"""Offline tests for sanitized, batch-aware runtime readiness."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
import requests

import src.readiness as readiness_module
from src.readiness import (
    ChannelApplicability,
    ChannelReadiness,
    ChannelState,
    ReadinessService,
    WorkflowState,
    aggregate_channel_health,
)


class _Response:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _HTTP:
    def __init__(self, *, organizer_count: int = 3, fail: bool = False) -> None:
        self.organizer_count = organizer_count
        self.fail = fail
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        if self.fail:
            raise requests.ConnectionError("secret-token C:/private/archive.zip")
        if url.endswith("/api/tags") and method == "GET":
            return _Response(
                200,
                {
                    "models": [
                        {
                            "name": "qwen2.5vl:3b-q4_K_M",
                            "digest": "fb90415cde1e-test",
                        }
                    ]
                },
            )
        if url.endswith("/api/show") and method == "POST":
            return _Response(200, {"capabilities": ["completion", "vision"]})
        if url.endswith("/collections/organizer-v1") and method == "GET":
            return _Response(
                200,
                {
                    "result": {
                        "status": "green",
                        "optimizer_status": "ok",
                        "points_count": 99,
                        "config": {"params": {"vectors": {"size": 512}}},
                    }
                },
            )
        if url.endswith("/collections/organizer-v1/points/count"):
            return _Response(200, {"result": {"count": self.organizer_count}})
        if "/collections/" in url:
            return _Response(404, {})
        if url.endswith("/indexes/metadata-v1/search"):
            return _Response(
                200,
                {
                    "estimatedTotalHits": 1,
                    "facetDistribution": {"batch_id": {"aic25-b1-reference": 12}},
                },
            )
        if "/indexes/" in url:
            return _Response(200, {"estimatedTotalHits": 0})
        raise AssertionError(f"unexpected readiness request: {method} {url}")


def _artifact_database(path: Path) -> None:
    first_image = path.parent / "frame-1.jpg"
    second_image = path.parent / "frame-2.jpg"
    other_image = path.parent / "other.webp"
    first_image.write_bytes(b"frame-1")
    second_image.write_bytes(b"frame-2")
    other_image.write_bytes(b"other")
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE canonical_frames (
                batch_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                image_uri TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO canonical_frames VALUES (?, ?, ?)",
            [
                ("aic25-b1-reference", "L01_V001", str(first_image)),
                ("aic25-b1-reference", "L01_V001", str(second_image)),
                ("other", "L99_V999", str(other_image)),
            ],
        )


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "profile": "laptop",
        "readiness": {
            "cache_seconds": 10,
            "batches": {
                "aic25-b1-reference": {
                    "expected_frames": 2,
                    "expected_videos": 1,
                    "workflows": {
                        "kis": ["organizer_clip", "metadata"],
                        "trake": ["organizer_clip"],
                        "qa": [
                            "organizer_clip",
                            "metadata",
                            "canonical_images",
                            "ollama_vlm",
                        ],
                    },
                    "channels": {
                        "organizer_clip": "required",
                        "metadata": "required",
                        "canonical_images": "required",
                        "ollama_vlm": "required",
                        "openrouter_vlm": "optional",
                        "visual": "optional",
                        "semantic_text": "optional",
                        "ocr": "optional",
                        "asr": "optional",
                    },
                }
            },
            "native_defaults": {
                "workflows": {
                    "kis": ["visual"],
                    "qa": ["visual", "ollama_vlm"],
                    "trake": ["visual"],
                },
                "channels": {"visual": "required", "ollama_vlm": "required"},
            },
        },
        "data": {"artifact_db": "artifacts.sqlite3"},
        "services": {
            "ollama_vlm": {
                "url": "http://127.0.0.1:11434",
                "model": "qwen2.5vl:3b-q4_K_M",
                "expected_digest_prefix": "fb90415cde1e",
                "connect_timeout_seconds": 1,
            },
            "openrouter_vlm": {"api_key": None},
            "qdrant": {
                "url": "http://qdrant.test",
                "collections": {
                    "native_frames": "native_{batch}",
                    "organizer_frames": "organizer-v1",
                    "accelerated_frames": "accelerated_{batch}",
                    "text_segments": "text_{batch}",
                },
            },
            "meilisearch": {
                "url": "http://meili.test",
                "indexes": {"ocr": "ocr-v1", "asr": "asr-v1", "metadata": "metadata-v1"},
            },
        },
        "models": {
            "visual": {"embedding_dim": 768, "model_name": "visual", "revision": "r1"},
            "organizer_clip": {
                "enabled": True,
                "embedding_dim": 512,
                "model_name": "ViT-B-32-quickgelu",
                "pretrained": "openai",
                "verification_record": "verified.json",
                "verification_sample_size": 16,
                "verification_median_cosine": 0.999,
            },
            "accelerated_visual": {
                "enabled": True,
                "enabled_profiles": ["accelerated"],
                "embedding_dim": 768,
            },
            "text": {"embedding_dim": 768, "model_name": "text", "revision": "r2"},
        },
        "retrieval": {"rrf_k": 60, "candidate_depths": {}, "channel_weights": {}},
    }


def test_reference_readiness_is_batch_scoped_and_optional_channels_do_not_degrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifact_database(tmp_path / "artifacts.sqlite3")
    http = _HTTP()
    monkeypatch.setattr(readiness_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(readiness_module, "organizer_verification_passed", lambda *_a, **_k: True)

    service = ReadinessService(_config(tmp_path), project_root=tmp_path, http_client=http)
    report = service.check("aic25-b1-reference")
    channels = {item.name: item for item in report.channels}

    assert report.kis.state is WorkflowState.READY
    assert report.qa.state is WorkflowState.DEGRADED
    assert service.workflow_readiness(report, "trake").state is WorkflowState.READY
    assert report.selected_batch_id == "aic25-b1-reference"
    assert report.launch_identity["canonical"] is False
    assert channels["organizer_clip"].state is ChannelState.READY
    assert channels["organizer_clip"].count == 3
    assert channels["metadata"].count == 12
    assert channels["canonical_images"].count == 2
    assert channels["visual"].state is ChannelState.NOT_PROVISIONED
    assert channels["ollama_vlm"].state is ChannelState.READY
    organizer_count = next(call for call in http.calls if call[1].endswith("/points/count"))
    assert organizer_count[2]["json"]["filter"]["must"][0]["match"]["value"] == (
        "aic25-b1-reference"
    )


def test_aggregate_health_uses_required_channel_and_exposes_only_safe_fields() -> None:
    statuses = aggregate_channel_health(
        (
            ChannelReadiness(
                "visual",
                ChannelApplicability.REQUIRED,
                ChannelState.INCOMPATIBLE_INDEX,
                "vector_dimension_mismatch",
                resource_name="secret-collection",
            ),
            ChannelReadiness(
                "organizer_clip",
                ChannelApplicability.OPTIONAL,
                ChannelState.READY,
                count=100,
            ),
            ChannelReadiness(
                "ocr",
                ChannelApplicability.OPTIONAL,
                ChannelState.READY,
                count=20,
            ),
            ChannelReadiness(
                "asr",
                ChannelApplicability.OPTIONAL,
                ChannelState.SERVICE_UNAVAILABLE,
                "meilisearch_unavailable",
            ),
        )
    )

    assert statuses == {
        "visual_vector": {
            "state": "incompatible_index",
            "reason_code": "vector_dimension_mismatch",
            "count": None,
        },
        "ocr_text": {"state": "ready", "reason_code": None, "count": 20},
        "asr_transcript": {
            "state": "service_unavailable",
            "reason_code": "meilisearch_unavailable",
            "count": None,
        },
        "metadata": {
            "state": "not_provisioned",
            "reason_code": "channel_not_applicable",
            "count": None,
        },
    }
    assert "secret-collection" not in str(statuses)


def test_trake_is_blocked_when_required_dense_space_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifact_database(tmp_path / "artifacts.sqlite3")
    monkeypatch.setattr(readiness_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(readiness_module, "organizer_verification_passed", lambda *_a, **_k: True)
    service = ReadinessService(
        _config(tmp_path), project_root=tmp_path, http_client=_HTTP(organizer_count=0)
    )

    report = service.check("aic25-b1-reference")
    trake = service.workflow_readiness(report, "trake")

    assert trake.state is WorkflowState.BLOCKED
    assert trake.unavailable_channels == ("organizer_clip",)


def test_trake_is_degraded_but_dense_retrieval_remains_usable_when_thumbnails_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "artifacts.sqlite3"
    _artifact_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM canonical_frames WHERE batch_id = ? AND image_uri LIKE ?",
            ("aic25-b1-reference", "%frame-2.jpg"),
        )
    monkeypatch.setattr(readiness_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(readiness_module, "organizer_verification_passed", lambda *_a, **_k: True)
    service = ReadinessService(_config(tmp_path), project_root=tmp_path, http_client=_HTTP())

    report = service.check("aic25-b1-reference")
    channels = {item.name: item for item in report.channels}
    trake = service.workflow_readiness(report, "trake")

    assert channels["organizer_clip"].usable is True
    assert channels["canonical_images"].reason_code == "canonical_image_coverage_incomplete"
    assert trake.state is WorkflowState.DEGRADED
    assert trake.required_channels == ("organizer_clip",)
    assert trake.unavailable_channels == ("canonical_images",)


def test_global_readiness_defers_unresolvable_media_to_candidate_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "artifacts.sqlite3"
    _artifact_database(database)
    secret_missing_path = tmp_path / "private-secret-thumbnail.jpg"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE canonical_frames SET image_uri = ? WHERE batch_id = ? AND image_uri LIKE ?",
            (str(secret_missing_path), "aic25-b1-reference", "%frame-1.jpg"),
        )
    monkeypatch.setattr(readiness_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(readiness_module, "organizer_verification_passed", lambda *_a, **_k: True)
    service = ReadinessService(_config(tmp_path), project_root=tmp_path, http_client=_HTTP())

    report = service.check("aic25-b1-reference")
    channels = {item.name: item for item in report.channels}
    trake = service.workflow_readiness(report, "trake")

    assert channels["organizer_clip"].usable is True
    assert channels["canonical_images"].state is ChannelState.READY
    assert channels["canonical_images"].reason_code is None
    assert trake.state is WorkflowState.READY
    assert trake.unavailable_channels == ()
    assert "private-secret-thumbnail" not in str(report.to_dict())


def test_canonical_image_readiness_uses_strict_configured_coverage_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "artifacts.sqlite3"
    _artifact_database(database)
    with sqlite3.connect(database) as connection:
        connection.executemany(
            "INSERT INTO canonical_frames VALUES (?, ?, ?)",
            (
                ("aic25-b1-reference", "L01_V001", str(tmp_path / f"frame-{i}.jpg"))
                for i in range(3, 100)
            ),
        )
    config = _config(tmp_path)
    specification = config["readiness"]["batches"]["aic25-b1-reference"]
    specification.pop("expected_frames")
    specification["expected_canonical_frames"] = 100
    specification["canonical_image_min_coverage"] = 0.99
    monkeypatch.setattr(readiness_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        readiness_module, "organizer_verification_passed", lambda *_a, **_k: True
    )
    exactly_99 = ReadinessService(config, project_root=tmp_path, http_client=_HTTP())
    report = exactly_99.check("aic25-b1-reference")
    channels = {item.name: item for item in report.channels}

    assert channels["canonical_images"].count == 99
    assert channels["canonical_images"].coverage == pytest.approx(0.99)
    assert channels["canonical_images"].reason_code == "canonical_image_coverage_incomplete"

    specification["expected_canonical_frames"] = 100
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO canonical_frames VALUES (?, ?, ?)",
            ("aic25-b1-reference", "L01_V001", str(tmp_path / "frame-100.jpg")),
        )
    complete = ReadinessService(config, project_root=tmp_path, http_client=_HTTP())
    channels = {
        item.name: item for item in complete.check("aic25-b1-reference").channels
    }
    assert channels["canonical_images"].state is ChannelState.READY

    specification["expected_canonical_frames"] = 101
    above_99 = ReadinessService(config, project_root=tmp_path, http_client=_HTTP())
    channels = {
        item.name: item for item in above_99.check("aic25-b1-reference").channels
    }
    assert channels["canonical_images"].coverage == pytest.approx(100 / 101)
    assert channels["canonical_images"].state is ChannelState.READY


def test_readiness_never_exposes_raw_service_errors_or_credentials(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = ReadinessService(
        config,
        project_root=tmp_path,
        http_client=_HTTP(fail=True),
        launch_identity={
            "canonical": False,
            "contract": "direct",
            "batch_id": "aic25-b1-reference",
            "profile": "laptop",
            "reason_code": "noncanonical_launch",
            "api_key": "launch-secret",
            "archive_path": "C:/private/keyframes.zip",
        },
    )

    payload = service.check("aic25-b1-reference").to_dict()
    rendered = str(payload)

    assert "secret-token" not in rendered
    assert "private/archive" not in rendered
    assert "test-only-key" not in rendered
    assert "launch-secret" not in rendered
    assert "keyframes.zip" not in rendered
    assert "qdrant_unavailable" in rendered
    assert "meilisearch_unavailable" in rendered


def test_readiness_cache_and_retrieval_fingerprint_invalidate_on_resource_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifact_database(tmp_path / "artifacts.sqlite3")
    http = _HTTP(organizer_count=3)
    monkeypatch.setattr(readiness_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(readiness_module, "organizer_verification_passed", lambda *_a, **_k: True)
    service = ReadinessService(_config(tmp_path), project_root=tmp_path, http_client=http)

    first = service.retrieval_fingerprint(("aic25-b1-reference",))
    call_count = len(http.calls)
    assert service.retrieval_fingerprint(("aic25-b1-reference",)) == first
    assert len(http.calls) == call_count

    http.organizer_count = 4
    service.invalidate()
    second = service.retrieval_fingerprint(("aic25-b1-reference",))
    assert second != first


def test_invalid_batch_and_mode_are_rejected_without_network(tmp_path: Path) -> None:
    http = _HTTP()
    service = ReadinessService(_config(tmp_path), project_root=tmp_path, http_client=http)

    with pytest.raises(ValueError, match="safe non-empty"):
        service.check("bad batch")
    with pytest.raises(ValueError, match="mode must be"):
        service.check("valid-batch", mode="network")
    with pytest.raises(ValueError, match="at least one"):
        service.retrieval_fingerprint(())
    assert http.calls == []


def test_app_readiness_exposes_optional_query_translation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifact_database(tmp_path / "artifacts.sqlite3")
    config = _config(tmp_path)
    config["models"]["query_translation"] = {
        "enabled": True,
        "converted_path": "models/translation",
        "tokenizer_path": "models/tokenizer",
        "asset_manifest": "checksums/query-translation.sha256",
    }
    monkeypatch.setattr(
        readiness_module,
        "_module_available",
        lambda name: name != "ctranslate2",
    )
    monkeypatch.setattr(
        readiness_module, "organizer_verification_passed", lambda *_a, **_k: True
    )

    report = ReadinessService(
        config, project_root=tmp_path, http_client=_HTTP()
    ).check("aic25-b1-reference")
    check = next(item for item in report.checks if item.name == "query_translation")

    assert check.state is ChannelState.BLOCKED_DEPENDENCY
    assert check.reason_code == "translator_dependency_missing"
    assert report.kis.state is WorkflowState.READY


def test_query_translation_readiness_is_side_effect_free_and_asset_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifact_database(tmp_path / "artifacts.sqlite3")
    config = _config(tmp_path)
    config["models"]["query_translation"] = {
        "enabled": True,
        "converted_path": "models/translation",
        "tokenizer_path": "models/tokenizer",
        "asset_manifest": "checksums/query-translation.sha256",
    }
    for relative in (
        "models/translation/model.bin",
        "models/translation/config.json",
        "models/tokenizer/source.spm",
        "models/tokenizer/target.spm",
        "checksums/query-translation.sha256",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    monkeypatch.setattr(readiness_module, "_module_available", lambda _name: True)

    report = ReadinessService(
        config, project_root=tmp_path, http_client=_HTTP()
    ).check("aic25-b1-reference")
    check = next(item for item in report.checks if item.name == "query_translation")

    assert check.state is ChannelState.READY
