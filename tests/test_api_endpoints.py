"""Offline tests for FastAPI transport using only injected runtime fakes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from src.api import create_app
from src.contracts import QueryRequest
from src.readiness import (
    ChannelApplicability,
    ChannelReadiness,
    ChannelState,
    ReadinessReport,
    WorkflowReadiness,
    WorkflowState,
)
from src.retrieval.textual_kis import TextualKISPipeline


class FakePipeline:
    """Retrieval fake that records validated requests passed by the API."""

    def __init__(
        self,
        *,
        search_hits: list[dict[str, Any]] | None = None,
        refine_hits: list[dict[str, Any]] | None = None,
        channel_errors: dict[str, str] | None = None,
        refine_error: Exception | None = None,
    ) -> None:
        self.search_hits = list(search_hits or [])
        self.refine_hits = list(refine_hits or [])
        self.last_channel_errors = dict(channel_errors or {})
        self.last_channel_statuses = {
            name: {
                "state": "service_unavailable",
                "reason_code": "channel_unavailable",
            }
            for name in self.last_channel_errors
        }
        self.refine_error = refine_error
        self.search_calls: list[tuple[QueryRequest, int]] = []
        self.search_refine_raw: list[bool] = []
        self.refine_calls: list[dict[str, Any]] = []

    def search(
        self, request: QueryRequest, *, limit: int, refine_raw: bool = True
    ) -> list[dict[str, Any]]:
        self.search_calls.append((request, limit))
        self.search_refine_raw.append(refine_raw)
        return [dict(item) for item in self.search_hits]

    def _plan(self, request: QueryRequest):
        return TextualKISPipeline._plan(request)

    def refine(
        self,
        request: QueryRequest,
        *,
        positive_ids,
        negative_ids,
        limit: int,
    ) -> list[dict[str, Any]]:
        if self.refine_error is not None:
            raise self.refine_error
        self.refine_calls.append(
            {
                "request": request,
                "positive_ids": positive_ids,
                "negative_ids": negative_ids,
                "limit": limit,
            }
        )
        return [dict(item) for item in self.refine_hits]


@dataclass
class FakeStatus:
    batch_id: str
    total_videos: int
    active_videos: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "total_videos": self.total_videos,
            "active_videos": self.active_videos,
            "failed_videos": 1,
            "processing_videos": 1,
            "pending_videos": 0,
            "readiness": self.active_videos / self.total_videos,
            "completed_stage_timings": 10,
            "average_seconds_per_active_video": 4.0,
            "eta_seconds": 8.0,
            "stage_counts": {},
        }


class FakeLedger:
    def __init__(self, *, active: bool = True, status_error: Exception | None = None) -> None:
        self.active = active
        self.status_error = status_error
        self.status_calls: list[str] = []
        self.active_calls: list[str] = []

    def status_summary(self, batch_id: str) -> FakeStatus:
        self.status_calls.append(batch_id)
        if self.status_error is not None:
            raise self.status_error
        return FakeStatus(batch_id, 10, 10 if self.active else 8)

    def is_batch_active(self, batch_id: str) -> bool:
        self.active_calls.append(batch_id)
        return self.active

    def is_batch_registered(self, _batch_id: str) -> bool:
        return False

    def core_ingestion_projection(self, batch_id: str, **kwargs):
        assert batch_id
        if kwargs["preload_window_seconds"] is not None and kwargs["preload_window_seconds"] <= 0:
            raise ValueError("preload_window_seconds must be positive when supplied")
        return SimpleNamespace(
            as_dict=lambda: {
                "sample_target_videos": kwargs["sample_videos"],
                "sampled_videos": 10,
                "projection_ready": True,
                "projected_core_seconds": 40.0,
                "preload_window_seconds": kwargs["preload_window_seconds"],
                "preload_budget_seconds": 48.0,
                "preload_deadline_fraction": kwargs["deadline_fraction"],
                "fits_preload_budget": True,
                "accelerated_profile_mandatory": False,
            }
        )


class FakeArtifactRepository:
    def frame_index_mappings(self, batch_id: str) -> dict[str, dict[int, int]]:
        assert batch_id == "aic25-b1"
        return {
            "L21_V001": {7: 181},
            "L21_V002": {1: 240},
        }


def _context(tmp_path: Path, pipeline: FakePipeline, ledger: FakeLedger | None = None):
    return SimpleNamespace(
        config={
            "competition": {"output_dir": str(tmp_path)},
            "data": {
                "scratch": str(tmp_path / "scratch"),
                "scratch_limit_gib": 12,
                "free_space_reserve_gib": 30,
            },
        },
        pipeline=pipeline,
        ledger=ledger or FakeLedger(),
        artifact_repository=FakeArtifactRepository(),
    )


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def test_b02_ordinal_image_and_neighbor_routes_preserve_collisions(tmp_path):
    context = _context(tmp_path, FakePipeline())
    frames = {n: SimpleNamespace(
        batch_id="aic26-b2", video_id="M08_V001", keyframe_ordinal=n,
        source_frame_idx=4751, timestamp_ms=190040,
    ) for n in (72, 73)}
    calls = []

    def get(batch, video, ordinal):
        calls.append((batch, video, ordinal))
        return frames.get(ordinal) if (batch, video) == ("aic26-b2", "M08_V001") else None

    context.artifact_repository = SimpleNamespace(get_ordinal_frame=get)
    context.qa_image_resolver = SimpleNamespace(resolve=lambda frame: SimpleNamespace(
        data=f"jpeg-{frame.keyframe_ordinal}".encode(), mime_type="image/jpeg",
        content_sha256=str(frame.keyframe_ordinal),
    ))
    app = create_app(context)
    image = _endpoint(app, "/v1/keyframes/{batch_id}/{video_id}/{keyframe_ordinal}/image", "GET")
    assert image("aic26-b2", "M08_V001", 72).body == b"jpeg-72"
    assert image("aic26-b2", "M08_V001", 73).body == b"jpeg-73"
    with pytest.raises(HTTPException) as missing:
        image("aic26-b2", "M08_V001", 74)
    assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as invalid:
        image("aic25-b1", "M08_V001", 72)
    assert invalid.value.status_code == 422
    neighbors = _endpoint(app, "/v1/keyframes/{batch_id}/{video_id}/{keyframe_ordinal}/neighbors", "GET")
    values = neighbors("aic26-b2", "M08_V001", 72, 1)
    assert [row["keyframe_ordinal"] for row in values["neighbors"]] == [72, 73]
    assert len({row["thumbnail_url"] for row in values["neighbors"]}) == 2
    assert "image_uri" not in str(values)
    with pytest.raises(HTTPException) as radius:
        neighbors("aic26-b2", "M08_V001", 72, 200)
    assert radius.value.status_code == 422


def test_edit_preview_does_not_search_and_confirmation_never_inferred(tmp_path):
    from unittest.mock import Mock

    pipeline = FakePipeline()
    context = _context(tmp_path, pipeline)
    logger = Mock(mode="shadow")
    context.interaction_logger = logger
    context.artifact_repository = SimpleNamespace(get_frame=lambda *a: object())
    app = create_app(context)
    preview = _endpoint(app, "/v1/kis/preview-edit", "POST")
    result = preview({"query": "a red shirt", "instruction": "replace red shirt with blue shirt"})
    assert result["draft_text"] == "a blue shirt" and not pipeline.search_calls
    logger.record_confirmation.assert_not_called()
    confirm = _endpoint(app, "/v1/kis/confirm-answer", "POST")
    request = QueryRequest("q", "a blue shirt", interaction_session_id="s1", interaction_turn=2,
                           event_source="benchmark")
    payload = {"request": request.to_dict(), "frame": {"batch_id": "b", "video_id": "v", "source_frame_idx": 0},
               "elapsed_ms": 1200, "confirmed": True}
    assert confirm(payload) == {"recorded": True, "human_label": False}
    with pytest.raises(HTTPException):
        confirm({**payload, "confirmed": False})
    assert logger.record_confirmation.call_count == 1


def test_interactive_feedback_contract_and_safe_proof(tmp_path: Path) -> None:
    pipeline = FakePipeline(refine_hits=[{
        "batch_id": "b", "video_id": "L21_V001", "source_frame_idx": 0,
        "evidence": [{"channel": "temporal_sequence", "temporal_steps": [
            {"step": 0, "source_frame_idx": 0, "timestamp_ms": 0, "path": "private"},
            {"step": 1, "source_frame_idx": 10, "timestamp_ms": 1000},
        ]}],
    }])
    context = _context(tmp_path, pipeline)
    route = _endpoint(create_app(context), "/v1/kis/refine", "POST")
    value = route({"query_id": "q", "original_text": "người đang đi", "batch_filters": ["b"],
                   "reference_frame": {"batch_id": "b", "video_id": "L21_V001", "source_frame_idx": 0}})
    assert pipeline.refine_calls[0]["request"].reference_frame.source_frame_idx == 0
    assert value["hits"][0]["evidence"]["items"][0]["temporal_steps"][0] == {
        "step": 0, "source_frame_idx": 0, "timestamp_ms": 0,
    }


def test_evidence_api_caps_budget_and_honors_rollback(tmp_path: Path) -> None:
    context = _context(tmp_path, FakePipeline())
    context.artifact_repository = SimpleNamespace(
        get_frame=lambda *args: SimpleNamespace(timestamp_ms=1000),
        text_segments_in_range=lambda *args, **kwargs: (),
    )
    route = _endpoint(create_app(context), "/v1/frames/{batch_id}/{video_id}/{source_frame_idx}/evidence", "GET")
    assert route("b", "v", 0)["anchor"]["timestamp_ms"] == 1000
    with pytest.raises(HTTPException) as invalid:
        route("b", "v", 0, 201)
    assert invalid.value.status_code == 422
    context.config["retrieval"] = {"interactive": {"evidence_enabled": False}}
    with pytest.raises(HTTPException) as disabled:
        route("b", "v", 0)
    assert disabled.value.status_code == 409


def test_evidence_api_uses_indexed_text_channels(tmp_path: Path) -> None:
    from unittest.mock import Mock

    pipeline = FakePipeline()
    ocr_store = SimpleNamespace(search=Mock(return_value=[{
        "id": "ocr-segment", "payload": {
            "batch_id": "b", "video_id": "v", "kind": "ocr",
            "start_ms": 0, "end_ms": 0, "source_frame_idx": 0,
            "text_nfc": "VISIBLE SIGN",
        },
    }]))
    pipeline.lexical_channels = (SimpleNamespace(name="ocr", store=ocr_store),)
    context = _context(tmp_path, pipeline)
    context.artifact_repository = SimpleNamespace(
        get_frame=lambda *args: SimpleNamespace(timestamp_ms=0),
        text_segments_in_range=lambda *args, **kwargs: (),
    )

    route = _endpoint(create_app(context), "/v1/frames/{batch_id}/{video_id}/{source_frame_idx}/evidence", "GET")
    response = route("b", "v", 0)

    assert [item["text"] for item in response["segments"]] == ["VISIBLE SIGN"]
    assert ocr_store.search.call_args.kwargs["top_k"] == 100


def test_search_validates_request_and_returns_editable_explicit_plan(tmp_path: Path) -> None:
    hit = {
        "rank": 1,
        "video_id": "L21_V001",
        "source_frame_idx": 181,
        "score": 0.5,
    }
    pipeline = FakePipeline(search_hits=[hit], channel_errors={"asr": "not indexed"})
    endpoint = _endpoint(create_app(_context(tmp_path, pipeline)), "/v1/kis/search", "POST")

    response = endpoint(
        {
            "query_id": "kis-1",
            "original_text": "người rời quán",
            "mode": "interactive",
            "batch_filters": ["aic25-b1"],
            "limit": 25,
            "before_clause": "người thanh toán",
            "current_clause": "người rời quán",
            "after_clause": "xe chạy đi",
        }
    )

    request, limit = pipeline.search_calls[0]
    assert isinstance(request, QueryRequest)
    assert request.batch_filters == ("aic25-b1",)
    assert limit == 25
    assert pipeline.search_refine_raw == [True]
    assert response["query_id"] == "kis-1"
    assert response["hits"][0]["rank"] == 1
    assert response["hits"][0]["video_id"] == hit["video_id"]
    assert response["hits"][0]["source_frame_idx"] == hit["source_frame_idx"]
    assert response["hits"][0]["fused_score"] == hit["score"]
    assert response["channel_errors"] == {"asr": "channel_unavailable"}
    assert response["query_plan"]["before"] == "người thanh toán"
    assert response["query_plan"]["current"] == "người rời quán"
    assert response["query_plan"]["after"] == "xe chạy đi"


def test_search_defaults_and_validates_batches_before_retrieval(tmp_path: Path) -> None:
    pipeline = FakePipeline(search_hits=[{"video_id": "L21_V001", "source_frame_idx": 1}])
    context = _context(tmp_path, pipeline)
    context.selected_batch_id = "batch-2"
    context.validate_batch_filters = lambda values=None: tuple(values or ("batch-2",))
    endpoint = _endpoint(create_app(context), "/v1/kis/search", "POST")

    endpoint({"query_id": "kis-default-batch", "original_text": "query"})

    assert pipeline.search_calls[0][0].batch_filters == ("batch-2",)

    def reject(_values=None):
        raise ValueError("unknown batch filter")

    context.validate_batch_filters = reject
    with pytest.raises(HTTPException) as raised:
        endpoint({"query_id": "kis-invalid-batch", "original_text": "query"})
    assert raised.value.status_code == 422
    assert len(pipeline.search_calls) == 1


def test_search_rejects_control_only_text_before_retrieval(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    endpoint = _endpoint(create_app(_context(tmp_path, pipeline)), "/v1/kis/search", "POST")

    with pytest.raises(HTTPException) as raised:
        endpoint({"query_id": "malformed", "original_text": "\x00\u202e\x01"})

    assert raised.value.status_code == 422
    assert raised.value.detail["code"] == "kis_request_invalid"
    assert pipeline.search_calls == []


def test_search_reports_channel_statuses_and_allows_mixed_empty_results(
    tmp_path: Path,
) -> None:
    pipeline = FakePipeline(channel_errors={"asr": "unavailable"})
    pipeline.last_channel_statuses["metadata"] = {
        "state": "ready",
        "reason_code": "query_completed",
    }
    endpoint = _endpoint(create_app(_context(tmp_path, pipeline)), "/v1/kis/search", "POST")

    response = endpoint({"query_id": "empty-mixed", "original_text": "không có kết quả"})

    assert response["hits"] == []
    assert response["channel_statuses"]["metadata"]["state"] == "ready"
    assert response["channel_statuses"]["asr"]["state"] == "service_unavailable"


def test_warmup_gates_search_and_exposes_runtime_health(tmp_path: Path) -> None:
    class _Warmup:
        def __init__(self) -> None:
            self.started = 0

        def start(self) -> bool:
            self.started += 1
            return True

        @staticmethod
        def snapshot() -> dict[str, Any]:
            return {"state": "warming", "search_usable": False, "channels": []}

    context = _context(tmp_path, FakePipeline())
    context.warmup = _Warmup()
    app = create_app(context)
    search = _endpoint(app, "/v1/kis/search", "POST")
    warmup = _endpoint(app, "/v1/runtime/warmup", "GET")
    ready = _endpoint(app, "/readyz", "GET")

    assert warmup()["state"] == "warming"
    with pytest.raises(HTTPException) as search_error:
        search({"query_id": "warming", "original_text": "query"})
    assert search_error.value.status_code == 503
    assert search_error.value.detail["code"] == "runtime_warming"
    assert search_error.value.headers == {"Retry-After": "2"}
    with pytest.raises(HTTPException) as ready_error:
        ready()
    assert ready_error.value.status_code == 503


def test_warmup_start_releases_local_vlm_and_restarts_evicted_runtime(
    tmp_path: Path,
) -> None:
    class _Warmup:
        def __init__(self) -> None:
            self.state = "evicted"
            self.started = 0

        def start(self) -> bool:
            self.started += 1
            self.state = "warming"
            return True

        def snapshot(self) -> dict[str, Any]:
            return {
                "state": self.state,
                "search_usable": False,
                "channels": [],
            }

    class _Reasoner:
        def __init__(self) -> None:
            self.unloaded = 0

        def unload(self) -> None:
            self.unloaded += 1

    context = _context(tmp_path, FakePipeline())
    context.warmup = _Warmup()
    context.qa_reasoner = _Reasoner()
    endpoint = _endpoint(create_app(context), "/v1/runtime/warmup/start", "POST")

    assert endpoint()["state"] == "warming"
    assert context.warmup.started == 1
    assert context.qa_reasoner.unloaded == 1


def test_workflow_gate_unlocks_kis_while_trake_is_still_warming(tmp_path: Path) -> None:
    class _Warmup:
        @staticmethod
        def start() -> bool:
            return False

        @staticmethod
        def snapshot() -> dict[str, Any]:
            return {
                "schema_version": 2,
                "state": "degraded",
                "search_usable": True,
                "workflows": {
                    "kis": {"state": "ready", "usable": True},
                    "qa": {"state": "ready", "usable": True},
                    "trake": {
                        "state": "warming",
                        "usable": False,
                        "reason_code": "trajectory_cache_warming",
                    },
                },
                "channels": [],
            }

    pipeline = FakePipeline()
    context = _context(tmp_path, pipeline)
    context.warmup = _Warmup()
    app = create_app(context)
    search = _endpoint(app, "/v1/kis/search", "POST")
    ready = _endpoint(app, "/readyz", "GET")

    assert search({"query_id": "staged", "original_text": "query"})["hits"] == []
    assert ready()["workflow"] == "kis"
    with pytest.raises(HTTPException) as trake_ready:
        ready("trake")
    assert trake_ready.value.status_code == 503
    assert trake_ready.value.detail["code"] == "runtime_warming"
    assert trake_ready.value.detail["workflow"] == "trake"


def test_qa_answer_readiness_reuses_published_warmup_report(tmp_path: Path) -> None:
    report = SimpleNamespace(
        batch_id="aic25-b1-reference",
        channels=(
            SimpleNamespace(name="canonical_images", usable=True),
            SimpleNamespace(name="ollama_vlm", usable=True),
        ),
    )

    class _Warmup:
        @staticmethod
        def readiness_report() -> Any:
            return report

        @staticmethod
        def snapshot() -> dict[str, Any]:
            return {"state": "ready", "search_usable": True, "workflows": {}}

    context = _context(tmp_path, FakePipeline())
    context.selected_batch_id = "aic25-b1-reference"
    context.warmup = _Warmup()
    context.readiness_service = SimpleNamespace(
        check=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("published readiness must avoid a synchronous rescan")
        )
    )
    ready = _endpoint(create_app(context), "/readyz", "GET")

    assert ready("qa", "answer") == {
        "status": "ready",
        "workflow": "qa",
        "stage": "answer",
    }


def test_readiness_endpoint_reuses_published_warmup_report(tmp_path: Path) -> None:
    workflow = WorkflowReadiness(WorkflowState.READY, (), ())
    report = ReadinessReport(
        profile="laptop",
        batch_id="aic25-b1-reference",
        selected_batch_id="aic25-b1-reference",
        launch_identity={},
        mode="app",
        kis=workflow,
        qa=workflow,
        channels=(
            ChannelReadiness(
                "canonical_images", ChannelApplicability.REQUIRED, ChannelState.READY
            ),
            ChannelReadiness(
                "ollama_vlm", ChannelApplicability.REQUIRED, ChannelState.READY
            ),
        ),
        checks=(),
        checked_at_unix=1.0,
    )

    class _Warmup:
        @staticmethod
        def readiness_report() -> ReadinessReport:
            return report

        @staticmethod
        def snapshot() -> dict[str, Any]:
            return {"state": "ready", "search_usable": True, "workflows": {}}

    context = _context(tmp_path, FakePipeline())
    context.selected_batch_id = "aic25-b1-reference"
    context.warmup = _Warmup()
    context.readiness_service = SimpleNamespace(
        check=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("published readiness must avoid a synchronous rescan")
        )
    )
    endpoint = _endpoint(create_app(context), "/v1/readiness", "GET")

    payload = endpoint()

    assert payload["batch_id"] == "aic25-b1-reference"
    assert payload["qa_stages"]["retrieve"]["usable"] is True
    assert payload["qa_stages"]["answer"]["usable"] is True


def test_provisioning_reuses_readiness_and_keeps_reference_ingest_null(
    tmp_path: Path,
) -> None:
    report = {
        "profile": "laptop",
        "batch_id": "aic25-b1",
        "kis": {"state": "ready"},
        "qa": {"state": "degraded"},
        "channels": [{"name": "visual", "state": "not_provisioned", "reason_code": "missing"}],
        "checks": [],
        "checked_at_unix": 1.0,
    }
    context = _context(tmp_path, FakePipeline())
    context.selected_batch_id = "aic25-b1"
    context.validate_batch_filters = lambda values=None: tuple(values or ("aic25-b1",))
    context.readiness_service = SimpleNamespace(
        check=lambda batch_id, mode="app": SimpleNamespace(to_dict=lambda: dict(report))
    )
    app = create_app(context)
    endpoint = _endpoint(app, "/v1/health/provisioning", "GET")
    alias = _endpoint(app, "/api/v1/health/provisioning", "GET")

    response = endpoint("aic25-b1")
    assert response["batch_id"] == "aic25-b1"
    assert response["channels"][0]["state"] == "not_provisioned"
    assert response["ingest"] is None
    assert alias("aic25-b1") == response


def test_health_status_reuses_snapshot_and_returns_sanitized_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Alias equivalence must not depend on host RSS/disk changes between calls.
    monkeypatch.setattr("src.api._resource_snapshot", lambda context: {"process_rss_bytes": 100})
    channels = (
        ChannelReadiness(
            "organizer_clip",
            ChannelApplicability.REQUIRED,
            ChannelState.READY,
            count=177_321,
            resource_name="private-organizer-collection",
        ),
        ChannelReadiness(
            "visual",
            ChannelApplicability.OPTIONAL,
            ChannelState.NOT_PROVISIONED,
            "collection_missing",
        ),
        ChannelReadiness(
            "ocr",
            ChannelApplicability.OPTIONAL,
            ChannelState.NOT_PROVISIONED,
            "index_missing",
        ),
        ChannelReadiness(
            "asr",
            ChannelApplicability.OPTIONAL,
            ChannelState.SERVICE_UNAVAILABLE,
            "meilisearch_unavailable",
        ),
        ChannelReadiness(
            "metadata",
            ChannelApplicability.REQUIRED,
            ChannelState.READY,
            count=873,
        ),
    )
    report_payload = {
        "profile": "laptop",
        "batch_id": "aic25-b1",
        "kis": {"state": "ready"},
        "qa": {"state": "degraded"},
        "checked_at_unix": 10.0,
    }
    report = SimpleNamespace(channels=channels, to_dict=lambda: dict(report_payload))
    trake = WorkflowReadiness(WorkflowState.READY, ("organizer_clip",), ())
    ledger = FakeLedger()
    ledger.is_batch_registered = lambda _batch_id: True  # type: ignore[method-assign]
    context = _context(tmp_path, FakePipeline(), ledger)
    context.selected_batch_id = "aic25-b1"
    context.validate_batch_filters = lambda values=None: tuple(values or ("aic25-b1",))
    context.readiness_service = SimpleNamespace(
        check=lambda batch_id, mode="app": report,
        workflow_readiness=lambda current, name: trake,
    )
    app = create_app(context)
    endpoint = _endpoint(app, "/v1/health/status", "GET")
    alias = _endpoint(app, "/api/v1/health/status", "GET")

    response = endpoint("aic25-b1")

    assert response["trake"] == trake.to_dict()
    assert response["channel_statuses"] == {
        "visual_vector": {"state": "ready", "reason_code": None, "count": 177_321},
        "ocr_text": {
            "state": "not_provisioned",
            "reason_code": "index_missing",
            "count": None,
        },
        "asr_transcript": {
            "state": "service_unavailable",
            "reason_code": "meilisearch_unavailable",
            "count": None,
        },
        "metadata": {"state": "ready", "reason_code": None, "count": 873},
    }
    assert response["ingest"]["stage_counts"] == {}
    assert "private-organizer-collection" not in str(response)
    assert alias("aic25-b1") == response


def test_health_status_rejects_unknown_batch_without_probing(tmp_path: Path) -> None:
    context = _context(tmp_path, FakePipeline())
    context.validate_batch_filters = lambda _values=None: (_ for _ in ()).throw(
        ValueError("unknown batch")
    )
    context.readiness_service = SimpleNamespace(
        check=lambda *_args, **_kwargs: pytest.fail("readiness should not be probed")
    )
    endpoint = _endpoint(create_app(context), "/v1/health/status", "GET")

    with pytest.raises(HTTPException) as raised:
        endpoint("unknown")

    assert raised.value.status_code == 422
    assert raised.value.detail == "health status request is invalid"


def test_search_redacts_internal_media_uris_and_backend_evidence(
    tmp_path: Path,
) -> None:
    secret_path = "zip://D:/private/Keyframes.zip!/L21_V001/0001.jpg"
    pipeline = FakePipeline(
        search_hits=[
            {
                "video_id": "L21_V001",
                "batch_id": "aic25-b1",
                "source_frame_idx": 181,
                "image_uri": secret_path,
                "thumbnail_uri": "D:/private/thumb.webp",
                "channel_ranks": {"organizer_clip": 1, "D:/private": 2},
                "evidence": [
                    {
                        "channel": "organizer_clip",
                        "rank": 1,
                        "raw_score": 0.75,
                        "image_uri": secret_path,
                        "provider_body": "must-not-appear",
                    }
                ],
            }
        ]
    )
    endpoint = _endpoint(create_app(_context(tmp_path, pipeline)), "/v1/kis/search", "POST")

    response = endpoint({"query_id": "safe-hit", "original_text": "query"})

    hit = response["hits"][0]
    assert hit["thumbnail_uri"] == "/v1/frames/aic25-b1/L21_V001/181/image"
    assert hit["channel_ranks"] == {"organizer_clip": 1}
    assert hit["evidence"] == {
        "items": [{"channel": "organizer_clip", "rank": 1, "raw_score": 0.75}]
    }
    assert "Keyframes.zip" not in str(response)
    assert "must-not-appear" not in str(response)


def test_search_reports_503_when_no_configured_channel_produces_a_result(
    tmp_path: Path,
) -> None:
    secret = "provider-secret C:/private/archive.zip"
    pipeline = FakePipeline(channel_errors={"visual": secret, "ocr": secret})
    endpoint = _endpoint(create_app(_context(tmp_path, pipeline)), "/v1/kis/search", "POST")

    with pytest.raises(HTTPException) as raised:
        endpoint({"query_id": "kis-1", "original_text": "query"})

    assert raised.value.status_code == 503
    assert raised.value.detail["channels"] == {
        "visual": "channel_unavailable",
        "ocr": "channel_unavailable",
    }
    assert "provider-secret" not in str(raised.value.detail)
    assert "archive.zip" not in str(raised.value.detail)


@pytest.mark.parametrize(
    "payload",
    [
        {"query_id": "kis-1", "original_text": ""},
        {"query_id": "kis-1", "original_text": "query", "limit": 101},
        {"query_id": "kis-1", "original_text": "query", "mode": "unknown"},
    ],
)
def test_search_maps_contract_validation_to_422(tmp_path: Path, payload: dict[str, Any]) -> None:
    endpoint = _endpoint(
        create_app(_context(tmp_path, FakePipeline())),
        "/v1/kis/search",
        "POST",
    )

    with pytest.raises(HTTPException) as raised:
        endpoint(payload)

    assert raised.value.status_code == 422


def test_refine_parses_canonical_feedback_ids_and_forwards_limit(tmp_path: Path) -> None:
    hit = {
        "rank": 1,
        "video_id": "L21_V001",
        "source_frame_idx": 190,
        "score": 0.7,
    }
    pipeline = FakePipeline(refine_hits=[hit])
    endpoint = _endpoint(create_app(_context(tmp_path, pipeline)), "/v1/kis/refine", "POST")

    response = endpoint(
        {
            "query_id": "kis-refine",
            "original_text": "người áo đỏ",
            "limit": 12,
            "positive_feedback_ids": ["L21_V001:181"],
            "negative_feedback_ids": ["L21_V002:240"],
        }
    )

    assert response["query_id"] == "kis-refine"
    assert response["hits"][0]["source_frame_idx"] == 190
    assert response["hits"][0]["fused_score"] == 0.7
    assert pipeline.refine_calls[0]["positive_ids"] == (("L21_V001", 181),)
    assert pipeline.refine_calls[0]["negative_ids"] == (("L21_V002", 240),)
    assert pipeline.refine_calls[0]["limit"] == 12


@pytest.mark.parametrize(
    "feedback_id",
    ["missing-separator", "L21_V001:not-an-int", "L21_V001:-1"],
)
def test_refine_rejects_malformed_or_negative_source_frame_feedback(
    tmp_path: Path,
    feedback_id: str,
) -> None:
    endpoint = _endpoint(
        create_app(_context(tmp_path, FakePipeline())),
        "/v1/kis/refine",
        "POST",
    )

    with pytest.raises(HTTPException) as raised:
        endpoint(
            {
                "query_id": "kis-refine",
                "original_text": "query",
                "positive_feedback_ids": [feedback_id],
            }
        )

    assert raised.value.status_code == 422


def test_refine_maps_unavailable_feedback_backend_to_409(tmp_path: Path) -> None:
    pipeline = FakePipeline(refine_error=RuntimeError("feedback vectors unavailable"))
    endpoint = _endpoint(create_app(_context(tmp_path, pipeline)), "/v1/kis/refine", "POST")

    with pytest.raises(HTTPException) as raised:
        endpoint({"query_id": "kis-refine", "original_text": "query"})

    assert raised.value.status_code == 409


def test_status_returns_ledger_summary_and_maps_unknown_batch_to_404(tmp_path: Path) -> None:
    ledger = FakeLedger()
    endpoint = _endpoint(
        create_app(_context(tmp_path, FakePipeline(), ledger)),
        "/v1/ingest/status",
        "GET",
    )

    response = endpoint("aic25-b1", preload_window_seconds=60.0)

    assert response["batch_id"] == "aic25-b1"
    assert response["readiness"] == 1.0
    assert response["resources"]["scratch_used_bytes"] == 0
    assert response["resources"]["scratch_limit_bytes"] == 12 * 1024**3
    assert response["resources"]["disk_free_bytes"] > 0
    assert response["core_ingestion_projection"]["fits_preload_budget"] is True
    assert response["core_ingestion_projection"]["preload_window_seconds"] == 60.0
    assert ledger.status_calls == ["aic25-b1"]

    with pytest.raises(HTTPException) as invalid_deadline:
        endpoint("aic25-b1", preload_window_seconds=-1.0)
    assert invalid_deadline.value.status_code == 422

    failed_endpoint = _endpoint(
        create_app(
            _context(
                tmp_path,
                FakePipeline(),
                FakeLedger(status_error=KeyError("unknown batch")),
            )
        ),
        "/v1/ingest/status",
        "GET",
    )
    with pytest.raises(HTTPException) as raised:
        failed_endpoint("unknown")
    assert raised.value.status_code == 404


def test_export_validates_and_writes_canonical_source_frame_csv(tmp_path: Path) -> None:
    ledger = FakeLedger(active=True)
    endpoint = _endpoint(
        create_app(_context(tmp_path, FakePipeline(), ledger)),
        "/v1/submissions/export",
        "POST",
    )

    response = endpoint(
        {
            "batch_id": "aic25-b1",
            "filename": "kis-result.csv",
            "hits": [
                {"video_id": "l21_v001.mp4", "source_frame_idx": 181},
                {"video_id": "L21_V002", "source_frame_idx": 240},
            ],
        }
    )

    assert isinstance(response, FileResponse)
    target = Path(response.path)
    assert target == tmp_path / "kis-result.csv"
    assert target.read_text(encoding="utf-8") == "L21_V001,181\nL21_V002,240\n"
    assert ledger.active_calls == ["aic25-b1"]


def test_kis_export_uses_candidate_scoped_membership_and_mapping_lookups(
    tmp_path: Path,
) -> None:
    class ScopedRepository:
        def existing_source_frames(self, batch_id, identities, *, include_refined=True):
            assert batch_id == "aic25-b1"
            assert identities == [("L21_V001", 181)]
            assert include_refined is True
            return frozenset(identities)

        def frame_index_mappings_for_sources(self, batch_id, identities):
            assert batch_id == "aic25-b1"
            assert identities == [("L21_V001", 181)]
            return {"L21_V001": {7: 181}}

        def frame_index_mappings(self, _batch_id):
            raise AssertionError("full-batch mapping scan must not run")

    context = _context(tmp_path, FakePipeline())
    context.artifact_repository = ScopedRepository()
    endpoint = _endpoint(create_app(context), "/v1/submissions/export", "POST")

    response = endpoint(
        {
            "batch_id": "aic25-b1",
            "filename": "scoped.csv",
            "hits": [{"rank": 1, "video_id": "L21_V001", "source_frame_idx": 181}],
        }
    )

    assert Path(response.path).read_bytes() == b"L21_V001,181\n"


def test_kis_export_rejects_unknown_batch_before_artifact_or_ledger_calls(
    tmp_path: Path,
) -> None:
    ledger = FakeLedger(active=True)
    context = _context(tmp_path, FakePipeline(), ledger)

    def reject(_values=None):
        raise ValueError("unknown batch")

    context.validate_batch_filters = reject
    endpoint = _endpoint(
        create_app(context),
        "/v1/submissions/export",
        "POST",
    )

    with pytest.raises(HTTPException) as raised:
        endpoint(
            {
                "batch_id": "unknown",
                "filename": "kis.csv",
                "hits": [{"video_id": "L21_V001", "source_frame_idx": 181}],
            }
        )

    assert raised.value.status_code == 422
    assert raised.value.detail == "KIS submission export is invalid"
    assert ledger.active_calls == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "batch_id": "aic25-b1",
                "filename": "../escape.csv",
                "hits": [{"video_id": "L21_V001", "source_frame_idx": 1}],
            },
            "filename",
        ),
        (
            {
                "batch_id": "aic25-b1",
                "filename": "duplicate.csv",
                "hits": [
                    {"video_id": "L21_V001", "source_frame_idx": 1},
                    {"video_id": "L21_V001", "source_frame_idx": 1},
                ],
            },
            "duplicate",
        ),
        (
            {
                "batch_id": "aic25-b1",
                "filename": "ordinal.csv",
                "hits": [{"video_id": "L21_V001", "keyframe_ordinal": 7}],
            },
            "keyframe_ordinal",
        ),
        (
            {
                "batch_id": "aic25-b1",
                "filename": "mislabeled-ordinal.csv",
                "hits": [
                    {
                        "video_id": "L21_V001",
                        "source_frame_idx": 7,
                        "keyframe_ordinal": 7,
                    }
                ],
            },
            "keyframe_ordinal",
        ),
    ],
)
def test_export_maps_filename_and_submission_validation_to_422(
    tmp_path: Path,
    payload: dict[str, Any],
    message: str,
) -> None:
    endpoint = _endpoint(
        create_app(_context(tmp_path, FakePipeline())),
        "/v1/submissions/export",
        "POST",
    )

    with pytest.raises(HTTPException, match=message) as raised:
        endpoint(payload)

    assert raised.value.status_code == 422


def test_export_rejects_an_inactive_batch_before_writing(tmp_path: Path) -> None:
    ledger = FakeLedger(active=False)
    endpoint = _endpoint(
        create_app(_context(tmp_path, FakePipeline(), ledger)),
        "/v1/submissions/export",
        "POST",
    )

    with pytest.raises(HTTPException) as raised:
        endpoint(
            {
                "batch_id": "aic25-b1",
                "filename": "inactive.csv",
                "hits": [{"video_id": "L21_V001", "source_frame_idx": 181}],
            }
        )

    assert raised.value.status_code == 422
    assert "not active" in str(raised.value.detail)
    assert not (tmp_path / "inactive.csv").exists()
