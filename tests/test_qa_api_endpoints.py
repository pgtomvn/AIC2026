"""Offline transport tests for Visual Q&A routes and KIS coexistence."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image

from src.api import create_app
from src.contracts import (
    QACandidateFrame,
    QACandidateRequest,
    QACompetitionSpec,
    QAQueryRequest,
    QAReasoningResult,
    QASingleFrameAnswerRequest,
)
from src.retrieval.visual_qa import VisualQATotalFailureError


def _prediction(rank: int = 1, frame: int = 10) -> dict[str, Any]:
    return {
        "rank": rank,
        "batch_id": "b1",
        "video_id": "L21_V001",
        "source_frame_idx": frame,
        "timestamp_ms": frame * 100,
        "answer": "3",
        "confidence_score": 0.8,
        "answer_source": "ollama-qwen/qwen2.5vl:test",
        "window_id": "b1:L21_V001:10",
        "raw_answer": "three",
        "model_name": "qwen2.5vl:test",
        "evidence": {
            "provider": "ollama-qwen",
            "cache_status": "miss",
            "latency_ms": 10.0,
            "attempts": 1,
        },
    }


def _outcome(status: str, *, window_id: str = "w1") -> dict[str, Any]:
    value: dict[str, Any] = {
        "window_id": window_id,
        "batch_id": "b1",
        "video_id": "L21_V001",
        "source_frame_idx": 10,
        "timestamp_ms": 1_000,
        "shot_id": "shot-1",
        "retrieval_rank": 1,
        "retrieval_score": 0.75,
        "frames": [
            {
                "position": position,
                "batch_id": "b1",
                "video_id": "L21_V001",
                "source_frame_idx": frame,
                "timestamp_ms": frame * 100,
            }
            for position, frame in enumerate((9, 10, 11), 1)
        ],
        "status": status,
    }
    if status in {"answered", "unanswerable"}:
        value.update(
            {
                "answer": "3" if status == "answered" else "",
                "raw_answer": "three" if status == "answered" else "",
                "normalized_answer": "3" if status == "answered" else "",
                "answerable": status == "answered",
                "evidence_frame_indices": [2] if status == "answered" else [],
                "provider": "ollama-qwen",
                "model_name": "qwen2.5vl:test",
                "prompt_version": "qa-v1",
                "provider_latency_ms": 10.0,
                "served_latency_ms": 1.0,
                "attempts": 1,
                "cache_status": "hit",
                "cache_layers": {"reasoning": "hit"},
            }
        )
    elif status.endswith("_failed"):
        value.update(
            {
                "failure_code": "provider_request_failed",
                "failure_stage": "provider",
                "provider": "ollama-qwen",
                "model_name": "qwen2.5vl:test",
                "attempts": 2,
                "provider_error_category": "rate_limited",
                "provider_status_code": 429,
            }
        )
    return value


class _KISPipeline:
    last_channel_errors: dict[str, str] = {}


class _QAPipeline:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.search_calls: list[tuple[QAQueryRequest, str]] = []
        self.expand_calls: list[tuple[str, int]] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.retrieve_calls: list[tuple[QACandidateRequest, str]] = []

    def start_candidate_search(
        self,
        request: QACandidateRequest,
        *,
        session_id: str,
        legacy_request: dict[str, Any] | None = None,
    ) -> tuple[QACandidateFrame, ...]:
        if self.failure is not None:
            raise self.failure
        self.retrieve_calls.append((request, session_id))
        candidate = QACandidateFrame(
            candidate_id="b1:L21_V001:10",
            retrieval_rank=1,
            retrieval_score=0.8,
            batch_id="b1",
            video_id="L21_V001",
            source_frame_idx=10,
            timestamp_ms=1_000,
            shot_id="shot-1",
            channel_ranks={"organizer_clip": 1},
        )
        stored_request = request.to_dict()
        if legacy_request is not None:
            stored_request["legacy_request"] = dict(legacy_request)
        self.snapshots[session_id] = {
            "session_id": session_id,
            "request": stored_request,
            "state_version": 3,
            "status": "candidates_ready",
            "state": {
                "state_version": 3,
                "status": "candidates_ready",
                "candidate_cache_status": "miss",
            },
            "session_candidates": [candidate.to_dict()],
            "candidate_count": 1,
            "candidate_retrieval_ms": 2.0,
            "predictions": [],
            "answers": {},
            "failures": {},
            "channel_statuses": {
                "organizer_clip": {"state": "ready", "applicability": "required"}
            },
            "retrieval_errors": {},
            "expires_at": 1234.0,
        }
        return (candidate,)

    def search(self, request: QAQueryRequest, *, session_id: str) -> list[dict[str, Any]]:
        if self.failure is not None:
            raise self.failure
        self.search_calls.append((request, session_id))
        prediction = _prediction()
        self.snapshots[session_id] = {
            "session_id": session_id,
            "request": request.to_dict(),
            "state": {"predictions": [prediction], "results": {"w1": {}}, "failures": {}},
            "target_count": 1,
            "reasoned_count": 1,
            "expires_at": 1234.0,
        }
        return [prediction]

    def expand(self, search_id: str, target_clusters: int) -> list[dict[str, Any]]:
        self.expand_calls.append((search_id, target_clusters))
        snapshot = self.snapshots[search_id]
        snapshot["target_count"] = target_clusters
        return list(snapshot["state"]["predictions"])

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        return self.snapshots.get(search_id)

    def answer_candidate(
        self, search_id: str, request: QASingleFrameAnswerRequest
    ) -> dict[str, Any]:
        snapshot = self.snapshots[search_id]
        assert request.candidate_id == "b1:L21_V001:10"
        prediction = _prediction()
        prediction["evidence"]["evidence_frame_indices"] = [1]
        snapshot["status"] = "answered"
        snapshot["selected_candidate_id"] = request.candidate_id
        snapshot["predictions"] = [prediction]
        snapshot["answers"] = {
            request.candidate_id: {
                "candidate_id": request.candidate_id,
                "status": "answered",
                "reasoning": {
                    "answer": "3",
                    "answerable": True,
                    "provider": "ollama-qwen",
                    "model_name": "qwen2.5vl:test",
                    "latency_ms": 10.0,
                },
                "stage_timings": {"frame_prepare_ms": 1.0, "vlm_inference_ms": 10.0},
            }
        }
        return snapshot["answers"][request.candidate_id]


class _OutcomePipeline(_QAPipeline):
    def __init__(
        self,
        outcomes: list[dict[str, Any]],
        *,
        predictions: list[dict[str, Any]] | None = None,
        raise_total_failure: bool = False,
    ) -> None:
        super().__init__()
        self.outcomes = outcomes
        self.predictions = predictions or []
        self.raise_total_failure = raise_total_failure

    def search(self, request: QAQueryRequest, *, session_id: str) -> list[dict[str, Any]]:
        self.search_calls.append((request, session_id))
        statuses = [item["status"] for item in self.outcomes]
        answered = statuses.count("answered")
        unanswerable = statuses.count("unanswerable")
        failed = sum(status.endswith("_failed") for status in statuses)
        status = (
            "failed"
            if failed and not answered and not unanswerable
            else "partial"
            if failed
            else "completed"
            if answered
            else "completed_no_answer"
        )
        self.snapshots[session_id] = {
            "session_id": session_id,
            "request": request.to_dict(),
            "state": {
                "state_version": 2,
                "predictions": list(self.predictions),
                "results": {},
                "failures": {},
                "window_outcomes": {
                    item["window_id"]: dict(item) for item in self.outcomes
                },
            },
            "status": status,
            "window_outcomes": list(self.outcomes),
            "target_count": len(self.outcomes),
            "requested_target_clusters": 1,
            "retrieved_clusters": len(self.outcomes),
            "retrieval_exhausted": len(self.outcomes) < 1,
            "reasoned_count": answered + unanswerable,
            "answerable_count": answered,
            "unanswerable_count": unanswerable,
            "failed_count": failed,
            "pending_count": 0,
            "channel_statuses": {
                "organizer_clip": {
                    "state": "ready",
                    "applicability": "required",
                    "count": 177_321,
                }
            },
            "retrieval_errors": {"native_visual": "not_provisioned"},
            "operator_edits": [],
            "expires_at": 1234.0,
        }
        if self.raise_total_failure:
            raise VisualQATotalFailureError(
                session_id, ["provider_request_failed"]
            )
        return list(self.predictions)


class _Ledger:
    def is_batch_active(self, batch_id: str) -> bool:
        return batch_id == "b1"


class _Repository:
    frame = object()

    def frame_index_mappings(self, batch_id: str) -> dict[str, dict[int, int]]:
        assert batch_id == "b1"
        return {"L21_V001": {1: 10, 2: 11}}

    def get_frame(self, batch_id: str, video_id: str, source_frame_idx: int):
        if (batch_id, video_id, source_frame_idx) == ("b1", "L21_V001", 10):
            return self.frame
        return None


class _Resolver:
    def resolve(self, frame: object):
        assert frame is _Repository.frame
        return SimpleNamespace(
            data=b"jpeg-bytes",
            mime_type="image/jpeg",
            content_sha256="a" * 64,
        )


class _Readiness:
    def check(self, batch_id: str, *, mode: str = "app"):
        if mode not in {"app", "full"}:
            raise ValueError("unsupported test mode")
        class Channel:
            def __init__(self, name: str, usable: bool) -> None:
                self.name = name
                self.usable = usable

            def to_dict(self) -> dict[str, Any]:
                return {
                    "name": self.name,
                    "state": "ready" if self.usable else "blocked_asset",
                }

        channels = (
            Channel("canonical_images", True),
            Channel("ollama_vlm", True),
            Channel("openrouter_vlm", False),
        )
        return SimpleNamespace(
            channels=channels,
            to_dict=lambda: {
                "profile": "laptop",
                "batch_id": batch_id,
                "mode": mode,
                "kis": {"state": "ready"},
                "qa": {"state": "ready"},
                "channels": [
                    {"name": channel.name, "state": "ready" if channel.usable else "blocked_asset"}
                    for channel in channels
                ],
            }
        )


def _context(tmp_path: Path, qa_pipeline: _QAPipeline):
    return SimpleNamespace(
        config={
            "profile": "laptop",
            "competition": {"output_dir": str(tmp_path)},
            "services": {
                "ollama_vlm": {"model": "qwen2.5vl:test"},
                "openrouter_vlm": {"model": "cloud/test"},
            },
            "retrieval": {
                "visual_qa": {
                    "initial_clusters": 1,
                    "expansion_step": 5,
                    "max_clusters": 20,
                    "automatic_targets": [5, 10, 15, 20],
                    "default_answer_spec_id": QACompetitionSpec.conservative_general_text().spec_id,
                }
            },
        },
        pipeline=_KISPipeline(),
        qa_pipeline=qa_pipeline,
        qa_reasoner=SimpleNamespace(unload=lambda: None),
        ledger=_Ledger(),
        artifact_repository=_Repository(),
        qa_image_resolver=_Resolver(),
        readiness_service=_Readiness(),
        selected_batch_id="b1",
    )


def _route(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


def _request(mode: str = "interactive") -> dict[str, Any]:
    return {
        "query_id": "qa-1",
        "event_description": "A person holds three apples.",
        "question": "How many apples?",
        "mode": mode,
        "answer_spec_id": QACompetitionSpec.historical_numeric().spec_id,
        "batch_filters": ["b1"],
        "limit": 100,
    }


def test_qa_search_alias_is_retrieval_only_and_deprecated(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    app = create_app(_context(tmp_path, pipeline))
    endpoint = _route(app, "/v1/qa/search", "POST").endpoint
    alias = _route(app, "/api/v1/search/qa", "POST")

    response = endpoint(_request("automatic"))

    assert isinstance(pipeline.retrieve_calls[0][0], QACandidateRequest)
    assert pipeline.search_calls == []
    assert pipeline.expand_calls == []
    assert response["state_version"] == 3
    assert response["candidates"][0]["source_frame_idx"] == 10
    assert response["predictions"] == []
    assert alias.deprecated is True


def test_qa_search_defaults_and_validates_batches_before_pipeline(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    context = _context(tmp_path, pipeline)
    context.selected_batch_id = "b1"
    context.validate_batch_filters = lambda values=None: tuple(values or ("b1",))
    endpoint = _route(create_app(context), "/v1/qa/search", "POST").endpoint
    payload = _request()
    payload.pop("batch_filters")

    endpoint(payload)

    assert pipeline.retrieve_calls[0][0].batch_filters == ("b1",)

    def reject(_values=None):
        raise ValueError("unknown batch filter")

    context.validate_batch_filters = reject
    with pytest.raises(HTTPException) as raised:
        endpoint(payload)
    assert raised.value.status_code == 422
    assert len(pipeline.retrieve_calls) == 1


def test_qa_expand_is_permanently_removed(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    app = create_app(_context(tmp_path, pipeline))
    search = _route(app, "/v1/qa/search", "POST").endpoint
    response = search(_request())
    expand = _route(app, "/v1/qa/search/{search_id}/expand", "POST").endpoint

    with pytest.raises(HTTPException) as raised:
        expand(response["search_id"], {"target_clusters": 2})
    assert raised.value.status_code == 410
    assert raised.value.detail["code"] == "qa_expand_removed"
    assert pipeline.expand_calls == []


def test_qa_answer_runs_only_for_a_persisted_candidate(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    app = create_app(_context(tmp_path, pipeline))
    retrieved = _route(app, "/v1/qa/retrieve", "POST").endpoint(
        {
            "query_id": "qa-retrieve",
            "event_description": "A person holds three apples.",
            "batch_filters": ["b1"],
            "top_k": 5,
        }
    )
    answer = _route(app, "/v1/qa/search/{search_id}/answer", "POST").endpoint

    response = answer(
        retrieved["search_id"],
        {
            "candidate_id": "b1:L21_V001:10",
            "question": "How many apples?",
            "answer_spec_id": QACompetitionSpec.historical_numeric().spec_id,
            "reasoning_policy": "local_only",
        },
    )

    assert response["status"] == "answered"
    assert response["predictions"][0]["source_frame_idx"] == 10
    with pytest.raises(HTTPException) as outsider:
        answer(
            retrieved["search_id"],
            {
                "candidate_id": "b1:L21_V001:11",
                "question": "How many apples?",
                "answer_spec_id": QACompetitionSpec.historical_numeric().spec_id,
            },
        )
    assert outsider.value.status_code == 422


def test_qa_restore_and_total_failure_are_safe(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    app = create_app(_context(tmp_path, pipeline))
    searched = _route(app, "/v1/qa/search", "POST").endpoint(_request())
    restore = _route(app, "/v1/qa/search/{search_id}", "GET").endpoint
    assert restore(searched["search_id"])["query_id"] == "qa-1"
    with pytest.raises(HTTPException) as missing:
        restore("missing")
    assert missing.value.status_code == 404

    secret = "fake-provider-secret"
    failed = _route(
        create_app(_context(tmp_path, _QAPipeline(failure=RuntimeError(secret)))),
        "/v1/qa/search",
        "POST",
    ).endpoint
    with pytest.raises(HTTPException) as raised:
        failed(_request())
    assert raised.value.status_code == 503
    assert secret not in str(raised.value.detail)


def test_qa_restore_allowlists_predictions_and_drops_media_paths(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    prediction = _prediction()
    prediction.update(
        {
            "thumbnail_uri": "zip://D:/private/Keyframes.zip!/frame.jpg",
            "image_uri": "D:/private/frame.webp",
            "provider_body": "must-not-appear",
        }
    )
    prediction["evidence"].update(
        {
            "window_frames": [
                {
                    "position": 1,
                    "source_frame_idx": 10,
                    "timestamp_ms": 1_000,
                    "image_uri": "zip://D:/private/archive.zip!/frame.jpg",
                }
            ],
            "request_headers": {"authorization": "must-not-appear"},
        }
    )
    pipeline.snapshots["safe-restore"] = {
        "session_id": "safe-restore",
        "request": _request(),
        "state": {"predictions": [prediction]},
        "status": "completed",
        "target_count": 1,
        "requested_target_clusters": 1,
        "retrieved_clusters": 1,
    }
    restore = _route(
        create_app(_context(tmp_path, pipeline)),
        "/v1/qa/search/{search_id}",
        "GET",
    ).endpoint

    response = restore("safe-restore")

    restored = response["predictions"][0]
    assert "thumbnail_uri" not in restored
    assert "image_uri" not in restored
    assert "request_headers" not in restored["evidence"]
    assert restored["evidence"]["window_frames"] == [
        {"position": 1, "source_frame_idx": 10, "timestamp_ms": 1_000}
    ]
    assert "Keyframes.zip" not in str(response)
    assert "must-not-appear" not in str(response)


def test_deprecated_search_does_not_invoke_legacy_reasoning_outcomes(tmp_path: Path) -> None:
    pipeline = _OutcomePipeline([_outcome("unanswerable")])
    endpoint = _route(
        create_app(_context(tmp_path, pipeline)), "/v1/qa/search", "POST"
    ).endpoint

    response = endpoint(_request())

    assert response["status"] == "candidates_ready"
    assert response["predictions"] == []
    assert pipeline.search_calls == []
    assert response["channel_statuses"]["organizer_clip"]["state"] == "ready"
    assert response["retrieval_errors"] == {}


def test_deprecated_search_ignores_legacy_total_failure_mode(tmp_path: Path) -> None:
    partial = _OutcomePipeline(
        [_outcome("answered"), _outcome("provider_failed", window_id="w2")],
        predictions=[_prediction()],
    )
    response = _route(
        create_app(_context(tmp_path, partial)), "/v1/qa/search", "POST"
    ).endpoint(_request())
    assert response["status"] == "candidates_ready"
    assert response["predictions"] == []
    assert partial.search_calls == []

    failed = _OutcomePipeline(
        [_outcome("provider_failed")], raise_total_failure=True
    )
    endpoint = _route(
        create_app(_context(tmp_path, failed)), "/v1/qa/search", "POST"
    ).endpoint
    response = endpoint(_request())
    assert response["status"] == "candidates_ready"
    assert failed.search_calls == []


def test_canonical_image_endpoint_returns_only_resolved_bytes(tmp_path: Path) -> None:
    endpoint = _route(
        create_app(_context(tmp_path, _QAPipeline())),
        "/v1/frames/{batch_id}/{video_id}/{source_frame_idx}/image",
        "GET",
    ).endpoint

    response = endpoint("b1", "L21_V001", 10)

    assert isinstance(response, Response)
    assert response.body == b"jpeg-bytes"
    assert response.media_type == "image/jpeg"
    assert response.headers["etag"] == f'"{"a" * 64}"'


def test_readiness_is_distinct_from_liveness_and_validates_mode(tmp_path: Path) -> None:
    context = _context(tmp_path, _QAPipeline())
    context.selected_batch_id = "b1"
    context.launch_identity = {
        "canonical": True,
        "contract": "run_app-v2",
        "batch_id": "b1",
        "api_key": "must-not-appear",
        "workspace_path": "D:/secret/archive.zip",
    }
    validated: list[tuple[str, ...] | None] = []

    def validate(values=None):
        validated.append(None if values is None else tuple(values))
        if values == ("unknown",):
            raise ValueError("unknown batch filter")
        return tuple(values or ("b1",))

    context.validate_batch_filters = validate
    app = create_app(context)
    readiness = _route(app, "/v1/readiness", "GET").endpoint
    health = _route(app, "/healthz", "GET").endpoint

    report = readiness(None, "app")

    assert health() == {"status": "ok"}
    assert report["batch_id"] == "b1"
    assert report["qa"]["state"] == "ready"
    assert report["selected_batch_id"] == "b1"
    assert report["launch_identity"]["canonical"] is True
    assert "must-not-appear" not in str(report)
    assert "archive.zip" not in str(report)
    assert validated == [None]
    with pytest.raises(HTTPException) as unknown:
        readiness("unknown", "app")
    assert unknown.value.status_code == 422
    with pytest.raises(HTTPException) as raised:
        readiness("b1", "networked")
    assert raised.value.status_code == 422


def test_runtime_capabilities_expose_safe_local_and_cloud_states(tmp_path: Path) -> None:
    endpoint = _route(
        create_app(_context(tmp_path, _QAPipeline())),
        "/v1/runtime/capabilities",
        "GET",
    ).endpoint

    response = endpoint()

    assert response["default_batch_id"] == "b1"
    assert response["qa"]["default_policy"] == "gemini_then_local"
    assert response["qa"]["local"]["status"]["state"] == "ready"
    assert response["qa"]["cloud"]["status"]["state"] == "blocked_asset"
    assert "api_key" not in str(response)


def test_runtime_capabilities_do_not_block_on_readiness_while_warming(tmp_path: Path) -> None:
    context = _context(tmp_path, _QAPipeline())
    context.warmup = SimpleNamespace(readiness_report=lambda: None)
    context.readiness_service = SimpleNamespace(
        check=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("capabilities must not run a readiness probe")
        )
    )
    endpoint = _route(
        create_app(context), "/v1/runtime/capabilities", "GET"
    ).endpoint

    response = endpoint()

    assert response["qa"]["local"]["status"]["state"] == "warming"


def test_image_diagnostic_is_ephemeral_structured_and_consent_gated(tmp_path: Path) -> None:
    context = _context(tmp_path, _QAPipeline())
    calls: list[tuple[Any, ...]] = []

    class Reasoner:
        def reason(self, request, images, spec):
            calls.append((request, images, spec))
            return QAReasoningResult(
                raw_answer="three",
                normalized_answer="3",
                answerable=True,
                evidence_frame_indices=(1,),
                provider="ollama-qwen",
                model_name="qwen2.5vl:test",
                prompt_version="test-v1",
                latency_ms=12.0,
                attempts=1,
            )

        def unload(self):
            return None

    context.qa_reasoner = Reasoner()
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(output, format="PNG")
    payload = {
        "image_base64": base64.b64encode(output.getvalue()).decode("ascii"),
        "mime_type": "image/png",
        "question": "Có bao nhiêu hình?",
        "reasoning_policy": "local_only",
    }
    endpoint = _route(
        create_app(context), "/v1/qa/diagnostics/image", "POST"
    ).endpoint

    response = endpoint(payload)

    assert response["answer"] == "3"
    assert response["actual_provider"] == "ollama-qwen"
    assert response["fallback_used"] is False
    assert "raw_answer" not in response
    assert len(calls) == 1
    assert calls[0][1][0].mime_type == "image/webp"

    with pytest.raises(HTTPException) as no_consent:
        endpoint({**payload, "reasoning_policy": "gemini_then_local"})
    assert no_consent.value.status_code == 422


def test_operator_edit_uses_api_audit_boundary(tmp_path: Path) -> None:
    pipeline = _OutcomePipeline([_outcome("answered")], predictions=[_prediction()])
    request = QAQueryRequest.from_dict(_request())
    pipeline.search(request, session_id="edit-session")
    context = _context(tmp_path, pipeline)
    recorded: list[tuple[Any, ...]] = []
    context.qa_session_store = SimpleNamespace(
        record_operator_edit=lambda *args: recorded.append(args)
    )
    endpoint = _route(
        create_app(context), "/v1/qa/search/{search_id}/edits", "POST"
    ).endpoint

    response = endpoint(
        "edit-session", {"window_id": "w1", "answer": "3"}
    )

    assert response["status"] == "completed"
    assert recorded[0][:2] == ("edit-session", "w1")
    assert recorded[0][2]["normalized_answer"] == "3"


def test_legacy_restore_normalizes_failures_and_recovers_safe_frames(
    tmp_path: Path,
) -> None:
    pipeline = _QAPipeline()
    pipeline.snapshots["legacy"] = {
        "session_id": "legacy",
        "request": _request(),
        "targets": [
            {
                "window_id": "w1",
                "batch_id": "b1",
                "video_id": "L21_V001",
                "source_frame_idx": 10,
                "timestamp_ms": 1_000,
            },
            {
                "window_id": "w2",
                "batch_id": "b1",
                "video_id": "L21_V002",
                "source_frame_idx": 20,
                "timestamp_ms": 2_000,
            },
        ],
        "window_outcomes": [
            {"window_id": "w1", "status": "unanswerable", "answerable": False},
            {
                "window_id": "w2",
                "status": "provider_failed",
                "failure_code": "reasoning_failed",
                "failure_stage": "provider",
            },
        ],
        "state": {
            "predictions": [
                {
                    "window_id": "w1",
                    "batch_id": "b1",
                    "video_id": "L21_V001",
                    "source_frame_idx": 10,
                    "evidence": {
                        "window_frames": [
                            {"source_frame_idx": frame, "timestamp_ms": frame * 100}
                            for frame in (9, 10, 11)
                        ]
                    },
                }
            ]
        },
    }
    restore = _route(
        create_app(_context(tmp_path, pipeline)),
        "/v1/qa/search/{search_id}",
        "GET",
    ).endpoint

    response = restore("legacy")

    outcomes = response["window_outcomes"]
    assert [len(value["frames"]) for value in outcomes] == [3, 1]
    assert outcomes[1]["status"] == "internal_failed"
    assert outcomes[1]["failure_code"] == "internal_reasoning_error"
    assert outcomes[1]["failure_stage"] == "internal"
    assert "reasoning_failed" not in str(response)


def test_qa_export_writes_three_columns_without_header(tmp_path: Path) -> None:
    endpoint = _route(
        create_app(_context(tmp_path, _QAPipeline())),
        "/v1/qa/submissions/export",
        "POST",
    ).endpoint
    response = endpoint(
        {
            "batch_id": "b1",
            "filename": "qa.csv",
            "answer_spec_id": QACompetitionSpec.historical_numeric().spec_id,
            "predictions": [_prediction()],
        }
    )

    assert isinstance(response, FileResponse)
    assert Path(response.path).read_text(encoding="utf-8") == "L21_V001,10,3\n"


def test_qa_export_rejects_unknown_batch_before_artifact_or_ledger_calls(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _QAPipeline())

    def reject(_values=None):
        raise ValueError("unknown batch")

    context.validate_batch_filters = reject
    endpoint = _route(
        create_app(context), "/v1/qa/submissions/export", "POST"
    ).endpoint

    with pytest.raises(HTTPException) as raised:
        endpoint(
            {
                "batch_id": "unknown",
                "filename": "qa.csv",
                "answer_spec_id": QACompetitionSpec.historical_numeric().spec_id,
                "predictions": [_prediction()],
            }
        )

    assert raised.value.status_code == 422
    assert raised.value.detail == "QA submission export is invalid"
