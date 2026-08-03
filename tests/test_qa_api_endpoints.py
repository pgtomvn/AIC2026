"""Offline transport tests for Visual Q&A routes and KIS coexistence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from src.api import create_app
from src.contracts import QACompetitionSpec, QAQueryRequest
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
        "answer_source": "gemini/gemini-3.6-flash",
        "window_id": "b1:L21_V001:10",
        "raw_answer": "three",
        "model_name": "gemini-3.6-flash",
        "evidence": {
            "provider": "gemini",
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
                "provider": "gemini",
                "model_name": "gemini-3.6-flash",
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
                "provider": "gemini",
                "model_name": "gemini-3.6-flash",
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

    def search(self, request: QAQueryRequest, *, session_id: str) -> list[dict[str, Any]]:
        if self.failure is not None:
            raise self.failure
        self.search_calls.append((request, session_id))
        prediction = _prediction()
        self.snapshots[session_id] = {
            "session_id": session_id,
            "request": request.to_dict(),
            "state": {"predictions": [prediction], "results": {"w1": {}}, "failures": {}},
            "target_count": 5,
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
            "requested_target_clusters": 5,
            "retrieved_clusters": len(self.outcomes),
            "retrieval_exhausted": len(self.outcomes) < 5,
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
        return SimpleNamespace(
            to_dict=lambda: {
                "profile": "laptop",
                "batch_id": batch_id,
                "mode": mode,
                "kis": {"state": "ready"},
                "qa": {"state": "ready"},
                "channels": [],
            }
        )


def _context(tmp_path: Path, qa_pipeline: _QAPipeline):
    return SimpleNamespace(
        config={
            "competition": {"output_dir": str(tmp_path)},
            "retrieval": {
                "visual_qa": {
                    "initial_clusters": 5,
                    "expansion_step": 5,
                    "max_clusters": 20,
                    "default_answer_spec_id": QACompetitionSpec.historical_numeric().spec_id,
                }
            },
        },
        pipeline=_KISPipeline(),
        qa_pipeline=qa_pipeline,
        ledger=_Ledger(),
        artifact_repository=_Repository(),
        qa_image_resolver=_Resolver(),
        readiness_service=_Readiness(),
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


def test_qa_search_alias_and_automatic_five_cluster_expansions(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    app = create_app(_context(tmp_path, pipeline))
    endpoint = _route(app, "/v1/qa/search", "POST").endpoint
    alias = _route(app, "/api/v1/search/qa", "POST")

    response = endpoint(_request("automatic"))

    search_id = response["search_id"]
    assert isinstance(pipeline.search_calls[0][0], QAQueryRequest)
    assert pipeline.expand_calls == [(search_id, 10), (search_id, 15), (search_id, 20)]
    assert response["target_clusters"] == 20
    assert response["predictions"][0]["source_frame_idx"] == 10
    assert "session_id" not in response["predictions"][0]
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

    assert pipeline.search_calls[0][0].batch_filters == ("b1",)

    def reject(_values=None):
        raise ValueError("unknown batch filter")

    context.validate_batch_filters = reject
    with pytest.raises(HTTPException) as raised:
        endpoint(payload)
    assert raised.value.status_code == 422
    assert len(pipeline.search_calls) == 1


def test_qa_interactive_expand_is_idempotent_and_bounded_to_five(tmp_path: Path) -> None:
    pipeline = _QAPipeline()
    app = create_app(_context(tmp_path, pipeline))
    search = _route(app, "/v1/qa/search", "POST").endpoint
    response = search(_request())
    expand = _route(app, "/v1/qa/search/{search_id}/expand", "POST").endpoint

    expanded = expand(response["search_id"], {"target_clusters": 10})
    replayed = expand(response["search_id"], {"target_clusters": 10})

    assert expanded["target_clusters"] == replayed["target_clusters"] == 10
    assert pipeline.expand_calls == [(response["search_id"], 10)]
    with pytest.raises(HTTPException) as raised:
        expand(response["search_id"], {"target_clusters": 20})
    assert raised.value.status_code == 422


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


def test_qa_all_unanswerable_is_successful_and_exposes_safe_outcomes(tmp_path: Path) -> None:
    pipeline = _OutcomePipeline([_outcome("unanswerable")])
    endpoint = _route(
        create_app(_context(tmp_path, pipeline)), "/v1/qa/search", "POST"
    ).endpoint

    response = endpoint(_request())

    assert response["status"] == "completed_no_answer"
    assert response["predictions"] == []
    assert response["reasoned_clusters"] == 1
    assert response["answerable_clusters"] == 0
    assert response["unanswerable_clusters"] == 1
    assert response["failed_clusters"] == 0
    assert response["window_outcomes"][0]["status"] == "unanswerable"
    assert response["operational_traces"] == [
        {
            "window_id": "w1",
            "outcome": "unanswerable",
            "provider": "gemini",
            "model_name": "gemini-3.6-flash",
            "cache_status": "hit",
            "provider_latency_ms": 10.0,
            "served_latency_ms": 1.0,
            "attempts": 1,
            "cache_layers": {"reasoning": "hit"},
        }
    ]
    assert response["channel_statuses"]["organizer_clip"]["state"] == "ready"
    assert response["retrieval_errors"] == {"native_visual": "not_provisioned"}


def test_qa_partial_is_http_success_and_total_failure_is_recoverable(tmp_path: Path) -> None:
    partial = _OutcomePipeline(
        [_outcome("answered"), _outcome("provider_failed", window_id="w2")],
        predictions=[_prediction()],
    )
    response = _route(
        create_app(_context(tmp_path, partial)), "/v1/qa/search", "POST"
    ).endpoint(_request())
    assert response["status"] == "partial"
    assert response["answerable_clusters"] == response["failed_clusters"] == 1

    failed = _OutcomePipeline(
        [_outcome("provider_failed")], raise_total_failure=True
    )
    endpoint = _route(
        create_app(_context(tmp_path, failed)), "/v1/qa/search", "POST"
    ).endpoint
    with pytest.raises(HTTPException) as raised:
        endpoint(_request())
    detail = raised.value.detail
    assert raised.value.status_code == 503
    assert detail["recoverable"] is True
    assert detail["search_id"] in failed.snapshots
    assert detail["status"] == "failed"
    assert detail["failed_clusters"] == 1
    assert detail["operational_traces"][0]["failure_code"] == "provider_request_failed"
    assert detail["operational_traces"][0]["provider"] == "gemini"
    assert detail["operational_traces"][0]["model_name"] == "gemini-3.6-flash"
    assert detail["operational_traces"][0]["attempts"] == 2
    assert detail["operational_traces"][0]["provider_error_category"] == "rate_limited"
    assert detail["operational_traces"][0]["provider_status_code"] == 429


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
        "contract": "run_app-v1",
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
