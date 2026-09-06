"""Offline tests for the Streamlit-to-FastAPI thin client boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from app.http_backend import BackendError, _safe_error, build_http_runtime
from src.contracts import (
    QACompetitionSpec,
    QAFrameInjectionRequest,
    QAQueryRequest,
    QASingleFrameAnswerRequest,
    QueryRequest,
)


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.content = b"image"

    def json(self) -> Any:
        return self.payload


class _HTTP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        if url.endswith("/v1/runtime/capabilities"):
            return _Response(
                {
                    "profile": "laptop",
                    "default_batch_id": "b1",
                    "qa": {
                        "local": {"status": {"state": "ready"}},
                        "cloud": {"status": {"state": "blocked_asset"}},
                    },
                }
            )
        if url.endswith("/v1/runtime/warmup"):
            return _Response(
                {
                    "state": "evicted",
                    "search_usable": False,
                    "workflows": {"trake": {"state": "ready", "usable": True}},
                }
            )
        if url.endswith("/v1/runtime/warmup/start"):
            return _Response({"state": "warming", "search_usable": False})
        if url.endswith("/v1/readiness"):
            return _Response(
                {
                    "kis": {"state": "ready"},
                    "qa": {"state": "ready"},
                    "warmup": {"workflows": {"trake": {"state": "ready", "usable": True}}},
                }
            )
        if url.endswith("/v1/kis/search") or url.endswith("/v1/kis/refine"):
            return _Response(
                {
                    "hits": [{"video_id": "L21_V001", "source_frame_idx": 10}],
                    "channel_statuses": {"organizer_clip": {"state": "ready"}},
                }
            )
        if url.endswith("/v1/frames/validate"):
            return _Response(
                {
                    "batch_active": True,
                    "existing": [{"video_id": "L21_V001", "source_frame_idx": 10}],
                    "ordinal_mappings": {},
                    "timestamps_ms": {"L21_V001:10": 1_000},
                }
            )
        raise AssertionError(f"unexpected request: {method} {url}")


def test_thin_client_preserves_fastapi_as_the_only_runtime_owner() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "from src.runtime import build_runtime" not in source
    assert "build_http_runtime(" in source


def test_kis_adapter_serializes_feedback_ids_without_unknown_fields() -> None:
    http = _HTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)
    request = QueryRequest(
        query_id="q1",
        original_text="một người đi bộ",
        batch_filters=("b1",),
    )

    context.pipeline.search(request, limit=20, refine_raw=True)
    context.pipeline.refine(
        request,
        positive_ids=(("L21_V001", 10),),
        negative_ids=(("L21_V002", 20),),
        limit=20,
    )

    search_payload = http.calls[1][2]["json"]
    refine_payload = http.calls[2][2]["json"]
    assert search_payload["refine_raw"] is True
    assert refine_payload["positive_feedback_ids"] == ["L21_V001:10"]
    assert refine_payload["negative_feedback_ids"] == ["L21_V002:20"]
    assert context.pipeline.last_channel_statuses["organizer_clip"]["state"] == "ready"


def test_repository_uses_bounded_validation_api_for_export_membership() -> None:
    http = _HTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)

    existing = context.artifact_repository.existing_source_frames("b1", (("L21_V001", 10),))
    record = context.artifact_repository.get_source_frame("b1", "L21_V001", 10)

    assert existing == {("L21_V001", 10)}
    assert record is not None and record.timestamp_ms == 1_000
    validation_calls = [call for call in http.calls if call[1].endswith("/v1/frames/validate")]
    assert len(validation_calls) == 2


class _NeighborsHTTP(_HTTP):
    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        if url.endswith("/v1/frames/b1/L21_V001/neighbors") and method == "GET":
            self.calls.append((method, url, kwargs))
            return _Response(
                {
                    "neighbors": [
                        {
                            "batch_id": "b1",
                            "video_id": "L21_V001",
                            "source_frame_idx": 12,
                            "timestamp_ms": 1_250,
                            "shot_id": "shot-1",
                        }
                    ]
                }
            )
        return super().request(method, url, **kwargs)


def test_repository_neighbors_uses_timestamp_route_and_bounded_parameters() -> None:
    http = _NeighborsHTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)

    frames = context.artifact_repository.neighbors(
        "b1",
        "L21_V001",
        1_250,
        radius_ms=4_000,
        limit=5,
    )

    assert [(frame.source_frame_idx, frame.timestamp_ms, frame.shot_id) for frame in frames] == [
        (12, 1_250, "shot-1")
    ]
    method, url, kwargs = http.calls[-1]
    assert method == "GET"
    assert url.endswith("/v1/frames/b1/L21_V001/neighbors")
    assert "/1250/neighbors" not in url
    assert kwargs["params"] == {
        "timestamp_ms": 1_250,
        "radius_ms": 4_000,
        "limit": 5,
    }


class _QAInjectionHTTP(_HTTP):
    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        if url.endswith("/v1/qa/inject") and method == "POST":
            self.calls.append((method, url, kwargs))
            return _Response(
                {
                    "search_id": "server-injected",
                    "candidate_count": 1,
                    "selected_candidate_id": "b1:L21_V001:10",
                    "candidates": [
                        {
                            "candidate_id": "b1:L21_V001:10",
                            "retrieval_rank": 1,
                            "retrieval_score": 0.0,
                            "batch_id": "b1",
                            "video_id": "L21_V001",
                            "source_frame_idx": 10,
                            "timestamp_ms": 1_000,
                            "shot_id": "shot-1",
                            "channel_ranks": {},
                            "evidence": {"origin": "kis_bridge"},
                        }
                    ],
                    "predictions": [],
                }
            )
        return super().request(method, url, **kwargs)


def test_qa_thin_client_injects_frame_and_keeps_authoritative_search_id() -> None:
    http = _QAInjectionHTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)
    request = QAFrameInjectionRequest(
        "bridge", "b1", "L21_V001", 10, "kis_bridge", "a news presenter"
    )

    candidates = context.qa_pipeline.inject(request, session_id="client-correlation")

    assert len(candidates) == 1
    assert context.qa_pipeline.last_snapshot["session_id"] == "server-injected"
    assert context.qa_pipeline.aliases["client-correlation"] == "server-injected"
    assert http.calls[-1][2]["json"] == request.to_dict()
    assert http.calls[-1][2]["timeout"] == (3, 30.0)


def test_thin_client_warmup_poll_is_read_only_while_qwen_is_resident() -> None:
    http = _HTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)

    snapshot = context.warmup.snapshot()

    assert snapshot["state"] == "evicted"
    assert http.calls[-1][0:2] == (
        "GET",
        "http://127.0.0.1:8080/v1/runtime/warmup",
    )
    assert not any(call[1].endswith("/v1/runtime/warmup/start") for call in http.calls)


class _RecoverableQAHTTP(_HTTP):
    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        if url.endswith("/v1/qa/retrieve") and method == "POST":
            self.calls.append((method, url, kwargs))
            return _Response(
                {
                    "detail": {
                        "code": "qa_reasoning_unavailable",
                        "search_id": "server-search-id",
                        "recoverable": True,
                        "status": "failed",
                        "failed_clusters": 1,
                        "target_clusters": 1,
                    }
                },
                503,
            )
        if url.endswith("/v1/qa/search/server-search-id") and method == "GET":
            self.calls.append((method, url, kwargs))
            return _Response(
                {
                    "search_id": "server-search-id",
                    "status": "failed",
                    "predictions": [],
                    "window_outcomes": [{"window_id": "w1", "status": "provider_failed"}],
                    "candidate_windows": [{"window_id": "w1", "status": "pending"}],
                }
            )
        return super().request(method, url, **kwargs)


def test_recoverable_503_keeps_authoritative_search_id_and_restores_snapshot() -> None:
    http = _RecoverableQAHTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)
    request = QAQueryRequest(
        query_id="correlation-only",
        event_description="A man is broadcasting TV news.",
        question="Does the man wear glasses?",
        answer_spec_id=QACompetitionSpec.conservative_general_text().spec_id,
        batch_filters=("b1",),
        reasoning_policy="local_only",
    )

    with pytest.raises(BackendError) as captured:
        context.qa_pipeline.search(request)

    error = captured.value
    assert error.status_code == 503
    assert error.search_id == "server-search-id"
    assert error.recoverable is True
    assert error.snapshot is not None
    assert error.snapshot["session_id"] == "server-search-id"
    assert context.qa_pipeline.last_snapshot == error.snapshot
    assert not any("correlation-only/expand" in call[1] for call in http.calls)


class _MissingCandidateHTTP(_HTTP):
    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        if url.endswith("/v1/qa/search/s1/answer") and method == "POST":
            return _Response(
                {
                    "detail": {
                        "code": "qa_candidate_media_unavailable",
                        "candidate_id": "b1:L21_V001:10",
                        "recoverable": True,
                    }
                },
                409,
            )
        if url.endswith("/v1/qa/search/s1") and method == "GET":
            return _Response(
                {
                    "search_id": "s1",
                    "state_version": 3,
                    "status": "failed",
                    "candidates": [],
                    "predictions": [],
                    "failures": {"b1:L21_V001:10": "media_unavailable"},
                }
            )
        return super().request(method, url, **kwargs)


def test_answer_media_failure_restores_persisted_candidate_state() -> None:
    context = build_http_runtime(
        "http://127.0.0.1:8080", http=_MissingCandidateHTTP()
    )

    with pytest.raises(BackendError) as captured:
        context.qa_pipeline.answer_candidate(
            "s1",
            QASingleFrameAnswerRequest(
                candidate_id="b1:L21_V001:10",
                question="How many apples?",
                reasoning_policy="local_only",
            ),
        )

    error = captured.value
    assert error.status_code == 409
    assert error.detail["candidate_id"] == "b1:L21_V001:10"
    assert error.snapshot is not None
    assert error.snapshot["failures"] == {"b1:L21_V001:10": "media_unavailable"}


def test_thin_client_refreshes_capabilities_and_uses_warmup_trake_readiness() -> None:
    http = _HTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)
    report = context.readiness_service.check("b1")

    capabilities = context.refresh_capabilities()
    trake = context.readiness_service.workflow_readiness(report, "trake")

    assert capabilities["qa"]["local"]["status"]["state"] == "ready"
    assert trake.to_dict() == {"state": "ready", "usable": True}


class _BundleHTTP(_HTTP):
    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        if url.endswith("/v1/submissions/bundles") and method == "POST":
            self.calls.append((method, url, kwargs))
            return _Response(
                {
                    "run_id": "run_20260817T120000123456Z",
                    "filename": "AIC2026_submission_20260817T120000123456Z.zip",
                    "sha256": hashlib.sha256(b"zip-bytes").hexdigest(),
                    "file_count": 3,
                    "byte_count": 9,
                    "download_url": (
                        "/v1/submissions/bundles/run_20260817T120000123456Z/download"
                    ),
                }
            )
        return super().request(method, url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append(("GET", url, kwargs))
        response = _Response({}, 200)
        response.content = b"zip-bytes"
        return response


def test_thin_client_builds_then_downloads_bundle_bytes() -> None:
    http = _BundleHTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)

    result = context.create_submission_bundle(
        {"mode": "all", "batch_id": "b1", "entries": []}
    )

    assert result["data"] == b"zip-bytes"
    assert http.calls[-2][0] == "POST"
    assert http.calls[-1][1].endswith(
        "/v1/submissions/bundles/run_20260817T120000123456Z/download"
    )


def test_bundle_validation_error_keeps_only_sanitized_issues() -> None:
    response = _Response(
        {
            "detail": {
                "code": "bundle_validation_failed",
                "issues": [
                    {
                        "filename": "query-1-kis.csv",
                        "task": "kis",
                        "code": "submission_frame_not_canonical",
                        "message": "frame is invalid",
                        "row": 2,
                        "line": 2,
                        "suggestion": "select another frame",
                        "path": "D:/secret",
                    }
                ],
            }
        },
        422,
    )

    error = _safe_error(response)

    assert error.detail["issues"][0]["row"] == 2
    assert "path" not in error.detail["issues"][0]


def test_thin_client_rejects_bundle_download_hash_mismatch() -> None:
    http = _BundleHTTP()
    original_request = http.request

    def mismatched(method: str, url: str, **kwargs: Any) -> _Response:
        response = original_request(method, url, **kwargs)
        if url.endswith("/v1/submissions/bundles"):
            response.payload["sha256"] = "0" * 64
        return response

    http.request = mismatched  # type: ignore[method-assign]
    context = build_http_runtime("http://127.0.0.1:8080", http=http)
    with pytest.raises(BackendError) as caught:
        context.create_submission_bundle({"mode": "current", "entries": [{}]})
    assert caught.value.code == "bundle_download_integrity_failed"


class _StagingHTTP(_HTTP):
    snapshot = {
        "schema_version": 1,
        "revision": 4,
        "updated_at_utc": "2026-08-20T09:30:00Z",
        "entries": {},
    }

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        if "/v1/submissions/staging" in url:
            self.calls.append((method, url, kwargs))
            return _Response(dict(self.snapshot))
        return super().request(method, url, **kwargs)


def test_thin_client_owns_staging_through_fastapi_only() -> None:
    http = _StagingHTTP()
    context = build_http_runtime("http://127.0.0.1:8080", http=http)

    assert context.load_submission_staging()["revision"] == 4
    context.stage_submission_entry(
        {"expected_revision": 4, "batch_id": "b1", "entry": {}}
    )
    context.mutate_submission_rows(
        "query-p1-15-qa.csv",
        {"action": "delete", "expected_revision": 4, "rank": 1},
    )
    context.delete_submission_entry("query-p1-15-qa.csv", expected_revision=4)
    context.clear_submission_staging(expected_revision=4)

    staging_calls = [call for call in http.calls if "/v1/submissions/staging" in call[1]]
    assert [call[0] for call in staging_calls] == [
        "GET",
        "POST",
        "POST",
        "DELETE",
        "DELETE",
    ]
    assert staging_calls[2][1].endswith(
        "/v1/submissions/staging/entries/query-p1-15-qa.csv/rows"
    )
    assert staging_calls[3][1].endswith(
        "/v1/submissions/staging/entries/query-p1-15-qa.csv"
    )
    assert staging_calls[3][2]["params"] == {"expected_revision": 4}
