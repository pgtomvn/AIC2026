"""Offline FastAPI transport tests for TRAKE search and gating."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import create_app
from src.contracts import (
    TRAKEResponse,
    TRAKEResponseItem,
    TRAKEStageMatch,
)
from src.retrieval.trake_router import TRAKENotReadyError


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def _payload() -> dict:
    return {
        "query_id": "trake-api-1",
        "global_description": "một người mua hàng rồi rời đi",
        "stages": [
            {"stage_index": 1, "description": "người chọn món hàng"},
            {"stage_index": 2, "description": "người rời cửa hàng"},
        ],
        "top_k": 10,
    }


class _TRAKEPipeline:
    def __init__(self, *, unavailable: bool = False, empty: bool = False) -> None:
        self.unavailable = unavailable
        self.empty = empty
        self.calls = []

    def search(self, request, *, default_batch_id):
        self.calls.append((request, default_batch_id))
        if self.unavailable:
            raise TRAKENotReadyError(
                "organizer_clip",
                "not_provisioned",
                channel_statuses={
                    "organizer_clip": {
                        "state": "not_provisioned",
                        "reason_code": "collection_missing",
                        "raw_exception": "D:/private/qdrant.log",
                    }
                },
            )
        results = ()
        if not self.empty:
            results = (
                TRAKEResponseItem(
                    rank=1,
                    batch_id=request.batch_filters[0],
                    video_id="L21_V001",
                    stage_matches=(
                        TRAKEStageMatch(
                            stage_index=1,
                            source_frame_idx=10,
                            timestamp_ms=1_000,
                            similarity=0.8,
                        ),
                        TRAKEStageMatch(
                            stage_index=2,
                            source_frame_idx=20,
                            timestamp_ms=2_000,
                            similarity=0.7,
                        ),
                    ),
                    confidence_score=0.9,
                ),
            )
        return TRAKEResponse(
            query_id=request.query_id,
            results=results,
            channel_statuses={
                "organizer_clip": {"state": "ready", "attempted": True}
            },
        )

    @staticmethod
    def channel_readiness(_batch_id):
        return True, {"organizer_clip": {"state": "ready"}}


class _Warmup:
    def start(self):
        return True

    @staticmethod
    def snapshot():
        return {"state": "warming", "search_usable": False, "channels": []}


def _context(pipeline: _TRAKEPipeline, *, warmup=None):
    selected = "aic25-b1-reference"
    return SimpleNamespace(
        trake_pipeline=pipeline,
        pipeline=SimpleNamespace(),
        qa_pipeline=None,
        ledger=SimpleNamespace(is_batch_registered=lambda _batch: False),
        artifact_repository=None,
        readiness_service=None,
        selected_batch_id=selected,
        validate_batch_filters=lambda values=None: tuple(values or (selected,)),
        warmup=warmup,
        config={"warmup": {"retry_after_seconds": 2}},
    )


def test_canonical_and_deprecated_trake_routes_serialize_frame_ids() -> None:
    pipeline = _TRAKEPipeline()
    app = create_app(_context(pipeline))

    for path in ("/v1/trake/search", "/api/v1/search/trake"):
        response = _endpoint(app, path, "POST")(_payload())
        assert response["results"][0]["frame_ids"] == [10, 20]
        assert response["results"][0]["stage_matches"][0]["source_frame_idx"] == 10
        assert response["channel_statuses"] == {
            "organizer_clip": {"state": "ready"}
        }
    assert all(call[0].batch_filters == ("aic25-b1-reference",) for call in pipeline.calls)


def test_trake_returns_empty_200_after_successful_dense_query() -> None:
    endpoint = _endpoint(
        create_app(_context(_TRAKEPipeline(empty=True))),
        "/v1/trake/search",
        "POST",
    )

    response = endpoint(_payload())

    assert response["results"] == []
    assert response["channel_statuses"]["organizer_clip"]["state"] == "ready"


def test_trake_contract_and_warmup_failures_are_structured() -> None:
    invalid = _endpoint(
        create_app(_context(_TRAKEPipeline())), "/v1/trake/search", "POST"
    )
    malformed = _payload()
    malformed["stages"][1]["stage_index"] = 3
    with pytest.raises(HTTPException) as contract_error:
        invalid(malformed)
    assert contract_error.value.status_code == 422

    warming = _endpoint(
        create_app(_context(_TRAKEPipeline(), warmup=_Warmup())),
        "/v1/trake/search",
        "POST",
    )
    with pytest.raises(HTTPException) as warmup_error:
        warming(_payload())
    assert warmup_error.value.status_code == 503
    assert warmup_error.value.detail["code"] == "runtime_warming"
    assert warmup_error.value.headers == {"Retry-After": "2"}


def test_trake_not_ready_redacts_backend_details() -> None:
    endpoint = _endpoint(
        create_app(_context(_TRAKEPipeline(unavailable=True))),
        "/v1/trake/search",
        "POST",
    )

    with pytest.raises(HTTPException) as raised:
        endpoint(_payload())

    assert raised.value.status_code == 503
    assert raised.value.detail["code"] == "trake_not_ready"
    assert "private" not in str(raised.value.detail)
    assert "raw_exception" not in str(raised.value.detail)


def test_trake_export_route_validates_and_uses_required_filename(tmp_path) -> None:
    selected = "aic25-b1-reference"

    class _Repository:
        @staticmethod
        def frame_index_mappings(_batch_id):
            return {"L21_V001": {1: 10, 2: 20}}

        @staticmethod
        def get_frame(batch_id, video_id, frame_idx):
            if batch_id == selected and video_id == "L21_V001" and frame_idx in {10, 20}:
                return SimpleNamespace(timestamp_ms=frame_idx * 100)
            return None

    context = _context(_TRAKEPipeline())
    context.artifact_repository = _Repository()
    context.ledger = SimpleNamespace(
        is_batch_active=lambda _batch: True,
        is_batch_registered=lambda _batch: False,
    )
    context.config = {
        "warmup": {"retry_after_seconds": 2},
        "competition": {"output_dir": str(tmp_path)},
    }
    endpoint = _endpoint(
        create_app(context), "/v1/trake/submissions/export", "POST"
    )
    result = _payload()
    result["batch_filters"] = [selected]
    response = _endpoint(
        create_app(context), "/v1/trake/search", "POST"
    )(result)

    exported = endpoint(
        {
            "query_id": "trake-api-1",
            "batch_id": selected,
            "stage_count": 2,
            "results": response["results"],
        }
    )

    assert exported.filename == "trake-api-1-trake-submission.csv"
    assert (tmp_path / exported.filename).read_text(encoding="utf-8") == "L21_V001,10,20\n"


def test_trake_export_route_rejects_non_monotonic_canonical_frame_ids(
    tmp_path: Path,
) -> None:
    selected = "aic25-b1-reference"

    class _Repository:
        @staticmethod
        def frame_index_mappings(_batch_id):
            return {"L21_V001": {1: 10, 2: 20}}

        @staticmethod
        def get_frame(batch_id, video_id, frame_idx):
            if batch_id == selected and video_id == "L21_V001" and frame_idx in {10, 20}:
                return SimpleNamespace(timestamp_ms=100)
            return None

    context = _context(_TRAKEPipeline())
    context.artifact_repository = _Repository()
    context.ledger = SimpleNamespace(
        is_batch_active=lambda _batch: True,
        is_batch_registered=lambda _batch: False,
    )
    context.config = {
        "warmup": {"retry_after_seconds": 2},
        "competition": {"output_dir": str(tmp_path)},
    }
    endpoint = _endpoint(
        create_app(context), "/v1/trake/submissions/export", "POST"
    )

    with pytest.raises(HTTPException) as raised:
        endpoint(
            {
                "query_id": "trake-api-bad-time",
                "batch_id": selected,
                "stage_count": 2,
                "results": [
                    {
                        "rank": 1,
                        "batch_id": selected,
                        "video_id": "L21_V001",
                        "frame_ids": [10, 20],
                    }
                ],
            }
        )

    assert raised.value.status_code == 422
    assert "strictly increasing" in str(raised.value.detail)
