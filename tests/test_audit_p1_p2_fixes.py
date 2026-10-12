"""Reproducers for the audited P1/P2 KIS pipeline defects."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.contracts import QueryRequest
from src.query.translator import QueryTextAdapter
from src.retrieval.color_filter import AdvancedKISReranker
from src.retrieval.query_plan import build_query_plan
from src.retrieval.spatial_rerank import ObjectEvidence
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline


class _Encoder:
    fingerprint = "audit:encoder"

    def encode_text(self, text: str) -> np.ndarray:
        del text
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _DenseStore:
    def __init__(self, results: list[dict[str, Any]], *, depth: int = 20) -> None:
        self.results = results
        self.depth = depth
        self.depths: list[int] = []

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        del vector, filters
        self.depths.append(top_k)
        return [dict(item) for item in self.results[:top_k]]


class _MetadataStore:
    def __init__(self, video_id: str) -> None:
        self.video_id = video_id

    def search_many(
        self,
        queries: tuple[tuple[str, tuple[str, ...] | None], ...],
        *,
        top_k: int,
        filters: str | None,
    ) -> list[list[dict[str, Any]]]:
        del top_k, filters
        return [
            [
                {
                    "id": f"metadata:{self.video_id}",
                    "score": 10.0,
                    "payload": {
                        "video_id": self.video_id,
                        "title": query,
                    },
                }
            ]
            for query, _locales in queries
        ]


def _frame(video_id: str, frame_idx: int, score: float) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "source_frame_idx": frame_idx,
        "timestamp_ms": frame_idx * 100,
        "score": score,
    }


def test_unbound_color_no_crash() -> None:
    prompt = (
        "CLB FANA tổ chức chương trình từ thiện tại Khánh Hòa, các tình nguyện viên "
        "mặc áo màu [đỏ, xanh], trao các túi quà màu [trắng, vàng] cho trẻ em nghèo. "
        "Màu tím."
    )
    pipeline = TextualKISPipeline(temporal_windows_ms=())
    app = create_app(SimpleNamespace(pipeline=pipeline, config={}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/kis/search",
            json={"query_id": "audit-unbound-color", "original_text": prompt},
        )

    assert response.status_code == 200
    constraints = response.json()["query_processing"]["color_constraints"]
    assert any(
        item.get("color") == "purple" and item.get("bound") is False
        for item in constraints
    )


def _metadata_pipeline(frame_zero_score: float) -> TextualKISPipeline:
    dense = _DenseStore(
        [
            _frame("V_MATCH", 42, 0.90),
            _frame("V_MATCH", 0, frame_zero_score),
        ]
    )
    return TextualKISPipeline(
        dense_channels=(DenseChannel("visual", _Encoder(), dense, depth=20),),
        lexical_channels=(
            LexicalChannel("metadata", _MetadataStore("V_MATCH"), depth=20),
        ),
        temporal_windows_ms=(),
        min_separation_ms=0,
    )


def test_metadata_frame_zero_suppressed() -> None:
    below_threshold = _metadata_pipeline(0.27)
    hits = below_threshold.search("generic metadata keyword")

    assert hits
    assert hits[0]["source_frame_idx"] != 0
    assert all(item["source_frame_idx"] != 0 for item in hits)
    assert hits[0]["metadata_video_boost"] == pytest.approx(1.15)
    assert hits[0]["score"] == pytest.approx(1.15 / 61.0)
    assert below_threshold.last_channel_statuses["metadata"]["result_count"] == 0
    assert below_threshold.last_channel_statuses["metadata"]["metadata_video_count"] == 1

    at_threshold = _metadata_pipeline(0.28)
    threshold_hits = at_threshold.search("generic metadata keyword")
    assert any(item["source_frame_idx"] == 0 for item in threshold_hits)


class _OCRStore:
    def __init__(self, exact_phrase: str) -> None:
        self.exact_phrase = exact_phrase
        self.calls: list[tuple[str, ...]] = []

    def search_many(
        self,
        queries: tuple[tuple[str, tuple[str, ...] | None], ...],
        *,
        top_k: int,
        filters: str | None,
    ) -> list[list[dict[str, Any]]]:
        del top_k, filters
        texts = tuple(query for query, _locales in queries)
        self.calls.append(texts)
        output: list[list[dict[str, Any]]] = []
        for query in texts:
            if query == f'"{self.exact_phrase}"':
                output.append(
                    [
                        {
                            **_frame("V_EXACT", 20, 4.0),
                            "ocr_text": self.exact_phrase,
                        }
                    ]
                )
            else:
                output.append(
                    [
                        {
                            **_frame("V_NORMAL", 10, 9.0),
                            "ocr_text": "generic charity program",
                        }
                    ]
                )
        return output


def test_quoted_ocr_independent_search() -> None:
    phrase = "Chương trình: Trao kinh phí hỗ trợ"
    store = _OCRStore(phrase)
    pipeline = TextualKISPipeline(
        lexical_channels=(LexicalChannel("ocr", store, depth=20),),
        temporal_windows_ms=(),
        min_separation_ms=0,
    )

    hits = pipeline.search(f'Tìm bảng chữ "{phrase}" trong sự kiện')

    assert store.calls
    assert f'"{phrase}"' in store.calls[0]
    assert any(query != f'"{phrase}"' for query in store.calls[0])
    assert hits[0]["video_id"] == "V_EXACT"
    assert pipeline.last_channel_statuses["ocr"]["query_variant_count"] >= 2


def test_quoted_ocr_prior_rejects_backend_phrase_false_positive() -> None:
    phrase = "Trao kinh phí hỗ trợ đồng bào khó khăn"

    class FalsePositiveStore:
        def search_many(self, queries, *, top_k, filters):
            del top_k, filters
            output = []
            for query, _locales in queries:
                if query.startswith('"'):
                    output.append(
                        [{**_frame("V_FALSE", 20, 99.0), "ocr_text": "kinh phí hỗ trợ khác"}]
                    )
                else:
                    output.append(
                        [{**_frame("V_NORMAL", 10, 1.0), "ocr_text": "generic charity"}]
                    )
            return output

    pipeline = TextualKISPipeline(
        lexical_channels=(LexicalChannel("ocr", FalsePositiveStore(), depth=20),),
        temporal_windows_ms=(),
        min_separation_ms=0,
    )
    hits = pipeline.search(f'Bảng ghi "{phrase}"')
    assert all(value["video_id"] != "V_FALSE" for value in hits)


def test_temporal_late_interaction_rewards_clause_order() -> None:
    pipeline = TextualKISPipeline(temporal_windows_ms=())
    plan = build_query_plan("mở cửa trước khi bước vào")

    def candidate(video_id: str, score: float, before_ms: int, current_ms: int) -> dict:
        return {
            **_frame(video_id, 10, score),
            "evidence": [
                {
                    "channel": "visual",
                    "timestamp_ms": before_ms,
                    "nested_channel_ranks": {"visual:temporal_before:query_chunk_1": 1},
                },
                {
                    "channel": "visual",
                    "timestamp_ms": current_ms,
                    "nested_channel_ranks": {"visual:temporal_current:query_chunk_1": 1},
                },
            ],
        }

    reranked = pipeline._rerank_temporal_evidence(
        [candidate("V_BAD", 1.0, 2_000, 1_000), candidate("V_GOOD", 0.98, 1_000, 2_000)],
        plan,
    )
    assert reranked[0]["video_id"] == "V_GOOD"
    assert reranked[0]["temporal_order_verified"] is True


class _FrameRepository:
    def get_frame(
        self,
        batch_id: str,
        video_id: str,
        source_frame_idx: int,
    ) -> None:
        del batch_id, video_id, source_frame_idx
        return None


class _ObjectArchive:
    fingerprint = "audit:objects"
    diagnostics = {"archive_state": "ready"}

    def get(self, frame: Any) -> ObjectEvidence:
        del frame
        return ObjectEvidence("not_applicable")


class _ColorVerifier:
    fingerprint = "audit:color"
    diagnostics = {"cache_hits": 0, "cache_misses": 0}

    def ratio(
        self,
        frame: Any,
        color: str,
        *,
        box: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, bool]:
        del frame, color, box
        return 0.0, False


class _AdvancingClock:
    def __init__(self, step: float = 0.006) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def test_tier2_monotonic_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = AdvancedKISReranker(
        _FrameRepository(),
        _ObjectArchive(),
        _ColorVerifier(),
        spatial_depth=100,
    )
    processing = QueryTextAdapter().prepare("a person left of a car")
    stage1 = [
        {
            "batch_id": "audit",
            "video_id": "V_TIMEOUT",
            "source_frame_idx": index + 1,
            "score": 1.0 - index / 1000,
        }
        for index in range(100)
    ]
    clock = _AdvancingClock()
    wall_clock = time.perf_counter
    monkeypatch.setattr("src.retrieval.color_filter.time.perf_counter", clock)

    wall_started = wall_clock()
    output = reranker(processing, stage1, precision_profile="standard")
    wall_elapsed_ms = (wall_clock() - wall_started) * 1_000

    assert wall_elapsed_ms <= 25.0
    assert output == stage1
    assert all("advanced_rank_delta" not in item for item in output)
    assert reranker.last_diagnostics["state"] == "deadline_exceeded"
    assert reranker.last_diagnostics["precision_profile"] == "automatic"


class _Tier2Recorder:
    fingerprint = "audit:tier2"
    last_diagnostics = {"state": "ready"}

    def __init__(self, events: list[tuple[str, str | None]]) -> None:
        self.events = events

    def __call__(
        self,
        processing: Any,
        candidates: list[dict[str, Any]],
        *,
        precision_profile: str = "standard",
    ) -> list[dict[str, Any]]:
        del processing
        self.events.append(("tier2", precision_profile))
        return candidates


def _routing_run(
    text: str,
    profile: str,
    *,
    advanced_rerank: bool = True,
) -> tuple[list[tuple[str, str | None]], list[int], dict[str, Any]]:
    events: list[tuple[str, str | None]] = []
    store = _DenseStore([_frame("V_ROUTE", 9, 0.9)])

    def raw_refiner(
        plan: Any,
        candidates: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del plan, kwargs
        events.append(("raw", None))
        return candidates

    cross_modal_calls: list[bool] = []

    def cross_modal(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        del args, kwargs
        cross_modal_calls.append(True)
        return []

    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", _Encoder(), store, depth=20),),
        temporal_windows_ms=(),
        raw_refiner=raw_refiner,
        advanced_reranker=_Tier2Recorder(events),
        cross_modal_rescorer=cross_modal,
    )
    pipeline.search(
        QueryRequest("audit-routing", text, precision_profile=profile),
        advanced_rerank=advanced_rerank,
    )
    assert not cross_modal_calls
    return events, store.depths, pipeline.last_query_processing


def test_autonomous_constraint_routing() -> None:
    for text in (
        "a person left of a car",
        "a red car",
        "a street without a car",
    ):
        events, depths, diagnostics = _routing_run(
            text,
            "standard",
            advanced_rerank=False,
        )
        assert events == [("tier2", "automatic"), ("raw", None)]
        assert depths[-1] == 300
        assert diagnostics["precision_profile"] == "automatic"

    events, depths, _diagnostics = _routing_run("a person", "standard")
    assert events == [("raw", None)]
    assert depths[-1] == 20

    routes = []
    for profile in ("standard", "advanced", "experimental_cross_modal"):
        events, depths, diagnostics = _routing_run("a person left of a car", profile)
        routes.append((events, depths[-1], diagnostics["precision_profile"]))
    assert routes[0] == routes[1] == routes[2]
