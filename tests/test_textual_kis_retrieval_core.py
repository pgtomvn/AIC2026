"""CPU-only tests for Textual KIS query planning, fusion, and orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from src.contracts import QueryRequest
from src.retrieval.fusion import (
    ChannelRanking,
    aggregate_temporal_windows,
    canonical_key,
    diversify,
    weighted_rrf,
)
from src.retrieval.query_plan import accent_fold, build_query_plan
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline


def _result(
    video_id: str,
    source_frame_idx: int,
    *,
    score: float = 1.0,
    timestamp_ms: int | None = None,
    shot_id: str | None = None,
    batch_id: str = "aic25-b1",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "video_id": video_id,
        "source_frame_idx": source_frame_idx,
    }
    if timestamp_ms is not None:
        payload["timestamp_ms"] = timestamp_ms
    if shot_id is not None:
        payload["shot_id"] = shot_id
    return {"score": score, "payload": payload}


class RecordingEncoder:
    """Small deterministic encoder fake that records text calls."""

    def __init__(self, vector: Sequence[float] = (3.0, 4.0), error: Exception | None = None):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.error = error
        self.calls: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.vector.copy()


class RecordingStore:
    """Dense/lexical store fake with a common search signature."""

    def __init__(
        self,
        results: Sequence[dict[str, Any]] = (),
        error: Exception | None = None,
    ) -> None:
        self.results = [dict(item) for item in results]
        self.error = error
        self.calls: list[tuple[Any, int, Any]] = []

    def search(self, query: Any, *, top_k: int, filters: Any = None) -> list[dict[str, Any]]:
        recorded_query = query.copy() if isinstance(query, np.ndarray) else query
        self.calls.append((recorded_query, top_k, filters))
        if self.error is not None:
            raise self.error
        return [dict(item) for item in self.results]


class UnavailableRecordingStore(RecordingStore):
    def is_available(self, filters: Any = None) -> bool:
        self.availability_filters = filters
        return False


def test_query_plan_preserves_unicode_quotes_entities_actions_and_attributes() -> None:
    original = "  Người ở Hà Nội nói “Cà phê Đỏ” rồi chạy  "

    plan = build_query_plan(original)

    assert plan.original_text == original
    assert plan.normalized_text == "Người ở Hà Nội nói “Cà phê Đỏ” rồi chạy"
    assert plan.accent_folded_text == "nguoi o ha noi noi “ca phe do” roi chay"
    assert plan.quoted_text == ("Cà phê Đỏ",)
    assert plan.named_entities == ("Hà Nội",)
    assert plan.objects == ("Người",)
    assert plan.actions == ("nói", "chạy")
    assert plan.attributes == ("Đỏ",)
    assert plan.before == "Người ở Hà Nội nói “Cà phê Đỏ”"
    assert plan.current == "chạy"
    assert plan.after is None
    assert accent_fold("ĐẮK  LẮK") == "dak lak"


def test_missing_batch_collection_skips_dense_model_loading() -> None:
    encoder = RecordingEncoder()
    store = UnavailableRecordingStore()
    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", encoder, store),),
        top_k=10,
    )
    request = QueryRequest(
        query_id="missing-batch",
        original_text="người qua cầu",
        batch_filters=("AIC26-B2",),
    )

    assert pipeline.search(request) == []
    assert encoder.calls == []
    assert store.availability_filters == {"batch_id": ["AIC26-B2"]}
    assert pipeline.last_channel_errors == {}
    assert pipeline.last_channel_statuses["visual"] == {
        "state": "not_provisioned",
        "reason_code": "no_compatible_points",
    }


def test_query_request_can_override_every_editable_generated_clause() -> None:
    request = QueryRequest(
        query_id="editable",
        original_text="người cầm điện thoại rồi chạy",
        quoted_text=("BẾN XE",),
        named_entities=("Bến Thành",),
        objects=("điện thoại",),
        actions=("cầm",),
        attributes=("đỏ",),
        before_clause="đứng yên",
        current_clause="cầm điện thoại",
        after_clause="chạy",
    )

    plan = TextualKISPipeline._plan(request)

    assert plan.original_text == request.original_text
    assert plan.quoted_text == request.quoted_text
    assert plan.named_entities == request.named_entities
    assert plan.objects == request.objects
    assert plan.actions == request.actions
    assert plan.attributes == request.attributes
    assert (plan.before, plan.current, plan.after) == (
        "đứng yên",
        "cầm điện thoại",
        "chạy",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Người áo đỏ chạy trước khi xe rời đi",
            ("Người áo đỏ chạy", "xe rời đi", None),
        ),
        (
            "Mở cửa rồi người áo đỏ chạy",
            ("Mở cửa", "người áo đỏ chạy", None),
        ),
        (
            "Người áo đỏ chạy sau khi cửa mở",
            ("cửa mở", "Người áo đỏ chạy", None),
        ),
    ],
)
def test_query_plan_assigns_temporal_clauses_by_event_role(
    text: str,
    expected: tuple[str | None, str | None, str | None],
) -> None:
    plan = build_query_plan(text)

    assert (plan.before, plan.current, plan.after) == expected


@pytest.mark.parametrize("value", ["", "   ", None, 42])
def test_query_plan_rejects_empty_or_non_text_queries(value: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        build_query_plan(value)  # type: ignore[arg-type]


def test_weighted_rrf_deduplicates_each_channel_and_retains_rank_evidence() -> None:
    visual = ChannelRanking(
        "visual",
        (
            _result("L21_V002", 20, score=0.99),
            _result("L21_V001", 10, score=0.01),
            _result("L21_V002", 20, score=0.50),
        ),
    )
    lexical = ChannelRanking(
        "ocr",
        (
            _result("L21_V001", 10, score=99.0),
            _result("L21_V003", 30, score=98.0),
        ),
    )

    fused = weighted_rrf((visual, lexical), k=0)

    assert [canonical_key(item) for item in fused] == [
        ("L21_V001", 10),
        ("L21_V002", 20),
        ("L21_V003", 30),
    ]
    assert fused[0]["score"] == pytest.approx(1 / 2 + 1)
    assert fused[0]["channel_ranks"] == {"visual": 2, "ocr": 1}
    assert fused[1]["score"] == pytest.approx(1.0)
    assert fused[1]["evidence"] == [{"channel": "visual", "rank": 1, "raw_score": 0.99}]


def test_weighted_rrf_is_deterministic_on_ties_and_tolerates_missing_modalities() -> None:
    rankings = (
        ChannelRanking("visual", (_result("L21_V002", 5),), weight=0.8),
        ChannelRanking("ocr", (), weight=0.8),
        ChannelRanking("asr", (_result("L21_V001", 9),), weight=0.8),
    )

    first = weighted_rrf(rankings)
    second = weighted_rrf(rankings)

    assert first == second
    assert [canonical_key(item) for item in first] == [
        ("L21_V001", 9),
        ("L21_V002", 5),
    ]


def test_weighted_rrf_preserves_ocr_polygons_as_evidence() -> None:
    result = _result("L21_V001", 10, score=0.8)
    result["payload"] = {
        **result["payload"],
        "segment_id": "ocr-1",
        "kind": "ocr",
        "text_raw": "quán cà phê",
        "polygons": [[[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]],
    }

    fused = weighted_rrf((ChannelRanking("ocr", (result,)),))

    assert fused[0]["evidence"][0]["segment_id"] == "ocr-1"
    assert fused[0]["evidence"][0]["polygons"][0][2] == [3.0, 4.0]


def test_fusion_prefers_source_frame_idx_and_removes_legacy_frame_id() -> None:
    result = {
        "score": 0.8,
        "payload": {
            "video_id": "L21_V001",
            "source_frame_idx": 181,
            "frame_id": 7,
        },
    }

    fused = weighted_rrf((ChannelRanking("visual", (result,)),))

    assert canonical_key(result) == ("L21_V001", 181)
    assert fused[0]["source_frame_idx"] == 181
    assert "frame_id" not in fused[0]


@pytest.mark.parametrize(
    "result",
    [
        {"video_id": "", "source_frame_idx": 1},
        {"video_id": "L21_V001", "source_frame_idx": -1},
        {"video_id": "L21_V001", "source_frame_idx": True},
    ],
)
def test_canonical_key_rejects_invalid_frame_identity(result: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError):
        canonical_key(result)


def test_temporal_window_aggregation_never_crosses_shot_boundaries() -> None:
    candidates = [
        {
            "video_id": "L21_V001",
            "source_frame_idx": 10,
            "timestamp_ms": 1000,
            "shot_id": "shot-a",
            "score": 0.9,
        },
        {
            "video_id": "L21_V001",
            "source_frame_idx": 20,
            "timestamp_ms": 2000,
            "shot_id": "shot-a",
            "score": 0.3,
        },
        {
            "video_id": "L21_V001",
            "source_frame_idx": 15,
            "timestamp_ms": 1500,
            "shot_id": "shot-b",
            "score": 0.8,
        },
    ]

    aggregated = aggregate_temporal_windows(candidates, window_ms=4000)
    by_frame = {item["source_frame_idx"]: item for item in aggregated}

    assert set(by_frame) == {10, 15}
    assert by_frame[10]["window_evidence_count"] == 2
    assert by_frame[10]["score"] == pytest.approx(0.9 + (0.9 + 0.3) / 2)
    assert by_frame[15]["window_evidence_count"] == 1
    assert by_frame[15]["score"] == pytest.approx(1.6)


def test_diversify_enforces_temporal_nms_without_suppressing_distinct_shots() -> None:
    candidates = [
        {
            "video_id": "L21_V001",
            "source_frame_idx": 1,
            "timestamp_ms": 0,
            "shot_id": "shot-a",
            "score": 1.0,
        },
        {
            "video_id": "L21_V001",
            "source_frame_idx": 1,
            "timestamp_ms": 0,
            "shot_id": "shot-a",
            "score": 0.99,
        },
        {
            "video_id": "L21_V001",
            "source_frame_idx": 2,
            "timestamp_ms": 2999,
            "shot_id": "shot-a",
            "score": 0.98,
        },
        {
            "video_id": "L21_V001",
            "source_frame_idx": 3,
            "timestamp_ms": 2999,
            "shot_id": "shot-b",
            "score": 0.97,
        },
        {
            "video_id": "L21_V001",
            "source_frame_idx": 4,
            "timestamp_ms": 3000,
            "shot_id": "shot-a",
            "score": 0.96,
        },
    ]

    selected = diversify(
        candidates,
        early_per_video=10,
        late_per_video=10,
    )

    assert [item["source_frame_idx"] for item in selected] == [1, 3, 4]
    assert [item["rank"] for item in selected] == [1, 2, 3]


def test_diversify_applies_early_and_late_per_video_quotas() -> None:
    candidates = [
        {
            "video_id": "L21_V001",
            "source_frame_idx": index,
            "timestamp_ms": index * 100,
            "shot_id": f"a-{index}",
            "score": 100.0 - index,
        }
        for index in range(2)
    ]
    candidates.extend(
        {
            "video_id": f"L22_V{index:03d}",
            "source_frame_idx": index,
            "timestamp_ms": 0,
            "shot_id": "only",
            "score": 90.0 - index,
        }
        for index in range(1, 19)
    )
    candidates.extend(
        {
            "video_id": "L21_V001",
            "source_frame_idx": index,
            "timestamp_ms": index * 4000,
            "shot_id": f"a-{index}",
            "score": 50.0 - index,
        }
        for index in range(2, 7)
    )

    selected = diversify(candidates)
    a_hits = [item for item in selected if item["video_id"] == "L21_V001"]

    assert len([item for item in selected[:20] if item["video_id"] == "L21_V001"]) == 2
    assert len(a_hits) == 5


def test_pipeline_propagates_batch_filters_and_returns_only_canonical_frames() -> None:
    encoder = RecordingEncoder()
    dense_store = RecordingStore(
        [
            _result("L21_V001", 181, score=0.1),
            _result("L21_V002", 240, score=0.9),
        ]
    )
    lexical_store = RecordingStore(
        [
            {
                "score": 99.0,
                "payload": {
                    "batch_id": "aic25-b1",
                    "video_id": "L21_V001",
                    "frame_id": 181,
                },
            }
        ]
    )
    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", encoder, dense_store, 1.0, 17),),
        lexical_channels=(LexicalChannel("ocr", lexical_store, 0.8, 11),),
        temporal_windows_ms=(),
    )
    request = QueryRequest(
        query_id="kis-1",
        original_text="  Cà   phê  ",
        batch_filters=("aic25-b1", "aic26-b2"),
        limit=10,
    )

    hits = pipeline.search(request, refine_raw=False)

    assert encoder.calls == ["Cà phê"]
    dense_query, dense_depth, dense_filter = dense_store.calls[0]
    assert dense_query == pytest.approx(np.asarray([0.6, 0.8], dtype=np.float32))
    assert dense_depth == 17
    assert dense_filter == {"batch_id": ["aic25-b1", "aic26-b2"]}
    assert lexical_store.calls == [
        (
            "Cà phê",
            11,
            "batch_id = 'aic25-b1' OR batch_id = 'aic26-b2'",
        )
    ]
    assert hits[0]["video_id"] == "L21_V001"
    assert hits[0]["source_frame_idx"] == 181
    assert "frame_id" not in hits[0]
    assert [item["rank"] for item in hits] == [1, 2]


def test_pipeline_isolates_a_failed_channel_and_keeps_successful_modalities() -> None:
    failed_store = RecordingStore(error=RuntimeError("qdrant unavailable"))
    lexical_store = RecordingStore([_result("L21_V001", 12)])
    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", RecordingEncoder(), failed_store),),
        lexical_channels=(LexicalChannel("ocr", lexical_store),),
        temporal_windows_ms=(),
    )

    hits = pipeline.search("người đi bộ", refine_raw=False)

    assert [canonical_key(item) for item in hits] == [("L21_V001", 12)]
    assert pipeline.last_channel_errors == {"visual": "qdrant unavailable"}
    assert pipeline.last_channel_statuses["visual"]["state"] == "service_unavailable"


def test_pipeline_fingerprint_is_stable_and_changes_with_retrieval_config() -> None:
    first = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", RecordingEncoder(), RecordingStore([])),),
        temporal_windows_ms=(),
        rrf_k=60,
    )
    same = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", RecordingEncoder(), RecordingStore([])),),
        temporal_windows_ms=(),
        rrf_k=60,
    )
    changed = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", RecordingEncoder(), RecordingStore([])),),
        temporal_windows_ms=(),
        rrf_k=40,
    )

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_pipeline_resolves_timestamped_text_segments_to_canonical_frames() -> None:
    lexical_store = RecordingStore(
        [
            {
                "score": 1.0,
                "payload": {
                    "batch_id": "aic25-b1",
                    "video_id": "L21_V001",
                    "start_ms": 1000,
                    "end_ms": 2000,
                    "text_raw": "xin chào",
                },
            }
        ]
    )
    resolved_inputs: list[dict[str, Any]] = []

    def resolve(result: dict[str, Any]) -> dict[str, Any]:
        resolved_inputs.append(result)
        payload = dict(result["payload"])
        payload.update(
            {
                "source_frame_idx": 45,
                "timestamp_ms": 1500,
                "shot_id": "shot-1",
            }
        )
        return {**result, "payload": payload}

    pipeline = TextualKISPipeline(
        lexical_channels=(LexicalChannel("asr", lexical_store),),
        frame_resolver=resolve,
        temporal_windows_ms=(),
    )

    hits = pipeline.search("xin chào", refine_raw=False)

    assert len(resolved_inputs) == 1
    assert [canonical_key(item) for item in hits] == [("L21_V001", 45)]


def test_pipeline_enriches_existing_text_frame_identity_without_changing_rank_data() -> None:
    resolved_inputs: list[dict[str, Any]] = []

    def resolve(result: dict[str, Any]) -> dict[str, Any]:
        resolved_inputs.append(result)
        payload = dict(result["payload"])
        payload.update(
            {
                "source_frame_idx": 0,
                "timestamp_ms": 2_400,
                "shot_id": "shot-canonical",
                "keyframe_ordinal": 7,
            }
        )
        return {**result, "payload": payload}

    pipeline = TextualKISPipeline(frame_resolver=resolve, temporal_windows_ms=())
    source = {
        "rank": 3,
        "score": 0.875,
        "payload": {
            "batch_id": "aic25-b1",
            "video_id": "L21_V001",
            "source_frame_idx": 0,
            "timestamp_ms": 0,
            "kind": "asr",
            "segment_id": "asr:aic25-b1:L21_V001:0",
            "text_raw": "xin chào",
        },
    }

    (result,) = pipeline._resolve_results((source,))

    assert resolved_inputs == [source]
    assert result["rank"] == 3
    assert result["score"] == 0.875
    assert result["payload"]["source_frame_idx"] == 0
    assert result["payload"]["timestamp_ms"] == 2_400
    assert result["payload"]["shot_id"] == "shot-canonical"
    assert result["payload"]["keyframe_ordinal"] == 7
    assert result["payload"]["text_raw"] == "xin chào"


def test_explicit_temporal_clauses_reach_raw_refinement_unchanged() -> None:
    store = RecordingStore([_result("L21_V001", 181)])
    captured_plans = []

    def refiner(plan, candidates):
        captured_plans.append(plan)
        return [dict(item) for item in candidates]

    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", RecordingEncoder(), store),),
        raw_refiner=refiner,
        temporal_windows_ms=(),
    )
    request = QueryRequest(
        query_id="kis-clauses",
        original_text="người rời quán",
        mode="interactive",
        before_clause="người thanh toán",
        current_clause="người rời quán",
        after_clause="xe chạy đi",
    )

    pipeline.search(request)

    assert len(captured_plans) == 1
    assert captured_plans[0].before == "người thanh toán"
    assert captured_plans[0].current == "người rời quán"
    assert captured_plans[0].after == "xe chạy đi"
    assert captured_plans[0].original_text == request.original_text


def test_refine_applies_normalized_rocchio_vector_before_fresh_search() -> None:
    encoder = RecordingEncoder((1.0, 0.0))
    store = RecordingStore([_result("L21_V001", 181)])
    feedback_calls: list[tuple[str, Sequence[tuple[str, int]]]] = []

    def feedback(channel: str, keys: Sequence[tuple[str, int]]) -> np.ndarray:
        feedback_calls.append((channel, keys))
        if keys[0][1] == 10:
            return np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", encoder, store),),
        feedback_provider=feedback,
        temporal_windows_ms=(),
    )

    hits = pipeline.refine(
        "target",
        positive_ids=(("L21_V001", 10), ("L21_V002", 10)),
        negative_ids=(("L21_V003", 20),),
        alpha=1.0,
        beta=0.75,
        gamma=0.25,
    )

    searched_vector = store.calls[0][0]
    assert searched_vector == pytest.approx(np.asarray([2**-0.5, 2**-0.5], dtype=np.float32))
    assert feedback_calls == [
        ("visual", (("L21_V001", 10), ("L21_V002", 10))),
        ("visual", (("L21_V003", 20),)),
    ]
    assert [canonical_key(item) for item in hits] == [("L21_V001", 181)]


def test_refine_requires_feedback_and_dense_channel_dependencies() -> None:
    with pytest.raises(RuntimeError, match="feedback_provider"):
        TextualKISPipeline().refine("query")

    pipeline = TextualKISPipeline(feedback_provider=lambda _channel, _keys: np.ones((1, 2)))
    with pytest.raises(RuntimeError, match="dense channel"):
        pipeline.refine("query")


def test_refine_falls_back_to_channel_that_contains_feedback_frames() -> None:
    first_store = RecordingStore([_result("L21_V001", 1)])
    second_store = RecordingStore([_result("L21_V002", 2)])

    def feedback(channel: str, _keys: Sequence[tuple[str, int]]) -> np.ndarray:
        if channel == "native":
            raise ValueError("frame is absent from the native batch")
        return np.asarray([[0.0, 1.0]], dtype=np.float32)

    pipeline = TextualKISPipeline(
        dense_channels=(
            DenseChannel("native", RecordingEncoder((1.0, 0.0)), first_store),
            DenseChannel("organizer", RecordingEncoder((1.0, 0.0)), second_store),
        ),
        feedback_provider=feedback,
        temporal_windows_ms=(),
    )

    hits = pipeline.refine("target", positive_ids=(("L21_V002", 2),))

    assert second_store.calls[0][0] == pytest.approx(np.asarray([0.8, 0.6], dtype=np.float32))
    assert {canonical_key(hit) for hit in hits} == {
        ("L21_V001", 1),
        ("L21_V002", 2),
    }
