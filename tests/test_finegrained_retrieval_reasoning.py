"""Offline reproducers for long-tail, fine-grained, and bilingual KIS queries."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pytest

from src.contracts import FrameRecord, QueryRequest
from src.evaluation.textual_kis import GroundTruthInterval, GroundTruthQuery
from src.evaluation.tuning import (
    CachedChannelRanking,
    observed_threshold_midpoints,
    tune_semantic_view_policy,
)
from src.query.entity_extractor import NamedEntity
from src.query.query_decomposer import decompose_query
from src.query.translator import QueryTextAdapter
from src.retrieval.color_filter import AdvancedKISReranker
from src.retrieval.fusion import canonical_key
from src.retrieval.query_plan import build_query_plan
from src.retrieval.spatial_rerank import (
    ObjectDetection,
    ObjectEvidence,
    labels_compatible,
    labels_positive_compatible,
)
from src.retrieval.textual_kis import DenseChannel, TextualKISPipeline

CULINARY_QUERY = (
    "A chef plating vegetarian fresh spring rolls (gỏi cuốn chay) wrapped in "
    "yellow and purple rice paper filled with tofu and rolled greens, delicately "
    "garnished on the plate with fresh green leaves and purple-and-yellow edible "
    "pansy flowers."
)
MULTI_VEHICLE_QUERY = (
    "Footage capturing vehicles wading through flood water, where a yellow car, "
    "a red car, and a black car prepare one after another to cross a bridge."
)


def _frame() -> FrameRecord:
    return FrameRecord(
        batch_id="aic25-b1-reference",
        video_id="L26_V056",
        keyframe_ordinal=1,
        source_frame_idx=6362,
        pts=6362,
        time_base="1/25",
        timestamp_ms=254_480,
        fps_num=25,
        fps_den=1,
        shot_id="finegrained-seed",
        sampling_reasons=("reference",),
        image_uri="memory://frame",
        video_uri="memory://video",
        source_sha256="a" * 64,
        extractor_version="test:v1",
    )


def _hit(video_id: str, frame_idx: int, score: float = 1.0) -> dict[str, Any]:
    return {
        "score": score,
        "payload": {
            "batch_id": "aic25-b1-reference",
            "video_id": video_id,
            "source_frame_idx": frame_idx,
            "timestamp_ms": frame_idx * 40,
        },
    }


def test_long_query_decomposes_into_stable_semantic_views() -> None:
    first = QueryTextAdapter().prepare(CULINARY_QUERY)
    second = QueryTextAdapter().prepare(CULINARY_QUERY)

    assert first.decomposition is not None
    assert second.decomposition is not None
    views = first.decomposition.semantic_views_en
    assert views == second.decomposition.semantic_views_en
    assert [view.role for view in views] == [
        "global",
        "action",
        "foreground_subject",
        "distinctive_attributes",
    ]
    assert len(views) <= 4
    assert "yellow and purple rice paper" in views[-1].text
    assert "yellow and purple edible pansy flowers" in views[-1].text


def test_parenthetical_vietnamese_is_lexical_not_primary_visual_text() -> None:
    adapter = QueryTextAdapter()
    processing = adapter.prepare(CULINARY_QUERY)

    assert processing.translation.detected_language == "en"
    dense = adapter.dense_variants("organizer_clip", processing)
    lexical = adapter.lexical_variants("ocr", processing)
    assert dense
    assert all("gỏi cuốn chay" not in variant.text for variant in dense)
    assert any(variant.text == "gỏi cuốn chay" for variant in lexical)
    assert processing.decomposition is not None
    assert [cue.text for cue in processing.decomposition.parenthetical_cues] == [
        "gỏi cuốn chay"
    ]


def test_operator_culinary_hint_is_not_hypernymized_as_organization() -> None:
    hint = NamedEntity("gỏi cuốn chay", "organization", source="operator_hint")
    decomposition = decompose_query(
        "A chef plating gỏi cuốn chay",
        ((hint, "optional"),),
        english_text="A chef plating gỏi cuốn chay",
    )

    assert decomposition.visual_description_en == "A chef plating gỏi cuốn chay"
    assert "an organization" not in decomposition.visual_description_en
    assert decomposition.lexical_aliases == ("gỏi cuốn chay",)


def test_coordinated_colors_bind_to_the_intended_subjects() -> None:
    decomposition = QueryTextAdapter().prepare(CULINARY_QUERY).decomposition
    assert decomposition is not None
    assert {(value.subject, value.color) for value in decomposition.color_constraints} == {
        ("rice paper", "yellow"),
        ("rice paper", "purple"),
        ("leaves", "green"),
        ("edible pansy flowers", "yellow"),
        ("edible pansy flowers", "purple"),
    }


def test_ordering_idiom_and_prospective_relation_do_not_create_false_clauses() -> None:
    plan = build_query_plan(MULTI_VEHICLE_QUERY)

    assert (plan.before, plan.current, plan.after) == (
        None,
        MULTI_VEHICLE_QUERY,
        None,
    )
    assert [value.source_text for value in plan.sequence_constraints] == [
        "one after another"
    ]
    assert plan.sequence_constraints[0].prospective_action == "cross"
    assert [(value.action, value.anchor) for value in plan.prospective_relations] == [
        ("cross", "a bridge")
    ]


@pytest.mark.parametrize(
    ("text", "source"),
    (
        ("Vehicles wait one behind another near a bridge.", "one behind another"),
        ("Car after car moves through flood water.", "Car after car"),
        ("Các xe lần lượt đi qua vùng nước ngập.", "lần lượt"),
        ("Các xe nối đuôi nhau dưới cầu.", "nối đuôi nhau"),
    ),
)
def test_sequence_idiom_variants_remain_ordered_groups(
    text: str,
    source: str,
) -> None:
    plan = build_query_plan(text)

    assert plan.before is None
    assert plan.after is None
    assert len(plan.sequence_constraints) == 1
    assert plan.sequence_constraints[0].kind == "ordered_group"
    assert plan.sequence_constraints[0].source_text == source


def test_repeated_colored_cars_become_distinct_vehicle_instances() -> None:
    processing = QueryTextAdapter().prepare(MULTI_VEHICLE_QUERY)
    decomposition = processing.decomposition
    assert decomposition is not None

    assert [value.instance_id for value in decomposition.object_instances] == [
        "vehicle_1",
        "vehicle_2",
        "vehicle_3",
    ]
    assert [value.attributes for value in decomposition.object_instances] == [
        ("yellow",),
        ("red",),
        ("black",),
    ]
    assert {value.group_id for value in decomposition.object_instances} == {
        "vehicle_group_1"
    }
    assert [
        (value.subject, value.color, value.node_id)
        for value in decomposition.color_constraints
    ] == [
        ("car", "yellow", "vehicle_1"),
        ("car", "red", "vehicle_2"),
        ("car", "black", "vehicle_3"),
    ]
    assert [value.label for value in decomposition.object_nodes] == [
        "vehicle",
        "vehicle",
        "vehicle",
    ]
    views = decomposition.semantic_views_en
    assert [value.role for value in views] == [
        "global",
        "scene_action",
        "instance_attributes",
        "relation_relaxed",
    ]
    assert views[-1].text == (
        "yellow, red and black vehicles in water at a bridge"
    )
    assert views[-1].alternatives == (
        "yellow, red and black vehicles in water by a bridge",
        "yellow, red and black vehicles in water near a bridge",
    )
    diagnostics = processing.to_dict()
    assert diagnostics["prospective_relations"][0]["verification_policy"] == (
        "positive_only"
    )


class _CoverageEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _PartialDenseStore:
    def is_available(self, _filters: Any = None) -> bool:
        return True

    def count(self, _filters: Any = None) -> int:
        return 569

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("partial native channel must not be searched")


def test_partial_native_channel_is_excluded_from_fusion() -> None:
    encoder = _CoverageEncoder()
    pipeline = TextualKISPipeline(
        dense_channels=(
            DenseChannel(
                "visual",
                encoder,
                _PartialDenseStore(),
                require_complete=True,
                expected_count=176_707,
            ),
        ),
        temporal_windows_ms=(),
    )
    request = QueryRequest(
        query_id="coverage",
        original_text=CULINARY_QUERY,
        batch_filters=("aic25-b1-reference",),
    )

    assert pipeline.search(request, limit=3) == []
    assert encoder.calls == []
    assert pipeline.last_channel_statuses["visual"] == {
        "state": "not_provisioned",
        "reason_code": "collection_coverage_incomplete",
        "count": 569,
        "expected_count": 176_707,
        "coverage": pytest.approx(569 / 176_707),
    }


class _ViewEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if text.startswith("A chef plating vegetarian fresh spring rolls") and ";" not in text:
            index = 0 if "wrapped in" in text else 1
        elif text.startswith("rice paper;"):
            index = 2
        else:
            index = 3
        vector = np.zeros(4, dtype=np.float32)
        vector[index] = 1.0
        return vector


class _ViewStore:
    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        filters: Any = None,
    ) -> list[dict[str, Any]]:
        del top_k, filters
        target = _hit("L26_V056", 6362)
        by_view = {
            0: [
                _hit("L26_V001", 10),
                _hit("L26_V002", 20),
                _hit("L26_V003", 30),
                _hit("L26_V004", 40),
                target,
            ],
            1: [_hit("L26_V101", 10), _hit("L26_V102", 20), target],
            2: [_hit("L26_V201", 10), target],
            3: [target, _hit("L26_V301", 10)],
        }
        return [dict(item) for item in by_view[int(np.argmax(vector))]]


def test_semantic_late_fusion_recovers_detail_candidate_in_top_three() -> None:
    encoder = _ViewEncoder()
    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("organizer_clip", encoder, _ViewStore()),),
        semantic_view_fusion="weighted_rrf",
        temporal_windows_ms=(),
        min_separation_ms=0,
    )

    hits = pipeline.search(CULINARY_QUERY, limit=3)

    assert ("L26_V056", 6362) in {canonical_key(item) for item in hits[:3]}
    assert len(encoder.calls) == 4
    assert pipeline.last_query_processing["rank_trace"]["tier1_fusion"]
    assert pipeline.last_query_processing["stage_rank_delta"]


class _VehicleViewEncoder:
    def encode_text(self, text: str) -> np.ndarray:
        vector = np.zeros(4, dtype=np.float32)
        if " at a bridge" in text or " by a bridge" in text or " near a bridge" in text:
            vector[3] = 1.0
        elif ";" in text:
            vector[2] = 1.0
        elif text.startswith("Footage capturing vehicles wading") and "," not in text:
            vector[1] = 1.0
        else:
            vector[0] = 1.0
        return vector


class _VehicleViewStore:
    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        filters: Any = None,
    ) -> list[dict[str, Any]]:
        del top_k, filters
        index = int(np.argmax(vector))
        if index == 3:
            return [
                _hit("L21_V003", 26220),
                _hit("L21_V030", 300),
            ]
        return [
            _hit(f"L29_V{index + 1:03d}", index + 10),
            _hit(f"L30_V{index + 1:03d}", index + 20),
        ]


def test_relation_relaxed_view_rescues_multi_vehicle_target_before_tier2() -> None:
    pipeline = TextualKISPipeline(
        dense_channels=(
            DenseChannel(
                "organizer_clip",
                _VehicleViewEncoder(),
                _VehicleViewStore(),
            ),
        ),
        semantic_view_fusion="max_rrf",
        temporal_windows_ms=(),
        min_separation_ms=0,
    )

    hits = pipeline.search(MULTI_VEHICLE_QUERY, limit=3, refine_raw=False)

    assert canonical_key(hits[0]) == ("L21_V003", 26220)
    assert pipeline.last_channel_statuses["organizer_clip"][
        "semantic_view_roles"
    ] == [
        "global",
        "scene_action",
        "instance_attributes",
        "relation_relaxed",
    ]
    view_trace = pipeline.last_query_processing["semantic_view_rankings"]
    relaxed_rows = [
        rows for name, rows in view_trace.items() if "relation_relaxed" in name
    ]
    assert len(relaxed_rows) == 3
    assert all(rows[0]["source_frame_idx"] == 26220 for rows in relaxed_rows)
    assert pipeline.last_query_processing["sequence_evidence"] == {
        "state": "inconclusive",
        "reason_code": "instance_tracking_not_provisioned",
        "positive_only": True,
        "shot_bounded": True,
        "constraints": [
            {
                "kind": "ordered_group",
                "source_text": "one after another",
                "prospective_action": "cross",
            }
        ],
    }


def test_offline_semantic_cache_bypasses_runtime_global_only_without_mutation() -> None:
    pipeline = TextualKISPipeline(
        dense_channels=(
            DenseChannel(
                "organizer_clip",
                _VehicleViewEncoder(),
                _VehicleViewStore(),
            ),
        ),
        semantic_view_fusion="global_only",
        temporal_windows_ms=(),
    )

    rankings = pipeline.collect_semantic_view_rankings(
        MULTI_VEHICLE_QUERY,
        channel_name="organizer_clip",
    )

    assert [ranking.name for ranking in rankings] == [
        "global",
        "scene_action",
        "instance_attributes",
        "relation_relaxed",
    ]
    relaxed = next(
        ranking for ranking in rankings if ranking.name == "relation_relaxed"
    )
    assert canonical_key(relaxed.results[0]) == ("L21_V003", 26220)
    assert pipeline.semantic_view_fusion == "global_only"


class _Repository:
    frame = _frame()

    def get_frame(self, batch_id: str, video_id: str, source_frame_idx: int):
        return (
            self.frame
            if (batch_id, video_id, source_frame_idx)
            == (self.frame.batch_id, self.frame.video_id, self.frame.source_frame_idx)
            else None
        )


class _OOVObjects:
    fingerprint = "objects:oov"
    diagnostics = {"archive_state": "ready"}

    def get(self, _frame: FrameRecord) -> ObjectEvidence:
        return ObjectEvidence(
            "ready",
            (ObjectDetection("Flower", 0.95, (0.1, 0.1, 0.5, 0.5)),),
        )


class _SingleCarObjects:
    fingerprint = "objects:single-car"
    diagnostics = {"archive_state": "ready"}

    def get(self, _frame: FrameRecord) -> ObjectEvidence:
        return ObjectEvidence(
            "ready",
            (ObjectDetection("Car", 0.95, (0.1, 0.1, 0.3, 0.3)),),
        )


class _AmphibiousVehicleObjects:
    fingerprint = "objects:amphibious-vehicles"
    diagnostics = {"archive_state": "ready"}

    def get(self, _frame: FrameRecord) -> ObjectEvidence:
        return ObjectEvidence(
            "ready",
            (
                ObjectDetection("Tank", 0.92, (0.55, 0.35, 0.90, 0.64)),
                ObjectDetection("Boat", 0.76, (0.40, 0.72, 0.65, 0.99)),
                ObjectDetection("Car", 0.60, (0.44, 0.55, 0.53, 0.66)),
            ),
        )


class _FallbackColor:
    fingerprint = "color:fallback"
    diagnostics = {"cache_hits": 0, "cache_misses": 2}

    def __init__(self, ratio: float) -> None:
        self.value = ratio

    def ratio(self, _frame: FrameRecord, _color: str, *, box=None):
        del box
        return self.value, False


def _candidate() -> dict[str, Any]:
    frame = _Repository.frame
    return {
        "batch_id": frame.batch_id,
        "video_id": frame.video_id,
        "source_frame_idx": frame.source_frame_idx,
        "score": 1.0,
    }


def test_oov_unlocalized_color_mismatch_is_neutral_but_fallback_can_reward() -> None:
    neutral = AdvancedKISReranker(
        _Repository(), _OOVObjects(), _FallbackColor(0.01)
    )
    neutral_output = neutral(QueryTextAdapter().prepare("purple tofu"), [_candidate()])
    assert neutral_output[0]["score"] == pytest.approx(1.0)
    assert neutral_output[0]["color_rerank_state"] == "inconclusive"
    assert "color_penalty_applied" not in neutral_output[0]
    assert neutral_output[0]["verification_inconclusive_reason"] == (
        "object_or_roi_unlocalized"
    )

    positive = AdvancedKISReranker(
        _Repository(), _OOVObjects(), _FallbackColor(0.20)
    )
    positive_output = positive(QueryTextAdapter().prepare("purple tofu"), [_candidate()])
    assert positive_output[0]["score"] > 1.0
    assert positive_output[0]["color_reward_applied"] is True
    assert positive_output[0]["color_rerank_state"] == "verified"


def test_one_roi_cannot_satisfy_three_colored_vehicle_instances() -> None:
    reranker = AdvancedKISReranker(
        _Repository(),
        _SingleCarObjects(),
        _FallbackColor(0.01),
        spatial_depth=1,
        color_depth=1,
    )

    output = reranker(
        QueryTextAdapter().prepare(MULTI_VEHICLE_QUERY),
        [_candidate()],
    )

    assert output[0]["score"] == pytest.approx(1.0)
    assert "color_penalty_applied" not in output[0]
    color_evidence = next(
        value for value in output[0]["evidence"] if value["channel"] == "hsv_color"
    )
    assert color_evidence["assigned_instance_count"] == 1
    assert color_evidence["instance_assignment_complete_exact"] is False
    assert sum(
        value.get("matched_label") == "Car"
        for value in color_evidence["constraints"]
    ) == 1


def test_vehicle_ontology_is_broad_for_positive_evidence_only() -> None:
    assert labels_positive_compatible("vehicle", "Tank") is True
    assert labels_positive_compatible("vehicle", "Boat") is True
    assert labels_positive_compatible("vehicle", "Car") is True
    assert labels_compatible("car", "Tank") is False
    assert labels_compatible("car", "Boat") is False

    reranker = AdvancedKISReranker(
        _Repository(),
        _AmphibiousVehicleObjects(),
        _FallbackColor(0.20),
        spatial_depth=1,
        color_depth=1,
    )
    output = reranker(
        QueryTextAdapter().prepare(MULTI_VEHICLE_QUERY),
        [_candidate()],
    )

    assert output[0]["score"] > 1.0
    assert output[0]["color_reward_applied"] is True
    assert "color_penalty_applied" not in output[0]
    color_evidence = next(
        value for value in output[0]["evidence"] if value["channel"] == "hsv_color"
    )
    assert color_evidence["assigned_instance_count"] == 3
    assert color_evidence["instance_assignment_complete_exact"] is False


def test_oov_negation_absence_is_inconclusive_not_verified() -> None:
    reranker = AdvancedKISReranker(
        _Repository(), _OOVObjects(), _FallbackColor(0.0)
    )

    output = reranker(QueryTextAdapter().prepare("a plate without tofu"), [_candidate()])

    assert output[0]["score"] == pytest.approx(1.0)
    assert output[0]["negation_rerank_state"] == "inconclusive"
    assert "negation_penalty_applied" not in output[0]


class _AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.006
        return self.value


def test_tier2_deadline_returns_exact_stage1_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reranker = AdvancedKISReranker(
        _Repository(), _OOVObjects(), _FallbackColor(0.20), spatial_depth=100
    )
    stage1 = [
        {
            "batch_id": "aic25-b1-reference",
            "video_id": f"L26_V{index:03d}",
            "source_frame_idx": index + 1,
            "score": 1.0 - index / 1000,
        }
        for index in range(100)
    ]
    before = json.dumps(stage1, sort_keys=True, separators=(",", ":"))
    wall_clock = time.perf_counter
    monkeypatch.setattr(
        "src.retrieval.color_filter.time.perf_counter", _AdvancingClock()
    )

    wall_started = wall_clock()
    output = reranker(QueryTextAdapter().prepare("a purple tofu"), stage1)
    wall_ms = (wall_clock() - wall_started) * 1_000

    assert wall_ms <= 25.0
    assert json.dumps(output, sort_keys=True, separators=(",", ":")) == before
    assert reranker.last_diagnostics["state"] == "deadline_exceeded"


def test_color_threshold_candidates_come_from_observed_class_boundaries() -> None:
    assert observed_threshold_midpoints(
        positive_values=(0.14, 0.16, 0.20),
        negative_values=(0.00, 0.07, 0.09),
    ) == (pytest.approx(0.115),)


def test_semantic_view_tuning_derives_weights_and_searches_every_rrf_k() -> None:
    roles = ("global", "scene_action", "instance_attributes", "relation_relaxed")

    def ranking(target_rank: int, prefix: str) -> CachedChannelRanking:
        rows = [
            _hit(f"L29_{prefix}_{index}", index)
            for index in range(1, target_rank)
        ]
        rows.append(_hit("L21_V003", 26220))
        rows.extend(
            _hit(f"L30_{prefix}_{index}", index)
            for index in range(target_rank + 1, 5)
        )
        return CachedChannelRanking(tuple(rows), latency_ms=1.0)

    cached = {
        "train-1": {
            role: ranking(index, f"train1-{role}")
            for index, role in enumerate(roles, start=1)
        },
        "train-2": {
            role: ranking(index, f"train2-{role}")
            for index, role in enumerate(roles, start=1)
        },
        "validation-1": {
            role: ranking(1 if role == "relation_relaxed" else 4, f"val-{role}")
            for role in roles
        },
    }
    interval = (GroundTruthInterval("L21_V003", 26220, 26220),)
    training = (
        GroundTruthQuery("train-1", interval, text="train one"),
        GroundTruthQuery("train-2", interval, text="train two"),
    )
    validation = (
        GroundTruthQuery("validation-1", interval, text=MULTI_VEHICLE_QUERY),
    )

    report = tune_semantic_view_policy(
        cached,
        training,
        validation,
        roles=roles,
        candidate_depth=4,
        consensus_alpha_values=(0.5, 1.0),
        qrels_provenance={"locked_test_sha256": "a" * 64},
        fingerprints={
            "model": "model:v1",
            "ranking_cache": "cache:v1",
            "query_adapter": "adapter:v1",
        },
    )

    assert sum(report.selected.weights.values()) == pytest.approx(1.0)
    assert report.standalone_train_mrr == {
        "global": pytest.approx(1.0),
        "scene_action": pytest.approx(0.5),
        "instance_attributes": pytest.approx(1 / 3),
        "relation_relaxed": pytest.approx(0.25),
    }
    # Six policy hypotheses x (1 + 2 + 3 + 4) complete k/depth pairs.
    assert report.candidates_evaluated == 60
    assert 1 <= report.selected.rrf_k <= report.selected.candidate_depth <= 4
    payload = report.to_dict()
    assert payload["activation_gate"]["passed"] is False
    assert payload["qrels_provenance"]["locked_test_sha256"] == "a" * 64
