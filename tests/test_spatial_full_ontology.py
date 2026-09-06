"""Offline tests for the full Advanced-KIS spatial ontology."""

from __future__ import annotations

import math

import pytest

from src.api import _safe_query_processing
from src.query.query_decomposer import RelativeSpatialConstraint, decompose_query
from src.query.spatial_types import SpatialRelationEnum
from src.retrieval.spatial_graph import (
    SpatialGraphEvaluator,
    SpatialOntologyConfig,
    evaluate_spatial_graph,
    relation_score,
)
from src.retrieval.spatial_rerank import ObjectDetection


def _d(label: str, box: tuple[float, float, float, float], confidence: float = 0.9) -> ObjectDetection:
    return ObjectDetection(label, confidence, box)


def _binary(relation: str, subject: str = "person", target: str = "car") -> RelativeSpatialConstraint:
    return RelativeSpatialConstraint(subject, relation, target, relation, "a", "b")


def _between() -> RelativeSpatialConstraint:
    return RelativeSpatialConstraint(
        "person", "between", "car", "between", "middle", "left", "bus", "right"
    )


@pytest.mark.parametrize(
    ("query", "relation"),
    (
        ("a person left of a car", "left_of"),
        ("một người bên phải xe ô tô", "right_of"),
        ("a bird above a car", "above"),
        ("con mèo nằm dưới cái bàn", "below"),
        ("a person adjacent to a car", "near"),
        ("một người cách xa xe máy", "far_from"),
        ("a person contained in a car", "inside"),
        ("a crowd surrounding a person", "surrounding"),
        ("a crowd around a person", "surrounding"),
        ("a crowd contains a person", "surrounding"),
        ("a box overlapping a sign", "overlapping"),
        ("a person between a car and a bus", "between"),
        ("a cyclist ahead of a car", "in_front_of"),
        ("a cyclist in the foreground of a car", "in_front_of"),
        ("một người đứng phía sau xe ô tô", "behind"),
        ("a person in the background of a car", "behind"),
    ),
)
def test_bilingual_ontology_parser(query: str, relation: str) -> None:
    result = decompose_query(query, (), english_text=query)
    assert result.relative_constraints
    assert result.relative_constraints[0].relation == relation
    assert result.relative_constraints[0].bound
    assert not result.unsupported_relations


def test_all_canonical_relations_are_unique() -> None:
    assert len(SpatialRelationEnum) == 12
    assert len({item.value for item in SpatialRelationEnum}) == 12


def test_negation_precedes_positive_spatial_and_color_parsing() -> None:
    result = decompose_query(
        "a yellow cyclist left of a blue car without a red bus",
        (),
        english_text="a yellow cyclist left of a blue car without a red bus",
    )
    assert [(item.subject, item.color) for item in result.negation_constraints] == [("bus", "red")]
    assert [(item.subject, item.relation, item.object_subject) for item in result.relative_constraints] == [
        ("cyclist", "left_of", "car")
    ]
    assert {(item.subject, item.color) for item in result.color_constraints} == {
        ("cyclist", "yellow"),
        ("car", "blue"),
    }


def test_between_requires_two_explicit_anchors() -> None:
    bound = decompose_query(
        "một người ở giữa xe màu đỏ và xe màu xanh dương", (), english_text="same"
    )
    constraint = bound.relative_constraints[0]
    assert constraint.arity == 3
    assert (constraint.subject, constraint.object_subject, constraint.second_object_subject) == (
        "người", "xe", "xe",
    )
    ambiguous = decompose_query("a person amidst a crowd", (), english_text="same")
    assert not ambiguous.relative_constraints
    assert ambiguous.ambiguous_relations[0].reason == "requires_two_anchors"
    assert not ambiguous.unsupported_relations


def test_directional_scores_are_continuous_and_inverse() -> None:
    left = _d("person", (0.1, 0.20, 0.3, 0.40))
    boundary = _d("car", (0.1, 0.23, 0.3, 0.43))
    half = _d("car", (0.1, 0.355, 0.3, 0.555))
    full = _d("car", (0.1, 0.48, 0.3, 0.68))
    assert relation_score(_binary("left_of"), left, boundary) == pytest.approx(0.0)
    assert relation_score(_binary("left_of"), left, half) == pytest.approx(0.5)
    assert relation_score(_binary("left_of"), left, full) == pytest.approx(1.0)
    assert relation_score(_binary("right_of"), full, left) == pytest.approx(1.0)


def test_near_and_far_scores_use_normalized_center_distance() -> None:
    first = _d("person", (0.1, 0.1, 0.2, 0.2))
    close = _d("car", (0.1, 0.2, 0.2, 0.3))
    far = _d("car", (0.8, 0.8, 0.9, 0.9))
    assert relation_score(_binary("near"), first, close) > 0.7
    assert relation_score(_binary("far_from"), first, close) == 0.0
    assert relation_score(_binary("near"), first, far) == 0.0
    assert relation_score(_binary("far_from"), first, far) == 1.0


def test_containment_surrounding_and_overlap_scores() -> None:
    inner = _d("person", (0.2, 0.2, 0.4, 0.4))
    outer = _d("car", (0.1, 0.1, 0.5, 0.5))
    partial = _d("sign", (0.3, 0.3, 0.6, 0.6))
    assert relation_score(_binary("inside"), inner, outer) == 1.0
    assert relation_score(_binary("surrounding"), outer, inner) == 1.0
    overlap = relation_score(_binary("overlapping", "person", "sign"), inner, partial)
    assert 0.0 < overlap < 1.0


def test_between_centered_off_axis_reversed_and_degenerate() -> None:
    middle = _d("person", (0.45, 0.45, 0.55, 0.55))
    left = _d("car", (0.45, 0.05, 0.55, 0.15))
    right = _d("bus", (0.45, 0.85, 0.55, 0.95))
    off_axis = _d("person", (0.80, 0.45, 0.90, 0.55))
    assert relation_score(_between(), middle, left, right) == pytest.approx(1.0)
    assert relation_score(_between(), middle, right, left) == pytest.approx(1.0)
    assert relation_score(_between(), off_axis, left, right) == 0.0
    assert relation_score(_between(), middle, left, left) == 0.0


def test_depth_is_soft_heuristic_with_half_reliability() -> None:
    foreground = _d("person", (0.35, 0.25, 0.95, 0.75))
    background = _d("car", (0.10, 0.40, 0.30, 0.60))
    constraint = _binary("in_front_of")
    assert constraint.verification_type == "heuristic_2d"
    assert relation_score(constraint, foreground, background) > 0.99
    assert relation_score(_binary("behind"), background, foreground) > 0.99
    config = SpatialOntologyConfig(perspective_weight=0.5)
    evaluator = SpatialGraphEvaluator(config)
    geometry = RelativeSpatialConstraint("person", "left_of", "bus", "", "p", "bus")
    perspective = RelativeSpatialConstraint("person", "in_front_of", "car", "", "p", "car")
    bus = _d("bus", (0.4, 0.75, 0.6, 0.95))
    match = evaluator.evaluate((geometry, perspective), (foreground, background, bus))
    assert match.state == "verified"
    assert 0.0 <= float(match.score) <= 1.0
    assert {item["verification_type"] for item in match.relation_states} == {
        "geometry", "heuristic_2d"
    }


def test_multi_entity_assignment_is_distinct_and_maximizes_geometry() -> None:
    detections = (
        _d("person", (0.1, 0.1, 0.3, 0.3), 0.95),
        _d("person", (0.1, 0.7, 0.3, 0.9), 0.85),
        _d("car", (0.4, 0.4, 0.6, 0.6), 0.9),
    )
    constraints = (
        RelativeSpatialConstraint("person", "left_of", "person", "", "left", "right"),
        RelativeSpatialConstraint("person", "above", "car", "", "left", "car"),
    )
    match = SpatialGraphEvaluator().evaluate(constraints, detections)
    assert match.state == "verified"
    assert len({assignment.detection.box for assignment in match.assignments}) == 3
    assert float(match.score) > 0.0


def test_evaluate_entrypoint_accepts_safe_mapping_and_invalid_is_neutral() -> None:
    detections = [
        {"label": "person", "confidence": 0.9, "box": [0.1, 0.1, 0.2, 0.2]},
        {"label": "car", "confidence": 0.9, "box": [0.1, 0.7, 0.2, 0.8]},
        {"label": "bad", "confidence": math.nan, "box": [0, 0, 1, 1]},
    ]
    assert evaluate_spatial_graph(detections, (_binary("left_of"),)) == 1.0
    assert evaluate_spatial_graph([], (_binary("left_of"),)) == 0.0


def test_api_projection_exposes_no_box_or_source_text() -> None:
    result = _safe_query_processing(
        {
            "relative_constraints": [
                {
                    "subject": "person",
                    "relation": "between",
                    "object_subject": "car",
                    "second_object_subject": "bus",
                    "arity": 3,
                    "verification_type": "geometry",
                    "bound": True,
                    "box": [0, 0, 1, 1],
                    "source_text": "secret",
                }
            ],
            "ambiguous_relations": [
                {"relation": "between", "reason": "requires_two_anchors"}
            ],
        }
    )
    assert result["relative_constraints"][0] == {
        "relation": "between",
        "bound": True,
        "arity": 3,
        "verification_type": "geometry",
        "subject": "person",
        "object": "car",
        "second_object": "bus",
    }
    assert result["ambiguous_relations"] == [
        {"relation": "between", "reason": "requires_two_anchors"}
    ]
    assert "box" not in repr(result)
    assert "secret" not in repr(result)


def test_config_rejects_non_finite_and_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        SpatialOntologyConfig(near_threshold=math.nan)
    with pytest.raises(ValueError):
        SpatialOntologyConfig(far_min=0.8, far_max=0.7)
    with pytest.raises(ValueError):
        SpatialOntologyConfig(beam_width=0)