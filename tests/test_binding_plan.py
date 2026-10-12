"""Synthetic contract checks; these are not a natural-video quality benchmark."""

from dataclasses import asdict

import pytest

from src.query.binding_plan import parse_binding_query
from src.retrieval.conservative_evidence import FrameScope
from src.retrieval.region_binding import evaluate_region_binding


def _scope() -> FrameScope:
    return FrameScope("synthetic", "fixture", 0, 0, "view-a")


def _region(oid: str, label: str, color: str, x: float) -> dict:
    return {"detection_id": "ids:"+oid, "label": label, "detector_score": 0.9,
            "box_yxyx": [0.1, x-0.05, 0.9, x+0.05], "hsv_fractions": {color: 0.8}}


def _record(*regions: dict) -> dict:
    return {"scope": asdict(_scope()), "identity_revision": "ids", "object_state": "ready", "regions": list(regions)}


@pytest.mark.parametrize("query,owner", [
    ("người đội nón có màu đỏ", "hat"),
    ("người mặc áo đen", "shirt"),
    ("a red car on the left of the frame", "car"),
    ("xe ô tô màu đỏ ở bên trái khung hình", "car"),
    ("nón đỏ. Sau đó áo trắng", "hat"),
])
def test_lexical_owner_and_offsets(query: str, owner: str) -> None:
    plan = parse_binding_query(query)
    attr = plan.attributes[0]
    obj = next(o for o in plan.objects if o.variable == attr.variable)
    assert obj.semantic_class == owner
    for a in plan.attributes:
        assert query[slice(*a.source_span)] == a.source_text
    for o in plan.objects:
        assert query[slice(*o.source_span)] == o.source_text


def test_text_events_do_not_merge_object_variables() -> None:
    p = parse_binding_query("nón đỏ. Sau đó nón trắng")
    assert len({a.variable for a in p.attributes}) == 2
    assert len({a.event_id for a in p.attributes}) == 2
    assert not p.grammar_complete


@pytest.mark.parametrize("q", ["nón hồng tím", "nón đỏ và vàng", "áo trắng ngà", "áo đen sọc"])
def test_qualified_color_is_blocked(q: str) -> None:
    p = parse_binding_query(q)
    assert p.attributes[0].blocked_reason == "qualified_or_multiple_color"
    assert not p.grammar_complete


@pytest.mark.parametrize("q", ["không có nón đỏ", "not a red hat", "ba người mặc áo đen", "áo đỏ đang bay"])
def test_incomplete_or_negative_not_silently_promoted(q: str) -> None:
    p = parse_binding_query(q)
    assert not p.grammar_complete
    assert p.unparsed_spans


def test_ambiguous_vietnamese_xanh_is_not_blue() -> None:
    assert not parse_binding_query("ghe xanh").attributes
    assert parse_binding_query("ghe xanh lá").attributes[0].value == "green"


@pytest.mark.parametrize("q", ["áo đỏ trong không gian rộng", "áo đỏ trong không khí lạnh"])
def test_lexical_khong_gian_and_khong_khi_are_not_negation(q: str) -> None:
    p = parse_binding_query(q)
    assert not p.attributes[0].blocked_reason
    assert not p.grammar_complete


def test_wrong_object_swap_keeps_independent_support_but_not_joint() -> None:
    p = parse_binding_query("red car on the left of the frame")
    r = _record(_region("a", "Car", "red", 0.8), _region("b", "Car", "blue", 0.2))
    result = evaluate_region_binding(p, r, expected_scope=_scope())
    assert result["independent_promotion_eligible"]
    assert not result["promotion_eligible"]
    assert result["events"][0]["binding_gap"]


def test_consistent_witness_positive_control_and_order_invariance() -> None:
    p = parse_binding_query("red car on the left of the frame")
    r = _record(_region("a", "Car", "red", 0.2), _region("b", "Car", "blue", 0.8))
    first = evaluate_region_binding(p, r, expected_scope=_scope())
    r["regions"].reverse()
    second = evaluate_region_binding(p, r, expected_scope=_scope())
    assert first == second
    assert first["promotion_eligible"]


def test_person_region_never_verifies_hat_or_shirt_color() -> None:
    for q in ("nón đỏ", "áo đỏ"):
        r = evaluate_region_binding(parse_binding_query(q), _record(_region("a", "Person", "red", 0.2)), expected_scope=_scope())
        assert r["support_count"] == 0
        assert not r["promotion_eligible"]


@pytest.mark.parametrize("field,value", [
    ("detector_score", float("nan")), ("detector_score", -1),
    ("detection_id", "other:a"), ("box_yxyx", [0, 0.9, 1, 0.1]),
    ("hsv_fractions", {"red": float("nan")}),
])
def test_invalid_region_measurement_cannot_support(field: str, value: object) -> None:
    region = _region("a", "Car", "red", 0.2)
    region[field] = value
    r = evaluate_region_binding(parse_binding_query("red car"), _record(region), expected_scope=_scope())
    assert not r["promotion_eligible"]


def test_revision_mismatch_abstains() -> None:
    record = _record(_region("a", "Car", "red", 0.2))
    record["scope"]["view_revision"] = "another-view"
    r = evaluate_region_binding(parse_binding_query("red car"), record, expected_scope=_scope())
    assert r["batch_problem"] == "frame_or_view_mismatch"
    assert not r["promotion_eligible"]


def test_missing_low_color_and_temporal_context_do_not_become_negatives() -> None:
    for q, record in [("red car", _record()),
                      ("red car", _record(_region("a", "Car", "blue", 0.2))),
                      ("red car. Then blue car", _record(_region("a", "Car", "red", 0.2)))]:
        r = evaluate_region_binding(parse_binding_query(q), record, expected_scope=_scope())
        assert not r["promotion_eligible"]
        assert all(e["joint"]["verdict"] != "contradiction" for e in r["events"])


def test_join_budget_never_leaks_partial_positive() -> None:
    p = parse_binding_query("red car on the left of the frame")
    r = evaluate_region_binding(p, _record(_region("a", "Car", "red", 0.2)), expected_scope=_scope(), max_comparisons=1)
    assert r["events"][0]["joint"]["reason"] == "join_budget_exhausted"
    assert not r["promotion_eligible"]


def test_no_color_noun_without_attributes_is_unknown() -> None:
    result = evaluate_region_binding(parse_binding_query("a car"), _record(_region("a", "Car", "red", 0.2)), expected_scope=_scope())
    assert result["predicate_count"] == 0
    assert not result["promotion_eligible"]
