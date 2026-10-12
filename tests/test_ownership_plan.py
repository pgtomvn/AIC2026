"""CPU-only role/scope checks; saved query wording is not an image oracle."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.query.ownership_plan import parse_ownership_query
from src.retrieval.conservative_evidence import EvidenceObservation, FrameScope
from src.retrieval.grounded_assignment import BoundObservation
from src.retrieval.ownership_binding import evaluate_ownership_witness, ownership_requirements

# Exact sentence excerpt from saved development query user-20260905-07.
MUSIC = "Có 1 người mặc áo trắng ngồi giữa 2 người mặc áo đen."
SCOPE = FrameScope("unit", "fictional-role-world", 0, 0, "unit-view")


def test_saved_music_excerpt_keeps_owner_count_relation_and_offsets():
    p = parse_ownership_query(MUSIC)
    assert [g.mentioned_count for g in p.groups] == [1, 2]
    assert [c.value for c in p.constraints if c.kind == "color"] == ["white", "black", "black"]
    assert sum(c.kind == "wears" for c in p.constraints) == 3
    assert len(p.distinct_pairs) == 3
    between = next(c for c in p.constraints if c.kind == "between")
    assert len(set(between.variables)) == 3
    assert between.value == "perspective_unspecified"
    assert any(c.kind == "pose" and c.value == "seated" for c in p.constraints)
    for x in (*p.groups, *p.constraints):
        assert MUSIC[slice(*x.source_span)] == x.source_text
    assert not p.grammar_complete


def test_hat_count_stays_on_hats_not_people_and_action_is_unhandled():
    text = "ba người đội nón có màu đỏ và cùng chạm mũi chân"
    p = parse_ownership_query(text)
    assert len(p.groups) == 1
    assert p.groups[0].mentioned_count == 3
    colors = [c for c in p.constraints if c.kind == "color"]
    assert len(colors) == 3
    assert all(c.variables[0] in p.groups[0].garment_variables for c in colors)
    assert any("chạm" in text[slice(*span)] for span in p.unhandled_spans)


@pytest.mark.parametrize("text,reason", [
    ("hơn 3 người đội nón đỏ", "modified_quantity"),
    ("ít nhất 2 người mặc áo trắng", "modified_quantity"),
    ("khoảng ba người đội nón đỏ", "modified_quantity"),
    ("0 người mặc áo trắng", "unsupported_role_count"),
    ("9999999999 người mặc áo trắng", "unsupported_role_count"),
    ("một người mặc nón đỏ", "unsupported_wearing_target"),
    ("một người đội áo đỏ", "unsupported_wearing_target"),
    ("không có một người mặc áo trắng", "negated_or_unresolved_attribute"),
    ("một người mặc áo đỏ đậm", "negated_or_unresolved_attribute"),
])
def test_unsupported_compositions_stay_blocked(text, reason):
    p = parse_ownership_query(text)
    assert len(p.groups) == 1
    assert p.groups[0].blocked_reason == reason
    assert not p.groups[0].person_variables
    assert not p.constraints
    assert not p.grammar_complete
    assert p.unhandled_spans


@pytest.mark.parametrize("text", [
    "một người mặc áo trắng. Sau đó có hai người mặc áo đen.",
    "một người mặc áo trắng ngồi cạnh hai người mặc áo đen.",
    "một người mặc áo trắng ngồi giữa ba người mặc áo đen.",
    "hai người mặc áo trắng ngồi giữa hai người mặc áo đen.",
])
def test_between_is_not_invented_across_unsupported_phrase(text):
    p = parse_ownership_query(text)
    assert not any(c.kind == "between" for c in p.constraints)


def test_unknown_color_does_not_assign_blue():
    p = parse_ownership_query("một người mặc áo xanh")
    assert not any(c.kind == "color" for c in p.constraints)
    assert not p.grammar_complete


def test_lexical_space_not_negation():
    p = parse_ownership_query("Trong không gian rộng có một người mặc áo trắng")
    assert not p.groups[0].blocked_reason


def test_pronoun_is_not_silently_linked_to_previous_owner():
    p = parse_ownership_query("một người mặc áo trắng. Người kia mặc áo đỏ.")
    assert len(p.groups) == 1
    assert sum(c.kind == "wears" for c in p.constraints) == 1
    assert not p.grammar_complete


def observations(plan):
    """Explicit artificial oracle assignments, never natural-video labels."""
    values = []
    for c in plan.constraints:
        evidence = EvidenceObservation(
            c.predicate_id, plan.query_revision, SCOPE, "visual", "available", "support",
            "localized_measurement", "synthetic-oracle", "v1", "unit world only",
        )
        values.append(BoundObservation(evidence, c.event_id, "ids",
                                       tuple((v, "ids:" + v) for v in c.variables)))
    return values


def test_complete_local_oracle_still_cannot_promote_full_query():
    p = parse_ownership_query(MUSIC)
    result = evaluate_ownership_witness(p, observations(p), scope=SCOPE, event_id="e0",
                                        identity_revision="ids")
    assert result.local_witness.verdict == "support"
    assert not result.promotion_eligible


def test_two_anchor_people_cannot_reuse_one_detection():
    p = parse_ownership_query(MUSIC)
    a, b = p.groups[1].person_variables
    rows = [replace(o, bindings=tuple((v, "ids:"+a if v == b else oid)
                                     for v, oid in o.bindings)) for o in observations(p)]
    result = evaluate_ownership_witness(p, rows, scope=SCOPE, event_id="e0", identity_revision="ids")
    assert result.local_witness.verdict == "unknown"


def test_swapped_garment_witness_cannot_borrow_other_person_color():
    p = parse_ownership_query(MUSIC)
    color = next(c for c in p.constraints if c.kind == "color" and c.value == "white")
    wrong_shirt = p.groups[1].garment_variables[0]
    rows = [replace(o, bindings=((color.variables[0], "ids:"+wrong_shirt),))
            if o.evidence.predicate_id == color.predicate_id else o for o in observations(p)]
    result = evaluate_ownership_witness(p, rows, scope=SCOPE, event_id="e0", identity_revision="ids")
    assert result.local_witness.verdict == "unknown"


def test_color_without_localized_wearing_relation_is_insufficient():
    p = parse_ownership_query(MUSIC)
    colors = {c.predicate_id for c in p.constraints if c.kind == "color"}
    rows = [o for o in observations(p) if o.evidence.predicate_id in colors]
    result = evaluate_ownership_witness(p, rows, scope=SCOPE, event_id="e0", identity_revision="ids")
    assert result.local_witness.verdict == "unknown"


@pytest.mark.parametrize("failure", ["scope", "identity", "query", "event", "budget"])
def test_stale_or_incomplete_contract_does_not_return_local_support(failure):
    p = parse_ownership_query(MUSIC)
    rows = observations(p)
    kwargs = dict(scope=SCOPE, event_id="e0", identity_revision="ids")
    if failure == "scope":
        kwargs["scope"] = replace(SCOPE, view_revision="different")
    elif failure == "identity":
        kwargs["identity_revision"] = "other"
    elif failure == "event":
        kwargs["event_id"] = "e1"
    elif failure == "query":
        rows = [replace(o, evidence=replace(o.evidence, query_revision="stale")) for o in rows]
    else:
        kwargs["max_comparisons"] = 1
    result = evaluate_ownership_witness(p, rows, **kwargs)
    assert result.local_witness.verdict == "unknown"


def test_required_event_does_not_merge_constraints_across_scenes():
    p = parse_ownership_query("một người mặc áo trắng. Sau đó hai người đội nón đỏ.")
    assert {c.event_id for c in p.constraints} == {"e0", "e1"}
    assert all(r.event_id == "e1" for r in ownership_requirements(p, "e1"))


def test_deterministic_replay():
    assert parse_ownership_query(MUSIC) == parse_ownership_query(MUSIC)
