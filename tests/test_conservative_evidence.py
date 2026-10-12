"""Offline policy counterexamples, not claims of retrieval accuracy."""

from dataclasses import replace

import pytest

from src.query.query_decomposer import decompose_query
from src.retrieval.conservative_adapter import declared_predicates, observations_from_legacy
from src.retrieval.conservative_evidence import (
    ConservativePolicy,
    EvidenceObservation,
    FrameScope,
    QueryPredicate,
)

SCOPE = FrameScope("batch", "video", 30, 1000, "clean-content-sha256")
P = QueryPredicate("red-car", "query-v1", "car is red", "visual")
Q = QueryPredicate("left", "query-v1", "car is on the left", "visual")


def observation(predicate=P, verdict="support", **changes):
    value = EvidenceObservation(predicate.predicate_id, predicate.query_revision, SCOPE,
                                predicate.modality, "available", verdict,
                                {"support": "localized_measurement",
                                 "contradiction": "localized_counterexample",
                                 "unknown": "unverified"}[verdict],
                                "verifier", "source-v1", "test_assessment")
    return replace(value, **changes)


def score(observations=(), predicates=(P,), scope=SCOPE):
    return ConservativePolicy().score(.03, scope, predicates, observations)


def test_missing_is_neutral_and_cannot_complete_query():
    value, diagnostic = score()
    assert value == .03
    assert diagnostic["unknown_count"] == 1
    assert not diagnostic["whole_query_verified"]
    assert not diagnostic["essential_support_complete"]


def test_essential_unknown_blocks_partial_bonus_without_penalty():
    assert score([observation()], (P, Q))[0] == .03
    value, diagnostic = score([observation(), observation(Q)], (P, Q))
    assert value == pytest.approx(.0375)
    assert diagnostic["essential_support_complete"]


@pytest.mark.parametrize("change", [
    {"modality": "speech"}, {"query_revision": "other-query"},
    {"scope": replace(SCOPE, batch_id="other-batch")},
    {"scope": replace(SCOPE, video_id="other-video")},
    {"scope": replace(SCOPE, source_frame_idx=31)},
    {"scope": replace(SCOPE, timestamp_ms=1001)},
    {"scope": replace(SCOPE, view_revision="blurred")},
    {"predicate_id": "not-requested"},
])
def test_wrong_modality_query_time_frame_or_revision_is_ineligible(change):
    value, diagnostic = score([observation(**change)])
    assert value == .03
    assert diagnostic["support_count"] == 0
    assert len(diagnostic["ignored"]) == 1


def test_speech_cannot_verify_visible_text_but_can_address_speech_predicate():
    visible = replace(P, modality="visible_text")
    spoken = replace(P, modality="speech")
    assert score([observation(spoken)], (visible,))[0] == .03
    assert score([observation(spoken)], (spoken,))[0] > .03


@pytest.mark.parametrize("availability", ["unavailable", "degraded"])
def test_unavailable_cannot_be_conclusive(availability):
    with pytest.raises(ValueError, match="must be unknown"):
        observation(availability=availability)
    assert score([observation(verdict="unknown", availability=availability)])[0] == .03


@pytest.mark.parametrize("verdict", ["support", "contradiction"])
def test_detector_non_detection_cannot_support_or_contradict(verdict):
    with pytest.raises(ValueError, match="localized basis"):
        observation(verdict=verdict, basis="non_detection")


def test_duplicate_channels_do_not_increase_bonus():
    evidence = observation()
    assert score([evidence, evidence, replace(evidence, source_id="repeated-view")])[0] == score([evidence])[0]


def test_conflicting_evidence_abstains_and_reports_conflict():
    value, diagnostic = score([observation(), observation(verdict="contradiction")])
    assert value == .03
    assert diagnostic["predicates"][0]["reason"] == "conflicting_evidence"


def test_contradiction_blocks_reward_and_penalty_is_bounded():
    value, _ = score([observation(), observation(Q, "contradiction")], (P, Q))
    assert value == pytest.approx(.03 * .75)
    assert score([observation(verdict="contradiction")])[0] == pytest.approx(.015)


def test_withholding_all_evidence_resets_to_base_not_previous_reward():
    assert score([observation()])[0] > .03
    assert score()[0] == .03
    assert score([observation(verdict="contradiction")])[0] < score()[0]


def test_empty_and_optional_only_predicates_do_not_trigger_bonus():
    assert score(predicates=())[0] == .03
    assert score([observation()], (replace(P, essential=False),))[0] == .03


@pytest.mark.parametrize("value", [-.1, float("inf"), float("nan")])
def test_invalid_base_score_rejected(value):
    with pytest.raises(ValueError):
        ConservativePolicy().score(value, SCOPE, [P], [])


@pytest.mark.parametrize("value", [-1, 1.1, float("nan")])
def test_unbounded_policy_rejected(value):
    with pytest.raises(ValueError):
        ConservativePolicy(support_cap=value)


def test_duplicate_predicate_and_mixed_query_rejected():
    with pytest.raises(ValueError, match="unique"):
        score(predicates=(P, P))
    with pytest.raises(ValueError, match="one query"):
        score(predicates=(P, replace(Q, query_revision="other")))


def declared(query):
    return declared_predicates(decompose_query(query, (), english_text=query))


def convert(predicates, states, channel="hsv_color", complete=True):
    return observations_from_legacy(predicates, SCOPE,
                                    [{"channel": channel, "constraints": states}],
                                    source_revision="captured-inputs", complete=complete)


@pytest.mark.parametrize("state", ["present_fallback", "absent", "inconclusive_mismatch"])
def test_legacy_fallback_or_low_ratio_is_unknown(state):
    predicates = declared("a red car")
    c = predicates[0].constraint
    evidence = convert(predicates, [{"subject": c["subject"], "color": c["color"],
                                   "instance_id": c.get("node_id"), "state": state,
                                   "ratio": .02, "scope": "full_frame"}])
    assert all(o.verdict == "unknown" for o in evidence)


def test_legacy_localized_color_support_requires_exact_assignment():
    predicates = declared("a red car")
    c = predicates[0].constraint
    state = {"subject": c["subject"], "color": c["color"], "instance_id": c.get("node_id"),
             "state": "present", "scope": "roi", "ratio": .5,
             "exact_class": True, "matched_label": "car"}
    evidence = convert(predicates, [state])
    assert evidence[0].verdict == "support"
    assert convert(predicates, [dict(state, exact_class=False)])[0].verdict == "unknown"
    assert convert(predicates, [state, state])[0].reason == "ambiguous_trace_binding"
    assert convert(predicates, [state], complete=False)[0].verdict == "unknown"


def test_negative_detection_counterexample_but_missing_is_unknown():
    predicates = declared("a road without cars")
    negative = next(p for p in predicates if p.family == "negation")
    state = {"subject": negative.constraint["subject"], "state": "forbidden_present", "matched_labels": ["car"]}
    observations = convert(predicates, [state], "negative_object")
    assert next(o for o in observations if o.predicate_id == negative.predicate.predicate_id).verdict == "contradiction"
    observations = convert(predicates, [dict(state, state="inconclusive_unlocalized", matched_labels=[])], "negative_object")
    assert all(o.verdict == "unknown" for o in observations)


def test_legacy_multiple_constraints_require_unavailable_joint_binding_guard():
    predicates = declared("a red car on the left")
    assert any(p.family == "joint_binding" for p in predicates)
    observations = convert(predicates, [])
    joint = next(p for p in predicates if p.family == "joint_binding")
    record = next(o for o in observations if o.predicate_id == joint.predicate.predicate_id)
    assert record.verdict == "unknown" and record.availability == "unavailable"


def test_repeated_decomposition_has_stable_identity_and_changed_query_does_not():
    assert declared("a red car") == declared("a red car")
    assert declared("a red car")[0].predicate.query_revision != declared("a blue car")[0].predicate.query_revision
