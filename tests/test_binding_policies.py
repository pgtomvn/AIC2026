"""CPU/offline policy counterexamples; these do not estimate research effectiveness."""

import itertools
import sys
from dataclasses import replace

import pytest

from src.evaluation.selective_metrics import CandidateJudgment, selective_metrics
from src.retrieval.binding_policies import (
    POLICIES,
    AssignmentRow,
    AssignmentTable,
    PolicyDecision,
    build_assignment_table,
    compare_binding_policies,
)
from src.retrieval.conservative_evidence import (
    EvidenceObservation,
    FrameScope,
    QueryPredicate,
)
from src.retrieval.grounded_assignment import BoundObservation, BoundRequirement

SCOPE = FrameScope("batch", "video", 30, 1000, "view-v1")
EVENT = "event-1"
NAMESPACE = "detections-v1"
P = QueryPredicate("a-red", "query-v1", "object is red", "visual")
Q = QueryPredicate("b-left", "query-v1", "object is on the left", "visual")
REQUIREMENTS = (BoundRequirement(P, EVENT, ("x",)), BoundRequirement(Q, EVENT, ("x",)))


def observation(
    predicate: QueryPredicate = P,
    identity: str = "object-1",
    verdict: str = "support",
    *,
    bindings: tuple[tuple[str, str], ...] | None = None,
) -> BoundObservation:
    evidence = EvidenceObservation(
        predicate.predicate_id, predicate.query_revision, SCOPE, predicate.modality,
        "available", verdict,
        {"support": "localized_measurement", "contradiction": "localized_counterexample",
         "unknown": "unverified"}[verdict],
        "test-verifier", "verifier-v1", "offline_test_assessment",
    )
    return BoundObservation(
        evidence, EVENT, NAMESPACE,
        bindings if bindings is not None else (("x", f"{NAMESPACE}:{identity}"),),
    )


def table(
    observations: tuple[BoundObservation, ...] = (),
    requirements: tuple[BoundRequirement, ...] = REQUIREMENTS,
    **kwargs,
) -> AssignmentTable:
    return build_assignment_table(
        requirements, observations, scope=SCOPE, event_id=EVENT,
        identity_revision=NAMESPACE, origin="synthetic_oracle", **kwargs,
    )


def decisions(value: AssignmentTable, **kwargs) -> dict[str, PolicyDecision]:
    return {d.policy: d for d in compare_binding_policies(.8, value, **kwargs)}


def policy_decision(action: str, *, observed_witness: bool = False) -> PolicyDecision:
    delta = {"promote": .25, "demote": -.5, "unchanged": 0.0}[action]
    return PolicyDecision(
        "evibind_selective", .8 * (1 + delta), action, "test_decision", delta,
        observed_witness,
    )


def oracle_domains(*identities: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (("x", tuple(f"{NAMESPACE}:{identity}" for identity in identities)),)


def test_all_five_policies_use_the_same_joint_witness_and_base_score():
    value = table((observation(), observation(Q)))
    actual = compare_binding_policies(.8, value, full_query_covered=True)
    assert tuple(d.policy for d in actual) == POLICIES
    assert actual[0].score == .8 and actual[0].action == "unchanged"
    assert all(d.score == pytest.approx(1.0) and d.action == "promote" for d in actual[1:])
    assert all(d.observed_joint_witness for d in actual)
    assert all(d.witness == (("x", f"{NAMESPACE}:object-1"),) for d in actual)


def test_cross_object_misbinding_separates_independent_from_joint_policies():
    value = table((observation(P, "red-car"), observation(Q, "left-car")))
    actual = decisions(value, full_query_covered=True)
    assert value.state == "ready" and len(value.rows) == 2
    assert {row.states for row in value.rows} == {("support", "unknown"), ("unknown", "support")}
    assert actual["independent"].score == pytest.approx(1.0)
    assert actual["soft_joint"].score == pytest.approx(.9)
    for name in ("hard_joint", "evibind_selective"):
        assert actual[name].score == .8
        assert actual[name].action == "unchanged"
    assert not any(d.observed_joint_witness or d.witness for d in actual.values())


def test_missing_predicate_is_unknown_while_local_counterexample_is_contradiction():
    missing = table(
        (observation(),), domain_is_physically_exhaustive=True,
        exhaustive_domains=oracle_domains("object-1"),
    )
    contradicted = table(
        (observation(), observation(Q, verdict="contradiction")),
        domain_is_physically_exhaustive=True,
        exhaustive_domains=oracle_domains("object-1"),
    )
    assert missing.rows[0].states == ("support", "unknown")
    assert missing.rows[0].reasons[1] == "missing_or_unobservable"
    assert contradicted.rows[0].states == ("support", "contradiction")
    assert decisions(missing, full_query_covered=True)["evibind_selective"].score == .8
    assert decisions(contradicted, full_query_covered=True)["evibind_selective"].score == pytest.approx(.6)


@pytest.mark.parametrize("exhaustive", [False, True])
def test_one_contradicted_assignment_cannot_reject_another_possible_assignment(exhaustive):
    value = table(
        (observation(P, "one", "contradiction"), observation(Q, "one"),
         observation(P, "two")),
        domain_is_physically_exhaustive=exhaustive,
        exhaustive_domains=oracle_domains("one", "two") if exhaustive else (),
    )
    assert {r.states for r in value.rows} == {("contradiction", "support"), ("support", "unknown")}
    assert decisions(value, full_query_covered=True)["evibind_selective"].action == "unchanged"


def test_counterexample_for_one_assignment_does_not_cancel_another_joint_witness():
    value = table(
        (observation(P, "one", "contradiction"), observation(Q, "one"),
         observation(P, "two"), observation(Q, "two")),
        domain_is_physically_exhaustive=True,
        exhaustive_domains=oracle_domains("one", "two"),
    )
    actual = decisions(value, full_query_covered=True)
    assert actual["evibind_selective"].action == "promote"
    assert actual["evibind_selective"].witness == (("x", f"{NAMESPACE}:two"),)


@pytest.mark.parametrize("exhaustive,expected", [(False, "unchanged"), (True, "demote")])
def test_selective_demotion_requires_oracle_exhaustiveness_and_every_assignment_contradicted(
    exhaustive, expected,
):
    value = table(
        (observation(P, "one", "contradiction"), observation(Q, "one"),
         observation(P, "two", "contradiction"), observation(Q, "two", "contradiction")),
        domain_is_physically_exhaustive=exhaustive,
        exhaustive_domains=oracle_domains("one", "two") if exhaustive else (),
    )
    actual = decisions(value, full_query_covered=True)
    assert actual["hard_joint"].action == "unchanged"
    assert actual["evibind_selective"].action == expected
    if exhaustive:
        assert actual["evibind_selective"].relative_delta == -.25


@pytest.mark.parametrize("verdict", ["support", "contradiction"])
def test_incomplete_parser_coverage_is_a_common_gate_for_every_policy(verdict):
    value = table(
        (observation(verdict=verdict), observation(Q, verdict=verdict)),
        domain_is_physically_exhaustive=True,
        exhaustive_domains=oracle_domains("object-1"),
    )
    for actual in (decisions(value), decisions(value, full_query_covered=False)):
        assert all(d.score == .8 and d.action == "unchanged" for d in actual.values())
        assert all(d.reason == "full_query_not_covered" for name, d in actual.items() if name != "stage1")


@pytest.mark.parametrize("evidence_changes", [
    {"predicate_id": "unrequested"}, {"query_revision": "another-query"},
    {"modality": "speech"},
    {"scope": replace(SCOPE, batch_id="another-batch")},
    {"scope": replace(SCOPE, video_id="another-video")},
    {"scope": replace(SCOPE, source_frame_idx=31)},
    {"scope": replace(SCOPE, timestamp_ms=1001)},
    {"scope": replace(SCOPE, view_revision="another-view")},
])
def test_observations_cannot_cross_query_modality_or_any_scope_coordinate(evidence_changes):
    invalid = observation(Q)
    invalid = replace(invalid, evidence=replace(invalid.evidence, **evidence_changes))
    value = table((observation(), invalid))
    assert value.ignored_observations == 1
    assert value.rows[0].states == ("support", "unknown")
    assert decisions(value, full_query_covered=True)["hard_joint"].action == "unchanged"


@pytest.mark.parametrize("changes", [
    {"event_id": "another-event"}, {"identity_revision": "another-detector"},
    {"bindings": (("x", "another-detector:object-1"),)},
    {"bindings": (("wrong-variable", f"{NAMESPACE}:object-1"),)},
    {"bindings": ()},
])
def test_wrong_event_identity_namespace_or_variables_are_ignored(changes):
    value = table((observation(), replace(observation(Q), **changes)))
    assert value.ignored_observations == 1
    assert value.rows[0].states == ("support", "unknown")


@pytest.mark.parametrize("availability", ["available", "degraded", "unavailable"])
def test_unknown_and_unavailable_observations_never_become_contradictions(availability):
    unknown = observation(Q, verdict="unknown")
    unknown = replace(unknown, evidence=replace(unknown.evidence, availability=availability))
    value = table(
        (observation(), unknown), domain_is_physically_exhaustive=True,
        exhaustive_domains=oracle_domains("object-1"),
    )
    assert value.rows[0].states == ("support", "unknown")
    assert decisions(value, full_query_covered=True)["evibind_selective"].action == "unchanged"


def test_duplicate_sources_have_one_vote_and_conflicts_abstain_locally():
    first = observation()
    repeat = replace(first, evidence=replace(first.evidence, source_id="another-source"))
    original = table((first, observation(Q)))
    assert table((first, first, repeat, observation(Q))) == original
    conflict = table((first, repeat, observation(verdict="contradiction"), observation(Q)))
    assert conflict.rows[0].states == ("unknown", "support")
    assert conflict.rows[0].reasons == ("conflicting_evidence", "localized_assessment")
    assert decisions(conflict, full_query_covered=True)["evibind_selective"].action == "unchanged"


def test_requirement_observation_and_binding_orders_do_not_change_tables_or_witnesses():
    records = (observation(P, "b"), observation(Q, "b"), observation(P, "a"), observation(Q, "a"))
    expected = table(records)
    expected_decisions = decisions(expected, full_query_covered=True)
    for permutation in itertools.permutations(records):
        actual = table(permutation, tuple(reversed(REQUIREMENTS)))
        assert actual == expected
        assert decisions(actual, full_query_covered=True) == expected_decisions
    assert expected_decisions["hard_joint"].witness == (("x", f"{NAMESPACE}:a"),)
    relation = (BoundRequirement(P, EVENT, ("y", "x")),)
    bindings = (("y", f"{NAMESPACE}:b"), ("x", f"{NAMESPACE}:a"))
    assert table((observation(bindings=bindings),), relation) == table(
        (observation(bindings=tuple(reversed(bindings))),), relation,
    )


def test_distinct_roles_filter_self_binding_without_merging_object_identities():
    reqs = (REQUIREMENTS[0], BoundRequirement(Q, EVENT, ("y",)))
    records = (
        observation(), observation(Q, bindings=(("y", f"{NAMESPACE}:object-1"),)),
        observation(Q, bindings=(("y", f"{NAMESPACE}:object-2"),)),
    )
    value = table(records, reqs, distinct=(("x", "y"),))
    assert len(value.rows) == 1 and value.rows[0].states == ("support", "support")
    assert dict(value.rows[0].bindings) == {"x": f"{NAMESPACE}:object-1", "y": f"{NAMESPACE}:object-2"}
    impossible = table(records[:2], reqs, distinct=(("x", "y"),))
    assert impossible.state == "no_distinct_assignment"
    assert all(d.action == "unchanged" for d in decisions(impossible, full_query_covered=True).values())


def test_assignment_budget_refusal_is_neutral_and_does_not_materialize_partial_rows():
    value = table((observation(P, "a"), observation(Q, "a"), observation(P, "b")), max_assignments=1)
    assert value.state == "assignment_budget_exhausted" and value.rows == ()
    actual = decisions(value, full_query_covered=True)
    assert all(d.score == .8 and not d.observed_joint_witness for d in actual.values())


@pytest.mark.parametrize("requirements,state", [
    ((), "empty_or_mixed_event_requirements"),
    ((REQUIREMENTS[0], replace(REQUIREMENTS[1], event_id="another-event")), "empty_or_mixed_event_requirements"),
    (REQUIREMENTS, "missing_object_domain"),
])
def test_empty_missing_and_mixed_event_tables_do_not_promote(requirements, state):
    value = table(requirements=requirements)
    assert value.state == state and value.rows == ()
    assert all(d.action == "unchanged" for d in decisions(value, full_query_covered=True).values())


@pytest.mark.parametrize("requirements,message", [
    ((REQUIREMENTS[0], REQUIREMENTS[0]), "duplicate predicate"),
    ((REQUIREMENTS[0], replace(REQUIREMENTS[1], predicate=replace(Q, query_revision="other"))), "mixed query"),
    ((replace(REQUIREMENTS[0], predicate=replace(P, essential=False)),), "essential"),
])
def test_incoherent_requirement_sets_are_rejected(requirements, message):
    with pytest.raises(ValueError, match=message):
        table(requirements=requirements)


@pytest.mark.parametrize("kwargs", [
    {"max_assignments": 0}, {"max_assignments": True}, {"max_assignments": 1.5},
    {"domain_is_physically_exhaustive": 1},
    {"distinct": (("x", "x"),)}, {"distinct": (("x", "undeclared"),)},
])
def test_invalid_table_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        table(**kwargs)


def test_automatic_evidence_cannot_assert_physical_exhaustiveness():
    with pytest.raises(ValueError, match="closed physical domain"):
        build_assignment_table(
            REQUIREMENTS, (), scope=SCOPE, event_id=EVENT, identity_revision=NAMESPACE,
            origin="automatic_verified", domain_is_physically_exhaustive=True,
        )


def test_wholly_unobserved_oracle_object_prevents_selective_rejection():
    value = table(
        (observation(P, "seen", "contradiction"), observation(Q, "seen")),
        domain_is_physically_exhaustive=True,
        exhaustive_domains=oracle_domains("seen", "unobserved-true-match"),
    )
    assert {row.states for row in value.rows} == {
        ("contradiction", "support"), ("unknown", "unknown"),
    }
    # A reference judgment may know the unobserved object satisfies the query;
    # that knowledge must not be inferred from the system's observed table.
    result = decisions(value, full_query_covered=True)["evibind_selective"]
    assert result.action == "unchanged" and result.score == .8
    metric = selective_metrics([("candidate", result)], {
        "candidate": CandidateJudgment(True, False),
    })
    assert metric["demotions"] == 0


def test_exhaustiveness_requires_explicit_complete_domains_and_cannot_drop_unseen_objects():
    with pytest.raises(ValueError, match="explicit oracle domains"):
        table((observation(),), domain_is_physically_exhaustive=True)
    with pytest.raises(ValueError, match="outside"):
        table(
            (observation(),), domain_is_physically_exhaustive=True,
            exhaustive_domains=oracle_domains("some-other-object"),
        )
    incomplete = table((observation(), observation(Q)))
    with pytest.raises(ValueError, match="unobserved assignments"):
        replace(
            incomplete, domain_is_physically_exhaustive=True,
            exhaustive_domains=oracle_domains("object-1", "entirely-unobserved"),
        )


@pytest.mark.parametrize("changes", [
    {"predicate_ids": ()}, {"predicate_ids": (P.predicate_id, P.predicate_id)},
    {"rows": ()}, {"state": "unknown-state"}, {"ignored_observations": -1},
    {"origin": "undeclared"}, {"scope": "not-a-frame"},
    {"query_revision": ""}, {"identity_revision": "wrong-namespace"},
    {"origin": "automatic_verified", "domain_is_physically_exhaustive": True},
])
def test_directly_constructed_tables_cannot_bypass_integrity_checks(changes):
    valid = table((observation(), observation(Q)))
    with pytest.raises(ValueError):
        replace(valid, **changes)


@pytest.mark.parametrize("changes", [
    {"bindings": ()}, {"bindings": (("x", "id:one"), ("x", "id:two"))},
    {"states": ()}, {"states": ("made-up-verdict",)}, {"reasons": ()},
])
def test_directly_constructed_rows_cannot_bypass_integrity_checks(changes):
    valid = AssignmentRow((("x", f"{NAMESPACE}:one"),), ("support",), ("assessment",))
    with pytest.raises(ValueError):
        replace(valid, **changes)


@pytest.mark.parametrize("base", [-1, float("nan"), float("inf")])
def test_invalid_base_score_is_rejected(base):
    with pytest.raises(ValueError):
        compare_binding_policies(base, table())


@pytest.mark.parametrize("kwargs", [
    {"full_query_covered": 1}, {"support_cap": -.1}, {"contradiction_cap": 1.1},
    {"support_cap": float("nan")}, {"contradiction_cap": float("inf")},
])
def test_invalid_coverage_and_score_caps_are_rejected(kwargs):
    with pytest.raises(ValueError):
        compare_binding_policies(.8, table(), **kwargs)


def test_adjusted_score_overflow_is_rejected_instead_of_returning_infinity():
    with pytest.raises(ValueError, match="finite"):
        compare_binding_policies(
            sys.float_info.max, table((observation(), observation(Q))), full_query_covered=True,
        )


def test_zero_base_score_has_no_score_change_even_with_a_supported_witness():
    actual = compare_binding_policies(0, table((observation(), observation(Q))), full_query_covered=True)
    assert all(d.score == 0 and d.action == "unchanged" for d in actual)
    assert all(d.observed_joint_witness for d in actual)
    assert selective_metrics([("candidate", actual[-1])], {})["promotion_coverage"] == 0


def test_empty_metrics_keep_zero_denominators_undefined():
    result = selective_metrics([], {})
    for key in ("candidates", "promotions", "demotions", "unchanged", "judged_actions"):
        assert result[key] == 0
    for key in (
        "score_change_coverage", "promotion_coverage", "selective_risk", "judged_only_risk",
        "promotion_precision", "judged_only_promotion_precision", "promotion_precision_bounds",
        "reference_unsupported_promotion_rate",
    ):
        assert result[key] is None


def test_unchanged_candidates_count_in_coverage_without_inventing_a_precision_denominator():
    result = selective_metrics([("one", policy_decision("unchanged"))], {})
    assert result["score_change_coverage"] == 0
    assert result["promotion_coverage"] == 0
    assert result["promotion_precision"] is None and result["selective_risk"] is None


def test_fully_judged_metrics_count_both_wrong_promotions_and_wrong_demotions():
    records = [(cid, policy_decision(action)) for cid, action in (
        ("promote-correct", "promote"), ("promote-wrong", "promote"),
        ("demote-correct", "demote"), ("demote-wrong", "demote"), ("untouched", "unchanged"),
    )]
    judgments = {
        "promote-correct": CandidateJudgment(True, False),
        "promote-wrong": CandidateJudgment(False, True),
        "demote-correct": CandidateJudgment(False, None),
        "demote-wrong": CandidateJudgment(True, None),
        "not-a-candidate": CandidateJudgment(False, False),
    }
    result = selective_metrics(records, judgments)
    assert result["candidates"] == 5 and result["judged_actions"] == 4
    assert result["wrong_judged_actions"] == 2 and result["selective_risk"] == .5
    assert result["score_change_coverage"] == .8 and result["promotion_coverage"] == .4
    assert result["promotion_precision"] == .5
    assert result["reference_unsupported_promotion_rate"] == .5


def test_partial_labels_leave_full_metrics_undefined_and_report_judged_metrics_and_bounds():
    records = [(cid, policy_decision(action)) for cid, action in (
        ("yes", "promote"), ("no", "promote"), ("missing", "promote"), ("demote", "demote"),
    )]
    judgments = {
        "yes": CandidateJudgment(True, None), "no": CandidateJudgment(False, False),
        "demote": CandidateJudgment(True, True),
    }
    result = selective_metrics(records, judgments)
    assert result["selective_risk"] is None and result["promotion_precision"] is None
    assert result["judged_only_risk"] == pytest.approx(2 / 3)
    assert result["judged_only_promotion_precision"] == .5
    assert result["promotion_precision_bounds"] == pytest.approx([1 / 3, 2 / 3])
    assert result["unjudged_promotions"] == 1
    assert result["reference_unjudged_promotions"] == 2
    assert result["reference_unsupported_promotion_rate"] is None


def test_reference_metrics_never_use_the_policy_own_witness_flag():
    judgments = {"one": CandidateJudgment(True, False)}
    false_flag = selective_metrics([("one", policy_decision("promote", observed_witness=False))], judgments)
    true_flag = selective_metrics([("one", policy_decision("promote", observed_witness=True))], judgments)
    assert false_flag == true_flag
    assert true_flag["reference_unsupported_promotion_rate"] == 1
    missing_labels = selective_metrics([("one", policy_decision("promote", observed_witness=True))], {})
    assert missing_labels["reference_unsupported_promotion_rate"] is None
    assert missing_labels["promotion_precision"] is None


def test_duplicate_candidates_and_mixed_policy_evaluations_are_rejected():
    decision = policy_decision("promote")
    with pytest.raises(ValueError, match="unique"):
        selective_metrics([("same", decision), ("same", decision)], {})
    with pytest.raises(ValueError, match="one policy"):
        selective_metrics([("one", decision), ("two", replace(decision, policy="hard_joint"))], {})


@pytest.mark.parametrize("changes", [{"action": "promtoe"}, {"policy": "undeclared-policy"}])
def test_invalid_decision_actions_and_policies_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(policy_decision("promote"), **changes)


@pytest.mark.parametrize("field", ["score", "relative_delta"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_decision_scores_and_deltas_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(policy_decision("promote"), **{field: value})


@pytest.mark.parametrize("candidate_id", ["", " \t", None, 7])
def test_metrics_reject_malformed_candidate_ids(candidate_id):
    with pytest.raises(ValueError):
        selective_metrics([(candidate_id, policy_decision("promote"))], {})


@pytest.mark.parametrize("value", [0, 1, .8, "true"])
def test_judgments_must_be_explicit_booleans_or_unknown(value):
    with pytest.raises(ValueError, match="bool or unknown"):
        CandidateJudgment(value, None)
    with pytest.raises(ValueError, match="bool or unknown"):
        CandidateJudgment(None, value)
