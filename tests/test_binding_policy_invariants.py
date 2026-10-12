"""Finite contract checks and counterexamples; no natural-video efficacy claims."""

import itertools
from dataclasses import replace

import pytest

from src.retrieval.binding_policies import (
    AssignmentRow,
    AssignmentTable,
    build_assignment_table,
    compare_binding_policies,
)
from src.retrieval.conservative_evidence import EvidenceObservation, FrameScope, QueryPredicate
from src.retrieval.grounded_assignment import BoundObservation, BoundRequirement

SCOPE = FrameScope("audit", "video", 1, 40, "view-v1")
EVENT = "event-v1"
NAMESPACE = "objects-v1"
PREDICATES = tuple(QueryPredicate(name, "query-v1", name, "visual") for name in ("p", "q"))
REQUIREMENTS = tuple(BoundRequirement(p, EVENT, ("x",)) for p in PREDICATES)
STATES = ("support", "contradiction", "unknown")
GATED = ("hard_joint", "evibind_selective")


def _record(predicate=0, identity="a", state="support", *, variable="x"):
    p = PREDICATES[predicate]
    evidence = EvidenceObservation(
        p.predicate_id, p.query_revision, SCOPE, p.modality, "available", state,
        {"support": "localized_measurement", "contradiction": "localized_counterexample",
         "unknown": "unverified"}[state],
        "audit-source", "audit-v1", "finite_contract_fixture",
    )
    return BoundObservation(evidence, EVENT, NAMESPACE, ((variable, f"{NAMESPACE}:{identity}"),))


def _build(records, requirements=REQUIREMENTS, **kwargs):
    return build_assignment_table(
        requirements, records, scope=SCOPE, event_id=EVENT,
        identity_revision=NAMESPACE, origin="synthetic_oracle", **kwargs,
    )


def _decisions(table, base=1.0, **kwargs):
    return {d.policy: d for d in compare_binding_policies(
        base, table, full_query_covered=True, **kwargs,
    )}


def _fixed_table(states, *, exhaustive=False):
    """Keep both assignments present independently of their resolved states."""
    identities = tuple(f"{NAMESPACE}:{identity}" for identity in ("a", "b"))
    rows = tuple(AssignmentRow((("x", identity),), tuple(row), ("audit", "audit"))
                 for identity, row in zip(identities, (states[:2], states[2:])))
    return AssignmentTable(
        ("p", "q"), rows, 0, "ready", SCOPE, EVENT, "query-v1", NAMESPACE,
        "synthetic_oracle", exhaustive, (("x", identities),) if exhaustive else (),
    )


def test_builder_permutations_and_admissible_duplicates_preserve_conflicts_and_decisions():
    records = (_record(), _record(state="contradiction"), _record(1),
               _record(identity="b"), _record(1, "b"))
    expected = _build(records)
    for requirements in itertools.permutations(REQUIREMENTS):
        for permutation in itertools.permutations(records):
            repeated = permutation + tuple(
                replace(r, evidence=replace(r.evidence, source_id="duplicate-source"))
                for r in permutation
            )
            actual = _build(repeated, requirements)
            assert actual == expected
            assert _decisions(actual) == _decisions(expected)
    assert expected.rows[0].states == ("unknown", "support")
    assert _decisions(expected)["hard_joint"].witness == (("x", f"{NAMESPACE}:b"),)


def test_repeating_rejected_records_changes_only_diagnostic_count():
    rejected = replace(_record(), event_id="wrong-event")
    once = _build((_record(), _record(1), rejected))
    twice = _build((_record(), _record(1), rejected, rejected))
    assert replace(twice, ignored_observations=once.ignored_observations) == once
    assert twice.ignored_observations == 2
    assert _decisions(twice) == _decisions(once)


@pytest.mark.parametrize("exhaustive", [False, True])
def test_all_two_by_two_state_tables_are_invariant_to_assignment_row_order(exhaustive):
    for states in itertools.product(STATES, repeat=4):
        table = _fixed_table(states, exhaustive=exhaustive)
        assert _decisions(replace(table, rows=tuple(reversed(table.rows)))) == _decisions(table)


@pytest.mark.parametrize("exhaustive", [False, True])
@pytest.mark.parametrize("base", [0.0, 1.0])
def test_fixed_table_erasure_cannot_create_gated_actions_or_increase_their_magnitude(
    exhaustive, base,
):
    for states in itertools.product(STATES, repeat=4):
        table = _fixed_table(states, exhaustive=exhaustive)
        before = _decisions(table, base)
        for keep in itertools.product((False, True), repeat=4):
            erased = tuple(state if retain else "unknown" for state, retain in zip(states, keep))
            after = _decisions(_fixed_table(erased, exhaustive=exhaustive), base)
            for name in GATED:
                if after[name].action != "unchanged":
                    assert after[name].action == before[name].action
                assert abs(after[name].relative_delta) <= abs(before[name].relative_delta)
                if after[name].observed_joint_witness:
                    assert before[name].observed_joint_witness


@pytest.mark.parametrize("name", ["independent", "soft_joint"])
def test_fixed_state_erasure_can_create_promotion_for_signed_fraction_comparators(name):
    before = _fixed_table(("support", "contradiction") * 2)
    after = _fixed_table(("support", "unknown") * 2)
    assert _decisions(before)[name].action == "demote"
    assert _decisions(after)[name].action == "promote"
    assert all(_decisions(after)[gated].action == "unchanged" for gated in GATED)


@pytest.mark.parametrize("exhaustive", [False, True])
def test_deleting_conflicting_record_can_create_gated_promotion(exhaustive):
    kwargs = {"domain_is_physically_exhaustive": exhaustive,
              "exhaustive_domains": (("x", (f"{NAMESPACE}:a",)),) if exhaustive else ()}
    conflicting = _build((_record(), _record(state="contradiction"), _record(1)), **kwargs)
    removed = _build((_record(), _record(1)), **kwargs)
    assert conflicting.rows[0].states == ("unknown", "support")
    assert removed.rows[0].states == ("support", "support")
    for name in GATED:
        assert _decisions(conflicting)[name].action == "unchanged"
        assert _decisions(removed)[name].action == "promote"


def test_deleting_a_domain_record_can_unblock_budget_and_create_promotion():
    records = (_record(), _record(1), _record(identity="b", state="unknown"))
    refused = _build(records, max_assignments=1)
    reduced = _build(records[:2], max_assignments=1)
    assert refused.state == "assignment_budget_exhausted"
    assert reduced.state == "ready"
    assert _decisions(refused)["hard_joint"].action == "unchanged"
    assert _decisions(reduced)["hard_joint"].action == "promote"


@pytest.mark.parametrize("exhaustive", [False, True])
def test_sound_oracle_gated_promotions_have_true_and_observed_witnesses(exhaustive):
    identities = ("a", "b")
    for truth in itertools.product((False, True), repeat=4):
        for observed in itertools.product((False, True), repeat=4):
            records = tuple(
                _record(i % 2, identities[i // 2], "support" if truth[i] else "contradiction")
                for i in range(4) if observed[i]
            )
            table = _build(
                records, domain_is_physically_exhaustive=exhaustive,
                exhaustive_domains=(("x", tuple(f"{NAMESPACE}:{x}" for x in identities)),)
                if exhaustive else (),
            )
            for name in GATED:
                decision = _decisions(table)[name]
                if decision.action == "promote":
                    index = identities.index(dict(decision.witness)["x"].split(":")[-1]) * 2
                    assert all(truth[index:index + 2])
                    assert all(observed[index:index + 2])
                if decision.action == "demote":
                    assert exhaustive
                    assert not any(all(truth[i:i + 2]) for i in (0, 2))


def test_false_support_breaks_conditional_soundness_despite_matching_scope_metadata():
    # Independent reference truth says q(a) is false; the scorer cannot inspect it.
    truth = (True, False)
    decisions = _decisions(_build((_record(), _record(1))))
    assert not all(truth)
    assert all(decisions[name].action == "promote" for name in GATED)


@pytest.mark.parametrize("changes", [
    {"batch_id": "other"}, {"video_id": "other"}, {"source_frame_idx": 2},
    {"timestamp_ms": 41}, {"view_revision": "other"},
])
def test_rejected_scope_records_cannot_add_domain_ids_or_exhaust_budget(changes):
    records = (_record(), _record(1))
    rejected = tuple(replace(r, evidence=replace(r.evidence, scope=replace(SCOPE, **changes)))
                     for r in (_record(identity="b"), _record(1, "b")))
    expected = _build(records, max_assignments=1)
    actual = _build(records + rejected, max_assignments=1)
    assert actual.ignored_observations == 2
    assert replace(actual, ignored_observations=0) == expected
    assert _decisions(actual) == _decisions(expected)


@pytest.mark.parametrize("base", [0.0, 1.0])
def test_selective_equals_hard_joint_structurally_on_every_open_state_table(base):
    for states in itertools.product(STATES, repeat=4):
        decisions = _decisions(_fixed_table(states), base)
        hard, selective = (decisions[name] for name in GATED)
        assert (hard.score, hard.action, hard.relative_delta, hard.witness) == (
            selective.score, selective.action, selective.relative_delta, selective.witness,
        )


def test_default_budget_accepts_cartesian_product_of_exactly_4096_assignments():
    requirements = (REQUIREMENTS[0], BoundRequirement(PREDICATES[1], EVENT, ("y",)))
    records = tuple(_record(predicate, str(i), variable=variable)
                    for predicate, variable in ((0, "x"), (1, "y")) for i in range(64))
    table = _build(records, requirements)
    assert table.state == "ready"
    assert len(table.rows) == 4096
    assert _decisions(table)["hard_joint"].action == "promote"


def test_default_budget_rejects_4097_without_partial_witness_but_explicit_budget_can_allow_it():
    requirements = REQUIREMENTS[:1]
    records = tuple(_record(identity=str(i)) for i in range(4097))
    refused = _build(records, requirements)
    assert refused.state == "assignment_budget_exhausted" and refused.rows == ()
    assert all(d.action == "unchanged" and d.witness == () for d in _decisions(refused).values())
    allowed = _build(records, requirements, max_assignments=4097)
    assert allowed.state == "ready" and len(allowed.rows) == 4097


def test_over_budget_is_rejected_before_cartesian_iterator_even_if_distinct_rows_would_fit(monkeypatch):
    requirements = (REQUIREMENTS[0], BoundRequirement(PREDICATES[1], EVENT, ("y",)))
    records = tuple(_record(predicate, str(i), variable=variable)
                    for predicate, variable in ((0, "x"), (1, "y")) for i in range(65))

    def forbidden_product(*args, **kwargs):
        raise AssertionError("over-budget inputs must not start Cartesian materialization")

    # 65**2 exceeds 4,200, although the 65**2 - 65 distinct assignments fit it.
    monkeypatch.setattr("src.retrieval.binding_policies.itertools.product", forbidden_product)
    refused = _build(records, requirements, distinct=(("x", "y"),), max_assignments=4200)
    assert refused.state == "assignment_budget_exhausted" and refused.rows == ()
    assert all(d.action == "unchanged" for d in _decisions(refused).values())


@pytest.mark.parametrize("budget", [65537, True, 4096.0])
def test_assignment_budget_type_and_upper_bound_are_enforced(budget):
    with pytest.raises(ValueError, match="max_assignments"):
        _build((), max_assignments=budget)


def test_hard_maximum_budget_is_valid_without_materializing_an_empty_domain():
    assert _build((), max_assignments=65536).state == "missing_object_domain"
