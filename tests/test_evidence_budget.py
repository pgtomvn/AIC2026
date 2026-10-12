"""Offline invariants of an experimental scheduler, not retrieval accuracy tests."""

import copy

import pytest

from src.query.query_decomposer import decompose_query
from src.query.translator import QueryProcessing, TranslationResult
from src.retrieval.evidence_budget import (
    BudgetedEvidenceReranker,
    ExistingEvidenceVerifier,
    VerificationOutcome,
    requested_evidence,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def candidates():
    return [
        {"video_id": "v", "source_frame_idx": i, "score": 1 / (i + 1),
         "evidence": [{"channel": "base"}], "channel_ranks": {"a": i + 1, "b": i + 1}}
        for i in range(3)
    ]


def test_overrun_retains_completed_action_without_starting_more():
    clock = Clock()
    source = candidates()
    saved = copy.deepcopy(source)

    def verify(item, kind):
        clock.now += .060
        item["score"] = 0.01
        item["evidence"].append({"channel": kind})
        return VerificationOutcome(item, True, "ready")

    output, diagnostic = BudgetedEvidenceReranker(budget_ms=50, clock=clock).run(
        source, ["geometry"], verify,
    )
    assert source == saved
    assert diagnostic["attempted_actions"] == 1
    assert diagnostic["overrun_ms"] == pytest.approx(10)
    assert output[-1]["source_frame_idx"] == 0
    assert len(output[-1]["evidence"]) == 2
    assert sorted(x["source_frame_idx"] for x in output) == [0, 1, 2]


@pytest.mark.parametrize("budget,kinds", [(0, ["full"]), (5, ["geometry"]), (200, [])])
def test_no_feasible_action_is_exact_noop(budget, kinds):
    def forbidden(*args):
        raise AssertionError("must not call")

    source = candidates()
    output, diagnostic = BudgetedEvidenceReranker(budget_ms=budget).run(source, kinds, forbidden)
    assert output == source
    assert diagnostic["attempted_actions"] == 0


def test_failed_mutating_action_does_not_erase_previously_completed_evidence():
    clock = Clock()

    def verify(item, kind):
        clock.now += .01
        item["score"] = .75 if kind == "geometry" else 99
        if kind == "full":
            raise RuntimeError("model failed")
        return VerificationOutcome(item, True, "ready")

    output, diagnostic = BudgetedEvidenceReranker(budget_ms=300, clock=clock).run(
        candidates()[:1], ["geometry", "full"], verify,
    )
    assert output[0]["score"] == .75
    assert diagnostic["completed_actions"] == 1
    assert diagnostic["trace"][-1]["reason"] == "verifier_error"


def test_full_action_replaces_original_score_not_compounded_geometry():
    observed = []

    def verify(item, kind):
        observed.append(item["score"])
        item["score"] *= .5 if kind == "geometry" else .8
        return VerificationOutcome(item, True, "ready")

    output, _ = BudgetedEvidenceReranker(budget_ms=300).run(
        candidates()[:1], ["geometry", "full"], verify,
    )
    assert observed == [1.0, 1.0]
    assert output[0]["score"] == .8


def test_identity_change_is_rejected_and_failed_action_consumes_budget():
    clock = Clock()

    def verify(item, kind):
        clock.now += .025
        item["video_id"] = "wrong"
        return VerificationOutcome(item, True, "ready")

    source = candidates()
    output, diagnostic = BudgetedEvidenceReranker(budget_ms=20, clock=clock).run(
        source, ["geometry"], verify,
    )
    assert output == source
    assert diagnostic["attempted_actions"] == 1
    assert diagnostic["completed_actions"] == 0
    assert diagnostic["trace"][0]["reason"] == "invalid_candidate_result"


def test_rank_discount_and_disagreement_produce_deterministic_priority():
    source = candidates()
    source[0]["channel_ranks"] = {"a": 1, "b": 1}
    source[1]["channel_ranks"] = {"a": 1, "b": 100}
    # Rank discount remains dominant here; disagreement is a bounded heuristic.
    clock = Clock()

    def verify(item, kind):
        clock.now += .011
        return VerificationOutcome(item, True, "ready")

    _, diagnostic = BudgetedEvidenceReranker(budget_ms=25, clock=clock).run(source, ["geometry"], verify)
    assert [x["candidate_rank"] for x in diagnostic["trace"]] == [1, 2]


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan")])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        BudgetedEvidenceReranker(budget_ms=budget)


def test_duplicate_identity_rejected_before_verification():
    source = candidates()[:1] * 2
    with pytest.raises(ValueError, match="unique"):
        BudgetedEvidenceReranker(budget_ms=100).run(source, [], lambda *_: None)


def processing(text):
    translation = TranslationResult(text, text, text, "en", text, "test", "unused", 0)
    return QueryProcessing(translation, {}, decompose_query(text, (), english_text=text))


@pytest.mark.parametrize("query,kinds", [
    ("a waterfall in a forest", ()),
    ("on the right", ()),  # Unbound position must not cause evidence I/O.
    ("a red car", ("full",)),
    ("a red car on the left", ("geometry", "full")),
])
def test_query_to_scheduler_evidence_contract(query, kinds):
    assert requested_evidence(processing(query)) == kinds


def test_geometry_adapter_preserves_color_query_for_later_full_verification():
    query = processing("a red car on the left")
    observed = []

    class Verifier:
        last_diagnostics = {"state": "ready"}

        def __call__(self, value, items):
            observed.append((len(value.decomposition.color_constraints),
                             len(value.decomposition.spatial_constraints)))
            return items

    adapter = ExistingEvidenceVerifier(Verifier(), query)
    adapter(candidates()[0], "geometry")
    adapter(candidates()[0], "full")
    assert observed == [(0, 1), (1, 1)]
    assert len(query.decomposition.color_constraints) == 1


def test_degraded_verifier_cannot_commit_a_changed_candidate():
    class Verifier:
        last_diagnostics = {"state": "degraded"}

        def __call__(self, query, items):
            items[0]["score"] = 999
            return items

    source = candidates()[0]
    outcome = ExistingEvidenceVerifier(Verifier(), processing("a red car"))(source, "full")
    assert not outcome.complete
    assert outcome.candidate == source
