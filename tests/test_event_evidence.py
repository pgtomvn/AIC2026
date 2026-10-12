"""Synthetic contract regressions, not human labels or retrieval benchmarks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from src.evaluation.predicate_review import validate_predicate_review
from src.query.event_scope import scope_explicit_events
from src.retrieval.conservative_evidence import EvidenceObservation, FrameScope, QueryPredicate
from src.retrieval.grounded_assignment import (
    BoundObservation,
    BoundRequirement,
    join_event_witness,
)

SCOPE = FrameScope("synthetic", "test-video", 1, 40, "test-image-v1")


def req(pid: str, variables: tuple[str, ...] = ("car",), event: str = "e0"):
    return BoundRequirement(QueryPredicate(pid, "q1", pid, "visual"), event, variables)


def obs(pid: str, bindings: tuple[tuple[str, str], ...], verdict: str = "support"):
    return BoundObservation(EvidenceObservation(
        pid, "q1", SCOPE, "visual", "available", verdict,
        {"support": "localized_measurement", "contradiction": "localized_counterexample",
         "unknown": "unverified"}[verdict], "synthetic-verifier", "v1", "synthetic-test-only",
    ), "e0", "detections-v1", bindings)


def join(requirements, observations, **kwargs):
    return join_event_witness(requirements, observations, scope=SCOPE, event_id="e0",
                              identity_revision="detections-v1", **kwargs)


def test_mixed_object_positive_evidence_cannot_certify_conjunction():
    result = join([req("red"), req("left")],
                  [obs("red", (("car", "right-car"),)), obs("left", (("car", "left-car"),))])
    assert result.verdict == "unknown" and result.reason == "no_shared_assignment"


def test_shared_witness_supports_and_is_order_invariant():
    rs = [req("red"), req("left")]
    evidence = [obs("red", (("car", "c1"),)), obs("left", (("car", "c1"),))]
    a, b = join(rs, evidence), join(list(reversed(rs)), list(reversed(evidence)))
    assert a == b
    assert a.verdict == "support" and a.assignment == (("car", "c1"),)


def test_two_object_relation_shares_both_variables():
    rs = [req("red"), req("left-of", ("car", "person"))]
    a = [obs("red", (("car", "c1"),)),
         obs("left-of", (("car", "c1"), ("person", "p1")))]
    assert join(rs, a, distinct=[("car", "person")]).verdict == "support"
    bad = [a[0], obs("left-of", (("car", "c1"), ("person", "c1")))]
    assert join(rs, bad, distinct=[("car", "person")]).verdict == "unknown"


@pytest.mark.parametrize("change", [
    {"event_id": "e1"}, {"identity_revision": "detections-v2"}, {"bindings": ()},
])
def test_wrong_binding_provenance_is_ignored(change):
    o = replace(obs("red", (("car", "c1"),)), **change)
    result = join([req("red")], [o])
    assert result.verdict == "unknown" and result.ignored_observations == 1


@pytest.mark.parametrize("change", [
    {"query_revision": "q2"}, {"scope": replace(SCOPE, timestamp_ms=80)},
    {"scope": replace(SCOPE, view_revision="v2")}, {"modality": "speech"},
    {"scope": replace(SCOPE, video_id="other-video")},
])
def test_wrong_observation_provenance_is_ignored(change):
    o = obs("red", (("car", "c1"),))
    result = join([req("red")], [replace(o, evidence=replace(o.evidence, **change))])
    assert result.verdict == "unknown" and result.ignored_observations == 1


def test_conflict_only_blocks_its_own_assignment_and_duplicates_do_not_vote():
    a = obs("red", (("car", "c1"),))
    conflict = obs("red", a.bindings, "contradiction")
    assert join([req("red")], [a, a, conflict]).verdict == "unknown"
    second = obs("red", (("car", "c2"),))
    result = join([req("red")], [a, conflict, second])
    assert result.verdict == "support" and result.assignment == (("car", "c2"),)


def test_unknown_and_non_detection_never_become_negative_conjunction():
    assert join([req("red")], []).verdict == "unknown"
    assert join([req("red")], [obs("red", (("car", "c1"),), "unknown")]).verdict == "unknown"
    assert join([req("red")], [obs("red", (("car", "c1"),), "contradiction")]).verdict == "unknown"


def test_join_budget_does_not_return_partial_positive():
    rs = [req("red"), req("left")]
    a = [obs("red", (("car", "c1"),)), obs("left", (("car", "c1"),))]
    assert join(rs, a, max_comparisons=1).reason == "join_budget_exhausted"


def test_mixed_event_requirements_not_silently_dropped():
    rs = [req("red"), req("green", event="e1")]
    assert join(rs, [obs("red", (("car", "c1"),))]).reason == "mixed_or_wrong_event_requirements"


def test_duplicate_predicates_and_invalid_distinct_pairs_rejected():
    with pytest.raises(ValueError):
        join([req("red"), req("red")], [])
    with pytest.raises(ValueError):
        join([req("red")], [], distinct=[("car", "undeclared")])


@pytest.mark.parametrize("text,cue", [
    ("A red car. Sau đó có một người.", "Sau đó"),
    ("Xe chạy. Tiếp theo có người đi bộ.", "Tiếp theo"),
    ("Người chờ. Trước đó xe chạy.", "Trước đó"),
    ("A red car. Then a blue car.", "Then"),
    ("A red car. Previously a bus stopped.", "Previously"),
])
def test_explicit_event_partition_preserves_source(text, cue):
    plan = scope_explicit_events(text)
    assert len(plan.events) == 2 and plan.events[1].cue == cue
    assert ''.join(text[e.start:e.end] for e in plan.events) == text


def test_begin_end_and_spatial_after_are_distinct():
    text = "Đoạn clip bắt đầu với xe đỏ. Đoạn clip kết thúc với xe xanh."
    plan = scope_explicit_events(text)
    assert [e.relation_to_previous for e in plan.events] == ["clip_start", "clip_end"]
    assert len(scope_explicit_events("Người phía sau xe và trước nhà.").events) == 1
    assert len(scope_explicit_events("The car is behind a bus before a wall.").events) == 1


def test_repeated_and_nonexact_source_phrases_remain_auditable():
    plan = scope_explicit_events("Xe đỏ. Sau đó có áo đỏ.")
    assert plan.locate("đỏ").reason == "ambiguous_across_events"
    assert len(plan.locate("đỏ").occurrences) == 2
    assert plan.locate("do").reason == "source_phrase_not_exact"
    assert plan.locate(" ").reason == "missing_source_phrase"
    assert plan.locate("đỏ. Sau đó").reason == "crosses_event_boundary"


def test_same_event_repetition_keeps_all_occurrences():
    result = scope_explicit_events("Áo đỏ và mũ đỏ.").locate("đỏ")
    assert len(result.occurrences) == 2 and result.event_ids == ("e0",)
    assert result.reason == "explicit_partition_only"


def packet():
    c = {"case_id": "C01", "query_id": "q", "scope": {"view_revision": "x"},
         "image_sha256": "a"*64, "annotation_scope": "visible_evidence_in_this_exact_frame",
         "predicates": [{"predicate_id": "p"}]}
    m = {"schema": "fourbigpigs-predicate-review-v1", "source_probe_sha256": "b"*64,
         "reviewer_count": 1, "cases": [c]}
    r = deepcopy(m)
    r.update({"reviewer": "synthetic-reviewer", "export_status": "complete"})
    r['cases'][0].update({"image_checked": True, "predicates": {"p": {
        "verdict": "unknown", "parse_match": "no", "event_scope": "uncertain", "reason": ""}}})
    return m, r


def test_manual_export_immutable_and_unknown_not_negative():
    m, r = packet()
    before = deepcopy(r)
    result = validate_predicate_review(r, m)
    assert r == before
    assert result['counts']['verdict'] == {"unknown": 1}
    assert not result['quality_metric_computed']


@pytest.mark.parametrize("field,value", [
    ("image_sha256", "c"*64), ("query_id", "other"), ("image_checked", False),
    ("scope", {"view_revision": "wrong"}), ("predicates", {}),
])
def test_wrong_manual_packet_rejected(field, value):
    m, r = packet()
    r['cases'][0][field] = value
    with pytest.raises(ValueError):
        validate_predicate_review(r, m)


def test_incomplete_duplicate_and_reasonless_export_rejected():
    m, r = packet()
    r['export_status'] = 'draft'
    with pytest.raises(ValueError):
        validate_predicate_review(r, m)
    r['export_status'] = 'complete'
    r['cases'].append(deepcopy(r['cases'][0]))
    with pytest.raises(ValueError):
        validate_predicate_review(r, m)
    r['cases'].pop()
    r['cases'][0]['predicates']['p']['verdict'] = 'contradiction'
    with pytest.raises(ValueError):
        validate_predicate_review(r, m)
