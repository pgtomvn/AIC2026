"""Identity alias regressions; synthetic rows do not verify visual correctness."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest

from src.query.binding_plan import parse_binding_query
from src.retrieval.conservative_evidence import FrameScope
from src.retrieval.region_binding import evaluate_region_binding

SCOPE = FrameScope("synthetic", "identity-alias", 0, 0, "view-v1")
PLAN = parse_binding_query("red car on the left of the frame")


def region(identity: str, *, x: float = 0.2, color: str = "red", label: str = "Car") -> dict:
    return {"detection_id": identity, "label": label, "detector_score": 0.9,
            "box_yxyx": [0.1, x - 0.05, 0.9, x + 0.05], "hsv_fractions": {color: 0.8}}


def record(*regions: dict) -> dict:
    return {"scope": asdict(SCOPE), "identity_revision": "ids",
            "object_state": "ready", "regions": list(regions)}


@pytest.mark.parametrize("second", [
    region("ids:a", x=0.2, color="blue"),
    region("ids:a", x=0.8, color="blue"),
    region("ids:a", x=0.8, color="red", label="Person"),
    region("ids:a", x=0.8, color="red"),
])
def test_reused_identity_cannot_join_incompatible_or_repeated_region_rows(second: dict) -> None:
    source = record(region("ids:a", x=0.8), second)
    before = deepcopy(source)
    result = evaluate_region_binding(PLAN, source, expected_scope=SCOPE)
    assert source == before
    assert result["batch_problem"] == "duplicate_region_identity"
    assert result["observation_count"] == 0
    assert result["support_count"] == 0
    assert not result["joint_local_support"]
    assert not result["independent_local_support"]
    assert not result["promotion_eligible"]
    assert not result["independent_promotion_eligible"]
    assert all(event["joint"]["verdict"] == "unknown" for event in result["events"])


def test_alias_refusal_is_permutation_invariant_and_not_a_negative_label() -> None:
    source = record(region("ids:a", x=0.8), region("ids:a", color="blue"))
    first = evaluate_region_binding(PLAN, source, expected_scope=SCOPE)
    source["regions"].reverse()
    assert evaluate_region_binding(PLAN, source, expected_scope=SCOPE) == first
    assert all(event["joint"]["assignment"] == () for event in first["events"])


def test_exact_duplicates_cannot_silently_certify_a_shared_identity() -> None:
    same = region("ids:a")
    result = evaluate_region_binding(PLAN, record(same, deepcopy(same)), expected_scope=SCOPE)
    assert result["batch_problem"] == "duplicate_region_identity"
    assert not result["promotion_eligible"]


def test_unused_class_alias_still_invalidates_the_identity_revision() -> None:
    source = record(region("ids:a"), region("ids:b", label="Person"),
                    region("ids:b", label="Person", x=0.8))
    result = evaluate_region_binding(PLAN, source, expected_scope=SCOPE)
    assert result["batch_problem"] == "duplicate_region_identity"
    assert not result["promotion_eligible"]


def test_unique_ids_do_not_borrow_a_different_regions_attribute() -> None:
    result = evaluate_region_binding(PLAN, record(region("ids:a", x=0.8),
                                    region("ids:b", color="blue")), expected_scope=SCOPE)
    assert result["batch_problem"] == ""
    assert result["independent_local_support"]
    assert not result["joint_local_support"]
    assert result["events"][0]["joint"]["reason"] == "no_shared_assignment"


def test_unique_positive_control_keeps_support_without_geometric_id_merging() -> None:
    # Coincident geometry is not an identity proof. Distinct supplied IDs stay
    # separate, and a single complete region can still supply a local witness.
    result = evaluate_region_binding(PLAN, record(region("ids:a"), region("ids:b")),
                                     expected_scope=SCOPE)
    assert result["batch_problem"] == ""
    assert result["promotion_eligible"]
    assert result["events"][0]["joint"]["assignment"] == ((PLAN.objects[0].variable, "ids:a"),)


@pytest.mark.parametrize("change,reason", [
    ({"scope": {**asdict(SCOPE), "view_revision": "stale"}}, "frame_or_view_mismatch"),
    ({"object_state": "unavailable"}, "object_state_not_ready"),
    ({"identity_revision": ""}, "missing_identity_revision"),
])
def test_prior_scope_failure_keeps_its_reason_when_duplicate_ids_are_also_present(
    change: dict, reason: str,
) -> None:
    source = record(region("ids:a"), region("ids:a"))
    source.update(change)
    result = evaluate_region_binding(PLAN, source, expected_scope=SCOPE)
    assert result["batch_problem"] == reason
    assert not result["promotion_eligible"]
