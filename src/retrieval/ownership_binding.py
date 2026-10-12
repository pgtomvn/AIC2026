"""Compile text role proposals into the existing research witness-join contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.query.ownership_plan import OwnershipPlan
from src.retrieval.conservative_evidence import FrameScope, QueryPredicate
from src.retrieval.grounded_assignment import (
    BoundObservation,
    BoundRequirement,
    WitnessResult,
    join_event_witness,
)


@dataclass(frozen=True)
class OwnershipWitness:
    local_witness: WitnessResult
    promotion_eligible: bool = False
    reason: str = "full_query_and_visual_grounding_incomplete"


def ownership_requirements(plan: OwnershipPlan, event_id: str) -> tuple[BoundRequirement, ...]:
    """Compile only the requested text event, explicitly as local requirements."""
    return tuple(BoundRequirement(
        QueryPredicate(c.predicate_id, plan.query_revision, c.source_text, "visual"),
        c.event_id, c.variables,
    ) for c in plan.constraints if c.event_id == event_id)


def evaluate_ownership_witness(
    plan: OwnershipPlan, observations: Sequence[BoundObservation], *, scope: FrameScope,
    event_id: str, identity_revision: str, max_comparisons: int = 4096,
) -> OwnershipWitness:
    """Require distinct grounded people and all local roles; never grant a bonus.

    Geometry or a Clothing/Person box alone must not be supplied as a verified
    wears relation. This compiler produces no observations and infers no IDs.
    """
    requirements = ownership_requirements(plan, event_id)
    variables = {v for r in requirements for v in r.object_variables}
    distinct = tuple((a, b) for a, b in plan.distinct_pairs if a in variables and b in variables)
    result = join_event_witness(requirements, observations, scope=scope, event_id=event_id,
                               identity_revision=identity_revision, distinct=distinct,
                               max_comparisons=max_comparisons)
    return OwnershipWitness(result)
