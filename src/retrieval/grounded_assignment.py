"""Research-only bounded witness join for same-event object predicates.

Relational joins and object/event variables are established techniques, not a
novelty claim. Inputs must come from grounded, shared-identity observations;
this module cannot infer object identities from class names or human labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.retrieval.conservative_evidence import EvidenceObservation, FrameScope, QueryPredicate


@dataclass(frozen=True)
class BoundRequirement:
    predicate: QueryPredicate
    event_id: str
    object_variables: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.object_variables:
            raise ValueError("event and object variables are required")
        if (not isinstance(self.object_variables, tuple)
                or len(set(self.object_variables)) != len(self.object_variables)
                or any(not isinstance(v, str) or not v.strip() for v in self.object_variables)):
            raise ValueError("object variables must be unique nonempty strings")


@dataclass(frozen=True)
class BoundObservation:
    evidence: EvidenceObservation
    event_id: str
    identity_revision: str
    bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.identity_revision.strip():
            raise ValueError("event and detection identity revision required")
        if not isinstance(self.bindings, tuple):
            raise ValueError("bindings must be immutable")
        keys = []
        for variable, identity in self.bindings:
            if not variable.strip() or not identity.strip():
                raise ValueError("binding values must be nonempty")
            keys.append(variable)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate object variable")


@dataclass(frozen=True)
class WitnessResult:
    verdict: str
    reason: str
    assignment: tuple[tuple[str, str], ...] = ()
    comparisons: int = 0
    ignored_observations: int = 0


def join_event_witness(
    requirements: Sequence[BoundRequirement], observations: Sequence[BoundObservation],
    *, scope: FrameScope, event_id: str, identity_revision: str,
    distinct: Sequence[tuple[str, str]] = (), max_comparisons: int = 4096,
) -> WitnessResult:
    """Require a common object assignment; empty/truncated search stays unknown.

    A contradiction rejects only the same assignment. Failure to find a witness
    cannot establish absence in an incompletely observed image. Requirements
    must explicitly belong to the requested event; this function never silently
    drops predicates from other events to make a conjunction pass.
    """
    if type(max_comparisons) is not int or max_comparisons < 1:
        raise ValueError("max_comparisons must be a positive integer")
    if not event_id.strip() or not identity_revision.strip():
        raise ValueError("expected event and identity revision required")
    if not requirements:
        return WitnessResult("unknown", "no_requirements")
    by_id = {r.predicate.predicate_id: r for r in requirements}
    if len(by_id) != len(requirements):
        raise ValueError("duplicate requirement")
    if len({r.predicate.query_revision for r in requirements}) != 1:
        raise ValueError("mixed query revisions")
    if any(r.event_id != event_id for r in requirements):
        return WitnessResult("unknown", "mixed_or_wrong_event_requirements")
    variables = {v for r in requirements for v in r.object_variables}
    if any(a == b or a not in variables or b not in variables for a, b in distinct):
        raise ValueError("distinct pairs must name two declared variables")
    states: dict[str, dict[tuple[tuple[str, str], ...], set[str]]] = {pid: {} for pid in by_id}
    ignored = 0
    for record in observations:
        o = record.evidence
        r = by_id.get(o.predicate_id)
        if (r is None or o.query_revision != r.predicate.query_revision
                or o.scope != scope or o.modality != r.predicate.modality
                or record.event_id != event_id or record.identity_revision != identity_revision
                or set(dict(record.bindings)) != set(r.object_variables)
                or o.availability != "available"):
            ignored += 1
            continue
        key = tuple(sorted(record.bindings))
        states[o.predicate_id].setdefault(key, set()).add(o.verdict)
    candidates = []
    for pid in sorted(by_id):
        rows = sorted(k for k, v in states[pid].items()
                      if "support" in v and "contradiction" not in v)
        if not rows:
            return WitnessResult("unknown", "missing_or_conflicting_witness", ignored_observations=ignored)
        candidates.append(rows)
    partials: set[tuple[tuple[str, str], ...]] = {()}
    comparisons = 0
    for rows in candidates:
        merged = set()
        for partial in sorted(partials):
            left = dict(partial)
            for row in rows:
                if comparisons >= max_comparisons:
                    return WitnessResult("unknown", "join_budget_exhausted", comparisons=comparisons,
                                         ignored_observations=ignored)
                comparisons += 1
                right = dict(row)
                if any(v in left and left[v] != oid for v, oid in right.items()):
                    continue
                combined = left | right
                if any(a in combined and b in combined and combined[a] == combined[b]
                       for a, b in distinct):
                    continue
                merged.add(tuple(sorted(combined.items())))
        partials = merged
        if not partials:
            return WitnessResult("unknown", "no_shared_assignment", comparisons=comparisons,
                                 ignored_observations=ignored)
    return WitnessResult("support", "shared_assignment_witness", min(partials), comparisons, ignored)
