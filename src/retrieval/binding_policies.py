"""Five controlled research policies over the same assignment/evidence table.

No models, labels or I/O. Whole-query coverage and physical domain completeness
are external obligations; the default grants neither. Joint means consistency
of assignments, not jointly trained representations.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.retrieval.conservative_evidence import FrameScope
from src.retrieval.grounded_assignment import BoundObservation, BoundRequirement

POLICIES = ("stage1", "independent", "hard_joint", "soft_joint", "evibind_selective")


@dataclass(frozen=True)
class AssignmentRow:
    bindings: tuple[tuple[str, str], ...]
    states: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.bindings, tuple) or not self.bindings
                or any(not isinstance(pair, tuple) or len(pair) != 2
                       or any(not isinstance(x, str) or not x.strip() for x in pair)
                       for pair in self.bindings)):
            raise ValueError("bindings must be immutable nonempty variable/identity pairs")
        if len(dict(self.bindings)) != len(self.bindings):
            raise ValueError("duplicate assignment variable")
        if (not isinstance(self.states, tuple) or not self.states
                or any(x not in {"support", "contradiction", "unknown"} for x in self.states)):
            raise ValueError("assignment states must be a nonempty tuple of supported verdicts")
        if (not isinstance(self.reasons, tuple) or len(self.reasons) != len(self.states)
                or any(not isinstance(x, str) or not x.strip() for x in self.reasons)):
            raise ValueError("every state needs a reason")


@dataclass(frozen=True)
class AssignmentTable:
    predicate_ids: tuple[str, ...]
    rows: tuple[AssignmentRow, ...]
    ignored_observations: int
    state: str
    scope: FrameScope
    event_id: str
    query_revision: str
    identity_revision: str
    origin: str
    # This is only enumeration over observed IDs, not a closed physical world.
    domain_is_physically_exhaustive: bool = False
    exhaustive_domains: tuple[tuple[str, tuple[str, ...]], ...] = ()
    distinct_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.predicate_ids, tuple)
                or any(not isinstance(p, str) or not p.strip() for p in self.predicate_ids)
                or len(set(self.predicate_ids)) != len(self.predicate_ids)):
            raise ValueError("predicate IDs must be unique immutable strings")
        if not isinstance(self.rows, tuple) or any(not isinstance(r, AssignmentRow) for r in self.rows):
            raise ValueError("assignment rows must be an immutable tuple")
        if any(len(r.states) != len(self.predicate_ids) for r in self.rows):
            raise ValueError("assignment state count must equal predicate count")
        if self.state not in {"ready", "empty_or_mixed_event_requirements", "missing_object_domain",
                              "assignment_budget_exhausted", "no_distinct_assignment"}:
            raise ValueError("unsupported assignment table state")
        if (self.state == "ready") != bool(self.rows):
            raise ValueError("only a ready table has assignment rows")
        if self.origin not in {"synthetic_oracle", "human_oracle", "automatic_verified"}:
            raise ValueError("an explicit evidence origin is required")
        if type(self.domain_is_physically_exhaustive) is not bool or (
                self.domain_is_physically_exhaustive and self.origin == "automatic_verified"):
            raise ValueError("only oracle branches may attest a complete physical domain")
        if not isinstance(self.scope, FrameScope):
            raise ValueError("scope must be FrameScope")
        if any(not isinstance(x, str) or not x.strip() for x in (self.event_id, self.identity_revision)):
            raise ValueError("event and identity revision must be nonempty")
        if self.rows and (not isinstance(self.query_revision, str) or not self.query_revision.strip()):
            raise ValueError("ready tables require a query revision")
        if type(self.ignored_observations) is not int or self.ignored_observations < 0:
            raise ValueError("ignored observation count must be nonnegative")
        keys = {tuple(sorted(v for v, _ in r.bindings)) for r in self.rows}
        if len(keys) > 1 or len({tuple(sorted(r.bindings)) for r in self.rows}) != len(self.rows):
            raise ValueError("assignments must have the same variables and be unique")
        if any(not oid.startswith(self.identity_revision + ":") for r in self.rows for _, oid in r.bindings):
            raise ValueError("assignment IDs must belong to the declared namespace")
        if self.domain_is_physically_exhaustive:
            declared = _validate_domains(self.exhaustive_domains, self.identity_revision)
            if self.rows:
                if set(declared) != set(dict(self.rows[0].bindings)):
                    raise ValueError("exhaustive domains must cover every assignment variable")
                if any(a == b or a not in declared or b not in declared for a, b in self.distinct_pairs):
                    raise ValueError("invalid distinct pair in declared domain")
                if math.prod(len(x) for x in declared.values()) > 65536:
                    raise ValueError("exhaustive ready table exceeds hard enumeration limit")
                expected = set()
                variables = sorted(declared)
                for values in itertools.product(*(declared[v] for v in variables)):
                    row = dict(zip(variables, values))
                    if not any(row[a] == row[b] for a, b in self.distinct_pairs):
                        expected.add(tuple(sorted(row.items())))
                if expected != {tuple(sorted(r.bindings)) for r in self.rows}:
                    raise ValueError("an exhaustive table must include unobserved assignments")


@dataclass(frozen=True)
class PolicyDecision:
    policy: str
    score: float
    action: str
    reason: str
    relative_delta: float
    observed_joint_witness: bool
    witness: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.policy not in POLICIES or self.action not in {"promote", "demote", "unchanged"}:
            raise ValueError("unsupported policy or score-change action")
        if (isinstance(self.score, bool) or not math.isfinite(self.score) or self.score < 0
                or isinstance(self.relative_delta, bool) or not math.isfinite(self.relative_delta)
                or not -1 <= self.relative_delta <= 1):
            raise ValueError("decision score and delta must be finite and within their bounds")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("decision reason is required")
        if type(self.observed_joint_witness) is not bool:
            raise ValueError("observed witness flag must be boolean")


def _validate_domains(
    domains: tuple[tuple[str, tuple[str, ...]], ...], namespace: str,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(domains, tuple) or not domains:
        raise ValueError("physical completeness requires explicit oracle domains")
    result = {}
    for pair in domains:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("domains must contain immutable variable/identities pairs")
        variable, values = pair
        if not isinstance(variable, str) or not variable.strip() or variable in result:
            raise ValueError("domain variables must be unique nonempty strings")
        if (not isinstance(values, tuple) or not values
                or any(not isinstance(v, str) or not v.startswith(namespace + ":") for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("oracle identities must be unique, nonempty and namespaced")
        result[variable] = values
    return result


def build_assignment_table(
    requirements: Sequence[BoundRequirement], observations: Sequence[BoundObservation], *,
    scope: FrameScope, event_id: str, identity_revision: str, origin: str,
    distinct: Sequence[tuple[str, str]] = (), max_assignments: int = 4096,
    domain_is_physically_exhaustive: bool = False,
    exhaustive_domains: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> AssignmentTable:
    """Materialize a bounded local table; reject cross-frame and identity leakage.

    Exhaustiveness can be attested only in an explicitly labelled oracle branch.
    Observed IDs alone, an IoU match or a missing detector output never establish
    that all possible physical assignments in a video have been considered.
    """
    if origin not in {"synthetic_oracle", "human_oracle", "automatic_verified"}:
        raise ValueError("an explicit evidence origin is required")
    if type(domain_is_physically_exhaustive) is not bool:
        raise ValueError("domain completeness must be boolean")
    if domain_is_physically_exhaustive and origin == "automatic_verified":
        raise ValueError("automatic records cannot assert a closed physical domain")
    if type(max_assignments) is not int or not 1 <= max_assignments <= 65536:
        raise ValueError("max_assignments must be positive and at most 65536")
    if not event_id.strip() or not identity_revision.strip():
        raise ValueError("event and identity namespace are required")
    reqs = sorted(requirements, key=lambda r: r.predicate.predicate_id)
    if len({r.predicate.predicate_id for r in reqs}) != len(reqs):
        raise ValueError("duplicate predicate")
    if len({r.predicate.query_revision for r in reqs}) > 1:
        raise ValueError("mixed query revisions")
    if any(not r.predicate.essential for r in reqs):
        raise ValueError("this prototype scores conjunctions of essential predicates only")
    variables = sorted({v for r in reqs for v in r.object_variables})
    if any(a == b or a not in variables or b not in variables for a, b in distinct):
        raise ValueError("distinct roles must be declared different variables")
    declared = _validate_domains(exhaustive_domains, identity_revision) if domain_is_physically_exhaustive else {}
    if declared and set(declared) != set(variables):
        raise ValueError("oracle domain must cover exactly the declared variables")
    if exhaustive_domains and not domain_is_physically_exhaustive:
        raise ValueError("exhaustive domains require an explicit oracle completeness attestation")
    revision = reqs[0].predicate.query_revision if reqs else ""
    ids = tuple(r.predicate.predicate_id for r in reqs)
    ignored = 0

    def result(rows: tuple[AssignmentRow, ...], state: str) -> AssignmentTable:
        return AssignmentTable(ids, rows, ignored, state, scope, event_id, revision,
                               identity_revision, origin, domain_is_physically_exhaustive,
                               exhaustive_domains, tuple(distinct))

    if not reqs or any(r.event_id != event_id for r in reqs):
        return result((), "empty_or_mixed_event_requirements")
    by_id = {r.predicate.predicate_id: r for r in reqs}
    facts: dict[tuple[str, tuple[tuple[str, str], ...]], set[str]] = {}
    domains: dict[str, set[str]] = {v: set(declared.get(v, ())) for v in variables}
    for row in observations:
        o = row.evidence
        req = by_id.get(o.predicate_id)
        if (req is None or o.query_revision != revision or o.scope != scope
                or row.event_id != event_id or row.identity_revision != identity_revision
                or o.modality != req.predicate.modality
                or set(dict(row.bindings)) != set(req.object_variables)
                or any(not oid.startswith(identity_revision + ":") for _, oid in row.bindings)):
            ignored += 1
            continue
        key = (o.predicate_id, tuple(sorted(row.bindings)))
        if declared and any(oid not in declared[var] for var, oid in row.bindings):
            raise ValueError("observed object lies outside the attested complete domain")
        state = o.verdict if o.availability == "available" else "unknown"
        facts.setdefault(key, set()).add(state)
        for var, oid in row.bindings:
            domains[var].add(oid)
    if any(not domain for domain in domains.values()):
        return result((), "missing_object_domain")
    # Bound the full Cartesian search before materialization. Refusal can be
    # conservative when distinct-role constraints would shrink the final set.
    size = math.prod(len(domains[v]) for v in variables)
    if size > max_assignments:
        return result((), "assignment_budget_exhausted")
    output = []
    for values in itertools.product(*(sorted(domains[v]) for v in variables)):
        assignment = dict(zip(variables, values))
        if any(assignment[a] == assignment[b] for a, b in distinct):
            continue
        states, reasons = [], []
        for req in reqs:
            key = (req.predicate.predicate_id,
                   tuple(sorted((v, assignment[v]) for v in req.object_variables)))
            evidence = facts.get(key, set()) - {"unknown"}
            state = next(iter(evidence)) if len(evidence) == 1 else "unknown"
            states.append(state)
            reasons.append("conflicting_evidence" if len(evidence) > 1 else
                           "localized_assessment" if evidence else "missing_or_unobservable")
        output.append(AssignmentRow(tuple(sorted(assignment.items())), tuple(states), tuple(reasons)))
    return result(tuple(output), "ready" if output else "no_distinct_assignment")


def compare_binding_policies(
    base_score: float, table: AssignmentTable, *, full_query_covered: bool = False,
    support_cap: float = .25, contradiction_cap: float = .50,
) -> tuple[PolicyDecision, ...]:
    """Score the same finite table under five explicit, uncalibrated policies.

    Selective v1 shares hard-joint's positive gate. It only adds demotion when
    every assignment in an oracle-attested complete domain has a contradiction.
    Hence it does NOT recover promotion coverage beyond hard joint. Measuring
    that equality is an intended diagnostic, not an improvement claim.
    """
    if isinstance(base_score, bool) or not math.isfinite(base_score) or base_score < 0:
        raise ValueError("base_score must be finite and nonnegative")
    if type(full_query_covered) is not bool:
        raise ValueError("full-query coverage must be boolean")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [support_cap, contradiction_cap]):
        raise ValueError("caps must be finite in [0, 1]")
    complete = [r for r in table.rows if r.states and all(s == "support" for s in r.states)]
    witness = min((r.bindings for r in complete), default=())

    def decision(name: str, delta: float, reason: str) -> PolicyDecision:
        score = base_score * (1 + delta)
        if not math.isfinite(score):
            raise ValueError("adjusted score must remain finite")
        action = "promote" if score > base_score else "demote" if score < base_score else "unchanged"
        return PolicyDecision(name, score, action, reason, delta, bool(complete), witness)

    base = decision("stage1", 0, "unchanged_baseline")
    if not full_query_covered or table.state != "ready" or not table.rows:
        reason = "full_query_not_covered" if not full_query_covered else table.state
        return (base, *(decision(p, 0, reason) for p in POLICIES[1:]))
    n = len(table.predicate_ids)

    def delta(states: Sequence[str]) -> float:
        return (support_cap * states.count("support") - contradiction_cap * states.count("contradiction")) / n

    # Deliberately ungated comparison: each predicate picks its best assignment.
    order = {"contradiction": -1, "unknown": 0, "support": 1}
    independent = tuple(max((r.states[i] for r in table.rows), key=order.get) for i in range(n))
    hard = support_cap if complete else 0
    soft = max(delta(r.states) for r in table.rows)
    selective, reason = hard, "joint_support" if complete else "insufficient_joint_evidence"
    if not complete and table.domain_is_physically_exhaustive and all(
            "contradiction" in r.states for r in table.rows):
        selective = -contradiction_cap * min(r.states.count("contradiction") / n for r in table.rows)
        reason = "all_assignments_contradicted_in_oracle_complete_domain"
    return (base, decision("independent", delta(independent), "predicatewise_best"),
            decision("hard_joint", hard, "all_support_gate"),
            decision("soft_joint", soft, "best_joint_signed_fraction"),
            decision("evibind_selective", selective, reason))
