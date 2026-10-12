"""Explicit conversion of legacy KIS traces for a research-only clean-view replay.

The caller must bind each trace to the frame/query/input snapshot that produced
it. A ready backend or an increased legacy score is never itself evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.query.query_decomposer import DecomposedQuery
from src.retrieval.conservative_evidence import (
    EvidenceObservation,
    FrameScope,
    QueryPredicate,
)


@dataclass(frozen=True)
class LegacyPredicate:
    predicate: QueryPredicate
    family: str
    constraint_json: str

    @property
    def constraint(self) -> dict[str, Any]:
        return json.loads(self.constraint_json)


def declared_predicates(decomposition: DecomposedQuery) -> tuple[LegacyPredicate, ...]:
    """Include recognized unsupported clauses as unknown essential predicates.

    Completeness of the natural-language parser is NOT assumed. A binding guard
    abstains on multi-constraint conjunctions: legacy traces do not expose a
    shared object-assignment proof for every combination of predicate families.
    """

    revision = hashlib.sha256(json.dumps(asdict(decomposition), sort_keys=True,
                                        ensure_ascii=False).encode()).hexdigest()
    output = []

    def add(family: str, value: dict[str, Any]) -> None:
        serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
        predicate_id = hashlib.sha256((family + serialized).encode()).hexdigest()[:24]
        if any(p.predicate.predicate_id == predicate_id for p in output):
            return
        output.append(LegacyPredicate(QueryPredicate(
            predicate_id, revision, f"{family}: {serialized}", "visual"
        ), family, serialized))

    for family in ("spatial", "relative", "color", "negation"):
        for constraint in getattr(decomposition, f"{family}_constraints"):
            value = asdict(constraint)
            value["bound"] = constraint.bound
            add(family, value)
    bound_count = sum(p.constraint["bound"] for p in output)
    for family, values in (
        ("sequence", decomposition.sequence_constraints),
        ("prospective", decomposition.prospective_relations),
        ("unsupported", decomposition.unsupported_relations),
        ("ambiguous", decomposition.ambiguous_relations),
    ):
        for constraint in values:
            add(family, asdict(constraint))
    for term in decomposition.lexical_must_match:
        # A lexical term alone does not specify visible text vs spoken content.
        # This unresolved requirement receives no converted observation.
        add("unresolved_lexical_modality", {"term": term})
    if bound_count > 1:
        add("joint_binding", {"reason": "legacy_trace_joint_assignment_not_certified"})
    return tuple(output)


def observations_from_legacy(
    predicates: Sequence[LegacyPredicate], scope: FrameScope,
    trace: Sequence[Mapping[str, Any]], *, source_revision: str,
    complete: bool, action_cost_ms: float = 0.0,
) -> tuple[EvidenceObservation, ...]:
    """Convert only explicit localized assessments; preserve unknown reasons.

    Costs belong to the whole action, so are not repeated on every predicate.
    Callers record action_cost_ms separately. Individual measurement costs are
    unavailable and are deliberately left zero here.
    """

    if action_cost_ms < 0:
        raise ValueError("action cost cannot be negative")
    records = []
    channels = {"spatial": "spatial_7x7", "relative": "spatial_relation",
                "color": "hsv_color", "negation": "negative_object"}
    for declared in predicates:
        predicate, constraint = declared.predicate, declared.constraint
        channel = channels.get(declared.family, "unsupported")
        matches = []
        for item in trace:
            if item.get("channel") != channel:
                continue
            for state in item.get("constraints", ()):
                if not isinstance(state, Mapping):
                    continue
                if state.get("subject") != constraint.get("subject"):
                    continue
                if declared.family == "color" and (
                    state.get("color") != constraint.get("color") or state.get("negated")
                    or state.get("instance_id") != constraint.get("node_id")
                ):
                    continue
                if declared.family == "spatial" and state.get("zone") != constraint.get("zone"):
                    continue
                if declared.family == "relative" and any(
                    (state.get(key) or "") != (constraint.get(key) or "")
                    for key in ("relation", "object_subject", "second_object_subject")
                ):
                    continue
                matches.append(state)
        verdict, basis = "unknown", "unverified"
        availability = "available" if complete else "degraded"
        reason = "legacy_action_incomplete" if not complete else "no_localized_evidence"
        measurements: list[tuple[str, float]] = []
        if complete and len(matches) == 1 and constraint.get("bound"):
            state = matches[0]
            reason = "legacy_state_" + str(state.get("state", "unresolved"))
            if declared.family == "color":
                if isinstance(state.get("ratio"), (int, float)):
                    measurements.append(("hsv_pixel_fraction", float(state["ratio"])))
                if (state.get("state") == "present" and state.get("scope") == "roi"
                        and state.get("exact_class") is True and state.get("matched_label")):
                    verdict, basis, reason = "support", "localized_measurement", "localized_hsv_positive"
                elif state.get("state") in {"absent", "inconclusive_mismatch"}:
                    basis, reason = "non_detection", "low_hsv_fraction_is_not_counterexample"
            elif declared.family in {"spatial", "relative"}:
                value = float(state.get("score", 0.0))
                measurements.append(("geometry_score", value))
                proxy = (constraint.get("perspective_proxy") or
                         state.get("verification_type") == "heuristic_2d")
                if value >= 0.5 and state.get("matched_labels") and not proxy:
                    verdict, basis, reason = "support", "localized_measurement", "geometry_positive_heuristic"
                else:
                    reason = "geometry_inconclusive_or_depth_proxy"
            elif declared.family == "negation" and constraint.get("color") is None:
                if state.get("state") == "forbidden_present" and state.get("matched_labels"):
                    verdict, basis, reason = ("contradiction", "localized_counterexample",
                                              "detected_forbidden_object_heuristic")
                else:
                    basis, reason = "non_detection", "detector_absence_is_not_proof"
        elif complete and len(matches) > 1:
            reason = "ambiguous_trace_binding"
        if channel == "unsupported":
            availability, reason = "unavailable", "predicate_backend_or_joint_binding_unavailable"
        records.append(EvidenceObservation(
            predicate.predicate_id, predicate.query_revision, scope, "visual", availability,
            verdict, basis, f"legacy:{channel}", source_revision, reason,
            measurements=tuple(measurements),
        ))
    return tuple(records)
