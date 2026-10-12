"""Experimental bridge from nominal query proposals to measured region rows.

Same-event joins are diagnostics for complex queries. Only the small complete
single-clause grammar can license a heuristic score bonus. This is not activated
by the production runtime, and detector/HSV support is not human ground truth.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from src.query.binding_plan import BindingPlan
from src.retrieval.conservative_evidence import EvidenceObservation, FrameScope, QueryPredicate
from src.retrieval.grounded_assignment import BoundObservation, BoundRequirement, join_event_witness

VERSION = "evibind-region-bridge-v1"


def _finite_unit(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def evaluate_region_binding(
    plan: BindingPlan, record: dict[str, Any], *, expected_scope: FrameScope,
    min_detector_score: float = 0.5, min_color_fraction: float = 0.2,
    screen_margin: float = 0.05, max_comparisons: int = 4096,
) -> dict[str, Any]:
    """Compare independent predicate support and a common witness on identical rows.

    No detector run, image read, model call, label use or corpus access occurs.
    Color thresholds are development heuristics, not calibrated probabilities.
    Non-detection and low color fractions remain unknown, not contradictions.
    """
    if not all(_finite_unit(v) for v in (min_detector_score, min_color_fraction, screen_margin)):
        raise ValueError("thresholds must be finite in [0,1]")
    if min_color_fraction <= 0 or screen_margin >= 0.5:
        raise ValueError("color threshold must be positive and screen margin below 0.5")
    identity_revision = record.get("identity_revision", "")
    batch_problem = ("frame_or_view_mismatch" if record.get("scope") != asdict(expected_scope)
                     else "object_state_not_ready" if record.get("object_state") != "ready"
                     else "missing_identity_revision" if not isinstance(identity_revision, str) or not identity_revision
                     else "")
    # An ID must denote one region record in this input revision. Otherwise
    # distinct boxes can alias one object and manufacture a shared witness.
    # Even identical duplicate records fail closed: callers may deduplicate
    # upstream, but this bridge must not silently repair identity provenance.
    region_ids = [region.get("detection_id") for region in record.get("regions", [])
                  if isinstance(region, dict) and isinstance(region.get("detection_id"), str)
                  and region["detection_id"].strip()]
    if not batch_problem and len(region_ids) != len(set(region_ids)):
        batch_problem = "duplicate_region_identity"
    objects = {o.variable: o for o in plan.objects}
    observations: list[BoundObservation] = []
    requirements: list[BoundRequirement] = []
    diagnostics = []
    for attr in plan.attributes:
        obj = objects[attr.variable]
        predicate = QueryPredicate(attr.predicate_id, plan.query_revision, attr.source_text, "visual")
        requirements.append(BoundRequirement(predicate, attr.event_id, (attr.variable,)))
        reason = batch_problem or attr.blocked_reason
        accepted = 0
        if not reason:
            for region in record.get("regions", []):
                if region.get("label") not in obj.detector_labels:
                    continue
                score = region.get("detector_score")
                if not _finite_unit(score) or score < min_detector_score:
                    continue
                detection_id = region.get("detection_id")
                if not isinstance(detection_id, str) or not detection_id.startswith(identity_revision + ":"):
                    continue
                box = region.get("box_yxyx", [])
                if (len(box) != 4 or not all(_finite_unit(v) for v in box)
                        or not (box[0] < box[2] and box[1] < box[3])):
                    continue
                value = (region.get("hsv_fractions", {}).get(attr.value) if attr.kind == "color"
                         else (box[1] + box[3]) / 2)
                if not _finite_unit(value):
                    continue
                supported = (value >= min_color_fraction if attr.kind == "color"
                             else value < 0.5-screen_margin if attr.value == "left"
                             else value > 0.5+screen_margin)
                verdict = "support" if supported else "unknown"
                evidence = EvidenceObservation(
                    predicate.predicate_id, plan.query_revision, expected_scope, "visual", "available",
                    verdict, "localized_measurement" if supported else "unverified",
                    detection_id, identity_revision, "region_proxy_threshold" if supported else "inconclusive_region_proxy",
                    measurements=(("hsv_fraction" if attr.kind == "color" else "center_x", value),),
                )
                observations.append(BoundObservation(
                    evidence, attr.event_id, identity_revision, ((attr.variable, detection_id),)
                ))
                accepted += 1
            if not accepted:
                reason = "no_compatible_region_measurement"
        diagnostics.append({"predicate_id": attr.predicate_id, "variable": attr.variable,
                            "event_id": attr.event_id, "kind": attr.kind, "value": attr.value,
                            "source_text": attr.source_text, "rows": accepted, "reason": reason})
    supported_ids = {o.evidence.predicate_id for o in observations if o.evidence.verdict == "support"}
    per_event = []
    for eid in sorted({r.event_id for r in requirements}):
        group = [r for r in requirements if r.event_id == eid]
        result = join_event_witness(
            group, observations, scope=expected_scope, event_id=eid,
            identity_revision=identity_revision or "unavailable", max_comparisons=max_comparisons,
        )
        independent = all(r.predicate.predicate_id in supported_ids for r in group)
        per_event.append({"event_id": eid, "predicates": len(group),
                          "independent_support": independent, "joint": asdict(result),
                          "binding_gap": independent and result.reason == "no_shared_assignment",
                          "frame_event_grounded": plan.grammar_complete})
    independent_all = bool(requirements) and all(r.predicate.predicate_id in supported_ids for r in requirements)
    joint_all = bool(per_event) and all(e["joint"]["verdict"] == "support" for e in per_event)
    score_gate = plan.grammar_complete and len(per_event) == 1 and not batch_problem
    return {
        "version": VERSION, "query_revision": plan.query_revision, "scope": asdict(expected_scope),
        "thresholds": {"detector": min_detector_score, "color": min_color_fraction, "margin": screen_margin},
        "predicate_count": len(requirements), "support_count": len(supported_ids),
        "observation_count": len(observations), "predicates": diagnostics, "events": per_event,
        "independent_local_support": independent_all, "joint_local_support": joint_all,
        "grammar_complete": plan.grammar_complete, "promotion_eligible": score_gate and joint_all,
        "independent_promotion_eligible": score_gate and independent_all,
        "scope_reason": "single_clause_existential" if score_gate else "incomplete_query_or_event_grounding",
        "human_labels_used": False, "production_activation": False, "ground_truth_claim": False,
        "batch_problem": batch_problem,
    }
