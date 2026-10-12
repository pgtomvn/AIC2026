"""Research-only predicate evidence and bounded reranking; no I/O at import.

Verdicts are verifier assessments, not ground truth or probabilities. Scope is
one exact frame and view revision. This contract does not verify whole videos.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

Modality = Literal["visual", "visible_text", "speech"]
Verdict = Literal["support", "contradiction", "unknown"]
Availability = Literal["available", "degraded", "unavailable"]
Basis = Literal["localized_measurement", "localized_counterexample", "non_detection", "unverified"]


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


@dataclass(frozen=True)
class FrameScope:
    """Caller-supplied identity, timestamp and content/input revision."""

    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    view_revision: str

    def __post_init__(self) -> None:
        for name in ("batch_id", "video_id", "view_revision"):
            _text(getattr(self, name), name)
        for name in ("source_frame_idx", "timestamp_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class QueryPredicate:
    predicate_id: str
    query_revision: str
    description: str
    modality: Modality
    essential: bool = True

    def __post_init__(self) -> None:
        for name in ("predicate_id", "query_revision", "description"):
            _text(getattr(self, name), name)
        if self.modality not in {"visual", "visible_text", "speech"}:
            raise ValueError("unsupported predicate modality")
        if not isinstance(self.essential, bool):
            raise ValueError("essential must be boolean")


@dataclass(frozen=True)
class EvidenceObservation:
    predicate_id: str
    query_revision: str
    scope: FrameScope
    modality: Modality
    availability: Availability
    verdict: Verdict
    basis: Basis
    source_id: str
    source_revision: str
    reason: str
    cost_ms: float = 0.0
    # Named measurements, e.g. HSV fraction, are NOT confidence probabilities.
    measurements: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for name in ("predicate_id", "query_revision", "source_id", "source_revision", "reason"):
            _text(getattr(self, name), name)
        if not isinstance(self.scope, FrameScope):
            raise ValueError("scope must be FrameScope")
        if self.modality not in {"visual", "visible_text", "speech"}:
            raise ValueError("unsupported evidence modality")
        if self.availability not in {"available", "degraded", "unavailable"}:
            raise ValueError("unsupported availability")
        if self.verdict not in {"support", "contradiction", "unknown"}:
            raise ValueError("unsupported verdict")
        if self.basis not in {
            "localized_measurement", "localized_counterexample", "non_detection", "unverified"
        }:
            raise ValueError("unsupported basis")
        if self.availability != "available" and self.verdict != "unknown":
            raise ValueError("degraded/unavailable observations must be unknown")
        required = {"support": "localized_measurement", "contradiction": "localized_counterexample"}
        if self.verdict in required and self.basis != required[self.verdict]:
            raise ValueError("conclusive verdict requires a localized basis")
        if not math.isfinite(self.cost_ms) or self.cost_ms < 0:
            raise ValueError("cost_ms must be finite and nonnegative")
        if not isinstance(self.measurements, tuple):
            raise ValueError("measurements must be an immutable tuple")
        names = []
        for name, value in self.measurements:
            _text(name, "measurement name")
            if not math.isfinite(value):
                raise ValueError("measurement must be finite")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("measurement names must be unique")


@dataclass(frozen=True)
class ConservativePolicy:
    """Fixed, uncalibrated relative score bounds. Essential support gates bonus."""

    support_cap: float = 0.25
    contradiction_cap: float = 0.50

    def __post_init__(self) -> None:
        for value in (self.support_cap, self.contradiction_cap):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("score caps must be finite in [0, 1]")

    def score(
        self, base_score: float, scope: FrameScope, predicates: Sequence[QueryPredicate],
        observations: Sequence[EvidenceObservation],
    ) -> tuple[float, dict[str, Any]]:
        """Deduplicate per predicate, ignore scope leakage, abstain on conflicts.

        Unknown is neutral relative to the original Stage-1 score. Losing known
        support removes a bonus; losing a contradiction removes a penalty.
        Neither operation proves that a missing-evidence candidate is relevant.
        """

        if not math.isfinite(base_score) or base_score < 0:
            raise ValueError("base_score must be finite and nonnegative")
        by_id = {p.predicate_id: p for p in predicates}
        if len(by_id) != len(predicates):
            raise ValueError("predicate identities must be unique")
        if len({p.query_revision for p in predicates}) > 1:
            raise ValueError("predicates must share one query revision")
        accepted: dict[str, set[EvidenceObservation]] = {p: set() for p in by_id}
        ignored = []
        for observation in observations:
            predicate = by_id.get(observation.predicate_id)
            reason = None
            if predicate is None:
                reason = "predicate_not_requested"
            elif predicate.query_revision != observation.query_revision:
                reason = "query_revision_mismatch"
            elif observation.scope != scope:
                reason = "frame_time_or_view_mismatch"
            elif observation.modality != predicate.modality:
                reason = "modality_mismatch"
            if reason:
                ignored.append({"predicate_id": observation.predicate_id, "reason": reason})
            else:
                accepted[observation.predicate_id].add(observation)
        resolutions = []
        for predicate in predicates:
            evidence = accepted[predicate.predicate_id]
            verdicts = {o.verdict for o in evidence if o.verdict != "unknown"}
            conflict = len(verdicts) > 1
            verdict = next(iter(verdicts)) if len(verdicts) == 1 else "unknown"
            resolutions.append({
                "predicate_id": predicate.predicate_id, "description": predicate.description,
                "essential": predicate.essential, "verdict": verdict,
                "reason": "conflicting_evidence" if conflict else
                "localized_assessment" if verdict != "unknown" else "no_conclusive_evidence",
                "observation_count": len(evidence),
                "observation_reasons": sorted({o.reason for o in evidence}),
            })
        total = len(resolutions)
        supported = sum(p["verdict"] == "support" for p in resolutions)
        contradicted = sum(p["verdict"] == "contradiction" for p in resolutions)
        essential = [p for p in resolutions if p["essential"]]
        gate = bool(essential) and all(p["verdict"] == "support" for p in essential)
        # Contradiction prevents reward; each predicate has one vote regardless
        # of repeated detections or repeated evidence channels.
        reward = self.support_cap * supported / total if total and gate and not contradicted else 0.0
        penalty = self.contradiction_cap * contradicted / total if total else 0.0
        score = base_score * (1.0 + reward - penalty)
        if not math.isfinite(score):
            raise ValueError("adjusted score is not finite")
        return score, {
            "version": "research-conservative-evidence-v1", "policy": asdict(self),
            "base_score": base_score, "score": score, "relative_delta": reward - penalty,
            "support_count": supported, "contradiction_count": contradicted,
            "unknown_count": total - supported - contradicted,
            "essential_support_complete": gate, "predicates": resolutions, "ignored": ignored,
            "claim_scope": "declared_predicates_at_exact_frame_only",
            "whole_query_verified": False, "calibrated_probability": False,
            "production_activation": False,
        }
