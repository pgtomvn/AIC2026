"""Denominator-aware research metrics with independently supplied judgments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.retrieval.binding_policies import PolicyDecision


@dataclass(frozen=True)
class CandidateJudgment:
    """Independent labels for whole-query match and the shared evidence set.

    joint_witness certifies that the evidence available to scoring contains a
    valid joint witness under an external reference protocol. It is not merely
    existence of a matching object somewhere in an unseen video, nor automatic
    acceptance of a policy's own witness flag. Unknown remains None.
    """
    query_match: bool | None
    joint_witness: bool | None

    def __post_init__(self) -> None:
        if any(x is not None and type(x) is not bool for x in (self.query_match, self.joint_witness)):
            raise ValueError("judgments must be bool or unknown, never inferred from scores")


def selective_metrics(
    decisions: Sequence[tuple[str, PolicyDecision]], judgments: Mapping[str, CandidateJudgment],
) -> dict[str, object]:
    """Keep zero-denominator and incompletely judged quantities undefined.

    Joint-witness judgments are external reference judgments, never copied from
    a policy's own observed_joint_witness flag. Coverage is score-change coverage.
    This function estimates neither corpus recall nor statistical significance.
    """
    if any(not isinstance(cid, str) or not cid.strip() for cid, _ in decisions):
        raise ValueError("candidate IDs must be nonempty strings")
    if len({cid for cid, _ in decisions}) != len(decisions):
        raise ValueError("candidate IDs must be unique within one policy evaluation")
    if len({d.policy for _, d in decisions}) > 1:
        raise ValueError("evaluate one policy at a time")
    selected = [(cid, d) for cid, d in decisions if d.action != "unchanged"]
    promoted = [(cid, d) for cid, d in selected if d.action == "promote"]
    empty = CandidateJudgment(None, None)
    labeled_actions = [(d, judgments.get(cid, empty).query_match) for cid, d in selected
                       if judgments.get(cid, empty).query_match is not None]
    wrong = sum((d.action == "promote") != label for d, label in labeled_actions)
    positive = [judgments.get(cid, empty).query_match for cid, _ in promoted]
    correct = sum(x is True for x in positive)
    unlabeled = sum(x is None for x in positive)
    witness_labels = [judgments.get(cid, empty).joint_witness for cid, _ in promoted]
    witness_unknown = sum(x is None for x in witness_labels)
    unsupported = sum(x is False for x in witness_labels)
    total, chosen, promotions = len(decisions), len(selected), len(promoted)
    return {
        "candidates": total, "promotions": promotions,
        "demotions": chosen - promotions, "unchanged": total - chosen,
        "score_change_coverage": chosen / total if total else None,
        "promotion_coverage": promotions / total if total else None,
        "judged_actions": len(labeled_actions), "wrong_judged_actions": wrong,
        "selective_risk": wrong / chosen if chosen and len(labeled_actions) == chosen else None,
        "judged_only_risk": wrong / len(labeled_actions) if labeled_actions else None,
        "promotion_precision": correct / promotions if promotions and not unlabeled else None,
        "judged_only_promotion_precision": correct / (promotions-unlabeled) if promotions > unlabeled else None,
        "promotion_precision_bounds": [correct/promotions, (correct+unlabeled)/promotions] if promotions else None,
        "unjudged_promotions": unlabeled,
        "reference_unsupported_promotion_rate": unsupported/promotions if promotions and not witness_unknown else None,
        "reference_unjudged_promotions": witness_unknown,
        "reference_source": "external judgments; not policy self-certification",
    }
