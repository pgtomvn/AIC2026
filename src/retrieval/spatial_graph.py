"""Finite-safe spatial graph verification for Advanced Textual KIS.

Boxes use organizer ``[ymin, xmin, ymax, xmax]`` coordinates.  The evaluator is
clean-room and deliberately keeps image/object evidence outside public results.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.query.query_decomposer import RelativeSpatialConstraint
from src.query.spatial_types import SpatialRelationEnum
from src.retrieval.spatial_rerank import ObjectDetection, labels_compatible

RELATION_MARGIN = 0.03
INSIDE_COVERAGE = 0.90
DEFAULT_NMS_IOU = 0.50
DEFAULT_MAX_DETECTIONS_PER_NODE = 6
DEFAULT_BEAM_WIDTH = 512

_RELATION_ALIASES = {
    "adjacent": SpatialRelationEnum.NEAR.value,
    "adjacent_to": SpatialRelationEnum.NEAR.value,
    "contained_in": SpatialRelationEnum.INSIDE.value,
    "contains": SpatialRelationEnum.SURROUNDING.value,
    "around": SpatialRelationEnum.SURROUNDING.value,
    "foreground": SpatialRelationEnum.IN_FRONT_OF.value,
    "background": SpatialRelationEnum.BEHIND.value,
    "ahead_of": SpatialRelationEnum.IN_FRONT_OF.value,
}


@dataclass(frozen=True, slots=True)
class SpatialOntologyConfig:
    """Validated scoring and bounded-assignment parameters."""

    directional_margin: float = 0.03
    directional_scale: float = 0.25
    near_threshold: float = 0.35
    far_min: float = 0.30
    far_max: float = 0.70
    inside_satisfied_coverage: float = 0.90
    between_projection_min: float = 0.10
    between_projection_max: float = 0.90
    between_perpendicular_threshold: float = 0.20
    depth_area_ratio: float = 1.20
    depth_sigmoid_scale: float = 10.0
    depth_vertical_tolerance: float = 0.05
    perspective_weight: float = 0.50
    nms_iou_threshold: float = DEFAULT_NMS_IOU
    max_detections_per_node: int = DEFAULT_MAX_DETECTIONS_PER_NODE
    beam_width: int = DEFAULT_BEAM_WIDTH

    def __post_init__(self) -> None:
        values = (
            self.directional_margin,
            self.directional_scale,
            self.near_threshold,
            self.far_min,
            self.far_max,
            self.inside_satisfied_coverage,
            self.between_projection_min,
            self.between_projection_max,
            self.between_perpendicular_threshold,
            self.depth_area_ratio,
            self.depth_sigmoid_scale,
            self.depth_vertical_tolerance,
            self.perspective_weight,
            self.nms_iou_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("spatial ontology thresholds must be finite")
        if not 0.0 <= self.directional_margin < 1.0:
            raise ValueError("directional_margin must be in [0, 1)")
        if self.directional_scale <= 0.0 or self.near_threshold <= 0.0:
            raise ValueError("directional_scale and near_threshold must be positive")
        if not 0.0 <= self.far_min < self.far_max <= math.sqrt(2.0):
            raise ValueError("far thresholds are invalid")
        if not 0.0 < self.inside_satisfied_coverage <= 1.0:
            raise ValueError("inside coverage must be in (0, 1]")
        if not 0.0 <= self.between_projection_min < self.between_projection_max <= 1.0:
            raise ValueError("between projection bounds are invalid")
        if self.between_perpendicular_threshold <= 0.0:
            raise ValueError("between perpendicular threshold must be positive")
        if self.depth_area_ratio <= 0.0 or self.depth_sigmoid_scale <= 0.0:
            raise ValueError("depth scoring parameters must be positive")
        if not 0.0 <= self.depth_vertical_tolerance <= 1.0:
            raise ValueError("depth_vertical_tolerance must be in [0, 1]")
        if not 0.0 < self.perspective_weight <= 1.0:
            raise ValueError("perspective_weight must be in (0, 1]")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1]")
        if self.max_detections_per_node < 1 or self.beam_width < 1:
            raise ValueError("graph search bounds must be positive")


@dataclass(frozen=True, slots=True)
class NodeAssignment:
    node_id: str
    detection: ObjectDetection


@dataclass(frozen=True, slots=True)
class SpatialGraphMatch:
    state: str
    score: float | None = None
    assignments: tuple[NodeAssignment, ...] = ()
    relation_states: tuple[dict[str, object], ...] = ()
    geometry_latency_ms: float = 0.0

    def detection_for(self, node_id: str | None) -> ObjectDetection | None:
        if node_id is None:
            return None
        for assignment in self.assignments:
            if assignment.node_id == node_id:
                return assignment.detection
        return None


def _clamp(value: float) -> float:
    return 0.0 if not math.isfinite(value) else min(1.0, max(0.0, value))


def _valid_detection(value: ObjectDetection) -> bool:
    try:
        confidence = float(value.confidence)
        ymin, xmin, ymax, xmax = (float(item) for item in value.box)
    except (TypeError, ValueError):
        return False
    return (
        bool(str(value.label).strip())
        and math.isfinite(confidence)
        and all(math.isfinite(item) for item in (ymin, xmin, ymax, xmax))
        and 0.0 <= ymin < ymax <= 1.0
        and 0.0 <= xmin < xmax <= 1.0
    )


def _coerce_detection(value: ObjectDetection | Mapping[str, Any]) -> ObjectDetection | None:
    if isinstance(value, ObjectDetection):
        return value if _valid_detection(value) else None
    if not isinstance(value, Mapping):
        return None
    raw_box = value.get("box", value.get("bbox"))
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        detection = ObjectDetection(
            str(value.get("label", "")).strip(),
            float(value.get("confidence", value.get("score", 0.0))),
            tuple(float(item) for item in raw_box),
        )
    except (TypeError, ValueError):
        return None
    return detection if _valid_detection(detection) else None


def intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Return normalized intersection area for yxyx boxes."""

    ly0, lx0, ly1, lx1 = left
    ry0, rx0, ry1, rx1 = right
    height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    return height * width


def box_area(box: tuple[float, float, float, float]) -> float:
    ymin, xmin, ymax, xmax = box
    return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)


def iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = intersection_area(left, right)
    union = box_area(left) + box_area(right) - intersection
    return 0.0 if union <= 0.0 else _clamp(intersection / union)


def class_aware_nms(
    detections: Sequence[ObjectDetection],
    *,
    iou_threshold: float = DEFAULT_NMS_IOU,
) -> tuple[ObjectDetection, ...]:
    """Suppress duplicate same-label boxes using deterministic confidence order."""

    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between zero and one")
    ordered = sorted(
        ((index, value) for index, value in enumerate(detections) if _valid_detection(value)),
        key=lambda value: (
            -float(value[1].confidence),
            value[1].label.casefold(),
            value[1].box,
            value[0],
        ),
    )
    kept: list[tuple[int, ObjectDetection]] = []
    for original_index, detection in ordered:
        if any(
            previous.label.casefold() == detection.label.casefold()
            and iou(previous.box, detection.box) > iou_threshold
            for _index, previous in kept
        ):
            continue
        kept.append((original_index, detection))
    kept.sort(key=lambda value: value[0])
    return tuple(value for _index, value in kept)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    ymin, xmin, ymax, xmax = box
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def _canonical_relation(value: str) -> str:
    normalized = str(value).strip().casefold()
    normalized = _RELATION_ALIASES.get(normalized, normalized)
    return SpatialRelationEnum(normalized).value


def _directional(delta: float, config: SpatialOntologyConfig) -> float:
    if delta <= config.directional_margin:
        return 0.0
    return _clamp((delta - config.directional_margin) / config.directional_scale)


def _inside_score(inner: ObjectDetection, outer: ObjectDetection) -> float:
    area = box_area(inner.box)
    return 0.0 if area <= 0.0 else _clamp(intersection_area(inner.box, outer.box) / area)


def _between_score(
    middle: ObjectDetection,
    first: ObjectDetection,
    second: ObjectDetection,
    config: SpatialOntologyConfig,
) -> float:
    bx, by = _center(middle.box)
    ax, ay = _center(first.box)
    cx, cy = _center(second.box)
    ux, uy = cx - ax, cy - ay
    vx, vy = bx - ax, by - ay
    norm_squared = ux * ux + uy * uy
    if norm_squared <= 1e-12:
        return 0.0
    norm = math.sqrt(norm_squared)
    projection = (ux * vx + uy * vy) / norm_squared
    perpendicular = abs(ux * vy - uy * vx) / norm
    if not (
        config.between_projection_min <= projection <= config.between_projection_max
        and perpendicular <= config.between_perpendicular_threshold
    ):
        return 0.0
    alignment = 1.0 - perpendicular / config.between_perpendicular_threshold
    midpoint = 1.0 - 2.0 * abs(projection - 0.5)
    return _clamp(alignment * midpoint)


def _depth_score(
    foreground: ObjectDetection,
    background: ObjectDetection,
    config: SpatialOntologyConfig,
) -> float:
    background_area = box_area(background.box)
    if background_area <= 0.0:
        return 0.0
    _foreground_x, foreground_y = _center(foreground.box)
    _background_x, background_y = _center(background.box)
    if foreground_y < background_y - config.depth_vertical_tolerance:
        return 0.0
    signal = config.depth_sigmoid_scale * (
        box_area(foreground.box) / background_area - config.depth_area_ratio
    )
    signal = min(60.0, max(-60.0, signal))
    return _clamp(1.0 / (1.0 + math.exp(-signal)))


def relation_score(
    constraint: RelativeSpatialConstraint,
    subject: ObjectDetection,
    object_detection: ObjectDetection,
    second_object: ObjectDetection | None = None,
    *,
    config: SpatialOntologyConfig | None = None,
) -> float:
    """Return a finite continuous score for one bound relation."""

    settings = config or SpatialOntologyConfig()
    if not (_valid_detection(subject) and _valid_detection(object_detection)):
        return 0.0
    relation = _canonical_relation(constraint.relation)
    subject_x, subject_y = _center(subject.box)
    object_x, object_y = _center(object_detection.box)
    if relation == SpatialRelationEnum.LEFT_OF.value:
        return _directional(object_x - subject_x, settings)
    if relation == SpatialRelationEnum.RIGHT_OF.value:
        return _directional(subject_x - object_x, settings)
    if relation == SpatialRelationEnum.ABOVE.value:
        return _directional(object_y - subject_y, settings)
    if relation == SpatialRelationEnum.BELOW.value:
        return _directional(subject_y - object_y, settings)
    if relation == SpatialRelationEnum.NEAR.value:
        distance = math.hypot(subject_x - object_x, subject_y - object_y)
        return _clamp(1.0 - distance / settings.near_threshold)
    if relation == SpatialRelationEnum.FAR_FROM.value:
        distance = math.hypot(subject_x - object_x, subject_y - object_y)
        return _clamp((distance - settings.far_min) / (settings.far_max - settings.far_min))
    if relation == SpatialRelationEnum.INSIDE.value:
        return _inside_score(subject, object_detection)
    if relation == SpatialRelationEnum.SURROUNDING.value:
        return _inside_score(object_detection, subject)
    if relation == SpatialRelationEnum.OVERLAPPING.value:
        return iou(subject.box, object_detection.box)
    if relation == SpatialRelationEnum.BETWEEN.value:
        if second_object is None or not _valid_detection(second_object):
            return 0.0
        return _between_score(subject, object_detection, second_object, settings)
    if relation == SpatialRelationEnum.IN_FRONT_OF.value:
        return _depth_score(subject, object_detection, settings)
    if relation == SpatialRelationEnum.BEHIND.value:
        return _depth_score(object_detection, subject, settings)
    return 0.0


def relation_satisfied(
    constraint: RelativeSpatialConstraint,
    subject: ObjectDetection,
    object_detection: ObjectDetection,
    *,
    margin: float = RELATION_MARGIN,
    inside_coverage: float = INSIDE_COVERAGE,
    second_object: ObjectDetection | None = None,
) -> bool:
    """Backward-compatible boolean projection of continuous relation scoring."""

    config = SpatialOntologyConfig(
        directional_margin=margin,
        inside_satisfied_coverage=inside_coverage,
    )
    score = relation_score(
        constraint,
        subject,
        object_detection,
        second_object,
        config=config,
    )
    relation = _canonical_relation(constraint.relation)
    if relation in {SpatialRelationEnum.INSIDE.value, SpatialRelationEnum.SURROUNDING.value}:
        return score >= inside_coverage
    return score > 0.0


def _assignment_key(assignments: Mapping[str, tuple[int, ObjectDetection]]) -> tuple[object, ...]:
    return tuple(
        (node_id, value[1].label.casefold(), -value[1].confidence, value[1].box)
        for node_id, value in sorted(assignments.items())
    )


def _constraint_nodes(constraint: RelativeSpatialConstraint) -> tuple[str, ...]:
    values = [constraint.subject_node_id, constraint.object_node_id]
    if constraint.arity == 3 and constraint.second_object_node_id:
        values.append(constraint.second_object_node_id)
    return tuple(values)


def _constraint_weight(
    constraint: RelativeSpatialConstraint,
    config: SpatialOntologyConfig,
) -> float:
    return config.perspective_weight if constraint.verification_type == "heuristic_2d" else 1.0


class SpatialGraphEvaluator:
    """Evaluate a bounded distinct-detection spatial constraint graph."""

    def __init__(self, config: SpatialOntologyConfig | None = None) -> None:
        self.config = config or SpatialOntologyConfig()

    def _resolved_score(
        self,
        constraints: Sequence[RelativeSpatialConstraint],
        assignments: Mapping[str, tuple[int, ObjectDetection]],
    ) -> tuple[float, int]:
        weighted = total_weight = 0.0
        resolved = 0
        for constraint in constraints:
            node_ids = _constraint_nodes(constraint)
            if any(node_id not in assignments for node_id in node_ids):
                continue
            subject = assignments[constraint.subject_node_id][1]
            object_detection = assignments[constraint.object_node_id][1]
            second = (
                assignments[constraint.second_object_node_id][1]
                if constraint.second_object_node_id
                else None
            )
            weight = _constraint_weight(constraint, self.config)
            weighted += weight * relation_score(
                constraint, subject, object_detection, second, config=self.config
            )
            total_weight += weight
            resolved += 1
        return (weighted / total_weight if total_weight else 0.0), resolved

    def evaluate(
        self,
        constraints: Sequence[RelativeSpatialConstraint],
        detections: Sequence[ObjectDetection],
        *,
        detections_are_nms: bool = False,
    ) -> SpatialGraphMatch:
        started = time.perf_counter()
        bound = tuple(value for value in constraints if value.bound)
        if not bound:
            return SpatialGraphMatch("not_requested")

        filtered = (
            tuple(detections)
            if detections_are_nms
            else class_aware_nms(
                detections, iou_threshold=self.config.nms_iou_threshold
            )
        )
        node_labels: dict[str, str] = {}
        for constraint in bound:
            assert constraint.subject is not None and constraint.object_subject is not None
            node_labels.setdefault(constraint.subject_node_id, constraint.subject)
            node_labels.setdefault(constraint.object_node_id, constraint.object_subject)
            if constraint.second_object_node_id and constraint.second_object_subject:
                node_labels.setdefault(
                    constraint.second_object_node_id, constraint.second_object_subject
                )

        node_candidates: dict[str, tuple[tuple[int, ObjectDetection], ...]] = {}
        for node_id, label in node_labels.items():
            matches = [
                (index, detection)
                for index, detection in enumerate(filtered)
                if labels_compatible(label, detection.label)
            ]
            matches.sort(
                key=lambda value: (
                    -value[1].confidence,
                    value[1].label.casefold(),
                    value[1].box,
                    value[0],
                )
            )
            if not matches:
                return SpatialGraphMatch(
                    "inconclusive",
                    geometry_latency_ms=round((time.perf_counter() - started) * 1_000, 6),
                )
            node_candidates[node_id] = tuple(
                matches[: self.config.max_detections_per_node]
            )

        states: list[tuple[dict[str, tuple[int, ObjectDetection]], frozenset[int], float]] = [
            ({}, frozenset(), 0.0)
        ]
        for node_id in sorted(node_labels):
            expanded: list[
                tuple[dict[str, tuple[int, ObjectDetection]], frozenset[int], float]
            ] = []
            for assignments, used, confidence_sum in states:
                for detection_index, detection in node_candidates[node_id]:
                    if detection_index in used:
                        continue
                    updated = dict(assignments)
                    updated[node_id] = (detection_index, detection)
                    expanded.append(
                        (updated, used | {detection_index}, confidence_sum + detection.confidence)
                    )
            if not expanded:
                return SpatialGraphMatch(
                    "inconclusive",
                    geometry_latency_ms=round((time.perf_counter() - started) * 1_000, 6),
                )
            scored_expanded = [
                (*self._resolved_score(bound, value[0]), value)
                for value in expanded
            ]
            scored_expanded.sort(
                key=lambda value: (
                    -value[0], -value[1], -value[2][2], _assignment_key(value[2][0])
                )
            )
            states = [value[2] for value in scored_expanded[: self.config.beam_width]]

        ranked: list[
            tuple[float, float, tuple[object, ...], dict[str, tuple[int, ObjectDetection]]]
        ] = []
        for assignments, _used, confidence_sum in states:
            score, _resolved = self._resolved_score(bound, assignments)
            ranked.append((score, confidence_sum, _assignment_key(assignments), assignments))
        ranked.sort(key=lambda value: (-value[0], -value[1], value[2]))
        best_score, _confidence, _key, best_assignments = ranked[0]

        relation_states: list[dict[str, object]] = []
        for constraint in bound:
            subject = best_assignments[constraint.subject_node_id][1]
            object_detection = best_assignments[constraint.object_node_id][1]
            second = (
                best_assignments[constraint.second_object_node_id][1]
                if constraint.second_object_node_id
                else None
            )
            score = relation_score(
                constraint, subject, object_detection, second, config=self.config
            )
            labels = [subject.label, object_detection.label]
            if second is not None:
                labels.append(second.label)
            relation_states.append(
                {
                    "relation": _canonical_relation(constraint.relation),
                    "arity": constraint.arity,
                    "subject": constraint.subject or "",
                    "object_subject": constraint.object_subject or "",
                    "second_object_subject": constraint.second_object_subject or "",
                    "score": round(score, 6),
                    "matched_labels": tuple(labels),
                    "verification_type": constraint.verification_type,
                }
            )

        assignments = tuple(
            NodeAssignment(node_id, value[1])
            for node_id, value in sorted(best_assignments.items())
        )
        return SpatialGraphMatch(
            "verified",
            _clamp(best_score),
            assignments,
            tuple(relation_states),
            round((time.perf_counter() - started) * 1_000, 6),
        )


def match_spatial_graph(
    constraints: Sequence[RelativeSpatialConstraint],
    detections: Sequence[ObjectDetection],
    *,
    margin: float = RELATION_MARGIN,
    inside_coverage: float = INSIDE_COVERAGE,
    max_detections_per_node: int = DEFAULT_MAX_DETECTIONS_PER_NODE,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    config: SpatialOntologyConfig | None = None,
    detections_are_nms: bool = False,
) -> SpatialGraphMatch:
    """Compatibility wrapper returning assignments and sanitized graph diagnostics."""

    settings = config or SpatialOntologyConfig(
        directional_margin=margin,
        inside_satisfied_coverage=inside_coverage,
        max_detections_per_node=max_detections_per_node,
        beam_width=beam_width,
    )
    return SpatialGraphEvaluator(settings).evaluate(
        constraints, detections, detections_are_nms=detections_are_nms
    )


def evaluate_spatial_graph(
    detections: Sequence[ObjectDetection | Mapping[str, Any]],
    constraints: Sequence[RelativeSpatialConstraint],
    *,
    config: SpatialOntologyConfig | None = None,
) -> float:
    """Return aggregate alignment in ``[0, 1]``; unavailable evidence maps to zero."""

    normalized = tuple(
        detection
        for raw in detections
        if (detection := _coerce_detection(raw)) is not None
    )
    match = SpatialGraphEvaluator(config).evaluate(constraints, normalized)
    return float(match.score) if match.state == "verified" and match.score is not None else 0.0


def forbidden_object_matches(
    label: str,
    detections: Sequence[ObjectDetection],
    *,
    confidence_threshold: float = 0.60,
) -> tuple[ObjectDetection, ...]:
    """Return exact object-evidence matches for a plain negation constraint."""

    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    return tuple(
        value
        for value in class_aware_nms(detections)
        if value.confidence > confidence_threshold and labels_compatible(label, value.label)
    )


__all__ = [
    "DEFAULT_BEAM_WIDTH",
    "DEFAULT_MAX_DETECTIONS_PER_NODE",
    "DEFAULT_NMS_IOU",
    "INSIDE_COVERAGE",
    "NodeAssignment",
    "RELATION_MARGIN",
    "SpatialGraphEvaluator",
    "SpatialGraphMatch",
    "SpatialOntologyConfig",
    "SpatialRelationEnum",
    "box_area",
    "class_aware_nms",
    "evaluate_spatial_graph",
    "forbidden_object_matches",
    "intersection_area",
    "iou",
    "match_spatial_graph",
    "relation_satisfied",
    "relation_score",
]