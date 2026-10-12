"""Bounded HSV verification and Advanced Textual KIS reranking."""

from __future__ import annotations

import hashlib
import io
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from src.contracts import FrameRecord
from src.query.query_decomposer import (
    ColorConstraint,
    NegationConstraint,
    ObjectInstance,
)
from src.query.translator import QueryProcessing
from src.retrieval.fusion import canonical_key
from src.retrieval.qa_media import ImageResolver
from src.retrieval.spatial_graph import (
    SpatialGraphMatch,
    SpatialOntologyConfig,
    class_aware_nms,
    forbidden_object_matches,
    iou,
    match_spatial_graph,
)
from src.retrieval.spatial_rerank import (
    FrameRepository,
    ObjectDetection,
    ObjectEvidence,
    OrganizerObjectArchive,
    labels_compatible,
    labels_positive_compatible,
    spatial_constraint_score,
)

_COLORS = ("red", "blue", "yellow", "green", "black", "white", "pink", "purple")


class HSVColorVerifier:
    """Compute all supported color ratios once per frame/ROI in a bounded LRU."""

    def __init__(
        self,
        resolver: ImageResolver,
        *,
        max_entries: int = 512,
        max_cache_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if min(max_entries, max_cache_bytes) < 1:
            raise ValueError("color cache bounds must be positive")
        self.resolver = resolver
        self.max_entries = int(max_entries)
        self.max_cache_bytes = int(max_cache_bytes)
        self._guard = threading.RLock()
        self._cache: OrderedDict[str, tuple[dict[str, float], int]] = OrderedDict()
        self._cache_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._request = threading.local()

    def begin_request(self) -> None:
        """At most eight decoded frames, shared by ROIs within this thread's request."""
        self._request.images = {}

    @property
    def fingerprint(self) -> str:
        resolver_value = str(getattr(self.resolver, "fingerprint", type(self.resolver).__name__))
        payload = (
            f"hsv-color-v3-batch-roi:{resolver_value}:{self.max_entries}:"
            f"{self.max_cache_bytes}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def diagnostics(self) -> dict[str, int]:
        with self._guard:
            return {
                "cache_entries": len(self._cache),
                "cache_bytes": self._cache_bytes,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

    @staticmethod
    def _mask(hsv: np.ndarray, color: str) -> np.ndarray:
        hue = hsv[..., 0]
        saturation = hsv[..., 1]
        value = hsv[..., 2]
        saturated = (saturation >= 70) & (value >= 45)
        if color == "red":
            return saturated & ((hue <= 12) | (hue >= 244))
        if color == "yellow":
            return saturated & (hue >= 24) & (hue <= 48)
        if color == "green":
            return saturated & (hue >= 50) & (hue <= 112)
        if color == "blue":
            return saturated & (hue >= 125) & (hue <= 180)
        if color == "purple":
            return saturated & (hue >= 181) & (hue <= 220)
        if color == "pink":
            return (saturation >= 45) & (value >= 90) & (hue >= 221) & (hue <= 243)
        if color == "black":
            return value <= 55
        if color == "white":
            return (value >= 205) & (saturation <= 50)
        raise ValueError("unsupported color constraint")

    @staticmethod
    def _crop(
        image: Image.Image,
        box: tuple[float, float, float, float] | None,
    ) -> Image.Image:
        if box is None:
            return image
        ymin, xmin, ymax, xmax = box
        width, height = image.size
        left = max(0, min(width - 1, int(xmin * width)))
        upper = max(0, min(height - 1, int(ymin * height)))
        right = max(left + 1, min(width, int(np.ceil(xmax * width))))
        lower = max(upper + 1, min(height, int(np.ceil(ymax * height))))
        return image.crop((left, upper, right, lower))

    def ratios(
        self,
        frame: FrameRecord,
        *,
        box: tuple[float, float, float, float] | None,
    ) -> tuple[dict[str, float], bool]:
        box_key = "full" if box is None else ",".join(f"{value:.5f}" for value in box)
        frame_revision = (
            f"{frame.source_sha256}:{frame.extractor_version}:"
            f"{frame.batch_id}:{frame.video_id}:{frame.source_frame_idx}:{frame.image_uri}"
        )
        key = hashlib.sha256(
            f"{self.fingerprint}:{frame_revision}:{box_key}".encode()
        ).hexdigest()
        with self._guard:
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
                self._cache_hits += 1
                return dict(cached[0]), True
            self._cache_misses += 1

        images = getattr(self._request, "images", None)
        rgb = images.get(frame_revision) if images is not None else None
        if rgb is None:
            resolved = self.resolver.resolve(frame)
            with Image.open(io.BytesIO(resolved.data)) as source:
                rgb = source.convert("RGB")
            if images is not None and len(images) < 8 and rgb.width * rgb.height <= 1024 * 1024:
                images[frame_revision] = rgb
        hsv = np.asarray(self._crop(rgb, box).convert("HSV"), dtype=np.uint8)
        if hsv.size == 0:
            raise ValueError("empty color region")
        denominator = hsv.shape[0] * hsv.shape[1]
        descriptor = {
            color: float(np.count_nonzero(self._mask(hsv, color)) / denominator)
            for color in _COLORS
        }
        size = 512
        with self._guard:
            prior = self._cache.pop(key, None)
            if prior is not None:
                self._cache_bytes -= prior[1]
            self._cache[key] = (dict(descriptor), size)
            self._cache_bytes += size
            while self._cache and (
                len(self._cache) > self.max_entries
                or self._cache_bytes > self.max_cache_bytes
            ):
                _old_key, (_old_value, old_size) = self._cache.popitem(last=False)
                self._cache_bytes -= old_size
        return descriptor, False

    def ratio(
        self,
        frame: FrameRecord,
        color: str,
        *,
        box: tuple[float, float, float, float] | None,
    ) -> tuple[float, bool]:
        descriptor, cache_hit = self.ratios(frame, box=box)
        if color not in descriptor:
            raise ValueError("unsupported color constraint")
        return descriptor[color], cache_hit


class AdvancedKISReranker:
    """Verify absolute/relative spatial, color binding, and negation evidence."""

    def __init__(
        self,
        repository: FrameRepository,
        object_archive: OrganizerObjectArchive,
        color_verifier: HSVColorVerifier,
        *,
        spatial_depth: int = 100,
        color_depth: int = 12,
        spatial_beta: float = 0.35,
        color_absent_multiplier: float = 0.65,
        color_workers: int = 2,
        relation_beta: float = 0.40,
        strict_color_threshold: float = 0.12,
        strict_color_beta: float = 0.25,
        strict_mismatch_multiplier: float = 0.10,
        negation_confidence_threshold: float = 0.60,
        negation_multiplier: float = 0.10,
        spatial_ontology: SpatialOntologyConfig | None = None,
        verification_budget_ms: float = 20.0,
    ) -> None:
        if not 1 <= spatial_depth <= 100 or not 1 <= color_depth <= spatial_depth:
            raise ValueError("advanced KIS depths are invalid")
        for name, value in (
            ("spatial_beta", spatial_beta),
            ("relation_beta", relation_beta),
            ("strict_color_beta", strict_color_beta),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name, value in (
            ("color_absent_multiplier", color_absent_multiplier),
            ("strict_mismatch_multiplier", strict_mismatch_multiplier),
            ("negation_multiplier", negation_multiplier),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0.0 < strict_color_threshold <= 1.0:
            raise ValueError("strict_color_threshold must be in (0, 1]")
        if not 0.0 <= negation_confidence_threshold <= 1.0:
            raise ValueError("negation confidence threshold must be in [0, 1]")
        if not 1 <= color_workers <= 2:
            raise ValueError("color_workers must be one or two")
        if not 0 < verification_budget_ms <= 5_000:
            raise ValueError("verification_budget_ms must be in (0, 5000]")
        self.repository = repository
        self.object_archive = object_archive
        self.color_verifier = color_verifier
        self.spatial_depth = int(spatial_depth)
        self.color_depth = int(color_depth)
        self.spatial_beta = float(spatial_beta)
        self.color_absent_multiplier = float(color_absent_multiplier)
        self.color_workers = int(color_workers)
        self.relation_beta = float(relation_beta)
        self.strict_color_threshold = float(strict_color_threshold)
        self.strict_color_beta = float(strict_color_beta)
        self.strict_mismatch_multiplier = float(strict_mismatch_multiplier)
        self.negation_confidence_threshold = float(negation_confidence_threshold)
        self.negation_multiplier = float(negation_multiplier)
        self.spatial_ontology = spatial_ontology or SpatialOntologyConfig()
        self.verification_budget_ms = float(verification_budget_ms)
        self._state = threading.local()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            (
                f"advanced-kis-v6-configured-budget:{self.object_archive.fingerprint}:"
                f"{self.color_verifier.fingerprint}:{self.spatial_depth}:{self.color_depth}:"
                f"{self.spatial_beta}:{self.color_absent_multiplier}:{self.relation_beta}:"
                f"{self.strict_color_threshold}:{self.strict_color_beta}:"
                f"{self.strict_mismatch_multiplier}:{self.negation_confidence_threshold}:"
                f"{self.negation_multiplier}:{self.spatial_ontology!r}:"
                f"{self.verification_budget_ms}"
            ).encode()
        ).hexdigest()

    @property
    def last_diagnostics(self) -> dict[str, Any]:
        return dict(getattr(self._state, "diagnostics", {}))

    @staticmethod
    def _batch_id(item: Mapping[str, Any]) -> str | None:
        payload = item.get("payload")
        source = payload if isinstance(payload, Mapping) else item
        value = source.get("batch_id")
        return value if isinstance(value, str) and value else None

    def _frame(self, item: Mapping[str, Any]) -> FrameRecord | None:
        batch_id = self._batch_id(item)
        if batch_id is None:
            return None
        try:
            video_id, source_frame_idx = canonical_key(item)
            return self.repository.get_frame(batch_id, video_id, source_frame_idx)
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _best_box(
        constraint: ColorConstraint | NegationConstraint,
        evidence: ObjectEvidence,
        graph_match: SpatialGraphMatch | None = None,
    ) -> tuple[float, float, float, float] | None:
        node_id = getattr(constraint, "node_id", None)
        assigned = graph_match.detection_for(node_id) if graph_match is not None else None
        if assigned is not None:
            return assigned.box
        if constraint.subject is None or evidence.state != "ready":
            return None
        matching = [
            value
            for value in evidence.detections
            if labels_compatible(constraint.subject, value.label)
        ]
        return max(matching, key=lambda value: value.confidence).box if matching else None

    def _assign_instance_detections(
        self,
        constraints: Sequence[ColorConstraint],
        instances: Sequence[ObjectInstance],
        evidence: ObjectEvidence,
        graph_match: SpatialGraphMatch | None,
    ) -> dict[str, tuple[ObjectDetection, bool]]:
        """Return a bounded maximum-cardinality one-to-one instance assignment."""

        if evidence.state != "ready":
            return {}
        instance_by_id = {value.instance_id: value for value in instances}
        node_ids = tuple(
            dict.fromkeys(
                value.node_id
                for value in constraints
                if value.bound and value.node_id and value.node_id in instance_by_id
            )
        )
        if not node_ids:
            return {}

        candidates: dict[str, tuple[tuple[int, ObjectDetection, bool], ...]] = {}
        for node_id in node_ids:
            instance = instance_by_id[node_id]
            assigned = graph_match.detection_for(node_id) if graph_match is not None else None
            matches: list[tuple[int, ObjectDetection, bool]] = []
            for index, detection in enumerate(evidence.detections):
                if assigned is not None and detection.box != assigned.box:
                    continue
                if not labels_positive_compatible(
                    instance.canonical_label,
                    detection.label,
                ):
                    continue
                matches.append(
                    (
                        index,
                        detection,
                        labels_compatible(instance.label, detection.label),
                    )
                )
            matches.sort(
                key=lambda value: (
                    not value[2],
                    -float(value[1].confidence),
                    value[1].label.casefold(),
                    value[1].box,
                    value[0],
                )
            )
            candidates[node_id] = tuple(
                matches[: self.spatial_ontology.max_detections_per_node]
            )

        states: list[
            tuple[dict[str, tuple[int, ObjectDetection, bool]], frozenset[int]]
        ] = [({}, frozenset())]
        for node_id in node_ids:
            expanded = list(states)
            for selected, used in states:
                for detection_index, detection, exact in candidates[node_id]:
                    if detection_index in used:
                        continue
                    if any(
                        iou(detection.box, previous[1].box)
                        > self.spatial_ontology.nms_iou_threshold
                        for previous in selected.values()
                    ):
                        continue
                    updated = dict(selected)
                    updated[node_id] = (detection_index, detection, exact)
                    expanded.append((updated, used | {detection_index}))

            def state_key(
                value: tuple[
                    dict[str, tuple[int, ObjectDetection, bool]],
                    frozenset[int],
                ],
            ) -> tuple[object, ...]:
                selected = value[0]
                return (
                    -len(selected),
                    -sum(item[2] for item in selected.values()),
                    -sum(float(item[1].confidence) for item in selected.values()),
                    tuple(
                        (
                            key,
                            item[1].label.casefold(),
                            item[1].box,
                        )
                        for key, item in sorted(selected.items())
                    ),
                )

            expanded.sort(key=state_key)
            states = expanded[: self.spatial_ontology.beam_width]

        best = states[0][0] if states else {}
        return {
            node_id: (value[1], value[2])
            for node_id, value in best.items()
        }

    @staticmethod
    def _sort(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            values,
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", 0)),
            ),
        )

    @staticmethod
    def _apply_penalty(
        output: dict[str, Any],
        multiplier: float,
        *,
        field: str,
    ) -> None:
        if output.get(field):
            return
        output["score"] = float(output.get("score", 0.0)) * multiplier
        output[field] = True

    @staticmethod
    def _apply_rank_calibrated_delta(
        output: dict[str, Any],
        delta_ratio: float,
        *,
        field: str,
    ) -> None:
        """Blend bounded evidence additively relative to the Stage-1 score."""

        if output.get(field):
            return
        base_score = float(output.get("score", 0.0))
        delta = base_score * float(delta_ratio)
        output["score"] = max(0.0, base_score + delta)
        output[field] = True
        output[f"{field}_delta"] = delta

    def _deadline_fallback(
        self,
        stage1: list[dict[str, Any]],
        *,
        started: float,
        phase: str,
        processed_candidates: int,
    ) -> list[dict[str, Any]]:
        """Discard partial Tier-2 mutations and return the untouched Stage-1 order."""

        self._state.diagnostics = {
            "state": "deadline_exceeded",
            "stage_timings_ms": dict(getattr(self._state, "stage_timings_ms", {})),
            "precision_profile": "automatic",
            "deadline_budget_ms": self.verification_budget_ms,
            "deadline_phase": phase,
            "processed_candidate_count": processed_candidates,
            "prefetch_latency_ms": round(
                float(getattr(self._state, "prefetch_latency_ms", 0.0)),
                3,
            ),
            "spatial_candidates": 0,
            "relation_candidates": 0,
            "negation_candidates": 0,
            "color_candidates": 0,
            "total_latency_ms": round(
                max(0.0, (time.perf_counter() - started) * 1_000),
                3,
            ),
        }
        return stage1

    def _verify_colors(
        self,
        item: dict[str, Any],
        frame: FrameRecord | None,
        evidence: ObjectEvidence,
        constraints: Sequence[ColorConstraint],
        instances: Sequence[ObjectInstance],
        negations: Sequence[NegationConstraint],
        *,
        graph_match: SpatialGraphMatch | None,
        strict: bool,
    ) -> dict[str, Any]:
        output = dict(item)
        if frame is None:
            output["color_rerank_state"] = "inconclusive"
            return output

        states: list[dict[str, Any]] = []
        verified_positive_ratios: list[float] = []
        verified_mismatch = False
        forbidden_color_present = False
        fallback_box = (2.0 / 7.0, 2.0 / 7.0, 5.0 / 7.0, 5.0 / 7.0)
        instance_by_id = {value.instance_id: value for value in instances}
        assignments = self._assign_instance_detections(
            constraints,
            instances,
            evidence,
            graph_match,
        )
        required_nodes = {
            value.node_id
            for value in constraints
            if value.bound and value.node_id and value.node_id in instance_by_id
        }
        complete_exact_assignment = bool(required_nodes) and all(
            node_id in assignments and assignments[node_id][1]
            for node_id in required_nodes
        )

        for constraint in constraints:
            if not constraint.bound:
                states.append({"color": constraint.color, "state": "unbound"})
                continue
            assigned = assignments.get(constraint.node_id or "")
            matched_detection = assigned[0] if assigned is not None else None
            exact_class = assigned[1] if assigned is not None else False
            if matched_detection is not None:
                box = matched_detection.box
            elif constraint.node_id and constraint.node_id in instance_by_id:
                box = None
            else:
                box = self._best_box(constraint, evidence, graph_match)
                exact_class = box is not None
            if box is None and constraint.color in {"black", "white"}:
                states.append(
                    {
                        "color": constraint.color,
                        "subject": constraint.subject or "",
                        "state": "unlocalized",
                        "reason": "fallback_background_color_unsafe",
                    }
                )
                continue
            if box is None:
                try:
                    full_ratio, full_cache_hit = self.color_verifier.ratio(
                        frame,
                        constraint.color,
                        box=None,
                    )
                    center_ratio, center_cache_hit = self.color_verifier.ratio(
                        frame,
                        constraint.color,
                        box=fallback_box,
                    )
                except Exception:
                    states.append(
                        {
                            "color": constraint.color,
                            "subject": constraint.subject or "",
                            "state": "unlocalized",
                            "reason": "fallback_color_unavailable",
                        }
                    )
                    continue
                ratio = max(full_ratio, center_ratio)
                present = ratio >= self.strict_color_threshold
                if present:
                    verified_positive_ratios.append(ratio)
                states.append(
                    {
                        "color": constraint.color,
                        "subject": constraint.subject or "",
                        "state": "present_fallback" if present else "unlocalized",
                        "scope": (
                            "center_3x3" if center_ratio >= full_ratio else "full_frame"
                        ),
                        "ratio": round(ratio, 5),
                        "full_frame_ratio": round(full_ratio, 5),
                        "center_3x3_ratio": round(center_ratio, 5),
                        "cache_hit": full_cache_hit and center_cache_hit,
                        "positive_only": True,
                        "reason": None if present else "fallback_mismatch_is_neutral",
                    }
                )
                continue
            try:
                ratio, cache_hit = self.color_verifier.ratio(
                    frame,
                    constraint.color,
                    box=box,
                )
            except Exception:
                states.append({"color": constraint.color, "state": "inconclusive"})
                continue
            threshold = self.strict_color_threshold if strict else 0.03
            present = ratio >= threshold
            mismatch_is_conclusive = (
                not required_nodes
                or (complete_exact_assignment and exact_class)
            )
            verified_mismatch = verified_mismatch or (
                not present and mismatch_is_conclusive
            )
            if present:
                verified_positive_ratios.append(ratio)
            states.append(
                {
                    "color": constraint.color,
                    "subject": constraint.subject or "",
                    "instance_id": constraint.node_id,
                    "state": (
                        "present"
                        if present
                        else "absent"
                        if mismatch_is_conclusive
                        else "inconclusive_mismatch"
                    ),
                    "scope": "roi" if box is not None else "full_frame",
                    "ratio": round(ratio, 5),
                    "cache_hit": cache_hit,
                    "matched_label": (
                        matched_detection.label if matched_detection is not None else None
                    ),
                    "exact_class": exact_class,
                    "positive_only": not mismatch_is_conclusive,
                    "reason": (
                        None
                        if present or mismatch_is_conclusive
                        else "broad_or_partial_assignment_mismatch_is_neutral"
                    ),
                }
            )

        if strict:
            for constraint in negations:
                if not constraint.bound or constraint.color is None:
                    continue
                box = self._best_box(constraint, evidence, graph_match)
                if box is None:
                    states.append(
                        {
                            "color": constraint.color,
                            "subject": constraint.subject or "",
                            "state": "inconclusive",
                            "negated": True,
                        }
                    )
                    continue
                try:
                    ratio, cache_hit = self.color_verifier.ratio(
                        frame,
                        constraint.color,
                        box=box,
                    )
                except Exception:
                    states.append(
                        {
                            "color": constraint.color,
                            "subject": constraint.subject or "",
                            "state": "inconclusive",
                            "negated": True,
                        }
                    )
                    continue
                present = ratio >= self.strict_color_threshold
                forbidden_color_present = forbidden_color_present or present
                states.append(
                    {
                        "color": constraint.color,
                        "subject": constraint.subject or "",
                        "state": "forbidden_present" if present else "forbidden_absent",
                        "scope": "roi",
                        "ratio": round(ratio, 5),
                        "cache_hit": cache_hit,
                        "negated": True,
                    }
                )

        if verified_mismatch or forbidden_color_present:
            multiplier = (
                self.strict_mismatch_multiplier if strict else self.color_absent_multiplier
            )
            self._apply_rank_calibrated_delta(
                output,
                multiplier - 1.0,
                field="color_penalty_applied",
            )
        elif strict and verified_positive_ratios:
            color_score = float(sum(verified_positive_ratios) / len(verified_positive_ratios))
            self._apply_rank_calibrated_delta(
                output,
                self.strict_color_beta * color_score,
                field="color_reward_applied",
            )
            output["color_score"] = color_score

        output["color_rerank_state"] = (
            "verified"
            if any(
                value["state"]
                in {
                    "present",
                    "absent",
                    "present_fallback",
                    "forbidden_present",
                    "forbidden_absent",
                }
                for value in states
            )
            else "inconclusive"
        )
        if output["color_rerank_state"] == "inconclusive":
            output["verification_inconclusive_reason"] = "object_or_roi_unlocalized"
        if states:
            evidence_items = list(output.get("evidence", ()))
            evidence_items.append(
                {
                    "channel": "hsv_color",
                    "constraints": states,
                    "instance_assignment_complete_exact": complete_exact_assignment,
                    "assigned_instance_count": len(assignments),
                }
            )
            output["evidence"] = evidence_items
        return output

    def __call__(
        self,
        processing: QueryProcessing,
        candidates: list[dict[str, Any]],
        *,
        precision_profile: str = "automatic",
    ) -> list[dict[str, Any]]:
        if precision_profile not in {
            "automatic", "standard", "advanced", "experimental_cross_modal"
        }:
            raise ValueError("unsupported KIS precision profile")
        stage1 = [dict(item) for item in candidates]
        strict = True
        started = time.perf_counter()
        self._state.prefetch_latency_ms = 0.0
        self._state.stage_timings_ms = {}
        begin_colors = getattr(self.color_verifier, "begin_request", None)
        if callable(begin_colors):
            begin_colors()
        decomposition = processing.decomposition
        requested = bool(decomposition and decomposition.bound_constraint_count)
        if not requested or decomposition is None:
            self._state.diagnostics = {
                "state": "not_requested",
                "precision_profile": "automatic",
                "spatial_candidates": 0,
                "relation_candidates": 0,
                "negation_candidates": 0,
                "color_candidates": 0,
            }
            return stage1

        identities = [(self._batch_id(item), *canonical_key(item)) for item in stage1]
        initial_ranks = {key: rank for rank, key in enumerate(identities, start=1)}
        rank_by_object = {
            id(item): initial_ranks[key] for item, key in zip(stage1, identities, strict=True)
        }
        spatial_started = time.perf_counter()
        evidence_by_key: dict[tuple[str | None, str, int], ObjectEvidence] = {}
        frame_by_key: dict[tuple[str | None, str, int], FrameRecord | None] = {}
        graph_by_key: dict[tuple[str | None, str, int], SpatialGraphMatch] = {}
        object_states: set[str] = set()
        reranked: list[dict[str, Any]] = []
        verified_spatial = verified_relations = verified_negations = 0
        relation_elapsed_ms = 0.0
        geometry_elapsed_ms = 0.0
        negation_elapsed_ms = 0.0

        prefetch_started = time.perf_counter()
        frame_snapshot: dict[tuple[str, str, int], FrameRecord | None] = {}
        evidence_snapshot: dict[tuple[str | None, str, int], ObjectEvidence] = {}
        get_frames = getattr(self.repository, "get_frames", None)
        metadata_started = time.perf_counter()
        if callable(get_frames):
            identities_by_batch: dict[str, list[tuple[str, int]]] = {}
            for batch_id, video_id, source_frame_idx in identities[: self.spatial_depth]:
                if batch_id is None:
                    continue
                identities_by_batch.setdefault(batch_id, []).append(
                    (video_id, source_frame_idx)
                )
            try:
                for batch_id, frame_ids in identities_by_batch.items():
                    for key, frame in get_frames(batch_id, frame_ids).items():
                        frame_snapshot[(batch_id, key[0], key[1])] = frame
            except (KeyError, TypeError, ValueError):
                frame_snapshot = {}

        self._state.stage_timings_ms["metadata"] = (time.perf_counter()-metadata_started)*1000
        object_started = time.perf_counter()
        for raw, (batch_id, *key) in zip(
            stage1[: self.spatial_depth], identities[: self.spatial_depth], strict=True
        ):
            frame = (
                frame_snapshot.get((batch_id, key[0], key[1]))
                if batch_id is not None
                else None
            )
            if frame is None:
                frame = self._frame(raw)
                if frame is not None and batch_id is not None:
                    frame_snapshot[(batch_id, key[0], key[1])] = frame
            evidence_snapshot[(batch_id, *key)] = (
                self.object_archive.get(frame)
                if frame is not None
                else ObjectEvidence("not_applicable")
            )
            if batch_id is not None:
                frame_snapshot[(batch_id, key[0], key[1])] = frame

        self._state.stage_timings_ms["object_prefetch"] = (time.perf_counter()-object_started)*1000
        prefetch_ms = (time.perf_counter() - prefetch_started) * 1_000
        self._state.prefetch_latency_ms = prefetch_ms
        # perf_counter is monotonic and high-resolution on Python 3.11 Windows;
        # monotonic's ~15 ms ticks are unsuitable for this 20 ms budget.
        deadline = time.perf_counter() + self.verification_budget_ms / 1_000
        deadline_exceeded = False

        def fallback_if_expired(
            phase: str,
            processed_candidates: int,
        ) -> list[dict[str, Any]] | None:
            if time.perf_counter() <= deadline:
                return None
            return self._deadline_fallback(
                stage1,
                started=started,
                phase=phase,
                processed_candidates=processed_candidates,
            )

        for index, raw in enumerate(stage1[: self.spatial_depth]):
            if time.perf_counter() > deadline:
                deadline_exceeded = True
                break
            item = dict(raw)
            rank_by_object[id(item)] = rank_by_object[id(raw)]
            batch_id, *key = identities[index]
            frame = (
                frame_snapshot.get((batch_id, key[0], key[1]))
                if batch_id is not None
                else None
            )
            if frame is None and (batch_id, key[0], key[1]) not in frame_snapshot:
                frame = self._frame(item)
            fallback = fallback_if_expired("frame_resolution", index)
            if fallback is not None:
                return fallback
            scoped_key = (batch_id, *key)
            frame_by_key[scoped_key] = frame
            evidence = evidence_snapshot.get(scoped_key)
            if evidence is None:
                evidence = (
                    self.object_archive.get(frame)
                    if frame is not None
                    else ObjectEvidence("not_applicable")
                )
            if strict and evidence.state == "ready":
                evidence = ObjectEvidence(
                    "ready",
                    class_aware_nms(
                        evidence.detections,
                        iou_threshold=self.spatial_ontology.nms_iou_threshold,
                    ),
                    evidence.cache_hit,
                    evidence.read_ms,
                )
            evidence_by_key[scoped_key] = evidence
            object_states.add(evidence.state)
            fallback = fallback_if_expired("object_evidence", index)
            if fallback is not None:
                return fallback

            scores: list[float] = []
            matches: list[dict[str, Any]] = []
            for constraint in decomposition.spatial_constraints:
                score, detections = spatial_constraint_score(constraint, evidence.detections)
                if score is None:
                    continue
                scores.append(score)
                matches.append(
                    {
                        "subject": constraint.subject,
                        "zone": constraint.zone,
                        "score": round(score, 5),
                        "matched_labels": sorted({value.label for value in detections}),
                        "perspective_proxy": constraint.perspective_proxy,
                    }
                )
            if scores:
                spatial_score = sum(scores) / len(scores)
                item["score"] = float(item.get("score", 0.0)) * (
                    1.0 + self.spatial_beta * spatial_score
                )
                item["spatial_score"] = spatial_score
                item["spatial_rerank_state"] = "verified"
                verified_spatial += 1
                evidence_items = list(item.get("evidence", ()))
                evidence_items.append(
                    {
                        "channel": "spatial_7x7",
                        "score": round(spatial_score, 5),
                        "constraints": matches,
                    }
                )
                item["evidence"] = evidence_items
            else:
                item["spatial_rerank_state"] = (
                    "inconclusive" if evidence.state == "ready" else evidence.state
                )

            fallback = fallback_if_expired("absolute_spatial", index)
            if fallback is not None:
                return fallback
            if strict and decomposition.relative_constraints:
                relation_started = time.perf_counter()
                graph_match = match_spatial_graph(
                    decomposition.relative_constraints,
                    evidence.detections,
                    config=self.spatial_ontology,
                    detections_are_nms=True,
                )
                graph_by_key[scoped_key] = graph_match
                item["relation_rerank_state"] = graph_match.state
                if graph_match.state == "verified" and graph_match.score is not None:
                    item["score"] = float(item.get("score", 0.0)) * (
                        1.0 + self.relation_beta * graph_match.score
                    )
                    item["relation_score"] = graph_match.score
                    item["_object_assignments"] = {
                        value.node_id: value.detection.box
                        for value in graph_match.assignments
                    }
                    verified_relations += 1
                    evidence_items = list(item.get("evidence", ()))
                    evidence_items.append(
                        {
                            "channel": "spatial_relation",
                            "score": round(graph_match.score, 5),
                            "constraints": list(graph_match.relation_states),
                        }
                    )
                    item["evidence"] = evidence_items
                geometry_elapsed_ms += graph_match.geometry_latency_ms
                relation_elapsed_ms += (time.perf_counter() - relation_started) * 1_000
                fallback = fallback_if_expired("relative_spatial", index)
                if fallback is not None:
                    return fallback

            if strict and evidence.state == "ready":
                negation_started = time.perf_counter()
                forbidden_states: list[dict[str, Any]] = []
                forbidden_present = False
                for constraint in decomposition.negation_constraints:
                    if (
                        not constraint.bound
                        or constraint.subject is None
                        or constraint.color is not None
                    ):
                        continue
                    detections = forbidden_object_matches(
                        constraint.subject,
                        evidence.detections,
                        confidence_threshold=self.negation_confidence_threshold,
                    )
                    present = bool(detections)
                    forbidden_present = forbidden_present or present
                    forbidden_states.append(
                        {
                            "subject": constraint.subject,
                            "state": (
                                "forbidden_present"
                                if present
                                else "inconclusive_unlocalized"
                            ),
                            "matched_labels": sorted({value.label for value in detections}),
                            "reason": None if present else "detector_absence_is_not_proof",
                        }
                    )
                if forbidden_states:
                    if forbidden_present:
                        verified_negations += 1
                    item["negation_rerank_state"] = (
                        "verified" if forbidden_present else "inconclusive"
                    )
                    if forbidden_present:
                        self._apply_penalty(
                            item,
                            self.negation_multiplier,
                            field="negation_penalty_applied",
                        )
                    evidence_items = list(item.get("evidence", ()))
                    evidence_items.append(
                        {"channel": "negative_object", "constraints": forbidden_states}
                    )
                    item["evidence"] = evidence_items
                negation_elapsed_ms += (time.perf_counter() - negation_started) * 1_000
                fallback = fallback_if_expired("negation", index)
                if fallback is not None:
                    return fallback
            reranked.append(item)
            fallback = fallback_if_expired("candidate_complete", index + 1)
            if fallback is not None:
                return fallback


        if deadline_exceeded:
            return self._deadline_fallback(
                stage1,
                started=started,
                phase="candidate_start",
                processed_candidates=len(reranked),
            )
        # These candidates have no Tier-2 work; keep all of them and their scores.
        reranked.extend(stage1[self.spatial_depth :])
        fallback = fallback_if_expired("candidate_loop", len(stage1))
        if fallback is not None:
            return fallback
        reranked = self._sort(reranked)
        fallback = fallback_if_expired("spatial_sort", len(stage1))
        if fallback is not None:
            return fallback
        spatial_ms = (time.perf_counter() - spatial_started) * 1_000
        self._state.stage_timings_ms["geometry"] = geometry_elapsed_ms

        color_started = time.perf_counter()
        head = reranked[: self.color_depth]
        tail = reranked[self.color_depth :]
        color_requested = bool(
            decomposition.color_constraints
            or any(value.color is not None for value in decomposition.negation_constraints)
        )
        if color_requested and head:
            verified_head: list[dict[str, Any]] = []
            for color_index, item in enumerate(head):
                fallback = fallback_if_expired("color_start", len(stage1) + color_index)
                if fallback is not None:
                    return fallback
                key = (self._batch_id(item), *canonical_key(item))
                color_item_started = time.perf_counter()
                verified_item = self._verify_colors(
                    item,
                    frame_by_key[key] if key in frame_by_key else self._frame(item),
                    evidence_by_key.get(key, ObjectEvidence("not_applicable")),
                    decomposition.color_constraints,
                    decomposition.object_instances,
                    decomposition.negation_constraints,
                    graph_match=graph_by_key.get(key),
                    strict=strict,
                )
                rank_by_object[id(verified_item)] = rank_by_object[id(item)]
                verified_head.append(verified_item)
                self._state.stage_timings_ms["color"] = self._state.stage_timings_ms.get("color", 0.0) + (time.perf_counter()-color_item_started)*1000
                fallback = fallback_if_expired(
                    "color_verification",
                    len(stage1) + color_index + 1,
                )
                if fallback is not None:
                    return fallback
            head = verified_head
        fallback = fallback_if_expired("color_complete", len(stage1) + len(head))
        if fallback is not None:
            return fallback
        fusion_started = time.perf_counter()
        output = self._sort([*head, *tail]) if color_requested else reranked
        finalized: list[dict[str, Any]] = []
        for rank, item in enumerate(output, start=1):
            fallback = fallback_if_expired("rank_delta", len(stage1) + rank - 1)
            if fallback is not None:
                return fallback
            initial_rank = rank_by_object[id(item)]
            # The untouched tail shares the fallback snapshot. Never add rank
            # diagnostics to it until the complete operation has succeeded.
            item = dict(item)
            item["advanced_rank_delta"] = initial_rank - rank
            finalized.append(item)
        self._state.stage_timings_ms["fusion"] = (time.perf_counter()-fusion_started)*1000
        color_ms = (time.perf_counter() - color_started) * 1_000
        color_verified = sum(
            value.get("color_rerank_state") == "verified" for value in head
        )
        object_evidence_degraded = bool(
            object_states & {"archive_missing", "member_missing", "invalid"}
        )
        self._state.diagnostics = {
            "state": "degraded" if object_evidence_degraded else "ready",
            "stage_timings_ms": dict(self._state.stage_timings_ms),
            "reason_code": (
                "object_evidence_unavailable" if object_evidence_degraded else "query_completed"
            ),
            "precision_profile": "automatic",
            "spatial_candidates": min(len(stage1), self.spatial_depth),
            "spatial_verified": verified_spatial,
            "relation_candidates": (
                min(len(stage1), self.spatial_depth)
                if strict and decomposition.relative_constraints
                else 0
            ),
            "relation_verified": verified_relations,
            "negation_candidates": (
                min(len(stage1), self.spatial_depth)
                if strict and decomposition.negation_constraints
                else 0
            ),
            "negation_verified": verified_negations,
            "spatial_latency_ms": round(spatial_ms, 3),
            "relation_latency_ms": round(relation_elapsed_ms, 3),
            "geometry_latency_ms": round(geometry_elapsed_ms, 3),
            "negation_latency_ms": round(negation_elapsed_ms, 3),
            "color_candidates": min(len(stage1), self.color_depth),
            "color_verified": color_verified,
            "color_latency_ms": round(color_ms, 3),
            "prefetch_latency_ms": round(prefetch_ms, 3),
            "deadline_budget_ms": self.verification_budget_ms,
            "total_latency_ms": round((time.perf_counter() - started) * 1_000, 3),
            "object_cache": self.object_archive.diagnostics,
            "object_states": sorted(object_states),
            "color_cache": self.color_verifier.diagnostics,
        }
        return finalized


__all__ = ["AdvancedKISReranker", "HSVColorVerifier"]
