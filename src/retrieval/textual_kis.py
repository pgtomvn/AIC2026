"""Deterministic hybrid retrieval core for Textual Known Item Search."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from inspect import Parameter, signature
from typing import Any

import numpy as np

from src.embedding.text_chunks import encode_query_mean
from src.query.entity_extractor import fold_entity_text
from src.query.translator import QueryProcessing, QueryTextAdapter, QueryVariant
from src.retrieval.candidate_rescue import (
    CandidateRescueConfig,
    run_candidate_rescue,
)
from src.retrieval.fusion import (
    ChannelRanking,
    aggregate_temporal_windows,
    canonical_key,
    diversify,
    weighted_rrf,
)
from src.retrieval.ocr_quality import (
    OCRQualityConfig,
    OCRQualityStore,
    calibrate_ocr_results,
)
from src.retrieval.query_plan import QueryPlan, build_query_plan
from src.retrieval.video_local import VideoLocalConfig, score_video_local


@dataclass(frozen=True)
class DenseChannel:
    """A cross-modal encoder and vector index pair."""

    name: str
    encoder: Any
    store: Any
    weight: float = 1.0
    depth: int = 300
    require_complete: bool = False
    expected_count: int | None = None


@dataclass(frozen=True)
class LexicalChannel:
    """A Meilisearch-like lexical retrieval channel."""

    name: str
    store: Any
    weight: float = 0.8
    depth: int = 200


FrameResolver = Callable[[dict[str, Any]], dict[str, Any] | None]
RawRefiner = Callable[[QueryPlan, list[dict[str, Any]]], list[dict[str, Any]]]
FeedbackProvider = Callable[[str, Sequence[tuple[str, int]]], np.ndarray]
AdvancedReranker = Callable[[QueryProcessing, list[dict[str, Any]]], list[dict[str, Any]]]
CrossModalRescorer = Callable[..., list[dict[str, Any]]]


_AUTOMATIC_TIER1_DEPTH = 300
_EXACT_OCR_RRF_WEIGHT = 2.0
_EXACT_ASR_RRF_WEIGHT = 1.5
_FRAME_ZERO_DENSE_MIN_SCORE = 0.28
_METADATA_VIDEO_BOOST = 1.15
_VISUAL_DENSE_CHANNELS = frozenset({"visual", "organizer_clip", "accelerated_visual"})


class TextualKISPipeline:
    """Search dense and lexical channels, fuse ranks, and diversify moments.

    Construction and imports are side-effect free.  Model loading and service
    connections remain lazy in their respective adapters.

    ``clip_encoder``, ``vector_store`` and ``text_store`` are retained as a
    compatibility path for the original scaffold.  New code should pass named
    ``dense_channels`` and ``lexical_channels``.
    """

    def __init__(
        self,
        clip_encoder: Any = None,
        vector_store: Any = None,
        text_store: Any = None,
        reranker: Any = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 100,
        *,
        dense_channels: Sequence[DenseChannel] | None = None,
        lexical_channels: Sequence[LexicalChannel] | None = None,
        rrf_k: int = 60,
        frame_resolver: FrameResolver | None = None,
        raw_refiner: RawRefiner | None = None,
        feedback_provider: FeedbackProvider | None = None,
        temporal_windows_ms: Sequence[int] = (4_000, 8_000),
        min_separation_ms: int = 3_000,
        early_cutoff: int = 20,
        early_per_video: int = 2,
        late_per_video: int = 5,
        query_adapter: QueryTextAdapter | None = None,
        entity_lexical_multiplier: float = 3.0,
        entity_bonus_alpha: float = 0.25,
        advanced_reranker: AdvancedReranker | None = None,
        cross_modal_rescorer: CrossModalRescorer | None = None,
        intent_policies: Mapping[str, Mapping[str, float | int]] | None = None,
        semantic_view_fusion: str = "weighted_rrf",
        semantic_view_weights: Mapping[str, float] | None = None,
        semantic_consensus_alpha: float = 0.0,
        asr_enabled: bool = True,
        asr_max_weight: float = 0.7,
        asr_temporal_overlap_bonus: float = 0.3,
        asr_temporal_padding_ms: int = 1_000,
        candidate_rescue: Mapping[str, Any] | CandidateRescueConfig | None = None,
        video_local: Mapping[str, Any] | VideoLocalConfig | None = None,
        video_local_trajectories: Any = None,
        ocr_quality: Mapping[str, Any] | OCRQualityConfig | None = None,
        ocr_quality_store: OCRQualityStore | None = None,
    ) -> None:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        if min_separation_ms < 0:
            raise ValueError("min_separation_ms cannot be negative")
        if not 1 <= early_cutoff <= 100:
            raise ValueError("early_cutoff must be between 1 and 100")
        if early_per_video < 1 or late_per_video < early_per_video:
            raise ValueError("per-video quotas must be positive and non-decreasing")
        if entity_lexical_multiplier < 1.0:
            raise ValueError("entity_lexical_multiplier must be at least 1")
        if not 0.0 <= entity_bonus_alpha <= 1.0:
            raise ValueError("entity_bonus_alpha must be between 0 and 1")
        if semantic_view_fusion not in {
            "global_only", "weighted_rrf", "max_rrf", "rrf_consensus"
        }:
            raise ValueError("unsupported semantic view fusion policy")
        if not 0.0 <= semantic_consensus_alpha <= 1.0:
            raise ValueError("semantic consensus alpha must be between zero and one")
        if not 0.0 <= asr_max_weight <= 1.0:
            raise ValueError("ASR max weight must be between zero and one")
        if not 0.0 <= asr_temporal_overlap_bonus <= 1.0:
            raise ValueError("ASR temporal overlap bonus must be between zero and one")
        if asr_temporal_padding_ms < 0:
            raise ValueError("ASR temporal padding must be non-negative")
        self.clip_encoder = clip_encoder
        self.vector_store = vector_store
        self.text_store = text_store
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.frame_resolver = frame_resolver
        self.raw_refiner = raw_refiner
        self.feedback_provider = feedback_provider
        self.temporal_windows_ms = tuple(temporal_windows_ms)
        self.min_separation_ms = min_separation_ms
        self.early_cutoff = early_cutoff
        self.early_per_video = early_per_video
        self.late_per_video = late_per_video
        self.query_adapter = query_adapter or QueryTextAdapter()
        self.entity_lexical_multiplier = float(entity_lexical_multiplier)
        self.entity_bonus_alpha = float(entity_bonus_alpha)
        self.advanced_reranker = advanced_reranker
        self.cross_modal_rescorer = cross_modal_rescorer
        self.semantic_view_fusion = semantic_view_fusion
        self.semantic_view_weights = {
            str(name): float(weight)
            for name, weight in (semantic_view_weights or {}).items()
        }
        if any(
            not np.isfinite(weight) or weight < 0.0
            for weight in self.semantic_view_weights.values()
        ):
            raise ValueError("semantic view weights must be finite and non-negative")
        self.semantic_consensus_alpha = float(semantic_consensus_alpha)
        self.asr_enabled = bool(asr_enabled)
        self.asr_max_weight = float(asr_max_weight)
        self.asr_temporal_overlap_bonus = float(asr_temporal_overlap_bonus)
        self.asr_temporal_padding_ms = int(asr_temporal_padding_ms)
        self.candidate_rescue = (
            candidate_rescue
            if isinstance(candidate_rescue, CandidateRescueConfig)
            else CandidateRescueConfig.from_mapping(candidate_rescue)
        )
        self.video_local = (
            video_local
            if isinstance(video_local, VideoLocalConfig)
            else VideoLocalConfig.from_mapping(video_local)
        )
        self.video_local_trajectories = video_local_trajectories
        self.ocr_quality = (
            ocr_quality
            if isinstance(ocr_quality, OCRQualityConfig)
            else OCRQualityConfig.from_mapping(ocr_quality)
        )
        self.ocr_quality_store = ocr_quality_store
        default_intent_policies: dict[str, dict[str, float | int]] = {
            "visual_dense": {"rrf_k": rrf_k, "dense_share": 0.70, "text_share": 0.30},
            "lexical_dominant": {"rrf_k": min(rrf_k, 40), "dense_share": 0.25, "text_share": 0.75},
            "spatial_color": {"rrf_k": rrf_k, "dense_share": 0.70, "text_share": 0.30},
        }
        for name, raw in (intent_policies or {}).items():
            policy = {
                "rrf_k": int(raw.get("rrf_k", rrf_k)),
                "dense_share": float(raw.get("dense_share", 0.70)),
                "text_share": float(raw.get("text_share", 0.30)),
            }
            if policy["rrf_k"] < 0 or any(
                not 0.0 <= float(policy[key]) <= 1.0
                for key in ("dense_share", "text_share")
            ) or not np.isclose(
                float(policy["dense_share"]) + float(policy["text_share"]), 1.0
            ):
                raise ValueError("intent RRF policies require non-negative k and shares summing to 1")
            default_intent_policies[str(name)] = policy
        self.intent_policies = default_intent_policies

        configured_dense = list(dense_channels or ())
        configured_lexical = list(lexical_channels or ())
        if clip_encoder is not None and vector_store is not None and not configured_dense:
            configured_dense.append(
                DenseChannel("visual", clip_encoder, vector_store, vector_weight, 300)
            )
        if text_store is not None and not configured_lexical:
            configured_lexical.append(LexicalChannel("lexical", text_store, bm25_weight, 200))
        self.dense_channels = tuple(configured_dense)
        self.lexical_channels = tuple(configured_lexical)
        self._execution_state = threading.local()
        self.last_channel_errors = {}
        self.last_channel_statuses = {}
        self.last_query_processing = {}
        self._execution_state.rank_trace = {}

    @property
    def last_channel_errors(self) -> dict[str, str]:
        return getattr(self._execution_state, "channel_errors", {})

    @last_channel_errors.setter
    def last_channel_errors(self, value: Mapping[str, Any]) -> None:
        self._execution_state.channel_errors = dict(value)

    @property
    def last_channel_statuses(self) -> dict[str, dict[str, Any]]:
        return getattr(self._execution_state, "channel_statuses", {})

    @last_channel_statuses.setter
    def last_channel_statuses(self, value: Mapping[str, Any]) -> None:
        self._execution_state.channel_statuses = {
            str(name): dict(status) for name, status in value.items()
        }

    @property
    def last_query_processing(self) -> dict[str, Any]:
        return getattr(self._execution_state, "query_processing", {})

    @last_query_processing.setter
    def last_query_processing(self, value: Mapping[str, Any]) -> None:
        self._execution_state.query_processing = dict(value)

    @property
    def last_rank_trace(self) -> dict[str, list[dict[str, Any]]]:
        """Return the bounded per-stage trace for offline diagnosis."""

        return {
            str(stage): [dict(item) for item in values]
            for stage, values in getattr(self._execution_state, "rank_trace", {}).items()
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable, secret-free retrieval configuration fingerprint."""

        def channel_value(channel: DenseChannel | LexicalChannel) -> dict[str, Any]:
            store = channel.store
            encoder = getattr(channel, "encoder", None)
            return {
                "name": channel.name,
                "weight": channel.weight,
                "depth": channel.depth,
                "collection": getattr(
                    store, "collection_template", getattr(store, "collection_name", None)
                ),
                "index": getattr(store, "index_name", None),
                "dimension": getattr(store, "vector_size", None),
                "encoder": getattr(encoder, "fingerprint", None),
                "text_max_chunks": getattr(encoder, "text_max_chunks", None),
                "text_chunk_overlap": getattr(encoder, "text_chunk_overlap", None),
                "require_complete": getattr(channel, "require_complete", False),
                "expected_count": getattr(channel, "expected_count", None),
            }

        payload = {
            "version": "textual-kis-v9-semantic-late-fusion",
            "rrf_k": self.rrf_k,
            "top_k": self.top_k,
            "temporal_windows_ms": self.temporal_windows_ms,
            "diversity": {
                "min_separation_ms": self.min_separation_ms,
                "early_cutoff": self.early_cutoff,
                "early_per_video": self.early_per_video,
                "late_per_video": self.late_per_video,
            },
            "dense": [channel_value(channel) for channel in self.dense_channels],
            "lexical": [channel_value(channel) for channel in self.lexical_channels],
            "query_adapter": self.query_adapter.fingerprint,
            "entity_policy": {
                "lexical_multiplier": self.entity_lexical_multiplier,
                "bonus_alpha": self.entity_bonus_alpha,
                "required_scope": "video_all_lexical",
            },
            "semantic_views": {
                "fusion": self.semantic_view_fusion,
                "weights": dict(sorted(self.semantic_view_weights.items())),
                "consensus_alpha": self.semantic_consensus_alpha,
            },
            "asr_routing": {
                "enabled": self.asr_enabled,
                "max_weight": self.asr_max_weight,
                "temporal_overlap_bonus": self.asr_temporal_overlap_bonus,
                "temporal_padding_ms": self.asr_temporal_padding_ms,
            },
            "candidate_rescue": {
                "mode": self.candidate_rescue.mode,
                "sample_size": self.candidate_rescue.sample_size,
                "min_unique_videos": self.candidate_rescue.min_unique_videos,
                "max_top_video_share": self.candidate_rescue.max_top_video_share,
                "max_video_hhi": self.candidate_rescue.max_video_hhi,
                "expanded_depth": self.candidate_rescue.expanded_depth,
                "max_depth": self.candidate_rescue.max_depth,
                "grouped_videos": self.candidate_rescue.grouped_videos,
                "frames_per_video": self.candidate_rescue.frames_per_video,
                "overfetch_weight": self.candidate_rescue.overfetch_weight,
                "grouped_weight": self.candidate_rescue.grouped_weight,
            },
            "video_local": {
                "mode": self.video_local.mode,
                "source_channel": self.video_local.source_channel,
                "candidate_videos": self.video_local.candidate_videos,
                "frames_per_video": self.video_local.frames_per_video,
                "weight": self.video_local.weight,
                "support_window_ms": self.video_local.support_window_ms,
                "gaussian_sigma_ms": self.video_local.gaussian_sigma_ms,
            },
            "ocr_quality": {
                "mode": self.ocr_quality.mode,
                "feature_version": "ocr-quality-v1",
                "min_multiplier": self.ocr_quality.min_multiplier,
                "max_multiplier": self.ocr_quality.max_multiplier,
                "candidate_limit": self.ocr_quality.candidate_limit,
            },
            "adaptive_rrf": self.intent_policies,
            "advanced_reranker": getattr(self.advanced_reranker, "fingerprint", None),
            "cross_modal_rescorer": getattr(self.cross_modal_rescorer, "fingerprint", None),
            "routing": {
                "mode": "automatic",
                "tier1_constraint_depth": _AUTOMATIC_TIER1_DEPTH,
                "metadata_video_boost": _METADATA_VIDEO_BOOST,
                "frame_zero_dense_min_score": _FRAME_ZERO_DENSE_MIN_SCORE,
                "exact_ocr_rrf_weight": _EXACT_OCR_RRF_WEIGHT,
                "legacy_precision_profiles_accepted": (
                    "standard", "advanced", "experimental_cross_modal"
                ),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _failure_status(error: Exception) -> dict[str, str]:
        """Return a stable, non-sensitive status for an optional channel."""

        if getattr(error, "restart_required", False):
            return {"state": "degraded", "reason_code": "cuda_restart_required"}
        message = str(error).casefold()
        if isinstance(error, (ImportError, ModuleNotFoundError)) or any(
            token in message
            for token in ("optional; install", "support is unavailable", "no module named")
        ):
            return {
                "state": "blocked_dependency",
                "reason_code": "dependency_missing",
            }
        if any(token in message for token in ("connection", "connect", "timeout", "unavailable")):
            return {
                "state": "service_unavailable",
                "reason_code": "service_unavailable",
            }
        return {"state": "service_unavailable", "reason_code": "channel_query_failed"}

    @staticmethod
    def _query_text(request: str | Mapping[str, Any] | Any) -> str:
        if isinstance(request, str):
            return request
        if isinstance(request, Mapping):
            value = request.get("original_text", request.get("text"))
        else:
            value = getattr(request, "original_text", getattr(request, "text", None))
        if not isinstance(value, str):
            raise ValueError("request must include original_text")
        return value

    @classmethod
    def _plan(cls, request: str | Mapping[str, Any] | Any) -> QueryPlan:
        plan = build_query_plan(cls._query_text(request))
        if isinstance(request, str):
            return plan

        def value(mapping_name: str, attribute_name: str) -> str | None:
            if isinstance(request, Mapping):
                raw = request.get(mapping_name)
            else:
                raw = getattr(request, attribute_name, None)
            return raw if isinstance(raw, str) and raw.strip() else None

        before = value("before_clause", "before_clause")
        current = value("current_clause", "current_clause")
        after = value("after_clause", "after_clause")
        updates: dict[str, Any] = {
            "before": before if before is not None else plan.before,
            "current": current if current is not None else plan.current,
            "after": after if after is not None else plan.after,
        }
        for field_name in ("quoted_text", "named_entities", "objects", "actions", "attributes"):
            raw = (
                request.get(field_name)
                if isinstance(request, Mapping)
                else getattr(request, field_name, None)
            )
            if raw is not None:
                if isinstance(raw, str):
                    raw = (raw,)
                updates[field_name] = tuple(str(value) for value in raw)
        return replace(plan, **updates)

    @staticmethod
    def _batch_filters(request: str | Mapping[str, Any] | Any) -> tuple[dict | None, str | None]:
        if isinstance(request, str):
            return None, None
        filters = (
            request.get("batch_filters")
            if isinstance(request, Mapping)
            else getattr(request, "batch_filters", None)
        )
        if not filters:
            return None, None
        values = [str(value) for value in filters]
        vector_filter = {"batch_id": values}
        escaped = [value.replace("'", "\\'") for value in values]
        lexical_filter = " OR ".join(f"batch_id = '{value}'" for value in escaped)
        return vector_filter, lexical_filter

    def _resolve_results(self, results: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        prefetch = getattr(self.frame_resolver, "prefetch", None)
        if callable(prefetch):
            pending: list[dict[str, Any]] = []
            for result in results:
                source = self._result_source(result)
                kind = str(source.get("kind", "")).casefold()
                segment = str(source.get("segment_id", result.get("id", ""))).casefold()
                if kind in {"ocr", "asr"} or segment.startswith(("ocr:", "asr:")):
                    pending.append(result)
                else:
                    try:
                        canonical_key(result)
                    except ValueError:
                        pending.append(result)
            prefetch(pending)
        resolved: list[dict[str, Any]] = []
        for result in results:
            source = self._result_source(result)
            kind = str(source.get("kind", "")).casefold()
            segment_id = str(source.get("segment_id", result.get("id", ""))).casefold()
            is_text_hit = kind in {"ocr", "asr"} or segment_id.startswith(("ocr:", "asr:"))
            if is_text_hit and self.frame_resolver is not None:
                mapped = self.frame_resolver(result)
                if mapped is not None:
                    resolved.append(mapped)
                    continue
            try:
                canonical_key(result)
                resolved.append(result)
                continue
            except ValueError:
                pass
            if self.frame_resolver is not None:
                mapped = self.frame_resolver(result)
                if mapped is not None:
                    resolved.append(mapped)
        return tuple(resolved)

    @staticmethod
    def _result_source(result: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = result.get("payload")
        return payload if isinstance(payload, Mapping) else result

    @classmethod
    def _result_video_id(cls, result: Mapping[str, Any]) -> str | None:
        value = cls._result_source(result).get("video_id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _verification_requested(processing: QueryProcessing) -> bool:
        decomposition = processing.decomposition
        return bool(decomposition and decomposition.bound_constraint_count)

    def _record_dense_frame_zero_permissions(
        self,
        channel_name: str,
        results: Sequence[dict[str, Any]],
    ) -> None:
        if channel_name.casefold() not in _VISUAL_DENSE_CHANNELS:
            return
        allowed: set[tuple[str, int]] = getattr(
            self._execution_state, "allowed_dense_frame_zero", set()
        )
        for result in results:
            try:
                raw_score = float(result.get("score", float("-inf")))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(raw_score) or raw_score < _FRAME_ZERO_DENSE_MIN_SCORE:
                continue
            for resolved in self._resolve_results((result,)):
                try:
                    key = canonical_key(resolved)
                except ValueError:
                    continue
                if key[1] == 0:
                    allowed.add(key)
        self._execution_state.allowed_dense_frame_zero = allowed

    def _filter_frame_zero_rankings(
        self,
        rankings: Sequence[ChannelRanking],
    ) -> list[ChannelRanking]:
        allowed: set[tuple[str, int]] = getattr(
            self._execution_state, "allowed_dense_frame_zero", set()
        )
        filtered: list[ChannelRanking] = []
        for ranking in rankings:
            results: list[dict[str, Any]] = []
            for result in ranking.results:
                try:
                    key = canonical_key(result)
                except ValueError:
                    continue
                if key[1] != 0 or key in allowed:
                    results.append(result)
            filtered.append(replace(ranking, results=tuple(results)))
        return filtered

    def _apply_metadata_video_boost(
        self,
        fused: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matched_videos: set[str] = getattr(
            self._execution_state, "metadata_boost_videos", set()
        )
        if not matched_videos:
            return list(fused)
        boosted: list[dict[str, Any]] = []
        for result in fused:
            item = dict(result)
            if self._result_video_id(item) in matched_videos:
                item["score"] = float(item.get("score", 0.0)) * _METADATA_VIDEO_BOOST
                item["metadata_video_boost"] = _METADATA_VIDEO_BOOST
                evidence = list(item.get("evidence", ()))
                evidence.append(
                    {"channel": "metadata_video", "boost_factor": _METADATA_VIDEO_BOOST}
                )
                item["evidence"] = evidence
            boosted.append(item)
        boosted.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", 0)),
            )
        )
        return boosted

    def _rerank_temporal_evidence(
        self, fused: Sequence[dict[str, Any]], plan: QueryPlan
    ) -> list[dict[str, Any]]:
        """Apply bounded rank-based late interaction for temporal clauses."""

        required = tuple(
            label
            for label, clause in (
                ("before", plan.before),
                ("current", plan.current),
                ("after", plan.after),
            )
            if isinstance(clause, str) and clause.strip()
        )
        if len(required) < 2:
            return list(fused)
        reranked: list[dict[str, Any]] = []
        covered_candidates = 0
        ordered_candidates = 0
        for raw in fused:
            item = dict(raw)
            best: dict[str, tuple[int, int]] = {}
            for evidence in item.get("evidence", ()):
                if not isinstance(evidence, Mapping):
                    continue
                timestamp = int(
                    evidence.get("timestamp_ms", item.get("timestamp_ms", 0)) or 0
                )
                names: list[tuple[str, int]] = []
                channel = evidence.get("channel")
                if isinstance(channel, str):
                    names.append((channel, int(evidence.get("rank", 1))))
                nested = evidence.get("nested_channel_ranks")
                if isinstance(nested, Mapping):
                    names.extend((str(name), int(rank)) for name, rank in nested.items())
                for label in required:
                    matches = [
                        (rank, timestamp)
                        for name, rank in names
                        if f"temporal_{label}" in name
                    ]
                    if matches:
                        candidate = min(matches)
                        if label not in best or candidate < best[label]:
                            best[label] = candidate
            coverage = len(best) / len(required)
            ordered = False
            if len(best) == len(required):
                timestamps = [best[label][1] for label in required]
                ordered = timestamps == sorted(timestamps)
            boost = 1.0 + 0.08 * coverage + (0.04 if ordered else 0.0)
            if best:
                covered_candidates += 1
                item["temporal_clause_coverage"] = round(coverage, 6)
                item["temporal_clause_timestamps"] = {
                    label: best[label][1] for label in required if label in best
                }
            if ordered:
                ordered_candidates += 1
                item["temporal_order_verified"] = True
            item["score"] = float(item.get("score", 0.0)) * boost
            reranked.append(item)
        reranked.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", 0)),
            )
        )
        processing = dict(self.last_query_processing)
        processing["temporal_evidence"] = {
            "required_clauses": list(required),
            "covered_candidates": covered_candidates,
            "ordered_candidates": ordered_candidates,
            "max_boost": 1.12,
        }
        self.last_query_processing = processing
        return reranked

    @staticmethod
    def _normalized_vector(value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim == 2:
            if vector.shape[0] != 1:
                raise ValueError("a query encoder must return exactly one vector")
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError("query embedding must be one-dimensional")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("query embedding must have a finite non-zero norm")
        return vector / norm

    @staticmethod
    def _payload_text_values(result: Mapping[str, Any]) -> tuple[str, ...]:
        payload = result.get("payload")
        source = payload if isinstance(payload, Mapping) else result
        values: list[str] = []
        for field in (
            "text_raw",
            "text_nfc",
            "text_accent_folded",
            "text",
            "ocr_text",
            "asr_text",
            "title",
            "description",
            "objects",
        ):
            value = source.get(field)
            if isinstance(value, str) and value:
                values.append(value)
            elif isinstance(value, (list, tuple)):
                values.extend(item for item in value if isinstance(item, str) and item)
        return tuple(values)

    @staticmethod
    def _phrase_matches(text: str, term: str) -> bool:
        def fold(value: str) -> str:
            normalized = unicodedata.normalize("NFD", value)
            without_marks = "".join(
                character
                for character in normalized
                if unicodedata.category(character) != "Mn"
            )
            return " ".join(
                without_marks.replace("đ", "d").replace("Đ", "D").casefold().split()
            )

        haystack = fold(text)
        needle = fold(term)
        return bool(
            needle
            and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack)
        )

    @staticmethod
    def _exact_ocr_query(term: str) -> str:
        """Build a quoted Meilisearch seed while preserving full client verification."""

        words = term.strip().split()
        seed = " ".join(words[:10])
        escaped = seed.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @classmethod
    def _asr_variants(
        cls, variants: Sequence[QueryVariant], plan: QueryPlan
    ) -> list[QueryVariant]:
        """Prefer a speech-focused query while retaining translated fallbacks."""

        output = list(variants)
        query = " ".join(plan.asr_query.split())
        if not query:
            return output
        identities = {" ".join(value.text.split()).casefold() for value in output}
        if query.casefold() not in identities:
            output.insert(0, QueryVariant("asr_query", query, "vie"))
        return output

    def _rerank_asr_coverage(
        self,
        results: Sequence[dict[str, Any]],
        plan: QueryPlan,
    ) -> list[dict[str, Any]]:
        """Apply a bounded phrase/token-coverage lift to long ASR segments."""

        terms = tuple(plan.asr_terms or plan.quoted_text)
        if not terms:
            terms = tuple(token for token in plan.tokens if len(token) >= 3)[:12]
        if not terms:
            return list(results)
        reranked: list[dict[str, Any]] = []
        matched = 0
        for raw in results:
            item = dict(raw)
            texts = self._payload_text_values(item)
            coverage = sum(
                1 for term in terms if any(self._phrase_matches(text, term) for text in texts)
            ) / len(terms)
            if coverage > 0.0:
                matched += 1
                item["asr_term_coverage"] = round(coverage, 6)
                item["score"] = float(item.get("score", 0.0)) * (1.0 + 0.25 * coverage)
            reranked.append(item)
        reranked.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str((item.get("payload") or item).get("video_id", "")),
                str(item.get("id", "")),
            )
        )
        processing = dict(self.last_query_processing)
        processing["asr_coverage_rerank"] = {
            "terms": list(terms),
            "matched_results": matched,
            "max_multiplier": 1.25,
        }
        self.last_query_processing = processing
        return reranked

    def _apply_asr_temporal_bonus(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        enabled: bool,
    ) -> list[dict[str, Any]]:
        """Lift only frames overlapping their best ASR segment; never penalize silence."""

        if not enabled or self.asr_temporal_overlap_bonus <= 0.0:
            return list(candidates)
        output: list[dict[str, Any]] = []
        boosted = 0
        for raw in candidates:
            item = dict(raw)
            timestamp_ms = int(item.get("timestamp_ms", 0) or 0)
            best = 0.0
            best_segment: str | None = None
            for evidence in item.get("evidence", ()):
                if not isinstance(evidence, Mapping):
                    continue
                channel = str(evidence.get("channel", "")).casefold()
                if "asr" not in channel and evidence.get("kind") != "asr":
                    continue
                try:
                    start_ms = int(evidence["start_ms"])
                    end_ms = int(evidence["end_ms"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    start_ms - self.asr_temporal_padding_ms
                    <= timestamp_ms
                    <= end_ms + self.asr_temporal_padding_ms
                ):
                    confidence = evidence.get("confidence", 1.0)
                    confidence_value = (
                        float(confidence)
                        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                        else 1.0
                    )
                    candidate = max(0.0, min(1.0, confidence_value))
                    if candidate > best:
                        best = candidate
                        best_segment = str(evidence.get("segment_id", "")) or None
            if best > 0.0:
                bonus = self.asr_temporal_overlap_bonus * best
                item["score"] = float(item.get("score", 0.0)) * (1.0 + bonus)
                item["asr_temporal_bonus"] = round(bonus, 6)
                if best_segment is not None:
                    item["asr_segment_id"] = best_segment
                boosted += 1
            output.append(item)
        output.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", 0)),
            )
        )
        processing = dict(self.last_query_processing)
        processing["asr_temporal_fusion"] = {
            "boosted_candidates": boosted,
            "max_bonus": self.asr_temporal_overlap_bonus,
            "padding_ms": self.asr_temporal_padding_ms,
            "absence_penalized": False,
        }
        self.last_query_processing = processing
        return output

    @staticmethod
    def _with_temporal_variants(
        variants: Sequence[QueryVariant], plan: QueryPlan
    ) -> list[QueryVariant]:
        """Add distinct clause variants only for an explicitly temporal query."""

        before = plan.before.strip() if isinstance(plan.before, str) else ""
        after = plan.after.strip() if isinstance(plan.after, str) else ""
        output = list(variants)
        if not before and not after:
            return output
        seen = {" ".join(value.text.split()).casefold() for value in output}
        for label, raw_clause in (
            ("before", plan.before),
            ("current", plan.current),
            ("after", plan.after),
        ):
            if not isinstance(raw_clause, str):
                continue
            clause = " ".join(raw_clause.split())
            identity = clause.casefold()
            if not clause or identity in seen:
                continue
            seen.add(identity)
            output.append(QueryVariant(f"temporal_{label}", clause))
        return output

    def _annotate_entity_matches(
        self,
        results: Sequence[dict[str, Any]],
        processing: QueryProcessing,
        channel_name: str,
    ) -> list[dict[str, Any]]:
        decomposition = processing.decomposition
        if decomposition is None or not decomposition.named_entities:
            return list(results)
        video_entities: dict[str, dict[str, dict[str, Any]]] = getattr(
            self._execution_state, "video_entities", {}
        )
        entity_videos: dict[str, set[str]] = getattr(
            self._execution_state, "entity_videos", {}
        )
        annotated: list[dict[str, Any]] = []
        for raw in results:
            result = dict(raw)
            texts = self._payload_text_values(result)
            matches: list[dict[str, Any]] = []
            payload = result.get("payload")
            source = payload if isinstance(payload, Mapping) else result
            video_id = source.get("video_id")
            for entity in decomposition.named_entities:
                if any(
                    self._phrase_matches(text, term)
                    for text in texts
                    for term in entity.lexical_terms
                ):
                    match = {
                        "text": entity.text,
                        "identity": fold_entity_text(entity.text),
                        "category": entity.category,
                        "mode": entity.mode,
                        "channel": channel_name,
                    }
                    matches.append(match)
                    if isinstance(video_id, str) and video_id:
                        entity_videos.setdefault(match["identity"], set()).add(video_id)
                        video_entities.setdefault(video_id, {}).setdefault(
                            match["identity"], match
                        )
            if matches:
                result["entity_matches"] = matches
                if isinstance(result.get("payload"), Mapping):
                    result["payload"] = {**dict(result["payload"]), "entity_matches": matches}
            annotated.append(result)
        self._execution_state.video_entities = video_entities
        self._execution_state.entity_videos = entity_videos
        return annotated

    def _annotate_exact_text_matches(
        self,
        results: Sequence[dict[str, Any]],
        terms: Sequence[str],
        channel_name: str,
    ) -> list[dict[str, Any]]:
        """Mark exact OCR/ASR phrase evidence without exposing backend payloads."""

        if not terms or not any(token in channel_name.casefold() for token in ("ocr", "asr")):
            return list(results)
        exact_videos: set[str] = getattr(self._execution_state, "exact_ocr_videos", set())
        annotated: list[dict[str, Any]] = []
        for raw in results:
            result = dict(raw)
            texts = self._payload_text_values(result)
            matched = tuple(
                term for term in terms if any(self._phrase_matches(text, term) for text in texts)
            )
            if matched:
                payload = result.get("payload")
                source = payload if isinstance(payload, Mapping) else result
                video_id = source.get("video_id")
                if isinstance(video_id, str) and video_id:
                    exact_videos.add(video_id)
                result["exact_text_matches"] = list(matched)
                if isinstance(payload, Mapping):
                    result["payload"] = {**dict(payload), "exact_text_matches": list(matched)}
            annotated.append(result)
        self._execution_state.exact_ocr_videos = exact_videos
        return annotated

    def _apply_entity_policy(
        self,
        fused: Sequence[dict[str, Any]],
        rankings: Sequence[ChannelRanking],
        processing: QueryProcessing,
        *,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        decomposition = processing.decomposition
        if decomposition is None:
            return list(fused)
        exact_ocr_videos: set[str] = getattr(
            self._execution_state, "exact_ocr_videos", set()
        )
        if not decomposition.named_entities and not exact_ocr_videos:
            return list(fused)
        entity_videos: dict[str, set[str]] = getattr(
            self._execution_state, "entity_videos", {}
        )
        video_entities: dict[str, dict[str, dict[str, Any]]] = getattr(
            self._execution_state, "video_entities", {}
        )
        required = tuple(
            fold_entity_text(entity.text)
            for entity in decomposition.required_entities
            if not entity.source.startswith("operator_")
        )
        lexical_ready = bool(getattr(self._execution_state, "entity_lexical_ready", False))
        eligible: set[str] | None = None
        if required:
            if lexical_ready:
                sets = [set(entity_videos.get(identity, set())) for identity in required]
                eligible = set.intersection(*sets) if sets else set()
                reason = "required_entity_match" if eligible else "required_entity_no_match"
            else:
                eligible = set()
                reason = "required_entity_evidence_unavailable"
            processing_value = dict(self.last_query_processing)
            processing_value["required_entity_status"] = reason
            self.last_query_processing = processing_value

        optional_identities = {
            fold_entity_text(entity.text)
            for entity in decomposition.named_entities
            if entity.mode == "optional" and not entity.source.startswith("operator_")
        }
        rrf_ceiling = sum(max(0.0, ranking.weight) / (rrf_k + 1) for ranking in rankings)
        bonus = (
            self.entity_bonus_alpha * rrf_ceiling
            if optional_identities or exact_ocr_videos
            else 0.0
        )
        output: list[dict[str, Any]] = []
        for raw in fused:
            video_id = str(raw.get("video_id", ""))
            if eligible is not None and video_id not in eligible:
                continue
            matches = video_entities.get(video_id, {})
            item = dict(raw)
            if matches:
                item["entity_matches"] = list(matches.values())
                evidence = list(item.get("evidence", ()))
                evidence.append(
                    {
                        "channel": "entity_video",
                        "entity_matches": list(matches.values()),
                    }
                )
                item["evidence"] = evidence
            exact_match = video_id in exact_ocr_videos
            if optional_identities & set(matches) or exact_match:
                item["score"] = float(item.get("score", 0.0)) + bonus
                item["entity_bonus"] = bonus
                item["retrieval_bonus_reason"] = (
                    "entity_and_exact_text"
                    if optional_identities & set(matches) and exact_match
                    else "exact_text" if exact_match else "entity"
                )
                if exact_match:
                    evidence = list(item.get("evidence", ()))
                    evidence.append({"channel": "exact_text", "matched": True})
                    item["evidence"] = evidence
            output.append(item)
        output.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", 0)),
            )
        )
        processing_value = dict(self.last_query_processing)
        processing_value["matched_entity_count"] = sum(
            1
            for entity in decomposition.named_entities
            if not entity.source.startswith("operator_")
            and entity_videos.get(fold_entity_text(entity.text))
        )
        processing_value["entity_bonus_normalized"] = round(bonus, 9)
        entities = processing_value.get("entities")
        if isinstance(entities, list):
            processing_value["entities"] = [
                {
                    **dict(value),
                    "boost_applied": bool(
                        not str(value.get("source", "")).startswith("operator_")
                        and entity_videos.get(fold_entity_text(str(value.get("text", ""))))
                    ),
                }
                for value in entities
                if isinstance(value, Mapping)
            ]
        self.last_query_processing = processing_value
        return output

    def _intent_rrf_policy(self, processing: QueryProcessing) -> tuple[int, float, float]:
        intent = (
            processing.decomposition.intent
            if processing.decomposition is not None
            else "visual_dense"
        )
        policy = self.intent_policies.get(intent, self.intent_policies["visual_dense"])
        return (
            int(policy["rrf_k"]),
            float(policy["dense_share"]),
            float(policy["text_share"]),
        )

    def _apply_adaptive_weights(
        self,
        rankings: Sequence[ChannelRanking],
        processing: QueryProcessing,
        *,
        verification_requested: bool = False,
    ) -> tuple[list[ChannelRanking], int, dict[str, float]]:
        effective_k, dense_share, text_share = self._intent_rrf_policy(processing)
        dense_names = {channel.name for channel in self.dense_channels}
        if verification_requested:
            dense_names.discard("semantic_text")
        dense = [
            value for value in rankings if value.name in dense_names and value.results
        ]
        textual = [
            value for value in rankings if value.name not in dense_names and value.results
        ]
        if dense and not textual:
            dense_share, text_share = 1.0, 0.0
        elif textual and not dense:
            dense_share, text_share = 0.0, 1.0

        def group_weights(
            values: Sequence[ChannelRanking],
            share: float,
        ) -> dict[str, float]:
            total = sum(max(0.0, value.weight) for value in values)
            if not values or share <= 0.0:
                return {}
            if total <= 0.0:
                equal = share / len(values)
                return {value.name: equal for value in values}
            return {
                value.name: share * max(0.0, value.weight) / total
                for value in values
            }

        weights = {
            **group_weights(dense, dense_share),
            **group_weights(textual, text_share),
        }
        if "asr" in weights:
            weights["asr"] = min(weights["asr"], self.asr_max_weight)
        weighted = [
            replace(value, weight=weights.get(value.name, 0.0))
            for value in rankings
        ]
        return weighted, effective_k, weights

    def _semantic_variant_weight(self, variant: QueryVariant) -> float:
        role = variant.semantic_role
        configured = self.semantic_view_weights.get(role, 1.0) if role else 1.0
        return float(variant.prior) * configured

    def _fuse_variant_rankings(
        self,
        rankings: Sequence[ChannelRanking],
        *,
        k: int,
    ) -> list[dict[str, Any]]:
        """Fuse per-view rankings without averaging their query embeddings."""

        if not rankings:
            return []
        if len(rankings) == 1:
            return list(rankings[0].results)
        if self.semantic_view_fusion in {"weighted_rrf", "rrf_consensus"}:
            fused = weighted_rrf(rankings, k=k)
            if self.semantic_view_fusion == "rrf_consensus" and self.semantic_consensus_alpha:
                denominator = max(1, len(rankings) - 1)
                for item in fused:
                    consensus = max(0, len(item.get("channel_ranks", {})) - 1) / denominator
                    item["score"] = float(item.get("score", 0.0)) * (
                        1.0 + self.semantic_consensus_alpha * consensus
                    )
                    item["semantic_consensus"] = consensus
                fused.sort(
                    key=lambda item: (
                        -float(item.get("score", 0.0)),
                        str(item.get("video_id", "")),
                        int(item.get("source_frame_idx", 0)),
                    )
                )
            return fused

        # Max reciprocal-rank preserves a rare-detail match even when the
        # candidate is absent from generic/global views. Evidence from all
        # views is still retained for diagnostics and later calibration.
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for ranking in rankings:
            for item in weighted_rrf((ranking,), k=k):
                key = canonical_key(item)
                previous = by_key.get(key)
                if previous is None:
                    by_key[key] = dict(item)
                    continue
                previous["evidence"] = [
                    *previous.get("evidence", ()),
                    *item.get("evidence", ()),
                ]
                previous["channel_ranks"] = {
                    **previous.get("channel_ranks", {}),
                    **item.get("channel_ranks", {}),
                }
                previous["score"] = max(
                    float(previous.get("score", 0.0)),
                    float(item.get("score", 0.0)),
                )
        return sorted(
            by_key.values(),
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", 0)),
            ),
        )

    def _trace_stage(
        self,
        stage: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        trace = dict(getattr(self._execution_state, "rank_trace", {}))
        rows: list[dict[str, Any]] = []
        for rank, item in enumerate(candidates[:20], start=1):
            try:
                video_id, source_frame_idx = canonical_key(item)
            except ValueError:
                continue
            rows.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "source_frame_idx": source_frame_idx,
                    "score": round(float(item.get("score", 0.0)), 12),
                    "channel_ranks": dict(item.get("channel_ranks", {})),
                    "semantic_view_ranks": {
                        str(name): int(value)
                        for evidence in item.get("evidence", ())
                        if isinstance(evidence, Mapping)
                        for name, value in (
                            evidence.get("nested_channel_ranks", {}).items()
                            if isinstance(evidence.get("nested_channel_ranks"), Mapping)
                            else ()
                        )
                        if "_view_" in str(name)
                    },
                }
            )
        trace[stage] = rows
        self._execution_state.rank_trace = trace


    def _collect_rankings(
        self,
        plan: QueryPlan,
        request: str | Mapping[str, Any] | Any,
        *,
        query_vectors: Mapping[str, np.ndarray] | None = None,
        processing: QueryProcessing | None = None,
        effective_k: int | None = None,
        verification_requested: bool = False,
        force_semantic_views: bool = False,
    ) -> list[ChannelRanking]:
        vector_filter, lexical_filter = self._batch_filters(request)
        rankings: list[ChannelRanking] = []
        self.last_channel_errors = {}
        self.last_channel_statuses = {}
        self._execution_state.entity_videos = {}
        self._execution_state.video_entities = {}
        self._execution_state.entity_lexical_ready = False
        self._execution_state.exact_ocr_videos = set()
        self._execution_state.metadata_boost_videos = set()
        self._execution_state.allowed_dense_frame_zero = set()
        self._execution_state.tier1_query_vectors = {}
        self._execution_state.tier1_query_views = {}
        self._execution_state.semantic_view_rankings = {}
        self._execution_state.semantic_view_results = {}
        tier1_depth = _AUTOMATIC_TIER1_DEPTH if verification_requested else None
        processing = processing or self._prepare_query(request, plan)
        effective_k = self.rrf_k if effective_k is None else effective_k
        request_mode = str(self._request_value(request, "mode") or "automatic")
        asr_requested = self.asr_enabled and request_mode != "visual_only" and (
            plan.asr_intent or request_mode == "asr_only"
        )
        processing_value = dict(self.last_query_processing)
        processing_value["asr_gate"] = {
            "enabled": asr_requested,
            "forced": request_mode == "asr_only",
            "suppressed": request_mode == "visual_only",
            "intent_detected": plan.asr_intent,
            "query": plan.asr_query if asr_requested else "",
            "query_text_source": (
                "original_vi"
                if processing.translation.detected_language == "vi"
                else "original_query"
            ),
            "terms": list(plan.asr_terms),
            "max_weight": self.asr_max_weight,
        }
        self.last_query_processing = processing_value
        for channel in self.dense_channels:
            channel_started = time.monotonic()
            if request_mode == "asr_only":
                self.last_channel_statuses[channel.name] = {
                    "state": "skipped",
                    "reason_code": "asr_only_mode",
                }
                continue
            try:
                is_available = getattr(channel.store, "is_available", None)
                if callable(is_available) and not is_available(vector_filter):
                    self.last_channel_statuses[channel.name] = {
                        "state": "not_provisioned",
                        "reason_code": "no_compatible_points",
                    }
                    continue
                if channel.require_complete:
                    count_method = getattr(channel.store, "count", None)
                    if not callable(count_method) or channel.expected_count is None:
                        self.last_channel_statuses[channel.name] = {
                            "state": "not_provisioned",
                            "reason_code": "coverage_contract_unavailable",
                        }
                        continue
                    point_count = int(count_method(vector_filter))
                    if point_count != channel.expected_count:
                        self.last_channel_statuses[channel.name] = {
                            "state": "not_provisioned",
                            "reason_code": "collection_coverage_incomplete",
                            "count": point_count,
                            "expected_count": channel.expected_count,
                            "coverage": (
                                point_count / channel.expected_count
                                if channel.expected_count > 0
                                else 0.0
                            ),
                        }
                        continue
                diagnostics: dict[str, Any] = {}
                if query_vectors and channel.name in query_vectors:
                    vectors = np.asarray(query_vectors[channel.name], dtype=np.float32)
                    variant_vectors = (("feedback", vectors),)
                else:
                    variants = list(
                        self.query_adapter.dense_variants(channel.name, processing)
                    )
                    if self.semantic_view_fusion == "global_only" and not force_semantic_views:
                        global_variants = [
                            value for value in variants if value.semantic_role in {None, "global"}
                        ]
                        variants = global_variants[:1]
                    if channel.name == "visual":
                        variants = self._with_temporal_variants(variants, plan)
                    if not variants:
                        self.last_channel_statuses[channel.name] = {
                            "state": "degraded",
                            "reason_code": "translation_unavailable",
                            "translation_applied": False,
                        }
                        continue
                    encoded_variants: list[tuple[str, np.ndarray]] = []
                    variant_diagnostics: list[dict[str, Any]] = []
                    for variant in variants:
                        diagnostic_encoder = getattr(
                            channel.encoder,
                            "encode_text_with_diagnostics",
                            None,
                        )
                        if callable(diagnostic_encoder):
                            encoded, diagnostic_value = diagnostic_encoder(variant.text)
                            to_dict = getattr(diagnostic_value, "to_dict", None)
                            if callable(to_dict):
                                variant_diagnostics.append(dict(to_dict()))
                        else:
                            encoded = channel.encoder.encode_text(variant.text)
                        encoded_variants.append(
                            (variant.name, np.asarray(encoded, dtype=np.float32))
                        )
                    variant_vectors = tuple(encoded_variants)
                    if variant_vectors:
                        rows: list[np.ndarray] = []
                        for _variant_name, value in variant_vectors:
                            matrix = np.asarray(value, dtype=np.float32)
                            rows.extend(matrix if matrix.ndim == 2 else matrix[None, :])
                        mean_vector = self._normalized_vector(np.stack(rows).mean(axis=0))
                        tier1_vectors = dict(self._execution_state.tier1_query_vectors)
                        tier1_vectors[channel.name] = mean_vector
                        self._execution_state.tier1_query_vectors = tier1_vectors
                        view_vectors = dict(self._execution_state.tier1_query_views)
                        view_vectors[channel.name] = tuple(
                            self._normalized_vector(row) for row in rows
                        )
                        self._execution_state.tier1_query_views = view_vectors
                    if variant_diagnostics:
                        diagnostics = dict(variant_diagnostics[0])
                        diagnostics["query_variant_count"] = len(variants)
                        diagnostics["translation_applied"] = (
                            processing.translation.translation_applied
                        )
                chunk_rankings: list[ChannelRanking] = []
                rankings_by_role: dict[str, list[ChannelRanking]] = {}
                role_variants: dict[str, QueryVariant] = {}
                for variant_name, vectors in variant_vectors:
                    if vectors.ndim == 1:
                        vectors = vectors[None, :]
                    if vectors.ndim != 2 or vectors.shape[0] < 1:
                        raise ValueError("query encoder must return a non-empty vector matrix")
                    for index, vector in enumerate(vectors):
                        results = channel.store.search(
                            self._normalized_vector(vector),
                            top_k=max(channel.depth, tier1_depth or channel.depth),
                            filters=vector_filter,
                        )
                        self._record_dense_frame_zero_permissions(channel.name, results)
                        if not (query_vectors and channel.name in query_vectors):
                            variant = next(
                                value for value in variants if value.name == variant_name
                            )
                            if variant.semantic_role is not None:
                                trace = dict(self._execution_state.semantic_view_rankings)
                                trace[f"{channel.name}:{variant_name}:query_chunk_{index + 1}"] = [
                                    {
                                        "rank": rank,
                                        "video_id": key[0],
                                        "source_frame_idx": key[1],
                                        "raw_score": result.get(
                                            "score",
                                            result.get("_rankingScore"),
                                        ),
                                        "role": variant.semantic_role,
                                    }
                                    for rank, result in enumerate(results[:20], start=1)
                                    if (
                                        key := canonical_key(result)
                                    )
                                ]
                                self._execution_state.semantic_view_rankings = trace
                        ranking = ChannelRanking(
                            f"{channel.name}:{variant_name}:query_chunk_{index + 1}",
                            self._resolve_results(results),
                            1.0,
                        )
                        if query_vectors and channel.name in query_vectors:
                            chunk_rankings.append(ranking)
                        else:
                            variant = next(
                                value for value in variants if value.name == variant_name
                            )
                            role = variant.semantic_role or variant.name
                            rankings_by_role.setdefault(role, []).append(ranking)
                            role_variants.setdefault(role, variant)

                if not (query_vectors and channel.name in query_vectors):
                    for role, role_chunks in rankings_by_role.items():
                        role_results = (
                            list(role_chunks[0].results)
                            if len(role_chunks) == 1
                            else weighted_rrf(role_chunks, k=effective_k)
                        )
                        resolved_role_results = tuple(
                            self._resolve_results(role_results)
                        )
                        semantic_results = dict(
                            self._execution_state.semantic_view_results
                        )
                        semantic_results[f"{channel.name}:{role}"] = (
                            resolved_role_results
                        )
                        self._execution_state.semantic_view_results = semantic_results
                        chunk_rankings.append(
                            ChannelRanking(
                                f"{channel.name}:semantic_role:{role}",
                                resolved_role_results,
                                self._semantic_variant_weight(role_variants[role]),
                            )
                        )
                results = (
                    list(chunk_rankings[0].results)
                    if len(chunk_rankings) == 1
                    else self._fuse_variant_rankings(chunk_rankings, k=effective_k)
                )
                rankings.append(
                    ChannelRanking(
                        channel.name,
                        self._resolve_results(results),
                        channel.weight,
                    )
                )
                self.last_channel_statuses[channel.name] = {
                    "state": "ready",
                    "reason_code": "query_completed",
                    "result_count": len(results),
                    "semantic_view_roles": list(dict.fromkeys(
                        value.semantic_role
                        for value in (
                            variants
                            if not (query_vectors and channel.name in query_vectors)
                            else ()
                        )
                        if value.semantic_role is not None
                    )),
                    "latency_ms": round(
                        (time.monotonic() - channel_started) * 1_000,
                        3,
                    ),
                    **diagnostics,
                }
                processing_value = dict(self.last_query_processing)
                processing_value["semantic_view_rankings"] = dict(
                    self._execution_state.semantic_view_rankings
                )
                self.last_query_processing = processing_value
            except Exception as error:  # isolate optional model/service channels
                self.last_channel_errors[channel.name] = str(error)
                self.last_channel_statuses[channel.name] = self._failure_status(error)

        for channel in self.lexical_channels:
            channel_started = time.monotonic()
            asr_channel = "asr" in channel.name.casefold()
            if request_mode == "visual_only":
                self.last_channel_statuses[channel.name] = {
                    "state": "skipped",
                    "reason_code": "visual_only_mode",
                }
                continue
            if request_mode == "asr_only" and not asr_channel:
                self.last_channel_statuses[channel.name] = {
                    "state": "skipped",
                    "reason_code": "asr_only_mode",
                }
                continue
            if asr_channel and not asr_requested:
                self.last_channel_statuses[channel.name] = {
                    "state": "skipped",
                    "reason_code": "asr_intent_not_detected",
                }
                continue
            try:
                is_available = getattr(channel.store, "is_available", None)
                if callable(is_available) and not is_available(lexical_filter):
                    self.last_channel_statuses[channel.name] = {
                        "state": "not_provisioned",
                        "reason_code": "no_compatible_documents",
                    }
                    continue
                self._execution_state.entity_lexical_ready = True
                variants = list(
                    self.query_adapter.lexical_variants(channel.name, processing)
                )
                if asr_channel:
                    variants = self._asr_variants(variants, plan)
                if any(token in channel.name.casefold() for token in ("ocr", "asr")):
                    variants = self._with_temporal_variants(variants, plan)
                exact_phrase_by_variant: dict[str, str] = {}
                if "ocr" in channel.name.casefold() or asr_channel:
                    exact_phrases: list[str] = []
                    seen_phrases: set[str] = set()
                    for phrase in plan.quoted_text:
                        if not phrase or not phrase.strip() or phrase in seen_phrases:
                            continue
                        seen_phrases.add(phrase)
                        exact_phrases.append(phrase)
                        if len(exact_phrases) == 16:
                            break
                    for index, phrase in enumerate(exact_phrases, start=1):
                        prefix = "exact_asr" if asr_channel else "exact_ocr"
                        name = f"{prefix}_{index}"
                        exact_phrase_by_variant[name] = phrase
                        variants.append(
                            QueryVariant(name, self._exact_ocr_query(phrase))
                        )
                metadata_channel = "metadata" in channel.name.casefold()
                variant_rankings: list[ChannelRanking] = []
                variant_results = self._search_lexical_variants(
                    channel.store,
                    variants,
                    top_k=max(channel.depth, tier1_depth or channel.depth),
                    filters=lexical_filter,
                )
                for variant, initial_results in zip(variants, variant_results, strict=True):
                    results = initial_results
                    if (
                        not results
                        and variant.name == "original"
                        and processing.translation.detected_language == "vi"
                        and processing.translation.accent_folded_text != variant.text.casefold()
                    ):
                        results = self._search_lexical(
                            channel.store,
                            QueryVariant(
                                "accent_folded",
                                processing.translation.accent_folded_text,
                                "vie",
                            ),
                            top_k=max(channel.depth, tier1_depth or channel.depth),
                            filters=lexical_filter,
                        )
                    results = self._annotate_entity_matches(
                        results,
                        processing,
                        channel.name,
                    )
                    results = self._annotate_exact_text_matches(
                        results,
                        plan.asr_terms if asr_channel else plan.quoted_text,
                        channel.name,
                    )
                    exact_phrase = exact_phrase_by_variant.get(variant.name)
                    if exact_phrase is not None:
                        verified_results: list[dict[str, Any]] = []
                        for result in results:
                            payload = result.get("payload")
                            payload_matches = (
                                payload.get("exact_text_matches", ())
                                if isinstance(payload, Mapping)
                                else ()
                            )
                            if (
                                exact_phrase in result.get("exact_text_matches", ())
                                or exact_phrase in payload_matches
                            ):
                                verified_results.append(result)
                        results = verified_results
                    if metadata_channel:
                        matched_videos: set[str] = getattr(
                            self._execution_state, "metadata_boost_videos", set()
                        )
                        matched_videos.update(
                            video_id
                            for result in results
                            if (video_id := self._result_video_id(result)) is not None
                        )
                        self._execution_state.metadata_boost_videos = matched_videos
                        continue
                    variant_rankings.append(
                        ChannelRanking(
                            f"{channel.name}:{variant.name}",
                            self._resolve_results(results),
                            (
                                _EXACT_OCR_RRF_WEIGHT
                                if variant.name.startswith("exact_ocr_")
                                else _EXACT_ASR_RRF_WEIGHT
                                if variant.name.startswith("exact_asr_")
                                else 1.0
                            ),
                        )
                    )
                results = (
                    list(variant_rankings[0].results)
                    if len(variant_rankings) == 1
                    else weighted_rrf(variant_rankings, k=effective_k)
                    if variant_rankings
                    else []
                )
                if (
                    processing.decomposition is not None
                    and processing.decomposition.named_entities
                ):
                    boosted: list[dict[str, Any]] = []
                    for result in results:
                        value = dict(result)
                        if value.get("entity_matches"):
                            value["score"] = (
                                float(value.get("score", 0.0))
                                * self.entity_lexical_multiplier
                            )
                        boosted.append(value)
                    boosted.sort(
                        key=lambda value: (
                            -float(value.get("score", 0.0)),
                            str((value.get("payload") or value).get("video_id", "")),
                            str(value.get("id", "")),
                        )
                    )
                    results = boosted
                if asr_channel:
                    results = self._rerank_asr_coverage(results, plan)
                ocr_quality_diagnostics: dict[str, Any] | None = None
                if "ocr" in channel.name.casefold():
                    results, ocr_quality_diagnostics = calibrate_ocr_results(
                        results,
                        config=self.ocr_quality,
                        store=self.ocr_quality_store,
                        exact_terms=tuple(plan.quoted_text)
                        + tuple(processing.decomposition.named_entities)
                        if processing.decomposition is not None
                        else tuple(plan.quoted_text),
                    )
                    processing_value = dict(self.last_query_processing)
                    processing_value["ocr_quality"] = dict(ocr_quality_diagnostics)
                    self.last_query_processing = processing_value
                if variant_rankings:
                    rankings.append(
                        ChannelRanking(
                            channel.name,
                            self._resolve_results(results),
                            channel.weight,
                        )
                    )
                self.last_channel_statuses[channel.name] = {
                    "state": "ready",
                    "reason_code": "query_completed",
                    "query_variant_count": len(variants),
                    "translation_applied": (
                        False
                        if asr_channel
                        else processing.translation.translation_applied
                    ),
                    **(
                        {
                            "query_text_source": (
                                "original_vi"
                                if processing.translation.detected_language == "vi"
                                else "original_query"
                            )
                        }
                        if asr_channel
                        else {}
                    ),
                    "result_count": len(results),
                    "metadata_video_count": (
                        len(self._execution_state.metadata_boost_videos)
                        if metadata_channel
                        else 0
                    ),
                    "latency_ms": round(
                        (time.monotonic() - channel_started) * 1_000,
                        3,
                    ),
                    **(
                        {"ocr_quality": ocr_quality_diagnostics}
                        if ocr_quality_diagnostics is not None
                        else {}
                    ),
                }
            except Exception as error:
                self.last_channel_errors[channel.name] = str(error)
                self.last_channel_statuses[channel.name] = self._failure_status(error)
        return self._filter_frame_zero_rankings(rankings)

    def _run_candidate_rescue(
        self,
        fused: list[dict[str, Any]],
        rankings: list[ChannelRanking],
        request: str | Mapping[str, Any] | Any,
        *,
        effective_k: int,
    ) -> tuple[list[dict[str, Any]], list[ChannelRanking]]:
        """Run the shadow-safe video-first rescue stage."""

        vector_filter, _lexical_filter = self._batch_filters(request)
        visual = next(
            (channel for channel in self.dense_channels if channel.name == "visual"),
            None,
        )
        query_vector = getattr(self._execution_state, "tier1_query_vectors", {}).get(
            "visual"
        )
        if visual is None or query_vector is None:
            diagnostics = {
                "mode": self.candidate_rescue.mode,
                "triggered": False,
                "applied": False,
                "reason_code": "visual_channel_unavailable",
            }
            processing = dict(self.last_query_processing)
            processing["candidate_rescue"] = diagnostics
            self.last_query_processing = processing
            return fused, rankings
        started = time.monotonic()
        try:
            outcome = run_candidate_rescue(
                config=self.candidate_rescue,
                initial_results=fused,
                store=visual.store,
                query_vector=query_vector,
                filters=vector_filter,
                channel_name=visual.name,
            )
            diagnostics = dict(outcome.diagnostics)
            diagnostics["latency_ms"] = round(
                (time.monotonic() - started) * 1_000, 3
            )
            if outcome.rankings:
                resolved = [
                    ChannelRanking(
                        branch.name,
                        tuple(self._resolve_results(branch.results)),
                        branch.weight,
                    )
                    for branch in outcome.rankings
                ]
                rankings = [*rankings, *resolved]
                fused = weighted_rrf(rankings, k=effective_k)
            self.last_channel_statuses["candidate_rescue"] = {
                "state": "ready",
                "reason_code": "query_completed",
                **diagnostics,
            }
        except Exception as error:
            diagnostics = {
                "mode": self.candidate_rescue.mode,
                "triggered": True,
                "applied": False,
                "reason_code": "rescue_failed",
                "latency_ms": round((time.monotonic() - started) * 1_000, 3),
            }
            self.last_channel_errors["candidate_rescue"] = str(error)
            self.last_channel_statuses["candidate_rescue"] = self._failure_status(error)
        processing = dict(self.last_query_processing)
        processing["candidate_rescue"] = diagnostics
        self.last_query_processing = processing
        return fused, rankings

    def _run_video_local(
        self,
        fused: list[dict[str, Any]],
        rankings: list[ChannelRanking],
        request: str | Mapping[str, Any] | Any,
        *,
        effective_k: int,
    ) -> list[dict[str, Any]]:
        """Run video-local late interaction without request-time cache warming."""

        vector_filter, _lexical_filter = self._batch_filters(request)
        raw_batch_id = (
            vector_filter.get("batch_id") if isinstance(vector_filter, Mapping) else None
        )
        if isinstance(raw_batch_id, str):
            batch_id = raw_batch_id
        elif (
            isinstance(raw_batch_id, Sequence)
            and not isinstance(raw_batch_id, (str, bytes))
            and len(raw_batch_id) == 1
        ):
            batch_id = str(raw_batch_id[0])
        else:
            batch_id = None
        query_views = getattr(self._execution_state, "tier1_query_views", {}).get(
            self.video_local.source_channel
        )
        started = time.monotonic()
        if self.video_local.mode == "off":
            diagnostics = {
                "mode": "off",
                "source_channel": self.video_local.source_channel,
                "applied": False,
                "reason_code": "feature_disabled",
                "latency_ms": 0.0,
            }
            self.last_channel_statuses["video_local"] = {
                "state": "not_provisioned",
                **diagnostics,
            }
            processing = dict(self.last_query_processing)
            processing["video_local"] = diagnostics
            self.last_query_processing = processing
            return fused
        if (
            not isinstance(batch_id, str)
            or not batch_id
            or query_views is None
            or self.video_local_trajectories is None
        ):
            reason_code = (
                "batch_filter_unavailable"
                if not isinstance(batch_id, str) or not batch_id
                else "source_query_vector_unavailable"
                if query_views is None
                else "trajectory_cache_unavailable"
            )
            diagnostics = {
                "mode": self.video_local.mode,
                "source_channel": self.video_local.source_channel,
                "applied": False,
                "reason_code": reason_code,
                "latency_ms": 0.0,
            }
            self.last_channel_statuses["video_local"] = {
                "state": "degraded",
                **diagnostics,
            }
        else:
            try:
                results, diagnostics = score_video_local(
                    config=self.video_local,
                    candidates=fused,
                    query_vectors=query_views,
                    trajectories=self.video_local_trajectories,
                    batch_id=batch_id,
                )
                diagnostics["latency_ms"] = round(
                    (time.monotonic() - started) * 1_000, 3
                )
                if self.video_local.mode == "active" and results:
                    branch = ChannelRanking(
                        "video_local_visual",
                        tuple(self._resolve_results(results)),
                        self.video_local.weight,
                    )
                    fused = weighted_rrf([*rankings, branch], k=effective_k)
                self.last_channel_statuses["video_local"] = {
                    "state": "ready",
                    "reason_code": diagnostics.get("reason_code", "query_completed"),
                    **diagnostics,
                }
            except Exception as error:
                diagnostics = {
                    "mode": self.video_local.mode,
                    "source_channel": self.video_local.source_channel,
                    "applied": False,
                    "reason_code": "video_local_failed",
                    "latency_ms": round((time.monotonic() - started) * 1_000, 3),
                }
                self.last_channel_errors["video_local"] = str(error)
                self.last_channel_statuses["video_local"] = self._failure_status(error)
        processing = dict(self.last_query_processing)
        processing["video_local"] = diagnostics
        self.last_query_processing = processing
        return fused

    @staticmethod
    def _request_value(request: str | Mapping[str, Any] | Any, name: str) -> Any:
        if isinstance(request, str):
            return None
        return request.get(name) if isinstance(request, Mapping) else getattr(request, name, None)

    def _prepare_query(
        self,
        request: str | Mapping[str, Any] | Any,
        plan: QueryPlan,
    ) -> QueryProcessing:
        language = self._request_value(request, "query_language") or "auto"
        override = self._request_value(request, "english_text_override")
        if override is None:
            override = self._request_value(request, "event_description_en")
        processing = self.query_adapter.prepare(
            plan.original_text,
            language_hint=str(language),
            english_override=override if isinstance(override, str) else None,
            entity_constraints=self._request_value(request, "entity_constraints") or (),
            legacy_entities=plan.named_entities,
        )
        self.last_query_processing = processing.to_dict()
        return processing

    def collect_channel_rankings(
        self,
        query: str | Mapping[str, Any] | Any,
    ) -> tuple[ChannelRanking, ...]:
        """Collect unfused logical-channel rankings for offline policy tuning."""

        plan = self._plan(query)
        processing = self._prepare_query(query, plan)
        effective_k, _dense_share, _text_share = self._intent_rrf_policy(processing)
        return tuple(
            self._collect_rankings(
                plan,
                query,
                processing=processing,
                effective_k=effective_k,
                verification_requested=self._verification_requested(processing),
            )
        )

    def collect_semantic_view_rankings(
        self,
        query: str | Mapping[str, Any] | Any,
        *,
        channel_name: str,
    ) -> tuple[ChannelRanking, ...]:
        """Collect complete per-view rankings for one dense encoder.

        This offline-only path deliberately bypasses a production
        ``global_only`` policy so train/validation can compare every semantic
        view. It does not alter the pipeline's configured runtime policy.
        """

        if channel_name not in {channel.name for channel in self.dense_channels}:
            raise ValueError(f"unknown dense channel: {channel_name}")
        plan = self._plan(query)
        processing = self._prepare_query(query, plan)
        effective_k, _dense_share, _text_share = self._intent_rrf_policy(processing)
        self._collect_rankings(
            plan,
            query,
            processing=processing,
            effective_k=effective_k,
            verification_requested=False,
            force_semantic_views=True,
        )
        prefix = f"{channel_name}:"
        semantic_results = dict(
            getattr(self._execution_state, "semantic_view_results", {})
        )
        rankings = [
            ChannelRanking(name.removeprefix(prefix), results, 1.0)
            for name, results in semantic_results.items()
            if name.startswith(prefix)
        ]
        if not rankings:
            status = self.last_channel_statuses.get(channel_name, {})
            reason = status.get("reason_code", "no_semantic_views")
            raise RuntimeError(
                f"dense channel {channel_name!r} produced no semantic views: {reason}"
            )
        return tuple(rankings)

    @staticmethod
    def _search_lexical_variants(
        store: Any,
        variants: Sequence[QueryVariant],
        *,
        top_k: int,
        filters: str | None,
    ) -> list[list[dict[str, Any]]]:
        search_many = getattr(store, "search_many", None)
        if callable(search_many) and variants:
            try:
                return list(
                    search_many(
                        tuple(
                            (variant.text, (variant.locale,) if variant.locale else None)
                            for variant in variants
                        ),
                        top_k=top_k,
                        filters=filters,
                    )
                )
            except (TypeError, ValueError, NotImplementedError):
                pass
        return [
            TextualKISPipeline._search_lexical(
                store,
                variant,
                top_k=top_k,
                filters=filters,
            )
            for variant in variants
        ]

    @staticmethod
    def _search_lexical(
        store: Any,
        variant: QueryVariant,
        *,
        top_k: int,
        filters: str | None,
    ) -> list[dict[str, Any]]:
        try:
            parameters = tuple(signature(store.search).parameters.values())
            supports_kwargs = any(value.kind is Parameter.VAR_KEYWORD for value in parameters)
            names = {value.name for value in parameters}
        except (TypeError, ValueError):
            supports_kwargs = False
            names = set()
        kwargs: dict[str, Any] = {"top_k": top_k, "filters": filters}
        if variant.locale and (supports_kwargs or "locales" in names):
            kwargs["locales"] = (variant.locale,)
        return store.search(variant.text, **kwargs)

    def _run_advanced_reranker(
        self,
        processing: QueryProcessing,
        fused: list[dict[str, Any]],
        *,
        precision_profile: str,
    ) -> list[dict[str, Any]]:
        """Run Tier 2 with compatibility for the pre-profile callable contract."""

        if self.advanced_reranker is None or not fused:
            return fused
        started = time.monotonic()
        try:
            try:
                parameters = tuple(signature(self.advanced_reranker).parameters.values())
                supports_kwargs = any(value.kind is Parameter.VAR_KEYWORD for value in parameters)
                names = {value.name for value in parameters}
            except (TypeError, ValueError):
                supports_kwargs, names = False, set()
            if supports_kwargs or "precision_profile" in names:
                output = self.advanced_reranker(
                    processing,
                    list(fused),
                    precision_profile=precision_profile,
                )
            else:
                output = self.advanced_reranker(processing, list(fused))
            diagnostics = getattr(self.advanced_reranker, "last_diagnostics", {})
            processing_value = dict(self.last_query_processing)
            if isinstance(diagnostics, Mapping):
                processing_value["advanced_rerank"] = dict(diagnostics)
            self.last_query_processing = processing_value
            diagnostic_state = (
                str(diagnostics.get("state"))
                if isinstance(diagnostics, Mapping)
                else "ready"
            )
            deadline_exceeded = diagnostic_state == "deadline_exceeded"
            degraded = deadline_exceeded or diagnostic_state == "degraded"
            self.last_channel_statuses["advanced_rerank"] = {
                "state": "degraded" if degraded else "ready",
                "reason_code": (
                    "tier2_deadline_exceeded"
                    if deadline_exceeded
                    else str(diagnostics.get("reason_code", "advanced_evidence_unavailable"))
                    if degraded and isinstance(diagnostics, Mapping)
                    else "query_completed"
                ),
                "result_count": len(output),
                "latency_ms": round((time.monotonic() - started) * 1_000, 3),
            }
            return list(output)
        except Exception as error:
            self.last_channel_errors["advanced_rerank"] = type(error).__name__
            self.last_channel_statuses["advanced_rerank"] = {
                "state": "degraded",
                "reason_code": "advanced_evidence_unavailable",
                "latency_ms": round((time.monotonic() - started) * 1_000, 3),
            }
            return fused

    def _run_cross_modal_rescorer(
        self,
        processing: QueryProcessing,
        fused: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run opt-in Tier 3, or preserve Tier-2 ordering when not provisioned."""

        if self.cross_modal_rescorer is None:
            processing_value = dict(self.last_query_processing)
            processing_value["cross_modal_rerank"] = {
                "state": "degraded",
                "reason_code": "cross_modal_not_provisioned",
            }
            self.last_query_processing = processing_value
            self.last_channel_statuses["cross_modal_rerank"] = {
                "state": "not_provisioned",
                "reason_code": "cross_modal_not_provisioned",
            }
            return fused
        started = time.monotonic()
        try:
            output = self.cross_modal_rescorer(
                processing,
                list(fused),
                query_vectors=dict(getattr(self._execution_state, "tier1_query_vectors", {})),
            )
            diagnostics = getattr(self.cross_modal_rescorer, "last_diagnostics", {})
            processing_value = dict(self.last_query_processing)
            if isinstance(diagnostics, Mapping):
                processing_value["cross_modal_rerank"] = dict(diagnostics)
            self.last_query_processing = processing_value
            self.last_channel_statuses["cross_modal_rerank"] = {
                "state": "ready",
                "reason_code": "query_completed",
                "result_count": len(output),
                "latency_ms": round((time.monotonic() - started) * 1_000, 3),
            }
            return list(output)
        except Exception as error:
            self.last_channel_errors["cross_modal_rerank"] = type(error).__name__
            self.last_channel_statuses["cross_modal_rerank"] = {
                "state": "degraded",
                "reason_code": "cross_modal_evidence_unavailable",
                "latency_ms": round((time.monotonic() - started) * 1_000, 3),
            }
            return fused

    def search(
        self,
        query: str | Mapping[str, Any] | Any,
        *,
        limit: int | None = None,
        query_vectors: Mapping[str, np.ndarray] | None = None,
        refine_raw: bool = True,
        advanced_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute hybrid retrieval and return canonical ranked frame hits."""

        self.last_channel_errors = {}
        self.last_channel_statuses = {}
        self.last_query_processing = {}
        self._execution_state.rank_trace = {}
        request_mode = str(self._request_value(query, "mode") or "automatic")
        set_request_mode = getattr(self.frame_resolver, "set_request_mode", None)
        if callable(set_request_mode):
            set_request_mode(request_mode)
        begin_request = getattr(self.frame_resolver, "begin_request", None)
        if callable(begin_request):
            begin_request()
        requested_limit = self.top_k if limit is None else limit
        if not 1 <= requested_limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        legacy_precision_profile = self._request_value(query, "precision_profile") or "standard"
        if request_mode == "asr_only":
            refine_raw = False
            advanced_rerank = False
        if legacy_precision_profile not in {
            "standard",
            "advanced",
            "experimental_cross_modal",
        }:
            raise ValueError("unsupported KIS precision profile")
        plan = self._plan(query)
        processing = self._prepare_query(query, plan)
        effective_k, _dense_share, _text_share = self._intent_rrf_policy(processing)
        verification_requested = self._verification_requested(processing)
        rankings = self._collect_rankings(
            plan,
            query,
            query_vectors=query_vectors,
            processing=processing,
            effective_k=effective_k,
            verification_requested=verification_requested,
        )
        rankings, effective_k, effective_weights = self._apply_adaptive_weights(
            rankings,
            processing,
            verification_requested=verification_requested,
        )
        processing_value = dict(self.last_query_processing)
        processing_value["precision_profile"] = "automatic"
        processing_value["legacy_precision_profile"] = legacy_precision_profile
        processing_value["effective_rrf_k"] = effective_k
        processing_value["effective_channel_weights"] = {
            name: round(weight, 6) for name, weight in sorted(effective_weights.items())
        }
        self.last_query_processing = processing_value
        fused = weighted_rrf(rankings, k=effective_k)
        fused, rankings = self._run_candidate_rescue(
            fused,
            rankings,
            query,
            effective_k=effective_k,
        )
        fused = self._run_video_local(
            fused,
            rankings,
            query,
            effective_k=effective_k,
        )
        self._trace_stage("tier1_fusion", fused)
        fused = self._apply_metadata_video_boost(fused)
        self._trace_stage("metadata", fused)
        if self.reranker is not None and fused:
            fused = self.reranker.rerank(plan.original_text, fused, top_k=len(fused))

        if self.temporal_windows_ms and fused:
            by_key: dict[tuple[str, int], dict[str, Any]] = {}
            for window_ms in self.temporal_windows_ms:
                for item in aggregate_temporal_windows(fused, window_ms=window_ms):
                    item = dict(item)
                    item["window_size_ms"] = window_ms
                    key = canonical_key(item)
                    if key not in by_key or float(item["score"]) > float(by_key[key]["score"]):
                        by_key[key] = item
            fused = sorted(by_key.values(), key=lambda item: -float(item["score"]))

        fused = self._rerank_temporal_evidence(fused, plan)
        fused = self._apply_asr_temporal_bonus(
            fused,
            enabled=(
                self.asr_enabled
                and request_mode != "visual_only"
                and (plan.asr_intent or request_mode == "asr_only")
            ),
        )
        self._trace_stage("asr_temporal", fused)
        if plan.sequence_constraints:
            processing_value = dict(self.last_query_processing)
            processing_value["sequence_evidence"] = {
                "state": "inconclusive",
                "reason_code": "instance_tracking_not_provisioned",
                "positive_only": True,
                "shot_bounded": True,
                "constraints": [
                    {
                        "kind": value.kind,
                        "source_text": value.source_text,
                        "prospective_action": value.prospective_action,
                    }
                    for value in plan.sequence_constraints
                ],
            }
            self.last_query_processing = processing_value

        fused = self._apply_entity_policy(fused, rankings, processing, rrf_k=effective_k)
        self._trace_stage("entity_policy", fused)

        # Constraint verification is an autonomous Tier-2 route.  The legacy
        # advanced_rerank flag only controls opt-in profile reranking; callers
        # cannot disable required verification for spatial/color/negative
        # constraints.  Transcript-only mode remains deliberately lexical.
        if verification_requested and request_mode != "asr_only":
            fused = self._run_advanced_reranker(
                processing,
                list(fused),
                precision_profile="automatic",
            )
        self._trace_stage("tier2", fused)

        if refine_raw and self.raw_refiner is not None and fused:
            coarse_keys = {canonical_key(item) for item in fused[:20]}
            try:
                raw_parameters = signature(self.raw_refiner).parameters.values()
                raw_kwargs = any(value.kind is Parameter.VAR_KEYWORD for value in raw_parameters)
                raw_names = {value.name for value in raw_parameters}
            except (TypeError, ValueError):
                raw_kwargs, raw_names = False, set()
            visual_texts = tuple(
                value.text for value in self.query_adapter.dense_variants("visual", processing)
            )
            refinement_started = time.monotonic()
            try:
                if raw_kwargs or "query_texts" in raw_names:
                    refined = self.raw_refiner(plan, fused[:20], query_texts=visual_texts)
                else:
                    refined = self.raw_refiner(plan, fused[:20])
                refined_keys = {canonical_key(item) for item in refined}
                replaced_keys = coarse_keys | refined_keys
                fused = list(refined) + [
                    item for item in fused if canonical_key(item) not in replaced_keys
                ]
                fused.sort(key=lambda item: -float(item["score"]))
                self.last_channel_statuses["raw_refinement"] = {
                    "state": "ready",
                    "reason_code": "query_completed",
                    "result_count": len(refined),
                    "latency_ms": round((time.monotonic() - refinement_started) * 1_000, 3),
                }
            except Exception as error:
                status = self._failure_status(error)
                if getattr(error, "restart_required", False):
                    status = {"state": "degraded", "reason_code": "cuda_restart_required"}
                self.last_channel_errors["raw_refinement"] = type(error).__name__
                self.last_channel_statuses["raw_refinement"] = {
                    **status,
                    "latency_ms": round((time.monotonic() - refinement_started) * 1_000, 3),
                }

        self._trace_stage("raw_refinement", fused)
        diversified = diversify(
            fused,
            limit=requested_limit,
            min_separation_ms=self.min_separation_ms,
            early_cutoff=self.early_cutoff,
            early_per_video=self.early_per_video,
            late_per_video=self.late_per_video,
        )
        self._trace_stage("diversity", diversified)
        rank_trace = self.last_rank_trace
        tier1_ranks = {
            (str(item["video_id"]), int(item["source_frame_idx"])): int(item["rank"])
            for item in rank_trace.get("tier1_fusion", ())
        }
        stage_rank_delta: list[dict[str, Any]] = []
        for item in diversified[:20]:
            key = canonical_key(item)
            initial_rank = tier1_ranks.get(key)
            if initial_rank is None:
                continue
            ranks: dict[str, int] = {}
            deltas: dict[str, int] = {}
            for stage, rows in rank_trace.items():
                stage_rank = next(
                    (
                        int(row["rank"])
                        for row in rows
                        if (row["video_id"], int(row["source_frame_idx"])) == key
                    ),
                    None,
                )
                if stage_rank is not None:
                    ranks[stage] = stage_rank
                    deltas[stage] = initial_rank - stage_rank
            stage_rank_delta.append(
                {
                    "video_id": key[0],
                    "source_frame_idx": key[1],
                    "initial_rank": initial_rank,
                    "stage_ranks": ranks,
                    "stage_rank_delta": deltas,
                }
            )
        processing_value = dict(self.last_query_processing)
        processing_value["rank_trace"] = rank_trace
        processing_value["stage_rank_delta"] = stage_rank_delta
        self.last_query_processing = processing_value
        return diversified

    def refine(
        self,
        query: str | Mapping[str, Any] | Any,
        *,
        positive_ids: Sequence[tuple[str, int]] = (),
        negative_ids: Sequence[tuple[str, int]] = (),
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.25,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Apply normalized Rocchio feedback and perform a fresh search."""

        if self.feedback_provider is None:
            raise RuntimeError("relevance feedback requires a feedback_provider")
        if not self.dense_channels:
            raise RuntimeError("relevance feedback requires a dense channel")
        plan = self._plan(query)
        processing = self._prepare_query(query, plan)
        errors: dict[str, str] = {}
        for channel in self.dense_channels:
            try:
                positive = (
                    np.asarray(self.feedback_provider(channel.name, positive_ids), dtype=np.float32)
                    if positive_ids
                    else None
                )
                negative = (
                    np.asarray(self.feedback_provider(channel.name, negative_ids), dtype=np.float32)
                    if negative_ids
                    else None
                )
                variants = self.query_adapter.dense_variants(channel.name, processing)
                if not variants:
                    raise RuntimeError("translation is unavailable for this dense channel")
                originals = np.stack(
                    [encode_query_mean(channel.encoder, value.text) for value in variants]
                )
                original = self._normalized_vector(originals.mean(axis=0))
                updated = alpha * original
                if positive is not None:
                    updated += beta * positive.mean(axis=0)
                if negative is not None:
                    updated -= gamma * negative.mean(axis=0)
                updated = self._normalized_vector(updated)
                return self.search(query, limit=limit, query_vectors={channel.name: updated})
            except Exception as error:
                errors[channel.name] = f"{type(error).__name__}: {error}"
        detail = "; ".join(f"{name}={message}" for name, message in errors.items())
        raise RuntimeError(f"no dense channel can apply the supplied feedback: {detail}")

    def _fuse_scores(
        self, vector_results: list[dict[str, Any]], text_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper around weighted RRF, never raw-score addition."""

        return weighted_rrf(
            (
                ChannelRanking("visual", tuple(vector_results), self.vector_weight),
                ChannelRanking("lexical", tuple(text_results), self.bm25_weight),
            ),
            k=self.rrf_k,
        )
