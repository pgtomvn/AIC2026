"""Deterministic hybrid retrieval core for Textual Known Item Search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.embedding.text_chunks import encode_query_mean
from src.retrieval.fusion import (
    ChannelRanking,
    aggregate_temporal_windows,
    canonical_key,
    diversify,
    weighted_rrf,
)
from src.retrieval.query_plan import QueryPlan, build_query_plan


@dataclass(frozen=True)
class DenseChannel:
    """A cross-modal encoder and vector index pair."""

    name: str
    encoder: Any
    store: Any
    weight: float = 1.0
    depth: int = 300


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
        self.last_channel_errors: dict[str, str] = {}
        self.last_channel_statuses: dict[str, dict[str, Any]] = {}

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
            }

        payload = {
            "version": "textual-kis-v3",
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
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _failure_status(error: Exception) -> dict[str, str]:
        """Return a stable, non-sensitive status for an optional channel."""

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
        resolved: list[dict[str, Any]] = []
        for result in results:
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

    def _collect_rankings(
        self,
        plan: QueryPlan,
        request: str | Mapping[str, Any] | Any,
        *,
        query_vectors: Mapping[str, np.ndarray] | None = None,
    ) -> list[ChannelRanking]:
        vector_filter, lexical_filter = self._batch_filters(request)
        rankings: list[ChannelRanking] = []
        self.last_channel_errors = {}
        self.last_channel_statuses = {}
        for channel in self.dense_channels:
            try:
                is_available = getattr(channel.store, "is_available", None)
                if callable(is_available) and not is_available(vector_filter):
                    self.last_channel_statuses[channel.name] = {
                        "state": "not_provisioned",
                        "reason_code": "no_compatible_points",
                    }
                    continue
                diagnostics: dict[str, Any] = {}
                if query_vectors and channel.name in query_vectors:
                    vectors = np.asarray(query_vectors[channel.name], dtype=np.float32)
                else:
                    diagnostic_encoder = getattr(
                        channel.encoder,
                        "encode_text_with_diagnostics",
                        None,
                    )
                    if callable(diagnostic_encoder):
                        encoded, diagnostic_value = diagnostic_encoder(plan.original_text)
                        vectors = np.asarray(encoded, dtype=np.float32)
                        to_dict = getattr(diagnostic_value, "to_dict", None)
                        if callable(to_dict):
                            diagnostics = dict(to_dict())
                    else:
                        vectors = np.asarray(
                            channel.encoder.encode_text(plan.original_text),
                            dtype=np.float32,
                        )
                if vectors.ndim == 1:
                    vectors = vectors[None, :]
                if vectors.ndim != 2 or vectors.shape[0] < 1:
                    raise ValueError("query encoder must return a non-empty vector matrix")
                chunk_rankings: list[ChannelRanking] = []
                for index, vector in enumerate(vectors):
                    results = channel.store.search(
                        self._normalized_vector(vector),
                        top_k=channel.depth,
                        filters=vector_filter,
                    )
                    chunk_rankings.append(
                        ChannelRanking(
                            f"{channel.name}:query_chunk_{index + 1}",
                            self._resolve_results(results),
                            1.0,
                        )
                    )
                results = (
                    list(chunk_rankings[0].results)
                    if len(chunk_rankings) == 1
                    else weighted_rrf(chunk_rankings, k=self.rrf_k)
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
                    **diagnostics,
                }
            except Exception as error:  # isolate optional model/service channels
                self.last_channel_errors[channel.name] = str(error)
                self.last_channel_statuses[channel.name] = self._failure_status(error)

        for channel in self.lexical_channels:
            try:
                is_available = getattr(channel.store, "is_available", None)
                if callable(is_available) and not is_available(lexical_filter):
                    self.last_channel_statuses[channel.name] = {
                        "state": "not_provisioned",
                        "reason_code": "no_compatible_documents",
                    }
                    continue
                # Quoted on-screen text remains part of the original query and
                # therefore naturally benefits exactness in Meilisearch.
                results = channel.store.search(
                    plan.normalized_text,
                    top_k=channel.depth,
                    filters=lexical_filter,
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
                }
            except Exception as error:
                self.last_channel_errors[channel.name] = str(error)
                self.last_channel_statuses[channel.name] = self._failure_status(error)
        return rankings

    def search(
        self,
        query: str | Mapping[str, Any] | Any,
        *,
        limit: int | None = None,
        query_vectors: Mapping[str, np.ndarray] | None = None,
        refine_raw: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute hybrid retrieval and return canonical ranked frame hits."""

        requested_limit = self.top_k if limit is None else limit
        if not 1 <= requested_limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        plan = self._plan(query)
        rankings = self._collect_rankings(plan, query, query_vectors=query_vectors)
        fused = weighted_rrf(rankings, k=self.rrf_k)
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

        if refine_raw and self.raw_refiner is not None and fused:
            coarse_keys = {canonical_key(item) for item in fused[:20]}
            refined = self.raw_refiner(plan, fused[:20])
            refined_keys = {canonical_key(item) for item in refined}
            replaced_keys = coarse_keys | refined_keys
            fused = list(refined) + [
                item for item in fused if canonical_key(item) not in replaced_keys
            ]
            fused.sort(key=lambda item: -float(item["score"]))
        return diversify(
            fused,
            limit=requested_limit,
            min_separation_ms=self.min_separation_ms,
            early_cutoff=self.early_cutoff,
            early_per_video=self.early_per_video,
            late_per_video=self.late_per_video,
        )

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
                original = encode_query_mean(channel.encoder, plan.original_text)
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
