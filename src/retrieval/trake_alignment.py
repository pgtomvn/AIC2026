"""Strict temporal alignment core for the TRAKE workflow."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from src.contracts import (
    TRAKEQueryRequest,
    TRAKEResponse,
    TRAKEResponseItem,
    TRAKEStage,
    TRAKEStageMatch,
)
from src.query.translator import QueryProcessing, QueryTextAdapter, QueryVariant
from src.retrieval.trake_trajectory_cache import TRAKETrajectoryCache

TRAKE_ALIGNMENT_VERSION = "trake-exact-alignment-v3"


class TextEncoder(Protocol):
    """Minimal interface shared by the compatible CLIP/SigLIP2 text towers."""

    def encode_text(self, text: str | list[str]) -> np.ndarray: ...


class TRAKEVectorStore(Protocol):
    """Dense-store operations required by the alignment engine."""

    vector_size: int

    def search_grouped(
        self,
        query_vector: np.ndarray,
        *,
        group_by: str,
        top_k_groups: int,
        group_size: int,
        filters: dict[str, Any],
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]: ...

    def iter_points_with_vectors(
        self,
        filters: dict[str, Any],
        *,
        batch_size: int = 256,
    ) -> Any: ...

    def retrieve_points_with_vectors(
        self,
        point_ids: list[Any] | tuple[Any, ...],
        *,
        filters: dict[str, Any] | None = None,
        batch_size: int = 4_096,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _FrameVector:
    source_frame_idx: int
    timestamp_ms: int
    vector: np.ndarray


@dataclass(frozen=True, slots=True)
class _DPState:
    score: float
    path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RankedAlignment:
    video_id: str
    source_frame_indices: tuple[int, ...]
    timestamps_ms: tuple[int, ...]
    similarities: tuple[float, ...]
    stage_a_score: float
    final_score: float

    @property
    def minimum_similarity(self) -> float:
        return min(self.similarities)


def _normalized_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N, D)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"{name} must not contain zero vectors")
    return matrix / norms


def _state_better(
    candidate: _DPState,
    current: _DPState | None,
    source_frame_indices: Sequence[int],
) -> bool:
    if current is None:
        return True
    if candidate.score != current.score:
        return candidate.score > current.score
    candidate_first = source_frame_indices[candidate.path[0]]
    current_first = source_frame_indices[current.path[0]]
    if candidate_first != current_first:
        # With the same future endpoint, a later start creates the shorter span.
        return candidate_first > current_first
    candidate_frames = tuple(source_frame_indices[index] for index in candidate.path)
    current_frames = tuple(source_frame_indices[index] for index in current.path)
    return candidate_frames < current_frames


def strict_monotonic_alignment(
    similarities: np.ndarray,
    source_frame_indices: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], float] | None:
    """Align all stages to increasing frames using prefix-max DP in ``O(NT)``.

    Returned positions index columns in ``similarities``. Ties prefer the
    shorter source-frame span, then the lexicographically earlier frame chain.
    """

    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("similarities must have shape (N, T)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("similarities must contain only finite values")
    stage_count, frame_count = matrix.shape
    if stage_count < 1:
        raise ValueError("similarities must contain at least one stage")
    if frame_count < stage_count:
        return None
    frames = (
        tuple(range(frame_count))
        if source_frame_indices is None
        else tuple(int(value) for value in source_frame_indices)
    )
    if len(frames) != frame_count:
        raise ValueError("source_frame_indices length must match the frame dimension")
    if any(left >= right for left, right in zip(frames, frames[1:])):
        raise ValueError("source_frame_indices must be strictly increasing")

    previous: list[_DPState | None] = [
        _DPState(float(matrix[0, frame_index]), (frame_index,))
        for frame_index in range(frame_count)
    ]
    for stage_index in range(1, stage_count):
        current: list[_DPState | None] = [None] * frame_count
        prefix_best: _DPState | None = None
        for frame_index in range(frame_count):
            prior_index = frame_index - 1
            if prior_index >= 0:
                prior = previous[prior_index]
                if prior is not None and _state_better(prior, prefix_best, frames):
                    prefix_best = prior
            if prefix_best is not None:
                current[frame_index] = _DPState(
                    prefix_best.score + float(matrix[stage_index, frame_index]),
                    prefix_best.path + (frame_index,),
                )
        previous = current

    best: _DPState | None = None
    for state in previous:
        if state is None:
            continue
        if best is None or state.score > best.score:
            best = state
            continue
        if state.score != best.score:
            continue
        state_frames = tuple(frames[index] for index in state.path)
        best_frames = tuple(frames[index] for index in best.path)
        state_span = state_frames[-1] - state_frames[0]
        best_span = best_frames[-1] - best_frames[0]
        if state_span < best_span or (state_span == best_span and state_frames < best_frames):
            best = state
    if best is None:
        return None
    return best.path, best.score / stage_count


class TRAKEPipeline:
    """Retrieve candidate videos and compute their best strict stage alignment."""

    def __init__(
        self,
        clip_encoder: TextEncoder | None = None,
        vector_store: TRAKEVectorStore | None = None,
        top_k_videos: int = 200,
        *,
        encoder: TextEncoder | None = None,
        channel_name: str = "visual",
        candidate_depth: int = 300,
        seed_frames_per_prompt: int = 4,
        grouped_search_workers: int = 4,
        vector_scroll_batch_size: int = 4_096,
        global_weight: float = 0.25,
        trajectory_cache: TRAKETrajectoryCache | None = None,
        query_adapter: QueryTextAdapter | None = None,
        lexical_channels: Sequence[Any] = (),
    ) -> None:
        selected_encoder = encoder or clip_encoder
        if encoder is not None and clip_encoder is not None and encoder is not clip_encoder:
            raise ValueError("pass encoder or clip_encoder, not both")
        if top_k_videos < 1 or top_k_videos > 200:
            raise ValueError("top_k_videos must be between 1 and 200")
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be positive")
        if not 1 <= seed_frames_per_prompt <= 32:
            raise ValueError("seed_frames_per_prompt must be between 1 and 32")
        if not 1 <= grouped_search_workers <= 12:
            raise ValueError("grouped_search_workers must be between 1 and 12")
        if not 256 <= vector_scroll_batch_size <= 16_384:
            raise ValueError("vector_scroll_batch_size must be between 256 and 16384")
        if not isinstance(channel_name, str) or not channel_name.strip():
            raise ValueError("channel_name must be a non-empty string")
        if (
            isinstance(global_weight, bool)
            or not isinstance(global_weight, (int, float))
            or not 0.0 <= float(global_weight) <= 1.0
        ):
            raise ValueError("global_weight must be between 0 and 1")
        self.clip_encoder = selected_encoder
        self.encoder = selected_encoder
        self.vector_store = vector_store
        self.top_k_videos = top_k_videos
        self.channel_name = channel_name
        self.candidate_depth = candidate_depth
        self.seed_frames_per_prompt = seed_frames_per_prompt
        self.grouped_search_workers = grouped_search_workers
        self.vector_scroll_batch_size = vector_scroll_batch_size
        self.global_weight = float(global_weight)
        self.trajectory_cache = trajectory_cache
        self.query_adapter = query_adapter
        self.lexical_channels = tuple(lexical_channels)
        self.last_query_processing: dict[str, Any] = {}
        self.last_channel_statuses: dict[str, dict[str, Any]] = {}

    def _dependencies(self) -> tuple[TextEncoder, TRAKEVectorStore]:
        if self.encoder is None or self.vector_store is None:
            raise RuntimeError("TRAKE encoder and vector store are not configured")
        return self.encoder, self.vector_store

    @staticmethod
    def _group_scores(groups: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for group in groups:
            group_id = group.get("group_id")
            if group_id is None:
                hits = group.get("hits", ())
                first = hits[0] if hits else group
                payload = first.get("payload", {}) if isinstance(first, Mapping) else {}
                group_id = payload.get("video_id")
            if group_id is None:
                continue
            score = group.get("score")
            if score is None:
                hits = group.get("hits", ())
                score = hits[0].get("score", -1.0) if hits else -1.0
            numeric = float(score)
            if np.isfinite(numeric):
                key = str(group_id)
                scores[key] = max(numeric, scores.get(key, -float("inf")))
        return scores

    def _candidate_scores(
        self,
        query_vectors: Sequence[np.ndarray],
        batch_id: str,
        lexical_video_order: Sequence[str] = (),
    ) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
        if isinstance(query_vectors, np.ndarray):
            if query_vectors.ndim != 2:
                raise ValueError("query vectors must be a two-dimensional matrix")
            query_vectors = tuple(row[None, :] for row in query_vectors)
        _, store = self._dependencies()
        per_prompt: list[dict[str, float]] = [dict() for _ in query_vectors]
        seed_points: list[dict[str, Any]] = []
        point_refs: list[tuple[str, Any]] = []
        grouped_search = store.search_grouped
        retrieve_points = getattr(store, "retrieve_points_with_vectors", None)
        use_exact_cache = self.trajectory_cache is not None
        supports_point_retrieval = callable(retrieve_points) and not use_exact_cache
        try:
            parameters = signature(grouped_search).parameters.values()
            supports_vectors = any(
                parameter.name == "with_vectors" or parameter.kind is Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            supports_payload_selection = any(
                parameter.name == "with_payload"
                or parameter.kind is Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            supports_vectors = True
            supports_payload_selection = True

        def retrieve(vector: np.ndarray) -> list[dict[str, Any]]:
            kwargs: dict[str, Any] = {
                "group_by": "video_id",
                "top_k_groups": self.candidate_depth,
                # Exact Stage-B reads full trajectories from cache; one hit is
                # sufficient to retain each group's maximum Stage-A score.
                "group_size": 1 if use_exact_cache else self.seed_frames_per_prompt,
                "filters": {"batch_id": batch_id},
            }
            if supports_vectors:
                # The production adapter retains private Qdrant point IDs, so
                # vectors can be fetched only after the top-video cut.  Older
                # stores keep the prior grouped-vector behavior.
                kwargs["with_vectors"] = not (
                    use_exact_cache or supports_point_retrieval
                )
            if supports_payload_selection:
                kwargs["with_payload"] = not use_exact_cache
            return grouped_search(vector, **kwargs)

        flattened: list[np.ndarray] = []
        prompt_indices: list[int] = []
        for prompt_index, variants in enumerate(query_vectors):
            if variants.ndim != 2 or not variants.shape[0]:
                raise ValueError("each TRAKE prompt requires at least one query variant")
            for vector in variants:
                flattened.append(vector)
                prompt_indices.append(prompt_index)
        worker_count = min(self.grouped_search_workers, len(flattened))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves prompt order even when requests complete out of order.
            grouped_results = tuple(executor.map(retrieve, flattened))
        for prompt_index, groups in zip(prompt_indices, grouped_results, strict=True):
            for video_id, score in self._group_scores(groups).items():
                per_prompt[prompt_index][video_id] = max(
                    score,
                    per_prompt[prompt_index].get(video_id, -float("inf")),
                )
            for group in groups:
                hits = group.get("hits", ())
                if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
                    continue
                group_id = group.get("group_id")
                if group_id is None and hits:
                    first = hits[0]
                    payload = first.get("payload", {}) if isinstance(first, Mapping) else {}
                    group_id = payload.get("video_id")
                for hit in hits:
                    if not isinstance(hit, Mapping):
                        continue
                    if hit.get("vector") is not None:
                        seed_points.append(dict(hit))
                    point_id = hit.get("_point_id")
                    if (
                        supports_point_retrieval
                        and group_id is not None
                        and point_id is not None
                    ):
                        point_refs.append((str(group_id), point_id))
        videos = set().union(*(scores.keys() for scores in per_prompt))
        stage_count = len(per_prompt) - 1
        stage_a: dict[str, float] = {}
        for video_id in videos:
            global_score = per_prompt[0].get(video_id, -1.0)
            stage_scores = [scores.get(video_id, -1.0) for scores in per_prompt[1:]]
            stage_a[video_id] = self.global_weight * global_score + (
                1.0 - self.global_weight
            ) * (sum(stage_scores) / stage_count)
        dense_order = sorted(stage_a, key=lambda video: (-stage_a[video], video))
        if lexical_video_order:
            dense_ranks = {video_id: rank for rank, video_id in enumerate(dense_order, 1)}
            lexical_ranks = {
                video_id: rank for rank, video_id in enumerate(lexical_video_order, 1)
            }
            fused_videos = set(dense_ranks) | set(lexical_ranks)
            ordered = sorted(
                fused_videos,
                key=lambda video_id: (
                    -(
                        (1.0 / (60 + dense_ranks[video_id]) if video_id in dense_ranks else 0.0)
                        + (
                            1.0 / (60 + lexical_ranks[video_id])
                            if video_id in lexical_ranks
                            else 0.0
                        )
                    ),
                    video_id,
                ),
            )
            for video_id in lexical_ranks:
                stage_a.setdefault(video_id, -1.0)
        else:
            ordered = dense_order
        selected = {video: stage_a[video] for video in ordered[: self.top_k_videos]}
        selected_ids = set(selected)
        if supports_point_retrieval and point_refs:
            requested_ids: list[Any] = []
            seen_ids: set[str] = set()
            for video_id, point_id in point_refs:
                if video_id not in selected_ids:
                    continue
                identity = str(point_id)
                if identity in seen_ids:
                    continue
                seen_ids.add(identity)
                requested_ids.append(point_id)
            if requested_ids:
                retrieve_kwargs: dict[str, Any] = {}
                try:
                    retrieve_parameters = signature(retrieve_points).parameters.values()
                    retrieve_kwargs_supported = any(
                        parameter.kind is Parameter.VAR_KEYWORD
                        for parameter in retrieve_parameters
                    )
                    retrieve_parameter_names = {
                        parameter.name for parameter in retrieve_parameters
                    }
                except (TypeError, ValueError):
                    retrieve_kwargs_supported = True
                    retrieve_parameter_names = set()
                if retrieve_kwargs_supported or "filters" in retrieve_parameter_names:
                    retrieve_kwargs["filters"] = {"batch_id": batch_id}
                if retrieve_kwargs_supported or "batch_size" in retrieve_parameter_names:
                    retrieve_kwargs["batch_size"] = self.vector_scroll_batch_size
                try:
                    fetched = retrieve_points(requested_ids, **retrieve_kwargs)
                except (AttributeError, NotImplementedError):
                    # Exact full-scroll fallback for clients without retrieve.
                    fetched = ()
                seed_points.extend(
                    dict(point)
                    for point in fetched
                    if isinstance(point, Mapping) and point.get("vector") is not None
                )
        selected_seed_points = tuple(
            point
            for point in seed_points
            if isinstance(point.get("payload"), Mapping)
            and point["payload"].get("video_id") in selected_ids
        )
        return selected, selected_seed_points

    @staticmethod
    def _video_ids_from_results(results: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        values: list[str] = []
        seen: set[str] = set()
        for item in results:
            payload = item.get("payload", item)
            video_id = payload.get("video_id") if isinstance(payload, Mapping) else None
            if isinstance(video_id, str) and video_id and video_id not in seen:
                seen.add(video_id)
                values.append(video_id)
        return tuple(values)

    def _lexical_candidate_order(
        self,
        processings: Sequence[QueryProcessing],
        batch_id: str,
    ) -> tuple[str, ...]:
        """Return bounded lexical video candidates without touching trajectories."""

        if not self.lexical_channels:
            return ()
        escaped_batch = batch_id.replace('"', '\\"')
        batch_filter = f'batch_id = "{escaped_batch}"'
        per_prompt_scores: list[dict[str, float]] = [defaultdict(float) for _ in processings]
        for channel in self.lexical_channels:
            channel_name = str(getattr(channel, "name", "metadata"))
            store = getattr(channel, "store", None)
            if store is None:
                continue
            try:
                available = getattr(store, "is_available", None)
                if callable(available) and not available(batch_filter):
                    self.last_channel_statuses[channel_name] = {
                        "state": "not_provisioned",
                        "reason_code": "no_compatible_documents",
                    }
                    continue
                depth = min(300, int(getattr(channel, "depth", 100)))
                for prompt_index, processing in enumerate(processings):
                    variants = (
                        self.query_adapter.lexical_variants(channel_name, processing)
                        if self.query_adapter is not None
                        else QueryTextAdapter().lexical_variants(channel_name, processing)
                    )
                    variant_orders: list[tuple[str, ...]] = []
                    for variant in variants:
                        kwargs: dict[str, Any] = {
                            "top_k": depth,
                            "filters": batch_filter,
                        }
                        try:
                            parameters = tuple(signature(store.search).parameters.values())
                            if variant.locale and any(
                                value.name == "locales" or value.kind is Parameter.VAR_KEYWORD
                                for value in parameters
                            ):
                                kwargs["locales"] = (variant.locale,)
                        except (TypeError, ValueError):
                            pass
                        results = store.search(variant.text, **kwargs)
                        variant_orders.append(self._video_ids_from_results(results))
                    for order in variant_orders:
                        for rank, video_id in enumerate(order, 1):
                            per_prompt_scores[prompt_index][video_id] += 1.0 / (60 + rank)
                self.last_channel_statuses[channel_name] = {
                    "state": "ready",
                    "reason_code": "query_completed",
                }
            except Exception:
                self.last_channel_statuses[channel_name] = {
                    "state": "service_unavailable",
                    "reason_code": "channel_query_failed",
                }
        combined: dict[str, float] = defaultdict(float)
        for scores in per_prompt_scores:
            ordered = sorted(scores, key=lambda video_id: (-scores[video_id], video_id))
            for rank, video_id in enumerate(ordered, 1):
                combined[video_id] += 1.0 / (60 + rank)
        return tuple(sorted(combined, key=lambda video_id: (-combined[video_id], video_id)))

    def _query_embeddings(
        self,
        encoder: TextEncoder,
        request: TRAKEQueryRequest,
    ) -> tuple[tuple[np.ndarray, ...], tuple[QueryProcessing, ...]]:
        texts = (request.global_description, *(stage.description for stage in request.stages))
        overrides = (
            request.global_description_en,
            *(stage.description_en for stage in request.stages),
        )
        processings: list[QueryProcessing] = []
        variants_by_prompt: list[tuple[QueryVariant, ...]] = []
        adapter = self.query_adapter or QueryTextAdapter()
        for text, override in zip(texts, overrides, strict=True):
            processing = adapter.prepare(
                text,
                language_hint=request.query_language,
                english_override=override,
            )
            variants = adapter.dense_variants(self.channel_name, processing)
            if not variants:
                raise RuntimeError("translation_unavailable")
            processings.append(processing)
            variants_by_prompt.append(variants)

        flattened = [variant.text for variants in variants_by_prompt for variant in variants]
        matrix = _normalized_matrix(encoder.encode_text(flattened), name="query embeddings")
        query_sets: list[np.ndarray] = []
        cursor = 0
        for variants in variants_by_prompt:
            stop = cursor + len(variants)
            query_sets.append(matrix[cursor:stop])
            cursor = stop
        global_processing = processings[0]
        self.last_query_processing = {
            **global_processing.to_dict(),
            "prompt_count": len(processings),
            "translated_prompt_count": sum(
                1
                for processing in processings
                if processing.translation.translation_applied
            ),
            "channel_policies": {
                self.channel_name: {
                    "policy": "+".join(
                        global_processing.policies.get(self.channel_name, ("identity",))
                    ),
                    "variant_count": sum(len(value) for value in variants_by_prompt),
                }
            },
        }
        return tuple(query_sets), tuple(processings)

    @staticmethod
    def _payload_int(payload: Mapping[str, Any], name: str) -> int | None:
        value = payload.get(name)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("+").isdigit():
            return int(value.strip())
        return None

    def _canonical_frames(
        self,
        batch_id: str,
        candidate_videos: Sequence[str],
        dimension: int,
        points: Any,
    ) -> dict[str, tuple[_FrameVector, ...]]:
        if not candidate_videos:
            return {}
        grouped: dict[str, dict[int, dict[str, tuple[int, np.ndarray]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        allowed = set(candidate_videos)
        for point in points:
            if not isinstance(point, Mapping):
                continue
            payload = point.get("payload", {})
            if not isinstance(payload, Mapping):
                continue
            video_id = payload.get("video_id")
            point_batch_id = payload.get("batch_id")
            frame_idx = self._payload_int(payload, "source_frame_idx")
            timestamp_ms = self._payload_int(payload, "timestamp_ms")
            if point_batch_id != batch_id:
                continue
            if not isinstance(video_id, str) or video_id not in allowed:
                continue
            if frame_idx is None or frame_idx < 0 or timestamp_ms is None or timestamp_ms < 0:
                continue
            raw_vector = point.get("vector")
            if isinstance(raw_vector, Mapping):
                if len(raw_vector) != 1:
                    continue
                raw_vector = next(iter(raw_vector.values()))
            try:
                vector = np.asarray(raw_vector, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            point_id = point.get("_point_id", point.get("id"))
            identity = (
                str(point_id)
                if point_id is not None
                else f"anonymous:{timestamp_ms}:{len(grouped[video_id][frame_idx])}"
            )
            grouped[video_id][frame_idx].setdefault(
                identity,
                (timestamp_ms, vector / norm),
            )

        result: dict[str, tuple[_FrameVector, ...]] = {}
        for video_id, by_frame in grouped.items():
            frames: list[_FrameVector] = []
            for frame_idx, points_by_id in by_frame.items():
                duplicates = tuple(points_by_id.values())
                vectors = np.stack([item[1] for item in duplicates])
                merged = np.mean(vectors, axis=0)
                norm = float(np.linalg.norm(merged))
                if norm <= 0 or not np.isfinite(norm):
                    continue
                frames.append(
                    _FrameVector(
                        source_frame_idx=frame_idx,
                        timestamp_ms=min(item[0] for item in duplicates),
                        vector=(merged / norm).astype(np.float32, copy=False),
                    )
                )
            frames.sort(key=lambda frame: (frame.timestamp_ms, frame.source_frame_idx))
            # A malformed timestamp mapping must not produce a non-canonical output chain.
            monotonic: list[_FrameVector] = []
            for frame in frames:
                if not monotonic or (
                    frame.source_frame_idx > monotonic[-1].source_frame_idx
                    and frame.timestamp_ms > monotonic[-1].timestamp_ms
                ):
                    monotonic.append(frame)
            result[video_id] = tuple(monotonic)
        return result

    def _video_frames(
        self,
        batch_id: str,
        candidate_videos: Sequence[str],
        dimension: int,
    ) -> dict[str, tuple[_FrameVector, ...]]:
        _, store = self._dependencies()
        if not candidate_videos:
            return {}
        points = store.iter_points_with_vectors(
            {"batch_id": batch_id, "video_id": list(candidate_videos)},
            batch_size=self.vector_scroll_batch_size,
        )
        return self._canonical_frames(batch_id, candidate_videos, dimension, points)

    def _seeded_video_frames(
        self,
        batch_id: str,
        candidate_videos: Sequence[str],
        dimension: int,
        stage_count: int,
        seed_points: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[_FrameVector, ...]]:
        """Use bounded grouped-search vectors with an old-backend fallback.

        Older stores and test fakes may omit vectors from grouped hits.  Those
        backends retain the exact paged-scroll behavior used before the seed
        optimization.  On a vector-capable backend, candidates with fewer than
        ``stage_count`` canonical seeds remain infeasible instead of triggering
        a latency spike that grows with the number of stages.
        """

        if not seed_points:
            return self._video_frames(batch_id, candidate_videos, dimension)
        seeded = self._canonical_frames(
            batch_id,
            candidate_videos,
            dimension,
            seed_points,
        )
        return seeded

    def search(
        self,
        request: TRAKEQueryRequest,
        *,
        default_batch_id: str | None = None,
        channel_statuses: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> TRAKEResponse:
        """Run grouped retrieval, exact-vector scoring, and strict temporal DP."""

        if not isinstance(request, TRAKEQueryRequest):
            request = TRAKEQueryRequest.model_validate(request)
        encoder, store = self._dependencies()
        batch_id = request.resolve_batch(default_batch_id)
        query_sets, processings = self._query_embeddings(encoder, request)
        dimension = query_sets[0].shape[1]
        if any(query_set.shape[1] != dimension for query_set in query_sets):
            raise ValueError("encoder returned inconsistent query embedding dimensions")
        if dimension != int(store.vector_size):
            raise ValueError(
                "query embedding dimension does not match the selected dense collection"
            )

        exact_batch = (
            self.trajectory_cache.require(batch_id)
            if self.trajectory_cache is not None
            else None
        )
        self.last_channel_statuses = {}
        lexical_video_order = (
            self._lexical_candidate_order(processings, batch_id)
            if exact_batch is not None
            else ()
        )
        stage_a, seed_points = self._candidate_scores(
            query_sets,
            batch_id,
            lexical_video_order,
        )
        frames_by_video = (
            {}
            if exact_batch is not None
            else self._seeded_video_frames(
                batch_id,
                tuple(stage_a),
                dimension,
                len(request.stages),
                seed_points,
            )
        )
        alignments: list[_RankedAlignment] = []

        def append_alignment(
            video_id: str,
            stage_a_score: float,
            source_frames: np.ndarray,
            timestamps_ms: np.ndarray,
            stage_similarities: np.ndarray,
            global_score: float,
        ) -> None:
            if (
                source_frames.ndim != 1
                or timestamps_ms.ndim != 1
                or source_frames.shape != timestamps_ms.shape
                or np.any(source_frames < 0)
                or np.any(timestamps_ms < 0)
                or np.any(np.diff(source_frames) <= 0)
                or np.any(np.diff(timestamps_ms) <= 0)
            ):
                # A malformed canonical mapping must isolate only its video;
                # it must never escape as a public non-monotonic alignment.
                return
            alignment = strict_monotonic_alignment(
                stage_similarities,
                source_frames,
            )
            if alignment is None:
                return
            positions, mean_stage_score = alignment
            selected_scores = tuple(
                float(stage_similarities[index, position])
                for index, position in enumerate(positions)
            )
            final_score = self.global_weight * global_score + (
                1.0 - self.global_weight
            ) * mean_stage_score
            alignments.append(
                _RankedAlignment(
                    video_id=video_id,
                    source_frame_indices=tuple(
                        int(source_frames[position]) for position in positions
                    ),
                    timestamps_ms=tuple(
                        int(timestamps_ms[position]) for position in positions
                    ),
                    similarities=selected_scores,
                    stage_a_score=stage_a_score,
                    final_score=final_score,
                )
            )

        if exact_batch is not None:
            exact_candidates: list[tuple[str, float, Any]] = []
            for video_id, stage_a_score in stage_a.items():
                trajectory = exact_batch.video(video_id)
                if trajectory is None or len(trajectory) < len(request.stages):
                    continue
                exact_candidates.append((video_id, stage_a_score, trajectory))
            if exact_candidates:
                # One large BLAS call is substantially faster than hundreds of
                # tiny per-video multiplications while retaining every frame.
                packed_vectors = np.concatenate(
                    [item[2].vectors for item in exact_candidates], axis=0
                )
                packed_similarities = np.stack(
                    [
                        np.max(query_set @ packed_vectors.T, axis=0)
                        for query_set in query_sets
                    ]
                )
                cursor = 0
                for video_id, stage_a_score, trajectory in exact_candidates:
                    stop = cursor + len(trajectory)
                    similarities = packed_similarities[:, cursor:stop]
                    append_alignment(
                        video_id,
                        stage_a_score,
                        trajectory.source_frame_indices,
                        trajectory.timestamps_ms,
                        similarities[1:],
                        float(np.max(similarities[0])),
                    )
                    cursor = stop
        else:
            global_vectors = query_sets[0]
            stage_vectors = query_sets[1:]
            for video_id, stage_a_score in stage_a.items():
                frames = frames_by_video.get(video_id, ())
                if not frames:
                    continue
                matrix = np.stack([frame.vector for frame in frames])
                source_frames = np.asarray(
                    [frame.source_frame_idx for frame in frames], dtype=np.int64
                )
                timestamps_ms = np.asarray(
                    [frame.timestamp_ms for frame in frames], dtype=np.int64
                )
                if matrix.shape[0] < len(request.stages):
                    continue
                stage_similarities = np.stack(
                    [np.max(vectors @ matrix.T, axis=0) for vectors in stage_vectors]
                )
                append_alignment(
                    video_id,
                    stage_a_score,
                    source_frames,
                    timestamps_ms,
                    stage_similarities,
                    float(np.max(global_vectors @ matrix.T)),
                )
        alignments.sort(
            key=lambda item: (
                -item.final_score,
                -item.minimum_similarity,
                -item.stage_a_score,
                item.video_id,
            )
        )

        results: list[TRAKEResponseItem] = []
        for rank, alignment in enumerate(alignments[: request.top_k], start=1):
            matches = tuple(
                TRAKEStageMatch(
                    stage_index=index,
                    source_frame_idx=alignment.source_frame_indices[index - 1],
                    timestamp_ms=alignment.timestamps_ms[index - 1],
                    similarity=max(-1.0, min(1.0, alignment.similarities[index - 1])),
                )
                for index in range(1, len(alignment.source_frame_indices) + 1)
            )
            confidence = max(0.0, min(1.0, (alignment.final_score + 1.0) / 2.0))
            results.append(
                TRAKEResponseItem(
                    rank=rank,
                    batch_id=batch_id,
                    video_id=alignment.video_id,
                    stage_matches=matches,
                    confidence_score=confidence,
                )
            )
        statuses = {name: dict(status) for name, status in (channel_statuses or {}).items()}
        statuses.update(self.last_channel_statuses)
        if self.channel_name not in statuses:
            statuses[self.channel_name] = {"state": "ready", "attempted": True}
        return TRAKEResponse(
            query_id=request.query_id,
            results=tuple(results),
            channel_statuses=statuses,
        )

    def warm(self, batch_id: str) -> dict[str, Any]:
        """Warm the exact trajectory cache without loading it at construction."""

        if self.trajectory_cache is None:
            return {
                "state": "not_configured",
                "reason_code": "trajectory_cache_not_configured",
            }
        return self.trajectory_cache.warm(batch_id)

    def align(
        self,
        event_descriptions: list[str],
        *,
        global_description: str | None = None,
        batch_id: str | None = None,
        query_id: str = "trake",
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """Compatibility entrypoint returning the historical list-of-dicts shape."""

        stages = tuple(
            TRAKEStage(stage_index=index, description=value)
            for index, value in enumerate(event_descriptions, start=1)
        )
        request = TRAKEQueryRequest(
            query_id=query_id,
            global_description=global_description or " ".join(event_descriptions),
            stages=stages,
            batch_filters=() if batch_id is None else (batch_id,),
            top_k=top_k,
        )
        response = self.search(request, default_batch_id=batch_id)
        return [item.model_dump(mode="json", by_alias=True) for item in response.results]

    def _dtw_align(
        self,
        event_embs: np.ndarray,
        frame_embs: np.ndarray,
    ) -> tuple[list[int], float]:
        """Compatibility name for the new strict prefix-maximum alignment."""

        events = _normalized_matrix(event_embs, name="event embeddings")
        frames = _normalized_matrix(frame_embs, name="frame embeddings")
        if events.shape[1] != frames.shape[1]:
            raise ValueError("event and frame embedding dimensions must match")
        result = strict_monotonic_alignment(events @ frames.T)
        if result is None:
            return [], -1.0
        positions, score = result
        return list(positions), score


__all__ = [
    "TRAKE_ALIGNMENT_VERSION",
    "TRAKEPipeline",
    "strict_monotonic_alignment",
]
