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
        trajectory_cache: TRAKETrajectoryCache | None = None,
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
        self.clip_encoder = selected_encoder
        self.encoder = selected_encoder
        self.vector_store = vector_store
        self.top_k_videos = top_k_videos
        self.channel_name = channel_name
        self.candidate_depth = candidate_depth
        self.seed_frames_per_prompt = seed_frames_per_prompt
        self.grouped_search_workers = grouped_search_workers
        self.vector_scroll_batch_size = vector_scroll_batch_size
        self.trajectory_cache = trajectory_cache

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
        query_vectors: np.ndarray,
        batch_id: str,
    ) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
        _, store = self._dependencies()
        per_prompt: list[dict[str, float]] = []
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

        worker_count = min(self.grouped_search_workers, len(query_vectors))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves prompt order even when requests complete out of order.
            grouped_results = tuple(executor.map(retrieve, query_vectors))
        for groups in grouped_results:
            per_prompt.append(self._group_scores(groups))
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
            stage_a[video_id] = 0.25 * global_score + 0.75 * (sum(stage_scores) / stage_count)
        ordered = sorted(stage_a, key=lambda video: (-stage_a[video], video))
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
        texts = [request.global_description, *(stage.description for stage in request.stages)]
        queries = _normalized_matrix(encoder.encode_text(texts), name="query embeddings")
        if queries.shape[0] != len(texts):
            raise ValueError("encoder returned the wrong number of query embeddings")
        if queries.shape[1] != int(store.vector_size):
            raise ValueError(
                "query embedding dimension does not match the selected dense collection"
            )

        stage_a, seed_points = self._candidate_scores(queries, batch_id)
        exact_batch = (
            self.trajectory_cache.require(batch_id)
            if self.trajectory_cache is not None
            else None
        )
        frames_by_video = (
            {}
            if exact_batch is not None
            else self._seeded_video_frames(
                batch_id,
                tuple(stage_a),
                queries.shape[1],
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
            final_score = 0.25 * global_score + 0.75 * mean_stage_score
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
                packed_similarities = queries @ packed_vectors.T
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
            global_vector = queries[0]
            stage_vectors = queries[1:]
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
                stage_similarities = stage_vectors @ matrix.T
                append_alignment(
                    video_id,
                    stage_a_score,
                    source_frames,
                    timestamps_ms,
                    stage_similarities,
                    float(np.max(matrix @ global_vector)),
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
