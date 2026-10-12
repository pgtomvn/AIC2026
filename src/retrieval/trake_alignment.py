"""Strict temporal alignment core for the TRAKE workflow."""

from __future__ import annotations

import heapq
import re
import threading
import unicodedata
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
from src.query.entity_extractor import fold_entity_text
from src.query.translator import QueryProcessing, QueryTextAdapter, QueryVariant
from src.retrieval.trake_scope import trake_filters, trake_video_allowed
from src.retrieval.trake_trajectory_cache import TRAKETrajectoryCache

TRAKE_ALIGNMENT_VERSION = "trake-exact-alignment-v5-entity-routing"


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
    shot_id: str | None = None


@dataclass(frozen=True, slots=True)
class _DPState:
    score: float
    path: tuple[int, ...]


class _TemporalPrefixBest:
    """Track the best predecessor under a linear post-threshold gap cost."""

    def __init__(
        self,
        timestamps_ms: Sequence[int],
        source_frame_indices: Sequence[int],
        *,
        max_gap_ms: int | None,
        gap_penalty_per_second: float,
    ) -> None:
        self.timestamps_ms = timestamps_ms
        self.source_frame_indices = source_frame_indices
        self.max_gap_ms = max_gap_ms
        self.gap_penalty_per_second = gap_penalty_per_second
        self._entries: list[tuple[int, _DPState]] = []
        self._old_cursor = 0
        self._recent_heap: list[
            tuple[tuple[float, int, tuple[int, ...]], int, _DPState]
        ] = []
        self._old_best: tuple[float, _DPState] | None = None

    def _priority(self, state: _DPState) -> tuple[float, int, tuple[int, ...]]:
        frames = tuple(self.source_frame_indices[index] for index in state.path)
        return (-state.score, -frames[0], frames)

    def add(self, frame_index: int, state: _DPState | None) -> None:
        if state is None:
            return
        self._entries.append((frame_index, state))
        heapq.heappush(
            self._recent_heap,
            (self._priority(state), frame_index, state),
        )

    def _move_old(self, timestamp_ms: int) -> None:
        if self.max_gap_ms is None or self.gap_penalty_per_second <= 0.0:
            return
        cutoff = timestamp_ms - self.max_gap_ms
        while self._old_cursor < len(self._entries):
            frame_index, state = self._entries[self._old_cursor]
            if self.timestamps_ms[frame_index] >= cutoff:
                break
            adjusted = state.score + (
                self.gap_penalty_per_second
                * float(self.timestamps_ms[frame_index])
                / 1_000.0
            )
            if self._old_best is None or _scored_state_better(
                adjusted,
                state,
                self._old_best[0],
                self._old_best[1],
                self.source_frame_indices,
            ):
                self._old_best = (adjusted, state)
            self._old_cursor += 1
        while (
            self._recent_heap
            and self.timestamps_ms[self._recent_heap[0][1]] < cutoff
        ):
            heapq.heappop(self._recent_heap)

    def best(self, timestamp_ms: int) -> tuple[float, _DPState] | None:
        self._move_old(timestamp_ms)
        selected: tuple[float, _DPState] | None = None
        if self._recent_heap:
            state = self._recent_heap[0][2]
            selected = (state.score, state)
        if self._old_best is not None:
            assert self.max_gap_ms is not None
            adjusted = self._old_best[0] - (
                self.gap_penalty_per_second
                * float(timestamp_ms - self.max_gap_ms)
                / 1_000.0
            )
            if selected is None or _scored_state_better(
                adjusted,
                self._old_best[1],
                selected[0],
                selected[1],
                self.source_frame_indices,
            ):
                selected = (adjusted, self._old_best[1])
        return selected


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


def _path_better(
    candidate: _DPState,
    current: _DPState,
    source_frame_indices: Sequence[int],
) -> bool:
    candidate_first = source_frame_indices[candidate.path[0]]
    current_first = source_frame_indices[current.path[0]]
    if candidate_first != current_first:
        # With the same future endpoint, a later start creates the shorter span.
        return candidate_first > current_first
    candidate_frames = tuple(source_frame_indices[index] for index in candidate.path)
    current_frames = tuple(source_frame_indices[index] for index in current.path)
    return candidate_frames < current_frames


def _state_better(
    candidate: _DPState,
    current: _DPState | None,
    source_frame_indices: Sequence[int],
) -> bool:
    if current is None:
        return True
    if candidate.score != current.score:
        return candidate.score > current.score
    return _path_better(candidate, current, source_frame_indices)


def _scored_state_better(
    candidate_score: float,
    candidate: _DPState,
    current_score: float,
    current: _DPState,
    source_frame_indices: Sequence[int],
) -> bool:
    if candidate_score != current_score:
        return candidate_score > current_score
    return _path_better(candidate, current, source_frame_indices)


def _nonnegative_penalty(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def strict_monotonic_alignment(
    similarities: np.ndarray,
    source_frame_indices: Sequence[int] | None = None,
    *,
    timestamps_ms: Sequence[int] | None = None,
    shot_ids: Sequence[str | None] | None = None,
    max_gap_ms: int | None = None,
    gap_penalty_per_second: float = 0.0,
    shot_boundary_penalty: float = 0.0,
) -> tuple[tuple[int, ...], float] | None:
    """Align stages to a strict timeline with optional transition penalties.

    The default zero-penalty path retains the original prefix-max ``O(NT)``
    behavior. When enabled, the temporal cost is linear in seconds beyond
    ``max_gap_ms``. A fixed shot cost applies only when both adjacent frames
    carry known, different shot IDs. Returned positions index columns in
    ``similarities``; the score is the penalized stage mean. Ties prefer the
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
    gap_penalty = _nonnegative_penalty(
        gap_penalty_per_second,
        name="gap_penalty_per_second",
    )
    boundary_penalty = _nonnegative_penalty(
        shot_boundary_penalty,
        name="shot_boundary_penalty",
    )
    if max_gap_ms is not None and (
        isinstance(max_gap_ms, bool)
        or not isinstance(max_gap_ms, (int, np.integer))
        or max_gap_ms < 0
    ):
        raise ValueError("max_gap_ms must be a non-negative integer or None")
    max_gap_ms = None if max_gap_ms is None else int(max_gap_ms)
    if gap_penalty > 0.0 and max_gap_ms is None:
        raise ValueError("max_gap_ms is required when gap penalties are enabled")

    if timestamps_ms is None:
        if gap_penalty > 0.0:
            raise ValueError("timestamps_ms is required when gap penalties are enabled")
        timestamps = tuple(range(frame_count))
    else:
        if len(timestamps_ms) != frame_count:
            raise ValueError("timestamps_ms length must match the frame dimension")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in timestamps_ms):
            raise ValueError("timestamps_ms must contain integers")
        timestamps = tuple(int(value) for value in timestamps_ms)
        if any(value < 0 for value in timestamps):
            raise ValueError("timestamps_ms must not contain negative values")
        if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps_ms must be strictly increasing")

    if shot_ids is None:
        shots: tuple[str | None, ...] = (None,) * frame_count
    else:
        if len(shot_ids) != frame_count:
            raise ValueError("shot_ids length must match the frame dimension")
        normalized_shots: list[str | None] = []
        for shot_id in shot_ids:
            if shot_id is None:
                normalized_shots.append(None)
            elif isinstance(shot_id, str):
                normalized_shots.append(shot_id.strip() or None)
            else:
                raise ValueError("shot_ids must contain strings or None")
        shots = tuple(normalized_shots)
    apply_shot_penalty = boundary_penalty > 0.0 and any(
        shot_id is not None for shot_id in shots
    )

    previous: list[_DPState | None] = [
        _DPState(float(matrix[0, frame_index]), (frame_index,))
        for frame_index in range(frame_count)
    ]
    for stage_index in range(1, stage_count):
        current: list[_DPState | None] = [None] * frame_count
        if gap_penalty == 0.0 and boundary_penalty == 0.0:
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
            continue
        global_prefix = _TemporalPrefixBest(
            timestamps,
            frames,
            max_gap_ms=max_gap_ms,
            gap_penalty_per_second=gap_penalty,
        )
        unknown_prefix = (
            _TemporalPrefixBest(
                timestamps,
                frames,
                max_gap_ms=max_gap_ms,
                gap_penalty_per_second=gap_penalty,
            )
            if apply_shot_penalty
            else None
        )
        shot_prefixes: dict[str, _TemporalPrefixBest] = {}
        for frame_index in range(frame_count):
            prior_index = frame_index - 1
            if prior_index >= 0:
                prior = previous[prior_index]
                global_prefix.add(prior_index, prior)
                prior_shot = shots[prior_index]
                if unknown_prefix is not None and prior_shot is None:
                    unknown_prefix.add(prior_index, prior)
                elif apply_shot_penalty and prior_shot is not None:
                    prefix = shot_prefixes.get(prior_shot)
                    if prefix is None:
                        prefix = _TemporalPrefixBest(
                            timestamps,
                            frames,
                            max_gap_ms=max_gap_ms,
                            gap_penalty_per_second=gap_penalty,
                        )
                        shot_prefixes[prior_shot] = prefix
                    prefix.add(prior_index, prior)

            selected = global_prefix.best(timestamps[frame_index])
            current_shot = shots[frame_index]
            if selected is not None and apply_shot_penalty and current_shot is not None:
                selected = (selected[0] - boundary_penalty, selected[1])
                compatible = (
                    unknown_prefix.best(timestamps[frame_index])
                    if unknown_prefix is not None
                    else None
                )
                same_shot = shot_prefixes.get(current_shot)
                if same_shot is not None:
                    same_shot_candidate = same_shot.best(timestamps[frame_index])
                    if same_shot_candidate is not None and (
                        compatible is None
                        or _scored_state_better(
                            same_shot_candidate[0],
                            same_shot_candidate[1],
                            compatible[0],
                            compatible[1],
                            frames,
                        )
                    ):
                        compatible = same_shot_candidate
                if compatible is not None and _scored_state_better(
                    compatible[0],
                    compatible[1],
                    selected[0],
                    selected[1],
                    frames,
                ):
                    selected = compatible
            if selected is not None:
                current[frame_index] = _DPState(
                    selected[0] + float(matrix[stage_index, frame_index]),
                    selected[1].path + (frame_index,),
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
        max_stage_gap_ms: int | None = None,
        temporal_gap_penalty_per_second: float = 0.0,
        shot_boundary_penalty: float = 0.0,
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
        if max_stage_gap_ms is not None and (
            isinstance(max_stage_gap_ms, bool)
            or not isinstance(max_stage_gap_ms, (int, np.integer))
            or max_stage_gap_ms < 0
        ):
            raise ValueError("max_stage_gap_ms must be a non-negative integer or None")
        max_stage_gap_ms = (
            None if max_stage_gap_ms is None else int(max_stage_gap_ms)
        )
        temporal_penalty = _nonnegative_penalty(
            temporal_gap_penalty_per_second,
            name="temporal_gap_penalty_per_second",
        )
        boundary_penalty = _nonnegative_penalty(
            shot_boundary_penalty,
            name="shot_boundary_penalty",
        )
        if temporal_penalty > 0.0 and max_stage_gap_ms is None:
            raise ValueError(
                "max_stage_gap_ms is required when temporal penalties are enabled"
            )
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
        self.max_stage_gap_ms = max_stage_gap_ms
        self.temporal_gap_penalty_per_second = temporal_penalty
        self.shot_boundary_penalty = boundary_penalty
        self.trajectory_cache = trajectory_cache
        self.query_adapter = query_adapter
        self.lexical_channels = tuple(lexical_channels)
        self._execution_state = threading.local()
        self.last_query_processing: dict[str, Any] = {}
        self.last_channel_statuses: dict[str, dict[str, Any]] = {}

    @property
    def last_query_processing(self) -> dict[str, Any]:
        return getattr(self._execution_state, "query_processing", {})

    @last_query_processing.setter
    def last_query_processing(self, value: Mapping[str, Any]) -> None:
        self._execution_state.query_processing = dict(value)

    @property
    def last_channel_statuses(self) -> dict[str, dict[str, Any]]:
        return getattr(self._execution_state, "channel_statuses", {})

    @last_channel_statuses.setter
    def last_channel_statuses(self, value: Mapping[str, Mapping[str, Any]]) -> None:
        self._execution_state.channel_statuses = {
            str(name): dict(status) for name, status in value.items()
        }

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
        eligible_videos: set[str] | None = None,
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
                "filters": trake_filters(batch_id),
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
        videos = {video for video in videos if trake_video_allowed(batch_id, video)}
        lexical_video_order = tuple(video for video in lexical_video_order
                                    if trake_video_allowed(batch_id, video))
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
            if eligible_videos is not None:
                fused_videos &= eligible_videos
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
            ordered = (
                [video_id for video_id in dense_order if video_id in eligible_videos]
                if eligible_videos is not None
                else dense_order
            )
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
                    retrieve_kwargs["filters"] = trake_filters(batch_id)
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

    @staticmethod
    def _payload_text_values(result: Mapping[str, Any]) -> tuple[str, ...]:
        payload = result.get("payload", result)
        if not isinstance(payload, Mapping):
            return ()
        values: list[str] = []
        for field in (
            "text_raw", "text_nfc", "text_accent_folded", "text",
            "ocr_text", "asr_text", "title", "description", "objects",
        ):
            value = payload.get(field)
            if isinstance(value, str) and value:
                values.append(value)
            elif isinstance(value, (list, tuple)):
                values.extend(item for item in value if isinstance(item, str) and item)
        return tuple(values)

    @staticmethod
    def _exact_entity_match(texts: Sequence[str], terms: Sequence[str]) -> bool:
        def fold(value: str) -> str:
            normalized = unicodedata.normalize("NFD", value)
            value = "".join(
                character
                for character in normalized
                if unicodedata.category(character) != "Mn"
            )
            return " ".join(value.replace("đ", "d").replace("Đ", "D").casefold().split())

        folded_texts = tuple(fold(value) for value in texts)
        for term in terms:
            needle = fold(term)
            pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
            if needle and any(pattern.search(text) for text in folded_texts):
                return True
        return False

    def _lexical_candidate_order(
        self,
        processings: Sequence[QueryProcessing],
        batch_id: str,
    ) -> tuple[tuple[str, ...], set[str] | None]:
        """Return bounded lexical video candidates without touching trajectories."""

        required_by_prompt = [
            tuple(processing.decomposition.required_entities)
            if processing.decomposition is not None
            else ()
            for processing in processings
        ]
        has_required = any(required_by_prompt)
        if not self.lexical_channels:
            return (), set() if has_required else None
        escaped_batch = batch_id.replace('"', '\\"')
        batch_filter = f'batch_id = "{escaped_batch}"'
        per_prompt_scores: list[dict[str, float]] = [defaultdict(float) for _ in processings]
        per_prompt_entity_videos: list[dict[str, set[str]]] = [
            defaultdict(set) for _ in processings
        ]
        ready_channels = 0
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
                ready_channels += 1
                depth = min(300, int(getattr(channel, "depth", 100)))
                specifications: list[tuple[int, QueryVariant]] = []
                for prompt_index, processing in enumerate(processings):
                    variants = (
                        self.query_adapter.lexical_variants(channel_name, processing)
                        if self.query_adapter is not None
                        else QueryTextAdapter().lexical_variants(channel_name, processing)
                    )
                    specifications.extend((prompt_index, variant) for variant in variants)
                search_many = getattr(store, "search_many", None)
                if callable(search_many) and specifications:
                    result_sets = list(
                        search_many(
                            tuple(
                                (variant.text, (variant.locale,) if variant.locale else None)
                                for _prompt_index, variant in specifications
                            ),
                            top_k=depth,
                            filters=batch_filter,
                        )
                    )
                else:
                    result_sets = []
                    for _prompt_index, variant in specifications:
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
                        result_sets.append(store.search(variant.text, **kwargs))
                if len(result_sets) != len(specifications):
                    raise RuntimeError("lexical multi-search result count mismatch")
                for (prompt_index, _variant), results in zip(
                    specifications, result_sets, strict=True
                ):
                    processing = processings[prompt_index]
                    decomposition = processing.decomposition
                    for rank, result in enumerate(results, 1):
                        payload = result.get("payload", result)
                        video_id = payload.get("video_id") if isinstance(payload, Mapping) else None
                        if not isinstance(video_id, str) or not video_id:
                            continue
                        matched = False
                        if decomposition is not None:
                            texts = self._payload_text_values(result)
                            for entity in decomposition.named_entities:
                                if self._exact_entity_match(texts, entity.lexical_terms):
                                    identity = fold_entity_text(entity.text)
                                    per_prompt_entity_videos[prompt_index][identity].add(video_id)
                                    matched = True
                        multiplier = 3.0 if matched else 1.0
                        per_prompt_scores[prompt_index][video_id] += multiplier / (60 + rank)
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
        eligible: set[str] | None = None
        if has_required:
            required_sets: list[set[str]] = []
            for prompt_index, entities in enumerate(required_by_prompt):
                for entity in entities:
                    required_sets.append(
                        set(per_prompt_entity_videos[prompt_index].get(fold_entity_text(entity.text), set()))
                    )
            eligible = set.intersection(*required_sets) if ready_channels and required_sets else set()
            diagnostics = dict(self.last_query_processing)
            diagnostics["required_entity_status"] = (
                "required_entity_match"
                if eligible
                else "required_entity_no_match"
                if ready_channels
                else "required_entity_evidence_unavailable"
            )
            self.last_query_processing = diagnostics
        ordered_videos = tuple(
            sorted(
                (
                    video_id
                    for video_id in combined
                    if eligible is None or video_id in eligible
                ),
                key=lambda video_id: (-combined[video_id], video_id),
            )
        )
        return ordered_videos, eligible

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
        constraints = (
            request.global_entity_constraints,
            *(stage.entity_constraints for stage in request.stages),
        )
        processings: list[QueryProcessing] = []
        variants_by_prompt: list[tuple[QueryVariant, ...]] = []
        adapter = self.query_adapter or QueryTextAdapter()
        for prompt_index, (text, override, entity_constraints) in enumerate(
            zip(texts, overrides, constraints, strict=True)
        ):
            processing = adapter.prepare(
                text,
                language_hint=request.query_language,
                english_override=override,
                entity_constraints=entity_constraints,
                prompt_index=prompt_index,
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
        all_entities = [
            entity
            for processing in processings
            for entity in processing.to_dict().get("entities", ())
        ]
        self.last_query_processing = {
            **global_processing.to_dict(),
            "prompt_count": len(processings),
            "translated_prompt_count": sum(
                1
                for processing in processings
                if processing.translation.translation_applied
            ),
            "entities": all_entities,
            "entity_boost_enabled": bool(all_entities),
            "required_entity_count": sum(
                len(processing.decomposition.required_entities)
                for processing in processings
                if processing.decomposition is not None
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
        grouped: dict[
            str,
            dict[int, dict[str, tuple[int, np.ndarray, str | None]]],
        ] = defaultdict(lambda: defaultdict(dict))
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
            raw_shot_id = payload.get("shot_id")
            shot_id = (
                raw_shot_id.strip()
                if isinstance(raw_shot_id, str) and raw_shot_id.strip()
                else None
            )
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
                (timestamp_ms, vector / norm, shot_id),
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
                duplicate_shots = {item[2] for item in duplicates}
                frames.append(
                    _FrameVector(
                        source_frame_idx=frame_idx,
                        timestamp_ms=min(item[0] for item in duplicates),
                        vector=(merged / norm).astype(np.float32, copy=False),
                        shot_id=(
                            next(iter(duplicate_shots))
                            if len(duplicate_shots) == 1
                            else None
                        ),
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
            trake_filters(batch_id, video_id=list(candidate_videos)),
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
        has_required_entities = any(
            processing.decomposition is not None
            and processing.decomposition.required_entities
            for processing in processings
        )
        lexical_video_order, eligible_videos = (
            self._lexical_candidate_order(processings, batch_id)
            if exact_batch is not None
            else ((), set() if has_required_entities else None)
        )
        if exact_batch is None and has_required_entities:
            diagnostics = dict(self.last_query_processing)
            diagnostics["required_entity_status"] = "required_entity_evidence_unavailable"
            self.last_query_processing = diagnostics
        stage_a, seed_points = self._candidate_scores(
            query_sets,
            batch_id,
            lexical_video_order,
            eligible_videos,
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
            shot_ids: Sequence[str | None] | None = None,
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
                timestamps_ms=timestamps_ms,
                shot_ids=shot_ids,
                max_gap_ms=self.max_stage_gap_ms,
                gap_penalty_per_second=self.temporal_gap_penalty_per_second,
                shot_boundary_penalty=self.shot_boundary_penalty,
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
                    tuple(frame.shot_id for frame in frames),
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
