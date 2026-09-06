"""Video-local late interaction over an already-warmed trajectory cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.retrieval.candidate_rescue import validate_rollout_mode
from src.retrieval.fusion import canonical_key


@dataclass(frozen=True, slots=True)
class VideoLocalConfig:
    """Validated late-interaction policy; defaults do not mutate rankings."""

    mode: str = "off"
    source_channel: str = "organizer_clip"
    candidate_videos: int = 12
    frames_per_video: int = 5
    weight: float = 0.6
    support_window_ms: int = 8_000
    gaussian_sigma_ms: int = 4_000

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "VideoLocalConfig":
        raw = value or {}
        config = cls(
            mode=validate_rollout_mode(raw.get("mode", "off"), name="video_local"),
            source_channel=str(raw.get("source_channel", "organizer_clip")).strip(),
            candidate_videos=int(raw.get("candidate_videos", 12)),
            frames_per_video=int(raw.get("frames_per_video", 5)),
            weight=float(raw.get("weight", 0.6)),
            support_window_ms=int(raw.get("support_window_ms", 8_000)),
            gaussian_sigma_ms=int(raw.get("gaussian_sigma_ms", 4_000)),
        )
        if min(
            config.candidate_videos,
            config.frames_per_video,
            config.support_window_ms,
            config.gaussian_sigma_ms,
        ) < 1:
            raise ValueError("video_local integer settings must be positive")
        if config.weight < 0.0:
            raise ValueError("video_local weight must be non-negative")
        if config.source_channel not in {"visual", "organizer_clip"}:
            raise ValueError("video_local source_channel must be visual or organizer_clip")
        return config


def _query_matrix(query_vectors: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(query_vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("video-local query vectors must be a non-empty matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("video-local query vectors must be finite and non-zero")
    return matrix / norms[:, None]


def shot_bounded_gaussian_support(
    scores: np.ndarray,
    timestamps_ms: np.ndarray,
    shot_codes: np.ndarray,
    *,
    window_ms: int,
    sigma_ms: int,
) -> np.ndarray:
    """Aggregate positive nearby support without crossing a shot boundary."""

    values = np.asarray(scores, dtype=np.float32)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    shots = np.asarray(shot_codes, dtype=np.int64)
    if values.ndim != 1 or timestamps.shape != values.shape or shots.shape != values.shape:
        raise ValueError("video-local support arrays must be aligned vectors")
    if window_ms < 1 or sigma_ms < 1:
        raise ValueError("video-local support window and sigma must be positive")
    support = np.zeros(values.shape, dtype=np.float32)
    for index in range(values.shape[0]):
        delta = np.abs(timestamps - timestamps[index])
        selected = (shots == shots[index]) & (delta <= window_ms)
        if not np.any(selected):
            continue
        weights = np.exp(-0.5 * (delta[selected] / float(sigma_ms)) ** 2)
        positive = np.maximum(values[selected], 0.0)
        support[index] = float(np.sum(positive * weights) / np.sum(weights))
    return support


def _candidate_video_ids(
    candidates: Sequence[Mapping[str, Any]], limit: int
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            video_id, _frame_idx = canonical_key(candidate)
        except ValueError:
            continue
        if video_id in seen:
            continue
        seen.add(video_id)
        ordered.append(video_id)
        if len(ordered) == limit:
            break
    return tuple(ordered)


def score_video_local(
    *,
    config: VideoLocalConfig,
    candidates: Sequence[Mapping[str, Any]],
    query_vectors: Sequence[np.ndarray] | np.ndarray,
    trajectories: Any,
    batch_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score all cached frames in top videos using MaxSim and local support."""

    diagnostics: dict[str, Any] = {
        "mode": config.mode,
        "source_channel": config.source_channel,
        "candidate_video_count": 0,
        "scored_video_count": 0,
        "result_count": 0,
        "applied": False,
    }
    snapshot = getattr(trajectories, "snapshot", None)
    if callable(snapshot):
        cache = snapshot()
        if isinstance(cache, Mapping):
            diagnostics.update(
                {
                    "cache_state": cache.get("state"),
                    "cache_fingerprint": cache.get("fingerprint"),
                    "cache_video_count": int(cache.get("video_count", 0) or 0),
                    "cache_frame_count": int(cache.get("frame_count", 0) or 0),
                    "cache_source": cache.get("source"),
                }
            )
    if config.mode == "off" or not candidates:
        return [], diagnostics
    query = _query_matrix(query_vectors)
    batch = trajectories.batch(batch_id)
    if batch is None:
        diagnostics["reason_code"] = "trajectory_cache_not_ready"
        return [], diagnostics
    video_ids = _candidate_video_ids(candidates, config.candidate_videos)
    diagnostics["candidate_video_count"] = len(video_ids)
    results: list[dict[str, Any]] = []
    incompatible_trajectories = 0
    for video_id in video_ids:
        trajectory = batch.video(video_id)
        if trajectory is None or len(trajectory) == 0:
            continue
        vectors = np.asarray(trajectory.vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != query.shape[1]:
            incompatible_trajectories += 1
            continue
        similarities = query @ vectors.T
        maxsim = similarities.max(axis=0)
        cue_coverage = (similarities > 0.0).mean(axis=0)
        raw_shots = getattr(trajectory, "shot_codes", None)
        if raw_shots is None:
            # Unknown shots cannot safely support a neighbouring frame.
            shots = -np.arange(1, len(trajectory) + 1, dtype=np.int64)
        else:
            shots = np.asarray(raw_shots, dtype=np.int64)
        support = shot_bounded_gaussian_support(
            maxsim,
            trajectory.timestamps_ms,
            shots,
            window_ms=config.support_window_ms,
            sigma_ms=config.gaussian_sigma_ms,
        )
        local_score = maxsim + 0.15 * cue_coverage + 0.20 * support
        order = np.lexsort((trajectory.source_frame_indices, -local_score))[
            : config.frames_per_video
        ]
        for index in order:
            results.append(
                {
                    "id": f"{batch_id}:{video_id}:{int(trajectory.source_frame_indices[index])}",
                    "score": float(local_score[index]),
                    "payload": {
                        "batch_id": batch_id,
                        "video_id": video_id,
                        "source_frame_idx": int(trajectory.source_frame_indices[index]),
                        "timestamp_ms": int(trajectory.timestamps_ms[index]),
                        "shot_code": int(shots[index]),
                        "video_local_maxsim": float(maxsim[index]),
                        "video_local_cue_coverage": float(cue_coverage[index]),
                        "video_local_gaussian_support": float(support[index]),
                    },
                }
            )
        diagnostics["scored_video_count"] += 1
    results.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["payload"]["video_id"]),
            int(item["payload"]["source_frame_idx"]),
        )
    )
    diagnostics["result_count"] = len(results)
    diagnostics["applied"] = config.mode == "active" and bool(results)
    diagnostics["reason_code"] = (
        "query_completed"
        if results
        else "vector_space_mismatch"
        if incompatible_trajectories
        else "candidate_trajectories_unavailable"
    )
    return results, diagnostics


__all__ = [
    "VideoLocalConfig",
    "score_video_local",
    "shot_bounded_gaussian_support",
]
