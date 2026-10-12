"""Manual KIS temporal branch using the existing warm Organizer-CLIP cache.

Applies existing TRAKE alignment, not an unverified WESp author formula.
Never warms/reloads the cache or mixes vector spaces in a request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.retrieval.interactive import unit_vector
from src.retrieval.trake_alignment import strict_monotonic_alignment


def score_sequence(
    candidates: Sequence[Mapping[str, Any]], *, cache: Any, batch_id: str,
    step_vectors: Sequence[np.ndarray], global_vector: np.ndarray, target: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one selected-step anchor per video, with strict ordered evidence."""
    diagnostics: dict[str, Any] = {
        "mode": "manual", "source_channel": "organizer_clip", "applied": False,
        "candidate_video_count": 0, "scored_video_count": 0, "result_count": 0,
    }
    if not 2 <= len(step_vectors) <= 6 or not 0 <= target < len(step_vectors):
        raise ValueError("invalid_temporal_steps")
    batch = cache.batch(batch_id) if cache is not None else None
    if batch is None:
        diagnostics["reason_code"] = "trajectory_cache_not_ready"
        return [], diagnostics
    diagnostics["cache_fingerprint"] = str(getattr(batch, "fingerprint", ""))
    query = unit_vector(global_vector)
    steps = np.stack([unit_vector(v, dimension=query.size) for v in step_vectors])
    videos = list(dict.fromkeys(
        str(item["video_id"]) for item in candidates if item.get("batch_id") == batch_id
    ))[:12]
    diagnostics["candidate_video_count"] = len(videos)
    results: list[dict[str, Any]] = []
    for video_id in videos:
        trajectory = batch.video(video_id)
        if trajectory is None:
            continue
        try:
            vectors = np.asarray(trajectory.vectors, dtype=np.float32)
            frames = trajectory.source_frame_indices
            times = trajectory.timestamps_ms
            if vectors.ndim != 2 or vectors.shape[1] != query.size:
                continue
            if not np.isfinite(vectors).all() or len(times) != len(vectors):
                continue
            if any(a >= b for a, b in zip(times, times[1:])):
                continue  # unknown/duplicate timestamps cannot prove before-after
            codes = getattr(trajectory, "shot_codes", None)
            shots = None if codes is None else [str(v) if v >= 0 else None for v in codes]
            alignment = strict_monotonic_alignment(
                steps @ vectors.T, frames, timestamps_ms=times, shot_ids=shots,
                max_gap_ms=120_000, gap_penalty_per_second=.001, shot_boundary_penalty=.02,
            )
            if alignment is None:
                continue
            positions, sequence_score = alignment
            index = positions[target]
            evidence = [
                {"step": step, "source_frame_idx": int(frames[pos]),
                 "timestamp_ms": int(times[pos])}
                for step, pos in enumerate(positions)
            ]
            results.append({
                "batch_id": batch_id, "video_id": video_id,
                "source_frame_idx": int(frames[index]), "timestamp_ms": int(times[index]),
                "score": .25 * float((vectors @ query).max()) + .75 * sequence_score,
                "temporal_steps": evidence,
            })
            diagnostics["scored_video_count"] += 1
        except (ValueError, TypeError, IndexError):
            continue
    results.sort(key=lambda row: (-row["score"], row["video_id"], row["source_frame_idx"]))
    diagnostics.update(result_count=len(results), applied=bool(results), reason_code=(
        "query_completed" if results else "valid_trajectory_unavailable"
    ))
    return results, diagnostics
