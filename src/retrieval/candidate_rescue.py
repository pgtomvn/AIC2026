"""Shadow-safe candidate concentration diagnostics and video-first rescue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.retrieval.fusion import ChannelRanking, canonical_key

ROLLOUT_MODES = frozenset({"off", "shadow", "active"})


def validate_rollout_mode(value: object, *, name: str) -> str:
    """Return a normalized rollout mode or fail before request-time work."""

    mode = str(value).strip().casefold()
    if mode not in ROLLOUT_MODES:
        raise ValueError(f"{name}.mode must be one of off, shadow, active")
    return mode


@dataclass(frozen=True, slots=True)
class CandidateConcentration:
    """Concentration statistics for the first ``sample_size`` candidates."""

    sample_count: int
    unique_videos: int
    top_video_share: float
    video_hhi: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "unique_videos": self.unique_videos,
            "top_video_share": round(self.top_video_share, 6),
            "video_hhi": round(self.video_hhi, 6),
        }


@dataclass(frozen=True, slots=True)
class CandidateRescueConfig:
    """Validated adaptive-overfetch and grouped-search policy."""

    mode: str = "off"
    sample_size: int = 50
    min_unique_videos: int = 15
    max_top_video_share: float = 0.25
    max_video_hhi: float = 0.12
    expanded_depth: int = 600
    max_depth: int = 900
    grouped_videos: int = 100
    frames_per_video: int = 2
    overfetch_weight: float = 0.4
    grouped_weight: float = 0.8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CandidateRescueConfig":
        raw = value or {}
        config = cls(
            mode=validate_rollout_mode(raw.get("mode", "off"), name="candidate_rescue"),
            sample_size=int(raw.get("sample_size", 50)),
            min_unique_videos=int(raw.get("min_unique_videos", 15)),
            max_top_video_share=float(raw.get("max_top_video_share", 0.25)),
            max_video_hhi=float(raw.get("max_video_hhi", 0.12)),
            expanded_depth=int(raw.get("expanded_depth", 600)),
            max_depth=int(raw.get("max_depth", 900)),
            grouped_videos=int(raw.get("grouped_videos", 100)),
            frames_per_video=int(raw.get("frames_per_video", 2)),
            overfetch_weight=float(raw.get("overfetch_weight", 0.4)),
            grouped_weight=float(raw.get("grouped_weight", 0.8)),
        )
        if min(
            config.sample_size,
            config.min_unique_videos,
            config.expanded_depth,
            config.max_depth,
            config.grouped_videos,
            config.frames_per_video,
        ) < 1:
            raise ValueError("candidate_rescue integer settings must be positive")
        if config.expanded_depth > config.max_depth:
            raise ValueError("candidate_rescue expanded_depth cannot exceed max_depth")
        if not 0.0 <= config.max_top_video_share <= 1.0:
            raise ValueError("candidate_rescue max_top_video_share must be in [0, 1]")
        if not 0.0 <= config.max_video_hhi <= 1.0:
            raise ValueError("candidate_rescue max_video_hhi must be in [0, 1]")
        if config.overfetch_weight < 0.0 or config.grouped_weight < 0.0:
            raise ValueError("candidate_rescue branch weights must be non-negative")
        return config


@dataclass(frozen=True, slots=True)
class CandidateRescueOutcome:
    diagnostics: Mapping[str, Any]
    rankings: tuple[ChannelRanking, ...]


def candidate_concentration(
    results: Sequence[Mapping[str, Any]], *, sample_size: int = 50
) -> CandidateConcentration:
    """Measure video diversity without depending on channel score scales."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    counts: dict[str, int] = {}
    accepted = 0
    for result in results[:sample_size]:
        try:
            video_id, _frame_idx = canonical_key(result)
        except ValueError:
            continue
        counts[video_id] = counts.get(video_id, 0) + 1
        accepted += 1
    if accepted == 0:
        return CandidateConcentration(0, 0, 0.0, 0.0)
    shares = tuple(count / accepted for count in counts.values())
    return CandidateConcentration(
        sample_count=accepted,
        unique_videos=len(counts),
        top_video_share=max(shares),
        video_hhi=sum(share * share for share in shares),
    )


def should_rescue(
    concentration: CandidateConcentration, config: CandidateRescueConfig
) -> tuple[bool, tuple[str, ...]]:
    """Return a deterministic trigger decision and its individual reasons."""

    reasons: list[str] = []
    if concentration.unique_videos < config.min_unique_videos:
        reasons.append("low_video_diversity")
    if concentration.top_video_share > config.max_top_video_share:
        reasons.append("top_video_concentration")
    if concentration.video_hhi > config.max_video_hhi:
        reasons.append("high_video_hhi")
    return bool(reasons), tuple(reasons)


def run_candidate_rescue(
    *,
    config: CandidateRescueConfig,
    initial_results: Sequence[Mapping[str, Any]],
    store: Any,
    query_vector: np.ndarray,
    filters: Mapping[str, Any] | None,
    channel_name: str,
) -> CandidateRescueOutcome:
    """Run overfetch and grouped search, returning branches only in active mode.

    Shadow mode deliberately performs the same external queries and records the
    same diagnostics, but its returned ranking tuple is empty.
    """

    concentration = candidate_concentration(
        initial_results, sample_size=config.sample_size
    )
    triggered, reasons = should_rescue(concentration, config)
    diagnostics: dict[str, Any] = {
        "mode": config.mode,
        "triggered": triggered,
        "trigger_reasons": list(reasons),
        "concentration": concentration.to_dict(),
        "overfetch_result_count": 0,
        "grouped_result_count": 0,
        "applied": False,
    }
    if config.mode == "off" or not triggered:
        return CandidateRescueOutcome(diagnostics, ())

    vector = np.asarray(query_vector, dtype=np.float32)
    expanded = store.search(
        vector,
        top_k=min(config.expanded_depth, config.max_depth),
        filters=dict(filters) if filters is not None else None,
    )
    grouped_search = getattr(store, "search_grouped", None)
    grouped: Sequence[Mapping[str, Any]] = ()
    if callable(grouped_search):
        grouped = grouped_search(
            vector,
            group_by="video_id",
            top_k_groups=config.grouped_videos,
            group_size=config.frames_per_video,
            filters=dict(filters) if filters is not None else None,
        )
    diagnostics["overfetch_result_count"] = len(expanded)
    diagnostics["grouped_result_count"] = len(grouped)
    if config.mode != "active":
        return CandidateRescueOutcome(diagnostics, ())

    branches: list[ChannelRanking] = []
    if expanded and config.overfetch_weight > 0.0:
        branches.append(
            ChannelRanking(
                f"{channel_name}:adaptive_overfetch",
                tuple(dict(item) for item in expanded),
                config.overfetch_weight,
            )
        )
    if grouped and config.grouped_weight > 0.0:
        branches.append(
            ChannelRanking(
                f"{channel_name}:video_grouped",
                tuple(dict(item) for item in grouped),
                config.grouped_weight,
            )
        )
    diagnostics["applied"] = bool(branches)
    return CandidateRescueOutcome(diagnostics, tuple(branches))


__all__ = [
    "CandidateConcentration",
    "CandidateRescueConfig",
    "CandidateRescueOutcome",
    "ROLLOUT_MODES",
    "candidate_concentration",
    "run_candidate_rescue",
    "should_rescue",
    "validate_rollout_mode",
]
