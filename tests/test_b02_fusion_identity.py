"""B02 ordinal identity must survive ranking without changing B01 output."""

from __future__ import annotations

import pytest

from src.retrieval.fusion import (
    ChannelRanking,
    aggregate_temporal_windows,
    canonical_key,
    diversify,
    frame_identity_key,
    weighted_rrf,
)


def _b02(ordinal: int, *, score: float = 0.8, batch: str = "aic26-b2") -> dict:
    return {
        "score": score,
        "payload": {
            "batch_id": batch,
            "video_id": "M08_V001",
            "keyframe_ordinal": ordinal,
            "source_frame_idx": 30,
            "timestamp_ms": 1_000,
            "shot_id": "shot-1",
        },
    }


def test_b02_rrf_preserves_distinct_ordinals_with_shared_source_frame_and_pts() -> None:
    visual = ChannelRanking("visual", (_b02(1), _b02(2), _b02(1, score=0.1)))
    metadata = ChannelRanking("metadata", (_b02(2),))

    fused = weighted_rrf((visual, metadata), k=0)

    assert [item["keyframe_ordinal"] for item in fused] == [2, 1]
    assert [item["source_frame_idx"] for item in fused] == [30, 30]
    assert fused[0]["channel_ranks"] == {"visual": 2, "metadata": 1}
    assert fused[1]["channel_ranks"] == {"visual": 1}
    assert canonical_key(fused[0]) == ("M08_V001", 30)
    assert frame_identity_key(fused[0]) == ("aic26-b2", "M08_V001", 2)


def test_b02_equal_time_images_survive_window_aggregation_and_diversification() -> None:
    candidates = [
        {**_b02(1)["payload"], "score": 0.9},
        {**_b02(2)["payload"], "score": 0.8},
    ]

    aggregated = aggregate_temporal_windows(candidates)
    selected = diversify(aggregated)

    assert {item["keyframe_ordinal"] for item in aggregated} == {1, 2}
    assert {item["keyframe_ordinal"] for item in selected} == {1, 2}
    assert [item["rank"] for item in selected] == [1, 2]


def test_b02_requires_positive_ordinal_instead_of_falling_back_to_frame() -> None:
    for invalid in (None, 0, True, 1.5, "1.5"):
        result = _b02(1)
        result["payload"]["keyframe_ordinal"] = invalid
        with pytest.raises(ValueError, match="keyframe_ordinal"):
            frame_identity_key(result)


def test_legacy_batch_still_deduplicates_on_source_frame_with_any_ordinal() -> None:
    first = _b02(1, batch="aic25-b1-reference")
    second = _b02(2, batch="aic25-b1-reference")

    fused = weighted_rrf((ChannelRanking("visual", (first, second)),))

    assert len(fused) == 1
    assert frame_identity_key(fused[0]) == ("M08_V001", 30)
