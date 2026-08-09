from __future__ import annotations

import math

from app.components.result_gallery import result_score


def test_result_score_prefers_public_fused_score() -> None:
    assert result_score(
        {"fused_score": 0.016393, "score": 99.0, "retrieval_score": 42.0}
    ) == 0.016393


def test_result_score_supports_legacy_internal_results() -> None:
    assert result_score({"score": 0.75}) == 0.75
    assert result_score({"retrieval_score": 0.5}) == 0.5


def test_result_score_never_fabricates_zero_for_missing_or_invalid_values() -> None:
    assert result_score({}) is None
    assert result_score({"fused_score": None}) is None
    assert result_score({"fused_score": math.nan}) is None
    assert result_score({"fused_score": True}) is None
