"""CPU-only contract tests for TRAKE."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import (
    TRAKECompetitionSpec,
    TRAKEQueryRequest,
    TRAKEResponse,
    TRAKEResponseItem,
    TRAKEStage,
    TRAKEStageMatch,
)


def _stages() -> tuple[TRAKEStage, ...]:
    return (
        TRAKEStage(stage_index=1, description="Một người bước vào phòng"),
        TRAKEStage(stage_index=2, description="Người đó ngồi xuống"),
    )


def _matches() -> tuple[TRAKEStageMatch, ...]:
    return (
        TRAKEStageMatch(
            stage_index=1,
            source_frame_idx=10,
            timestamp_ms=400,
            similarity=0.8,
        ),
        TRAKEStageMatch(
            stage_index=2,
            source_frame_idx=20,
            timestamp_ms=800,
            similarity=0.9,
        ),
    )


def test_request_resolves_one_default_batch_and_forbids_extra_fields() -> None:
    request = TRAKEQueryRequest(
        query_id="q-1",
        global_description="Một người đi vào và ngồi xuống",
        stages=_stages(),
    )

    assert request.resolve_batch("aic25-b1-reference") == "aic25-b1-reference"
    with pytest.raises(ValueError, match="exactly one batch"):
        request.resolve_batch()
    with pytest.raises(ValidationError, match="Extra inputs"):
        TRAKEQueryRequest.model_validate(
            {
                **request.model_dump(),
                "unexpected": True,
            }
        )


def test_request_rejects_non_contiguous_stages_and_multiple_batches() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        TRAKEQueryRequest(
            query_id="q-1",
            global_description="event",
            stages=(
                TRAKEStage(stage_index=1, description="one"),
                TRAKEStage(stage_index=3, description="three"),
            ),
        )
    with pytest.raises(ValidationError, match="exactly one batch"):
        TRAKEQueryRequest(
            query_id="q-1",
            global_description="event",
            stages=_stages(),
            batch_filters=("batch-a", "batch-b"),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "query_id": "q",
            "global_description": "event",
            "stages": [
                {"stage_index": True, "description": "one"},
                {"stage_index": 2, "description": "two"},
            ],
        },
        {
            "query_id": "q",
            "global_description": "event",
            "stages": [
                {"stage_index": 1, "description": "one"},
                {"stage_index": 2, "description": "two"},
            ],
            "top_k": True,
        },
    ],
)
def test_request_rejects_boolean_integer_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TRAKEQueryRequest.model_validate(payload)


def test_response_rejects_boolean_frame_fields() -> None:
    with pytest.raises(ValidationError):
        TRAKEStageMatch(
            stage_index=1,
            source_frame_idx=True,
            timestamp_ms=0,
            similarity=0.5,
        )


def test_response_uses_frame_ids_alias_and_validates_chronology() -> None:
    item = TRAKEResponseItem(
        rank=1,
        batch_id="aic25-b1-reference",
        video_id="L21_V001",
        stage_matches=_matches(),
        confidence_score=0.95,
    )
    response = TRAKEResponse(query_id="q-1", results=(item,), channel_statuses={})

    payload = response.model_dump(mode="json", by_alias=True)
    assert payload["results"][0]["frame_ids"] == [10, 20]
    assert "source_frame_indices" not in payload["results"][0]

    with pytest.raises(ValidationError, match="strictly increasing"):
        TRAKEResponseItem(
            rank=1,
            batch_id="batch",
            video_id="L21_V001",
            stage_matches=(
                TRAKEStageMatch(
                    stage_index=1,
                    source_frame_idx=20,
                    timestamp_ms=400,
                    similarity=0.8,
                ),
                TRAKEStageMatch(
                    stage_index=2,
                    source_frame_idx=10,
                    timestamp_ms=800,
                    similarity=0.9,
                ),
            ),
            confidence_score=0.5,
        )


@pytest.mark.parametrize("stage_count", range(2, 7))
@pytest.mark.parametrize("chronology", ["equal", "decreasing"])
def test_response_rejects_non_increasing_timestamps(
    stage_count: int,
    chronology: str,
) -> None:
    timestamps = (
        [400] * stage_count
        if chronology == "equal"
        else list(range(stage_count * 400, 0, -400))
    )
    with pytest.raises(ValidationError, match="timestamp_ms values must be strictly increasing"):
        TRAKEResponseItem(
            rank=1,
            batch_id="batch",
            video_id="L21_V001",
            stage_matches=tuple(
                TRAKEStageMatch(
                    stage_index=index,
                    source_frame_idx=index * 10,
                    timestamp_ms=timestamp_ms,
                    similarity=0.8,
                )
                for index, timestamp_ms in enumerate(timestamps, start=1)
            ),
            confidence_score=0.5,
        )


def test_response_requires_contiguous_ranks_and_unique_videos() -> None:
    base = dict(
        batch_id="batch",
        video_id="L21_V001",
        stage_matches=_matches(),
        confidence_score=0.5,
    )
    with pytest.raises(ValidationError, match="contiguous"):
        TRAKEResponse(
            query_id="q",
            results=(TRAKEResponseItem(rank=2, **base),),
        )
    with pytest.raises(ValidationError, match="one alignment per video"):
        TRAKEResponse(
            query_id="q",
            results=(
                TRAKEResponseItem(rank=1, **base),
                TRAKEResponseItem(rank=2, **base),
            ),
        )


def test_provisional_spec_normalizes_video_and_uses_inclusive_intervals() -> None:
    spec = TRAKECompetitionSpec.historical_2025_provisional()

    assert spec.spec_id == "historical_2025_provisional"
    assert spec.normalize_video_id("l21_v001.mp4") == "L21_V001"
    assert spec.contains_frame(10, 10, 20)
    assert spec.contains_frame(20, 10, 20)
