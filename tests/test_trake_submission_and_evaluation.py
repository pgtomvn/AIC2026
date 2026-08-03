"""Offline tests for TRAKE export and provisional evaluation."""

from __future__ import annotations

import csv

import pytest

from src.contracts import TRAKEResponseItem, TRAKEStageMatch
from src.evaluation.trake import (
    TRAKEGroundTruthQuery,
    TRAKEStageInterval,
    evaluate_trake_queries,
    evaluate_trake_query,
    trake_alignment_score,
)
from src.trake_submission import (
    TRAKEFrameIndexEvidence,
    TRAKESubmissionValidationError,
    export_trake_submission,
    validate_trake_submission,
)


def _item(
    *,
    rank: int = 1,
    video_id: str = "L21_V001",
    frames: tuple[int, int] = (10, 20),
) -> TRAKEResponseItem:
    return TRAKEResponseItem(
        rank=rank,
        batch_id="aic25-b1-reference",
        video_id=video_id,
        stage_matches=tuple(
            TRAKEStageMatch(
                stage_index=index,
                source_frame_idx=frame,
                timestamp_ms=frame * 40,
                similarity=0.9,
            )
            for index, frame in enumerate(frames, start=1)
        ),
        confidence_score=0.9,
    )


def _ground_truth() -> TRAKEGroundTruthQuery:
    return TRAKEGroundTruthQuery(
        query_id="q-1",
        video_id="L21_V001",
        stage_intervals=(
            TRAKEStageInterval(1, 8, 12),
            TRAKEStageInterval(2, 18, 22),
        ),
    )


def test_export_writes_atomic_headerless_rows_and_required_directory_name(tmp_path) -> None:
    target = export_trake_submission(
        (_item(), _item(rank=2, video_id="L21_V002", frames=(30, 40))),
        tmp_path,
        query_id="query 01",
        stage_count=2,
        batch_id="aic25-b1-reference",
        is_batch_active=lambda batch: batch == "aic25-b1-reference",
        is_canonical_source_frame=lambda _batch, _video, frame: frame % 10 == 0,
    )

    assert target.name == "query-01-trake-submission.csv"
    with target.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["L21_V001", "10", "20"],
            ["L21_V002", "30", "40"],
        ]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (
            (
                {
                    "rank": 1,
                    "batch_id": "aic25-b1-reference",
                    "video_id": "L21_V001",
                    "keyframe_ordinals": [1, 2],
                },
            ),
            "keyframe_ordinal",
        ),
        ((_item(rank=2),), "contiguous"),
        (
            (
                _item(),
                _item(rank=2),
            ),
            "one row per video",
        ),
    ],
)
def test_submission_rejects_invalid_identity_and_ranking(items, message: str) -> None:
    with pytest.raises(TRAKESubmissionValidationError, match=message):
        validate_trake_submission(items, stage_count=2)


def test_submission_rejects_noncanonical_and_ordinal_only_frames() -> None:
    with pytest.raises(TRAKESubmissionValidationError, match="canonical"):
        validate_trake_submission(
            (_item(),),
            stage_count=2,
            is_canonical_source_frame=lambda _batch, _video, frame: frame != 20,
        )
    with pytest.raises(TRAKESubmissionValidationError, match="keyframe_ordinal"):
        validate_trake_submission(
            (_item(frames=(1, 20)),),
            stage_count=2,
            frame_index_evidence=TRAKEFrameIndexEvidence(
                {"L21_V001": {1: 10, 2: 20}}
            ),
        )


def _mapping_item(
    *,
    stage_indexes: list[object],
    timestamps_ms: list[object] | None = None,
) -> dict[str, object]:
    timestamps = timestamps_ms or [index * 100 for index in range(1, len(stage_indexes) + 1)]
    return {
        "rank": 1,
        "batch_id": "aic25-b1-reference",
        "video_id": "L21_V001",
        "stage_matches": [
            {
                "stage_index": stage_index,
                "source_frame_idx": position * 10,
                "timestamp_ms": timestamp,
            }
            for position, (stage_index, timestamp) in enumerate(
                zip(stage_indexes, timestamps, strict=True),
                start=1,
            )
        ],
    }


@pytest.mark.parametrize("stage_count", range(2, 7))
@pytest.mark.parametrize(
    "malformation",
    ["missing", "boolean", "zero", "gap", "duplicate", "reversed"],
)
def test_mapping_stage_matches_require_strict_contiguous_indices(
    stage_count: int,
    malformation: str,
) -> None:
    indexes: list[object] = list(range(1, stage_count + 1))
    if malformation == "boolean":
        indexes[0] = True
    elif malformation == "zero":
        indexes[0] = 0
    elif malformation == "gap":
        indexes[1] = stage_count + 1
    elif malformation == "duplicate":
        indexes[1] = 1
    elif malformation == "reversed":
        indexes.reverse()
    item = _mapping_item(stage_indexes=indexes)
    if malformation == "missing":
        del item["stage_matches"][0]["stage_index"]  # type: ignore[index]

    with pytest.raises(TRAKESubmissionValidationError, match="stage_index"):
        validate_trake_submission((item,), stage_count=stage_count)


@pytest.mark.parametrize("stage_count", range(2, 7))
@pytest.mark.parametrize("chronology", ["equal", "decreasing"])
def test_mapping_stage_matches_require_strict_timestamps(
    stage_count: int,
    chronology: str,
) -> None:
    timestamps: list[object] = (
        [100] * stage_count
        if chronology == "equal"
        else list(range(stage_count * 100, 0, -100))
    )
    with pytest.raises(TRAKESubmissionValidationError, match="strictly increasing"):
        validate_trake_submission(
            (
                _mapping_item(
                    stage_indexes=list(range(1, stage_count + 1)),
                    timestamps_ms=timestamps,
                ),
            ),
            stage_count=stage_count,
        )


@pytest.mark.parametrize("malformation", ["missing", "boolean", "string", "negative"])
def test_mapping_stage_matches_require_strict_timestamp_values(
    malformation: str,
) -> None:
    item = _mapping_item(stage_indexes=[1, 2], timestamps_ms=[100, 200])
    matches = item["stage_matches"]
    assert isinstance(matches, list)
    first = matches[0]
    assert isinstance(first, dict)
    if malformation == "missing":
        del first["timestamp_ms"]
    elif malformation == "boolean":
        first["timestamp_ms"] = True
    elif malformation == "string":
        first["timestamp_ms"] = "100"
    else:
        first["timestamp_ms"] = -1

    with pytest.raises(TRAKESubmissionValidationError, match="timestamp_ms"):
        validate_trake_submission((item,), stage_count=2)


def test_direct_frame_ids_use_canonical_timestamp_lookup() -> None:
    item = {
        "rank": 1,
        "batch_id": "aic25-b1-reference",
        "video_id": "L21_V001",
        "frame_ids": [10, 20],
    }

    with pytest.raises(TRAKESubmissionValidationError, match="timestamp lookup"):
        validate_trake_submission((item,), stage_count=2)
    with pytest.raises(TRAKESubmissionValidationError, match="strictly increasing"):
        validate_trake_submission(
            (item,),
            stage_count=2,
            get_canonical_timestamp_ms=lambda _batch, _video, frame: {
                10: 100,
                20: 100,
            }.get(frame),
        )

    rows = validate_trake_submission(
        (item,),
        stage_count=2,
        get_canonical_timestamp_ms=lambda _batch, _video, frame: {
            10: 100,
            20: 200,
        }.get(frame),
    )
    assert rows[0].source_frame_indices == (10, 20)


def test_mapping_stage_timestamps_are_authoritative_after_strict_validation() -> None:
    item = _mapping_item(stage_indexes=[1, 2], timestamps_ms=[100, 200])

    rows = validate_trake_submission(
        (item,),
        stage_count=2,
        get_canonical_timestamp_ms=lambda _batch, _video, _frame: None,
    )

    assert rows[0].source_frame_indices == (10, 20)


def test_invalid_stage_matches_cannot_fall_back_to_valid_frame_ids() -> None:
    item = _mapping_item(stage_indexes=[2, 1], timestamps_ms=[100, 200])
    item["frame_ids"] = [10, 20]

    with pytest.raises(TRAKESubmissionValidationError, match="stage_index"):
        validate_trake_submission(
            (item,),
            stage_count=2,
            get_canonical_timestamp_ms=lambda _batch, _video, frame: frame * 10,
        )


def test_wrong_video_scores_zero_and_correct_video_receives_partial_credit() -> None:
    ground_truth = _ground_truth()

    assert trake_alignment_score(
        {"video_id": "L21_V999", "frame_ids": [10, 20]}, ground_truth
    ) == 0.0
    assert trake_alignment_score(
        {"video_id": "L21_V001", "frame_ids": [10, 99]}, ground_truth
    ) == 0.5


def test_query_evaluator_uses_best_score_at_each_cutoff() -> None:
    predictions = [
        {"video_id": "L21_V999", "frame_ids": [1, 2]},
        {"video_id": "L21_V001", "frame_ids": [10, 99]},
        {"video_id": "L21_V001", "frame_ids": [10, 20]},
    ]

    metrics = evaluate_trake_query(predictions, _ground_truth(), latency_ms=25.0)

    assert metrics.best_score_at[1] == 0.0
    assert metrics.best_score_at[5] == 1.0
    assert metrics.mean_top_k_r_score == pytest.approx(0.8)
    assert metrics.video_recall == 1.0
    assert metrics.stage_accuracy == 1.0
    assert metrics.exact_alignment_recall == 1.0
    assert metrics.first_video_rank == 2
    assert metrics.reciprocal_rank == 0.5


def test_aggregate_evaluator_reports_retrieval_alignment_and_latency() -> None:
    report = evaluate_trake_queries(
        {
            "q-1": [{"video_id": "L21_V001", "frame_ids": [10, 20]}],
            "q-2": [],
        },
        (
            _ground_truth(),
            TRAKEGroundTruthQuery(
                query_id="q-2",
                video_id="L21_V002",
                stage_intervals=(
                    TRAKEStageInterval(1, 1, 5),
                    TRAKEStageInterval(2, 6, 10),
                ),
            ),
        ),
        latencies_ms={"q-1": 10.0, "q-2": 30.0},
    )

    assert report.query_count == 2
    assert report.mean_top_k_r_score == 0.5
    assert report.video_recall == 0.5
    assert report.stage_accuracy == 0.5
    assert report.exact_alignment_recall == 0.5
    assert report.mrr == 0.5
    assert report.latency_ms["mean"] == 20.0
