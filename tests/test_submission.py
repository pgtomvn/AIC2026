"""Unit tests for strict Textual KIS submission export."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts import CompetitionSpec, SearchHit
from src.submission import (
    FrameIndexEvidence,
    SubmissionValidationError,
    export_submission,
    validate_submission,
)


def _hit(rank: int, video_id: str = "L21_V001", frame: int = 181) -> SearchHit:
    return SearchHit(
        rank=rank,
        video_id=video_id,
        source_frame_idx=frame,
        timestamp_ms=3000,
        shot_id=f"shot-{rank}",
        fused_score=1 / rank,
        thumbnail_uri=f"thumbs/{video_id}/{frame}.webp",
    )


def _active(batch_id: str) -> bool:
    return batch_id == "aic25-b1"


def _persisted(_batch_id: str, _video_id: str, _source_frame_idx: int) -> bool:
    return True


def test_export_validates_normalizes_and_writes_rank_order(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "submission.csv"

    written = export_submission(
        [_hit(2, "L21_V002.mp4", 250), _hit(1, "l21_v001", 181)],
        destination,
        batch_id="aic25-b1",
        is_batch_active=_active,
        is_canonical_source_frame=_persisted,
    )

    assert written == destination
    assert destination.read_text(encoding="utf-8") == "L21_V001,181\nL21_V002,250\n"
    assert not list(destination.parent.glob("*.tmp"))


def test_validation_rejects_inactive_batch_without_creating_output(tmp_path: Path) -> None:
    destination = tmp_path / "submission.csv"

    with pytest.raises(SubmissionValidationError, match="not active"):
        export_submission(
            [_hit(1)],
            destination,
            batch_id="aic25-b1",
            is_batch_active=lambda _: False,
            is_canonical_source_frame=_persisted,
        )

    assert not destination.exists()


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([_hit(1), _hit(2)], "duplicate result"),
        ([_hit(1, "../../bad")], "path"),
        ([{"video_id": "L21_V001", "keyframe_ordinal": 7}], "keyframe_ordinal"),
        ([_hit(2)], "contiguous"),
        ([], "at least one"),
    ],
)
def test_validation_rejects_malformed_duplicate_or_noncontiguous_results(
    items, message: str
) -> None:
    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(
            items,
            batch_id="aic25-b1",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
        )


def test_validation_rejects_more_than_top_100() -> None:
    items = [
        _hit(rank, f"L{21 + (rank - 1) // 99:02d}_V{(rank - 1) % 99 + 1:03d}", rank)
        for rank in range(1, 102)
    ]

    with pytest.raises(SubmissionValidationError, match="maximum is 100"):
        validate_submission(
            items,
            batch_id="aic25-b1",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
        )


def test_frame_map_evidence_rejects_known_jpeg_ordinal() -> None:
    evidence = FrameIndexEvidence({"L21_V001": {1: 0, 2: 60, 3: 120}})

    with pytest.raises(SubmissionValidationError, match="keyframe_ordinal"):
        validate_submission(
            [
                {
                    "rank": 1,
                    "video_id": "L21_V001",
                    "source_frame_idx": 2,
                    "keyframe_ordinal": 2,
                }
            ],
            batch_id="aic25-b1",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
            frame_index_evidence=evidence,
        )

    assert (
        validate_submission(
            [_hit(1, frame=60)],
            batch_id="aic25-b1",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
            frame_index_evidence=evidence,
        )[0].source_frame_idx
        == 60
    )
    # A raw-refined source ordinal may numerically collide with an organizer
    # keyframe ordinal. Explicit coarse provenance prevents a false rejection.
    refined = validate_submission(
        [
            {
                "rank": 1,
                "video_id": "L21_V001",
                "source_frame_idx": 2,
                "keyframe_ordinal": 2,
                "coarse_source_frame_idx": 60,
            }
        ],
        batch_id="aic25-b1",
        is_batch_active=_active,
        is_canonical_source_frame=_persisted,
        frame_index_evidence=evidence,
    )
    assert refined[0].source_frame_idx == 2


def test_frame_evidence_video_ids_use_competition_normalization() -> None:
    evidence = FrameIndexEvidence({"l21_v001.mp4": {2: 60}})

    with pytest.raises(SubmissionValidationError, match="keyframe_ordinal"):
        validate_submission(
            [
                {
                    "video_id": "L21_V001",
                    "source_frame_idx": 2,
                    "keyframe_ordinal": 2,
                }
            ],
            batch_id="aic25-b1",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
            frame_index_evidence=evidence,
        )


def test_custom_spec_can_emit_header_in_declared_column_order(tmp_path: Path) -> None:
    spec = CompetitionSpec(
        output_columns=("frame_idx", "video_name"),
        include_header=True,
    )
    destination = tmp_path / "submission.csv"

    export_submission(
        [("L21_V001", 42)],
        destination,
        spec=spec,
        batch_id="aic25-b1",
        is_batch_active=_active,
        is_canonical_source_frame=_persisted,
    )

    assert destination.read_text(encoding="utf-8") == ("frame_idx,video_name\n42,L21_V001\n")


def test_validation_rejects_missing_and_mixed_persisted_source_frames() -> None:
    with pytest.raises(SubmissionValidationError, match="not a canonical source frame"):
        validate_submission(
            [{"rank": 1, "video_id": "L21_V001", "source_frame_idx": 999999}],
            batch_id="aic25-b1",
            is_batch_active=_active,
            is_canonical_source_frame=lambda _batch, _video, _frame: False,
        )
    with pytest.raises(SubmissionValidationError, match="mix batches"):
        validate_submission(
            [
                {
                    "rank": 1,
                    "batch_id": "different-batch",
                    "video_id": "L21_V001",
                    "source_frame_idx": 181,
                }
            ],
            batch_id="aic25-b1",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
        )
