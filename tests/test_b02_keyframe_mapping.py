"""Batch 2 PTS and VFR keyframe contract, independent of production stores."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.ingest.b02_keyframe_mapping import (
    B02KeyframeIdentity,
    is_vfr_traffic_video,
    parse_b02_keyframe_row,
    pts_time_to_ms,
    trake_eligible,
)


@pytest.mark.parametrize(
    ("pts", "expected_ms"),
    [
        ("0", 0),
        ("2.0", 2000),
        ("4.080", 4080),
        (Decimal("1.2345"), 1234),
        ("1.2355", 1236),
        ("1.234567", 1235),
    ],
)
def test_pts_time_to_ms_uses_exact_pts(pts: str | Decimal, expected_ms: int) -> None:
    assert pts_time_to_ms(pts) == expected_ms


@pytest.mark.parametrize("pts", ["", "NaN", "Infinity", "-0.01", True, None])
def test_pts_time_to_ms_rejects_invalid_values(pts: object) -> None:
    with pytest.raises(ValueError, match="pts_time"):
        pts_time_to_ms(pts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("video_id", "expected"),
    [
        ("N001-V001", True),
        ("N100-V099", True),
        ("N001", True),
        ("N000-V001", False),
        ("N101-V001", False),
        ("N001V001", False),
        ("M05_V013", False),
        ("S01-V001", False),
    ],
)
def test_vfr_exception_and_trake_exclusion(video_id: str, expected: bool) -> None:
    assert is_vfr_traffic_video(video_id) is expected
    assert trake_eligible(video_id) is (not expected)


def test_relative_frame_idx_is_not_identity_or_submission_time() -> None:
    first = parse_b02_keyframe_row(
        "aic26-b2", "N001-V001", {"n": "1", "frame_idx": "50", "pts_time": "2.000", "fps": "25"}
    )
    second = parse_b02_keyframe_row(
        "aic26-b2", "N001-V001", {"n": "2", "frame_idx": "50", "pts_time": "2.520", "fps": "60"}
    )
    assert first.identity.key == ("aic26-b2", "N001-V001", 1)
    assert second.identity.key == ("aic26-b2", "N001-V001", 2)
    assert first.identity != second.identity
    assert first.organizer_frame_idx == second.organizer_frame_idx == 50
    assert (first.timestamp_ms, second.timestamp_ms) == (2000, 2520)
    assert first.is_vfr_traffic_video and not first.trake_eligible


def test_batch_and_video_are_part_of_identity_and_m_video_is_eligible() -> None:
    row = {"n": "1", "frame_idx": "0", "pts_time": "0.0"}
    m_frame = parse_b02_keyframe_row("aic26-b2", "M05_V013", row)
    another_batch = parse_b02_keyframe_row("another-batch", "M05_V013", row)
    another_video = parse_b02_keyframe_row("aic26-b2", "M05_V014", row)
    assert len({m_frame.identity, another_batch.identity, another_video.identity}) == 3
    assert m_frame.timestamp_ms == 0
    assert m_frame.trake_eligible


@pytest.mark.parametrize(
    "row",
    [
        {"n": "0", "frame_idx": "0", "pts_time": "0"},
        {"n": "1.0", "frame_idx": "0", "pts_time": "0"},
        {"n": "1", "frame_idx": "-1", "pts_time": "0"},
        {"n": "1", "frame_idx": "0"},
        {"n": "1", "pts_time": "0"},
    ],
)
def test_map_row_rejects_invalid_identity_or_time(row: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        parse_b02_keyframe_row("aic26-b2", "N001-V001", row)


def test_identity_rejects_ambiguous_batch_video_or_ordinal() -> None:
    with pytest.raises(ValueError, match="batch_id"):
        B02KeyframeIdentity("", "M05_V013", 1)
    with pytest.raises(ValueError, match="video_id"):
        B02KeyframeIdentity("aic26-b2", " M05_V013 ", 1)
    with pytest.raises(ValueError, match="keyframe_ordinal"):
        B02KeyframeIdentity("aic26-b2", "M05_V013", 0)
