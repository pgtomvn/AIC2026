"""Batch 2 keyframe identity and presentation-time conversion.

Organizer ``frame_idx`` values for N001–N100 traffic-camera videos are
relative, not exact source-frame positions.  Keep them as provenance only:
keyframes are identified by their one-based ordinal, and KIS/Q&A submission
time is derived exclusively from ``pts_time``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Mapping

_TRAFFIC_VIDEO_PATTERN = re.compile(r"^N(?P<camera>\d{3})(?:[-_]|$)")


def pts_time_to_ms(value: str | int | float | Decimal) -> int:
    """Convert organizer PTS seconds to integer milliseconds without using FPS.

    Decimal arithmetic preserves the CSV value exactly.  Half-millisecond ties
    use Python's usual round-to-even policy, matching the existing B01 mapper.
    """

    if isinstance(value, bool):
        raise ValueError("pts_time must be a finite nonnegative number")
    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("pts_time must be a finite nonnegative number") from exc
    if not seconds.is_finite() or seconds < 0:
        raise ValueError("pts_time must be a finite nonnegative number")
    return int((seconds * 1000).to_integral_value(rounding=ROUND_HALF_EVEN))


def is_vfr_traffic_video(video_id: str) -> bool:
    """Whether the organizer's N001–N100 VFR exception applies to a video."""

    match = _TRAFFIC_VIDEO_PATTERN.match(video_id)
    return match is not None and 1 <= int(match.group("camera")) <= 100


def trake_eligible(video_id: str) -> bool:
    """Exclude N001–N100 traffic-camera videos from Batch 2 TRAKE."""

    return not is_vfr_traffic_video(video_id)


@dataclass(frozen=True, slots=True)
class B02KeyframeIdentity:
    """Stable Batch 2 keyframe identity independent of relative ``frame_idx``."""

    batch_id: str
    video_id: str
    keyframe_ordinal: int

    def __post_init__(self) -> None:
        for name in ("batch_id", "video_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty, trimmed string")
        if (
            isinstance(self.keyframe_ordinal, bool)
            or not isinstance(self.keyframe_ordinal, int)
            or self.keyframe_ordinal < 1
        ):
            raise ValueError("keyframe_ordinal must be a positive integer")

    @property
    def key(self) -> tuple[str, str, int]:
        """Tuple suitable for in-memory deduplication and ordered manifests."""

        return self.batch_id, self.video_id, self.keyframe_ordinal


@dataclass(frozen=True, slots=True)
class B02MappedKeyframe:
    """Organizer map row with PTS-derived time and advisory frame index."""

    identity: B02KeyframeIdentity
    timestamp_ms: int
    organizer_frame_idx: int

    @property
    def is_vfr_traffic_video(self) -> bool:
        return is_vfr_traffic_video(self.identity.video_id)

    @property
    def trake_eligible(self) -> bool:
        return trake_eligible(self.identity.video_id)


def _nonnegative_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return parsed


def parse_b02_keyframe_row(
    batch_id: str, video_id: str, row: Mapping[str, object]
) -> B02MappedKeyframe:
    """Parse one CSV map row without assuming a constant frame rate.

    Repeated ``frame_idx`` values are intentionally accepted: on N-series
    videos they do not define identity or submitted time.
    """

    try:
        ordinal = _nonnegative_integer(row["n"], "n", minimum=1)
        organizer_frame_idx = _nonnegative_integer(row["frame_idx"], "frame_idx")
        timestamp_ms = pts_time_to_ms(row["pts_time"])
    except KeyError as exc:
        raise ValueError(f"missing organizer map column: {exc.args[0]}") from exc
    return B02MappedKeyframe(
        identity=B02KeyframeIdentity(batch_id, video_id, ordinal),
        timestamp_ms=timestamp_ms,
        organizer_frame_idx=organizer_frame_idx,
    )
