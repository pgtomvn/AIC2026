"""Validate a single manual role-review export without inferring research outcomes."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from typing import Any

ROI_ROLES = (
    "person_left", "person_center", "person_right",
    "shirt_left", "shirt_center", "shirt_right",
)
JUDGMENT_IDS = (
    "center_white", "left_black", "right_black", "three_distinct",
    "seated_between", "instruments", "bookcase", "whole_query",
)
_SOURCE_FIELDS = (
    "query_id", "query_sha256", "video_id", "source_frame_idx", "timestamp_ms",
    "image_sha256",
)
_REVIEW_FIELDS = {
    "schema", "packet_sha256", "source", "status", "demo", "exported_at",
    "reviewer", "start", "end", "intervalUnknown", "rois", "judgments", "notes",
    "interval_seconds",
}


def _mapping(value: Any, keys: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field}: missing or unexpected fields")
    return value


def _text(value: Any, field: str, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        raise ValueError(f"{field}: expected text")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field}: expected lowercase SHA-256")
    return value


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field}: expected finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field}: expected finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field}: expected finite number")
    return result


def _seconds(value: str, field: str) -> float:
    token = value.strip()
    if len(token) > 64 or not re.fullmatch(r"\d+:\d{1,2}(?:\.\d+)?", token, flags=re.ASCII):
        raise ValueError(f"{field}: expected minutes:seconds")
    minutes, seconds = token.split(":")
    second_value = float(seconds)
    if second_value >= 60:
        raise ValueError(f"{field}: seconds must be below 60")
    return _number(int(minutes) * 60 + second_value, field)


def validate_role_review(
    review: Mapping[str, Any], packet: Mapping[str, Any], *, packet_sha256: str,
    video_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Return a detached, descriptive copy of one complete, non-demo annotation.

    This validates interchange and provenance fields, not annotator correctness.
    The importer must hash-check packet/image/video bytes and reject duplicate JSON
    object keys before decoding: duplicates cannot be recovered from a Mapping.
    Notes are opaque text. Boxes describe only the packet's static frame; neither
    exhaustive object coverage nor tracking across the reported interval is implied.
    """
    _mapping(review, _REVIEW_FIELDS, "review")
    if not isinstance(packet, Mapping) or packet.get("schema") != "evibind-role-review-packet-v1":
        raise ValueError("packet schema mismatch")
    if review["schema"] != "evibind-role-review-v1":
        raise ValueError("review schema mismatch")
    if review["status"] != "complete":
        raise ValueError("review is not complete")
    if review["demo"] is not False:
        raise ValueError("demo must be exactly false")
    if type(packet.get("reviewer_count")) is not int or packet["reviewer_count"] != 1:
        raise ValueError("expected single-reviewer packet")
    _digest(packet_sha256, "packet_sha256")
    if review["packet_sha256"] != packet_sha256:
        raise ValueError("packet SHA-256 mismatch")

    # Exact roles prevent silently accepting a packet with duplicate or missing tasks.
    if packet.get("roi_roles") != list(ROI_ROLES):
        raise ValueError("packet ROI roles mismatch")
    predicates = packet.get("predicates")
    if (not isinstance(predicates, list) or len(predicates) != len(JUDGMENT_IDS)
            or any(not isinstance(item, list) or len(item) != 2 for item in predicates)):
        raise ValueError("packet predicates malformed")
    if [item[0] for item in predicates] != list(JUDGMENT_IDS):
        raise ValueError("packet predicate identities mismatch")
    for _, description in predicates:
        _text(description, "packet predicate description")

    source = _mapping(review["source"], set(_SOURCE_FIELDS), "source")
    for field in _SOURCE_FIELDS:
        expected = packet.get(field)
        if field.endswith("sha256"):
            _digest(expected, f"packet.{field}")
        elif field in ("source_frame_idx", "timestamp_ms"):
            if type(expected) is not int or expected < 0:
                raise ValueError(f"packet.{field}: expected nonnegative integer")
        else:
            _text(expected, f"packet.{field}")
        if type(source[field]) is not type(expected) or source[field] != expected:
            raise ValueError(f"source.{field} mismatch")

    _text(review["reviewer"], "reviewer")
    exported_at = _text(review["exported_at"], "exported_at")
    try:
        exported = datetime.fromisoformat(exported_at)
    except ValueError as error:
        raise ValueError("exported_at: expected ISO-8601 timestamp") from error
    if exported.tzinfo is None or exported.utcoffset() is None:
        raise ValueError("exported_at: timezone required")
    for field in ("start", "end", "notes"):
        _text(review[field], field, allow_blank=True)
    if type(review["intervalUnknown"]) is not bool:
        raise ValueError("intervalUnknown must be boolean")
    duration = None
    if video_duration_seconds is not None:
        duration = _number(video_duration_seconds, "video duration")
        if duration <= 0:
            raise ValueError("video duration must be positive")
    interval = review["interval_seconds"]
    if review["intervalUnknown"]:
        if interval is not None:
            raise ValueError("unknown interval must be null")
    else:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("interval_seconds must contain two numbers")
        start, end = (_number(item, "interval_seconds") for item in interval)
        if not 0 <= start < end:
            raise ValueError("interval must have nonnegative start and positive duration")
        for text_field, number in (("start", start), ("end", end)):
            if not math.isclose(_seconds(review[text_field], text_field), number,
                                rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{text_field} does not match interval_seconds")
        if duration is not None and end > duration:
            raise ValueError("interval exceeds video duration")

    rois = _mapping(review["rois"], set(ROI_ROLES), "rois")
    for role, value in rois.items():
        item = _mapping(value, {"box_xyxy_norm", "unobservable"}, f"rois.{role}")
        if type(item["unobservable"]) is not bool:
            raise ValueError(f"{role}: unobservable must be boolean")
        box = item["box_xyxy_norm"]
        if item["unobservable"]:
            if box is not None:
                raise ValueError(f"{role}: unobservable region must have null box")
        else:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"{role}: expected four normalized coordinates")
            x1, y1, x2, y2 = (_number(coordinate, role) for coordinate in box)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"{role}: invalid normalized box")
    judgments = _mapping(review["judgments"], set(JUDGMENT_IDS), "judgments")
    for key, value in judgments.items():
        if not isinstance(value, str) or value not in {"support", "contradiction", "unknown"}:
            raise ValueError(f"{key}: invalid judgment")

    # Copy every mutable container so callers cannot accidentally edit the raw record.
    result = dict(review)
    result["source"] = dict(source)
    result["rois"] = {
        role: {"unobservable": item["unobservable"],
               "box_xyxy_norm": None if item["box_xyxy_norm"] is None
               else list(item["box_xyxy_norm"])}
        for role, item in rois.items()
    }
    result["judgments"] = dict(judgments)
    result["interval_seconds"] = None if interval is None else list(interval)
    result.update({"reviewer_count": 1, "roi_count": len(rois),
                   "judgment_counts": dict(Counter(judgments.values())),
                   "quality_metric_computed": False,
                   "physical_domain_completeness_inferred": False,
                   "temporal_tracking_inferred": False})
    return result
