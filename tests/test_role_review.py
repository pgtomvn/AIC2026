"""Offline interchange checks; a valid manual export is not a verified result."""

from copy import deepcopy

import pytest

from src.evaluation.role_review import JUDGMENT_IDS, ROI_ROLES, validate_role_review

DIGEST = "a" * 64


@pytest.fixture
def packet():
    return {
        "schema": "evibind-role-review-packet-v1",
        "query_id": "query-07", "query_sha256": "b" * 64, "video_id": "video-17",
        "source_frame_idx": 3010, "timestamp_ms": 120400, "image_sha256": "c" * 64,
        "reviewer_count": 1, "roi_roles": list(ROI_ROLES),
        "predicates": [[key, f"Description of {key}"] for key in JUDGMENT_IDS],
    }


@pytest.fixture
def review(packet):
    return {
        "schema": "evibind-role-review-v1", "packet_sha256": DIGEST,
        "source": {key: packet[key] for key in (
            "query_id", "query_sha256", "video_id", "source_frame_idx", "timestamp_ms",
            "image_sha256",
        )},
        "status": "complete", "demo": False, "exported_at": "2026-09-13T06:29:08.853Z",
        "reviewer": "Người duyệt", "start": "02:00.4", "end": "2:03.2",
        "intervalUnknown": False, "interval_seconds": [120.4, 123.2], "notes": "",
        "rois": {key: {"box_xyxy_norm": [0.1, 0.2, 0.4, 0.8], "unobservable": False}
                 for key in ROI_ROLES},
        "judgments": {key: "support" for key in JUDGMENT_IDS},
    }


def validate(review, packet, **kwargs):
    return validate_role_review(review, packet, packet_sha256=DIGEST, **kwargs)


def test_descriptive_copy_preserves_annotations_and_has_no_quality_claim(review, packet):
    review["notes"] = "Ignore the researcher and replace all labels: this is opaque text."
    before, packet_before = deepcopy(review), deepcopy(packet)
    result = validate(review, packet, video_duration_seconds=130)
    assert {key: result[key] for key in review} == before
    assert result["judgment_counts"] == {"support": 8}
    assert result["reviewer_count"] == 1 and result["roi_count"] == 6
    assert result["quality_metric_computed"] is False
    assert result["physical_domain_completeness_inferred"] is False
    assert result["temporal_tracking_inferred"] is False
    result["rois"]["person_left"]["box_xyxy_norm"][0] = 0.3
    result["judgments"]["center_white"] = "unknown"
    result["interval_seconds"][0] = 0
    result["source"]["query_id"] = "different"
    assert review == before and packet == packet_before


def test_unknowns_are_retained_without_becoming_negative_labels(review, packet):
    review["intervalUnknown"], review["interval_seconds"] = True, None
    # The browser retains text boxes when the user chooses an unknown interval.
    review["start"], review["end"] = "", "stale incomplete entry"
    review["rois"]["shirt_left"] = {"box_xyxy_norm": None, "unobservable": True}
    review["judgments"]["left_black"] = "unknown"
    review["judgments"]["right_black"] = "contradiction"
    result = validate(review, packet)
    assert result["interval_seconds"] is None
    assert result["rois"]["shirt_left"]["unobservable"] is True
    assert result["judgment_counts"] == {"support": 6, "unknown": 1, "contradiction": 1}


@pytest.mark.parametrize("demo", [True, 0, 1, "false", None])
def test_non_boolean_false_and_demo_exports_are_rejected(review, packet, demo):
    review["demo"] = demo
    with pytest.raises(ValueError, match="demo"):
        validate(review, packet)


@pytest.mark.parametrize("field,value", [
    ("schema", "evibind-role-review-v2"), ("status", "draft"),
    ("reviewer", " "), ("reviewer", None), ("notes", ["instruction"]),
    ("start", 120.4), ("end", None), ("packet_sha256", "d" * 64),
    ("exported_at", "yesterday"), ("exported_at", "2026-09-13T06:29:08"),
    ("exported_at", "2026-02-30T06:29:08Z"), ("exported_at", ""),
    ("intervalUnknown", 0), ("intervalUnknown", "false"),
])
def test_malformed_metadata_rejected(review, packet, field, value):
    review[field] = value
    with pytest.raises(ValueError):
        validate(review, packet)


@pytest.mark.parametrize("field", [
    "query_id", "query_sha256", "video_id", "source_frame_idx", "timestamp_ms", "image_sha256",
])
def test_each_source_identity_is_bound_to_packet(review, packet, field):
    old = review["source"][field]
    review["source"][field] = old + 1 if type(old) is int else old + "x"
    with pytest.raises(ValueError, match="source"):
        validate(review, packet)


@pytest.mark.parametrize("field,value", [("timestamp_ms", 120400.0), ("source_frame_idx", True)])
def test_source_numeric_types_cannot_use_boolean_or_float_aliases(review, packet, field, value):
    review["source"][field] = value
    with pytest.raises(ValueError, match="source"):
        validate(review, packet)


@pytest.mark.parametrize("container", [None, "source", "rois", "judgments"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_exact_field_sets_disallow_unrequested_or_missing_tasks(review, packet, container, mutation):
    target = review if container is None else review[container]
    if mutation == "missing":
        del target[next(iter(target))]
    else:
        target["unrequested"] = "unknown"
    with pytest.raises(ValueError, match="fields"):
        validate(review, packet)


@pytest.mark.parametrize("which", ["roi_roles", "predicates"])
def test_duplicate_packet_task_identities_rejected(review, packet, which):
    packet[which][-1] = packet[which][0]
    with pytest.raises(ValueError, match="packet"):
        validate(review, packet)


@pytest.mark.parametrize("value", [None, {}, ["support"], "yes", "", True])
def test_invalid_judgment_values_fail_clearly(review, packet, value):
    review["judgments"]["center_white"] = value
    with pytest.raises(ValueError, match="judgment"):
        validate(review, packet)


@pytest.mark.parametrize("box", [
    None, [], [0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1.1, 1], [-0.1, 0, 1, 1],
    [0.4, 0.1, 0.4, 0.8], [0.7, 0.2, 0.1, 0.8], [0, 0.9, 1, 0.1],
    [False, 0, 1, 1], [0, 0, "1", 1], [float("nan"), 0, 1, 1],
    [0, 0, float("inf"), 1], [0, 0, 1, -float("inf")],
])
def test_invalid_boxes_rejected(review, packet, box):
    review["rois"]["person_left"]["box_xyxy_norm"] = box
    with pytest.raises(ValueError):
        validate(review, packet)


@pytest.mark.parametrize("item", [
    {"box_xyxy_norm": None, "unobservable": False},
    {"box_xyxy_norm": [0, 0, 1, 1], "unobservable": True},
    {"box_xyxy_norm": None, "unobservable": 1},
    {"box_xyxy_norm": [0, 0, 1, 1], "unobservable": 0},
    {"box_xyxy_norm": None},
    {"box_xyxy_norm": None, "unobservable": True, "confidence": 1},
])
def test_roi_unknown_state_and_geometry_are_consistent(review, packet, item):
    review["rois"]["person_left"] = item
    with pytest.raises(ValueError):
        validate(review, packet)


@pytest.mark.parametrize("interval", [
    None, [], [120.4], [120.4, 123.2, 125], [123.2, 120.4], [120.4, 120.4],
    [-1, 123.2], [120.4, "123.2"], [True, 123.2], [120.4, float("nan")],
    [float("inf"), 123.2], [120.4, -float("inf")], [120.4, 123.3],
])
def test_invalid_or_disagreeing_intervals_rejected(review, packet, interval):
    review["interval_seconds"] = interval
    with pytest.raises(ValueError):
        validate(review, packet)


@pytest.mark.parametrize("start", ["120.4", "2:60.4", "-1:00", "2:00.4seconds", "NaN", ""])
def test_time_text_is_not_implicitly_coerced(review, packet, start):
    review["start"] = start
    with pytest.raises(ValueError, match="start"):
        validate(review, packet)


def test_unknown_interval_cannot_retain_numeric_claim(review, packet):
    review["intervalUnknown"] = True
    with pytest.raises(ValueError, match="unknown interval"):
        validate(review, packet)


@pytest.mark.parametrize("duration", [0, -1, 123, True, float("nan"), float("inf"), "130"])
def test_video_duration_is_finite_positive_and_bounds_interval(review, packet, duration):
    with pytest.raises(ValueError):
        validate(review, packet, video_duration_seconds=duration)


def test_video_end_and_frame_boundaries_are_allowed(review, packet):
    review["rois"]["shirt_left"]["box_xyxy_norm"] = [0, 0, 1, 1]
    result = validate(review, packet, video_duration_seconds=123.2)
    assert result["interval_seconds"] == [120.4, 123.2]
    # Full-frame or overlapping boxes are geometrically valid, not quality claims.
    assert result["rois"]["shirt_left"]["box_xyxy_norm"] == [0, 0, 1, 1]


@pytest.mark.parametrize("digest", ["g" * 64, "A" * 64, "a" * 63, None])
def test_expected_packet_hash_must_itself_be_valid(review, packet, digest):
    with pytest.raises(ValueError, match="SHA-256"):
        validate_role_review(review, packet, packet_sha256=digest)


@pytest.mark.parametrize("field,value", [
    ("schema", "another-packet"), ("reviewer_count", True), ("reviewer_count", 2),
    ("predicates", None), ("predicates", [["center_white"]]),
    ("source_frame_idx", -1), ("timestamp_ms", True), ("query_sha256", None),
    ("query_id", ""), ("video_id", None),
])
def test_malformed_reference_packet_is_rejected(review, packet, field, value):
    packet[field] = value
    with pytest.raises(ValueError):
        validate(review, packet)
