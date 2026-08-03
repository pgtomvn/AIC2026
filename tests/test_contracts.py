"""Unit tests for canonical Textual KIS domain contracts."""

from __future__ import annotations

import json

import pytest

from src.contracts import CompetitionSpec, FrameRecord, QueryRequest, SearchHit, TextSegment


def _frame_record(**overrides) -> FrameRecord:
    values = {
        "batch_id": "aic25-b1",
        "video_id": "L21_V001",
        "keyframe_ordinal": 7,
        "source_frame_idx": 181,
        "pts": 271500,
        "time_base": "1/90000",
        "timestamp_ms": 3017,
        "fps_num": 30000,
        "fps_den": 1001,
        "shot_id": "shot-2",
        "sampling_reasons": ("scene", "max_gap"),
        "image_uri": "zip://keyframes/L21_V001/007.jpg",
        "video_uri": "zip://videos/L21_V001.mp4",
        "source_sha256": "AB" * 32,
        "extractor_version": "organizer-aic25-b1",
    }
    values.update(overrides)
    return FrameRecord(**values)


def test_frame_record_round_trip_preserves_both_frame_identities() -> None:
    record = _frame_record()

    payload = record.to_dict()
    restored = FrameRecord.from_dict(json.loads(json.dumps(payload)))

    assert restored == record
    assert restored.keyframe_ordinal == 7
    assert restored.source_frame_idx == 181
    assert restored.source_sha256 == ("ab" * 32)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"keyframe_ordinal": 0}, "keyframe_ordinal"),
        ({"source_frame_idx": -1}, "source_frame_idx"),
        ({"time_base": "0/1"}, "time_base"),
        ({"sampling_reasons": ("scene", "scene")}, "duplicates"),
        ({"source_sha256": "bad"}, "source_sha256"),
    ],
)
def test_frame_record_rejects_invalid_identity_and_timing(override, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _frame_record(**override)


def test_serialized_contracts_forbid_ambiguous_frame_id() -> None:
    payload = _frame_record().to_dict()
    payload["frame_id"] = payload.pop("source_frame_idx")

    with pytest.raises(ValueError, match="ambiguous frame_id"):
        FrameRecord.from_dict(payload)


def test_text_segment_validates_interval_unicode_and_confidence() -> None:
    segment = TextSegment(
        segment_id="ocr-L21_V001-181",
        batch_id="aic25-b1",
        video_id="L21_V001",
        kind="ocr",
        start_ms=3000,
        end_ms=3100,
        source_frame_idx=181,
        text_raw="Cà phê Sữa Đá",
        text_nfc="Cà phê Sữa Đá",
        text_accent_folded="Ca phe Sua Da",
        confidence=0.95,
        model_version="pp-ocrv5-mobile",
        polygons=(((10, 20), (30, 20), (30, 40), (10, 40)),),
    )

    assert TextSegment.from_dict(segment.to_dict()) == segment
    assert segment.polygons[0][2] == (30.0, 40.0)
    with pytest.raises(ValueError, match="end_ms"):
        TextSegment(**{**segment.to_dict(), "end_ms": 2999})
    with pytest.raises(ValueError, match="between 0 and 1"):
        TextSegment(**{**segment.to_dict(), "confidence": 1.1})
    with pytest.raises(ValueError, match="at least three"):
        TextSegment(**{**segment.to_dict(), "polygons": (((0, 0), (1, 1)),)})


def test_query_request_normalizes_sequences_and_rejects_feedback_overlap() -> None:
    request = QueryRequest(
        query_id="kis-1",
        original_text="Người đàn ông bước vào quán",
        mode="interactive",
        batch_filters=["aic25-b1"],
        objects=["người", "quán"],
        actions=["bước"],
        positive_feedback_ids=["hit-1"],
    )

    assert request.batch_filters == ("aic25-b1",)
    assert request.objects == ("người", "quán")
    assert QueryRequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError, match="disjoint"):
        QueryRequest(
            query_id="kis-1",
            original_text="query",
            positive_feedback_ids=("same",),
            negative_feedback_ids=("same",),
        )


def test_search_hit_and_competition_spec_validate_rank_and_video_name() -> None:
    hit = SearchHit(
        rank=1,
        video_id="L21_V001",
        source_frame_idx=181,
        timestamp_ms=3017,
        shot_id="shot-2",
        fused_score=0.4,
        channel_ranks={"visual": 1, "ocr": 6},
        evidence={"ocr": "Cà phê"},
        thumbnail_uri="thumbs/L21_V001/181.webp",
    )
    spec = CompetitionSpec.historical_2025()

    assert SearchHit.from_dict(hit.to_dict()) == hit
    assert spec.normalize_video_id("l21_v001.mp4") == "L21_V001"
    assert spec.contains_frame(10, 10, 20)
    assert spec.contains_frame(20, 10, 20)
    assert not spec.contains_frame(21, 10, 20)
    with pytest.raises(ValueError, match="path"):
        spec.normalize_video_id("../../L21_V001")


def test_search_hit_evidence_must_be_json_serializable() -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        SearchHit(
            rank=1,
            video_id="L21_V001",
            source_frame_idx=1,
            timestamp_ms=0,
            shot_id="shot-1",
            fused_score=1.0,
            thumbnail_uri="thumbs/L21_V001/1.webp",
            evidence={"bad": object()},
        )


def test_competition_spec_accepts_application_config_section() -> None:
    spec = CompetitionSpec.from_config(
        {
            "spec_id": "fixture-v2",
            "output_columns": ["video_name", "frame_idx"],
            "max_results": 50,
            "cutoffs": [1, 5, 20, 50],
            "include_header": True,
            "output_dir": "outputs/ignored-runtime-key",
        }
    )

    assert spec.spec_id == "fixture-v2"
    assert spec.max_results == 50
    assert spec.cutoffs == (1, 5, 20, 50)
    assert spec.include_header is True
