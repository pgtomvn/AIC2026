"""Malformed imported source times cannot become OCR/ASR navigation context."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.retrieval.frame_evidence import _indexed_segments, frame_evidence


class Store:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, str]] = []

    def search(self, query: str, *, top_k: int, filters: str) -> list[dict]:
        self.calls.append((query, top_k, filters))
        return self.hits


def hit(start: object, end: object, *, kind: str = "asr", ident: str = "segment") -> dict:
    return {"id": ident, "payload": {"batch_id": "batch", "video_id": "video",
            "kind": kind, "start_ms": start, "end_ms": end,
            "text_nfc": "source context", "source_frame_idx": 7}}


@pytest.mark.parametrize("kind", ["asr", "ocr"])
@pytest.mark.parametrize("interval", [
    (1000, 500), (-1, 500), (0, -1), (-1000, -500),
    (True, 500), (0, False), (0.0, 500), (0, 500.0),
    (None, 500), (0, None), (float("nan"), 500), (0, float("inf")),
])
def test_malformed_source_interval_is_skipped_not_repaired(kind: str, interval: tuple) -> None:
    source = Store([hit(*interval, kind=kind)])
    before = deepcopy(source.hits)
    assert _indexed_segments(source, "batch", "video", kind, 0, 30_000, 10) == []
    # NaN is unequal to itself, so check raw identity for that fixture rather
    # than normalize it into a plausible source time.
    assert source.hits[0]["payload"]["start_ms"] is interval[0]
    assert source.hits[0]["payload"]["end_ms"] is interval[1]
    assert source.hits[0]["payload"]["text_nfc"] == before[0]["payload"]["text_nfc"]


def test_bad_times_cannot_hide_a_valid_later_hit_or_change_its_scope() -> None:
    source = Store([hit(1000, 500, ident="reversed"), hit(-1, 500, ident="negative"),
                    hit(2000, 2500, ident="valid")])
    before = deepcopy(source.hits)
    result = _indexed_segments(source, "batch", "video", "asr", 0, 30_000, 10)
    assert source.hits == before
    assert result == [{"segment_id": "valid", "kind": "asr", "start_ms": 2000,
                       "end_ms": 2500, "text": "source context", "source_frame_idx": 7}]
    assert len(source.calls) == 1
    assert source.calls[0][1] == 10


@pytest.mark.parametrize("interval", [(0, 0), (0, 100), (30_000, 30_000), (29_000, 31_000)])
def test_valid_closed_overlap_including_point_and_endpoints_is_preserved(interval: tuple) -> None:
    source = Store([hit(*interval)])
    result = _indexed_segments(source, "batch", "video", "asr", 0, 30_000, 10)
    assert len(result) == 1
    assert (result[0]["start_ms"], result[0]["end_ms"]) == interval


@pytest.mark.parametrize("interval", [(0, 99), (30_001, 40_000)])
def test_nonoverlap_remains_filtered_without_converting_it_into_a_negative(interval: tuple) -> None:
    assert _indexed_segments(Store([hit(*interval)]), "batch", "video", "asr",
                             100, 30_000, 10) == []


@pytest.mark.parametrize("change", [{"batch_id": "other"}, {"video_id": "other"}, {"kind": "ocr"}])
def test_valid_times_do_not_relax_existing_source_channel_identity_checks(change: dict) -> None:
    value = hit(100, 200)
    value["payload"].update(change)
    assert _indexed_segments(Store([value]), "batch", "video", "asr", 0, 30_000, 10) == []


def test_filtering_keeps_the_existing_budget_and_stable_source_time_order() -> None:
    source = Store([hit(1000, 500, ident="bad"), hit(4000, 5000, ident="z"),
                    hit(1000, 2000, ident="a"), hit(2000, 3000, ident="outside-limit")])
    result = _indexed_segments(source, "batch", "video", "asr", 0, 30_000, 3)
    assert [row["segment_id"] for row in result] == ["a", "z"]
    assert source.calls[0][1] == 3
    assert all("verdict" not in row and "query_truth" not in row for row in result)


def repository(*segments: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        get_source_frame=lambda *args: SimpleNamespace(timestamp_ms=1000),
        text_segments_in_range=lambda *args, **kwargs:
        tuple(segment for segment in segments if getattr(segment, "kind", kwargs["kind"]) == kwargs["kind"]),
    )


def local_segment(start: object = 100, end: object = 200, **changes: object) -> SimpleNamespace:
    value = dict(segment_id="local", batch_id="batch", video_id="video", kind="asr",
                 start_ms=start, end_ms=end, text_nfc="source context", source_frame_idx=7)
    value.update(changes)
    return SimpleNamespace(**value)


@pytest.mark.parametrize("kind", ["asr", "ocr"])
@pytest.mark.parametrize("interval", [(1000, 500), (-1, 500), (0, -1),
                                     (True, 500), (0, False), (0.0, 500), (0, 500.0), (None, 500)])
def test_repository_fallback_obeys_the_same_time_guard(kind: str, interval: tuple) -> None:
    segment = local_segment(*interval, kind=kind)
    source = repository(segment)
    # A rejected imported record cannot bypass rejection via the local adapter.
    result = frame_evidence(source, "batch", "video", 0,
                            text_stores={kind: Store([hit(*interval, kind=kind)])})
    assert result["segments"] == []
    assert result["speech_is_not_visual_confirmation"]
    assert (segment.start_ms, segment.end_ms) == interval


@pytest.mark.parametrize("change", [{"batch_id": "other"}, {"video_id": "other"},
                                    {"text_nfc": None}, {"end_ms": 100_000, "start_ms": 90_000}])
def test_repository_fallback_cannot_escape_identity_or_window(change: dict) -> None:
    assert frame_evidence(repository(local_segment(**change)), "batch", "video", 0)["segments"] == []


@pytest.mark.parametrize("kind", ["asr", "ocr"])
def test_point_context_is_retained_and_never_upgraded_to_a_durational_witness(kind: str) -> None:
    result = frame_evidence(repository(local_segment(1000, 1000, kind=kind)), "batch", "video", 0)
    assert result["segments"] == [{"segment_id": "local", "kind": kind, "start_ms": 1000,
                                  "end_ms": 1000, "text": "source context", "source_frame_idx": 7}]
    assert result["speech_is_not_visual_confirmation"]


def test_repository_claimed_wrong_channel_cannot_fall_through_a_loose_adapter() -> None:
    source = repository()
    source.text_segments_in_range = lambda *args, **kwargs: (local_segment(kind="unrequested"),)
    assert frame_evidence(source, "batch", "video", 0)["segments"] == []
