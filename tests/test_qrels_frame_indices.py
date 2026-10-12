"""Qrels loading must preserve the source frame-index type before validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.qrels import load_qrels_pack


def _write_pack(tmp_path: Path, interval: dict[str, object]) -> Path:
    """Write a genuine payload and matching manifest so type checks are reached."""

    text = "two people feeding goats"
    record = {
        "query_id": "q02",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "intent": "visual_dense",
        "video_group": "g1",
        "intervals": [{"video_id": "V1", **interval}],
    }
    payload = (json.dumps(record) + "\n").encode("utf-8")
    (tmp_path / "qrels.jsonl").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "spec_id": "test:v1",
        "corpus_fingerprint": "test-corpus",
        "index_fingerprint": "test-index",
        "model_fingerprint": "test-model",
        "content_file": "qrels.jsonl",
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "split_metadata": {"method": "video_group", "seed": 7},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.mark.parametrize("field", ["start_frame_idx", "end_frame_idx"])
@pytest.mark.parametrize("value", [True, False, 1.0, 1.9, "1", None, [], {}])
def test_qrels_rejects_non_integer_frame_indices(
    tmp_path: Path, field: str, value: object
) -> None:
    interval: dict[str, object] = {"start_frame_idx": 0, "end_frame_idx": 2}
    interval[field] = value
    path = _write_pack(tmp_path, interval)

    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        load_qrels_pack(path)


@pytest.mark.parametrize("field", ["start_frame_idx", "end_frame_idx"])
def test_qrels_rejects_missing_frame_indices(tmp_path: Path, field: str) -> None:
    interval: dict[str, object] = {"start_frame_idx": 0, "end_frame_idx": 2}
    del interval[field]

    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        load_qrels_pack(_write_pack(tmp_path, interval))


@pytest.mark.parametrize("start,end", [(0, 0), (0, 2), (7940, 8108), (2**60, 2**60 + 1)])
def test_qrels_preserves_valid_integer_frame_indices(
    tmp_path: Path, start: int, end: int
) -> None:
    path = _write_pack(tmp_path, {"start_frame_idx": start, "end_frame_idx": end})

    interval = load_qrels_pack(path).queries[0].intervals[0]

    assert interval.start_frame_idx == start
    assert interval.end_frame_idx == end
    assert type(interval.start_frame_idx) is int
    assert type(interval.end_frame_idx) is int


@pytest.mark.parametrize(
    "start,end,message",
    [
        (-1, 2, "start_frame_idx must be a non-negative integer"),
        (0, -1, "end_frame_idx must be a non-negative integer"),
        (2, 1, "end_frame_idx must be at least start_frame_idx"),
    ],
)
def test_qrels_preserves_interval_bounds_validation(
    tmp_path: Path, start: int, end: int, message: str
) -> None:
    path = _write_pack(tmp_path, {"start_frame_idx": start, "end_frame_idx": end})

    with pytest.raises(ValueError, match=message):
        load_qrels_pack(path)
