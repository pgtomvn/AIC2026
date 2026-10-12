"""B02 reference coordinate export, separate from historical L-series adapters."""
from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any

from src.contracts import QACompetitionSpec
from src.ingest.b02_presentation import present_ordinal_frame
from src.ingest.frame_identity import resolve_image_frame
from src.retrieval.trake_scope import trake_video_allowed


def _trake_values(repository: Any, row: Mapping[str, Any]) -> list[Any]:
    video = row.get("video_id")
    if not isinstance(video, str) or not trake_video_allowed("aic26-b2", video):
        raise ValueError("video is outside the released B02 TRAKE scope")
    matches = row.get("stage_matches")
    if not isinstance(matches, (list, tuple)) or not 2 <= len(matches) <= 20:
        raise ValueError("TRAKE requires 2 to 20 canonical stages")
    coordinates = []
    for number, match in enumerate(matches, 1):
        if not isinstance(match, Mapping) or type(match.get("stage_index")) is not int or match["stage_index"] != number:
            raise ValueError("TRAKE stage indices must be contiguous")
        if (match.get("batch_id", "aic26-b2") != "aic26-b2"
                or match.get("video_id", video) != video):
            raise ValueError("TRAKE cannot cross videos or batches")
        frame = resolve_image_frame(repository, {**match, "batch_id": "aic26-b2", "video_id": video})
        if frame is None or type(match.get("timestamp_ms")) is not int or match["timestamp_ms"] != frame.timestamp_ms:
            raise ValueError("stale canonical TRAKE stage")
        coordinates.append((frame.keyframe_ordinal, frame.source_frame_idx, frame.timestamp_ms))
    if any(any(a >= b for a, b in zip(left, right)) for left, right in zip(coordinates, coordinates[1:])):
        raise ValueError("TRAKE ordinal, source and time must all increase strictly")
    return [video, *(item[1] for item in coordinates)]


def serialize_b02_submission(repository: Any, rows: Sequence[Mapping[str, Any]], *,
                             task: str = "kis", answer_spec_id: str | None = None) -> bytes:
    """Resolve every coordinate from canonical data; UI values cannot override PTS."""
    if task not in {"kis", "qa", "trake"}:
        raise ValueError("unsupported B02 submission task")
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 100:
        raise ValueError("B02 export requires 1 to 100 rows")
    spec = QACompetitionSpec.for_spec_id(answer_spec_id) if task == "qa" else None
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    seen = set()
    stage_count = None
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or row.get("batch_id") != "aic26-b2":
            raise ValueError("B02 export cannot mix batches")
        if "rank" in row and (isinstance(row["rank"], bool) or row["rank"] != rank):
            raise ValueError("export ranks must be contiguous from one")
        if task == "trake":
            values = _trake_values(repository, row)
            if values[0] in seen or stage_count not in (None, len(values) - 1):
                raise ValueError("TRAKE requires unique videos and equal stage counts")
            seen.add(values[0])
            stage_count = len(values) - 1
            writer.writerow(values)
            continue
        frame = resolve_image_frame(repository, row)
        if frame is None:
            raise ValueError("canonical ordinal is missing")
        public = present_ordinal_frame(frame)
        if "timestamp_ms" in row and row["timestamp_ms"] != frame.timestamp_ms:
            raise ValueError("stale canonical timestamp")
        values = [frame.video_id, public["submission_value"]]
        if spec is not None:
            answer = row.get("answer")
            if not isinstance(answer, str) or not answer.strip() or len(answer) > 100:
                raise ValueError("QA requires an explicit answer")
            values.append(spec.normalize_answer(answer))
        identity = tuple(values)
        if identity in seen:
            raise ValueError("duplicate submission coordinate; choose one of the equal-PTS images")
        seen.add(identity)
        writer.writerow(values)
    return output.getvalue().encode("utf-8")
