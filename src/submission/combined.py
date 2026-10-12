"""Read-only canonical CSV serialization across the two audited corpora."""
from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.contracts import QACompetitionSpec
from src.retrieval.batch_scope import B01, B02
from src.submission.b02 import serialize_b02_submission


def _b01_frame(repository: Any, row: Mapping[str, Any]) -> Any:
    video, source = row.get("video_id"), row.get("source_frame_idx")
    if not isinstance(video, str) or not re.fullmatch(r"L\d{2}_V\d{3}", video):
        raise ValueError("invalid B01 video")
    if type(source) is not int or source < 0:
        raise ValueError("invalid source frame")
    frame = repository.get_source_frame(B01, video, source)
    if frame is None or (frame.batch_id, frame.video_id, frame.source_frame_idx) != (B01, video, source):
        raise ValueError("source frame is not canonical")
    for key in ("timestamp_ms", "keyframe_ordinal"):
        if row.get(key) is not None and (type(row[key]) is not int or row[key] != getattr(frame, key)):
            raise ValueError("stale canonical frame metadata")
    return frame


def serialize_combined_submission(repository: Any, rows: Sequence[Mapping[str, Any]], *,
                                  task: str = "kis", answer_spec_id: str | None = None) -> bytes:
    """Preserve rank order; N uses PTS milliseconds, L/M/S source frames.

    This is a local CSV preview, not a DRES submission or server acceptance.
    The caller must separately check release flags and active-batch state.
    """
    if task not in {"kis", "qa", "trake"} or not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 100:
        raise ValueError("invalid task or row count")
    answer_spec = QACompetitionSpec.for_spec_id(answer_spec_id) if task == "qa" else None
    output, seen, width = io.StringIO(newline=""), set(), None
    writer = csv.writer(output, lineterminator="\n")
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or ("rank" in row and (type(row["rank"]) is not int or row["rank"] != rank)):
            raise ValueError("invalid ranked row")
        if row.get("batch_id") == B02:
            encoded = serialize_b02_submission(repository, [dict(row, rank=1)], task=task, answer_spec_id=answer_spec_id)
            values = next(csv.reader(io.StringIO(encoded.decode("utf-8"))))
        elif row.get("batch_id") == B01:
            if task == "trake":
                stages = row.get("stage_matches")
                if not isinstance(stages, (list, tuple)) or not 2 <= len(stages) <= 20:
                    raise ValueError("invalid TRAKE stages")
                frames = []
                for number, stage in enumerate(stages, 1):
                    if (not isinstance(stage, Mapping) or type(stage.get("stage_index")) is not int
                            or stage["stage_index"] != number or type(stage.get("timestamp_ms")) is not int
                            or stage.get("batch_id", B01) != B01
                            or stage.get("video_id", row.get("video_id")) != row.get("video_id")):
                        raise ValueError("invalid or cross-video TRAKE stage")
                    frames.append(_b01_frame(repository, dict(stage, video_id=row.get("video_id"))))
                if any(a.source_frame_idx >= b.source_frame_idx or a.timestamp_ms >= b.timestamp_ms
                       for a, b in zip(frames, frames[1:])):
                    raise ValueError("TRAKE source and time must strictly increase")
                values = [row["video_id"], *(frame.source_frame_idx for frame in frames)]
            else:
                frame = _b01_frame(repository, row)
                values = [frame.video_id, frame.source_frame_idx]
                if answer_spec is not None:
                    answer = row.get("answer")
                    if not isinstance(answer, str) or not answer.strip() or len(answer) > 100:
                        raise ValueError("QA requires an explicit answer")
                    values.append(answer_spec.normalize_answer(answer))
        else:
            raise ValueError("unknown row batch")
        identity = str(values[0]) if task == "trake" else tuple(str(item) for item in values)
        if identity in seen or width not in (None, len(values)):
            raise ValueError("duplicate coordinates or inconsistent stage count")
        seen.add(identity)
        width = len(values)
        writer.writerow(values)
    return output.getvalue().encode("utf-8")
