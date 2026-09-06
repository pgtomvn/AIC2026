"""RFC 4180-compatible CSV formatting for Visual Q&A submissions."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

QAExportRow = tuple[str, int | str, str]


def _validated_row(video_id: str, frame_idx: int | str, answer: str) -> QAExportRow:
    if not isinstance(video_id, str):
        raise TypeError("video_id must be a string")
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, (int, str)):
        raise TypeError("frame_idx must be an integer or string")
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    return video_id.strip(), str(frame_idx).strip(), answer


def format_qa_submission_row(
    video_id: str,
    frame_idx: int | str,
    answer: str,
) -> str:
    """Format one QA row without appending a record terminator.

    The answer is passed to :mod:`csv` unchanged. In particular, leading and
    trailing whitespace remains part of the field value.
    """

    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        delimiter=",",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="",
        doublequote=True,
    )
    writer.writerow(_validated_row(video_id, frame_idx, answer))
    return output.getvalue()


def export_qa_csv(rows: Iterable[QAExportRow]) -> str:
    """Serialize QA rows with LF terminators and RFC 4180 minimal quoting."""

    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        delimiter=",",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
        doublequote=True,
    )
    for video_id, frame_idx, answer in rows:
        writer.writerow(_validated_row(video_id, frame_idx, answer))
    return output.getvalue()


__all__ = ["QAExportRow", "export_qa_csv", "format_qa_submission_row"]
