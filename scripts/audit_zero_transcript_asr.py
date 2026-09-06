#!/usr/bin/env python3
"""Audit zero-transcript ASR videos without mutating production state."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import load_config  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rows(ledger_path: Path, batch_id: str) -> tuple[dict[str, Any], ...]:
    connection = sqlite3.connect(
        f"file:{ledger_path.resolve().as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        values = connection.execute(
            """
            SELECT DISTINCT artifacts.video_id, artifacts.kind, videos.source_uri
            FROM artifacts
            LEFT JOIN videos
              ON videos.batch_id = artifacts.batch_id
             AND videos.video_id = artifacts.video_id
            WHERE artifacts.batch_id = ?
              AND artifacts.stage_name = 'reference_asr'
              AND artifacts.kind IN ('asr_no_audio', 'asr_zero_transcript')
            ORDER BY artifacts.video_id
            """,
            (batch_id,),
        ).fetchall()
    finally:
        connection.close()
    return tuple(dict(row) for row in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config()
    configured_ledger = config.get("data", {}).get("catalog_db", "data/state/jobs.sqlite3")
    ledger_path = _project_path(args.ledger or configured_ledger)
    rows = _rows(ledger_path, args.batch_id)
    audited = [
        {
            "video_id": str(row["video_id"]),
            "ledger_artifact_kind": str(row["kind"]),
            "source_uri": str(row.get("source_uri") or ""),
            "status": "audit_deferred",
            "reason_code": "source_archive_unavailable",
            "production_mutated": False,
        }
        for row in rows
    ]
    payload = {
        "schema_version": 1,
        "audit_kind": "asr_zero_transcript",
        "batch_id": args.batch_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "production_mutated": False,
        "summary": {
            "zero_transcript_videos": len(audited),
            "audited": 0,
            "audit_deferred": len(audited),
            "no_audio_stream": 0,
            "no_speech_detected": 0,
        },
        "videos": audited,
    }
    output = _project_path(args.output)
    _atomic_json(output, payload)
    print(json.dumps({"output": str(output), **payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
