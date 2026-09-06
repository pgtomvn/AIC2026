#!/usr/bin/env python3
"""Freeze unjudged KIS queries for repeatable operational shadow benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/evaluation/kis-qrels-review-seed.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/kis-proxy-queries-v1.jsonl"),
    )
    args = parser.parse_args(argv)
    source = args.source.resolve()
    source_bytes = source.read_bytes()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source_bytes.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        query_id = str(value.get("query_id", "")).strip()
        text = str(
            value.get("text") or value.get("text_draft_not_ground_truth") or ""
        ).strip()
        if not query_id or not text or query_id in seen:
            raise ValueError(f"source row {line_number} has an invalid identity/text")
        seen.add(query_id)
        rows.append(
            {
                "schema_version": 1,
                "query_id": query_id,
                "text": text,
                "source_kind": str(value.get("source_kind") or "unknown"),
                "judged": False,
                "activation_eligible": False,
            }
        )
    if len(rows) != 27:
        raise ValueError("the frozen proxy contract requires exactly 27 queries")
    rows.sort(key=lambda value: str(value["query_id"]))
    output_bytes = (
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in rows
        )
    ).encode("utf-8")
    output = args.output.resolve()
    _atomic_write(output, output_bytes)
    manifest = {
        "schema_version": 1,
        "evaluation_kind": "proxy_unjudged",
        "activation_eligible": False,
        "query_count": len(rows),
        "source": str(source),
        "source_sha256": _sha256_bytes(source_bytes),
        "content_sha256": _sha256_bytes(output_bytes),
        "queries": [value["query_id"] for value in rows],
    }
    manifest_path = output.with_suffix(".manifest.json")
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({"output": str(output), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
