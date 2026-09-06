#!/usr/bin/env python3
"""Prepare an auditable KIS review-seed queue without fabricating qrels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _round2_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = payload.get("kis", ()) if isinstance(payload, Mapping) else ()
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        query_id, text = value.get("id"), value.get("text")
        if not isinstance(query_id, str) or not isinstance(text, str) or not text.strip():
            continue
        rows.append(
            {
                "query_id": query_id,
                "text": text.strip(),
                "source": str(path),
                "source_kind": "competition_query",
                "intent": "",
                "video_group": "",
                "tags": [],
                "seed_intervals": [],
                "human_video_verified": False,
                "review_status": "needs_ground_truth_review",
            }
        )
    return rows


def _round1_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = payload.get("queries", ()) if isinstance(payload, Mapping) else ()
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping) or value.get("task") != "kis":
            continue
        query_id = value.get("query_id")
        entities = value.get("entities", ())
        if not isinstance(query_id, str) or not isinstance(entities, list):
            continue
        # The manifest retained entity summaries, not the exact competition
        # wording. This draft is deliberately inadmissible until a reviewer
        # restores and verifies the original query text.
        draft = "; ".join(str(item) for item in entities if isinstance(item, str))
        rows.append(
            {
                "query_id": f"query-{query_id}-kis",
                "text_draft_not_ground_truth": draft,
                "text": "",
                "source": str(path),
                "source_kind": "entity_summary_only",
                "intent": "",
                "video_group": "",
                "tags": [],
                "seed_intervals": [],
                "human_video_verified": False,
                "review_status": "needs_original_query_text_and_ground_truth_review",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round2",
        type=Path,
        default=Path("outputs/submissions/round2_queries_en.json"),
    )
    parser.add_argument(
        "--round1-manifest",
        type=Path,
        default=Path("outputs/submissions/round1_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/kis-qrels-review-seed.jsonl"),
    )
    args = parser.parse_args(argv)
    rows = [*_round1_rows(args.round1_manifest), *_round2_rows(args.round2)]
    identities = [row["query_id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("review seed contains duplicate query IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: str(item["query_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "output": str(args.output),
        "seed_query_count": len(rows),
        "exact_text_query_count": sum(bool(row.get("text")) for row in rows),
        "entity_summary_only_count": sum(
            row["source_kind"] == "entity_summary_only" for row in rows
        ),
        "target_query_count": 120,
        "remaining_human_authored_queries": 120 - len(rows),
        "ground_truth_labels_created": 0,
        "production_ready": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
