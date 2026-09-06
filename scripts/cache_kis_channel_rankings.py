#!/usr/bin/env python3
"""Cache unfused KIS channel rankings for locked qrels or unjudged proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QueryRequest
from src.evaluation.qrels import (
    build_locked_qrels_split,
    load_qrels_pack,
    validate_production_qrels,
)
from src.runtime import build_pipeline


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_proxy_queries(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"proxy query row {line_number} must be an object")
            query_id = str(value.get("query_id", "")).strip()
            text = str(
                value.get("text") or value.get("text_draft_not_ground_truth") or ""
            ).strip()
            if not query_id or not text or query_id in seen:
                raise ValueError(f"proxy query row {line_number} has an invalid identity/text")
            seen.add(query_id)
            rows.append(
                {
                    "query_id": query_id,
                    "text": text,
                    "intent_class": str(value.get("intent") or "proxy_unjudged"),
                }
            )
    if not rows:
        raise ValueError("proxy query file is empty")
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    """Query each logical channel once and atomically persist reusable rankings."""

    parser = argparse.ArgumentParser(description=__doc__)
    query_source = parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--qrels-manifest", type=Path)
    query_source.add_argument(
        "--proxy-queries",
        type=Path,
        help="Unjudged JSONL queries for operational shadow benchmarking only",
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/kis-channel-rankings.json"),
    )
    parser.add_argument(
        "--semantic-views",
        action="store_true",
        help="Cache independent semantic-view rankings instead of logical channels",
    )
    parser.add_argument(
        "--dense-channel",
        default="organizer_clip",
        help="Dense encoder to cache when --semantic-views is selected",
    )
    args = parser.parse_args(argv)
    if args.qrels_manifest is not None:
        pack = load_qrels_pack(args.qrels_manifest)
        validate_production_qrels(pack)
        seed = int(pack.manifest.split_metadata["seed"])
        locked = build_locked_qrels_split(pack, seed=seed)
        queries = tuple(
            {
                "query_id": query.query_id,
                "text": query.text or "",
                "intent_class": query.intent_class,
            }
            for query in pack.queries
        )
        evaluation_kind = "locked_qrels"
        qrels_content_sha256: str | None = pack.manifest.content_sha256
        locked_test_sha256: str | None = locked.test_sha256
        query_source_sha256 = _sha256(args.qrels_manifest)
    else:
        queries = _load_proxy_queries(args.proxy_queries)
        evaluation_kind = "proxy_unjudged"
        qrels_content_sha256 = None
        locked_test_sha256 = None
        query_source_sha256 = _sha256(args.proxy_queries)
    pipeline = build_pipeline()
    rows: list[dict[str, object]] = []
    for index, query in enumerate(queries, start=1):
        request = QueryRequest(
            query_id=query["query_id"],
            original_text=query["text"],
            mode="automatic",
            batch_filters=(args.batch_id,),
        )
        started = time.monotonic()
        rankings = (
            pipeline.collect_semantic_view_rankings(
                request,
                channel_name=args.dense_channel,
            )
            if args.semantic_views
            else pipeline.collect_channel_rankings(request)
        )
        elapsed_ms = (time.monotonic() - started) * 1_000
        statuses = pipeline.last_channel_statuses
        channels: dict[str, object] = {}
        for ranking in rankings:
            if args.semantic_views:
                # Per-view searches share adapter/channel setup. Splitting the
                # measured wall time keeps their additive tuning latency equal
                # to the observed query cost instead of multiplying it.
                latency = elapsed_ms / max(1, len(rankings))
            else:
                status = statuses.get(ranking.name, {})
                latency = (
                    float(status.get("latency_ms", 0.0))
                    if isinstance(status, dict)
                    else 0.0
                )
            channels[ranking.name] = {
                "results": [dict(value) for value in ranking.results],
                "latency_ms": max(0.0, latency),
                "request_count": (
                    1
                    if args.semantic_views
                    else max(1, int(status.get("query_variant_count", 1)))
                    if isinstance(status, dict)
                    else 1
                ),
            }
        rows.append(
            {
                "query_id": query["query_id"],
                "intent_class": query["intent_class"],
                "elapsed_ms": max(0.0, elapsed_ms),
                "channels": channels,
            }
        )
        print(
            json.dumps(
                {
                    "query_id": query["query_id"],
                    "completed": index,
                    "total": len(queries),
                    "elapsed_ms": round(elapsed_ms, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "batch_id": args.batch_id,
        "evaluation_kind": evaluation_kind,
        "query_source_sha256": query_source_sha256,
        "qrels_content_sha256": qrels_content_sha256,
        "locked_test_sha256": locked_test_sha256,
        "pipeline_fingerprint": pipeline.fingerprint,
        "ranking_mode": "semantic_views" if args.semantic_views else "logical_channels",
        "dense_channel": args.dense_channel if args.semantic_views else None,
        "queries": rows,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "queries": len(rows),
                "evaluation_kind": evaluation_kind,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
