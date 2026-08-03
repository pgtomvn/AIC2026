"""Shared command-line orchestration for resumable native ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ingest.factory import build_native_ingestor
from src.ingest.native_pipeline import discover_video_sources
from src.utils.io_helpers import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Loose videos, ZIPs, or directories")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        type=Path,
        help="Repeatable directory/ZIP/video input alias",
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--competition", default="aic2026")
    parser.add_argument(
        "--cpu-workers",
        "--parallel-workers",
        dest="cpu_workers",
        type=int,
        help="CPU decode workers; --parallel-workers is a compatibility alias",
    )
    parser.add_argument("--queue-size", type=int)
    channel_mode = parser.add_mutually_exclusive_group()
    channel_mode.add_argument(
        "--core-only", action="store_true", help="Skip OCR/ASR/text enrichment"
    )
    channel_mode.add_argument(
        "--all-channels",
        action="store_true",
        help="Explicitly request visual, OCR, ASR, and semantic text",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--recreate-indexes", action="store_true")
    parser.add_argument(
        "--no-recover", action="store_true", help="Do not reclaim interrupted stages"
    )
    return parser


def _worker_defaults(config: dict[str, Any]) -> tuple[int, int]:
    profile = str(config.get("profile", "laptop"))
    configured = config.get("ingestion", {}).get("worker_profiles", {})
    values = configured.get(profile, {}) if isinstance(configured, dict) else {}
    fallback = (8, 8) if profile == "accelerated" else (2, 4)
    return (
        int(values.get("cpu_workers", fallback[0])),
        int(values.get("queue_size", fallback[1])),
    )


def _summary(outcomes: dict[str, str]) -> dict[str, Any]:
    core_failures = sorted(
        video_id for video_id, value in outcomes.items() if value.startswith("failed")
    )
    enrichment_failures = sorted(
        video_id
        for video_id, value in outcomes.items()
        if "enrichment failed" in value
    )
    return {
        "outcomes": outcomes,
        "core_failures": core_failures,
        "enrichment_failures": enrichment_failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    default_workers, default_queue = _worker_defaults(config)
    cpu_workers = default_workers if args.cpu_workers is None else args.cpu_workers
    queue_size = default_queue if args.queue_size is None else args.queue_size
    if cpu_workers < 1 or queue_size < 1:
        parser.error("worker and queue sizes must be positive")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")
    inputs = [*args.inputs, *args.input_dir]
    if not inputs:
        parser.error("at least one positional input or --input-dir is required")
    sources = discover_video_sources(inputs)
    if not sources:
        raise ValueError("no supported videos were discovered")
    if args.max_videos is not None:
        sources = sources[: args.max_videos]
    enrich = not args.core_only
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "batch_id": args.batch_id,
                    "profile": str(config.get("profile", "laptop")),
                    "channels": "all" if enrich else "core",
                    "cpu_workers": cpu_workers,
                    "queue_size": queue_size,
                    "video_count": len(sources),
                    "videos": [source.video_id for source in sources],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    ingestor = build_native_ingestor(
        config,
        batch_id=args.batch_id,
        competition=args.competition,
        enrichment=enrich,
    )
    archives = sorted({source.path.resolve() for source in sources if source.member})
    if archives:
        # Catalog verification is the authorization boundary for archive-backed
        # ingestion.  Initialize only the SQLite schema needed to read that
        # catalog, then reject unmanaged/incomplete archives before the full
        # ingestor can create (or recreate) any external index.
        ingestor.ledger.initialize()
    for archive in archives:
        ingestor.ledger.assert_archive_ready(archive)
    ingestor.initialize(
        recreate_indexes=args.recreate_indexes,
        recover_interrupted=not args.no_recover,
    )
    outcomes = ingestor.ingest_many(
        sources,
        cpu_workers=cpu_workers,
        queue_size=queue_size,
        enrich=enrich,
    )
    result = _summary(outcomes)
    result["status"] = ingestor.ledger.status_summary(args.batch_id).as_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["core_failures"] or result["enrichment_failures"] else 0


__all__ = ["build_parser", "main"]
