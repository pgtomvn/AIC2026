#!/usr/bin/env python3
"""Validate and import finalized cloud artifacts into local retrieval stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.cloud_artifacts import (
    CloudArtifactBundle,
    CloudArtifactImporter,
    cloud_model_contracts,
)
from src.runtime import build_runtime
from src.utils.io_helpers import load_config


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _channels(value: str) -> tuple[str, ...]:
    parts = tuple(item.strip().casefold() for item in value.split(",") if item.strip())
    allowed = ("siglip2", "ocr", "asr")
    if not parts or len(parts) != len(set(parts)) or any(item not in allowed for item in parts):
        raise argparse.ArgumentTypeError("channels must be a unique subset of siglip2,ocr,asr")
    return tuple(item for item in allowed if item in parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--channels", type=_channels, default=_channels("siglip2,ocr,asr"))
    parser.add_argument(
        "--asr-profile", choices=("production", "reference"), default="reference"
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly accept a validated cumulative channel snapshot below full coverage",
    )
    parser.add_argument(
        "--completion-path",
        type=Path,
        default=Path("data/checkpoints/cloud-import/b01-import-complete.json"),
    )
    parser.add_argument(
        "--channel-receipt-dir",
        type=Path,
        help="Per-channel completion receipts (defaults under data/checkpoints)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config()
    context = build_runtime(config)
    dense = {channel.name: channel for channel in context.pipeline.dense_channels}
    lexical = {channel.name: channel for channel in context.pipeline.lexical_channels}
    specification = config.get("readiness", {}).get("batches", {}).get(args.batch_id, {})
    bundle = CloudArtifactBundle(
        _path(args.artifact_dir),
        expected_batch_id=args.batch_id,
        expected_model_contracts=cloud_model_contracts(
            config, asr_profile=args.asr_profile
        ),
        expected_videos=int(specification.get("expected_videos", 0)) or None,
        expected_frames=int(specification.get("expected_canonical_frames", 0)) or None,
        allow_partial=args.allow_partial,
    )
    importer = CloudArtifactImporter(
        bundle=bundle,
        vector_store=dense["visual"].store.for_batch(args.batch_id),
        ocr_store=lexical["ocr"].store,
        asr_store=lexical["asr"].store,
        ledger=context.ledger,
        completion_path=_path(args.completion_path),
        channel_receipt_dir=(
            _path(args.channel_receipt_dir)
            if args.channel_receipt_dir is not None
            else _path(Path("data/checkpoints/cloud-import") / args.batch_id)
        ),
    )
    report = importer.run(
        channels=args.channels,
        resume=args.resume,
        validate_only=args.validate_only,
    )
    readiness = getattr(context, "readiness_service", None)
    if readiness is not None and not args.validate_only:
        readiness.invalidate()
    payload = report.to_dict()
    payload["target_seconds"] = 180
    payload["within_target"] = report.elapsed_ms <= 180_000
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
