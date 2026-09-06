#!/usr/bin/env python3
"""Build deterministic, archive-local OCR or ASR partition manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.cloud_artifacts import build_cloud_partition_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-zip", type=Path, required=True)
    parser.add_argument("--keyframes-zip", type=Path, action="append", required=True)
    parser.add_argument("--videos-zip", type=Path, action="append", default=[])
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--stage", choices=("ocr", "asr"), required=True)
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_cloud_partition_manifest(
        map_archive=args.maps_zip,
        keyframe_archives=args.keyframes_zip,
        video_archives=args.videos_zip,
        batch_id=args.batch_id,
        stage=args.stage,
        max_frames=args.max_frames,
        worker_count=args.worker_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_fingerprint": manifest["plan_fingerprint"],
                "partitions": len(manifest["partitions"]),
                "videos": manifest["expected_videos"],
                "frames": manifest["expected_frames"],
                "worker_loads": manifest["worker_loads"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
