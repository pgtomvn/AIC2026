#!/usr/bin/env python3
"""Package selected validated cloud channels with ZIP_STORED and SHA-256."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.cloud_artifacts import (
    ASR_TRANSCRIPTS_NAME,
    OCR_FEATURES_NAME,
    SIGLIP_FEATURES_NAME,
    SIGLIP_MANIFEST_NAME,
    CloudArtifactBundle,
    cloud_model_contracts,
    sha256_file,
)
from src.utils.io_helpers import load_config


def _channels(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip().casefold() for item in value.split(",") if item.strip())
    allowed = ("siglip2", "ocr", "asr")
    selected = tuple(item for item in allowed if item in requested)
    if not selected or len(selected) != len(requested) or len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError("channels must be a unique non-empty subset")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--channels", type=_channels, default=_channels("siglip2,ocr,asr"))
    parser.add_argument(
        "--asr-profile", choices=("production", "reference"), default="reference"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly package a validated cumulative partial channel snapshot",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config()
    specification = config.get("readiness", {}).get("batches", {}).get(args.batch_id, {})
    bundle = CloudArtifactBundle(
        args.artifact_dir,
        expected_batch_id=args.batch_id,
        expected_model_contracts=cloud_model_contracts(
            config, asr_profile=args.asr_profile
        ),
        expected_videos=int(specification.get("expected_videos", 0)) or None,
        expected_frames=int(specification.get("expected_canonical_frames", 0)) or None,
        allow_partial=args.allow_partial,
    )
    counts = bundle.validate(channels=args.channels)
    manifest_sha256 = bundle.manifest_sha256
    names = {
        "siglip2": SIGLIP_FEATURES_NAME,
        "ocr": OCR_FEATURES_NAME,
        "asr": ASR_TRANSCRIPTS_NAME,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    channel_tag = "-".join(args.channels)
    target = args.output_dir / f"b01_{channel_tag}_{manifest_sha256[:12]}.zip"
    temporary = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        archive.write(bundle.manifest_path, arcname=SIGLIP_MANIFEST_NAME)
        for stage in args.channels:
            archive.write(args.artifact_dir / names[stage], arcname=names[stage])
    os.replace(temporary, target)
    digest = sha256_file(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    os.replace(sidecar_tmp, sidecar)
    print(
        json.dumps(
            {
                "bundle": str(target),
                "bundle_sha256": digest,
                "manifest_sha256": manifest_sha256,
                "channels": list(args.channels),
                "counts": counts,
                "sha256_sidecar": str(sidecar),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
