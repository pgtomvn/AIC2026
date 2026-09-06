#!/usr/bin/env python3
"""Validate and fingerprint the locked 20% test partition of a KIS qrels pack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.qrels import build_locked_qrels_split, load_qrels_pack


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    """Print the deterministic split and optionally persist its locked hash."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically update only split_metadata in the qrels manifest",
    )
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    pack = load_qrels_pack(manifest_path)
    split = build_locked_qrels_split(pack, seed=args.seed)
    report: dict[str, object] = {
        "manifest": str(manifest_path),
        "seed": args.seed,
        "method": "video_group",
        "ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "query_counts": {
            "train": len(split.train_query_ids),
            "validation": len(split.validation_query_ids),
            "test": len(split.test_query_ids),
        },
        "locked_test_sha256": split.test_sha256,
        "written": False,
    }
    if args.write:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("qrels manifest must be an object")
        manifest["split_metadata"] = {
            "method": "video_group",
            "ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "seed": args.seed,
            "locked_test_sha256": split.test_sha256,
        }
        _atomic_json(manifest_path, manifest)
        report["written"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
