"""Freeze prepared model/tool directories into a runtime checksum manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assets import freeze_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "checksums/runtime-assets.sha256",
    )
    args = parser.parse_args(argv)
    count = freeze_assets(args.paths, root=PROJECT_ROOT, destination=args.output)
    print(f"Frozen {count} files in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
