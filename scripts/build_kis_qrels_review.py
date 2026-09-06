"""Build a human-review KIS candidate bundle from cached per-channel rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.review_bundle import build_qrels_review_bundle
from src.evaluation.tuning import load_cached_rankings


def _load_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"query line {line_number} must be an object")
        rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--cached-rankings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--thumbnail-base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args(argv)
    manifest = build_qrels_review_bundle(
        _load_queries(Path(args.queries)),
        load_cached_rankings(args.cached_rankings),
        args.output_dir,
        candidate_depth=args.candidate_depth,
        thumbnail_base_url=args.thumbnail_base_url,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
