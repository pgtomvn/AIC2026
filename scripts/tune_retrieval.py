"""Tune Textual KIS RRF parameters from offline cached channel rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.textual_kis import load_ground_truth
from src.evaluation.tuning import (
    load_cached_rankings,
    persist_tuning_report,
    tune_cached_rankings,
)
from src.utils.io_helpers import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune weighted RRF and report cumulative channel ablations"
    )
    parser.add_argument("--rankings", type=Path, help="Versioned cached-ranking JSON")
    parser.add_argument("--ground-truth", type=Path, help="Optional labeled interval CSV/JSON")
    parser.add_argument("--rrf-k", type=int, action="append", help="Repeat to replace k grid")
    parser.add_argument(
        "--depth",
        action="append",
        default=[],
        metavar="CHANNEL=VALUE[,VALUE]",
        help="Candidate-depth grid override; repeat per channel",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="CHANNEL=VALUE[,VALUE]",
        help="Weight grid override; repeat per channel",
    )
    parser.add_argument("--max-candidates", type=int, default=50_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/retrieval-tuning.json"),
    )
    return parser


def _grid(values: list[str], *, numeric: type[int] | type[float]) -> dict[str, tuple]:
    parsed: dict[str, tuple] = {}
    for value in values:
        channel, separator, raw_values = value.partition("=")
        if not separator or not channel or not raw_values:
            raise ValueError(f"invalid grid override: {value!r}")
        if channel in parsed:
            raise ValueError(f"duplicate grid override for channel: {channel}")
        parsed[channel] = tuple(numeric(item) for item in raw_values.split(","))
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Select parameters or explicitly preserve defaults when labels are absent."""

    args = _parser().parse_args(argv)
    if args.ground_truth and not args.rankings:
        raise ValueError("--rankings is required with --ground-truth")
    config = load_config()["retrieval"]
    default_depths = {str(key): int(value) for key, value in config["candidate_depths"].items()}
    default_weights = {
        str(key): float(value) for key, value in config["channel_weights"].items()
    }
    depth_values = _grid(args.depth, numeric=int)
    weight_values = _grid(args.weight, numeric=float)
    unknown = (set(depth_values) | set(weight_values)) - set(default_weights)
    if unknown:
        raise ValueError(f"grid overrides contain unknown channels: {sorted(unknown)}")
    cached = load_cached_rankings(args.rankings) if args.rankings else None
    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else None
    report = tune_cached_rankings(
        cached,
        ground_truth,
        default_rrf_k=int(config["rrf_k"]),
        default_depths=default_depths,
        default_weights=default_weights,
        rrf_k_values=args.rrf_k or config["tune_rrf_k"],
        depth_values=depth_values,
        weight_values=weight_values,
        max_candidates=args.max_candidates,
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    persist_tuning_report(report, output)
    payload = report.to_dict()
    payload["output"] = str(output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
