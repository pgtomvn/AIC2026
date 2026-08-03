"""Evaluate ranked Textual KIS predictions against source-frame intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import CompetitionSpec
from src.evaluation.textual_kis import evaluate_queries, load_ground_truth, load_predictions


def compute_r_at_k(predictions: list, ground_truth: dict, k: int) -> float:
    """Backward-compatible single-query interval recall helper."""
    from src.evaluation.textual_kis import _ground_truth_rows, first_relevant_rank, recall_at_k

    row = {"query_id": "default", **ground_truth}
    if "frame_id" in row and not any(
        name in row for name in ("source_frame_idx", "frame_idx", "start_frame_idx")
    ):
        row["source_frame_idx"] = row.pop("frame_id")
    query = _ground_truth_rows([row])[0]
    return recall_at_k(first_relevant_rank(predictions, query), k)


def compute_final_score(predictions: list, ground_truth: dict) -> float:
    """Backward-compatible mean historical score across standard cutoffs."""
    spec = CompetitionSpec.historical_2025()
    return sum(compute_r_at_k(predictions, ground_truth, k) for k in spec.cutoffs) / len(
        spec.cutoffs
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AIC Textual KIS source-frame predictions"
    )
    parser.add_argument("--ground-truth", type=Path, required=True, help="CSV or JSON intervals")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV or JSON rankings")
    parser.add_argument("--spec", type=Path, help="Optional CompetitionSpec JSON")
    parser.add_argument("--latencies", type=Path, help="Optional JSON query_id-to-ms mapping")
    parser.add_argument("--output", type=Path, help="Optional report JSON destination")
    parser.add_argument(
        "--compact", action="store_true", help="Omit per-query metrics from displayed JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load files, evaluate them, and emit deterministic JSON."""
    args = _parser().parse_args(argv)
    spec = CompetitionSpec.historical_2025()
    if args.spec:
        with args.spec.open("r", encoding="utf-8-sig") as handle:
            spec = CompetitionSpec.from_dict(json.load(handle))
    latencies = None
    if args.latencies:
        with args.latencies.open("r", encoding="utf-8-sig") as handle:
            raw_latencies = json.load(handle)
        if not isinstance(raw_latencies, dict):
            raise ValueError("latencies JSON must map query IDs to milliseconds")
        latencies = {str(key): float(value) for key, value in raw_latencies.items()}

    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    if set(predictions) == {"default"} and len(ground_truth) == 1:
        predictions = {ground_truth[0].query_id: predictions["default"]}
    report = evaluate_queries(predictions, ground_truth, spec=spec, latencies_ms=latencies)
    payload = report.to_dict()
    if args.compact:
        payload.pop("per_query", None)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
