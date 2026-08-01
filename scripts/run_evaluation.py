"""
Run Evaluation Script
=====================
Offline evaluation harness to measure retrieval quality using
the competition's R-Score metric.

Usage:
    python scripts/run_evaluation.py --ground-truth gt.json --predictions pred.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import logger


def compute_r_at_k(predictions: list, ground_truth: dict, k: int) -> float:
    """Compute R@k metric.

    R@k = max over ranks 1..k of R-Score(prediction_i)

    Args:
        predictions: Ordered list of predicted (video_id, frame_id) tuples.
        ground_truth: Dict with ground-truth video_id and frame_id.
        k: Cutoff rank.

    Returns:
        R@k score (0.0 to 1.0).
    """
    # TODO: Implement R-Score calculation per the competition spec
    raise NotImplementedError("compute_r_at_k")


def compute_final_score(
    predictions: list, ground_truth: dict
) -> float:
    """Compute the competition Final Score.

    Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5

    Args:
        predictions: Ordered list of predictions.
        ground_truth: Ground truth for the query.

    Returns:
        Final Score (0.0 to 1.0).
    """
    ks = [1, 5, 20, 50, 100]
    r_scores = [compute_r_at_k(predictions, ground_truth, k) for k in ks]
    return sum(r_scores) / len(ks)


def main() -> None:
    """Run evaluation."""
    parser = argparse.ArgumentParser(description="AIC 2026 Evaluation")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV")
    args = parser.parse_args()

    logger.info(f"Ground truth: {args.ground_truth}")
    logger.info(f"Predictions: {args.predictions}")

    # TODO: Load data, compute metrics, display results
    logger.warning("run_evaluation.py is a stub — implement the evaluation logic.")


if __name__ == "__main__":
    main()
