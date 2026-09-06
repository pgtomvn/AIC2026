#!/usr/bin/env python3
"""Compare visual-only, ASR-only, and gated fusion on held-out KIS qrels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.textual_kis import (
    Prediction,
    evaluate_queries,
    load_ground_truth,
)

MODES = ("visual_only", "asr_only", "automatic")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def _search(
    session: requests.Session,
    *,
    api_url: str,
    batch_id: str,
    query_id: str,
    text: str,
    mode: str,
    top_k: int,
    timeout_seconds: float,
) -> tuple[list[Prediction], float, dict[str, Any]]:
    started = time.perf_counter()
    response = session.post(
        f"{api_url.rstrip('/')}/v1/kis/search",
        json={
            "query_id": f"asr-hybrid-{mode}-{query_id}",
            "original_text": text,
            "query_language": "auto",
            "batch_filters": [batch_id],
            "limit": top_k,
            "refine_raw": False,
            "precision_profile": "standard",
            "mode": mode,
        },
        timeout=(2.0, timeout_seconds),
    )
    latency_ms = (time.perf_counter() - started) * 1_000
    response.raise_for_status()
    payload = response.json()
    predictions = [
        Prediction(str(hit["video_id"]), int(hit["source_frame_idx"]))
        for hit in payload.get("hits", ())
        if isinstance(hit, dict)
        and isinstance(hit.get("video_id"), str)
        and isinstance(hit.get("source_frame_idx"), int)
        and not isinstance(hit.get("source_frame_idx"), bool)
    ]
    processing = payload.get("query_processing", {})
    gate = processing.get("asr_gate", {}) if isinstance(processing, dict) else {}
    return predictions, latency_ms, dict(gate) if isinstance(gate, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.top_k < 1 or args.top_k > 1000:
        raise ValueError("top-k must be between 1 and 1000")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    queries = load_ground_truth(args.ground_truth)
    if not queries or any(
        not isinstance(query.text, str) or not query.text.strip() for query in queries
    ):
        raise ValueError("held-out ground truth must include non-empty text for every query")

    reports: dict[str, Any] = {}
    with requests.Session() as session:
        for mode in MODES:
            predictions: dict[str, list[Prediction]] = {}
            latencies: dict[str, float] = {}
            gates: dict[str, dict[str, Any]] = {}
            for query in queries:
                result, latency_ms, gate = _search(
                    session,
                    api_url=args.api_url,
                    batch_id=args.batch_id,
                    query_id=query.query_id,
                    text=str(query.text),
                    mode=mode,
                    top_k=args.top_k,
                    timeout_seconds=args.timeout_seconds,
                )
                predictions[query.query_id] = result
                latencies[query.query_id] = latency_ms
                gates[query.query_id] = gate
            report = evaluate_queries(predictions, queries, latencies_ms=latencies)
            reports[mode] = {
                **report.to_dict(),
                "asr_gate": gates,
                "asr_gate_enabled_queries": sum(
                    bool(value.get("enabled")) for value in gates.values()
                ),
            }

    payload = {
        "contract": "asr-hybrid-heldout-v1",
        "batch_id": args.batch_id,
        "ground_truth": str(args.ground_truth.resolve()),
        "query_count": len(queries),
        "modes": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
