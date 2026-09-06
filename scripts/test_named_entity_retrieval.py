"""Offline entity benchmark with an optional unlabeled live retrieval probe."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query.entity_extractor import EntityExtractor, merge_entity_constraints  # noqa: E402
from src.query.query_decomposer import decompose_query, mask_entity_mentions  # noqa: E402

CASES = (
    {
        "id": "charity_location",
        "query": (
            "Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA "
            "tại một xã thuộc tỉnh Khánh Hòa."
        ),
        "expected": ("FANA", "Khánh Hòa"),
        "english": "a charity club named FANA distributing gifts in a rural village in Khánh Hòa",
    },
    {
        "id": "historical_temple",
        "query": (
            "Trong đoạn video có 2 câu thơ của một nhà thơ ca ngợi anh hùng "
            "Nguyễn Trung Trực trong đình thần Nguyễn Trung Trực tại Kiên Giang."
        ),
        "expected": ("Nguyễn Trung Trực", "Kiên Giang"),
        "english": (
            "two verses praise hero Nguyễn Trung Trực in a traditional temple "
            "in Kiên Giang"
        ),
    },
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _offline(iterations: int) -> dict[str, Any]:
    extractor = EntityExtractor()
    timings: list[float] = []
    reports: list[dict[str, Any]] = []
    true_positive = predicted = expected_count = 0
    for case in CASES:
        extracted = extractor.extract(case["query"])
        predicted_values = tuple(value.text for value in extracted)
        expected_values = tuple(case["expected"])
        predicted_set, expected_set = set(predicted_values), set(expected_values)
        true_positive += len(predicted_set & expected_set)
        predicted += len(predicted_set)
        expected_count += len(expected_set)
        masked, mentions = mask_entity_mentions(case["query"], extracted)
        decomposition = decompose_query(
            case["query"],
            merge_entity_constraints(extracted),
            english_text=case["english"],
        )
        for _ in range(iterations):
            started = time.perf_counter()
            extractor.extract(case["query"])
            timings.append((time.perf_counter() - started) * 1_000)
        reports.append(
            {
                "id": case["id"],
                "expected_entities": expected_values,
                "extracted_entities": predicted_values,
                "exact_set_match": predicted_set == expected_set,
                "masked_mention_count": len(mentions),
                "masked_source_contains_entities": any(
                    value in masked for value in expected_values
                ),
                "visual_description_en": decomposition.visual_description_en,
                "lexical_must_match": decomposition.lexical_must_match,
            }
        )
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected_count if expected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p95 = _percentile(timings, 0.95)
    return {
        "mode": "offline_labeled_entities",
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "latency_ms": {
            "p50": statistics.median(timings) if timings else 0.0,
            "p95": p95,
            "max": max(timings, default=0.0),
        },
        "cases": reports,
        "passed": f1 == 1.0 and p95 <= 5.0 and all(
            value["exact_set_match"] and not value["masked_source_contains_entities"]
            for value in reports
        ),
    }


def _live(api_url: str, batch_id: str) -> dict[str, Any]:
    import requests

    reports: list[dict[str, Any]] = []
    for case in CASES:
        started = time.perf_counter()
        response = requests.post(
            f"{api_url.rstrip('/')}/v1/kis/search",
            json={
                "query_id": f"entity-{case['id']}",
                "original_text": case["query"],
                "batch_filters": [batch_id],
                "limit": 20,
                "query_language": "vi",
                "refine_raw": False,
            },
            timeout=30,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        response.raise_for_status()
        payload = response.json()
        processing = payload.get("query_processing", {})
        reports.append(
            {
                "id": case["id"],
                "latency_ms": latency_ms,
                "hit_count": len(payload.get("hits", ())),
                "channel_statuses": payload.get("channel_statuses", {}),
                "entities": processing.get("entities", ()),
                "matched_entity_count": processing.get("matched_entity_count", 0),
            }
        )
    return {
        "mode": "live_unlabeled_retrieval",
        "notice": "No retrieval accuracy or precision is reported without ground truth.",
        "cases": reports,
        "passed": all(value["hit_count"] >= 0 for value in reports),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    result = _offline(args.iterations)
    if args.live_api:
        result = {"offline": result, "live": _live(args.api_url, args.batch_id)}
        passed = result["offline"]["passed"] and result["live"]["passed"]
    else:
        passed = result["passed"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
