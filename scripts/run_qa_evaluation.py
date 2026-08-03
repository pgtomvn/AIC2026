"""Evaluate ranked Visual Q&A predictions against intervals and exact answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACompetitionSpec
from src.evaluation.textual_kis import GroundTruthInterval
from src.evaluation.visual_qa import QAGroundTruthQuery, evaluate_qa_queries


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _ground_truth(value: Any) -> tuple[QAGroundTruthQuery, ...]:
    if not isinstance(value, list):
        raise ValueError("ground-truth JSON must be a list")
    queries: list[QAGroundTruthQuery] = []
    for index, raw in enumerate(value):
        row = _mapping(raw, f"ground truth row {index}")
        intervals_raw = row.get("intervals")
        if not isinstance(intervals_raw, list):
            raise ValueError(f"ground truth row {index} intervals must be a list")
        intervals = tuple(
            GroundTruthInterval(**_mapping(item, "ground-truth interval"))
            for item in intervals_raw
        )
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"ground truth row {index} answers must be a list")
        queries.append(
            QAGroundTruthQuery(
                query_id=str(row["query_id"]),
                intervals=intervals,
                answers=tuple(str(answer) for answer in answers),
                event_description=row.get("event_description"),
                question=row.get("question"),
                intent_class=str(row.get("intent_class", "unspecified")),
            )
        )
    return tuple(queries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--answer-spec-id",
        default=QACompetitionSpec.historical_numeric().spec_id,
    )
    parser.add_argument("--spec", type=Path, help="Optional QACompetitionSpec JSON")
    parser.add_argument("--latencies", type=Path, help="Optional query_id-to-ms JSON")
    parser.add_argument("--provenance", type=Path, help="Optional per-query provider/cache JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    spec = QACompetitionSpec.for_spec_id(args.answer_spec_id)
    if args.spec:
        spec = QACompetitionSpec.from_dict(
            json.loads(args.spec.read_text(encoding="utf-8-sig"))
        )
    ground_truth = _ground_truth(
        json.loads(args.ground_truth.read_text(encoding="utf-8-sig"))
    )
    predictions_raw = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    predictions = _mapping(predictions_raw, "predictions")
    if any(not isinstance(items, list) for items in predictions.values()):
        raise ValueError("predictions must map every query ID to a list")
    latencies = None
    if args.latencies:
        latencies_raw = _mapping(
            json.loads(args.latencies.read_text(encoding="utf-8-sig")), "latencies"
        )
        latencies = {str(key): float(value) for key, value in latencies_raw.items()}
    provenance = None
    if args.provenance:
        provenance_raw = _mapping(
            json.loads(args.provenance.read_text(encoding="utf-8-sig")), "provenance"
        )
        if any(not isinstance(items, list) for items in provenance_raw.values()):
            raise ValueError("provenance must map every query ID to a list")
        provenance = provenance_raw
    report = evaluate_qa_queries(
        predictions,
        ground_truth,
        spec=spec,
        latencies_ms=latencies,
        provenance=provenance,
    )
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
