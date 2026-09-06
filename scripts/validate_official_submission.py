"""Validate a submission against an explicit historical or future rule pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.codabench_query_id import parse_codabench_query_filename
from src.evaluation.textual_kis import (
    GroundTruthInterval,
    evaluate_query,
    load_ground_truth,
)
from src.evaluation.trake import (
    TRAKEGroundTruthQuery,
    TRAKEStageInterval,
    evaluate_trake_query,
)
from src.evaluation.visual_qa import QAGroundTruthQuery, evaluate_qa_query
from src.export.preflight import (
    CODABENCH_SPEC_ID,
    BundleEntry,
    BundleValidationContext,
    BundleValidationError,
    validate_bundle_entries,
)
from src.ingest.artifacts import ArtifactRepository
from src.ingest.ledger import JobLedger, LedgerError
from src.official_submission import (
    OfficialSubmissionValidationError,
    parse_submission_csv,
    prediction_rows,
    validate_parsed_submission,
)
from src.utils.io_helpers import load_config


def _path(value: str, *, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--task", choices=("kis", "qa", "trake"), required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--artifact-db", type=Path)
    parser.add_argument("--catalog-db", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--query-id")
    parser.add_argument("--score", action="store_true")
    parser.add_argument(
        "--require-qa-quotes",
        action="store_true",
        help="Require the QA answer field to be explicitly double-quoted",
    )
    return parser


def _json_query(path: Path, query_id: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict) and "queries" in value:
        value = value["queries"]
    if isinstance(value, dict):
        selected = value.get(query_id)
        if isinstance(selected, dict):
            return {"query_id": query_id, **selected}
        if value.get("query_id") == query_id:
            return value
    if isinstance(value, list):
        for row in value:
            if isinstance(row, dict) and row.get("query_id") == query_id:
                return row
    raise OfficialSubmissionValidationError(
        "ground_truth_query_missing", "query ID is absent from ground truth"
    )


def _score(parsed, ground_truth: Path, query_id: str) -> dict[str, object]:
    predictions = prediction_rows(parsed)
    if parsed.task == "kis":
        queries = load_ground_truth(ground_truth)
        selected = next((item for item in queries if item.query_id == query_id), None)
        if selected is None:
            raise OfficialSubmissionValidationError(
                "ground_truth_query_missing", "query ID is absent from ground truth"
            )
        return evaluate_query(predictions, selected).to_dict()

    row = _json_query(ground_truth, query_id)
    raw_intervals = row.get("intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise OfficialSubmissionValidationError(
            "ground_truth_invalid", "ground truth requires non-empty intervals"
        )
    if parsed.task == "qa":
        intervals = tuple(
            GroundTruthInterval(
                str(item["video_id"]),
                int(item["start_frame_idx"]),
                int(item["end_frame_idx"]),
            )
            for item in raw_intervals
            if isinstance(item, dict)
        )
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise OfficialSubmissionValidationError(
                "ground_truth_invalid", "QA ground truth requires answers"
            )
        query = QAGroundTruthQuery(
            query_id=query_id,
            intervals=intervals,
            answers=tuple(str(answer) for answer in answers),
            question=None if row.get("question") is None else str(row["question"]),
        )
        return evaluate_qa_query(predictions, query).to_dict()

    stages = tuple(
        TRAKEStageInterval(
            int(item["stage_index"]),
            int(item["start_frame_idx"]),
            int(item["end_frame_idx"]),
        )
        for item in raw_intervals
        if isinstance(item, dict)
    )
    video_id = row.get("video_id")
    if not isinstance(video_id, str):
        raise OfficialSubmissionValidationError(
            "ground_truth_invalid", "TRAKE ground truth requires video_id"
        )
    query = TRAKEGroundTruthQuery(query_id, video_id, stages)
    return evaluate_trake_query(predictions, query).to_dict()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.score and args.ground_truth is None:
            raise OfficialSubmissionValidationError(
                "ground_truth_required_for_scoring",
                "--score requires --ground-truth",
            )
        if args.score and (not isinstance(args.query_id, str) or not args.query_id.strip()):
            raise OfficialSubmissionValidationError(
                "query_id_required_for_scoring",
                "--score requires --query-id for a single ranked CSV",
            )
        parsed = parse_submission_csv(
            args.submission,
            task=args.task,
            spec_id=args.spec,
            require_qa_answer_quotes=args.require_qa_quotes,
        )
        config = load_config()
        artifact_path = args.artifact_db or _path(config["data"]["artifact_db"])
        catalog_path = args.catalog_db or _path(config["data"]["catalog_db"])
        repository = ArtifactRepository(artifact_path)
        repository.initialize()
        ledger = JobLedger(catalog_path)
        ledger.initialize()
        identities = tuple(
            (row[0].strip().upper().removesuffix(".MP4"), int(frame))
            for row in parsed.rows
            for frame in (row[1:] if args.task == "trake" else row[1:2])
        )
        existing = repository.existing_source_frames(args.batch_id, identities)

        def is_active(batch_id: str) -> bool:
            try:
                return ledger.is_batch_active(batch_id)
            except LedgerError:
                configured = config.get("readiness", {}).get("batches", {}).get(batch_id, {})
                return configured.get("kind") == "reference" and bool(
                    repository.batch_counts(batch_id).frame_count
                )

        def canonical(batch_id: str, video_id: str, frame_idx: int) -> bool:
            return batch_id == args.batch_id and (video_id, frame_idx) in existing

        def timestamp(batch_id: str, video_id: str, frame_idx: int) -> int | None:
            if batch_id != args.batch_id:
                return None
            frame = repository.get_source_frame(batch_id, video_id, frame_idx)
            return None if frame is None else frame.timestamp_ms

        if args.spec == CODABENCH_SPEC_ID:
            try:
                parsed_query_id, parsed_task = parse_codabench_query_filename(
                    args.submission.name
                )
            except (TypeError, ValueError):
                raise OfficialSubmissionValidationError(
                    "submission_filename_invalid",
                    "Codabench filename does not match its task",
                ) from None
            if parsed_task != args.task:
                raise OfficialSubmissionValidationError(
                    "submission_filename_invalid",
                    "Codabench filename does not match its task",
                )
            data = args.submission.read_bytes()
            entry = BundleEntry(
                task=args.task,
                query_id=parsed_query_id,
                csv_bytes=data,
                batch_id=args.batch_id,
                source_fingerprint=hashlib.sha256(data).hexdigest(),
                stage_count=parsed.stage_count if args.task == "trake" else None,
            )
            try:
                validated = validate_bundle_entries(
                    (entry,),
                    context=BundleValidationContext(
                        is_batch_active=is_active,
                        is_canonical_source_frame=canonical,
                        get_canonical_timestamp_ms=timestamp,
                    ),
                    mode="current",
                    require_qa_answer_quotes=args.require_qa_quotes,
                )
            except BundleValidationError as error:
                issue = error.issues[0]
                raise OfficialSubmissionValidationError(
                    issue.code,
                    issue.message,
                    row=issue.row,
                    line=issue.line,
                    suggestion=issue.suggestion,
                ) from None
            parsed = validated[0].parsed
            report = validate_parsed_submission(
                parsed,
                batch_id=args.batch_id,
                is_batch_active=is_active,
                is_canonical_source_frame=canonical,
                get_canonical_timestamp_ms=timestamp,
                ground_truth_supplied=args.ground_truth is not None,
            ).to_dict()
        else:
            report = validate_parsed_submission(
                parsed,
                batch_id=args.batch_id,
                is_batch_active=is_active,
                is_canonical_source_frame=canonical,
                get_canonical_timestamp_ms=timestamp,
                ground_truth_supplied=args.ground_truth is not None,
            ).to_dict()
        if args.ground_truth is not None:
            report["ground_truth"] = {
                "path_name": args.ground_truth.name,
                "scored": bool(args.score),
            }
        if args.score:
            report["metrics"] = _score(
                parsed,
                args.ground_truth,
                args.query_id.strip(),
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except OfficialSubmissionValidationError as exc:
        error_payload = {"status": "invalid", "code": exc.code, "message": str(exc)}
        for name in ("line", "row", "suggestion"):
            value = getattr(exc, name, None)
            if value is not None:
                error_payload[name] = value
        print(
            json.dumps(error_payload, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "code": "submission_validation_failed",
                    "message": type(exc).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
