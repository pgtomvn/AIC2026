"""Validate and export a Textual KIS top-100 file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import CompetitionSpec
from src.ingest.artifacts import ArtifactRepository
from src.ingest.ledger import JobLedger
from src.submission import FrameIndexEvidence, export_submission
from src.utils.io_helpers import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="JSON list of canonical hits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args(argv)
    results = json.loads(args.results.read_text(encoding="utf-8-sig"))
    if not isinstance(results, list):
        raise ValueError("results JSON must be a list")
    config = load_config()
    spec = CompetitionSpec.from_config(config.get("competition", {}))
    if args.spec:
        spec = CompetitionSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8-sig")))
    artifact_path = Path(config["data"]["artifact_db"])
    ledger_path = Path(config["data"]["catalog_db"])
    if not artifact_path.is_absolute():
        artifact_path = PROJECT_ROOT / artifact_path
    if not ledger_path.is_absolute():
        ledger_path = PROJECT_ROOT / ledger_path
    repository = ArtifactRepository(artifact_path)
    repository.initialize()
    ledger = JobLedger(ledger_path)
    ledger.initialize()
    identities = [
        (
            spec.normalize_video_id(item.get("video_id", item.get("video_name"))),
            int(item.get("source_frame_idx", item.get("frame_idx"))),
        )
        for item in results
        if isinstance(item, dict)
    ]
    existing = repository.existing_source_frames(args.batch_id, identities)
    mappings = repository.frame_index_mappings_for_sources(args.batch_id, identities)
    target = export_submission(
        results,
        args.output,
        spec=spec,
        batch_id=args.batch_id,
        is_batch_active=ledger.is_batch_active,
        is_canonical_source_frame=lambda batch, video, frame: (
            batch == args.batch_id and (video, frame) in existing
        ),
        frame_index_evidence=FrameIndexEvidence(mappings),
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
