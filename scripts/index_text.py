"""Index canonical OCR or ASR segments into a Meilisearch lexical index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.text_store import TextStore
from src.ingest.artifacts import ArtifactRepository
from src.ingest.ledger import JobLedger
from src.utils.io_helpers import load_config


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--kind", choices=("ocr", "asr"), required=True)
    args = parser.parse_args(argv)
    config = load_config()
    meili = config["services"]["meilisearch"]
    store = TextStore(
        url=meili["url"],
        api_key=meili.get("api_key"),
        index_name=meili["indexes"][args.kind],
    )
    repository = ArtifactRepository(_project_path(config["data"]["artifact_db"]))
    repository.initialize()
    ledger = JobLedger(_project_path(config["data"]["catalog_db"]))
    ledger.initialize()
    ledger.retrieval_generation(args.batch_id)
    documents = [
        {"id": segment.segment_id, **segment.to_dict(), "text": segment.text_nfc}
        for segment in repository.iter_text_segments(
            batch_id=args.batch_id,
            kind=args.kind,
            index_ready_only=True,
        )
    ]
    if not documents:
        raise ValueError("no canonical text segments matched the request")
    store.create_index()
    store.add_documents(documents)
    generation = ledger.bump_retrieval_generation(args.batch_id)
    print(
        json.dumps(
            {
                "indexed": len(documents),
                "index": store.index_name,
                "retrieval_generation": generation,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
