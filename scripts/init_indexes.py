"""Create or validate every configured Textual KIS service index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import build_pipeline
from src.utils.io_helpers import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate-empty",
        action="store_true",
        help="Recreate configured indexes (destructive; intended only before ingestion)",
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Batch identity used for native visual and semantic-text collections",
    )
    args = parser.parse_args(argv)
    pipeline = build_pipeline(load_config())
    initialized: dict[str, str] = {}
    for channel in pipeline.dense_channels:
        store = (
            channel.store.for_batch(args.batch_id)
            if hasattr(channel.store, "for_batch")
            else channel.store
        )
        store.create_collection(recreate=args.recreate_empty)
        initialized[channel.name] = store.collection_name
    for channel in pipeline.lexical_channels:
        channel.store.create_index()
        initialized[channel.name] = channel.store.index_name
    print(json.dumps(initialized, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
