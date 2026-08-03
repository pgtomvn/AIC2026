"""Index canonical per-video embedding matrices into Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.vector_store import VectorStore, batch_collection_name
from src.ingest.ledger import JobLedger
from src.utils.io_helpers import load_config


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument(
        "--collection", choices=("native_frames", "organizer_frames"), default="native_frames"
    )
    parser.add_argument("--batch-id", help="Required for a batch-qualified native collection")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    qdrant = config["services"]["qdrant"]
    dimension = (
        config["models"]["visual"]["embedding_dim"]
        if args.collection == "native_frames"
        else config["models"]["organizer_clip"]["embedding_dim"]
    )
    configured_name = qdrant["collections"][args.collection]
    if args.collection == "native_frames":
        if not args.batch_id:
            parser.error("--batch-id is required for --collection native_frames")
        collection_name = batch_collection_name(configured_name, args.batch_id)
    else:
        collection_name = configured_name
    ledger = None
    if args.batch_id:
        ledger = JobLedger(_project_path(config["data"]["catalog_db"]))
        ledger.initialize()
        ledger.retrieval_generation(args.batch_id)
    store = VectorStore(
        url=qdrant["url"],
        api_key=qdrant.get("api_key"),
        collection_name=collection_name,
        vector_size=int(dimension),
        distance=qdrant["distance"],
    )
    store.create_collection(recreate=args.recreate)
    total = 0
    for matrix_path in sorted(args.embeddings_dir.glob("*.npy")):
        mapping_path = matrix_path.with_suffix(".json")
        if not mapping_path.is_file():
            raise FileNotFoundError(f"missing mapping for {matrix_path.name}")
        matrix = np.load(matrix_path, allow_pickle=False)
        mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
        frames = mapping.get("frames", [])
        if args.batch_id and any(frame.get("batch_id") != args.batch_id for frame in frames):
            raise ValueError(f"mapping batch does not match --batch-id for {matrix_path.name}")
        if matrix.shape != (len(frames), int(dimension)):
            raise ValueError(f"matrix/mapping mismatch for {matrix_path.name}")
        ids = [
            f"{frame['batch_id']}:{frame['video_id']}:{frame['source_frame_idx']}"
            for frame in frames
        ]
        store.upsert(ids, matrix, frames)
        total += len(ids)
    if total == 0:
        raise ValueError("no .npy matrices were found")
    generation = None
    if ledger is not None:
        generation = ledger.bump_retrieval_generation(args.batch_id)
    print(
        json.dumps(
            {
                "indexed": total,
                "collection_count": store.count(),
                "retrieval_generation": generation,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
