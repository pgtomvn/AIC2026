"""
Index Vectors Script
====================
Push precomputed CLIP embeddings into Qdrant for similarity search.

Usage:
    python scripts/index_vectors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import load_config
from src.utils.logger import logger, setup_logger_from_config


def main() -> None:
    """Push embeddings to Qdrant."""
    config = load_config()
    setup_logger_from_config(config)

    logger.info("Loading embeddings from disk...")

    # TODO:
    #   1. Load embeddings .npy and mapping JSON
    #   2. Initialize VectorStore from config
    #   3. Create/recreate collection
    #   4. Batch upsert embeddings with frame metadata payloads
    #   5. Log final count

    logger.warning("index_vectors.py is a stub — implement the pipeline above.")


if __name__ == "__main__":
    main()
