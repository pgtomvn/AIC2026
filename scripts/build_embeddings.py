"""
Build Embeddings Script
=======================
Batch encode all keyframe images into CLIP embeddings and save to disk.

Usage:
    python scripts/build_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import load_config, ensure_dirs
from src.utils.logger import logger


def main() -> None:
    """Batch embedding generation pipeline."""
    config = load_config()
    clip_config = config["models"]["clip"]

    keyframes_dir = PROJECT_ROOT / config["data"]["raw_keyframes"]
    embeddings_dir = PROJECT_ROOT / config["data"]["embeddings"]
    ensure_dirs(embeddings_dir)

    logger.info(f"Loading CLIP model: {clip_config['name']} ({clip_config['pretrained']})")

    # TODO:
    #   1. Initialize CLIPEncoder from config
    #   2. Glob all keyframe images
    #   3. Batch encode images
    #   4. Save embeddings as .npy files
    #   5. Save mapping (frame_path -> embedding_index) as JSON

    logger.warning("build_embeddings.py is a stub — implement the pipeline above.")


if __name__ == "__main__":
    main()
