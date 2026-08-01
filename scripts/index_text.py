"""
Index Text Script
=================
Push OCR-extracted text and Whisper ASR transcripts into Meilisearch
for BM25 keyword search.

Usage:
    python scripts/index_text.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import load_config
from src.utils.logger import logger, setup_logger_from_config


def main() -> None:
    """Push text data to Meilisearch."""
    config = load_config()
    setup_logger_from_config(config)

    logger.info("Loading OCR and ASR data...")

    # TODO:
    #   1. Load OCR JSON files from data/ocr/
    #   2. Load ASR JSON files from data/asr/
    #   3. Merge into documents with id, video_id, frame_id, text_content
    #   4. Initialize TextStore from config
    #   5. Create index and add documents
    #   6. Log final document count

    logger.warning("index_text.py is a stub — implement the pipeline above.")


if __name__ == "__main__":
    main()
