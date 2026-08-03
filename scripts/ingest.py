"""Compatibility wrapper for the shared native-ingestion CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
