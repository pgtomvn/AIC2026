"""
AIC 2026 — Multimodal Video Search & Retrieval Engine
=====================================================

Top-level package. Provides centralized config loading and logging setup.
"""

from pathlib import Path

# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

__version__ = "0.1.0"
