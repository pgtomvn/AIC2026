"""
Extract Data Script
===================
Unzip raw data archives and organize them into the data/ directory structure.

Usage:
    python scripts/extract_data.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import ensure_dirs, load_config
from src.utils.logger import logger


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract a zip archive to the specified directory.

    Args:
        archive_path: Path to the .zip file.
        output_dir: Directory to extract into.
    """
    logger.info(f"Extracting {archive_path.name} → {output_dir}")
    ensure_dirs(output_dir)

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)

    logger.success(f"Extracted {archive_path.name} ({len(list(output_dir.rglob('*')))} files)")


def main() -> None:
    """Main extraction pipeline."""
    config = load_config()

    archives = {
        "Keyframes_L21.zip": Path(config["data"]["raw_keyframes"]),
        "Videos_L21_a.zip": Path(config["data"]["raw_videos"]),
    }

    for archive_name, output_dir in archives.items():
        archive_path = PROJECT_ROOT / archive_name
        if archive_path.exists():
            extract_archive(archive_path, PROJECT_ROOT / output_dir)
        else:
            logger.warning(f"Archive not found: {archive_path}")

    # Create remaining data directories
    ensure_dirs(
        PROJECT_ROOT / config["data"]["embeddings"],
        PROJECT_ROOT / config["data"]["metadata"],
        PROJECT_ROOT / config["data"]["ocr"],
        PROJECT_ROOT / config["data"]["asr"],
    )

    logger.success("Data extraction complete!")


if __name__ == "__main__":
    main()
