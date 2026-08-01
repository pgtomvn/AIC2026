"""Regression tests for import-time and constructor filesystem writes."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_logger_import_does_not_create_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("src.utils.logger", None)

    importlib.import_module("src.utils.logger")

    assert list(tmp_path.iterdir()) == []


def test_keyframe_extractor_constructor_does_not_create_output(
    tmp_path: Path,
) -> None:
    from src.extraction.keyframe_extractor import KeyframeExtractor

    output_dir = tmp_path / "nested" / "keyframes"
    extractor = KeyframeExtractor(output_dir=output_dir)

    assert extractor.output_dir == output_dir
    assert not output_dir.exists()
