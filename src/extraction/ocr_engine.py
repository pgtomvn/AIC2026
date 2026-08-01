"""
OCR Engine
==========
Extract on-screen text from keyframe images using PaddleOCR.
Handles Vietnamese and English text detection & recognition.

Usage:
    engine = OCREngine(languages=["vi", "en"])
    results = engine.extract_text("path/to/frame.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class OCREngine:
    """PaddleOCR wrapper for on-screen text extraction.

    Args:
        languages: List of language codes for recognition.
        use_gpu: Whether to use GPU acceleration.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool = True,
    ) -> None:
        self.languages = languages or ["vi", "en"]
        self.use_gpu = use_gpu
        self._model = None  # Lazy-loaded PaddleOCR instance

    def _load_model(self) -> None:
        """Lazy-load the PaddleOCR model."""
        # TODO: from paddleocr import PaddleOCR
        #       self._model = PaddleOCR(use_angle_cls=True, lang=..., use_gpu=...)
        raise NotImplementedError("OCREngine._load_model")

    def extract_text(self, image_path: str | Path) -> list[dict]:
        """Extract text regions from a single image.

        Args:
            image_path: Path to the keyframe image.

        Returns:
            List of dicts with keys: text, confidence, bbox (x1,y1,x2,y2).
        """
        # TODO: Run OCR on image, parse results into structured dicts
        raise NotImplementedError("OCREngine.extract_text")

    def extract_batch(
        self, image_dir: str | Path, output_path: Optional[str | Path] = None
    ) -> dict[str, list[dict]]:
        """Extract text from all images in a directory.

        Args:
            image_dir: Directory containing keyframe images.
            output_path: Optional path to save results as JSON.

        Returns:
            Dict mapping filename -> list of text detections.
        """
        # TODO: Iterate images, call extract_text, optionally save to JSON
        raise NotImplementedError("OCREngine.extract_batch")
