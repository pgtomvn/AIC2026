"""Offline RapidOCR/ONNX adapter for reference-batch provisioning."""

from __future__ import annotations

import hashlib
import io
import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.retrieval.query_plan import accent_fold, normalize_nfc


class RapidOCREngine:
    """Lazy PP-OCRv6 multilingual wrapper with deterministic filtering."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        confidence_threshold: float = 0.50,
        min_text_characters: int = 2,
        local_files_only: bool = True,
        package_version: str = "3.9.2",
        source_revision: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if min_text_characters < 1:
            raise ValueError("min_text_characters must be positive")
        self.model_dir = Path(model_dir)
        self.confidence_threshold = float(confidence_threshold)
        self.min_text_characters = int(min_text_characters)
        self.local_files_only = bool(local_files_only)
        self.package_version = package_version
        self.source_revision = source_revision
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return "rapidocr-ppocrv6-small-multilingual"

    @property
    def fingerprint(self) -> str:
        files = []
        if self.model_dir.is_dir():
            files = []
            for path in sorted(self.model_dir.iterdir()):
                if path.is_file() and path.suffix.casefold() in {".onnx", ".txt"}:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    files.append((path.name, path.stat().st_size, digest))
        payload = {
            "adapter": "rapidocr-reference-v1",
            "package": self.package_version,
            "source_revision": self.source_revision,
            "model": self.model_name,
            "files": files,
            "confidence": self.confidence_threshold,
            "min_chars": self.min_text_characters,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    def _model_paths(self) -> dict[str, Path]:
        if not self.model_dir.is_dir():
            raise RuntimeError("offline RapidOCR model directory is missing")
        values = {path.name.casefold(): path for path in self.model_dir.iterdir() if path.is_file()}

        def select(token: str) -> Path:
            matches = [path for name, path in values.items() if token in name]
            if len(matches) != 1:
                raise RuntimeError(f"offline RapidOCR {token} model is missing or ambiguous")
            return matches[0]

        return {
            "det": select("det"),
            "rec": select("rec"),
            "cls": select("cls"),
            "dict": select("dict"),
        }

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from rapidocr import EngineType, RapidOCR

            actual_version = package_version("rapidocr")
        except (ImportError, PackageNotFoundError) as error:
            raise RuntimeError("RapidOCR is not installed in the locked runtime") from error
        if str(actual_version) != self.package_version:
            raise RuntimeError("RapidOCR package version does not match configuration")
        paths = self._model_paths()
        params = {
            "Global.log_level": "warning",
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.model_path": str(paths["det"].resolve()),
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.model_path": str(paths["rec"].resolve()),
            "Global.use_cls": False,
            "Rec.rec_keys_path": str(paths["dict"].resolve()),
        }
        self._model = RapidOCR(params=params)

    @staticmethod
    def _image_array(source: str | Path | bytes | Image.Image) -> np.ndarray:
        if isinstance(source, Image.Image):
            return np.asarray(source.convert("RGB"))
        if isinstance(source, bytes):
            with Image.open(io.BytesIO(source)) as image:
                return np.asarray(image.convert("RGB"))
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    @staticmethod
    def _raw_rows(output: Any) -> list[tuple[Any, Any, Any]]:
        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", getattr(output, "texts", None))
        scores = getattr(output, "scores", None)
        if boxes is not None and texts is not None and scores is not None:
            return list(zip(boxes, texts, scores, strict=False))
        if isinstance(output, tuple) and len(output) == 2:
            output = output[0]
        if not isinstance(output, (list, tuple)):
            return []
        rows: list[tuple[Any, Any, Any]] = []
        for row in output:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                rows.append((row[0], row[1], row[2]))
        return rows

    def extract_text(self, source: str | Path | bytes | Image.Image) -> list[dict[str, Any]]:
        """Return OCREngine-compatible records from one validated image."""

        self._load_model()
        output = self._model(self._image_array(source))
        detections: list[dict[str, Any]] = []
        for raw_polygon, raw_text, raw_score in self._raw_rows(output):
            try:
                text = normalize_nfc(str(raw_text))
                score = float(raw_score)
                polygon = tuple((float(point[0]), float(point[1])) for point in raw_polygon)
            except (TypeError, ValueError, IndexError):
                continue
            visible_characters = sum(not value.isspace() for value in text)
            if (
                score < self.confidence_threshold
                or visible_characters < self.min_text_characters
                or len(polygon) < 3
            ):
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            detections.append(
                {
                    "text_raw": text,
                    "text_nfc": text,
                    "text_accent_folded": accent_fold(text),
                    "confidence": score,
                    "polygon": [list(point) for point in polygon],
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                }
            )
        return detections

    def unload(self) -> None:
        self._model = None


__all__ = ["RapidOCREngine"]
