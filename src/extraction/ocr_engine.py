"""Offline-capable PaddleOCR adapter with Vietnamese-safe normalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.retrieval.query_plan import accent_fold, normalize_nfc
from src.utils.offline import enforce_huggingface_offline

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class OCREngine:
    """Lazy PP-OCRv5 mobile wrapper; CPU is the laptop-profile default."""

    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool = False,
        *,
        model_name: str = "latin_PP-OCRv5_mobile_rec",
        detection_model_name: str = "PP-OCRv5_mobile_det",
        cache_dir: str | Path = "models/paddleocr",
        local_files_only: bool = True,
    ) -> None:
        self.languages = languages or ["vi", "en"]
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.detection_model_name = detection_model_name
        self.cache_dir = Path(cache_dir)
        self.local_files_only = local_files_only
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.local_files_only and not self.cache_dir.is_dir():
            raise RuntimeError(
                f"offline PaddleOCR cache is missing: {self.cache_dir}; run prepare_models.py"
            )
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self.cache_dir.resolve()))
        if self.local_files_only:
            enforce_huggingface_offline()
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        try:
            # On Windows, importing Paddle before CUDA PyTorch can load an
            # incompatible DLL first.  The core pipeline is visual-then-OCR;
            # making that order explicit also keeps direct OCR use reliable.
            import torch  # noqa: F401
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "PaddleOCR is not installed in the competition ML profile"
            ) from error
        device = "gpu" if self.use_gpu else "cpu"
        try:
            self._model = PaddleOCR(
                ocr_version="PP-OCRv5",
                text_detection_model_name=self.detection_model_name,
                text_recognition_model_name=self.model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=device,
            )
        except TypeError:
            # PaddleOCR 2.x compatibility.  Latin recognition preserves
            # Vietnamese; language-specific models can be selected externally.
            self._model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self.use_gpu)

    @staticmethod
    def _record(text: str, confidence: float, polygon: Any) -> dict[str, Any]:
        points = [[float(point[0]), float(point[1])] for point in polygon]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return {
            "text_raw": text,
            "text_nfc": normalize_nfc(text),
            "text_accent_folded": accent_fold(text),
            "confidence": float(confidence),
            "polygon": points,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
        }

    def _parse_legacy(self, output: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        pages = output or []
        if pages and isinstance(pages[0], list) and pages[0] and len(pages[0]) == 2:
            pages = [pages]
        for page in pages:
            for line in page or []:
                if not isinstance(line, (list, tuple)) or len(line) != 2:
                    continue
                polygon, recognition = line
                if not recognition or len(recognition) < 2:
                    continue
                detections.append(self._record(str(recognition[0]), float(recognition[1]), polygon))
        return detections

    def _parse_v3(self, output: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for result in output or []:
            payload = getattr(result, "json", result)
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            if not isinstance(payload, dict):
                continue
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            polygons = payload.get("rec_polys", payload.get("dt_polys", []))
            for text, score, polygon in zip(texts, scores, polygons, strict=False):
                detections.append(self._record(str(text), float(score), polygon))
        return detections

    def extract_text(self, image_path: str | Path) -> list[dict[str, Any]]:
        """Extract regions while preserving raw, NFC, and accent-folded text."""

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._load_model()
        if hasattr(self._model, "predict"):
            return self._parse_v3(self._model.predict(str(path)))
        return self._parse_legacy(self._model.ocr(str(path), cls=True))

    def extract_batch(
        self, image_dir: str | Path, output_path: str | Path | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract all supported images and optionally persist UTF-8 JSON."""

        root = Path(image_dir)
        results = {
            str(path.relative_to(root)): self.extract_text(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.casefold() in _IMAGE_EXTENSIONS
        }
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
