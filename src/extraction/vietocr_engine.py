"""Lazy PP-OCR detector + Vietnamese VietOCR recognizer for isolated B02 jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VIETOCR_REVISION = "fe8c3a7fc714aec57ab81cec844eb3adf0c1636c"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VietOCREngine:
    """Read frozen assets only; constructing this object never downloads or loads a model.

    Detection uses the existing B01 ONNX detector on CPU. Text recognition is
    batched on CUDA. This avoids requiring a second CUDA/cuDNN runtime for ONNX.
    """

    def __init__(self, assets: Path, *, confidence: float = 0.5, batch_size: int = 16):
        if not 0 <= confidence <= 1 or batch_size < 1:
            raise ValueError("Invalid recognition settings")
        self.assets = assets
        self.confidence = confidence
        self.batch_size = batch_size
        self._detector: Any = None
        self._recognizer: Any = None
        self._identity: dict | None = None

    @property
    def identity(self) -> dict:
        if self._identity is None:
            manifest = json.loads((self.assets / "assets.json").read_text(encoding="utf-8"))
            expected = {"detector.onnx", "vgg_seq2seq.pth", "config.json"}
            if set(manifest["files"]) != expected or manifest["revision"] != VIETOCR_REVISION:
                raise ValueError("Unexpected VietOCR asset contract")
            for name, sha in manifest["files"].items():
                if digest_file(self.assets / name) != sha:
                    raise ValueError(f"VietOCR asset checksum mismatch: {name}")
            self._identity = {
                "adapter": "b02-vietocr-v1", "recognizer": "vgg_seq2seq",
                "revision": VIETOCR_REVISION, "files": manifest["files"],
                "detector": "PP-OCRv6_det_small", "detector_device": "cpu",
                "recognizer_device": "cuda", "confidence": self.confidence,
                "batch_size": self.batch_size, "min_text_characters": 2,
            }
        return self._identity

    def _load(self) -> None:
        if self._recognizer is not None:
            return
        self.identity
        import rapidocr
        import torch
        from rapidocr.ch_ppocr_det import TextDetector
        from rapidocr.utils.parse_parameters import ParseParams
        from vietocr.tool.predictor import Predictor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; no silent CPU recognition fallback")
        if importlib.metadata.version("rapidocr") != "3.9.2":
            raise RuntimeError("RapidOCR version differs from frozen profile")
        cfg = ParseParams.load(Path(rapidocr.__file__).parent / "config.yaml")
        cfg.Det.model_path = str(self.assets / "detector.onnx")
        cfg.Det.model_root_dir = str(self.assets)
        cfg.Det.engine_cfg = cfg.EngineConfig["onnxruntime"]
        cfg.Det.engine_cfg.use_cuda = False
        cfg.Det.engine_cfg.intra_op_num_threads = 2
        self._detector = TextDetector(cfg.Det)
        config = json.loads((self.assets / "config.json").read_text(encoding="utf-8"))
        config["weights"] = str(self.assets / "vgg_seq2seq.pth")
        config["device"] = "cuda:0"
        config["cnn"]["pretrained"] = False
        config["predictor"]["beamsearch"] = False
        self._recognizer = Predictor(config)

    def extract_text(self, image: Image.Image) -> list[dict]:
        """Detect in BGR, recognize RGB crops; errors propagate, not no-text receipts."""
        self._load()
        from rapidocr.utils.process_img import get_rotate_crop_image

        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        detected = self._detector(bgr)
        if detected.boxes is None:
            return []
        boxes = np.asarray(detected.boxes, dtype=np.float32)
        if boxes.ndim != 3 or boxes.shape[1:] != (4, 2) or not np.isfinite(boxes).all():
            raise ValueError("Invalid detector polygons")
        crops = []
        for box in boxes:
            crop = get_rotate_crop_image(bgr, box.copy())
            if not crop.size:
                raise ValueError("Empty detector crop")
            crops.append(Image.fromarray(crop[:, :, ::-1].copy()))
        results = []
        for start in range(0, len(crops), self.batch_size):
            selected = crops[start:start + self.batch_size]
            texts, probs = self._recognizer.predict_batch(selected, return_prob=True)
            if len(texts) != len(selected) or len(probs) != len(selected):
                raise ValueError("Recognizer dropped crops")
            for offset, (text, probability) in enumerate(zip(texts, probs, strict=True)):
                score = float(probability)
                text = unicodedata.normalize("NFC", str(text)).strip()
                # Upstream produces NaN for an empty decoded token sequence.
                if not text and math.isnan(score):
                    continue
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid recognition confidence")
                if len(text) < 2 or score < self.confidence:
                    continue
                box = boxes[start + offset]
                folded = "".join(c for c in unicodedata.normalize("NFD", text)
                                 if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")
                results.append({"text_raw": text, "text_nfc": text,
                                "text_accent_folded": folded, "confidence": score,
                                "polygon": box.tolist(), "bbox": [float(box[:, 0].min()),
                                float(box[:, 1].min()), float(box[:, 0].max()),
                                float(box[:, 1].max())]})
        return results
