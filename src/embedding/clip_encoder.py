"""Lazy OpenCLIP adapter used for the organizer reference baseline."""

from __future__ import annotations

import gc
import hashlib
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.contracts import TextEncodingDiagnostics
from src.embedding.cuda_health import CUDAEncoderRestartRequired, is_fatal_cuda_error
from src.embedding.text_chunks import chunk_text
from src.utils.offline import enforce_huggingface_offline


def _normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("encoder produced an invalid zero or non-finite vector")
    return values / norms


class CLIPEncoder:
    """OpenCLIP image/text encoder with side-effect-free lazy loading.

    The legacy constructor defaults are retained for API compatibility.  The
    project configuration no longer selects ViT-H/14 on the 4 GiB profile.
    """

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: str = "cuda",
        batch_size: int = 64,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        text_max_chunks: int = 4,
        text_chunk_overlap: int = 8,
    ) -> None:
        if text_max_chunks < 1 or text_chunk_overlap < 0:
            raise ValueError("invalid text chunking limits")
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.local_files_only = local_files_only
        self.text_max_chunks = int(text_max_chunks)
        self.text_chunk_overlap = int(text_chunk_overlap)
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._lifecycle_lock = threading.RLock()
        self._cuda_circuit_open = False

    @property
    def cuda_restart_required(self) -> bool:
        return self._cuda_circuit_open

    def _ensure_cuda_healthy(self) -> None:
        if self._cuda_circuit_open:
            raise CUDAEncoderRestartRequired()

    def _guard_cuda_error(self, error: RuntimeError) -> None:
        if self.device.startswith("cuda") and is_fatal_cuda_error(error):
            self._cuda_circuit_open = True
            raise CUDAEncoderRestartRequired() from None
        raise error

    @property
    def fingerprint(self) -> str:
        """Stable model/preprocessing identity for collection names."""

        value = f"openclip|{self.model_name}|{self.pretrained}|open_clip_default"
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    @property
    def embedding_dim(self) -> int | None:
        """Return the discovered output dimension after model loading."""

        if self._model is None:
            return None
        projection = getattr(self._model, "text_projection", None)
        shape = getattr(projection, "shape", ())
        return int(shape[-1]) if shape else None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.local_files_only:
            # Set process guards before importing OpenCLIP/Hugging Face.  Their
            # modules cache offline constants at import time.
            enforce_huggingface_offline()
        try:
            import open_clip
            import torch
        except ImportError as error:
            raise RuntimeError(
                "OpenCLIP is optional; install the competition ML profile before encoding"
            ) from error
        if self.local_files_only:
            pretrained_config = open_clip.get_pretrained_cfg(self.model_name, self.pretrained)
            if not pretrained_config or not (
                pretrained_config.get("file") or pretrained_config.get("hf_hub")
            ):
                raise RuntimeError(
                    f"OpenCLIP {self.model_name}/{self.pretrained} has no cache-only "
                    "pretrained source"
                )
        kwargs: dict[str, Any] = {}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            **kwargs,
        )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._torch = torch

    @staticmethod
    def _open_image(source: str | Path | bytes | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB").copy()
        if isinstance(source, bytes):
            with Image.open(BytesIO(source)) as image:
                return image.convert("RGB").copy()
        with Image.open(source) as image:
            return image.convert("RGB").copy()

    def encode_images(
        self, image_paths: list[str | Path | bytes | Image.Image]
    ) -> np.ndarray:
        """Serialize model loading and inference across warm-up and queries."""

        with self._lifecycle_lock:
            self._ensure_cuda_healthy()
            try:
                return self._encode_images_unlocked(image_paths)
            except RuntimeError as error:
                self._guard_cuda_error(error)

    def _encode_images_unlocked(
        self, image_paths: list[str | Path | bytes | Image.Image]
    ) -> np.ndarray:
        """Encode images in bounded batches and return normalized float32 rows."""

        if not image_paths:
            return np.empty((0, self.embedding_dim or 0), dtype=np.float32)
        self._load_model()
        rows: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(image_paths), self.batch_size):
                images = []
                for path in image_paths[start : start + self.batch_size]:
                    image = self._open_image(path)
                    try:
                        images.append(self._preprocess(image))
                    finally:
                        image.close()
                batch = self._torch.stack(images).to(self.device)
                encoded = self._model.encode_image(batch)
                rows.append(encoded.float().cpu().numpy())
        return _normalize(np.concatenate(rows, axis=0))

    def encode_text(self, text: str | list[str]) -> np.ndarray:
        """Serialize model loading and inference across warm-up and queries."""

        with self._lifecycle_lock:
            self._ensure_cuda_healthy()
            try:
                return self._encode_text_unlocked(text)
            except RuntimeError as error:
                self._guard_cuda_error(error)

    def _encode_text_unlocked(self, text: str | list[str]) -> np.ndarray:
        """Encode one or more text queries into normalized float32 rows."""

        values = [text] if isinstance(text, str) else list(text)
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("text must contain at least one non-empty string")
        self._load_model()
        with self._torch.inference_mode():
            tokens = self._tokenizer(values).to(self.device)
            encoded = self._model.encode_text(tokens).float().cpu().numpy()
        return _normalize(encoded)

    def encode_text_with_diagnostics(
        self,
        text: str,
    ) -> tuple[np.ndarray, TextEncodingDiagnostics]:
        """Encode tokenizer-aware query chunks without silently dropping content."""

        with self._lifecycle_lock:
            self._ensure_cuda_healthy()
            try:
                self._load_model()
                context_limit = int(getattr(self._model, "context_length", 77))
                content_limit = min(75, max(1, context_limit - 2))
                chunks = chunk_text(
                    self._tokenizer,
                    text,
                    context_limit=context_limit,
                    content_limit=content_limit,
                    max_chunks=self.text_max_chunks,
                    overlap=min(self.text_chunk_overlap, content_limit - 1),
                )
                return self._encode_text_unlocked(list(chunks.texts)), chunks.diagnostics
            except RuntimeError as error:
                self._guard_cuda_error(error)

    def compute_similarity(self, query_emb: np.ndarray, image_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity for already normalized embeddings."""

        return (image_embs @ query_emb.T).squeeze()

    def unload(self) -> None:
        """Serialize model release against warm-up and inference."""

        with self._lifecycle_lock:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        """Release a loaded OpenCLIP model while keeping the adapter reusable."""

        torch = getattr(self, "_torch", None)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None
        gc.collect()
        if torch is not None and self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass
