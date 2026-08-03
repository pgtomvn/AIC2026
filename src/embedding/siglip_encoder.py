"""Primary SigLIP2 encoder for the laptop and accelerated profiles."""

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
from src.embedding.clip_encoder import _normalize
from src.embedding.text_chunks import chunk_text
from src.utils.offline import enforce_huggingface_offline


class SigLIP2Encoder:
    """Lazy Transformers adapter with CUDA-OOM batch backoff.

    ``local_files_only`` defaults to true so competition-day code never starts
    an implicit network download.  Model acquisition is a separate, explicit
    preparation step.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        *,
        revision: str | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_batch_size: int = 32,
        text_max_length: int = 64,
        cache_dir: str | Path = "models/huggingface",
        local_files_only: bool = True,
        text_max_chunks: int = 4,
        text_chunk_overlap: int = 8,
    ) -> None:
        if batch_size < 1 or max_batch_size < batch_size:
            raise ValueError("invalid encoder batch sizes")
        if text_max_length < 1:
            raise ValueError("text_max_length must be positive")
        if text_max_chunks < 1 or text_chunk_overlap < 0:
            raise ValueError("invalid text chunking limits")
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.text_max_length = text_max_length
        self.cache_dir = Path(cache_dir)
        self.local_files_only = local_files_only
        self.text_max_chunks = int(text_max_chunks)
        self.text_chunk_overlap = int(text_chunk_overlap)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._embedding_dim: int | None = None
        self._lifecycle_lock = threading.RLock()

    @property
    def fingerprint(self) -> str:
        value = (
            f"transformers|{self.model_name}|{self.revision or 'unversioned'}|"
            f"224|processor=slow|padding=max_length|text_max_length={self.text_max_length}"
        )
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.local_files_only:
            enforce_huggingface_offline()
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "SigLIP2 requires the optional PyTorch/Transformers competition profile"
            ) from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for SigLIP2 but is not available")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        common = {
            "cache_dir": str(self.cache_dir),
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            common["revision"] = self.revision
        processor = AutoProcessor.from_pretrained(self.model_name, use_fast=False, **common)
        model = AutoModel.from_pretrained(self.model_name, dtype=dtype, **common)
        model_text_limit = int(model.config.text_config.max_position_embeddings)
        if self.text_max_length > model_text_limit:
            raise ValueError(
                f"text_max_length={self.text_max_length} exceeds the model limit "
                f"of {model_text_limit}"
            )
        self._processor = processor
        self._model = model.to(self.device).eval()
        self._torch = torch
        projection_dim = getattr(model.config, "projection_dim", None)
        if projection_dim is not None:
            self._embedding_dim = int(projection_dim)

    def _is_oom(self, error: RuntimeError) -> bool:
        return "out of memory" in str(error).casefold() and self.device.startswith("cuda")

    def _encode_image_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model.get_image_features(**inputs)
        return output.float().cpu().numpy()

    @staticmethod
    def _open_image(source: str | Path | bytes | Image.Image) -> Image.Image:
        """Return an owned RGB image for a path, byte payload, or PIL image."""

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
            return self._encode_images_unlocked(image_paths)

    def _encode_images_unlocked(
        self, image_paths: list[str | Path | bytes | Image.Image]
    ) -> np.ndarray:
        """Encode bounded image inputs with automatic CUDA-OOM batch backoff."""

        if not image_paths:
            return np.empty((0, self.embedding_dim or 0), dtype=np.float32)
        self._load_model()
        rows: list[np.ndarray] = []
        position = 0
        live_batch = self.batch_size
        successes = 0
        while position < len(image_paths):
            paths = image_paths[position : position + live_batch]
            images: list[Image.Image] = []
            try:
                for path in paths:
                    images.append(self._open_image(path))
                encoded = self._encode_image_batch(images)
            except RuntimeError as error:
                if not self._is_oom(error) or live_batch == 1:
                    raise
                self._torch.cuda.empty_cache()
                live_batch = max(1, live_batch // 2)
                successes = 0
                continue
            finally:
                for image in images:
                    image.close()
            rows.append(encoded)
            position += len(paths)
            successes += 1
            if successes >= 4 and live_batch < self.max_batch_size:
                live_batch = min(self.max_batch_size, live_batch * 2)
                successes = 0
        output = _normalize(np.concatenate(rows, axis=0))
        self._embedding_dim = int(output.shape[1])
        return output

    def encode_text(self, text: str | list[str]) -> np.ndarray:
        """Serialize model loading and inference across warm-up and queries."""

        with self._lifecycle_lock:
            return self._encode_text_unlocked(text)

    def _encode_text_unlocked(self, text: str | list[str]) -> np.ndarray:
        """Encode multilingual text with the matching SigLIP2 text tower."""

        values = [text] if isinstance(text, str) else list(text)
        if not values or any(not value.strip() for value in values):
            raise ValueError("text must contain at least one non-empty string")
        self._load_model()
        inputs = self._processor(
            text=values,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model.get_text_features(**inputs)
        normalized = _normalize(output.float().cpu().numpy())
        self._embedding_dim = int(normalized.shape[1])
        return normalized

    def encode_text_with_diagnostics(
        self,
        text: str,
    ) -> tuple[np.ndarray, TextEncodingDiagnostics]:
        """Encode actual-token chunks and return explicit truncation diagnostics."""

        with self._lifecycle_lock:
            self._load_model()
            tokenizer = self._processor.tokenizer
            special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
            content_limit = max(1, self.text_max_length - special_tokens)
            chunks = chunk_text(
                tokenizer,
                text,
                context_limit=self.text_max_length,
                content_limit=content_limit,
                max_chunks=self.text_max_chunks,
                overlap=min(self.text_chunk_overlap, content_limit - 1),
            )
            return self._encode_text_unlocked(list(chunks.texts)), chunks.diagnostics

    def unload(self) -> None:
        """Serialize model release against warm-up and inference."""

        with self._lifecycle_lock:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        """Release loaded model state and any cached CUDA allocations.

        The adapter remains reusable: the next encode call loads the same
        pinned model again.  Native ingestion calls this before handing the
        laptop GPU from visual indexing to ASR.
        """

        torch = self._torch
        self._model = None
        self._processor = None
        self._torch = None
        gc.collect()
        if torch is not None and self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass
