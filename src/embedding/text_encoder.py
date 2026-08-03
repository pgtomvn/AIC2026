"""Vietnamese semantic text embeddings for ASR and metadata segments."""

from __future__ import annotations

import gc
import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import TextEncodingDiagnostics
from src.embedding.clip_encoder import _normalize
from src.embedding.text_chunks import chunk_text
from src.utils.offline import enforce_huggingface_offline


class VietnameseTextEncoder:
    """Lazy mean-pooled Transformers encoder with an offline-safe default."""

    def __init__(
        self,
        model_name: str = "dangvantuan/vietnamese-embedding",
        *,
        revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        cache_dir: str | Path = "models/huggingface",
        local_files_only: bool = True,
        text_max_chunks: int = 4,
        text_chunk_overlap: int = 8,
    ) -> None:
        if text_max_chunks < 1 or text_chunk_overlap < 0:
            raise ValueError("invalid text chunking limits")
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.local_files_only = local_files_only
        self.text_max_chunks = int(text_max_chunks)
        self.text_chunk_overlap = int(text_chunk_overlap)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._embedding_dim: int | None = None
        self._lifecycle_lock = threading.RLock()

    @property
    def fingerprint(self) -> str:
        value = (
            f"transformers|{self.model_name}|{self.revision or 'unversioned'}|mean_pool|normalized"
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
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Vietnamese semantic encoding requires the optional Transformers profile"
            ) from error
        common = {"cache_dir": str(self.cache_dir), "local_files_only": self.local_files_only}
        if self.revision is not None:
            common["revision"] = self.revision
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common)
        self._model = AutoModel.from_pretrained(self.model_name, **common).to(self.device).eval()
        self._torch = torch
        hidden_size = getattr(self._model.config, "hidden_size", None)
        self._embedding_dim = int(hidden_size) if hidden_size is not None else None

    def encode_text(self, text: str | list[str]) -> np.ndarray:
        """Serialize model loading and inference across warm-up and queries."""

        with self._lifecycle_lock:
            return self._encode_text_unlocked(text)

    def _encode_text_unlocked(self, text: str | list[str]) -> np.ndarray:
        """Encode one or more segments into normalized float32 vectors."""

        values = [text] if isinstance(text, str) else list(text)
        if not values:
            return np.empty((0, self.embedding_dim or 0), dtype=np.float32)
        self._load_model()
        rows: list[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            inputs = self._tokenizer(
                values[start : start + self.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self._torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            rows.append(pooled.float().cpu().numpy())
        output = _normalize(np.concatenate(rows, axis=0))
        self._embedding_dim = int(output.shape[1])
        return output

    def encode_text_with_diagnostics(
        self,
        text: str,
    ) -> tuple[np.ndarray, TextEncodingDiagnostics]:
        """Encode tokenizer-aware chunks for semantic text retrieval."""

        with self._lifecycle_lock:
            self._load_model()
            model_limit = int(getattr(self._model.config, "max_position_embeddings", 512))
            tokenizer_limit = int(getattr(self._tokenizer, "model_max_length", model_limit))
            context_limit = min(model_limit, tokenizer_limit)
            special_tokens = int(self._tokenizer.num_special_tokens_to_add(pair=False))
            content_limit = max(1, context_limit - special_tokens)
            chunks = chunk_text(
                self._tokenizer,
                text,
                context_limit=context_limit,
                content_limit=content_limit,
                max_chunks=self.text_max_chunks,
                overlap=min(self.text_chunk_overlap, content_limit - 1),
            )
            return self._encode_text_unlocked(list(chunks.texts)), chunks.diagnostics

    def unload(self) -> None:
        """Release the cached text model while keeping the adapter reusable."""

        with self._lifecycle_lock:
            torch = self._torch
            self._model = None
            self._tokenizer = None
            self._torch = None
            gc.collect()
            if torch is not None and self.device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except (AttributeError, RuntimeError):
                    pass
