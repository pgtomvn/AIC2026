"""Optional offline Florence-2 adapter for explicit OCR and object-count tasks."""

from __future__ import annotations

import gc
import hashlib
import io
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.vlm.qa_reasoner import (
    PROMPT_VERSION,
    QAImage,
    QAProviderRequestError,
    QAProviderResponseError,
    QAProviderUnavailableError,
    _fingerprint,
    _validate_images,
)


class Florence2TaskReasoner:
    """CPU-only Florence-2 tasks selected explicitly by an operator."""

    provider_name = "florence2-experimental"

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_name: str = "microsoft/Florence-2-base",
        revision: str,
        expected_weights_sha256: str | None = None,
        max_new_tokens: int = 256,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.revision = revision
        self.expected_weights_sha256 = (
            expected_weights_sha256.casefold() if expected_weights_sha256 else None
        )
        self.max_new_tokens = int(max_new_tokens)
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._processor: Any = None
        self._model: Any = None
        self._last_metrics: dict[str, int | float] = {}
        self._max_attempts = 1

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.provider_name,
            {
                "model": self.model_name,
                "revision": self.revision,
                "weights": self.expected_weights_sha256 or "unverified",
                "device": "cpu",
                "tasks": ["ocr", "object_count"],
            },
        )

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def last_metrics(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._last_metrics)

    def _weights_path(self) -> Path:
        return self.model_path / "model.safetensors"

    def probe(self) -> dict[str, Any]:
        """Verify pinned local assets without importing or loading Transformers."""

        required = (
            self.model_path / "config.json",
            self.model_path / "preprocessor_config.json",
            self._weights_path(),
        )
        if not all(path.is_file() for path in required):
            raise QAProviderUnavailableError(
                "the optional Florence-2 model is not provisioned",
                provider=self.provider_name,
                model=self.model_name,
                category="model_missing",
            )
        if self.expected_weights_sha256:
            digest = hashlib.sha256()
            with self._weights_path().open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest().casefold() != self.expected_weights_sha256:
                raise QAProviderUnavailableError(
                    "the Florence-2 model fingerprint is incompatible",
                    provider=self.provider_name,
                    model=self.model_name,
                    category="model_unavailable",
                )
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "revision": self.revision,
            "device": "cpu",
        }

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._processor, self._model
        self.probe()
        try:
            # Transformers copies trusted local custom modules to its dynamic
            # module cache. Keep that generated state beside the pinned model
            # instead of touching the operator's home directory.
            modules_cache = self.model_path / ".cache" / "modules"
            modules_cache.mkdir(parents=True, exist_ok=True)
            import transformers.dynamic_module_utils as dynamic_modules

            dynamic_modules.HF_MODULES_CACHE = str(modules_cache)
            from transformers import AutoModelForCausalLM, AutoProcessor

            processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
                # The pinned Florence custom class predates Transformers'
                # automatic SDPA capability flag. Eager attention is the
                # stable CPU path and avoids claiming unsupported SDPA.
                attn_implementation="eager",
            ).to("cpu")
            model.eval()
        except Exception:
            raise QAProviderUnavailableError(
                "the optional Florence-2 model could not be loaded",
                provider=self.provider_name,
                model=self.model_name,
                category="model_unavailable",
            ) from None
        self._processor, self._model = processor, model
        return processor, model

    @staticmethod
    def _task_answer(parsed: Any, *, task: str, target_label: str | None) -> str:
        key = "<OCR>" if task == "ocr" else "<OD>"
        value = parsed.get(key) if isinstance(parsed, Mapping) else parsed
        if task == "ocr":
            if isinstance(value, str):
                return " ".join(value.split())
            if isinstance(value, (list, tuple)):
                return " ".join(str(item).strip() for item in value if str(item).strip())
            return ""
        if not isinstance(value, Mapping) or not isinstance(value.get("labels"), list):
            raise QAProviderResponseError(
                "Florence-2 returned an invalid object detection payload",
                provider="florence2-experimental", category="response_invalid",
            )
        normalized_target = " ".join(str(target_label or "").casefold().split())
        return str(
            sum(
                1
                for label in value["labels"]
                if " ".join(str(label).casefold().split()) == normalized_target
            )
        )

    def reason_task(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
        *,
        task: str,
        target_label: str | None = None,
    ) -> QAReasoningResult:
        if task not in {"ocr", "object_count"}:
            raise QAProviderRequestError(
                "Florence-2 does not support this question type",
                provider=self.provider_name,
                model=self.model_name,
                category="invalid_request",
            )
        if task == "object_count" and not str(target_label or "").strip():
            raise ValueError("object_count requires target_label")
        values = _validate_images(images, max_images=1, max_total_image_bytes=8 * 1024 * 1024)
        if len(values) != 1:
            raise ValueError("Florence-2 requires exactly one image")
        with self._lock:
            processor, model = self._load()
            try:
                with Image.open(io.BytesIO(values[0].data)) as source:
                    image = source.convert("RGB")
                task_prompt = "<OCR>" if task == "ocr" else "<OD>"
                started = self._monotonic()
                inputs = processor(text=task_prompt, images=image, return_tensors="pt")
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
                generated_text = processor.batch_decode(
                    generated, skip_special_tokens=False
                )[0]
                parsed = processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(image.width, image.height),
                )
                latency_ms = (self._monotonic() - started) * 1_000
            except Exception:
                raise QAProviderRequestError(
                    "Florence-2 inference failed",
                    provider=self.provider_name,
                    model=self.model_name,
                    category="provider_request_failed",
                ) from None
            raw_answer = self._task_answer(parsed, task=task, target_label=target_label)
            answerable = bool(raw_answer.strip())
            normalized = spec.normalize_answer(raw_answer, request.question) if answerable else ""
            self._last_metrics = {"wall_latency_ms": round(max(0.0, latency_ms), 3)}
            return QAReasoningResult(
                raw_answer=raw_answer,
                normalized_answer=normalized,
                answerable=answerable,
                evidence_frame_indices=(1,) if answerable else (),
                provider=self.provider_name,
                model_name=self.model_name,
                prompt_version=f"{PROMPT_VERSION}-florence-task-v1",
                latency_ms=latency_ms,
            )

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        del request, images, spec
        raise QAProviderRequestError(
            "Florence-2 requires an explicit OCR or object-count task",
            provider=self.provider_name,
            model=self.model_name,
            category="invalid_request",
        )

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            gc.collect()


__all__ = ["Florence2TaskReasoner"]
