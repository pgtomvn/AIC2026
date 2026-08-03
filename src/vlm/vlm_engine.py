"""Backward-compatible facade over the Visual Q&A reasoning providers."""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.vlm.qa_reasoner import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_QWEN_MODEL,
    GeminiInteractionsReasoner,
    QAImage,
    QAReasoner,
    Qwen3VLReasoner,
)


class VLMEngine:
    """Compatibility layer for legacy single-image VLM calls.

    Construction remains side-effect free. The selected API client or local
    model is created only on the first ``ask``/``_load_local_model`` call.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        max_new_tokens: int = 128,
        device: str = "cuda",
        use_api: bool = False,
        *,
        api_key: str | None = None,
        client_factory: Callable[..., Any] | None = None,
        reasoner: QAReasoner | None = None,
    ) -> None:
        self.model_name = model_name
        self.gemini_model = gemini_model
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_api = use_api
        self._api_key = api_key
        self._client_factory = client_factory
        self._reasoner = reasoner

    @property
    def fingerprint(self) -> str:
        """Expose the selected provider's secret-free cache fingerprint."""

        value = getattr(self._build_reasoner(), "fingerprint", None)
        if callable(value):
            value = value()
        if not isinstance(value, str) or not value:
            raise RuntimeError("the configured reasoner does not expose a fingerprint")
        return value

    def _build_reasoner(self) -> QAReasoner:
        if self._reasoner is not None:
            return self._reasoner
        if self.use_api:
            self._reasoner = GeminiInteractionsReasoner(
                api_key=self._api_key,
                model_name=self.gemini_model,
                client_factory=self._client_factory,
                max_attempts=5,
                requests_per_minute=13,
                max_concurrency=1,
                retry_base_seconds=5,
                retry_max_seconds=60,
                min_request_interval_seconds=4.5,
            )
        else:
            self._reasoner = Qwen3VLReasoner(
                model_name=self.model_name,
                max_new_tokens=self.max_new_tokens,
                device=self.device,
            )
        return self._reasoner

    def _load_local_model(self) -> None:
        """Load the cache-only local backend through the provider boundary."""

        if self.use_api:
            raise RuntimeError("cannot load a local model while use_api is enabled")
        reasoner = self._build_reasoner()
        load = getattr(reasoner, "load", None)
        if not callable(load):
            raise RuntimeError("the configured reasoner does not expose local model loading")
        load()

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        """Delegate the canonical multi-frame contract to the selected provider."""

        return self._build_reasoner().reason(request, images, spec)

    def ask(
        self,
        image_path: str | Path,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ask one visual question using the canonical provider contract.

        ``system_prompt`` is retained for signature compatibility and is treated
        as event context. Provider safety and output instructions remain fixed.
        """

        path = Path(image_path)
        data = path.read_bytes()
        image = QAImage(
            data=data,
            mime_type=_mime_type_for_path(path),
            index=1,
            source_frame_idx=0,
            timestamp_ms=0,
            sha256=hashlib.sha256(data).hexdigest(),
        )
        spec = QACompetitionSpec.conservative_general_text()
        event_description = (
            system_prompt.strip() if system_prompt and system_prompt.strip() else question
        )
        request = QAQueryRequest(
            query_id="legacy-vlm-engine",
            event_description=event_description,
            question=question,
            answer_spec_id=spec.spec_id,
            limit=1,
        )
        result = self._build_reasoner().reason(request, (image,), spec)
        return {
            "answer": result.answer,
            "confidence": 1.0 if result.answerable else 0.0,
            "answerable": result.answerable,
            "evidence_frame_indices": list(result.evidence_frame_indices),
            "provider": result.provider,
            "model_name": result.model_name,
        }

    def _ask_local(
        self,
        image_path: str | Path,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run one request through a local-configured facade."""

        if self.use_api:
            raise RuntimeError("local inference is unavailable while use_api is enabled")
        return self.ask(image_path, question, system_prompt)

    def _ask_gemini(
        self,
        image_path: str | Path,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run one request through an API-configured facade."""

        if not self.use_api:
            raise RuntimeError("Gemini inference requires use_api=True")
        return self.ask(image_path, question, system_prompt)

    def ask_batch(
        self,
        image_paths: list[str | Path],
        question: str,
    ) -> list[dict[str, Any]]:
        """Ask the same question of each image in deterministic input order."""

        return [self.ask(image_path, question) for image_path in image_paths]

    def unload(self) -> None:
        """Release the selected provider without constructing one first."""

        if self._reasoner is not None:
            self._reasoner.unload()


def _mime_type_for_path(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("VLMEngine supports JPEG, PNG, and WebP images")
    return mime_type


__all__ = ["VLMEngine"]
