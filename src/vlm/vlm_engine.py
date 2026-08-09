"""Backward-compatible single-image facade over the local QA reasoner."""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.vlm.local_qwen import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaQwenReasoner
from src.vlm.qa_reasoner import QAImage, QAReasoner


class VLMEngine:
    """Compatibility facade whose default is always the loopback Ollama provider."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        max_new_tokens: int = 64,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        reasoner: QAReasoner | None = None,
        use_api: bool = False,
        **legacy: Any,
    ) -> None:
        if use_api or legacy:
            raise ValueError(
                "legacy cloud provider selection was removed; inject an explicit QAReasoner"
            )
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.base_url = base_url
        self._reasoner = reasoner

    def _build_reasoner(self) -> QAReasoner:
        if self._reasoner is None:
            self._reasoner = OllamaQwenReasoner(
                base_url=self.base_url,
                model_name=self.model_name,
                max_new_tokens=self.max_new_tokens,
            )
        return self._reasoner

    @property
    def fingerprint(self) -> str:
        return str(self._build_reasoner().fingerprint)

    def reason(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
    ) -> QAReasoningResult:
        return self._build_reasoner().reason(request, images, spec)

    def ask(
        self,
        image_path: str | Path,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
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
        request = QAQueryRequest(
            query_id="legacy-vlm-engine",
            event_description=(
                system_prompt.strip() if system_prompt and system_prompt.strip() else question
            ),
            question=question,
            answer_spec_id=spec.spec_id,
            limit=1,
            reasoning_policy="local_only",
        )
        result = self.reason(request, (image,), spec)
        return {
            "answer": result.answer,
            "confidence": 1.0 if result.answerable else 0.0,
            "answerable": result.answerable,
            "evidence_frame_indices": list(result.evidence_frame_indices),
            "provider": result.provider,
            "model_name": result.model_name,
        }

    _ask_local = ask

    def ask_batch(self, image_paths: list[str | Path], question: str) -> list[dict[str, Any]]:
        return [self.ask(image_path, question) for image_path in image_paths]

    def unload(self) -> None:
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
