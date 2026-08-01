"""
VLM Engine
==========
Visual Language Model inference for answering visual questions.
Supports local models (Qwen2-VL via vLLM) and API fallback (Gemini).

Usage:
    engine = VLMEngine(model_name="Qwen/Qwen2-VL-7B-Instruct")
    answer = engine.ask(image_path="frame.jpg", question="What color is the car?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class VLMEngine:
    """Visual Language Model inference engine.

    Supports two backends:
    - **Local**: Qwen2-VL or LLaVA-NeXT via vLLM or HuggingFace
    - **API**: Google Gemini 1.5 Flash for high-throughput fallback

    Args:
        model_name: HuggingFace model ID for local inference.
        gemini_model: Gemini model name for API fallback.
        max_new_tokens: Maximum tokens to generate.
        device: "cuda" or "cpu".
        use_api: If True, use Gemini API instead of local model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        gemini_model: str = "gemini-1.5-flash",
        max_new_tokens: int = 256,
        device: str = "cuda",
        use_api: bool = False,
    ) -> None:
        self.model_name = model_name
        self.gemini_model = gemini_model
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_api = use_api
        self._model = None
        self._processor = None

    def _load_local_model(self) -> None:
        """Load the local VLM (Qwen2-VL) using vLLM or transformers."""
        # TODO: Load model and processor
        raise NotImplementedError("VLMEngine._load_local_model")

    def ask(
        self,
        image_path: str | Path,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Ask a visual question about an image.

        Args:
            image_path: Path to the image.
            question: Question to ask about the image.
            system_prompt: Optional system instruction.

        Returns:
            Dict with keys: answer (str), confidence (float).
        """
        if self.use_api:
            return self._ask_gemini(image_path, question, system_prompt)
        return self._ask_local(image_path, question, system_prompt)

    def _ask_local(
        self,
        image_path: str | Path,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Run inference with the local VLM."""
        # TODO: Preprocess image, construct prompt, generate
        raise NotImplementedError("VLMEngine._ask_local")

    def _ask_gemini(
        self,
        image_path: str | Path,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Run inference via Google Gemini API."""
        # TODO: Call Gemini API with image and question
        raise NotImplementedError("VLMEngine._ask_gemini")

    def ask_batch(
        self,
        image_paths: list[str | Path],
        question: str,
    ) -> list[dict]:
        """Batch VLM inference for multiple images with the same question.

        Args:
            image_paths: List of image file paths.
            question: Question to ask about each image.

        Returns:
            List of answer dicts.
        """
        # TODO: Batch inference for throughput
        raise NotImplementedError("VLMEngine.ask_batch")
