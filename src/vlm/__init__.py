"""Visual-language reasoning providers and compatibility facade."""

from src.vlm.florence2 import Florence2TaskReasoner
from src.vlm.gemini import GeminiFlashReasoner
from src.vlm.gpu_coordinator import GPUModelCoordinator
from src.vlm.local_qwen import OllamaQwenReasoner
from src.vlm.qa_reasoner import (
    ProviderChainReasoner,
    QAImage,
    QAProviderRequestError,
    QAProviderResponseError,
    QAProviderUnavailableError,
    QAReasoner,
    QAReasonerError,
    QAReasoningResult,
)
from src.vlm.qa_router import QAReasonerRouter
from src.vlm.vlm_engine import VLMEngine

__all__ = [
    "OllamaQwenReasoner",
    "GeminiFlashReasoner",
    "Florence2TaskReasoner",
    "GPUModelCoordinator",
    "ProviderChainReasoner",
    "QAImage",
    "QAProviderRequestError",
    "QAProviderResponseError",
    "QAProviderUnavailableError",
    "QAReasoner",
    "QAReasonerError",
    "QAReasoningResult",
    "QAReasonerRouter",
    "VLMEngine",
]
