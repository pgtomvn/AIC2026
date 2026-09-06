"""Stable CUDA failure classification for retrieval encoder circuit breakers."""

from __future__ import annotations

_FATAL_TOKENS = (
    "device-side assert",
    "illegal memory access",
    "unspecified launch failure",
    "misaligned address",
    "cuda error: an illegal",
)


def is_fatal_cuda_error(error: BaseException) -> bool:
    """Return true only for failures that make same-process CUDA reuse unsafe."""

    message = str(error).casefold()
    return any(token in message for token in _FATAL_TOKENS)


class CUDAEncoderRestartRequired(RuntimeError):
    """Signal that a CUDA encoder circuit remains open until process restart."""

    restart_required = True

    def __init__(self) -> None:
        super().__init__("CUDA encoder restart is required")


__all__ = ["CUDAEncoderRestartRequired", "is_fatal_cuda_error"]
