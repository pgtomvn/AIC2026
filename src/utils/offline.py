"""Process-local guards for model backends that otherwise probe the network."""

from __future__ import annotations

import os


def enforce_huggingface_offline() -> None:
    """Force Hugging Face-compatible libraries into cache-only operation.

    Assignment is deliberate rather than ``setdefault``: a model configured as
    local-only must override a conflicting ambient ``0`` value.
    """

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
