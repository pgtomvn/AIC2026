"""Explicit cache-only five-frame Qwen3-VL GPU readiness check."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.utils.io_helpers import load_config
from src.vlm.qa_reasoner import QAImage, Qwen3VLReasoner


def _image(index: int, color: str) -> QAImage:
    output = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(output, format="PNG")
    data = output.getvalue()
    return QAImage(
        data=data,
        mime_type="image/png",
        index=index,
        source_frame_idx=index - 1,
        timestamp_ms=(index - 1) * 100,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def main() -> int:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("run this check with `uv run --group competition`") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen readiness requires a CUDA GPU")
    detected_total = int(torch.cuda.get_device_properties(0).total_memory)
    if detected_total < int(11.5 * 1024**3):
        raise RuntimeError("Qwen accelerated profile requires a nominal 12 GiB or larger GPU")
    config = load_config()
    qwen = config.get("models", {}).get("qwen_vl", {})
    if not qwen.get("enabled", False):
        raise RuntimeError("models.qwen_vl.enabled must be true for this explicit check")
    reasoner = Qwen3VLReasoner(
        model_name=str(qwen["model_name"]),
        revision=qwen.get("revision"),
        cache_dir=str((PROJECT_ROOT / str(qwen["cache_dir"])).resolve()),
        device=str(qwen.get("device", "cuda")),
        dtype=str(qwen.get("dtype", "float16")),
        max_new_tokens=int(qwen.get("max_new_tokens", 32)),
        max_pixels=int(qwen.get("max_pixels", 589_824)),
        local_files_only=True,
    )
    request = QAQueryRequest(
        query_id="local-qwen-synthetic-check",
        event_description="Five simple frames contain red and blue squares.",
        question="How many provided frames are red?",
        answer_spec_id=QACompetitionSpec.historical_numeric().spec_id,
        limit=5,
    )
    images = tuple(
        _image(index, "red" if index <= 3 else "blue") for index in range(1, 6)
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        result = reasoner.reason(
            request,
            images,
            QACompetitionSpec.historical_numeric(),
        )
        peak = int(torch.cuda.max_memory_reserved())
        total = detected_total
    finally:
        reasoner.unload()
    if not result.answerable or result.normalized_answer != "3":
        raise RuntimeError("Qwen readiness check returned an unexpected structured answer")
    if peak <= 0 or peak > total:
        raise RuntimeError("Qwen readiness check reported invalid GPU memory use")
    print(
        "Qwen five-frame readiness passed: "
        f"model={result.model_name} peak_vram_gib={peak / 1024**3:.2f} "
        f"total_vram_gib={total / 1024**3:.2f} latency_ms={result.latency_ms:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
