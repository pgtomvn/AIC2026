"""Explicit Ollama/Qwen2.5-VL availability and latency integration check."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.utils.io_helpers import load_config
from src.vlm.local_qwen import OllamaQwenReasoner
from src.vlm.qa_reasoner import QAImage, QAReasonerError


def _synthetic_image() -> QAImage:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    for x in (24, 88, 152):
        draw.ellipse((x, 72, x + 48, 120), fill="red")
    output = io.BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    return QAImage(
        data=data,
        mime_type="image/png",
        index=1,
        source_frame_idx=0,
        timestamp_ms=0,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-latency", action="store_true")
    parser.add_argument("--warm-target-seconds", type=float, default=2.0)
    return parser


def _memory_usage() -> dict[str, int | None]:
    """Report memory when psutil is present without making it a core dependency."""

    try:
        import psutil
    except ImportError:
        return {"checker_rss_bytes": None, "system_ram_used_bytes": None}
    return {
        "checker_rss_bytes": psutil.Process().memory_info().rss,
        "system_ram_used_bytes": psutil.virtual_memory().used,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config()
    service = config.get("services", {}).get("ollama_vlm", {})
    reasoner = OllamaQwenReasoner(
        base_url=str(service.get("url", "http://127.0.0.1:11434")),
        model_name=str(service.get("model", "qwen2.5vl:3b-q4_K_M")),
        expected_digest_prefix=service.get("expected_digest_prefix"),
        connect_timeout_seconds=float(service.get("connect_timeout_seconds", 2)),
        timeout_seconds=float(service.get("timeout_seconds", 120)),
        keep_alive=service.get("keep_alive", "15m"),
        num_ctx=int(service.get("num_ctx", 4096)),
        max_new_tokens=int(service.get("max_new_tokens", 64)),
    )
    request = QAQueryRequest(
        query_id="offline-vlm-check",
        event_description="ba hình tròn màu đỏ trên nền trắng",
        question="Có bao nhiêu hình tròn màu đỏ?",
        answer_spec_id=QACompetitionSpec.conservative_general_text().spec_id,
        limit=1,
        reasoning_policy="local_only",
    )
    try:
        probe = reasoner.probe()
        timings: list[dict[str, object]] = []
        for phase in ("cold", "warm"):
            started = perf_counter()
            result = reasoner.reason(
                request,
                (_synthetic_image(),),
                QACompetitionSpec.conservative_general_text(),
            )
            timings.append(
                {
                    "phase": phase,
                    "wall_seconds": round(perf_counter() - started, 6),
                    "provider_latency_ms": round(result.latency_ms, 3),
                    "answer": result.answer,
                    "answerable": result.answerable,
                    "ollama": reasoner.last_metrics,
                }
            )
        warm_seconds = float(timings[-1]["wall_seconds"])
        payload = {
            "status": "passed" if warm_seconds <= args.warm_target_seconds else "warning",
            "provider": probe["provider"],
            "model": probe["model"],
            "digest": probe["digest"],
            "vision": probe["vision"],
            "warm_target_seconds": args.warm_target_seconds,
            "timings": timings,
            "resources": {**_memory_usage(), "ollama": reasoner.runtime_status()},
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 1 if args.strict_latency and warm_seconds > args.warm_target_seconds else 0
    except QAReasonerError as error:
        print(
            json.dumps(
                {"status": "failed", **error.safe_metadata()},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    finally:
        reasoner.unload()


if __name__ == "__main__":
    raise SystemExit(main())
