"""Bounded subprocess transport for the optional Florence task adapter."""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

from src import PROJECT_ROOT
from src.contracts import QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.vlm.florence2 import Florence2TaskReasoner
from src.vlm.qa_reasoner import QAImage, QAProviderRequestError, _validate_images


class IsolatedFlorence2TaskReasoner(Florence2TaskReasoner):
    """Load Florence only in a one-shot child; reclaim its memory after each call."""

    def __init__(self, *args: Any, timeout_seconds: float = 90.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Florence worker timeout must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)

    def reason_task(
        self,
        request: QAQueryRequest,
        images: Sequence[QAImage],
        spec: QACompetitionSpec,
        *,
        task: str,
        target_label: str | None = None,
    ) -> QAReasoningResult:
        if task not in {"ocr", "object_count"} or (
            task == "object_count" and not str(target_label or "").strip()
        ):
            raise ValueError("Florence requires an explicit OCR or labelled object-count task")
        values = _validate_images(images, max_images=1, max_total_image_bytes=8 * 1024 * 1024)
        if len(values) != 1:
            raise ValueError("Florence requires exactly one image")
        payload = {
            "config": {
                "model_path": str(self.model_path.resolve()),
                "model_name": self.model_name,
                "revision": self.revision,
                "expected_weights_sha256": self.expected_weights_sha256,
                "max_new_tokens": self.max_new_tokens,
            },
            "request": request.to_dict(), "spec": spec.to_dict(),
            "task": task, "target_label": target_label,
            "image": {"data": base64.b64encode(values[0].data).decode("ascii"),
                      "mime_type": values[0].mime_type, "index": values[0].index,
                      "source_frame_idx": values[0].source_frame_idx,
                      "timestamp_ms": values[0].timestamp_ms, "sha256": values[0].sha256},
        }
        with self._lock:
            started = time.monotonic()
            self._last_metrics = {}
            environment = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")
            process = subprocess.Popen(
                [sys.executable, "-m", "src.vlm.florence_worker"],
                cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                try:
                    output, _ = process.communicate(
                        json.dumps(payload).encode("utf-8"), timeout=self.timeout_seconds
                    )
                except subprocess.TimeoutExpired:
                    raise self._failure("inference_timeout") from None
                if process.returncode != 0:
                    self._last_metrics["worker_exit_code"] = process.returncode
                    raise self._failure("worker_terminated")
                try:
                    response = json.loads(output)
                    if "error" in response:
                        raise self._failure(response["error"].get("category") or "provider_request_failed")
                    result = QAReasoningResult.from_dict(response["result"])
                except (ValueError, TypeError, KeyError):
                    raise self._failure("response_invalid") from None
                self._last_metrics["wall_latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                return result
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def _failure(self, category: str) -> QAProviderRequestError:
        return QAProviderRequestError(
            "the isolated Florence worker could not complete inference",
            provider=self.provider_name, model=self.model_name, attempts=1, category=category,
        )
