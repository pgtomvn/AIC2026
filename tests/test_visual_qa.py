from __future__ import annotations

import hashlib
import io
import json

from PIL import Image

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.vlm.local_qwen import OllamaQwenReasoner
from src.vlm.qa_reasoner import QAImage


class _Response:
    status_code = 200

    def json(self):
        return {
            "message": {
                "content": json.dumps(
                    {"answer": "3", "answerable": True, "evidence_frame_indices": [1]}
                )
            }
        }


class _OllamaHTTP:
    def request(self, *_args, **_kwargs):
        return _Response()


def _image() -> QAImage:
    output = io.BytesIO()
    Image.new("RGB", (24, 16), "red").save(output, format="PNG")
    data = output.getvalue()
    return QAImage(
        data=data,
        mime_type="image/png",
        index=1,
        source_frame_idx=10,
        timestamp_ms=1_000,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _request() -> QAQueryRequest:
    return QAQueryRequest(
        query_id="qa-test",
        event_description="ba hình tròn màu đỏ",
        question="Có bao nhiêu hình tròn?",
        answer_spec_id=QACompetitionSpec.conservative_general_text().spec_id,
        batch_filters=("aic25-b1-reference",),
        limit=5,
        reasoning_policy="local_only",
    )


def test_local_multi_window_reasoning_has_no_rate_limit_state() -> None:
    http = _OllamaHTTP()
    reasoner = OllamaQwenReasoner(http=http)
    spec = QACompetitionSpec.conservative_general_text()
    results = [reasoner.reason(_request(), (_image(),), spec) for _ in range(20)]
    assert all(result.provider == "ollama-qwen" for result in results)
    assert all(result.attempts == 1 for result in results)
