"""CPU/offline checks of child failure, timeout and missing-model boundaries."""

import hashlib
import io
import subprocess
import sys

import pytest
from PIL import Image

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.vlm.florence2 import Florence2TaskReasoner
from src.vlm.isolated_florence import IsolatedFlorence2TaskReasoner
from src.vlm.qa_reasoner import QAImage, QAProviderRequestError


def _image() -> QAImage:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(output, format="PNG")
    data = output.getvalue()
    return QAImage(data, "image/png", 1, 10, 400, hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize("script,timeout,category", [
    ("import sys; sys.stdin.buffer.read(); sys.exit(17)", 5, "worker_terminated"),
    ("import sys,time; sys.stdin.buffer.read(); time.sleep(10)", 0.2, "inference_timeout"),
    ("import sys; sys.stdin.buffer.read(); print('invalid')", 5, "response_invalid"),
])
def test_child_failure_does_not_terminate_caller(monkeypatch, tmp_path, script, timeout, category):
    original = subprocess.Popen
    children = []

    def child(_command, **kwargs):
        process = original([sys.executable, "-c", script], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr("src.vlm.isolated_florence.subprocess.Popen", child)
    reasoner = IsolatedFlorence2TaskReasoner(tmp_path, revision="test", timeout_seconds=timeout)
    with pytest.raises(QAProviderRequestError) as raised:
        reasoner.reason_task(QAQueryRequest("q", "event", "Read text"), (_image(),),
                             QACompetitionSpec.conservative_general_text(), task="ocr")
    assert raised.value.category == category
    assert children[0].poll() is not None
    assert reasoner._model is None


def test_real_worker_reports_missing_assets_without_downloading(tmp_path):
    reasoner = IsolatedFlorence2TaskReasoner(tmp_path, revision="test", timeout_seconds=15)
    with pytest.raises(QAProviderRequestError) as raised:
        reasoner.reason_task(QAQueryRequest("q", "event", "Read text"), (_image(),),
                             QACompetitionSpec.conservative_general_text(), task="ocr")
    assert raised.value.category == "model_missing"
    assert not list(tmp_path.iterdir())


def test_invalid_detection_payload_is_not_fabricated_zero():
    from src.vlm.qa_reasoner import QAProviderResponseError

    with pytest.raises(QAProviderResponseError):
        Florence2TaskReasoner._task_answer({}, task="object_count", target_label="cake")
    assert Florence2TaskReasoner._task_answer(
        {"<OD>": {"labels": []}}, task="object_count", target_label="cake"
    ) == "0"
