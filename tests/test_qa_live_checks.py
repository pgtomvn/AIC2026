"""Offline checks for explicit, bounded Visual Q&A diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import check_qa_live, check_vlm_offline
from src.contracts import QACompetitionSpec, QAReasoningResult


class _Reasoner:
    provider_name = "ollama-qwen"
    model_name = "qwen2.5vl:test"

    def __init__(self, **_settings) -> None:
        self.calls: list[int] = []
        self.last_metrics: dict[str, float] = {}

    def probe(self) -> dict:
        return {
            "state": "ready",
            "provider": self.provider_name,
            "model": self.model_name,
            "digest": "test-digest",
            "vision": True,
        }

    def reason(self, _request, images, _spec) -> QAReasoningResult:
        self.calls.append(len(images))
        return QAReasoningResult(
            raw_answer="3",
            normalized_answer="3",
            answerable=True,
            evidence_frame_indices=(1,),
            provider=self.provider_name,
            model_name=self.model_name,
            prompt_version="test-v1",
            latency_ms=1.0,
            attempts=1,
        )

    def unload(self) -> None:
        return None

    def runtime_status(self) -> dict:
        return {"state": "resident", "size_vram_bytes": 1}


def test_offline_checker_runs_exactly_cold_and_warm_inferences(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reasoner = _Reasoner()
    monkeypatch.setattr(check_vlm_offline, "OllamaQwenReasoner", lambda **_kw: reasoner)

    assert check_vlm_offline.main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["vision"] is True
    assert [item["phase"] for item in payload["timings"]] == ["cold", "warm"]
    assert reasoner.calls == [1, 1]


def test_live_checker_is_local_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    initialized: list[str] = []
    live_calls: list[object] = []

    class Initializable:
        def __init__(self, name: str) -> None:
            self.name = name

        def initialize(self) -> None:
            initialized.append(self.name)

    class Pipeline:
        reasoner = _Reasoner()

        @staticmethod
        def retrieve_vqa_candidates(_event, top_k, *, batch_filters):
            assert top_k == 1 and batch_filters == ("b1",)
            return (
                SimpleNamespace(
                    candidate_id="b1:video-1:10",
                    batch_id="b1",
                    video_id="video-1",
                    source_frame_idx=10,
                ),
            )

        def answer_vqa_single_frame(self, frame, question, **kwargs):
            live_calls.append((frame, question, kwargs))
            return (
                {
                    "answerable": True,
                    "answer": "3",
                    "provider": "ollama-qwen",
                },
                {"vlm_inference_ms": 1.0},
            )

    class Context:
        qa_pipeline = Pipeline()
        qa_reasoner = qa_pipeline.reasoner
        class Repository(Initializable):
            def get_frame(self, batch_id, video_id, source_frame_idx):
                return SimpleNamespace(
                    batch_id=batch_id,
                    video_id=video_id,
                    source_frame_idx=source_frame_idx,
                    timestamp_ms=1_000,
                )

        artifact_repository = Repository("artifacts")
        ledger = Initializable("ledger")

        @staticmethod
        def validate_batch_filters(values):
            return tuple(values)

    monkeypatch.setattr(check_qa_live, "load_config", lambda: {})
    monkeypatch.setattr(check_qa_live, "build_runtime", lambda _config: Context())

    result = check_qa_live.main(
        [
            "--batch-id",
            "b1",
            "--event-description",
            "event",
            "--question",
            "question",
        ]
    )

    assert result == 0
    assert initialized == ["ledger", "artifacts"]
    assert len(live_calls) == 1
    assert live_calls[0][2]["reasoning_policy"] == "local_only"


def test_live_checker_cloud_requires_consent() -> None:
    result = check_qa_live.main(
        [
            "--batch-id",
            "b1",
            "--event-description",
            "event",
            "--question",
            "question",
            "--reasoning-policy",
            "gemini_then_local",
        ]
    )
    assert result == 2


def test_general_text_spec_used_by_live_checker() -> None:
    assert QACompetitionSpec.conservative_general_text().answer_normalization == "general_text"
