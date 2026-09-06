"""CPU-only contract tests for production ASR extraction and gated retrieval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.contracts import QAQueryRequest, QueryRequest
from src.extraction.asr_engine import ASREngine
from src.ingest.cloud_artifacts import cloud_model_contracts
from src.retrieval.query_plan import build_query_plan
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline
from src.retrieval.visual_qa import VisualQAPipeline
from src.utils.io_helpers import load_config
from src.vlm.qa_reasoner import _question_text


class _Encoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _Store:
    def __init__(self, *, kind: str) -> None:
        self.kind = kind
        self.calls: list[object] = []

    def search(self, query: object, *, top_k: int, filters: object = None):
        self.calls.append(query)
        payload = {
            "batch_id": "aic25-b1-reference",
            "video_id": "L21_V001",
            "source_frame_idx": 10,
            "timestamp_ms": 1_500,
            "shot_id": "shot-1",
        }
        if self.kind == "asr":
            payload.update(
                {
                    "kind": "asr",
                    "segment_id": "asr-1",
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "confidence": 0.9,
                    "text_nfc": "xin chào",
                }
            )
        return [{"score": 1.0, "payload": payload}]


def _pipeline() -> tuple[TextualKISPipeline, _Encoder, _Store]:
    encoder = _Encoder()
    dense = _Store(kind="visual")
    asr = _Store(kind="asr")
    return (
        TextualKISPipeline(
            dense_channels=(DenseChannel("visual", encoder, dense),),
            lexical_channels=(LexicalChannel("asr", asr),),
            temporal_windows_ms=(),
            asr_max_weight=0.7,
            asr_temporal_overlap_bonus=0.3,
        ),
        encoder,
        asr,
    )


def test_query_plan_detects_speech_names_and_numbers_but_not_plain_visuals() -> None:
    assert build_query_plan("a red car crossing a bridge").asr_intent is False
    assert build_query_plan('người đàn ông nói "xin chào"').asr_intent is True
    assert build_query_plan("phỏng vấn Nguyễn Văn An về số 2026").asr_intent is True


def test_query_modes_gate_asr_and_expose_isolated_baselines() -> None:
    pipeline, encoder, asr = _pipeline()

    pipeline.search(
        QueryRequest("visual", "a red car crossing a bridge", mode="automatic"),
        refine_raw=False,
    )
    assert encoder.calls
    assert asr.calls == []
    assert pipeline.last_query_processing["asr_gate"]["enabled"] is False

    encoder.calls.clear()
    pipeline.search(
        QueryRequest("speech", "người đàn ông nói xin chào", mode="asr_only"),
        refine_raw=False,
    )
    assert encoder.calls == []
    assert asr.calls
    assert pipeline.last_query_processing["asr_gate"]["max_weight"] == 0.7
    assert pipeline.last_query_processing["asr_gate"]["query_text_source"] == "original_vi"
    assert pipeline.last_channel_statuses["asr"]["query_text_source"] == "original_vi"
    assert pipeline.last_channel_statuses["asr"]["translation_applied"] is False

    asr.calls.clear()
    pipeline.search(
        QueryRequest("baseline", "người đàn ông nói xin chào", mode="visual_only"),
        refine_raw=False,
    )
    assert asr.calls == []
    assert pipeline.last_channel_statuses["asr"]["reason_code"] == "visual_only_mode"


def test_temporal_asr_bonus_is_bounded_and_never_penalizes_silence() -> None:
    pipeline, _, _ = _pipeline()
    candidates = [
        {
            "video_id": "L21_V001",
            "source_frame_idx": 10,
            "timestamp_ms": 1_500,
            "score": 1.0,
            "evidence": [
                {
                    "channel": "asr",
                    "kind": "asr",
                    "segment_id": "asr-1",
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "confidence": 0.5,
                }
            ],
        },
        {
            "video_id": "L21_V002",
            "source_frame_idx": 20,
            "timestamp_ms": 5_000,
            "score": 1.0,
            "evidence": [],
        },
    ]

    result = pipeline._apply_asr_temporal_bonus(candidates, enabled=True)

    assert result[0]["score"] == 1.15
    assert result[0]["asr_temporal_bonus"] == 0.15
    assert result[1]["score"] == 1.0
    assert pipeline.last_query_processing["asr_temporal_fusion"]["absence_penalized"] is False


def test_visual_qa_injects_only_bounded_nearby_transcript_context() -> None:
    calls: list[dict[str, object]] = []

    class _Repository:
        def text_segments_in_range(self, batch_id, video_id, **kwargs):
            calls.append({"batch_id": batch_id, "video_id": video_id, **kwargs})
            return (
                SimpleNamespace(start_ms=1_000, end_ms=2_000, text_nfc="xin chào"),
                SimpleNamespace(start_ms=2_000, end_ms=3_000, text_nfc="tôi là An"),
            )

    retriever = SimpleNamespace(frame_repository=_Repository())
    pipeline = VisualQAPipeline(
        retriever=retriever,
        asr_context_radius_ms=30_000,
        asr_context_max_characters=100,
    )
    request = QAQueryRequest("qa", "a man speaking", "What is his name?")

    enriched = pipeline._with_asr_context(
        request, "aic25-b1-reference", "L21_V001", 20_000
    )
    prompt = _question_text(enriched)

    assert calls[0]["start_ms"] == 0
    assert calls[0]["end_ms"] == 50_000
    assert calls[0]["kind"] == "asr"
    assert "Nearby ASR transcript" in prompt
    assert "[00:01.000-00:02.000] xin chào" in prompt
    assert request.transcript_context == ()


def test_production_asr_contract_and_engine_kwargs_are_fingerprinted() -> None:
    contract = cloud_model_contracts(load_config(), asr_profile="production")["asr"]
    assert contract["profile"] == "production"
    assert contract["name"] == "dropbox-dash/faster-whisper-large-v3-turbo"
    assert contract["compute_type"] == "int8_float16"
    assert contract["vad_parameters"] == {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    }

    calls: list[dict[str, object]] = []

    class _Model:
        def transcribe(self, _path: str, **kwargs):
            calls.append(kwargs)
            segment = SimpleNamespace(
                start=1.0,
                end=2.0,
                text="xin chào",
                avg_logprob=0.0,
                no_speech_prob=0.0,
                words=(),
            )
            info = SimpleNamespace(language="vi", language_probability=0.99, duration=2.0)
            return iter((segment,)), info

    engine = ASREngine(
        model_size=contract["name"],
        model_revision=contract["revision"],
        language=contract["language"],
        device="cuda",
        compute_type=contract["compute_type"],
        beam_size=contract["beam_size"],
        retry_beam_size=contract["retry_beam_size"],
        profile_name="production",
        vad_parameters=contract["vad_parameters"],
        condition_on_previous_text=contract["condition_on_previous_text"],
    )
    engine._model = _Model()
    with patch.object(Path, "is_file", return_value=True):
        result = engine.transcribe("synthetic.wav")

    assert calls[0]["vad_parameters"] == contract["vad_parameters"]
    assert calls[0]["condition_on_previous_text"] is True
    assert result["profile"] == "production"
    assert result["model_fingerprint"] == engine.fingerprint
