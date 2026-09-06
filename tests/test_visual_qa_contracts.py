"""Offline tests for Visual Q&A contracts and answer normalization."""

from __future__ import annotations

import pytest

from src.contracts import (
    QA_GENERAL_TEXT_SPEC_ID,
    QA_HISTORICAL_NUMERIC_SPEC_ID,
    FrameRecord,
    QACandidateWindow,
    QACompetitionSpec,
    QAPrediction,
    QAQueryRequest,
    QAReasoningResult,
    QASingleFrameAnswerRequest,
    resolve_qa_competition_spec,
)
from src.vlm.answer_normalizer import (
    AnswerNormalizationError,
    AnswerNormalizer,
    normalize_answer,
    normalize_general_text,
    normalize_numeric,
)


def _prediction(**overrides) -> QAPrediction:
    values = {
        "rank": 1,
        "batch_id": "aic25-b1-reference",
        "video_id": "L21_V001",
        "source_frame_idx": 181,
        "timestamp_ms": 3017,
        "answer": "5",
        "confidence_score": 0.8,
        "answer_source": "ollama-qwen",
        "window_id": "L21_V001:181:r4000:n5",
        "evidence": {"frame_indices": [170, 181, 192]},
        "raw_answer": "five",
        "model_name": "fixture-model",
    }
    values.update(overrides)
    return QAPrediction(**values)


def _frame(source_frame_idx: int) -> FrameRecord:
    return FrameRecord(
        batch_id="b1",
        video_id="L21_V001",
        keyframe_ordinal=source_frame_idx + 1,
        source_frame_idx=source_frame_idx,
        pts=source_frame_idx,
        time_base="1/10",
        timestamp_ms=source_frame_idx * 100,
        fps_num=10,
        fps_den=1,
        shot_id="shot-1",
        sampling_reasons=("fixture",),
        image_uri=f"frame-{source_frame_idx}.webp",
        video_uri="video.mp4",
        source_sha256="a" * 64,
        extractor_version="fixture-v1",
    )


def test_qa_query_request_round_trip_and_defaults() -> None:
    request = QAQueryRequest(
        query_id="qa-1",
        event_description="Một người bước vào cửa hàng",
        question="Có bao nhiêu người?",
        mode="interactive",
        batch_filters=["aic25-b1-reference"],
        limit=25,
        allow_external_frame_transfer=True,
    )

    assert request.answer_spec_id == QA_HISTORICAL_NUMERIC_SPEC_ID
    assert request.batch_filters == ("aic25-b1-reference",)
    assert QAQueryRequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError, match="mode"):
        QAQueryRequest(
            "qa-1", "event", "question", mode="unsupported", reasoning_policy="local_only"
        )


def test_single_frame_manual_answer_contract_is_local_only() -> None:
    request = QASingleFrameAnswerRequest(
        candidate_id="b1:L21_V001:181",
        question="Địa điểm này ở đâu?",
        reasoning_policy="local_only",
        answer_engine="manual",
        manual_answer="xã Cam Hiệp Bắc",
    )

    assert QASingleFrameAnswerRequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError, match="local-only"):
        QASingleFrameAnswerRequest(
            candidate_id=request.candidate_id,
            question=request.question,
            reasoning_policy="gemini_then_local",
            allow_external_frame_transfer=True,
            answer_engine="manual",
            manual_answer="answer",
        )
    with pytest.raises(ValueError, match="requires answer_engine=manual"):
        QASingleFrameAnswerRequest(
            candidate_id=request.candidate_id,
            question=request.question,
            reasoning_policy="local_only",
            answer_engine="vlm",
            manual_answer="answer",
        )
    with pytest.raises(ValueError, match="duplicates"):
        QAQueryRequest(
            "qa-1", "event", "question", batch_filters=("b1", "b1"),
            reasoning_policy="local_only",
        )
    with pytest.raises(ValueError, match="unknown Visual Q&A answer_spec_id"):
        QAQueryRequest(
            "qa-1", "event", "question", answer_spec_id="unknown-spec",
            reasoning_policy="local_only",
        )


def test_qa_candidate_window_is_canonical_ordered_and_bounded() -> None:
    frames = tuple(_frame(value) for value in (9, 10, 11))
    window = QACandidateWindow(
        cluster_id="b1:L21_V001:10",
        retrieval_rank=1,
        retrieval_score=0.75,
        batch_id="b1",
        video_id="L21_V001",
        source_frame_idx=10,
        timestamp_ms=1_000,
        shot_id="shot-1",
        frames=frames,
        frame_selection_scores={9: 0.9, 10: 0.8},
    )

    assert QACandidateWindow.from_dict(window.to_dict()) == window
    serialized = window.to_dict()
    serialized["frame_selection_scores"] = {"9": 0.9, "10": 0.8}
    assert QACandidateWindow.from_dict(serialized) == window
    with pytest.raises(ValueError, match="between 3 and 5"):
        QACandidateWindow(
            "short", 1, 0.5, "b1", "L21_V001", 10, 1_000, "shot-1", frames[:2]
        )
    with pytest.raises(ValueError, match="ordered chronologically"):
        QACandidateWindow(
            "unordered",
            1,
            0.5,
            "b1",
            "L21_V001",
            10,
            1_000,
            "shot-1",
            (frames[1], frames[0], frames[2]),
        )
    with pytest.raises(ValueError, match="anchor must be present"):
        QACandidateWindow(
            "missing-anchor",
            1,
            0.5,
            "b1",
            "L21_V001",
            12,
            1_200,
            "shot-1",
            frames,
        )
    with pytest.raises(ValueError, match="selected frames"):
        QACandidateWindow(
            "bad-score",
            1,
            0.5,
            "b1",
            "L21_V001",
            10,
            1_000,
            "shot-1",
            frames,
            frame_selection_scores={999: 0.9},
        )


def test_qa_prediction_and_reasoning_result_round_trip() -> None:
    prediction = _prediction()
    reasoning = QAReasoningResult(
        raw_answer="Five",
        normalized_answer="5",
        answerable=True,
        evidence_frame_indices=[170, 181, 192],
        provider="fixture",
        model_name="fixture-model",
        prompt_version="qa-v1",
        latency_ms=12.5,
    )

    assert QAPrediction.from_dict(prediction.to_dict()) == prediction
    assert QAReasoningResult.from_dict(reasoning.to_dict()) == reasoning
    assert reasoning.answer == "5"
    with pytest.raises(ValueError, match="ambiguous frame_id"):
        QAPrediction.from_dict({**prediction.to_dict(), "frame_id": 181})
    with pytest.raises(ValueError, match="between 0 and 1"):
        _prediction(confidence_score=1.1)
    with pytest.raises(ValueError, match="evidence frame"):
        QAReasoningResult(
            raw_answer="5",
            normalized_answer="5",
            answerable=True,
            evidence_frame_indices=(),
            provider="fixture",
            model_name="fixture-model",
            prompt_version="qa-v1",
            latency_ms=1,
        )


def test_answer_specs_are_explicit_and_unknown_ids_fail() -> None:
    numeric = resolve_qa_competition_spec(QA_HISTORICAL_NUMERIC_SPEC_ID)
    general = resolve_qa_competition_spec(QA_GENERAL_TEXT_SPEC_ID)

    assert numeric == QACompetitionSpec.historical_numeric()
    assert general == QACompetitionSpec.conservative_general_text()
    assert QACompetitionSpec.from_dict(numeric.to_dict()) == numeric
    with pytest.raises(ValueError, match="unknown Visual Q&A answer_spec_id"):
        resolve_qa_competition_spec("unknown-spec")


def test_numeric_and_general_text_normalization_are_conservative() -> None:
    question = "Có bao nhiêu người?"

    assert normalize_numeric("  ‘NĂM!’  ", question=question) == "5"
    assert normalize_numeric("05") == "5"
    assert normalize_numeric("three") == "3"
    assert normalize_answer("MƯỜI LĂM", mode="numeric") == "15"
    assert normalize_general_text("  “Màu đỏ!”  ") == "màu đỏ"
    assert normalize_answer("NĂM", mode="general_text") == "năm"
    assert AnswerNormalizer("general_text")("Màu đỏ.") == "màu đỏ"
    with pytest.raises(AnswerNormalizationError, match="numeric answer"):
        normalize_numeric("there are five")
    with pytest.raises(AnswerNormalizationError, match="quantity or number question"):
        normalize_numeric("tư", question="Đây là khu vực công hay tư?")
    assert normalize_numeric("tư", question="Có mấy người?") == "4"
