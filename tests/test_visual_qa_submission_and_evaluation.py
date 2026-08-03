"""Visual Q&A submission and exact end-to-end metric tests."""

from __future__ import annotations

import csv

import pytest

from src.contracts import QACompetitionSpec, QAPrediction
from src.evaluation.textual_kis import GroundTruthInterval
from src.evaluation.visual_qa import (
    QAGroundTruthQuery,
    evaluate_qa_queries,
    evaluate_qa_query,
    first_localized_qa_rank,
    first_relevant_qa_rank,
)
from src.qa_submission import (
    QASubmissionValidationError,
    export_qa_submission,
    validate_qa_submission,
)
from src.submission import FrameIndexEvidence


def _prediction(**overrides) -> QAPrediction:
    values = {
        "rank": 1,
        "batch_id": "aic25-b1-reference",
        "video_id": "L21_V001",
        "source_frame_idx": 110,
        "timestamp_ms": 3667,
        "answer": "five",
        "confidence_score": 0.9,
        "answer_source": "fixture",
        "window_id": "window-1",
        "evidence": {"frames": [100, 110, 120]},
        "raw_answer": "Five!",
        "model_name": "fixture-model",
    }
    values.update(overrides)
    return QAPrediction(**values)


def _ground_truth(*, answers=("5",)) -> QAGroundTruthQuery:
    return QAGroundTruthQuery(
        query_id="qa-1",
        intervals=(GroundTruthInterval("L21_V001", 100, 120),),
        answers=answers,
        event_description="A group enters a store",
        question="How many people are there?",
        intent_class="count",
    )


def _active(batch_id: str) -> bool:
    return batch_id == "aic25-b1-reference"


def _persisted(_batch_id: str, _video_id: str, _source_frame_idx: int) -> bool:
    return True


def test_validate_and_export_qa_submission_normalizes_answer_atomically(tmp_path) -> None:
    destination = tmp_path / "nested" / "qa.csv"

    written = export_qa_submission(
        [_prediction(video_id="l21_v001.mp4")],
        destination,
        batch_id="aic25-b1-reference",
        is_batch_active=_active,
        is_canonical_source_frame=_persisted,
    )

    assert written == destination
    with destination.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [["L21_V001", "110", "5"]]


def test_qa_submission_rejects_duplicate_frames_even_with_alternate_answers() -> None:
    with pytest.raises(QASubmissionValidationError, match="duplicate canonical frame"):
        validate_qa_submission(
            [
                _prediction(rank=1, answer="five"),
                _prediction(rank=2, answer="six"),
            ],
            batch_id="aic25-b1-reference",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
        )


def test_qa_submission_requires_batch_and_rejects_ordinal_provenance() -> None:
    with pytest.raises(QASubmissionValidationError, match="batch_id is required"):
        validate_qa_submission([("L21_V001", 110, "5")])

    evidence = FrameIndexEvidence({"L21_V001": {7: 110}})
    with pytest.raises(QASubmissionValidationError, match="keyframe_ordinal"):
        validate_qa_submission(
            [
                {
                    "video_id": "L21_V001",
                    "source_frame_idx": 7,
                    "keyframe_ordinal": 7,
                    "answer": "5",
                }
            ],
            batch_id="aic25-b1-reference",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
            frame_index_evidence=evidence,
        )


def test_qa_submission_rejects_missing_and_mixed_persisted_source_frames() -> None:
    with pytest.raises(QASubmissionValidationError, match="not a canonical source frame"):
        validate_qa_submission(
            [_prediction(source_frame_idx=999999)],
            batch_id="aic25-b1-reference",
            is_batch_active=_active,
            is_canonical_source_frame=lambda _batch, _video, _frame: False,
        )
    with pytest.raises(QASubmissionValidationError, match="does not match export batch"):
        validate_qa_submission(
            [_prediction(batch_id="different-batch")],
            batch_id="aic25-b1-reference",
            is_batch_active=_active,
            is_canonical_source_frame=_persisted,
        )


def test_exact_visual_qa_metric_requires_video_interval_and_answer() -> None:
    predictions = [
        ("L21_V001", 110, "4"),
        ("L21_V001", 110, "NĂM!"),
        ("L21_V002", 110, "5"),
    ]

    metrics = evaluate_qa_query(predictions, _ground_truth(), latency_ms=20)

    assert first_relevant_qa_rank(predictions, _ground_truth()) == 2
    assert first_localized_qa_rank(predictions, _ground_truth()) == 1
    assert metrics.first_localized_rank == 1
    assert metrics.first_relevant_rank == 2
    assert metrics.retrieval_recall_at == {
        1: 1.0,
        5: 1.0,
        20: 1.0,
        50: 1.0,
        100: 1.0,
    }
    assert metrics.recall_at == {1: 0.0, 5: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}
    assert metrics.answer_accuracy_given_retrieval_at == {
        1: 0.0,
        5: 1.0,
        20: 1.0,
        50: 1.0,
        100: 1.0,
    }
    assert metrics.mean_top_k_score == pytest.approx(0.8)
    assert metrics.reciprocal_rank == pytest.approx(0.5)


def test_general_text_spec_preserves_accents_and_aggregate_metrics() -> None:
    spec = QACompetitionSpec.conservative_general_text()
    first = QAGroundTruthQuery(
        query_id="qa-1",
        intervals=(GroundTruthInterval("L21_V001", 10, 20),),
        answers=("màu đỏ",),
        question="Chiếc áo màu gì?",
        intent_class="color",
    )
    second = QAGroundTruthQuery(
        query_id="qa-2",
        intervals=(GroundTruthInterval("L21_V002", 30, 40),),
        answers=("xanh",),
        question="Màu gì?",
        intent_class="color",
    )

    report = evaluate_qa_queries(
        {
            "qa-1": [("L21_V001", 15, "MÀU ĐỎ!")],
            "qa-2": [("L21_V002", 35, "đỏ")],
        },
        (first, second),
        spec=spec,
        latencies_ms={"qa-1": 10, "qa-2": 30},
        provenance={
            "qa-1": [{"window_id": "w1", "provider": "fixture", "cache_status": "hit"}],
            "qa-2": [{"window_id": "w2", "provider": "fixture", "cache_status": "miss"}],
        },
    )

    assert report.retrieval_recall_at == {
        1: 1.0,
        5: 1.0,
        20: 1.0,
        50: 1.0,
        100: 1.0,
    }
    assert report.recall_at == {1: 0.5, 5: 0.5, 20: 0.5, 50: 0.5, 100: 0.5}
    assert report.answer_accuracy_given_retrieval_at == {
        1: 0.5,
        5: 0.5,
        20: 0.5,
        50: 0.5,
        100: 0.5,
    }
    assert report.answer_accuracy_given_retrieval == pytest.approx(0.5)
    assert report.mean_top_k_score == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)
    assert report.latency_ms["median"] == pytest.approx(20)
    assert report.cache_hit_rate == pytest.approx(0.5)
    assert report.cache_observation_count == 2
    assert report.provider_usage == {"fixture": 2}
