"""CPU-only tests for historical Textual KIS interval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_evaluation import main
from src.evaluation.textual_kis import (
    CrossValidationFold,
    GroundTruthInterval,
    GroundTruthQuery,
    HoldoutSplit,
    Prediction,
    build_grouped_splits,
    evaluate_queries,
    evaluate_query,
    grouped_k_fold,
    grouped_train_validation_test_split,
    load_ground_truth,
    load_predictions,
)


def _ground_truth(query_id: str = "q1", video_id: str = "L21_V001") -> GroundTruthQuery:
    return GroundTruthQuery(
        query_id=query_id,
        intervals=(GroundTruthInterval(video_id, 100, 120),),
        text="target moment",
        intent_class="action",
    )


@pytest.mark.parametrize(
    ("correct_rank", "expected_score"),
    [(1, 1.0), (2, 0.8), (5, 0.8), (6, 0.6), (20, 0.6), (21, 0.4), (51, 0.2)],
)
def test_historical_rank_tiers_emerge_from_cutoff_recall(
    correct_rank: int, expected_score: float
) -> None:
    predictions = [Prediction("L22_V001", index) for index in range(correct_rank - 1)]
    predictions.append(Prediction("L21_V001", 110))

    metrics = evaluate_query(predictions, _ground_truth())

    assert metrics.first_relevant_rank == correct_rank
    assert metrics.mean_top_k_score == pytest.approx(expected_score)
    assert metrics.reciprocal_rank == pytest.approx(1 / correct_rank)


def test_interval_rule_accepts_any_interval_and_rejects_wrong_video_or_boundary() -> None:
    ground_truth = GroundTruthQuery(
        "q",
        (
            GroundTruthInterval("L21_V001", 100, 120),
            GroundTruthInterval("L22_V002", 200, 210),
        ),
    )

    result = evaluate_query(
        [
            Prediction("L21_V001", 99),
            Prediction("L23_V002", 205),
            Prediction("L22_V002", 210),
        ],
        ground_truth,
    )

    assert result.first_relevant_rank == 3
    assert result.recall_at[1] == 0.0
    assert result.recall_at[5] == 1.0


def test_multi_query_report_includes_misses_intents_and_latency() -> None:
    queries = (_ground_truth("q1", "L21_V001"), _ground_truth("q2", "L22_V002"))
    predictions = {"q1": [Prediction("L21_V001", 100)]}

    report = evaluate_queries(predictions, queries, latencies_ms={"q1": 10.0, "q2": 30.0})

    assert report.query_count == 2
    assert report.recall_at[1] == 0.5
    assert report.mean_top_k_score == 0.5
    assert report.mrr == 0.5
    assert report.median_rank == 51.0  # median of rank 1 and miss sentinel 101
    assert report.median_rank_miss_value == 101
    assert report.latency_ms == {"mean": 20.0, "median": 20.0, "p95": 29.0}
    assert report.by_intent_class["action"]["query_count"] == 2
    json.dumps(report.to_dict())


def test_multi_query_report_rejects_unknown_prediction_query() -> None:
    with pytest.raises(ValueError, match="unknown query IDs"):
        evaluate_queries({"typo": []}, (_ground_truth(),))


def _many_queries(count: int) -> tuple[GroundTruthQuery, ...]:
    return tuple(
        GroundTruthQuery(
            f"q{index}",
            (GroundTruthInterval(f"L{21 + index // 100:02d}_V{index % 1000:03d}", 0, 10),),
        )
        for index in range(count)
    )


def _videos(queries) -> set[str]:
    return {interval.video_id for query in queries for interval in query.intervals}


def test_grouped_holdout_split_is_deterministic_and_has_no_video_leakage() -> None:
    queries = _many_queries(50)

    first = grouped_train_validation_test_split(queries, seed=7)
    second = grouped_train_validation_test_split(queries, seed=7)

    assert isinstance(first, HoldoutSplit)
    assert first == second
    assert len(first.train) == 30
    assert len(first.validation) == 10
    assert len(first.test) == 10
    assert not (_videos(first.train) & _videos(first.validation))
    assert not (_videos(first.train) & _videos(first.test))
    assert not (_videos(first.validation) & _videos(first.test))
    assert isinstance(build_grouped_splits(queries), HoldoutSplit)


def test_grouped_five_fold_uses_each_query_once_for_validation() -> None:
    queries = _many_queries(10)

    folds = grouped_k_fold(queries, seed=9)

    assert len(folds) == 5
    assert all(isinstance(fold, CrossValidationFold) for fold in folds)
    validation_ids = [query.query_id for fold in folds for query in fold.validation]
    assert sorted(validation_ids) == sorted(query.query_id for query in queries)
    assert all(not (_videos(fold.train) & _videos(fold.validation)) for fold in folds)
    assert isinstance(build_grouped_splits(queries), tuple)


def test_grouping_connects_queries_that_share_any_target_video() -> None:
    queries = (
        GroundTruthQuery("q1", (GroundTruthInterval("L21_V001", 0, 1),)),
        GroundTruthQuery(
            "q2",
            (
                GroundTruthInterval("L21_V001", 0, 1),
                GroundTruthInterval("L21_V002", 0, 1),
            ),
        ),
        GroundTruthQuery("q3", (GroundTruthInterval("L21_V003", 0, 1),)),
        GroundTruthQuery("q4", (GroundTruthInterval("L21_V004", 0, 1),)),
        GroundTruthQuery("q5", (GroundTruthInterval("L21_V005", 0, 1),)),
        GroundTruthQuery("q6", (GroundTruthInterval("L21_V006", 0, 1),)),
    )

    folds = grouped_k_fold(queries, folds=5)
    assignment = {query.query_id: fold.fold_index for fold in folds for query in fold.validation}

    assert assignment["q1"] == assignment["q2"]


def test_loaders_support_nested_json_and_ranked_csv(tmp_path: Path) -> None:
    ground_truth_path = tmp_path / "gt.json"
    ground_truth_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "kis-1",
                        "text": "Một người đang đọc báo",
                        "intent_class": "action",
                        "intervals": [
                            {
                                "video_name": "L21_V001",
                                "start_frame_idx": 10,
                                "end_frame_idx": 20,
                            },
                            {
                                "video_name": "L21_V002",
                                "source_frame_idx": 30,
                            },
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    prediction_path = tmp_path / "predictions.csv"
    prediction_path.write_text(
        "query_id,video_name,frame_idx,rank\nkis-1,L21_V002,30,2\nkis-1,L21_V001,15,1\n",
        encoding="utf-8",
    )

    ground_truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(prediction_path)

    assert len(ground_truth[0].intervals) == 2
    assert predictions["kis-1"] == [
        Prediction("L21_V001", 15),
        Prediction("L21_V002", 30),
    ]


def test_prediction_loader_rejects_keyframe_ordinal(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps([{"video_id": "L21_V001", "keyframe_ordinal": 7}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keyframe_ordinal"):
        load_predictions(path)


def test_prediction_loader_rejects_rank_gaps(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text(
        "query_id,video_name,frame_idx,rank\nq1,L21_V001,10,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        load_predictions(path)


def test_evaluation_cli_emits_compact_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ground_truth = tmp_path / "ground-truth.csv"
    predictions = tmp_path / "predictions.csv"
    ground_truth.write_text(
        "query_id,video_name,start_frame_idx,end_frame_idx\nq1,L21_V001,10,20\n",
        encoding="utf-8",
    )
    predictions.write_text("L21_V001,15\n", encoding="utf-8")

    result = main(
        [
            "--ground-truth",
            str(ground_truth),
            "--predictions",
            str(predictions),
            "--compact",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["mean_top_k_score"] == 1.0
    assert "per_query" not in payload
