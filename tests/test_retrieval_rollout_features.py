from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from src.evaluation.interaction_log import InteractionLogger
from src.evaluation.rollout import (
    rollout_config_fingerprint,
    validate_rollout_activation,
)
from src.retrieval.candidate_rescue import (
    CandidateRescueConfig,
    candidate_concentration,
    run_candidate_rescue,
    should_rescue,
)
from src.retrieval.ocr_quality import (
    OCRQualityConfig,
    OCRQualityStore,
    build_ocr_quality_features,
    calibrate_ocr_results,
    write_ocr_quality_sidecar,
    write_ocr_quality_sidecar_streaming,
)
from src.retrieval.textual_kis import TextualKISPipeline
from src.retrieval.video_local import (
    VideoLocalConfig,
    score_video_local,
    shot_bounded_gaussian_support,
)


def _result(video_id: str, frame_idx: int, score: float = 1.0) -> dict:
    return {
        "id": f"batch:{video_id}:{frame_idx}",
        "score": score,
        "payload": {"batch_id": "batch", "video_id": video_id, "source_frame_idx": frame_idx},
    }


class _RescueStore:
    def __init__(self) -> None:
        self.search_calls = 0
        self.grouped_calls = 0

    def search(self, _vector, *, top_k, filters):
        self.search_calls += 1
        assert top_k == 600
        assert filters == {"batch_id": "batch"}
        return [_result("expanded", 1)]

    def search_grouped(self, _vector, **kwargs):
        self.grouped_calls += 1
        assert kwargs["top_k_groups"] == 100
        assert kwargs["group_size"] == 2
        return [_result("grouped", 2)]


def test_candidate_concentration_and_shadow_rescue_do_not_return_branches() -> None:
    initial = [_result("dominant", index) for index in range(40)] + [
        _result(f"other-{index}", 40 + index) for index in range(10)
    ]
    config = CandidateRescueConfig(mode="shadow")
    concentration = candidate_concentration(initial)
    triggered, reasons = should_rescue(concentration, config)
    assert triggered
    assert "top_video_concentration" in reasons
    store = _RescueStore()
    outcome = run_candidate_rescue(
        config=config,
        initial_results=initial,
        store=store,
        query_vector=np.array([1.0, 0.0], dtype=np.float32),
        filters={"batch_id": "batch"},
        channel_name="visual",
    )
    assert outcome.rankings == ()
    assert outcome.diagnostics["applied"] is False
    assert store.search_calls == store.grouped_calls == 1


def test_active_rescue_returns_independent_rrf_branches() -> None:
    store = _RescueStore()
    outcome = run_candidate_rescue(
        config=CandidateRescueConfig(mode="active"),
        initial_results=[_result("one", index) for index in range(50)],
        store=store,
        query_vector=np.array([1.0, 0.0], dtype=np.float32),
        filters={"batch_id": "batch"},
        channel_name="visual",
    )
    assert [branch.name for branch in outcome.rankings] == [
        "visual:adaptive_overfetch",
        "visual:video_grouped",
    ]


def test_gaussian_support_never_crosses_shots() -> None:
    support = shot_bounded_gaussian_support(
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0, 1_000, 2_000], dtype=np.int64),
        np.array([7, 8, 7], dtype=np.int64),
        window_ms=8_000,
        sigma_ms=4_000,
    )
    assert support[1] == 0.0
    assert support[0] > 0.0


@dataclass
class _Trajectory:
    source_frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    vectors: np.ndarray
    shot_codes: np.ndarray

    def __len__(self) -> int:
        return len(self.source_frame_indices)


class _Batch:
    def video(self, video_id: str):
        if video_id != "video":
            return None
        return _Trajectory(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0, 1_000, 2_000], dtype=np.int64),
            np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
            np.array([1, 1, 2], dtype=np.int64),
        )


class _Trajectories:
    def batch(self, batch_id: str):
        return _Batch() if batch_id == "batch" else None


def test_video_local_scores_cached_frames_without_warming() -> None:
    results, diagnostics = score_video_local(
        config=VideoLocalConfig(mode="shadow", frames_per_video=2),
        candidates=[_result("video", 99)],
        query_vectors=np.array([[1.0, 0.0], [0.7, 0.3]], dtype=np.float32),
        trajectories=_Trajectories(),
        batch_id="batch",
    )
    assert len(results) == 2
    assert diagnostics["scored_video_count"] == 1
    assert diagnostics["applied"] is False


def test_video_local_accepts_the_pipeline_single_batch_filter_contract() -> None:
    pipeline = object.__new__(TextualKISPipeline)
    pipeline.dense_channels = ()
    pipeline.video_local = VideoLocalConfig(mode="shadow", frames_per_video=2)
    pipeline.video_local_trajectories = _Trajectories()
    pipeline._execution_state = type("State", (), {})()
    pipeline._execution_state.tier1_query_views = {
        "organizer_clip": (np.array([1.0, 0.0], dtype=np.float32),)
    }
    pipeline.last_query_processing = {}
    pipeline.last_channel_statuses = {}
    pipeline.last_channel_errors = {}

    output = pipeline._run_video_local(
        [_result("video", 99)],
        [],
        {"batch_filters": ["batch"]},
        effective_k=60,
    )

    assert output == [_result("video", 99)]
    assert pipeline.last_query_processing["video_local"]["scored_video_count"] == 1
    assert pipeline.last_query_processing["video_local"]["applied"] is False
    assert pipeline.last_query_processing["video_local"]["source_channel"] == "organizer_clip"


def test_video_local_rejects_cross_vector_space_scoring() -> None:
    results, diagnostics = score_video_local(
        config=VideoLocalConfig(mode="shadow"),
        candidates=[_result("video", 99)],
        query_vectors=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        trajectories=_Trajectories(),
        batch_id="batch",
    )

    assert results == []
    assert diagnostics["scored_video_count"] == 0
    assert diagnostics["reason_code"] == "vector_space_mismatch"


def _ocr(segment_id: str, frame: int, text: str, confidence: float) -> dict:
    return {
        "segment_id": segment_id,
        "video_id": "video",
        "source_frame_idx": frame,
        "text_raw": text,
        "confidence": confidence,
        "polygons": [[[0, 0], [100, 0], [100, 20], [0, 20]]],
    }


def test_ocr_sidecar_is_idempotent_and_roi_fallback_is_explicit() -> None:
    rows = [_ocr(f"s{index}", index, "Breaking News", 0.9) for index in range(3)]
    without_roi = build_ocr_quality_features(rows)
    assert all(not item.roi_available for item in without_roi)
    with_roi = build_ocr_quality_features(rows, image_sizes={"video": (200, 100)})
    assert all(item.roi_available for item in with_roi)
    path = Path("outputs") / f".test-ocr-quality-{uuid4().hex}.sqlite3"
    try:
        first = write_ocr_quality_sidecar(path, with_roi, source_fingerprint="artifact")
        second = write_ocr_quality_sidecar(path, with_roi, source_fingerprint="artifact")
        assert first == second
        assert OCRQualityStore(path).load(["s0"])["s0"].persistence > 0.0
    finally:
        path.unlink(missing_ok=True)


def test_ocr_sidecar_streaming_is_repeatable() -> None:
    first = build_ocr_quality_features([_ocr("a", 1, "Alpha", 0.8)])
    second_row = _ocr("b", 2, "Beta", 0.9)
    second_row["video_id"] = "video-z"
    second = build_ocr_quality_features([second_row])
    streaming_path = Path("outputs") / f".test-ocr-stream-{uuid4().hex}.sqlite3"
    repeated_path = Path("outputs") / f".test-ocr-stream-repeat-{uuid4().hex}.sqlite3"
    try:
        actual, count = write_ocr_quality_sidecar_streaming(
            streaming_path, (first, second), source_fingerprint="artifact"
        )
        repeated, repeated_count = write_ocr_quality_sidecar_streaming(
            repeated_path, (first, second), source_fingerprint="artifact"
        )
        assert actual == repeated
        assert count == 2
        assert repeated_count == count
        assert set(OCRQualityStore(streaming_path).load(["a", "b"])) == {"a", "b"}
    finally:
        streaming_path.unlink(missing_ok=True)
        repeated_path.unlink(missing_ok=True)


def test_ocr_calibration_never_deletes_and_shadow_preserves_order() -> None:
    rows = (
        _ocr("weak", 1, "$%^", 0.1),
        _ocr("strong", 2, "London Zoo", 0.99),
    )
    features = build_ocr_quality_features(rows)
    path = Path("outputs") / f".test-ocr-quality-{uuid4().hex}.sqlite3"
    try:
        write_ocr_quality_sidecar(path, features, source_fingerprint="artifact")
        results = [
            {"score": 1.0, "payload": rows[0]},
            {"score": 0.95, "payload": rows[1]},
        ]
        shadow, diagnostics = calibrate_ocr_results(
            results,
            config=OCRQualityConfig(mode="shadow"),
            store=OCRQualityStore(path),
            exact_terms=("London Zoo",),
        )
        assert [item["payload"]["segment_id"] for item in shadow] == ["weak", "strong"]
        assert diagnostics["evaluated_candidate_count"] == 2
        assert diagnostics["quoted_text_exemptions"] == 1
        active, diagnostics = calibrate_ocr_results(
            results,
            config=OCRQualityConfig(mode="active"),
            store=OCRQualityStore(path),
            exact_terms=("London Zoo",),
        )
        assert len(active) == len(results)
        assert {item["payload"]["segment_id"] for item in active} == {"weak", "strong"}
        assert diagnostics["applied"] is True
    finally:
        path.unlink(missing_ok=True)


def test_active_rollout_requires_matching_locked_tuning_record() -> None:
    record = Path("outputs") / f".test-rollout-{uuid4().hex}.json"
    config = {
        "mode": "active",
        "depth": 600,
        "tuning_record": str(record),
    }
    try:
        with pytest.raises(FileNotFoundError):
            validate_rollout_activation("candidate_rescue", config, project_root=Path.cwd())
        record.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "feature": "candidate_rescue",
                    "config_fingerprint": rollout_config_fingerprint(config),
                    "qrels_locked_test_sha256": "a" * 64,
                    "activation_gate": {"passed": True},
                }
            ),
            encoding="utf-8",
        )
        validate_rollout_activation("candidate_rescue", config, project_root=Path.cwd())
    finally:
        record.unlink(missing_ok=True)


def test_interaction_log_is_append_only_and_bounded() -> None:
    path = Path("outputs") / f".test-interactions-{uuid4().hex}.jsonl"
    try:
        logger = InteractionLogger(path, mode="shadow", max_results=1)
        logger.record_search(
            query_id="q1",
            query_text="person reading a sign",
            retrieval_fingerprint="f" * 64,
            batch_ids=("batch",),
            results=(_result("v1", 1), _result("v2", 2)),
            latency_ms=12.5,
        )
        logger.record_feedback(
            query_id="q1",
            positive_ids=("batch:v1:1",),
            negative_ids=(),
            submitted_frame_id="batch:v1:1",
        )
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [event["event"] for event in events] == [
            "search_results_returned",
            "relevance_feedback",
        ]
        assert len(events[0]["results"]) == 1
    finally:
        path.unlink(missing_ok=True)
