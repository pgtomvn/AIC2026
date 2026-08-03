"""Offline tests for strict TRAKE retrieval and temporal alignment."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from src.contracts import TRAKEQueryRequest, TRAKEStage
from src.retrieval.trake import TRAKEPipeline, strict_monotonic_alignment
from src.retrieval.trake_trajectory_cache import TRAKETrajectoryCache


def test_strict_dp_avoids_greedy_trap() -> None:
    similarities = np.array(
        [
            [0.8, 0.2, 1.0],
            [0.1, 0.2, 0.9],
        ],
        dtype=np.float32,
    )

    result = strict_monotonic_alignment(similarities, (10, 20, 30))

    assert result is not None
    positions, score = result
    assert positions == (0, 2)
    assert score == pytest.approx(0.85)


def test_strict_dp_ties_prefer_shorter_span_then_earlier_chain() -> None:
    result = strict_monotonic_alignment(np.ones((2, 4)), (10, 20, 30, 40))

    assert result is not None
    assert result[0] == (0, 1)


def test_strict_dp_rejects_invalid_sequences_and_short_videos() -> None:
    assert strict_monotonic_alignment(np.ones((3, 2))) is None
    with pytest.raises(ValueError, match="strictly increasing"):
        strict_monotonic_alignment(np.ones((2, 3)), (10, 10, 20))
    with pytest.raises(ValueError, match="finite"):
        strict_monotonic_alignment(np.array([[0.1, np.nan]]))


class _Encoder:
    def encode_text(self, text: str | list[str]) -> np.ndarray:
        values = [text] if isinstance(text, str) else text
        assert len(values) == 3
        return np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )


class _Store:
    vector_size = 2

    def __init__(self) -> None:
        self.group_calls: list[dict[str, Any]] = []

    def search_grouped(self, query_vector: np.ndarray, **kwargs: Any) -> list[dict]:
        self.group_calls.append(kwargs)
        return [
            {"group_id": "L21_V002", "score": 0.8, "hits": []},
            {"group_id": "L21_V001", "score": 0.9, "hits": []},
        ]

    def iter_points_with_vectors(self, filters: dict, **_kwargs: Any):
        assert filters["batch_id"] == "aic25-b1-reference"
        assert set(filters["video_id"]) == {"L21_V001", "L21_V002"}
        points = [
            # Duplicate organizer ordinals for one canonical frame are merged.
            _point("L21_V001", 10, 100, [1.0, 0.0]),
            _point("L21_V001", 10, 101, [1.0, 0.0]),
            _point("L21_V001", 20, 200, [0.0, 1.0]),
            _point("L21_V002", 5, 50, [0.8, 0.2]),
            _point("L21_V002", 15, 150, [0.8, 0.2]),
            # Bad vectors and missing canonical mappings are ignored.
            _point("L21_V002", 25, 250, [np.nan, 0.0]),
            {"id": "bad", "payload": {"video_id": "L21_V002"}, "vector": [1.0, 0.0]},
        ]
        yield from points


class _SeedStore:
    vector_size = 2

    def __init__(self) -> None:
        self.group_calls: list[dict[str, Any]] = []
        self.scroll_calls = 0

    def search_grouped(self, query_vector: np.ndarray, **kwargs: Any) -> list[dict]:
        self.group_calls.append(kwargs)
        return [
            {
                "group_id": "L21_V001",
                "score": 0.95,
                "hits": [
                    {**_point("L21_V001", 10, 100, [1.0, 0.0]), "score": 0.95},
                    {**_point("L21_V001", 20, 200, [0.0, 1.0]), "score": 0.90},
                ],
            },
            {
                "group_id": "L21_V002",
                "score": 0.80,
                "hits": [
                    {**_point("L21_V002", 5, 50, [0.8, 0.2]), "score": 0.80},
                    {**_point("L21_V002", 15, 150, [0.2, 0.8]), "score": 0.75},
                ],
            },
        ]

    def iter_points_with_vectors(self, filters: dict, **_kwargs: Any):
        self.scroll_calls += 1
        raise AssertionError(f"seeded candidates must not scroll: {filters}")


class _PointFetchStore:
    vector_size = 2

    def __init__(self) -> None:
        self.group_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.scroll_calls = 0

    def search_grouped(self, query_vector: np.ndarray, **kwargs: Any) -> list[dict]:
        self.group_calls.append(kwargs)
        return [
            {
                "group_id": "L21_V001",
                "score": 0.95,
                "hits": [
                    {
                        "id": "frame-10",
                        "_point_id": "point-10",
                        "score": 0.95,
                        "payload": {"video_id": "L21_V001"},
                    },
                    {
                        "id": "frame-20",
                        "_point_id": "point-20",
                        "score": 0.90,
                        "payload": {"video_id": "L21_V001"},
                    },
                    {
                        "id": "frame-30",
                        "_point_id": "point-missing-vector",
                        "score": 0.85,
                        "payload": {"video_id": "L21_V001"},
                    },
                ],
            },
            {
                "group_id": "L21_V002",
                "score": 0.10,
                "hits": [
                    {
                        "id": "other-frame",
                        "_point_id": "other-point",
                        "score": 0.10,
                        "payload": {"video_id": "L21_V002"},
                    }
                ],
            },
        ]

    def retrieve_points_with_vectors(
        self,
        point_ids: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.retrieve_calls.append((list(point_ids), kwargs))
        # Deliberately reverse chronological order and include one missing
        # vector; the alignment output must remain deterministic and valid.
        return [
            {
                **_point("L21_V001", 20, 200, [0.0, 1.0]),
                "_point_id": "point-20",
            },
            {
                "id": "frame-30",
                "_point_id": "point-missing-vector",
                "payload": {
                    "batch_id": "aic25-b1-reference",
                    "video_id": "L21_V001",
                    "source_frame_idx": 30,
                    "timestamp_ms": 300,
                },
                "vector": None,
            },
            {
                **_point("L21_V001", 10, 100, [1.0, 0.0]),
                "_point_id": "point-10",
            },
        ]

    def iter_points_with_vectors(self, filters: dict, **_kwargs: Any):
        self.scroll_calls += 1
        raise AssertionError(f"point-fetch candidates must not scroll: {filters}")


def _point(
    video_id: str,
    source_frame_idx: int,
    timestamp_ms: int,
    vector: list[float],
) -> dict[str, Any]:
    return {
        "id": f"{video_id}:{source_frame_idx}",
        "payload": {
            "batch_id": "aic25-b1-reference",
            "video_id": video_id,
            "source_frame_idx": source_frame_idx,
            "timestamp_ms": timestamp_ms,
        },
        "vector": vector,
    }


def _request() -> TRAKEQueryRequest:
    return TRAKEQueryRequest(
        query_id="trake-1",
        global_description="A person enters and then sits",
        stages=(
            TRAKEStage(stage_index=1, description="A person enters"),
            TRAKEStage(stage_index=2, description="The person sits"),
        ),
        batch_filters=("aic25-b1-reference",),
        top_k=100,
    )


def test_pipeline_ranks_one_alignment_per_video_with_canonical_frames() -> None:
    store = _Store()
    pipeline = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=store,
        channel_name="organizer_clip",
    )

    response = pipeline.search(_request())

    assert [item.video_id for item in response.results] == ["L21_V001", "L21_V002"]
    assert response.results[0].source_frame_indices == (10, 20)
    assert response.results[0].rank == 1
    assert response.results[0].confidence_score == pytest.approx(1.0)
    assert response.channel_statuses["organizer_clip"]["state"] == "ready"
    assert len(store.group_calls) == 3
    assert all(call["top_k_groups"] == 300 for call in store.group_calls)
    assert all(call["group_size"] == 4 for call in store.group_calls)
    assert all(call["with_vectors"] is True for call in store.group_calls)


def test_fallback_canonicalizer_requires_frame_and_timestamp_monotonicity() -> None:
    pipeline = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=_Store(),
        channel_name="organizer_clip",
    )
    points = (
        _point("L21_V001", 10, 100, [1.0, 0.0]),
        _point("L21_V001", 20, 100, [0.0, 1.0]),
        _point("L21_V001", 30, 300, [0.0, 1.0]),
        _point("L21_V002", 20, 100, [1.0, 0.0]),
        _point("L21_V002", 10, 200, [0.0, 1.0]),
        _point("L21_V002", 30, 300, [0.0, 1.0]),
    )

    frames = pipeline._canonical_frames(
        "aic25-b1-reference",
        ("L21_V001", "L21_V002"),
        2,
        points,
    )

    assert [item.source_frame_idx for item in frames["L21_V001"]] == [10, 30]
    assert [item.timestamp_ms for item in frames["L21_V001"]] == [100, 300]
    assert [item.source_frame_idx for item in frames["L21_V002"]] == [20, 30]
    assert [item.timestamp_ms for item in frames["L21_V002"]] == [100, 300]


def test_fallback_pipeline_never_emits_equal_timestamps() -> None:
    class EqualTimestampStore(_Store):
        def iter_points_with_vectors(self, filters: dict, **_kwargs: Any):
            assert filters["batch_id"] == "aic25-b1-reference"
            yield _point("L21_V001", 10, 100, [1.0, 0.0])
            yield _point("L21_V001", 20, 100, [0.0, 1.0])

    response = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=EqualTimestampStore(),
        channel_name="organizer_clip",
    ).search(_request())

    assert response.results == ()


def test_pipeline_uses_bounded_grouped_vectors_without_full_scroll() -> None:
    store = _SeedStore()
    pipeline = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=store,
        channel_name="organizer_clip",
        seed_frames_per_prompt=2,
    )

    response = pipeline.search(_request())

    assert [item.video_id for item in response.results] == ["L21_V001", "L21_V002"]
    assert response.results[0].source_frame_indices == (10, 20)
    assert store.scroll_calls == 0
    assert all(call["group_size"] == 2 for call in store.group_calls)
    assert all(call["with_vectors"] is True for call in store.group_calls)


def test_pipeline_fetches_only_deduplicated_points_from_selected_videos() -> None:
    store = _PointFetchStore()
    pipeline = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=store,
        channel_name="organizer_clip",
        top_k_videos=1,
    )

    first = pipeline.search(_request())
    second = pipeline.search(_request())

    assert first.model_dump() == second.model_dump()
    assert first.results[0].source_frame_indices == (10, 20)
    assert store.scroll_calls == 0
    assert all(call["with_vectors"] is False for call in store.group_calls)
    assert store.retrieve_calls[0][0] == [
        "point-10",
        "point-20",
        "point-missing-vector",
    ]
    assert "other-point" not in store.retrieve_calls[0][0]
    assert store.retrieve_calls[0][1]["filters"] == {
        "batch_id": "aic25-b1-reference"
    }


def test_exact_cache_aligns_frames_outside_grouped_top_seeds() -> None:
    class ExactTimelineStore:
        vector_size = 2

        def __init__(self) -> None:
            self.group_calls: list[dict[str, Any]] = []
            self.stream_calls = 0
            self.retrieve_calls = 0

        @staticmethod
        def count(_filters: dict[str, Any]) -> int:
            return 4

        def search_grouped(self, _query: np.ndarray, **kwargs: Any) -> list[dict]:
            self.group_calls.append(kwargs)
            return [
                {
                    "group_id": "L21_V001",
                    "score": 0.8,
                    "hits": [
                        {
                            "id": "seed-10",
                            "_point_id": "seed-10",
                            "score": 0.8,
                            "payload": {"video_id": "L21_V001"},
                        },
                        {
                            "id": "seed-20",
                            "_point_id": "seed-20",
                            "score": 0.7,
                            "payload": {"video_id": "L21_V001"},
                        },
                    ],
                }
            ]

        def retrieve_points_with_vectors(self, *_args: Any, **_kwargs: Any):
            self.retrieve_calls += 1
            raise AssertionError("exact cache must not fetch only grouped seed points")

        def iter_points_with_vectors(self, filters: dict, **_kwargs: Any):
            assert filters == {"batch_id": "aic25-b1-reference"}
            self.stream_calls += 1
            yield _point("L21_V001", 10, 100, [0.7, 0.7])
            # Both exact optima are absent from grouped top seeds.
            yield _point("L21_V001", 15, 150, [1.0, 0.0])
            yield _point("L21_V001", 20, 200, [0.7, 0.7])
            yield _point("L21_V001", 30, 300, [0.0, 1.0])

    store = ExactTimelineStore()
    cache = TRAKETrajectoryCache(store)
    pipeline = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=store,
        channel_name="organizer_clip",
        trajectory_cache=cache,
    )

    cache.warm("aic25-b1-reference")
    first = pipeline.search(_request())
    second = pipeline.search(_request())

    assert first.results[0].source_frame_indices == (15, 30)
    assert first.model_dump() == second.model_dump()
    assert store.stream_calls == 1
    assert store.retrieve_calls == 0
    assert all(call["with_vectors"] is False for call in store.group_calls)
    assert all(call["with_payload"] is False for call in store.group_calls)
    assert all(call["group_size"] == 1 for call in store.group_calls)


def test_pipeline_falls_back_for_grouped_backend_without_vectors() -> None:
    class OldGroupedStore(_Store):
        def search_grouped(
            self,
            query_vector: np.ndarray,
            *,
            group_by: str,
            top_k_groups: int,
            group_size: int,
            filters: dict[str, Any],
        ) -> list[dict]:
            self.group_calls.append(
                {
                    "group_by": group_by,
                    "top_k_groups": top_k_groups,
                    "group_size": group_size,
                    "filters": filters,
                }
            )
            return [
                {"group_id": "L21_V002", "score": 0.8, "hits": []},
                {"group_id": "L21_V001", "score": 0.9, "hits": []},
            ]

    response = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=OldGroupedStore(),
    ).search(_request())

    assert [item.video_id for item in response.results] == ["L21_V001", "L21_V002"]


def test_grouped_prompt_search_is_parallel_and_keeps_prompt_order() -> None:
    class ParallelStore:
        vector_size = 3

        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def search_grouped(self, query_vector: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
            prompt_index = int(np.argmax(query_vector))
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep((3 - prompt_index) * 0.01)
            with self.lock:
                self.active -= 1
            scores = (
                {"A": 0.9, "B": 0.0},
                {"A": 0.0, "B": 0.4},
                {"A": 0.0, "B": 0.4},
            )[prompt_index]
            return [
                {"group_id": video_id, "score": score, "hits": []}
                for video_id, score in scores.items()
            ]

        def iter_points_with_vectors(self, filters: dict, **_kwargs: Any):
            return iter(())

    store = ParallelStore()
    pipeline = TRAKEPipeline(
        encoder=_Encoder(),
        vector_store=store,
        grouped_search_workers=3,
    )

    scores, _ = pipeline._candidate_scores(np.eye(3, dtype=np.float32), "batch")

    assert list(scores) == ["B", "A"]
    assert store.max_active >= 2


def test_pipeline_rejects_cross_space_dimension_mismatch() -> None:
    store = _Store()
    store.vector_size = 512
    pipeline = TRAKEPipeline(encoder=_Encoder(), vector_store=store)

    with pytest.raises(ValueError, match="dimension"):
        pipeline.search(_request())


def test_compatibility_constructor_and_alignment_name() -> None:
    pipeline = TRAKEPipeline(top_k_videos=5)

    assert pipeline.top_k_videos == 5
    positions, score = pipeline._dtw_align(
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
    )
    assert positions == [0, 1]
    assert score == pytest.approx(1.0)

    with pytest.raises(ValueError, match="grouped_search_workers"):
        TRAKEPipeline(grouped_search_workers=13)
