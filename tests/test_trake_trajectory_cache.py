"""Offline tests for the exact, compact TRAKE trajectory cache."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from src.indexing.vector_store import VectorPointBatch
from src.retrieval.trake_snapshot import TRAKETrajectorySnapshotStore
from src.retrieval.trake_trajectory_cache import (
    TRAKE_TRAJECTORY_CACHE_VERSION,
    TRAKETrajectoryCache,
    TRAKETrajectoryCacheError,
)


def _point(
    frame_idx: int,
    vector: list[float],
    *,
    video_id: str = "video-1",
    batch_id: str = "batch-1",
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{video_id}:{frame_idx}",
        "payload": {
            "batch_id": batch_id,
            "video_id": video_id,
            "source_frame_idx": frame_idx,
            "timestamp_ms": frame_idx if timestamp_ms is None else timestamp_ms,
        },
        "vector": vector,
    }


class _MutableStore:
    vector_size = 2

    def __init__(self) -> None:
        self.revision = 1
        self.points = [_point(10, [1.0, 0.0]), _point(20, [0.0, 1.0])]
        self.stream_calls = 0
        self.fail = False
        self.filters: list[dict[str, Any]] = []

    def trajectory_revision(self, _batch_id: str) -> int:
        return self.revision

    def count(self, _filters: dict[str, Any]) -> int:
        return len(self.points)

    def iter_points_with_vectors(self, filters: dict[str, Any], **_kwargs: Any):
        self.stream_calls += 1
        self.filters.append(filters)
        if self.fail:
            raise RuntimeError("secret Qdrant endpoint")
        yield from self.points


def test_cache_is_lazy_compact_and_rebuilds_when_fingerprint_changes() -> None:
    store = _MutableStore()
    cache = TRAKETrajectoryCache(store)

    assert cache.snapshot()["state"] == "idle"
    assert store.stream_calls == 0

    first = cache.warm("batch-1")
    cache.warm("batch-1")

    trajectory = cache.batch("batch-1")
    assert trajectory is not None
    video = trajectory.video("video-1")
    assert video is not None
    assert video.source_frame_indices.tolist() == [10, 20]
    assert video.vectors.flags.c_contiguous
    assert video.vectors.flags.writeable is False
    assert first["nbytes"] == 2 * (2 * 4 + 8 + 8 + 8)
    assert store.stream_calls == 1
    assert store.filters == [{"batch_id": "batch-1"}]

    store.revision = 2
    store.points = [_point(30, [1.0, 1.0])]
    cache.warm("batch-1")

    rebuilt = cache.batch("batch-1")
    assert rebuilt is not None
    assert rebuilt.video("video-1").source_frame_indices.tolist() == [30]  # type: ignore[union-attr]
    assert store.stream_calls == 2


def test_cache_canonicalizes_duplicates_and_skips_invalid_points() -> None:
    store = _MutableStore()
    store.points = [
        _point(10, [1.0, 0.0], timestamp_ms=101),
        _point(10, [1.0, 0.0], timestamp_ms=100),
        _point(20, [np.nan, 0.0]),
        _point(30, [0.0, 0.0]),
        _point(40, [0.0, 1.0], batch_id="other-batch"),
    ]
    cache = TRAKETrajectoryCache(store)

    cache.warm("batch-1")

    batch = cache.batch("batch-1")
    assert batch is not None
    video = batch.video("video-1")
    assert video is not None
    assert video.source_frame_indices.tolist() == [10]
    assert video.timestamps_ms.tolist() == [100]
    assert video.vectors.tolist() == [[1.0, 0.0]]


def test_cache_warm_is_single_flight_across_threads() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStore(_MutableStore):
        def iter_points_with_vectors(self, filters: dict[str, Any], **kwargs: Any):
            self.stream_calls += 1
            self.filters.append(filters)
            started.set()
            assert release.wait(timeout=5)
            yield from self.points

    store = BlockingStore()
    cache = TRAKETrajectoryCache(store, fingerprint_provider=lambda _batch: "stable")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cache.warm, "batch-1") for _ in range(4)]
        assert started.wait(timeout=5)
        release.set()
        snapshots = [future.result(timeout=5) for future in futures]

    assert store.stream_calls == 1
    assert all(snapshot["state"] == "ready" for snapshot in snapshots)


def test_cache_failure_is_sanitized_and_retryable() -> None:
    store = _MutableStore()
    store.fail = True
    cache = TRAKETrajectoryCache(store)

    with pytest.raises(TRAKETrajectoryCacheError, match="warm-up failed"):
        cache.warm("batch-1")

    failed = cache.snapshot()
    assert failed["state"] == "failed"
    assert failed["reason_code"] == "trajectory_cache_build_failed"
    assert "secret" not in str(failed)
    assert cache.batch("batch-1") is None

    store.fail = False
    ready = cache.warm("batch-1")
    assert ready["state"] == "ready"
    assert store.stream_calls == 2


def test_cache_prefers_contiguous_batch_iterator_when_available() -> None:
    class BatchStore(_MutableStore):
        def iter_point_batches_with_vectors(self, filters, **_kwargs):
            self.stream_calls += 1
            self.filters.append(filters)
            yield VectorPointBatch(
                point_ids=("point-10", "point-20"),
                payloads=tuple(point["payload"] for point in self.points),
                vectors=np.asarray([point["vector"] for point in self.points], dtype=np.float32),
                source_point_count=2,
            )

        def iter_points_with_vectors(self, *_args, **_kwargs):
            raise AssertionError("legacy point iterator must not be used")

    store = BatchStore()
    cache = TRAKETrajectoryCache(store)

    cache.warm("batch-1")

    batch = cache.batch("batch-1")
    assert batch is not None
    video = batch.video("video-1")
    assert video is not None
    assert video.source_frame_indices.tolist() == [10, 20]
    assert store.stream_calls == 1


def test_persistent_snapshot_is_checksummed_and_memory_mapped_across_instances(
    tmp_path,
) -> None:
    store = _MutableStore()
    snapshot_store = TRAKETrajectorySnapshotStore(
        tmp_path,
        channel="organizer_clip",
        vector_size=2,
        cache_version=TRAKE_TRAJECTORY_CACHE_VERSION,
    )
    first = TRAKETrajectoryCache(store, snapshot_store=snapshot_store)

    first_snapshot = first.warm("batch-1")

    assert first_snapshot["source"] == "rebuilt"
    assert store.stream_calls == 1
    first_batch = first.batch("batch-1")
    assert first_batch is not None
    assert isinstance(first_batch.vectors, np.memmap)
    manifest = next(tmp_path.rglob("manifest.json"))
    assert '"sha256"' in manifest.read_text(encoding="utf-8")

    store.fail = True
    second = TRAKETrajectoryCache(store, snapshot_store=snapshot_store)
    second_snapshot = second.warm("batch-1")

    assert second_snapshot["source"] == "disk"
    assert store.stream_calls == 1
    second_batch = second.batch("batch-1")
    assert second_batch is not None
    assert isinstance(second_batch.vectors, np.memmap)
    assert second_batch.video("video-1").source_frame_indices.tolist() == [10, 20]  # type: ignore[union-attr]


def test_invalid_persistent_snapshot_is_quarantined_and_rebuilt(tmp_path) -> None:
    store = _MutableStore()
    snapshot_store = TRAKETrajectorySnapshotStore(
        tmp_path,
        channel="visual",
        vector_size=2,
        cache_version=TRAKE_TRAJECTORY_CACHE_VERSION,
    )
    TRAKETrajectoryCache(store, snapshot_store=snapshot_store).warm("batch-1")
    vectors = next(tmp_path.rglob("vectors.npy"))
    with vectors.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\0")

    rebuilt = TRAKETrajectoryCache(store, snapshot_store=snapshot_store)
    snapshot = rebuilt.warm("batch-1")

    assert snapshot["source"] == "rebuilt"
    assert store.stream_calls == 2
    assert any(path.name.startswith(".invalid-") for path in vectors.parent.parent.iterdir())


def test_cache_filters_until_frames_and_timestamps_are_both_strict() -> None:
    store = _MutableStore()
    store.points = [
        _point(10, [1.0, 0.0], timestamp_ms=100),
        _point(20, [0.0, 1.0], timestamp_ms=100),
        _point(30, [1.0, 1.0], timestamp_ms=90),
        _point(40, [1.0, -1.0], timestamp_ms=110),
    ]
    cache = TRAKETrajectoryCache(store)

    cache.warm("batch-1")

    video = cache.batch("batch-1").video("video-1")  # type: ignore[union-attr]
    assert video is not None
    assert np.all(np.diff(video.source_frame_indices) > 0)
    assert np.all(np.diff(video.timestamps_ms) > 0)


def test_require_never_starts_a_request_time_build() -> None:
    store = _MutableStore()
    cache = TRAKETrajectoryCache(store)

    with pytest.raises(TRAKETrajectoryCacheError, match="not ready"):
        cache.require("batch-1")

    assert store.stream_calls == 0


def test_snapshot_lock_is_single_flight_across_cache_instances(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStore(_MutableStore):
        def iter_points_with_vectors(self, filters: dict[str, Any], **_kwargs: Any):
            self.stream_calls += 1
            self.filters.append(filters)
            started.set()
            assert release.wait(timeout=5)
            yield from self.points

    store = BlockingStore()
    snapshot_store = TRAKETrajectorySnapshotStore(
        tmp_path,
        channel="organizer_clip",
        vector_size=2,
        cache_version=TRAKE_TRAJECTORY_CACHE_VERSION,
    )
    first = TRAKETrajectoryCache(store, snapshot_store=snapshot_store)
    second = TRAKETrajectoryCache(store, snapshot_store=snapshot_store)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first.warm, "batch-1")
        assert started.wait(timeout=5)
        second_future = executor.submit(second.warm, "batch-1")
        release.set()
        snapshots = [first_future.result(timeout=5), second_future.result(timeout=5)]

    assert store.stream_calls == 1
    assert {snapshot["source"] for snapshot in snapshots} == {"rebuilt", "disk"}
