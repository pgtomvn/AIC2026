"""Compact, exact full-timeline cache for TRAKE Stage-B alignment."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from src.retrieval.trake_snapshot import (
    TRAKETrajectorySnapshot,
    TRAKETrajectorySnapshotError,
    TRAKETrajectorySnapshotStore,
)

TRAKE_TRAJECTORY_CACHE_VERSION = "trake-trajectory-v3"


class TRAKETrajectoryCacheError(RuntimeError):
    """Sanitized trajectory-cache build failure."""


@dataclass(frozen=True, slots=True)
class TRAKEVideoTrajectory:
    """Read-only views over one video's canonical full timeline."""

    source_frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    vectors: np.ndarray

    def __len__(self) -> int:
        return int(self.source_frame_indices.shape[0])


@dataclass(frozen=True, slots=True)
class _BatchTrajectories:
    batch_id: str
    fingerprint: str
    source_frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    vectors: np.ndarray
    video_slices: Mapping[str, slice]
    source_point_count: int

    def video(self, video_id: str) -> TRAKEVideoTrajectory | None:
        selected = self.video_slices.get(video_id)
        if selected is None:
            return None
        return TRAKEVideoTrajectory(
            source_frame_indices=self.source_frame_indices[selected],
            timestamps_ms=self.timestamps_ms[selected],
            vectors=self.vectors[selected],
        )

    @property
    def frame_count(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def nbytes(self) -> int:
        return int(
            self.vectors.nbytes
            + self.source_frame_indices.nbytes
            + self.timestamps_ms.nbytes
        )


class TRAKETrajectoryCache:
    """Lazy, single-flight cache of one channel's exact batch trajectories.

    Construction performs no I/O. ``warm`` streams the selected dense batch
    once, validates and canonicalizes every point, then atomically publishes
    compact shared NumPy arrays plus per-video slices.
    """

    def __init__(
        self,
        vector_store: Any,
        *,
        vector_size: int | None = None,
        batch_size: int = 16_384,
        fingerprint_provider: Callable[[str], str] | None = None,
        snapshot_store: TRAKETrajectorySnapshotStore | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        dimension = int(vector_size or getattr(vector_store, "vector_size", 0))
        if dimension < 1:
            raise ValueError("vector_size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.vector_store = vector_store
        self.vector_size = dimension
        self.batch_size = batch_size
        self.fingerprint_provider = fingerprint_provider
        self.snapshot_store = snapshot_store
        self._monotonic = monotonic
        self._condition = threading.Condition(threading.RLock())
        self._entry: _BatchTrajectories | None = None
        self._loading = False
        self._loading_batch_id: str | None = None
        self._generation = 0
        self._failure_generation = 0
        self._state = "idle"
        self._reason_code = "not_warmed"
        self._source: str | None = None
        self._load_duration_ms = 0.0
        self._build_duration_ms = 0.0
        self._persist_duration_ms = 0.0

    @staticmethod
    def _payload_int(payload: Mapping[str, Any], name: str) -> int | None:
        value = payload.get(name)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("+").isdigit():
            return int(value.strip())
        return None

    def _default_fingerprint(self, batch_id: str) -> str:
        store = self.vector_store
        bound_store = store
        bind = getattr(store, "for_batch", None)
        if callable(bind):
            bound_store = bind(batch_id)
        count: int | None = None
        count_points = getattr(bound_store, "count", None)
        if callable(count_points):
            count = int(count_points({"batch_id": batch_id}))
        revision = getattr(store, "trajectory_revision", None)
        if callable(revision):
            revision = revision(batch_id)
        payload = {
            "version": TRAKE_TRAJECTORY_CACHE_VERSION,
            "batch_id": batch_id,
            "store_type": f"{type(store).__module__}.{type(store).__qualname__}",
            "collection": str(
                getattr(bound_store, "collection_name", getattr(store, "collection_name", ""))
            ),
            "vector_size": self.vector_size,
            "distance": str(getattr(bound_store, "distance", "")),
            "count": count,
            "revision": None if revision is None else str(revision),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _fingerprint(self, batch_id: str) -> str:
        if self.fingerprint_provider is None:
            return self._default_fingerprint(batch_id)
        value = self.fingerprint_provider(batch_id)
        if not isinstance(value, str) or not value:
            raise ValueError("trajectory fingerprint provider returned an invalid value")
        return hashlib.sha256(
            (
                f"{TRAKE_TRAJECTORY_CACHE_VERSION}:"
                f"{self.vector_size}:{batch_id}:{value}"
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _from_snapshot(snapshot: TRAKETrajectorySnapshot) -> _BatchTrajectories:
        slices = {
            video_id: slice(
                int(snapshot.video_offsets[index]),
                int(snapshot.video_offsets[index + 1]),
            )
            for index, video_id in enumerate(snapshot.video_ids)
        }
        return _BatchTrajectories(
            batch_id=snapshot.batch_id,
            fingerprint=snapshot.fingerprint,
            source_frame_indices=snapshot.source_frame_indices,
            timestamps_ms=snapshot.timestamps_ms,
            vectors=snapshot.vectors,
            video_slices=MappingProxyType(slices),
            source_point_count=snapshot.source_point_count,
        )

    @staticmethod
    def _snapshot_layout(entry: _BatchTrajectories) -> tuple[tuple[str, ...], np.ndarray]:
        video_ids = tuple(entry.video_slices)
        offsets = np.empty(len(video_ids) + 1, dtype=np.int64)
        offsets[0] = 0
        for index, video_id in enumerate(video_ids, start=1):
            selected = entry.video_slices[video_id]
            offsets[index] = int(selected.stop or 0)
        return video_ids, offsets

    def _progress(self, reason_code: str) -> None:
        with self._condition:
            self._state = "warming"
            self._reason_code = reason_code

    def _build(self, batch_id: str, fingerprint: str) -> _BatchTrajectories:
        # Production adapters provide one contiguous NumPy matrix per Qdrant
        # page. Metadata stores only page/row coordinates, avoiding 177k small
        # ndarray and normalized-hit objects on the reference batch.
        vector_batches: list[np.ndarray] = []
        grouped: dict[str, list[tuple[int, int, int, int]]] = {}
        source_point_count = 0

        def retain_payload(
            payload: Any,
            vector_batch_index: int,
            vector_row_index: int,
        ) -> None:
            if not isinstance(payload, Mapping) or payload.get("batch_id") != batch_id:
                return
            video_id = payload.get("video_id")
            frame_idx = self._payload_int(payload, "source_frame_idx")
            timestamp_ms = self._payload_int(payload, "timestamp_ms")
            if (
                not isinstance(video_id, str)
                or not video_id
                or frame_idx is None
                or frame_idx < 0
                or timestamp_ms is None
                or timestamp_ms < 0
            ):
                return
            grouped.setdefault(video_id, []).append(
                (
                    frame_idx,
                    timestamp_ms,
                    vector_batch_index,
                    vector_row_index,
                )
            )

        batch_iterator = getattr(
            self.vector_store,
            "iter_point_batches_with_vectors",
            None,
        )
        if callable(batch_iterator):
            batches = batch_iterator(
                {"batch_id": batch_id},
                batch_size=self.batch_size,
            )
            for point_batch in batches:
                payloads = tuple(getattr(point_batch, "payloads", ()))
                matrix = np.asarray(
                    getattr(point_batch, "vectors", ()), dtype=np.float32
                )
                source_point_count += int(
                    getattr(point_batch, "source_point_count", len(payloads))
                )
                if matrix.shape != (len(payloads), self.vector_size):
                    continue
                batch_index = len(vector_batches)
                vector_batches.append(np.ascontiguousarray(matrix))
                for row_index, payload in enumerate(payloads):
                    retain_payload(payload, batch_index, row_index)
        else:
            points = self.vector_store.iter_points_with_vectors(
                {"batch_id": batch_id},
                batch_size=self.batch_size,
            )
            for point in points:
                source_point_count += 1
                if not isinstance(point, Mapping):
                    continue
                raw_vector = point.get("vector")
                if isinstance(raw_vector, Mapping):
                    if len(raw_vector) != 1:
                        continue
                    raw_vector = next(iter(raw_vector.values()))
                try:
                    vector = np.asarray(raw_vector, dtype=np.float32)
                except (TypeError, ValueError):
                    continue
                if vector.shape != (self.vector_size,):
                    continue
                batch_index = len(vector_batches)
                vector_batches.append(vector.reshape(1, self.vector_size))
                retain_payload(point.get("payload", {}), batch_index, 0)

        canonical: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        total_frames = 0
        for video_id in sorted(grouped):
            rows = grouped[video_id]
            matrix = np.stack(
                [vector_batches[row[2]][row[3]] for row in rows]
            )
            frames = np.fromiter((row[0] for row in rows), dtype=np.int64)
            times = np.fromiter((row[1] for row in rows), dtype=np.int64)
            finite = np.all(np.isfinite(matrix), axis=1)
            norms = np.linalg.norm(matrix, axis=1)
            valid = finite & np.isfinite(norms) & (norms > 0)
            if not np.any(valid):
                continue
            matrix = matrix[valid] / norms[valid, None]
            frames = frames[valid]
            times = times[valid]

            # Organizer ordinals can map to the same canonical source frame.
            # Merge their normalized vectors before ordering in real time.
            by_frame: dict[int, tuple[int, np.ndarray]] = {}
            for frame_idx, timestamp_ms, vector in zip(
                frames.tolist(), times.tolist(), matrix, strict=True
            ):
                existing = by_frame.get(frame_idx)
                if existing is None:
                    by_frame[frame_idx] = (timestamp_ms, vector.copy())
                else:
                    by_frame[frame_idx] = (
                        min(existing[0], timestamp_ms),
                        existing[1] + vector,
                    )
            merged_frames = np.fromiter(by_frame, dtype=np.int64)
            merged_times = np.fromiter(
                (item[0] for item in by_frame.values()), dtype=np.int64
            )
            merged_vectors = np.stack([item[1] for item in by_frame.values()])
            merged_norms = np.linalg.norm(merged_vectors, axis=1)
            merged_valid = np.isfinite(merged_norms) & (merged_norms > 0)
            if not np.any(merged_valid):
                continue
            merged_frames = merged_frames[merged_valid]
            merged_times = merged_times[merged_valid]
            merged_vectors = (
                merged_vectors[merged_valid] / merged_norms[merged_valid, None]
            ).astype(np.float32, copy=False)
            order = np.lexsort((merged_frames, merged_times))
            merged_frames = merged_frames[order]
            merged_times = merged_times[order]
            merged_vectors = merged_vectors[order]

            keep: list[int] = []
            previous_frame = -1
            previous_timestamp = -1
            for index, (frame_idx, timestamp_ms) in enumerate(
                zip(merged_frames, merged_times, strict=True)
            ):
                if int(frame_idx) > previous_frame and int(timestamp_ms) > previous_timestamp:
                    keep.append(index)
                    previous_frame = int(frame_idx)
                    previous_timestamp = int(timestamp_ms)
            if keep:
                canonical[video_id] = (
                    merged_frames[keep],
                    merged_times[keep],
                    merged_vectors[keep],
                )
                total_frames += len(keep)

        vectors = np.empty((total_frames, self.vector_size), dtype=np.float32)
        source_frames = np.empty(total_frames, dtype=np.int64)
        timestamps = np.empty(total_frames, dtype=np.int64)
        slices: dict[str, slice] = {}
        cursor = 0
        for video_id in sorted(canonical):
            video_frames, video_times, video_vectors = canonical[video_id]
            stop = cursor + len(video_frames)
            timestamps[cursor:stop] = video_times
            source_frames[cursor:stop] = video_frames
            vectors[cursor:stop] = video_vectors
            slices[video_id] = slice(cursor, stop)
            cursor = stop
        for array in (vectors, source_frames, timestamps):
            array.setflags(write=False)
        return _BatchTrajectories(
            batch_id=batch_id,
            fingerprint=fingerprint,
            source_frame_indices=source_frames,
            timestamps_ms=timestamps,
            vectors=vectors,
            video_slices=MappingProxyType(slices),
            source_point_count=source_point_count,
        )

    def warm(self, batch_id: str) -> dict[str, Any]:
        """Validate/rebuild one batch with single-flight publication."""

        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be a non-empty string")
        batch_id = batch_id.strip()
        with self._condition:
            observed_generation = self._generation
            observed_failure = self._failure_generation
            while self._loading:
                self._condition.wait()
                if self._failure_generation != observed_failure:
                    raise TRAKETrajectoryCacheError(
                        "TRAKE trajectory cache warm-up failed"
                    )
                if (
                    self._generation != observed_generation
                    and self._entry is not None
                    and self._entry.batch_id == batch_id
                ):
                    return self.snapshot()
            self._loading = True
            self._loading_batch_id = batch_id
            self._state = "warming"
            self._reason_code = "fingerprint_check"
            self._source = None
            self._load_duration_ms = 0.0
            self._build_duration_ms = 0.0
            self._persist_duration_ms = 0.0
            existing = self._entry if self._entry and self._entry.batch_id == batch_id else None

        try:
            fingerprint = self._fingerprint(batch_id)
            if existing is not None and existing.fingerprint == fingerprint:
                built = existing
                source = "memory"
                reason_code = "memory_hit"
            else:
                with self._condition:
                    self._entry = None
                built = None
                source = "memory"
                reason_code = "warmup_completed"
                snapshot_store = self.snapshot_store
                if snapshot_store is not None:
                    self._progress("snapshot_loading")
                    loaded = snapshot_store.load(batch_id, fingerprint)
                    self._load_duration_ms += loaded.duration_ms
                    if loaded.snapshot is not None:
                        built = self._from_snapshot(loaded.snapshot)
                        source = "disk"
                        reason_code = loaded.reason_code
                    else:
                        self._progress("waiting_for_snapshot_builder")
                        with snapshot_store.build_lock(batch_id):
                            self._progress("snapshot_loading")
                            loaded = snapshot_store.load(batch_id, fingerprint)
                            self._load_duration_ms += loaded.duration_ms
                            if loaded.snapshot is not None:
                                built = self._from_snapshot(loaded.snapshot)
                                source = "disk"
                                reason_code = loaded.reason_code
                            else:
                                self._progress("trajectory_cache_building")
                                build_started = self._monotonic()
                                built = self._build(batch_id, fingerprint)
                                self._build_duration_ms = round(
                                    (self._monotonic() - build_started) * 1_000,
                                    3,
                                )
                                video_ids, offsets = self._snapshot_layout(built)
                                persist_started = self._monotonic()
                                try:
                                    snapshot_store.publish(
                                        batch_id=batch_id,
                                        fingerprint=fingerprint,
                                        vectors=built.vectors,
                                        source_frame_indices=built.source_frame_indices,
                                        timestamps_ms=built.timestamps_ms,
                                        video_ids=video_ids,
                                        video_offsets=offsets,
                                        source_point_count=built.source_point_count,
                                    )
                                    self._persist_duration_ms = round(
                                        (self._monotonic() - persist_started) * 1_000,
                                        3,
                                    )
                                    reloaded = snapshot_store.load(batch_id, fingerprint)
                                    self._load_duration_ms += reloaded.duration_ms
                                    if reloaded.snapshot is None:
                                        raise TRAKETrajectorySnapshotError(
                                            "published trajectory snapshot could not be loaded"
                                        )
                                    built = self._from_snapshot(reloaded.snapshot)
                                    source = "rebuilt"
                                    reason_code = "snapshot_built"
                                except TRAKETrajectorySnapshotError:
                                    self._persist_duration_ms = round(
                                        (self._monotonic() - persist_started) * 1_000,
                                        3,
                                    )
                                    source = "memory_unpersisted"
                                    reason_code = "snapshot_persistence_failed"
                if built is None:
                    self._progress("trajectory_cache_building")
                    build_started = self._monotonic()
                    built = self._build(batch_id, fingerprint)
                    self._build_duration_ms = round(
                        (self._monotonic() - build_started) * 1_000,
                        3,
                    )
        except Exception as error:
            with self._condition:
                self._loading = False
                self._loading_batch_id = None
                self._state = "failed"
                self._reason_code = "trajectory_cache_build_failed"
                self._failure_generation += 1
                self._condition.notify_all()
            raise TRAKETrajectoryCacheError("TRAKE trajectory cache warm-up failed") from error

        with self._condition:
            self._entry = built
            self._loading = False
            self._loading_batch_id = None
            self._state = "ready"
            self._reason_code = reason_code
            self._source = source
            self._generation += 1
            self._condition.notify_all()
            return self.snapshot()

    def batch(self, batch_id: str) -> _BatchTrajectories | None:
        """Return the current immutable batch cache without blocking."""

        with self._condition:
            if self._entry is None or self._entry.batch_id != batch_id:
                return None
            return self._entry

    def get_or_warm(self, batch_id: str) -> _BatchTrajectories:
        """Return an exact batch cache, warming synchronously when absent."""

        cached = self.batch(batch_id)
        if cached is not None:
            return cached
        self.warm(batch_id)
        cached = self.batch(batch_id)
        if cached is None:  # pragma: no cover - guarded by warm's atomic publication
            raise TRAKETrajectoryCacheError("TRAKE trajectory cache is unavailable")
        return cached

    def require(self, batch_id: str) -> _BatchTrajectories:
        """Return an already-warmed cache without performing request-time I/O."""

        cached = self.batch(batch_id)
        if cached is None:
            raise TRAKETrajectoryCacheError("TRAKE trajectory cache is not ready")
        return cached

    def invalidate(self) -> None:
        """Drop the published entry; an in-flight build remains authoritative."""

        with self._condition:
            self._entry = None
            if not self._loading:
                self._state = "idle"
                self._reason_code = "invalidated"
                self._source = None

    def snapshot(self) -> dict[str, Any]:
        """Return sanitized, nonblocking cache state and memory usage."""

        with self._condition:
            entry = self._entry
            return {
                "state": self._state,
                "reason_code": self._reason_code,
                "batch_id": (
                    entry.batch_id if entry is not None else self._loading_batch_id
                ),
                "fingerprint": entry.fingerprint if entry is not None else None,
                "video_count": len(entry.video_slices) if entry is not None else 0,
                "frame_count": entry.frame_count if entry is not None else 0,
                "source_point_count": entry.source_point_count if entry is not None else 0,
                "nbytes": entry.nbytes if entry is not None else 0,
                "source": self._source,
                "load_duration_ms": self._load_duration_ms,
                "build_duration_ms": self._build_duration_ms,
                "persist_duration_ms": self._persist_duration_ms,
            }


__all__ = [
    "TRAKE_TRAJECTORY_CACHE_VERSION",
    "TRAKETrajectoryCache",
    "TRAKETrajectoryCacheError",
    "TRAKEVideoTrajectory",
]
