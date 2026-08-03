"""Persistent, checksummed snapshots for exact TRAKE trajectories.

Snapshots are immutable fingerprint directories containing ordinary ``.npy``
files so fresh API and Streamlit processes can memory-map the same arrays.  All
filesystem activity is deferred until an explicit load/build operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping
from uuid import uuid4

import numpy as np

TRAKE_SNAPSHOT_SCHEMA_VERSION = 1
_CACHE_DIRECTORY_VERSION = "v1"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 4 * 1024 * 1024


class TRAKETrajectorySnapshotError(RuntimeError):
    """Sanitized persistent-snapshot failure."""


@dataclass(frozen=True, slots=True)
class TRAKETrajectorySnapshot:
    """Validated memory-mapped trajectory arrays and their video boundaries."""

    batch_id: str
    channel: str
    fingerprint: str
    source_frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    vectors: np.ndarray
    video_ids: tuple[str, ...]
    video_offsets: np.ndarray
    source_point_count: int


@dataclass(frozen=True, slots=True)
class SnapshotLoadResult:
    """Non-throwing result used to distinguish misses from invalid snapshots."""

    snapshot: TRAKETrajectorySnapshot | None
    reason_code: str
    duration_ms: float


class _CrossProcessFileLock:
    """Small stdlib-only advisory lock working on Windows and POSIX."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._handle: Any = None

    def __enter__(self) -> "_CrossProcessFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        started = self._monotonic()
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return self
            except (BlockingIOError, OSError):
                if self._monotonic() - started >= self.timeout_seconds:
                    handle.close()
                    raise TRAKETrajectorySnapshotError(
                        "TRAKE trajectory snapshot lock timed out"
                    ) from None
                self._sleep(0.1)

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _safe_component(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized or _SAFE_COMPONENT.fullmatch(normalized) is None:
        raise ValueError(f"{name} contains unsupported path characters")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _strict_int(value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("snapshot integer field is invalid")
    return value


class TRAKETrajectorySnapshotStore:
    """Load and atomically publish immutable trajectory snapshots."""

    def __init__(
        self,
        root: str | Path,
        *,
        channel: str,
        vector_size: int,
        cache_version: str,
        lock_timeout_seconds: float = 600.0,
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be positive")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        self.root = Path(root)
        self.channel = _safe_component(channel, name="channel")
        self.vector_size = int(vector_size)
        self.cache_version = str(cache_version)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._wall_clock = wall_clock
        self._monotonic = monotonic
        self._sleep = sleep

    def _channel_root(self, batch_id: str) -> Path:
        batch = _safe_component(batch_id, name="batch_id")
        return self.root / _CACHE_DIRECTORY_VERSION / batch / self.channel

    def _snapshot_path(self, batch_id: str, fingerprint: str) -> Path:
        if _SHA256.fullmatch(fingerprint) is None:
            raise ValueError("fingerprint must be a lowercase SHA-256 digest")
        return self._channel_root(batch_id) / fingerprint

    @contextmanager
    def build_lock(self, batch_id: str) -> Iterator[None]:
        """Serialize snapshot construction across local processes."""

        lock = _CrossProcessFileLock(
            self._channel_root(batch_id) / ".build.lock",
            timeout_seconds=self.lock_timeout_seconds,
            monotonic=self._monotonic,
            sleep=self._sleep,
        )
        with lock:
            yield

    @staticmethod
    def _file_spec(
        manifest: Mapping[str, Any],
        name: str,
        *,
        dtype: str,
        shape: tuple[int, ...],
    ) -> Mapping[str, Any]:
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("snapshot file manifest is missing")
        spec = files.get(name)
        if not isinstance(spec, Mapping):
            raise ValueError("snapshot file entry is missing")
        if spec.get("dtype") != dtype or tuple(spec.get("shape", ())) != shape:
            raise ValueError("snapshot array contract does not match")
        _strict_int(spec.get("size_bytes"), minimum=1)
        checksum = spec.get("sha256")
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            raise ValueError("snapshot checksum is invalid")
        return spec

    def load(self, batch_id: str, fingerprint: str) -> SnapshotLoadResult:
        """Load a valid snapshot without exposing parse or filesystem details."""

        started = self._monotonic()
        path = self._snapshot_path(batch_id, fingerprint)
        if not path.is_dir():
            return SnapshotLoadResult(
                None,
                "snapshot_missing",
                round((self._monotonic() - started) * 1_000, 3),
            )
        try:
            manifest_path = path / "manifest.json"
            if manifest_path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("snapshot manifest is too large")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise ValueError("snapshot manifest must be an object")
            if manifest.get("schema_version") != TRAKE_SNAPSHOT_SCHEMA_VERSION:
                raise ValueError("snapshot schema is incompatible")
            if manifest.get("cache_version") != self.cache_version:
                raise ValueError("snapshot cache version is incompatible")
            if manifest.get("batch_id") != batch_id:
                raise ValueError("snapshot batch does not match")
            if manifest.get("channel") != self.channel:
                raise ValueError("snapshot channel does not match")
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("snapshot fingerprint does not match")
            if _strict_int(manifest.get("vector_size"), minimum=1) != self.vector_size:
                raise ValueError("snapshot vector dimension does not match")
            frame_count = _strict_int(manifest.get("frame_count"))
            video_count = _strict_int(manifest.get("video_count"))
            source_point_count = _strict_int(manifest.get("source_point_count"))
            raw_video_ids = manifest.get("video_ids")
            if not isinstance(raw_video_ids, list) or len(raw_video_ids) != video_count:
                raise ValueError("snapshot video list does not match")
            video_ids = tuple(
                _safe_component(value, name="video_id")
                if isinstance(value, str)
                else (_ for _ in ()).throw(ValueError("snapshot video ID is invalid"))
                for value in raw_video_ids
            )
            if tuple(sorted(set(video_ids))) != video_ids:
                raise ValueError("snapshot video IDs are not sorted and unique")

            expected = {
                "vectors.npy": ("<f4", (frame_count, self.vector_size)),
                "source_frame_indices.npy": ("<i8", (frame_count,)),
                "timestamps_ms.npy": ("<i8", (frame_count,)),
                "video_offsets.npy": ("<i8", (video_count + 1,)),
            }
            for filename, (dtype, shape) in expected.items():
                spec = self._file_spec(manifest, filename, dtype=dtype, shape=shape)
                candidate = path / filename
                if candidate.stat().st_size != int(spec["size_bytes"]):
                    raise ValueError("snapshot file size does not match")
                if _sha256(candidate) != spec["sha256"]:
                    raise ValueError("snapshot checksum does not match")

            vectors = np.load(path / "vectors.npy", mmap_mode="r", allow_pickle=False)
            source_frames = np.load(
                path / "source_frame_indices.npy", mmap_mode="r", allow_pickle=False
            )
            timestamps = np.load(path / "timestamps_ms.npy", mmap_mode="r", allow_pickle=False)
            offsets = np.load(path / "video_offsets.npy", mmap_mode="r", allow_pickle=False)
            arrays = {
                "vectors.npy": vectors,
                "source_frame_indices.npy": source_frames,
                "timestamps_ms.npy": timestamps,
                "video_offsets.npy": offsets,
            }
            for filename, array in arrays.items():
                dtype, shape = expected[filename]
                if array.dtype.str != dtype or array.shape != shape:
                    raise ValueError("snapshot array header does not match")
                array.setflags(write=False)
            if offsets.shape[0] == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != frame_count:
                raise ValueError("snapshot offsets do not span the trajectory")
            if video_count and not np.all(np.diff(offsets) > 0):
                raise ValueError("snapshot video offsets are not strictly increasing")
            for index in range(video_count):
                start = int(offsets[index])
                stop = int(offsets[index + 1])
                if not np.all(np.diff(source_frames[start:stop]) > 0):
                    raise ValueError("snapshot source frames are not strictly increasing")
                if not np.all(np.diff(timestamps[start:stop]) > 0):
                    raise ValueError("snapshot timestamps are not strictly increasing")
            return SnapshotLoadResult(
                TRAKETrajectorySnapshot(
                    batch_id=batch_id,
                    channel=self.channel,
                    fingerprint=fingerprint,
                    source_frame_indices=source_frames,
                    timestamps_ms=timestamps,
                    vectors=vectors,
                    video_ids=video_ids,
                    video_offsets=offsets,
                    source_point_count=source_point_count,
                ),
                "snapshot_hit",
                round((self._monotonic() - started) * 1_000, 3),
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return SnapshotLoadResult(
                None,
                "snapshot_invalid",
                round((self._monotonic() - started) * 1_000, 3),
            )

    @staticmethod
    def _array_manifest(path: Path, array: np.ndarray) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    def publish(
        self,
        *,
        batch_id: str,
        fingerprint: str,
        vectors: np.ndarray,
        source_frame_indices: np.ndarray,
        timestamps_ms: np.ndarray,
        video_ids: tuple[str, ...],
        video_offsets: np.ndarray,
        source_point_count: int,
    ) -> None:
        """Publish one complete snapshot; caller must hold ``build_lock``."""

        final = self._snapshot_path(batch_id, fingerprint)
        parent = final.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".tmp-{fingerprint}-{os.getpid()}-{uuid4().hex}"
        temporary.mkdir()
        try:
            arrays = {
                "vectors.npy": np.ascontiguousarray(vectors, dtype="<f4"),
                "source_frame_indices.npy": np.ascontiguousarray(
                    source_frame_indices, dtype="<i8"
                ),
                "timestamps_ms.npy": np.ascontiguousarray(timestamps_ms, dtype="<i8"),
                "video_offsets.npy": np.ascontiguousarray(video_offsets, dtype="<i8"),
            }
            files: dict[str, Mapping[str, Any]] = {}
            for filename, array in arrays.items():
                path = temporary / filename
                with path.open("wb") as handle:
                    np.save(handle, array, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                files[filename] = self._array_manifest(path, array)
            manifest = {
                "schema_version": TRAKE_SNAPSHOT_SCHEMA_VERSION,
                "cache_version": self.cache_version,
                "batch_id": batch_id,
                "channel": self.channel,
                "fingerprint": fingerprint,
                "vector_size": self.vector_size,
                "created_at_unix": self._wall_clock(),
                "source_point_count": int(source_point_count),
                "video_count": len(video_ids),
                "frame_count": int(vectors.shape[0]),
                "video_ids": list(video_ids),
                "files": {name: dict(spec) for name, spec in files.items()},
            }
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            if final.exists():
                invalid = parent / (
                    f".invalid-{fingerprint}-{int(self._wall_clock())}-{uuid4().hex}"
                )
                os.replace(final, invalid)
            os.replace(temporary, final)
            if os.name != "nt":
                descriptor = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except Exception as error:
            raise TRAKETrajectorySnapshotError(
                "TRAKE trajectory snapshot publication failed"
            ) from error


__all__ = [
    "SnapshotLoadResult",
    "TRAKE_SNAPSHOT_SCHEMA_VERSION",
    "TRAKETrajectorySnapshot",
    "TRAKETrajectorySnapshotError",
    "TRAKETrajectorySnapshotStore",
]
