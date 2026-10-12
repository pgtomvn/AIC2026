"""Canonical SQLite storage for extracted frames and timestamped text.

Construction is side-effect free.  Call :meth:`ArtifactRepository.initialize`
explicitly before using the repository.  Ordinary upserts treat canonical rows
as immutable: an exact retry is a no-op, while an identity collision with a
different record or processing fingerprint raises :class:`ArtifactConflictError`.
Intentional reprocessing uses :meth:`ArtifactRepository.replace_video`, which
replaces one video's records in a single transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.contracts import FrameRecord, RefinedFrameRecord, TextSegment


class ArtifactRepositoryError(RuntimeError):
    """Base error for canonical artifact storage."""


class RepositoryNotInitialized(ArtifactRepositoryError):
    """Raised when storage is used before explicit initialization."""


class ArtifactConflictError(ArtifactRepositoryError):
    """Raised when a canonical identity is reused for incompatible content."""


class RepositoryCorruptionError(ArtifactRepositoryError):
    """Raised when a persisted row cannot be reconstructed as a contract."""


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """Counts produced by one transactional, idempotent upsert."""

    inserted: int = 0
    unchanged: int = 0
    readiness_changed: int = 0


@dataclass(frozen=True, slots=True)
class VideoReplacement:
    """Counts committed by an atomic per-video replacement."""

    frame_count: int
    text_segment_count: int


@dataclass(frozen=True, slots=True)
class BatchArtifactCounts:
    """Canonical and index-ready artifact counts for one batch."""

    batch_id: str
    video_count: int
    frame_count: int
    text_segment_count: int
    index_ready_frame_count: int
    index_ready_text_segment_count: int
    text_segments_by_kind: dict[str, int] = field(default_factory=dict)


_SCHEMA_VERSION = 3
_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS canonical_frames (
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    source_frame_idx INTEGER NOT NULL CHECK (source_frame_idx >= 0),
    keyframe_ordinal INTEGER NOT NULL CHECK (keyframe_ordinal >= 1),
    pts INTEGER NOT NULL,
    time_base TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL CHECK (timestamp_ms >= 0),
    fps_num INTEGER NOT NULL CHECK (fps_num >= 1),
    fps_den INTEGER NOT NULL CHECK (fps_den >= 1),
    shot_id TEXT NOT NULL,
    sampling_reasons_json TEXT NOT NULL,
    image_uri TEXT NOT NULL,
    video_uri TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    artifact_fingerprint TEXT NOT NULL,
    record_fingerprint TEXT NOT NULL,
    index_ready INTEGER NOT NULL DEFAULT 1 CHECK (index_ready IN (0, 1)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (batch_id, video_id, source_frame_idx),
    UNIQUE (batch_id, video_id, keyframe_ordinal)
);
CREATE INDEX IF NOT EXISTS idx_canonical_frames_timestamp
ON canonical_frames(batch_id, video_id, timestamp_ms, source_frame_idx);
CREATE INDEX IF NOT EXISTS idx_canonical_frames_ready
ON canonical_frames(index_ready, batch_id, video_id, source_frame_idx);

CREATE TABLE IF NOT EXISTS canonical_frame_mappings (
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    keyframe_ordinal INTEGER NOT NULL CHECK (keyframe_ordinal >= 1),
    source_frame_idx INTEGER NOT NULL CHECK (source_frame_idx >= 0),
    PRIMARY KEY (batch_id, video_id, keyframe_ordinal),
    FOREIGN KEY (batch_id, video_id, source_frame_idx)
        REFERENCES canonical_frames(batch_id, video_id, source_frame_idx)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_canonical_frame_mappings_source
ON canonical_frame_mappings(batch_id, video_id, source_frame_idx);

CREATE TABLE IF NOT EXISTS canonical_text_segments (
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms >= start_ms),
    source_frame_idx INTEGER CHECK (source_frame_idx IS NULL OR source_frame_idx >= 0),
    text_raw TEXT NOT NULL,
    text_nfc TEXT NOT NULL,
    text_accent_folded TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    model_version TEXT NOT NULL,
    polygons_json TEXT NOT NULL DEFAULT '[]',
    artifact_fingerprint TEXT NOT NULL,
    record_fingerprint TEXT NOT NULL,
    index_ready INTEGER NOT NULL DEFAULT 1 CHECK (index_ready IN (0, 1)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (batch_id, video_id, segment_id)
);
CREATE INDEX IF NOT EXISTS idx_canonical_text_segments_time
ON canonical_text_segments(batch_id, video_id, start_ms, end_ms, segment_id);
CREATE INDEX IF NOT EXISTS idx_canonical_text_segments_ready
ON canonical_text_segments(index_ready, batch_id, kind, video_id, segment_id);
"""

_REFINED_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_refined_frames (
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    source_frame_idx INTEGER NOT NULL CHECK (source_frame_idx >= 0),
    coarse_source_frame_idx INTEGER NOT NULL CHECK (coarse_source_frame_idx >= 0),
    pts INTEGER NOT NULL,
    time_base TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL CHECK (timestamp_ms >= 0),
    fps_num INTEGER NOT NULL CHECK (fps_num >= 1),
    fps_den INTEGER NOT NULL CHECK (fps_den >= 1),
    shot_id TEXT NOT NULL,
    image_uri TEXT NOT NULL,
    video_uri TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    image_width INTEGER NOT NULL CHECK (image_width >= 1),
    image_height INTEGER NOT NULL CHECK (image_height >= 1),
    image_bytes INTEGER NOT NULL CHECK (image_bytes >= 1),
    refiner_fingerprint TEXT NOT NULL,
    record_fingerprint TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (batch_id, video_id, source_frame_idx),
    FOREIGN KEY (batch_id, video_id, coarse_source_frame_idx)
        REFERENCES canonical_frames(batch_id, video_id, source_frame_idx)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_raw_refined_frames_timestamp
ON raw_refined_frames(batch_id, video_id, timestamp_ms, source_frame_idx);
"""

_ORDINAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS canonical_ordinal_frames (
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    keyframe_ordinal INTEGER NOT NULL CHECK (keyframe_ordinal >= 1),
    source_frame_idx INTEGER NOT NULL CHECK (source_frame_idx >= 0),
    timestamp_ms INTEGER NOT NULL CHECK (timestamp_ms >= 0),
    record_json TEXT NOT NULL,
    artifact_fingerprint TEXT NOT NULL,
    record_fingerprint TEXT NOT NULL,
    index_ready INTEGER NOT NULL DEFAULT 0 CHECK (index_ready IN (0, 1)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (batch_id, video_id, keyframe_ordinal)
);
CREATE INDEX IF NOT EXISTS idx_canonical_ordinal_frames_timestamp
ON canonical_ordinal_frames(batch_id, video_id, timestamp_ms, keyframe_ordinal);
CREATE INDEX IF NOT EXISTS idx_canonical_ordinal_frames_ready
ON canonical_ordinal_frames(index_ready, batch_id, video_id, keyframe_ordinal);
"""

_SCHEMA = _BASE_SCHEMA + _REFINED_SCHEMA + _ORDINAL_SCHEMA


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_positive_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer or None")
    if value < 1:
        raise ValueError("limit must be positive")
    return value


def _canonical_json(value: FrameRecord | RefinedFrameRecord | TextSegment) -> str:
    return json.dumps(
        value.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_fingerprint(value: FrameRecord | RefinedFrameRecord | TextSegment) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _frame_processing_fingerprint(record: FrameRecord, supplied: str | None) -> str:
    if supplied is not None:
        return _require_nonempty(supplied, "fingerprint")
    return f"frame:{record.source_sha256}:{record.extractor_version}"


def _text_processing_fingerprint(segment: TextSegment, supplied: str | None) -> str:
    if supplied is not None:
        return _require_nonempty(supplied, "fingerprint")
    return f"text:{segment.model_version}"


@dataclass(frozen=True, slots=True)
class _PreparedFrame:
    record: FrameRecord
    artifact_fingerprint: str
    record_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreparedTextSegment:
    segment: TextSegment
    artifact_fingerprint: str
    record_fingerprint: str


class ArtifactRepository:
    """Explicitly initialized canonical frame and text-segment repository."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int):
            raise TypeError("busy_timeout_ms must be an integer")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialized = False

    def initialize(self) -> None:
        """Create or validate the schema and enable SQLite WAL mode."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1_000)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms:d}")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, 1, 2, _SCHEMA_VERSION):
                raise ArtifactRepositoryError(
                    "Unsupported artifact schema version "
                    f"{version}; expected 0, 1, 2, or {_SCHEMA_VERSION}"
                )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            schema = (
                _SCHEMA
                if version in (0, _SCHEMA_VERSION)
                else (_REFINED_SCHEMA + _ORDINAL_SCHEMA if version == 1 else _ORDINAL_SCHEMA)
            )
            connection.executescript(
                f"BEGIN IMMEDIATE;\n{schema}\nPRAGMA user_version = {_SCHEMA_VERSION};\nCOMMIT;"
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(canonical_text_segments)"
                ).fetchall()
            }
            if "polygons_json" not in columns:
                connection.execute(
                    "ALTER TABLE canonical_text_segments "
                    "ADD COLUMN polygons_json TEXT NOT NULL DEFAULT '[]'"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        self._initialized = True

    def _connection(self) -> sqlite3.Connection:
        if not self._initialized:
            raise RepositoryNotInitialized(
                "ArtifactRepository.initialize() must be called before storage access"
            )
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1_000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms:d}")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _prepare_frames(
        records: Iterable[FrameRecord], fingerprint: str | None
    ) -> tuple[_PreparedFrame, ...]:
        prepared_by_identity: dict[tuple[str, str, int], _PreparedFrame] = {}
        ordinals: dict[tuple[str, str, int], tuple[str, str, int]] = {}
        for record in records:
            if not isinstance(record, FrameRecord):
                raise TypeError("records must contain only FrameRecord instances")
            prepared = _PreparedFrame(
                record=record,
                artifact_fingerprint=_frame_processing_fingerprint(record, fingerprint),
                record_fingerprint=_content_fingerprint(record),
            )
            identity = (record.batch_id, record.video_id, record.source_frame_idx)
            previous = prepared_by_identity.get(identity)
            if previous is not None and previous != prepared:
                raise ArtifactConflictError(f"Conflicting frame identity {identity!r} in input")
            ordinal_identity = (record.batch_id, record.video_id, record.keyframe_ordinal)
            prior_identity = ordinals.get(ordinal_identity)
            if prior_identity is not None and prior_identity != identity:
                raise ArtifactConflictError(
                    f"Keyframe ordinal {ordinal_identity!r} maps to multiple source frames"
                )
            prepared_by_identity[identity] = prepared
            ordinals[ordinal_identity] = identity
        return tuple(prepared_by_identity.values())

    @staticmethod
    def _prepare_ordinal_frames(
        records: Iterable[FrameRecord], fingerprint: str | None
    ) -> tuple[_PreparedFrame, ...]:
        prepared_by_identity: dict[tuple[str, str, int], _PreparedFrame] = {}
        for record in records:
            if not isinstance(record, FrameRecord):
                raise TypeError("records must contain only FrameRecord instances")
            prepared = _PreparedFrame(
                record=record,
                artifact_fingerprint=_frame_processing_fingerprint(record, fingerprint),
                record_fingerprint=_content_fingerprint(record),
            )
            identity = (record.batch_id, record.video_id, record.keyframe_ordinal)
            previous = prepared_by_identity.get(identity)
            if previous is not None and previous != prepared:
                raise ArtifactConflictError(f"Conflicting ordinal frame identity {identity!r} in input")
            prepared_by_identity[identity] = prepared
        return tuple(prepared_by_identity.values())

    @staticmethod
    def _prepare_text_segments(
        segments: Iterable[TextSegment], fingerprint: str | None
    ) -> tuple[_PreparedTextSegment, ...]:
        prepared_by_identity: dict[tuple[str, str, str], _PreparedTextSegment] = {}
        for segment in segments:
            if not isinstance(segment, TextSegment):
                raise TypeError("segments must contain only TextSegment instances")
            prepared = _PreparedTextSegment(
                segment=segment,
                artifact_fingerprint=_text_processing_fingerprint(segment, fingerprint),
                record_fingerprint=_content_fingerprint(segment),
            )
            identity = (segment.batch_id, segment.video_id, segment.segment_id)
            previous = prepared_by_identity.get(identity)
            if previous is not None and previous != prepared:
                raise ArtifactConflictError(
                    f"Conflicting text-segment identity {identity!r} in input"
                )
            prepared_by_identity[identity] = prepared
        return tuple(prepared_by_identity.values())

    @staticmethod
    def _insert_frame(
        connection: sqlite3.Connection,
        prepared: _PreparedFrame,
        *,
        index_ready: bool,
        timestamp: float,
    ) -> None:
        record = prepared.record
        connection.execute(
            """
            INSERT INTO canonical_frames(
                batch_id, video_id, source_frame_idx, keyframe_ordinal, pts,
                time_base, timestamp_ms, fps_num, fps_den, shot_id,
                sampling_reasons_json, image_uri, video_uri, source_sha256,
                extractor_version, artifact_fingerprint, record_fingerprint,
                index_ready, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.batch_id,
                record.video_id,
                record.source_frame_idx,
                record.keyframe_ordinal,
                record.pts,
                record.time_base,
                record.timestamp_ms,
                record.fps_num,
                record.fps_den,
                record.shot_id,
                json.dumps(list(record.sampling_reasons), ensure_ascii=False),
                record.image_uri,
                record.video_uri,
                record.source_sha256,
                record.extractor_version,
                prepared.artifact_fingerprint,
                prepared.record_fingerprint,
                int(index_ready),
                timestamp,
                timestamp,
            ),
        )

    @staticmethod
    def _insert_refined_frame(
        connection: sqlite3.Connection,
        record: RefinedFrameRecord,
        *,
        record_fingerprint: str,
        timestamp: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO raw_refined_frames(
                batch_id, video_id, source_frame_idx, coarse_source_frame_idx,
                pts, time_base, timestamp_ms, fps_num, fps_den, shot_id,
                image_uri, video_uri, source_sha256, image_sha256,
                image_width, image_height, image_bytes, refiner_fingerprint,
                record_fingerprint, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.batch_id,
                record.video_id,
                record.source_frame_idx,
                record.coarse_source_frame_idx,
                record.pts,
                record.time_base,
                record.timestamp_ms,
                record.fps_num,
                record.fps_den,
                record.shot_id,
                record.image_uri,
                record.video_uri,
                record.source_sha256,
                record.image_sha256,
                record.image_width,
                record.image_height,
                record.image_bytes,
                record.refiner_fingerprint,
                record_fingerprint,
                timestamp,
                timestamp,
            ),
        )

    @staticmethod
    def _upsert_frame_mapping(
        connection: sqlite3.Connection,
        batch_id: str,
        video_id: str,
        keyframe_ordinal: int,
        source_frame_idx: int,
    ) -> None:
        existing = connection.execute(
            """
            SELECT source_frame_idx FROM canonical_frame_mappings
            WHERE batch_id = ? AND video_id = ? AND keyframe_ordinal = ?
            """,
            (batch_id, video_id, keyframe_ordinal),
        ).fetchone()
        if existing is not None and int(existing["source_frame_idx"]) != source_frame_idx:
            raise ArtifactConflictError(
                f"Keyframe ordinal {keyframe_ordinal} for {batch_id}/{video_id} "
                f"already maps to source frame {existing['source_frame_idx']}"
            )
        connection.execute(
            """
            INSERT INTO canonical_frame_mappings(
                batch_id, video_id, keyframe_ordinal, source_frame_idx
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(batch_id, video_id, keyframe_ordinal) DO NOTHING
            """,
            (batch_id, video_id, keyframe_ordinal, source_frame_idx),
        )

    @staticmethod
    def _insert_text_segment(
        connection: sqlite3.Connection,
        prepared: _PreparedTextSegment,
        *,
        index_ready: bool,
        timestamp: float,
    ) -> None:
        segment = prepared.segment
        connection.execute(
            """
            INSERT INTO canonical_text_segments(
                batch_id, video_id, segment_id, kind, start_ms, end_ms,
                source_frame_idx, text_raw, text_nfc, text_accent_folded,
                confidence, model_version, polygons_json, artifact_fingerprint,
                record_fingerprint, index_ready, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment.batch_id,
                segment.video_id,
                segment.segment_id,
                segment.kind,
                segment.start_ms,
                segment.end_ms,
                segment.source_frame_idx,
                segment.text_raw,
                segment.text_nfc,
                segment.text_accent_folded,
                segment.confidence,
                segment.model_version,
                json.dumps(segment.polygons, ensure_ascii=False),
                prepared.artifact_fingerprint,
                prepared.record_fingerprint,
                int(index_ready),
                timestamp,
                timestamp,
            ),
        )

    @staticmethod
    def _check_existing(
        row: sqlite3.Row,
        *,
        artifact_fingerprint: str,
        record_fingerprint: str,
        description: str,
    ) -> None:
        if row["artifact_fingerprint"] != artifact_fingerprint:
            raise ArtifactConflictError(
                f"{description} already exists with a different processing fingerprint"
            )
        if row["record_fingerprint"] != record_fingerprint:
            raise ArtifactConflictError(
                f"{description} already exists with different canonical content"
            )

    def upsert_frame(
        self,
        record: FrameRecord,
        *,
        fingerprint: str | None = None,
        index_ready: bool = True,
    ) -> UpsertResult:
        """Idempotently persist one validated canonical frame."""

        return self.upsert_frames((record,), fingerprint=fingerprint, index_ready=index_ready)

    def upsert_frames(
        self,
        records: Iterable[FrameRecord],
        *,
        fingerprint: str | None = None,
        index_ready: bool = True,
    ) -> UpsertResult:
        """Persist frames in one transaction, rejecting any identity conflict."""

        if not isinstance(index_ready, bool):
            raise TypeError("index_ready must be a boolean")
        prepared_records = self._prepare_frames(records, fingerprint)
        inserted = unchanged = readiness_changed = 0
        timestamp = time.time()
        with self._transaction() as connection:
            for batch_id, video_id in {
                (item.record.batch_id, item.record.video_id) for item in prepared_records
            }:
                ordinal_mode = connection.execute(
                    """SELECT 1 FROM canonical_ordinal_frames
                    WHERE batch_id = ? AND video_id = ? LIMIT 1""",
                    (batch_id, video_id),
                ).fetchone()
                if ordinal_mode is not None:
                    raise ArtifactConflictError(
                        f"{batch_id}/{video_id} already uses ordinal frame identity"
                    )
            for prepared in prepared_records:
                record = prepared.record
                existing = connection.execute(
                    """
                    SELECT artifact_fingerprint, record_fingerprint, index_ready
                    FROM canonical_frames
                    WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                    """,
                    (record.batch_id, record.video_id, record.source_frame_idx),
                ).fetchone()
                if existing is not None:
                    self._check_existing(
                        existing,
                        artifact_fingerprint=prepared.artifact_fingerprint,
                        record_fingerprint=prepared.record_fingerprint,
                        description=(
                            f"Frame {record.batch_id}/{record.video_id}/{record.source_frame_idx}"
                        ),
                    )
                    if bool(existing["index_ready"]) != index_ready:
                        connection.execute(
                            """
                            UPDATE canonical_frames SET index_ready = ?, updated_at = ?
                            WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                            """,
                            (
                                int(index_ready),
                                timestamp,
                                record.batch_id,
                                record.video_id,
                                record.source_frame_idx,
                            ),
                        )
                        readiness_changed += 1
                    else:
                        unchanged += 1
                    self._upsert_frame_mapping(
                        connection,
                        record.batch_id,
                        record.video_id,
                        record.keyframe_ordinal,
                        record.source_frame_idx,
                    )
                    continue
                ordinal = connection.execute(
                    """
                    SELECT source_frame_idx FROM canonical_frames
                    WHERE batch_id = ? AND video_id = ? AND keyframe_ordinal = ?
                    """,
                    (record.batch_id, record.video_id, record.keyframe_ordinal),
                ).fetchone()
                if ordinal is not None:
                    raise ArtifactConflictError(
                        f"Keyframe ordinal {record.keyframe_ordinal} for "
                        f"{record.batch_id}/{record.video_id} already maps to source frame "
                        f"{ordinal['source_frame_idx']}"
                    )
                self._insert_frame(
                    connection, prepared, index_ready=index_ready, timestamp=timestamp
                )
                self._upsert_frame_mapping(
                    connection,
                    record.batch_id,
                    record.video_id,
                    record.keyframe_ordinal,
                    record.source_frame_idx,
                )
                inserted += 1
        return UpsertResult(inserted, unchanged, readiness_changed)

    def upsert_ordinal_frames(
        self,
        records: Iterable[FrameRecord],
        *,
        fingerprint: str | None = None,
        index_ready: bool = False,
    ) -> UpsertResult:
        """Persist every organizer keyframe under its ordinal, including repeated source indices.

        This separate identity path keeps legacy source-frame rows and their
        foreign keys unchanged. Processing or content conflicts roll back the
        whole call; matching retries are idempotent.
        """

        if not isinstance(index_ready, bool):
            raise TypeError("index_ready must be a boolean")
        prepared_records = self._prepare_ordinal_frames(records, fingerprint)
        inserted = unchanged = readiness_changed = 0
        timestamp = time.time()
        with self._transaction() as connection:
            for batch_id, video_id in {
                (item.record.batch_id, item.record.video_id) for item in prepared_records
            }:
                source_mode = connection.execute(
                    """SELECT 1 FROM canonical_frames
                    WHERE batch_id = ? AND video_id = ? LIMIT 1""",
                    (batch_id, video_id),
                ).fetchone()
                if source_mode is not None:
                    raise ArtifactConflictError(
                        f"{batch_id}/{video_id} already uses source-frame identity"
                    )
            for prepared in prepared_records:
                record = prepared.record
                identity = (record.batch_id, record.video_id, record.keyframe_ordinal)
                existing = connection.execute(
                    """SELECT artifact_fingerprint, record_fingerprint, index_ready
                    FROM canonical_ordinal_frames
                    WHERE batch_id = ? AND video_id = ? AND keyframe_ordinal = ?""",
                    identity,
                ).fetchone()
                if existing is not None:
                    self._check_existing(
                        existing,
                        artifact_fingerprint=prepared.artifact_fingerprint,
                        record_fingerprint=prepared.record_fingerprint,
                        description=f"Ordinal frame {identity!r}",
                    )
                    if bool(existing["index_ready"]) != index_ready:
                        connection.execute(
                            """UPDATE canonical_ordinal_frames
                            SET index_ready = ?, updated_at = ?
                            WHERE batch_id = ? AND video_id = ? AND keyframe_ordinal = ?""",
                            (int(index_ready), timestamp, *identity),
                        )
                        readiness_changed += 1
                    else:
                        unchanged += 1
                    continue
                connection.execute(
                    """INSERT INTO canonical_ordinal_frames(
                        batch_id, video_id, keyframe_ordinal, source_frame_idx,
                        timestamp_ms, record_json, artifact_fingerprint,
                        record_fingerprint, index_ready, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        *identity,
                        record.source_frame_idx,
                        record.timestamp_ms,
                        _canonical_json(record),
                        prepared.artifact_fingerprint,
                        prepared.record_fingerprint,
                        int(index_ready),
                        timestamp,
                        timestamp,
                    ),
                )
                inserted += 1
        return UpsertResult(inserted, unchanged, readiness_changed)

    def upsert_refined_frame(self, record: RefinedFrameRecord) -> UpsertResult:
        """Persist one exact raw-refined frame without changing canonical counts.

        The retained canonical frame named by ``coarse_source_frame_idx`` is the
        durable identity anchor.  A retained frame with the selected exact
        identity remains authoritative and makes this operation a no-op.
        """

        if not isinstance(record, RefinedFrameRecord):
            raise TypeError("record must be a RefinedFrameRecord")
        record_fingerprint = _content_fingerprint(record)
        timestamp = time.time()
        with self._transaction() as connection:
            canonical = connection.execute(
                """
                SELECT source_sha256 FROM canonical_frames
                WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                """,
                (record.batch_id, record.video_id, record.source_frame_idx),
            ).fetchone()
            if canonical is not None:
                if str(canonical["source_sha256"]).casefold() != record.source_sha256:
                    raise ArtifactConflictError(
                        "refined frame source digest does not match retained source frame"
                    )
                return UpsertResult(unchanged=1)
            anchor = connection.execute(
                """
                SELECT source_sha256 FROM canonical_frames
                WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                """,
                (
                    record.batch_id,
                    record.video_id,
                    record.coarse_source_frame_idx,
                ),
            ).fetchone()
            if anchor is None:
                raise ArtifactConflictError(
                    "refined frame references a missing canonical source-frame anchor"
                )
            if str(anchor["source_sha256"]).casefold() != record.source_sha256:
                raise ArtifactConflictError(
                    "refined frame source digest does not match its canonical anchor"
                )
            existing = connection.execute(
                """
                SELECT record_fingerprint FROM raw_refined_frames
                WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                """,
                (record.batch_id, record.video_id, record.source_frame_idx),
            ).fetchone()
            if existing is not None:
                if str(existing["record_fingerprint"]) != record_fingerprint:
                    raise ArtifactConflictError(
                        "refined source frame already exists with different content"
                    )
                connection.execute(
                    """
                    UPDATE raw_refined_frames SET updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                    """,
                    (
                        timestamp,
                        record.batch_id,
                        record.video_id,
                        record.source_frame_idx,
                    ),
                )
                return UpsertResult(unchanged=1)
            self._insert_refined_frame(
                connection,
                record,
                record_fingerprint=record_fingerprint,
                timestamp=timestamp,
            )
        return UpsertResult(inserted=1)

    def upsert_frame_mappings(
        self,
        batch_id: str,
        video_id: str,
        ordinal_to_source: Mapping[int, int],
    ) -> int:
        """Persist complete organizer ordinal mappings, including duplicate frames."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        if not isinstance(ordinal_to_source, Mapping):
            raise TypeError("ordinal_to_source must be a mapping")
        normalized: dict[int, int] = {}
        for ordinal, source_frame_idx in ordinal_to_source.items():
            _require_nonnegative_int(source_frame_idx, "source_frame_idx")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise TypeError("keyframe_ordinal must be an integer")
            if ordinal < 1:
                raise ValueError("keyframe_ordinal must be positive")
            normalized[ordinal] = source_frame_idx
        with self._transaction() as connection:
            known_sources = {
                int(row["source_frame_idx"])
                for row in connection.execute(
                    """
                    SELECT source_frame_idx FROM canonical_frames
                    WHERE batch_id = ? AND video_id = ?
                    """,
                    (batch_id, video_id),
                )
            }
            missing = sorted(set(normalized.values()) - known_sources)
            if missing:
                raise ArtifactConflictError(
                    f"frame mappings for {batch_id}/{video_id} reference missing "
                    f"source frames; first={missing[0]}"
                )
            for ordinal, source_frame_idx in sorted(normalized.items()):
                self._upsert_frame_mapping(
                    connection, batch_id, video_id, ordinal, source_frame_idx
                )
        return len(normalized)

    def frame_index_mappings(
        self, batch_id: str, *, video_id: str | None = None
    ) -> dict[str, dict[int, int]]:
        """Return persisted ordinal-to-source evidence for submission validation."""

        _require_nonempty(batch_id, "batch_id")
        parameters: list[Any] = [batch_id]
        clause = "batch_id = ?"
        if video_id is not None:
            _require_nonempty(video_id, "video_id")
            clause += " AND video_id = ?"
            parameters.append(video_id)
        with closing(self._connection()) as connection:
            rows = connection.execute(
                f"""
                SELECT video_id, keyframe_ordinal, source_frame_idx
                FROM canonical_frame_mappings WHERE {clause}
                ORDER BY video_id, keyframe_ordinal
                """,
                parameters,
            ).fetchall()
        result: dict[str, dict[int, int]] = {}
        for row in rows:
            result.setdefault(str(row["video_id"]), {})[int(row["keyframe_ordinal"])] = int(
                row["source_frame_idx"]
            )
        return result

    def frame_index_mappings_for_sources(
        self,
        batch_id: str,
        identities: Iterable[tuple[str, int]],
    ) -> dict[str, dict[int, int]]:
        """Return only ordinal evidence relevant to submitted source identities."""

        _require_nonempty(batch_id, "batch_id")
        normalized: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for identity in identities:
            if not isinstance(identity, tuple) or len(identity) != 2:
                raise TypeError("identities must contain (video_id, source_frame_idx) tuples")
            video_id, source_frame_idx = identity
            _require_nonempty(video_id, "video_id")
            _require_nonnegative_int(source_frame_idx, "source_frame_idx")
            value = (video_id, source_frame_idx)
            if value not in seen:
                seen.add(value)
                normalized.append(value)
        if not normalized:
            return {}
        result: dict[str, dict[int, int]] = {}
        with closing(self._connection()) as connection:
            for offset in range(0, len(normalized), 400):
                chunk = normalized[offset : offset + 400]
                values = ",".join("(?, ?)" for _ in chunk)
                parameters: list[Any] = []
                for video_id, source_frame_idx in chunk:
                    parameters.extend((video_id, source_frame_idx))
                parameters.append(batch_id)
                rows = connection.execute(
                    f"""
                    WITH wanted(video_id, source_frame_idx) AS (VALUES {values})
                    SELECT DISTINCT mapping.video_id, mapping.keyframe_ordinal,
                                    mapping.source_frame_idx
                    FROM wanted
                    JOIN canonical_frame_mappings AS mapping
                      ON mapping.batch_id = ?
                     AND mapping.video_id = wanted.video_id
                     AND (
                         mapping.source_frame_idx = wanted.source_frame_idx
                         OR mapping.keyframe_ordinal = wanted.source_frame_idx
                     )
                    ORDER BY mapping.video_id, mapping.keyframe_ordinal
                    """,
                    parameters,
                ).fetchall()
                for row in rows:
                    result.setdefault(str(row["video_id"]), {})[
                        int(row["keyframe_ordinal"])
                    ] = int(row["source_frame_idx"])
        return result

    def upsert_text_segment(
        self,
        segment: TextSegment,
        *,
        fingerprint: str | None = None,
        index_ready: bool = True,
    ) -> UpsertResult:
        """Idempotently persist one validated timestamped text segment."""

        return self.upsert_text_segments(
            (segment,), fingerprint=fingerprint, index_ready=index_ready
        )

    def upsert_text_segments(
        self,
        segments: Iterable[TextSegment],
        *,
        fingerprint: str | None = None,
        index_ready: bool = True,
    ) -> UpsertResult:
        """Persist text segments in one transaction, rejecting identity conflicts."""

        if not isinstance(index_ready, bool):
            raise TypeError("index_ready must be a boolean")
        prepared_segments = self._prepare_text_segments(segments, fingerprint)
        inserted = unchanged = readiness_changed = 0
        timestamp = time.time()
        with self._transaction() as connection:
            for prepared in prepared_segments:
                segment = prepared.segment
                existing = connection.execute(
                    """
                    SELECT artifact_fingerprint, record_fingerprint, index_ready
                    FROM canonical_text_segments
                    WHERE batch_id = ? AND video_id = ? AND segment_id = ?
                    """,
                    (segment.batch_id, segment.video_id, segment.segment_id),
                ).fetchone()
                if existing is not None:
                    self._check_existing(
                        existing,
                        artifact_fingerprint=prepared.artifact_fingerprint,
                        record_fingerprint=prepared.record_fingerprint,
                        description=(
                            f"Text segment {segment.batch_id}/{segment.video_id}/"
                            f"{segment.segment_id}"
                        ),
                    )
                    if bool(existing["index_ready"]) != index_ready:
                        connection.execute(
                            """
                            UPDATE canonical_text_segments SET index_ready = ?, updated_at = ?
                            WHERE batch_id = ? AND video_id = ? AND segment_id = ?
                            """,
                            (
                                int(index_ready),
                                timestamp,
                                segment.batch_id,
                                segment.video_id,
                                segment.segment_id,
                            ),
                        )
                        readiness_changed += 1
                    else:
                        unchanged += 1
                    continue
                self._insert_text_segment(
                    connection, prepared, index_ready=index_ready, timestamp=timestamp
                )
                inserted += 1
        return UpsertResult(inserted, unchanged, readiness_changed)

    def replace_text_segments_for_video(
        self,
        batch_id: str,
        video_id: str,
        kind: str,
        segments: Iterable[TextSegment],
        *,
        fingerprint: str,
        index_ready: bool = True,
    ) -> UpsertResult:
        """Atomically replace one generated text channel for one canonical video.

        This deliberately cannot modify frames or another text kind. It is the
        scoped recovery boundary used when an OCR/ASR model fingerprint changes.
        """

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonempty(kind, "kind")
        _require_nonempty(fingerprint, "fingerprint")
        if not isinstance(index_ready, bool):
            raise TypeError("index_ready must be a boolean")
        values = tuple(segments)
        for segment in values:
            if (
                segment.batch_id != batch_id
                or segment.video_id != video_id
                or segment.kind != kind
            ):
                raise ValueError("replacement segments must match batch, video, and kind")
        prepared = self._prepare_text_segments(values, fingerprint)
        timestamp = time.time()
        with self._transaction() as connection:
            previous = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM canonical_text_segments
                    WHERE batch_id = ? AND video_id = ? AND kind = ?
                    """,
                    (batch_id, video_id, kind),
                ).fetchone()[0]
            )
            connection.execute(
                """
                DELETE FROM canonical_text_segments
                WHERE batch_id = ? AND video_id = ? AND kind = ?
                """,
                (batch_id, video_id, kind),
            )
            for value in prepared:
                self._insert_text_segment(
                    connection,
                    value,
                    index_ready=index_ready,
                    timestamp=timestamp,
                )
        return UpsertResult(len(prepared), 0, previous)

    def replace_video(
        self,
        batch_id: str,
        video_id: str,
        *,
        frames: Iterable[FrameRecord],
        text_segments: Iterable[TextSegment] = (),
        ordinal_to_source: Mapping[int, int] | None = None,
        frame_fingerprint: str | None = None,
        text_fingerprint: str | None = None,
        index_ready: bool = True,
    ) -> VideoReplacement:
        """Atomically replace one video, requiring explicit organizer aliases."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        if not isinstance(index_ready, bool):
            raise TypeError("index_ready must be a boolean")
        prepared_frames = self._prepare_frames(frames, frame_fingerprint)
        prepared_segments = self._prepare_text_segments(text_segments, text_fingerprint)
        supplied_mappings = None if ordinal_to_source is None else dict(ordinal_to_source)
        for prepared in prepared_frames:
            record = prepared.record
            if (record.batch_id, record.video_id) != (batch_id, video_id):
                raise ValueError("every replacement frame must belong to the requested video")
        for prepared in prepared_segments:
            segment = prepared.segment
            if (segment.batch_id, segment.video_id) != (batch_id, video_id):
                raise ValueError(
                    "every replacement text segment must belong to the requested video"
                )

        timestamp = time.time()
        with self._transaction() as connection:
            if connection.execute(
                """SELECT 1 FROM canonical_ordinal_frames
                WHERE batch_id = ? AND video_id = ? LIMIT 1""",
                (batch_id, video_id),
            ).fetchone() is not None:
                raise ArtifactConflictError(
                    f"{batch_id}/{video_id} uses ordinal frame identity; "
                    "source-frame replacement is unavailable"
                )
            existing_frame_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM canonical_frames
                    WHERE batch_id = ? AND video_id = ?
                    """,
                    (batch_id, video_id),
                ).fetchone()[0]
            )
            existing_mapping_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM canonical_frame_mappings
                    WHERE batch_id = ? AND video_id = ?
                    """,
                    (batch_id, video_id),
                ).fetchone()[0]
            )
            if supplied_mappings is None and existing_mapping_count > existing_frame_count:
                raise ArtifactConflictError(
                    "reference video has additional organizer ordinal aliases; "
                    "pass ordinal_to_source explicitly when replacing it"
                )
            connection.execute(
                "DELETE FROM canonical_text_segments WHERE batch_id = ? AND video_id = ?",
                (batch_id, video_id),
            )
            connection.execute(
                "DELETE FROM canonical_frames WHERE batch_id = ? AND video_id = ?",
                (batch_id, video_id),
            )
            for prepared in prepared_frames:
                self._insert_frame(
                    connection, prepared, index_ready=index_ready, timestamp=timestamp
                )
                record = prepared.record
                self._upsert_frame_mapping(
                    connection,
                    record.batch_id,
                    record.video_id,
                    record.keyframe_ordinal,
                    record.source_frame_idx,
                )
            for prepared in prepared_segments:
                self._insert_text_segment(
                    connection, prepared, index_ready=index_ready, timestamp=timestamp
                )
            if supplied_mappings is not None:
                known_sources = {prepared.record.source_frame_idx for prepared in prepared_frames}
                for ordinal, source_frame_idx in supplied_mappings.items():
                    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
                        raise ValueError("keyframe_ordinal must be a positive integer")
                    _require_nonnegative_int(source_frame_idx, "source_frame_idx")
                    if source_frame_idx not in known_sources:
                        raise ArtifactConflictError(
                            f"frame mapping references missing source frame {source_frame_idx}"
                        )
                    self._upsert_frame_mapping(
                        connection, batch_id, video_id, ordinal, source_frame_idx
                    )
        return VideoReplacement(len(prepared_frames), len(prepared_segments))

    @staticmethod
    def _frame_from_row(row: sqlite3.Row) -> FrameRecord:
        try:
            reasons = json.loads(row["sampling_reasons_json"])
            return FrameRecord(
                batch_id=row["batch_id"],
                video_id=row["video_id"],
                keyframe_ordinal=row["keyframe_ordinal"],
                source_frame_idx=row["source_frame_idx"],
                pts=row["pts"],
                time_base=row["time_base"],
                timestamp_ms=row["timestamp_ms"],
                fps_num=row["fps_num"],
                fps_den=row["fps_den"],
                shot_id=row["shot_id"],
                sampling_reasons=tuple(reasons),
                image_uri=row["image_uri"],
                video_uri=row["video_uri"],
                source_sha256=row["source_sha256"],
                extractor_version=row["extractor_version"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepositoryCorruptionError("Invalid canonical frame row") from exc

    @staticmethod
    def _ordinal_frame_from_row(row: sqlite3.Row) -> FrameRecord:
        try:
            record = FrameRecord.from_dict(json.loads(row["record_json"]))
            if (
                record.batch_id != row["batch_id"]
                or record.video_id != row["video_id"]
                or record.keyframe_ordinal != row["keyframe_ordinal"]
                or record.source_frame_idx != row["source_frame_idx"]
                or record.timestamp_ms != row["timestamp_ms"]
                or _content_fingerprint(record) != row["record_fingerprint"]
            ):
                raise ValueError("ordinal frame columns and record disagree")
            return record
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepositoryCorruptionError("Invalid canonical ordinal frame row") from exc

    def get_ordinal_frame(
        self, batch_id: str, video_id: str, keyframe_ordinal: int
    ) -> FrameRecord | None:
        """Return one exact keyframe; source-frame and PTS collisions are allowed."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        if (
            isinstance(keyframe_ordinal, bool)
            or not isinstance(keyframe_ordinal, int)
            or keyframe_ordinal < 1
        ):
            raise ValueError("keyframe_ordinal must be a positive integer")
        with closing(self._connection()) as connection:
            row = connection.execute(
                """SELECT * FROM canonical_ordinal_frames
                WHERE batch_id = ? AND video_id = ? AND keyframe_ordinal = ?""",
                (batch_id, video_id, keyframe_ordinal),
            ).fetchone()
        return None if row is None else self._ordinal_frame_from_row(row)

    def get_trake_frame(
        self, batch_id: str, video_id: str, source_frame_idx: int, *, timestamp_ms: int
    ) -> FrameRecord | None:
        """Resolve TRAKE evidence by exact source/PTS; never guess an ordinal."""
        from src.retrieval.trake_scope import trake_video_allowed

        if not trake_video_allowed(batch_id, video_id):
            return None
        if batch_id != "aic26-b2":
            frame = self.get_source_frame(batch_id, video_id, source_frame_idx)
            return frame if frame is not None and frame.timestamp_ms == timestamp_ms else None
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0
               for x in (source_frame_idx, timestamp_ms)):
            raise ValueError("invalid TRAKE coordinates")
        with closing(self._connection()) as connection:
            rows = connection.execute(
                """SELECT * FROM canonical_ordinal_frames
                WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                    AND timestamp_ms = ? LIMIT 2""",
                (batch_id, video_id, source_frame_idx, timestamp_ms),
            ).fetchall()
        return self._ordinal_frame_from_row(rows[0]) if len(rows) == 1 else None

    def iter_ordinal_frames(
        self,
        *,
        batch_id: str | None = None,
        video_id: str | None = None,
        index_ready_only: bool = False,
    ) -> Iterator[FrameRecord]:
        """Iterate all ordinal frames in organizer order, retaining equal PTS rows."""

        if video_id is not None and batch_id is None:
            raise ValueError("video_id requires batch_id")
        if batch_id is not None:
            _require_nonempty(batch_id, "batch_id")
        if video_id is not None:
            _require_nonempty(video_id, "video_id")
        if not isinstance(index_ready_only, bool):
            raise TypeError("index_ready_only must be a boolean")
        clauses: list[str] = []
        parameters: list[Any] = []
        if batch_id is not None:
            clauses.append("batch_id = ?")
            parameters.append(batch_id)
        if video_id is not None:
            clauses.append("video_id = ?")
            parameters.append(video_id)
        if index_ready_only:
            clauses.append("index_ready = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""SELECT * FROM canonical_ordinal_frames {where}
            ORDER BY batch_id, video_id, keyframe_ordinal"""

        def generate() -> Iterator[FrameRecord]:
            with closing(self._connection()) as connection:
                for row in connection.execute(query, parameters):
                    yield self._ordinal_frame_from_row(row)

        return generate()

    def count_ordinal_frames(self, batch_id: str, *, video_id: str | None = None) -> int:
        """Count distinct organizer keyframes without collapsing source indices or PTS."""

        _require_nonempty(batch_id, "batch_id")
        parameters: list[Any] = [batch_id]
        query = "SELECT COUNT(*) FROM canonical_ordinal_frames WHERE batch_id = ?"
        if video_id is not None:
            _require_nonempty(video_id, "video_id")
            query += " AND video_id = ?"
            parameters.append(video_id)
        with closing(self._connection()) as connection:
            return int(connection.execute(query, parameters).fetchone()[0])

    def nearest_ordinal_frame(
        self, batch_id: str, video_id: str, timestamp_ms: int
    ) -> FrameRecord | None:
        """Resolve a time to one keyframe, breaking equal-PTS ties by ordinal."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonnegative_int(timestamp_ms, "timestamp_ms")
        with closing(self._connection()) as connection:
            row = connection.execute(
                """SELECT * FROM canonical_ordinal_frames
                WHERE batch_id = ? AND video_id = ?
                ORDER BY ABS(timestamp_ms - ?), timestamp_ms, keyframe_ordinal
                LIMIT 1""",
                (batch_id, video_id, timestamp_ms),
            ).fetchone()
        return None if row is None else self._ordinal_frame_from_row(row)

    @staticmethod
    def _refined_frame_from_row(row: sqlite3.Row) -> RefinedFrameRecord:
        try:
            return RefinedFrameRecord(
                batch_id=row["batch_id"],
                video_id=row["video_id"],
                source_frame_idx=row["source_frame_idx"],
                coarse_source_frame_idx=row["coarse_source_frame_idx"],
                pts=row["pts"],
                time_base=row["time_base"],
                timestamp_ms=row["timestamp_ms"],
                fps_num=row["fps_num"],
                fps_den=row["fps_den"],
                shot_id=row["shot_id"],
                image_uri=row["image_uri"],
                video_uri=row["video_uri"],
                source_sha256=row["source_sha256"],
                image_sha256=row["image_sha256"],
                image_width=row["image_width"],
                image_height=row["image_height"],
                image_bytes=row["image_bytes"],
                refiner_fingerprint=row["refiner_fingerprint"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RepositoryCorruptionError("Invalid raw-refined frame row") from exc

    @staticmethod
    def _text_segment_from_row(row: sqlite3.Row) -> TextSegment:
        try:
            return TextSegment(
                segment_id=row["segment_id"],
                batch_id=row["batch_id"],
                video_id=row["video_id"],
                kind=row["kind"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                source_frame_idx=row["source_frame_idx"],
                text_raw=row["text_raw"],
                text_nfc=row["text_nfc"],
                text_accent_folded=row["text_accent_folded"],
                confidence=row["confidence"],
                model_version=row["model_version"],
                polygons=tuple(
                    tuple(tuple(point) for point in polygon)
                    for polygon in json.loads(row["polygons_json"])
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepositoryCorruptionError("Invalid canonical text-segment row") from exc

    def get_frame(self, batch_id: str, video_id: str, source_frame_idx: int) -> FrameRecord | None:
        """Return a frame by its canonical competition identity."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonnegative_int(source_frame_idx, "source_frame_idx")
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT * FROM canonical_frames
                WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                """,
                (batch_id, video_id, source_frame_idx),
            ).fetchone()
        return None if row is None else self._frame_from_row(row)

    def get_frames(
        self,
        batch_id: str,
        identities: Sequence[tuple[str, int]],
    ) -> dict[tuple[str, int], FrameRecord]:
        """Resolve a bounded candidate head in one read-only SQLite query."""

        _require_nonempty(batch_id, "batch_id")
        unique: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for video_id, source_frame_idx in identities:
            _require_nonempty(video_id, "video_id")
            _require_nonnegative_int(source_frame_idx, "source_frame_idx")
            key = (video_id, source_frame_idx)
            if key not in seen:
                seen.add(key)
                unique.append(key)
        if not unique:
            return {}
        if len(unique) > 100:
            raise ValueError("get_frames accepts at most 100 canonical identities")
        values = ",".join("(?, ?)" for _value in unique)
        parameters: list[Any] = []
        for video_id, source_frame_idx in unique:
            parameters.extend((video_id, source_frame_idx))
        parameters.append(batch_id)
        with closing(self._connection()) as connection:
            rows = connection.execute(
                f"""
                WITH requested(video_id, source_frame_idx) AS (VALUES {values})
                SELECT frame.* FROM canonical_frames AS frame
                JOIN requested
                  ON requested.video_id = frame.video_id
                 AND requested.source_frame_idx = frame.source_frame_idx
                WHERE frame.batch_id = ?
                """,
                tuple(parameters),
            ).fetchall()
        frames = [self._frame_from_row(row) for row in rows]
        return {(frame.video_id, frame.source_frame_idx): frame for frame in frames}

    def adjacent_frames(
        self,
        batch_id: str,
        video_id: str,
        source_frame_idx: int,
        *,
        radius: int = 5,
    ) -> list[tuple[int, FrameRecord]]:
        """Return canonical ordinal neighbors with stable signed offsets.

        Adjacent source frames are resolved through the ordinal mapping table;
        source-frame arithmetic is deliberately never used. Duplicate ordinal
        aliases are represented by the offset nearest to the anchor.
        """

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonnegative_int(source_frame_idx, "source_frame_idx")
        if isinstance(radius, bool) or not isinstance(radius, int) or not 1 <= radius <= 5:
            raise ValueError("radius must be between 1 and 5")
        anchor = self.get_frame(batch_id, video_id, source_frame_idx)
        if anchor is None:
            return []
        with closing(self._connection()) as connection:
            rows = connection.execute(
                """
                SELECT mapping.keyframe_ordinal AS adjacent_ordinal, frame.*
                FROM canonical_frame_mappings AS mapping
                JOIN canonical_frames AS frame
                  ON frame.batch_id = mapping.batch_id
                 AND frame.video_id = mapping.video_id
                 AND frame.source_frame_idx = mapping.source_frame_idx
                WHERE mapping.batch_id = ?
                  AND mapping.video_id = ?
                  AND mapping.keyframe_ordinal BETWEEN ? AND ?
                ORDER BY mapping.keyframe_ordinal, mapping.source_frame_idx
                """,
                (
                    batch_id,
                    video_id,
                    max(1, anchor.keyframe_ordinal - radius),
                    anchor.keyframe_ordinal + radius,
                ),
            ).fetchall()
        selected: dict[int, tuple[int, FrameRecord]] = {}
        for row in rows:
            offset = int(row["adjacent_ordinal"]) - anchor.keyframe_ordinal
            frame = self._frame_from_row(row)
            current = selected.get(frame.source_frame_idx)
            if current is None or (abs(offset), offset) < (abs(current[0]), current[0]):
                selected[frame.source_frame_idx] = (offset, frame)
        return sorted(
            selected.values(),
            key=lambda value: (
                value[0],
                value[1].source_frame_idx,
            ),
        )


    def get_refined_frame(
        self, batch_id: str, video_id: str, source_frame_idx: int
    ) -> RefinedFrameRecord | None:
        """Return a durable raw-refined frame without querying retained frames."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonnegative_int(source_frame_idx, "source_frame_idx")
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT * FROM raw_refined_frames
                WHERE batch_id = ? AND video_id = ? AND source_frame_idx = ?
                """,
                (batch_id, video_id, source_frame_idx),
            ).fetchone()
        return None if row is None else self._refined_frame_from_row(row)

    def get_source_frame(
        self, batch_id: str, video_id: str, source_frame_idx: int
    ) -> FrameRecord | RefinedFrameRecord | None:
        """Resolve one competition source frame, preferring retained metadata."""

        canonical = self.get_frame(batch_id, video_id, source_frame_idx)
        if canonical is not None:
            return canonical
        return self.get_refined_frame(batch_id, video_id, source_frame_idx)

    def existing_source_frames(
        self,
        batch_id: str,
        identities: Iterable[tuple[str, int]],
        *,
        include_refined: bool = True,
    ) -> frozenset[tuple[str, int]]:
        """Return candidate-scoped persisted source identities in bounded queries."""

        _require_nonempty(batch_id, "batch_id")
        if not isinstance(include_refined, bool):
            raise TypeError("include_refined must be a boolean")
        normalized: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for identity in identities:
            if not isinstance(identity, tuple) or len(identity) != 2:
                raise TypeError("identities must contain (video_id, source_frame_idx) tuples")
            video_id, source_frame_idx = identity
            _require_nonempty(video_id, "video_id")
            _require_nonnegative_int(source_frame_idx, "source_frame_idx")
            value = (video_id, source_frame_idx)
            if value not in seen:
                seen.add(value)
                normalized.append(value)
        if not normalized:
            return frozenset()
        found: set[tuple[str, int]] = set()
        table_names = ["canonical_frames"]
        if include_refined:
            table_names.append("raw_refined_frames")
        with closing(self._connection()) as connection:
            for offset in range(0, len(normalized), 400):
                chunk = normalized[offset : offset + 400]
                values = ",".join("(?, ?)" for _ in chunk)
                parameters: list[Any] = []
                for video_id, source_frame_idx in chunk:
                    parameters.extend((video_id, source_frame_idx))
                selects = [
                    (
                        "SELECT wanted.video_id, wanted.source_frame_idx "
                        f"FROM wanted JOIN {table} AS stored "
                        "ON stored.batch_id = ? "
                        "AND stored.video_id = wanted.video_id "
                        "AND stored.source_frame_idx = wanted.source_frame_idx"
                    )
                    for table in table_names
                ]
                query = (
                    f"WITH wanted(video_id, source_frame_idx) AS (VALUES {values}) "
                    + " UNION ".join(selects)
                )
                query_parameters = parameters + [batch_id] * len(table_names)
                for row in connection.execute(query, query_parameters):
                    found.add((str(row["video_id"]), int(row["source_frame_idx"])))
        return frozenset(found)

    def get_text_segment(self, batch_id: str, video_id: str, segment_id: str) -> TextSegment | None:
        """Return a text segment by its video-scoped canonical identity."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonempty(segment_id, "segment_id")
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT * FROM canonical_text_segments
                WHERE batch_id = ? AND video_id = ? AND segment_id = ?
                """,
                (batch_id, video_id, segment_id),
            ).fetchone()
        return None if row is None else self._text_segment_from_row(row)

    def neighbors(
        self,
        batch_id: str,
        video_id: str,
        timestamp_ms: int,
        radius_ms: int = 4_000,
        limit: int | None = None,
    ) -> list[FrameRecord]:
        """Return same-video frames around a timestamp, ordered chronologically."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonnegative_int(timestamp_ms, "timestamp_ms")
        _require_nonnegative_int(radius_ms, "radius_ms")
        limit = _require_positive_limit(limit)
        lower = max(0, timestamp_ms - radius_ms)
        upper = timestamp_ms + radius_ms
        parameters: list[Any] = [batch_id, video_id, lower, upper]
        if limit is None:
            query = """
                SELECT * FROM canonical_frames
                WHERE batch_id = ? AND video_id = ?
                    AND timestamp_ms BETWEEN ? AND ?
                ORDER BY timestamp_ms, source_frame_idx
            """
        else:
            query = """
                SELECT * FROM (
                    SELECT * FROM canonical_frames
                    WHERE batch_id = ? AND video_id = ?
                        AND timestamp_ms BETWEEN ? AND ?
                    ORDER BY ABS(timestamp_ms - ?), timestamp_ms, source_frame_idx
                    LIMIT ?
                ) ORDER BY timestamp_ms, source_frame_idx
            """
            parameters.extend((timestamp_ms, limit))
        with closing(self._connection()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._frame_from_row(row) for row in rows]

    def nearest_frame(self, batch_id: str, video_id: str, timestamp_ms: int) -> FrameRecord | None:
        """Return the nearest frame in one video, preferring the earlier frame on ties."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        _require_nonnegative_int(timestamp_ms, "timestamp_ms")
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT * FROM canonical_frames
                WHERE batch_id = ? AND video_id = ?
                ORDER BY ABS(timestamp_ms - ?), timestamp_ms, source_frame_idx
                LIMIT 1
                """,
                (batch_id, video_id, timestamp_ms),
            ).fetchone()
        return None if row is None else self._frame_from_row(row)

    def has_refinable_frames(self, batch_id: str) -> bool:
        """Return whether a batch has at least one exact raw-video anchor."""

        _require_nonempty(batch_id, "batch_id")
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM canonical_frames
                WHERE batch_id = ?
                    AND video_uri <> ''
                    AND pts IS NOT NULL
                    AND time_base IS NOT NULL
                    AND sampling_reasons_json NOT LIKE '%"estimated_pts"%'
                LIMIT 1
                """,
                (batch_id,),
            ).fetchone()
        return row is not None

    def nearest_frame_for_segment(
        self,
        batch_id: str,
        video_id: str,
        segment_id: str,
        *,
        require_asr: bool = True,
    ) -> FrameRecord | None:
        """Resolve a text segment's midpoint to the nearest same-video frame."""

        if not isinstance(require_asr, bool):
            raise TypeError("require_asr must be a boolean")
        segment = self.get_text_segment(batch_id, video_id, segment_id)
        if segment is None:
            return None
        if require_asr and segment.kind != "asr":
            raise ValueError(f"Text segment {segment_id!r} is not an ASR segment")
        midpoint_ms = segment.start_ms + (segment.end_ms - segment.start_ms) // 2
        return self.nearest_frame(batch_id, video_id, midpoint_ms)

    def list_frames(
        self, batch_id: str, video_id: str, *, index_ready_only: bool = False
    ) -> list[FrameRecord]:
        """List one video's canonical frames in timestamp order."""

        return list(
            self.iter_frames(
                batch_id=batch_id,
                video_id=video_id,
                index_ready_only=index_ready_only,
            )
        )

    def iter_frames(
        self,
        *,
        batch_id: str | None = None,
        video_id: str | None = None,
        index_ready_only: bool = False,
    ) -> Iterator[FrameRecord]:
        """Iterate canonical frames in stable batch/video/time order."""

        if video_id is not None and batch_id is None:
            raise ValueError("video_id requires batch_id")
        if batch_id is not None:
            _require_nonempty(batch_id, "batch_id")
        if video_id is not None:
            _require_nonempty(video_id, "video_id")
        if not isinstance(index_ready_only, bool):
            raise TypeError("index_ready_only must be a boolean")
        clauses: list[str] = []
        parameters: list[Any] = []
        if batch_id is not None:
            clauses.append("batch_id = ?")
            parameters.append(batch_id)
        if video_id is not None:
            clauses.append("video_id = ?")
            parameters.append(video_id)
        if index_ready_only:
            clauses.append("index_ready = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM canonical_frames {where}
            ORDER BY batch_id, video_id, timestamp_ms, source_frame_idx
        """

        def generate() -> Iterator[FrameRecord]:
            with closing(self._connection()) as connection:
                cursor = connection.execute(query, parameters)
                for row in cursor:
                    yield self._frame_from_row(row)

        return generate()

    def iter_index_ready_frames(
        self, *, batch_id: str | None = None, video_id: str | None = None
    ) -> Iterator[FrameRecord]:
        """Iterate only frames marked ready for downstream vector indexing."""

        return self.iter_frames(batch_id=batch_id, video_id=video_id, index_ready_only=True)

    def count_frames(self, batch_id: str, *, video_id: str | None = None) -> int:
        """Return the exact canonical frame count for a batch or one video."""

        _require_nonempty(batch_id, "batch_id")
        if video_id is not None:
            _require_nonempty(video_id, "video_id")
        query = "SELECT COUNT(*) FROM canonical_frames WHERE batch_id = ?"
        parameters: list[Any] = [batch_id]
        if video_id is not None:
            query += " AND video_id = ?"
            parameters.append(video_id)
        with closing(self._connection()) as connection:
            return int(connection.execute(query, parameters).fetchone()[0])

    def video_frame_counts(self, batch_id: str) -> dict[str, int]:
        """Return stable per-video canonical frame counts."""

        _require_nonempty(batch_id, "batch_id")
        with closing(self._connection()) as connection:
            rows = connection.execute(
                """
                SELECT video_id, COUNT(*) AS frame_count
                FROM canonical_frames
                WHERE batch_id = ?
                GROUP BY video_id
                ORDER BY video_id
                """,
                (batch_id,),
            ).fetchall()
        return {str(row["video_id"]): int(row["frame_count"]) for row in rows}

    def count_text_segments(
        self,
        batch_id: str,
        *,
        kind: str | None = None,
        video_id: str | None = None,
    ) -> int:
        """Return an exact artifact-database text-segment count."""

        _require_nonempty(batch_id, "batch_id")
        clauses = ["batch_id = ?"]
        parameters: list[Any] = [batch_id]
        for column, value in (("kind", kind), ("video_id", video_id)):
            if value is not None:
                _require_nonempty(value, column)
                clauses.append(f"{column} = ?")
                parameters.append(value)
        query = "SELECT COUNT(*) FROM canonical_text_segments WHERE " + " AND ".join(clauses)
        with closing(self._connection()) as connection:
            return int(connection.execute(query, parameters).fetchone()[0])

    def iter_text_segments(
        self,
        *,
        batch_id: str | None = None,
        video_id: str | None = None,
        kind: str | None = None,
        index_ready_only: bool = False,
    ) -> Iterator[TextSegment]:
        """Iterate text segments in stable batch/video/time order."""

        if video_id is not None and batch_id is None:
            raise ValueError("video_id requires batch_id")
        for name, value in (("batch_id", batch_id), ("video_id", video_id), ("kind", kind)):
            if value is not None:
                _require_nonempty(value, name)
        if not isinstance(index_ready_only, bool):
            raise TypeError("index_ready_only must be a boolean")
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (("batch_id", batch_id), ("video_id", video_id), ("kind", kind)):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if index_ready_only:
            clauses.append("index_ready = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM canonical_text_segments {where}
            ORDER BY batch_id, video_id, start_ms, end_ms, segment_id
        """

        def generate() -> Iterator[TextSegment]:
            with closing(self._connection()) as connection:
                cursor = connection.execute(query, parameters)
                for row in cursor:
                    yield self._text_segment_from_row(row)

        return generate()

    def iter_index_ready_text_segments(
        self,
        *,
        batch_id: str | None = None,
        video_id: str | None = None,
        kind: str | None = None,
    ) -> Iterator[TextSegment]:
        """Iterate only text segments marked ready for downstream indexing."""

        return self.iter_text_segments(
            batch_id=batch_id,
            video_id=video_id,
            kind=kind,
            index_ready_only=True,
        )

    def text_segments_in_range(
        self,
        batch_id: str,
        video_id: str,
        *,
        start_ms: int,
        end_ms: int,
        kind: str = "asr",
        index_ready_only: bool = True,
        limit: int = 200,
    ) -> tuple[TextSegment, ...]:
        """Return timestamped text overlapping one bounded video interval."""

        for name, value in (("batch_id", batch_id), ("video_id", video_id), ("kind", kind)):
            _require_nonempty(value, name)
        if isinstance(start_ms, bool) or not isinstance(start_ms, int) or start_ms < 0:
            raise ValueError("start_ms must be a non-negative integer")
        if isinstance(end_ms, bool) or not isinstance(end_ms, int) or end_ms < start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses = [
            "batch_id = ?",
            "video_id = ?",
            "kind = ?",
            "end_ms >= ?",
            "start_ms <= ?",
        ]
        parameters: list[Any] = [batch_id, video_id, kind, start_ms, end_ms]
        if index_ready_only:
            clauses.append("index_ready = 1")
        query = f"""
            SELECT * FROM canonical_text_segments
            WHERE {' AND '.join(clauses)}
            ORDER BY start_ms, end_ms, segment_id
            LIMIT ?
        """
        parameters.append(limit)
        with closing(self._connection()) as connection:
            return tuple(
                self._text_segment_from_row(row)
                for row in connection.execute(query, parameters)
            )

    def set_video_index_ready(
        self,
        batch_id: str,
        video_id: str,
        ready: bool,
        *,
        include_frames: bool = True,
        include_text_segments: bool = True,
    ) -> int:
        """Atomically change downstream-index readiness for one video's records."""

        _require_nonempty(batch_id, "batch_id")
        _require_nonempty(video_id, "video_id")
        for name, value in (
            ("ready", ready),
            ("include_frames", include_frames),
            ("include_text_segments", include_text_segments),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if not include_frames and not include_text_segments:
            return 0
        changed = 0
        timestamp = time.time()
        with self._transaction() as connection:
            if include_frames:
                cursor = connection.execute(
                    """
                    UPDATE canonical_frames SET index_ready = ?, updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND index_ready != ?
                    """,
                    (int(ready), timestamp, batch_id, video_id, int(ready)),
                )
                changed += cursor.rowcount
                cursor = connection.execute(
                    """
                    UPDATE canonical_ordinal_frames SET index_ready = ?, updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND index_ready != ?
                    """,
                    (int(ready), timestamp, batch_id, video_id, int(ready)),
                )
                changed += cursor.rowcount
            if include_text_segments:
                cursor = connection.execute(
                    """
                    UPDATE canonical_text_segments SET index_ready = ?, updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND index_ready != ?
                    """,
                    (int(ready), timestamp, batch_id, video_id, int(ready)),
                )
                changed += cursor.rowcount
        return changed

    def batch_counts(self, batch_id: str) -> BatchArtifactCounts:
        """Return consistent canonical and index-ready counts for one batch."""

        _require_nonempty(batch_id, "batch_id")
        with closing(self._connection()) as connection:
            connection.execute("BEGIN")
            summary = connection.execute(
                """
                SELECT
                    ((SELECT COUNT(*) FROM canonical_frames WHERE batch_id = ?)
                     + (SELECT COUNT(*) FROM canonical_ordinal_frames WHERE batch_id = ?)) AS frames,
                    (SELECT COUNT(*) FROM canonical_text_segments WHERE batch_id = ?) AS texts,
                    ((SELECT COUNT(*) FROM canonical_frames
                        WHERE batch_id = ? AND index_ready = 1)
                     + (SELECT COUNT(*) FROM canonical_ordinal_frames
                        WHERE batch_id = ? AND index_ready = 1)) AS ready_frames,
                    (SELECT COUNT(*) FROM canonical_text_segments
                        WHERE batch_id = ? AND index_ready = 1) AS ready_texts,
                    (SELECT COUNT(*) FROM (
                        SELECT video_id FROM canonical_frames WHERE batch_id = ?
                        UNION
                        SELECT video_id FROM canonical_ordinal_frames WHERE batch_id = ?
                        UNION
                        SELECT video_id FROM canonical_text_segments WHERE batch_id = ?
                    )) AS videos
                """,
                (batch_id,) * 9,
            ).fetchone()
            kinds = connection.execute(
                """
                SELECT kind, COUNT(*) AS count FROM canonical_text_segments
                WHERE batch_id = ? GROUP BY kind ORDER BY kind
                """,
                (batch_id,),
            ).fetchall()
            connection.commit()
        return BatchArtifactCounts(
            batch_id=batch_id,
            video_count=int(summary["videos"]),
            frame_count=int(summary["frames"]),
            text_segment_count=int(summary["texts"]),
            index_ready_frame_count=int(summary["ready_frames"]),
            index_ready_text_segment_count=int(summary["ready_texts"]),
            text_segments_by_kind={str(row["kind"]): int(row["count"]) for row in kinds},
        )

    def refined_frame_count(
        self, batch_id: str | None = None, *, video_id: str | None = None
    ) -> int:
        """Count raw-refined frames separately from ingestion/readiness artifacts."""

        if video_id is not None and batch_id is None:
            raise ValueError("video_id requires batch_id")
        clauses: list[str] = []
        parameters: list[str] = []
        if batch_id is not None:
            clauses.append("batch_id = ?")
            parameters.append(_require_nonempty(batch_id, "batch_id"))
        if video_id is not None:
            clauses.append("video_id = ?")
            parameters.append(_require_nonempty(video_id, "video_id"))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connection()) as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM raw_refined_frames {where}", parameters
            ).fetchone()
        return int(row[0])

    def refined_frame_usage(self) -> tuple[int, int]:
        """Return total persisted refined-frame count and encoded image bytes."""

        with closing(self._connection()) as connection:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(image_bytes), 0) FROM raw_refined_frames"
            ).fetchone()
        return int(row[0]), int(row[1])


__all__ = [
    "ArtifactConflictError",
    "ArtifactRepository",
    "ArtifactRepositoryError",
    "BatchArtifactCounts",
    "RepositoryCorruptionError",
    "RepositoryNotInitialized",
    "UpsertResult",
    "VideoReplacement",
]
