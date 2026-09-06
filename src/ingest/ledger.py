"""SQLite-backed, resumable ingestion job ledger."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator

from src.ingest.archive_catalog import ArchiveInspection


class LedgerError(RuntimeError):
    """Base error for invalid ledger operations."""


class StageFingerprintMismatch(LedgerError):
    """Raised when completed work is reused with incompatible inputs."""


class StageState(StrEnum):
    """Persisted states for one per-video processing stage."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """An immutable artifact registered as part of stage completion."""

    kind: str
    uri: str
    fingerprint: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class StageTransition:
    """Result of an idempotent stage transition."""

    changed: bool
    state: StageState
    attempts: int


@dataclass(frozen=True, slots=True)
class BatchStatus:
    """Readiness and ETA summary for one batch."""

    batch_id: str
    total_videos: int
    active_videos: int
    failed_videos: int
    processing_videos: int
    pending_videos: int
    readiness: float
    completed_stage_timings: int
    average_seconds_per_active_video: float | None
    eta_seconds: float | None
    stage_counts: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable status object."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoreIngestionProjection:
    """First-N core-ingestion timing projection for competition readiness."""

    sample_target_videos: int
    sampled_videos: int
    projection_ready: bool
    average_core_seconds_per_video: float | None
    projected_core_seconds: float | None
    preload_window_seconds: float | None
    preload_budget_seconds: float | None
    preload_deadline_fraction: float
    fits_preload_budget: bool | None
    accelerated_profile_mandatory: bool | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_SCHEMA = """
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS archives (
    archive_id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    sha256 TEXT,
    kinds_json TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    mtime_ns INTEGER NOT NULL,
    member_count INTEGER NOT NULL CHECK (member_count >= 0),
    file_count INTEGER NOT NULL CHECK (file_count >= 0),
    uncompressed_bytes INTEGER NOT NULL CHECK (uncompressed_bytes >= 0),
    crc_ok INTEGER,
    state TEXT NOT NULL DEFAULT 'cataloged',
    relocated_to TEXT,
    discovered_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_archives_sha256 ON archives(sha256);

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    competition TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'registered',
    source_label TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    activated_at REAL
);

CREATE TABLE IF NOT EXISTS videos (
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    source_sha256 TEXT,
    state TEXT NOT NULL DEFAULT 'pending',
    active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1)),
    last_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    activated_at REAL,
    PRIMARY KEY (batch_id, video_id),
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_videos_batch_state ON videos(batch_id, state);

CREATE TABLE IF NOT EXISTS stages (
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('running', 'completed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    fingerprint TEXT NOT NULL,
    started_at REAL,
    completed_at REAL,
    error TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (batch_id, video_id, stage_name),
    FOREIGN KEY (batch_id, video_id) REFERENCES videos(batch_id, video_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_stages_batch_state ON stages(batch_id, state);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER CHECK (size_bytes IS NULL OR size_bytes >= 0),
    created_at REAL NOT NULL,
    UNIQUE (batch_id, video_id, stage_name, kind, uri, fingerprint),
    FOREIGN KEY (batch_id, video_id) REFERENCES videos(batch_id, video_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_artifacts_video_kind
ON artifacts(batch_id, video_id, kind);

CREATE TABLE IF NOT EXISTS timings (
    timing_id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL NOT NULL,
    duration_seconds REAL NOT NULL CHECK (duration_seconds >= 0),
    succeeded INTEGER NOT NULL CHECK (succeeded IN (0, 1)),
    UNIQUE (batch_id, video_id, stage_name, attempt),
    FOREIGN KEY (batch_id, video_id) REFERENCES videos(batch_id, video_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_timings_batch_video ON timings(batch_id, video_id);

CREATE TABLE IF NOT EXISTS retrieval_generations (
    batch_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL DEFAULT 0 CHECK (generation >= 0),
    updated_at REAL NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
);
"""


class JobLedger:
    """Explicitly initialized SQLite state for resumable per-video jobs.

    Construction only stores the database path.  It does not create directories,
    open connections, or alter SQLite files.
    """

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms

    def initialize(self) -> None:
        """Create the database and schema, enabling WAL explicitly."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # sqlite3.Connection's context manager commits/rolls back but does not
        # close the handle. Explicit closing is required for deterministic
        # temporary-ledger cleanup on Windows.
        with closing(self._connection()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            connection.commit()

    def _connection(self) -> sqlite3.Connection:
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
    def _timestamp(now: float | None) -> float:
        return time.time() if now is None else now

    def register_batch(
        self,
        batch_id: str,
        competition: str,
        *,
        source_label: str | None = None,
        now: float | None = None,
    ) -> None:
        """Idempotently register a competition batch."""

        if not batch_id.strip() or not competition.strip():
            raise ValueError("batch_id and competition must be non-empty")
        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT competition FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if existing is not None and existing["competition"] != competition:
                raise LedgerError(
                    f"Batch {batch_id!r} already belongs to {existing['competition']!r}"
                )
            connection.execute(
                """
                INSERT INTO batches(
                    batch_id, competition, source_label, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    source_label = COALESCE(excluded.source_label, batches.source_label),
                    updated_at = excluded.updated_at
                """,
                (batch_id, competition, source_label, timestamp, timestamp),
            )

    def register_video(
        self,
        batch_id: str,
        video_id: str,
        source_uri: str,
        *,
        source_sha256: str | None = None,
        now: float | None = None,
    ) -> None:
        """Idempotently add a video while preserving completed state."""

        if not video_id.strip() or not source_uri.strip():
            raise ValueError("video_id and source_uri must be non-empty")
        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT source_sha256 FROM videos
                WHERE batch_id = ? AND video_id = ?
                """,
                (batch_id, video_id),
            ).fetchone()
            if (
                existing is not None
                and existing["source_sha256"] is not None
                and source_sha256 is not None
                and existing["source_sha256"] != source_sha256
            ):
                raise LedgerError(
                    f"Video {batch_id}/{video_id} was registered with a different source digest"
                )
            if existing is None:
                # Adding work to a previously complete batch must immediately
                # revoke export readiness until the new video is activated.
                connection.execute(
                    """
                    UPDATE batches SET state = 'registered', activated_at = NULL,
                        updated_at = ?
                    WHERE batch_id = ? AND state = 'active'
                    """,
                    (timestamp, batch_id),
                )
            connection.execute(
                """
                INSERT INTO videos(
                    batch_id, video_id, source_uri, source_sha256, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id, video_id) DO UPDATE SET
                    source_uri = excluded.source_uri,
                    source_sha256 = COALESCE(excluded.source_sha256, videos.source_sha256),
                    updated_at = excluded.updated_at
                """,
                (batch_id, video_id, source_uri, source_sha256, timestamp, timestamp),
            )

    def record_archive(
        self,
        inspection: ArchiveInspection,
        *,
        relocated_to: str | Path | None = None,
        now: float | None = None,
    ) -> None:
        """Upsert one inspected archive inventory."""

        timestamp = self._timestamp(now)
        kinds_json = json.dumps([kind.value for kind in inspection.kinds])
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO archives(
                    path, sha256, kinds_json, size_bytes, mtime_ns, member_count,
                    file_count, uncompressed_bytes, crc_ok, state, relocated_to,
                    discovered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    sha256 = excluded.sha256,
                    kinds_json = excluded.kinds_json,
                    size_bytes = excluded.size_bytes,
                    mtime_ns = excluded.mtime_ns,
                    member_count = excluded.member_count,
                    file_count = excluded.file_count,
                    uncompressed_bytes = excluded.uncompressed_bytes,
                    crc_ok = excluded.crc_ok,
                    relocated_to = COALESCE(excluded.relocated_to, archives.relocated_to),
                    state = CASE
                        WHEN excluded.relocated_to IS NOT NULL THEN 'relocated'
                        ELSE archives.state
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    str(inspection.path),
                    inspection.sha256,
                    kinds_json,
                    inspection.size_bytes,
                    inspection.mtime_ns,
                    inspection.member_count,
                    inspection.file_count,
                    inspection.uncompressed_bytes,
                    None if inspection.crc_ok is None else int(inspection.crc_ok),
                    "relocated" if relocated_to is not None else "cataloged",
                    None if relocated_to is None else str(relocated_to),
                    timestamp,
                    timestamp,
                ),
            )

    def archive_record(self, path: str | Path) -> dict[str, Any] | None:
        """Return a catalog row matching either its arrival or relocated path."""

        resolved = str(Path(path).resolve())
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT * FROM archives
                WHERE path = ? OR relocated_to = ?
                ORDER BY CASE WHEN relocated_to = ? THEN 0 ELSE 1 END
                LIMIT 1
                """,
                (resolved, resolved, resolved),
            ).fetchone()
        return None if row is None else dict(row)

    def assert_archive_ready(
        self,
        path: str | Path,
        *,
        protected_names: Iterable[str] = ("Keyframes_L21.zip", "Videos_L21_a.zip"),
    ) -> dict[str, Any]:
        """Require a stable, CRC-verified, hashed archive before ingestion.

        Ordinary arrivals must have completed canonical relocation.  The two
        immutable development archives are the sole exception: they may remain
        in place after the same CRC/SHA gate.
        """

        candidate = Path(path).resolve(strict=True)
        row = self.archive_record(candidate)
        if row is None:
            raise LedgerError(
                f"Archive {candidate} is unmanaged; run scripts/catalog.py with full "
                "verification and relocation before ingestion"
            )
        if row["crc_ok"] != 1 or not row["sha256"]:
            raise LedgerError(f"Archive {candidate} has not passed CRC and SHA-256 verification")
        protected = {name.casefold() for name in protected_names}
        allowed_in_place = candidate.name.casefold() in protected and row["state"] == "cataloged"
        if row["state"] != "relocated" and not allowed_in_place:
            raise LedgerError(f"Archive {candidate} has not completed canonical relocation")
        stat = candidate.stat()
        if int(row["size_bytes"]) != stat.st_size or int(row["mtime_ns"]) != stat.st_mtime_ns:
            raise LedgerError(f"Archive {candidate} changed after verification")
        if row["state"] == "relocated" and Path(row["relocated_to"]).resolve() != candidate:
            raise LedgerError(f"Archive {candidate} is not at its recorded canonical path")
        return row

    def start_stage(
        self,
        batch_id: str,
        video_id: str,
        stage_name: str,
        fingerprint: str,
        *,
        now: float | None = None,
        allow_reprocess: bool = False,
    ) -> StageTransition:
        """Start or retry a stage; repeated running/completed calls are no-ops."""

        if not stage_name.strip() or not fingerprint.strip():
            raise ValueError("stage_name and fingerprint must be non-empty")
        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT state, attempts, fingerprint FROM stages
                WHERE batch_id = ? AND video_id = ? AND stage_name = ?
                """,
                (batch_id, video_id, stage_name),
            ).fetchone()
            if row is not None:
                state = StageState(row["state"])
                attempts = int(row["attempts"])
                if state is StageState.RUNNING and row["fingerprint"] == fingerprint:
                    return StageTransition(False, state, attempts)
                if (
                    state is StageState.COMPLETED
                    and row["fingerprint"] == fingerprint
                    and not allow_reprocess
                ):
                    return StageTransition(False, state, attempts)
                if state is StageState.COMPLETED and not allow_reprocess:
                    raise StageFingerprintMismatch(
                        f"Completed stage {stage_name!r} has fingerprint "
                        f"{row['fingerprint']!r}, not {fingerprint!r}"
                    )
                if state is StageState.RUNNING:
                    raise StageFingerprintMismatch(
                        f"Running stage {stage_name!r} has a different fingerprint"
                    )
                attempts += 1
                if state is StageState.COMPLETED:
                    connection.execute(
                        """
                        DELETE FROM artifacts
                        WHERE batch_id = ? AND video_id = ? AND stage_name = ?
                        """,
                        (batch_id, video_id, stage_name),
                    )
                connection.execute(
                    """
                    UPDATE stages SET state = 'running', attempts = ?, fingerprint = ?,
                        started_at = ?, completed_at = NULL, error = NULL, updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND stage_name = ?
                    """,
                    (
                        attempts,
                        fingerprint,
                        timestamp,
                        timestamp,
                        batch_id,
                        video_id,
                        stage_name,
                    ),
                )
            else:
                attempts = 1
                connection.execute(
                    """
                    INSERT INTO stages(
                        batch_id, video_id, stage_name, state, attempts, fingerprint,
                        started_at, updated_at
                    ) VALUES (?, ?, ?, 'running', 1, ?, ?, ?)
                    """,
                    (batch_id, video_id, stage_name, fingerprint, timestamp, timestamp),
                )
            connection.execute(
                """
                UPDATE videos SET state = 'processing', last_error = NULL, updated_at = ?
                WHERE batch_id = ? AND video_id = ? AND active = 0
                """,
                (timestamp, batch_id, video_id),
            )
            return StageTransition(True, StageState.RUNNING, attempts)

    def _finish_stage(
        self,
        connection: sqlite3.Connection,
        *,
        batch_id: str,
        video_id: str,
        stage_name: str,
        state: StageState,
        timestamp: float,
        error: str | None,
    ) -> tuple[sqlite3.Row, bool]:
        row = connection.execute(
            """
            SELECT state, attempts, fingerprint, started_at FROM stages
            WHERE batch_id = ? AND video_id = ? AND stage_name = ?
            """,
            (batch_id, video_id, stage_name),
        ).fetchone()
        if row is None:
            raise LedgerError(f"Stage {stage_name!r} has not been started")
        if row["state"] == state.value:
            return row, False
        if row["state"] != StageState.RUNNING.value:
            raise LedgerError(
                f"Cannot transition stage {stage_name!r} from {row['state']!r} to {state.value!r}"
            )
        started_at = float(row["started_at"])
        duration = max(0.0, timestamp - started_at)
        connection.execute(
            """
            UPDATE stages SET state = ?, completed_at = ?, error = ?, updated_at = ?
            WHERE batch_id = ? AND video_id = ? AND stage_name = ?
            """,
            (state.value, timestamp, error, timestamp, batch_id, video_id, stage_name),
        )
        connection.execute(
            """
            INSERT INTO timings(
                batch_id, video_id, stage_name, attempt, started_at, finished_at,
                duration_seconds, succeeded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_id, video_id, stage_name, attempt) DO NOTHING
            """,
            (
                batch_id,
                video_id,
                stage_name,
                int(row["attempts"]),
                started_at,
                timestamp,
                duration,
                int(state is StageState.COMPLETED),
            ),
        )
        return row, True

    def complete_stage(
        self,
        batch_id: str,
        video_id: str,
        stage_name: str,
        *,
        artifacts: Iterable[ArtifactSpec] = (),
        now: float | None = None,
    ) -> StageTransition:
        """Atomically register outputs and mark a running stage complete."""

        timestamp = self._timestamp(now)
        artifact_list = tuple(artifacts)
        with self._transaction() as connection:
            row, changed = self._finish_stage(
                connection,
                batch_id=batch_id,
                video_id=video_id,
                stage_name=stage_name,
                state=StageState.COMPLETED,
                timestamp=timestamp,
                error=None,
            )
            if not changed:
                return StageTransition(False, StageState.COMPLETED, int(row["attempts"]))
            for artifact in artifact_list:
                if not artifact.kind.strip() or not artifact.uri.strip():
                    raise ValueError("artifact kind and uri must be non-empty")
                if artifact.fingerprint != row["fingerprint"]:
                    raise StageFingerprintMismatch(
                        f"Artifact {artifact.uri!r} does not match stage fingerprint"
                    )
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        batch_id, video_id, stage_name, kind, uri, fingerprint,
                        sha256, size_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(batch_id, video_id, stage_name, kind, uri, fingerprint)
                    DO NOTHING
                    """,
                    (
                        batch_id,
                        video_id,
                        stage_name,
                        artifact.kind,
                        artifact.uri,
                        artifact.fingerprint,
                        artifact.sha256,
                        artifact.size_bytes,
                        timestamp,
                    ),
                )
            return StageTransition(True, StageState.COMPLETED, int(row["attempts"]))

    def fail_stage(
        self,
        batch_id: str,
        video_id: str,
        stage_name: str,
        error: str,
        *,
        now: float | None = None,
    ) -> StageTransition:
        """Mark a running stage failed while preserving it for a later retry."""

        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            row, changed = self._finish_stage(
                connection,
                batch_id=batch_id,
                video_id=video_id,
                stage_name=stage_name,
                state=StageState.FAILED,
                timestamp=timestamp,
                error=error,
            )
            if changed:
                connection.execute(
                    """
                    UPDATE videos SET state = 'failed', last_error = ?, updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND active = 0
                    """,
                    (error, timestamp, batch_id, video_id),
                )
            return StageTransition(changed, StageState.FAILED, int(row["attempts"]))

    def activate_video(
        self,
        batch_id: str,
        video_id: str,
        *,
        required_stages: Iterable[str] = (),
        required_artifacts: Mapping[str, int] | None = None,
        now: float | None = None,
    ) -> bool:
        """Atomically activate a video after checking stage and artifact counts."""

        timestamp = self._timestamp(now)
        stages = tuple(dict.fromkeys(required_stages))
        artifact_counts = dict(required_artifacts or {})
        if any(count < 0 for count in artifact_counts.values()):
            raise ValueError("required artifact counts must be non-negative")
        with self._transaction() as connection:
            video = connection.execute(
                "SELECT active FROM videos WHERE batch_id = ? AND video_id = ?",
                (batch_id, video_id),
            ).fetchone()
            if video is None:
                raise LedgerError(f"Unknown video {batch_id}/{video_id}")
            if bool(video["active"]):
                return False
            if stages:
                placeholders = ",".join("?" for _ in stages)
                rows = connection.execute(
                    f"""
                    SELECT stage_name, state FROM stages
                    WHERE batch_id = ? AND video_id = ?
                        AND stage_name IN ({placeholders})
                    """,
                    (batch_id, video_id, *stages),
                ).fetchall()
                completed = {row["stage_name"] for row in rows if row["state"] == "completed"}
                missing = sorted(set(stages) - completed)
                if missing:
                    raise LedgerError(f"Required stages are incomplete: {', '.join(missing)}")
            for kind, minimum in artifact_counts.items():
                count = connection.execute(
                    """
                    SELECT COUNT(*) FROM artifacts
                    WHERE batch_id = ? AND video_id = ? AND kind = ?
                    """,
                    (batch_id, video_id, kind),
                ).fetchone()[0]
                if count < minimum:
                    raise LedgerError(
                        f"Artifact {kind!r} count {count} is below required {minimum}"
                    )
            connection.execute(
                """
                UPDATE videos SET active = 1, state = 'active', last_error = NULL,
                    activated_at = ?, updated_at = ?
                WHERE batch_id = ? AND video_id = ?
                """,
                (timestamp, timestamp, batch_id, video_id),
            )
            return True

    def deactivate_video(
        self,
        batch_id: str,
        video_id: str,
        error: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Revoke video and batch readiness after durable artifact validation fails."""

        if not error.strip():
            raise ValueError("error must be non-empty")
        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT active FROM videos WHERE batch_id = ? AND video_id = ?",
                (batch_id, video_id),
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown video {batch_id}/{video_id}")
            connection.execute(
                """
                UPDATE videos SET active = 0, state = 'failed', last_error = ?,
                    activated_at = NULL, updated_at = ?
                WHERE batch_id = ? AND video_id = ?
                """,
                (error, timestamp, batch_id, video_id),
            )
            connection.execute(
                """
                UPDATE batches SET state = 'registered', activated_at = NULL,
                    updated_at = ? WHERE batch_id = ? AND state = 'active'
                """,
                (timestamp, batch_id),
            )
            return bool(row["active"])

    def activate_batch(self, batch_id: str, *, now: float | None = None) -> bool:
        """Activate a batch only after every registered video is active."""

        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT b.state AS batch_state,
                    COUNT(v.video_id) AS total,
                    COALESCE(SUM(v.active), 0) AS active
                FROM batches b LEFT JOIN videos v ON v.batch_id = b.batch_id
                WHERE b.batch_id = ? GROUP BY b.batch_id
                """,
                (batch_id,),
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown batch {batch_id!r}")
            if row["batch_state"] == "active":
                return False
            if int(row["total"]) == 0 or int(row["active"]) != int(row["total"]):
                raise LedgerError("Batch cannot activate until every video is active")
            connection.execute(
                """
                UPDATE batches SET state = 'active', activated_at = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (timestamp, timestamp, batch_id),
            )
            return True

    def activate_batch_if_ready(self, batch_id: str, *, now: float | None = None) -> bool:
        """Atomically activate a fully ready batch, otherwise leave it pending.

        This is the ingestion-safe counterpart to :meth:`activate_batch`: a
        partial retry may process only some already-registered videos and must
        not turn that expected condition into a pipeline failure.
        """

        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT b.state AS batch_state,
                    COUNT(v.video_id) AS total,
                    COALESCE(SUM(v.active), 0) AS active
                FROM batches b LEFT JOIN videos v ON v.batch_id = b.batch_id
                WHERE b.batch_id = ? GROUP BY b.batch_id
                """,
                (batch_id,),
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown batch {batch_id!r}")
            if row["batch_state"] == "active":
                return False
            if int(row["total"]) == 0 or int(row["active"]) != int(row["total"]):
                return False
            connection.execute(
                """
                UPDATE batches SET state = 'active', activated_at = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (timestamp, timestamp, batch_id),
            )
            return True

    def is_batch_active(self, batch_id: str) -> bool:
        """Return the persisted activation state for a known batch."""

        with closing(self._connection()) as connection:
            row = connection.execute(
                "SELECT state FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise LedgerError(f"Unknown batch {batch_id!r}")
        return row["state"] == "active"

    def is_batch_registered(self, batch_id: str) -> bool:
        """Inspect batch registration read-only, even before normal initialization.

        This is used by runtime input validation.  It never creates the database
        or schema and therefore preserves side-effect-free runtime construction.
        """

        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        if not self.path.is_file():
            return False
        try:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            with closing(
                sqlite3.connect(uri, uri=True, timeout=self.busy_timeout_ms / 1_000)
            ) as connection:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'batches'"
                ).fetchone()
                if table is None:
                    return False
                row = connection.execute(
                    "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
                ).fetchone()
            return row is not None
        except sqlite3.Error as error:
            raise LedgerError("batch registry could not be inspected safely") from error

    def retrieval_generation(self, batch_id: str) -> int:
        """Return the monotonic retrieval generation for a known batch."""

        with closing(self._connection()) as connection:
            batch = connection.execute(
                "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise LedgerError(f"Unknown batch {batch_id!r}")
            row = connection.execute(
                "SELECT generation FROM retrieval_generations WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        return 0 if row is None else int(row["generation"])

    def bump_retrieval_generation(
        self, batch_id: str, *, now: float | None = None
    ) -> int:
        """Atomically invalidate retrieval-derived caches after index activation."""

        timestamp = self._timestamp(now)
        with self._transaction() as connection:
            batch = connection.execute(
                "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise LedgerError(f"Unknown batch {batch_id!r}")
            connection.execute(
                """
                INSERT INTO retrieval_generations(batch_id, generation, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    generation = retrieval_generations.generation + 1,
                    updated_at = excluded.updated_at
                """,
                (batch_id, timestamp),
            )
            row = connection.execute(
                "SELECT generation FROM retrieval_generations WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            assert row is not None
            return int(row["generation"])

    def get_stage(self, batch_id: str, video_id: str, stage_name: str) -> dict[str, Any] | None:
        """Return one stage record for diagnostics."""

        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT * FROM stages
                WHERE batch_id = ? AND video_id = ? AND stage_name = ?
                """,
                (batch_id, video_id, stage_name),
            ).fetchone()
        return None if row is None else dict(row)

    def list_video_ids(self, batch_id: str) -> tuple[str, ...]:
        """List registered videos without exposing their source URIs."""

        if not batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        with closing(self._connection()) as connection:
            rows = connection.execute(
                "SELECT video_id FROM videos WHERE batch_id = ? ORDER BY video_id",
                (batch_id,),
            ).fetchall()
        return tuple(str(row["video_id"]) for row in rows)

    def stage_records(
        self,
        batch_id: str,
        stage_name: str,
    ) -> tuple[dict[str, Any], ...]:
        """Return sanitized stage progress for readiness/checkpoint projections."""

        if not batch_id.strip() or not stage_name.strip():
            raise ValueError("batch_id and stage_name must be non-empty")
        with closing(self._connection()) as connection:
            rows = connection.execute(
                """
                SELECT video_id, state, attempts, fingerprint, started_at,
                       completed_at, error, updated_at
                FROM stages
                WHERE batch_id = ? AND stage_name = ?
                ORDER BY video_id
                """,
                (batch_id, stage_name),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def recover_interrupted(
        self,
        batch_id: str,
        *,
        older_than_seconds: float = 0.0,
        now: float | None = None,
    ) -> int:
        """Mark stale running stages failed so a single-worker run can resume.

        This is explicit rather than automatic because a multi-host deployment
        must not reclaim a stage that another worker is still processing.
        """

        if older_than_seconds < 0:
            raise ValueError("older_than_seconds must be non-negative")
        timestamp = self._timestamp(now)
        cutoff = timestamp - older_than_seconds
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT batch_id, video_id, stage_name, attempts, started_at
                FROM stages
                WHERE batch_id = ? AND state = 'running' AND started_at <= ?
                """,
                (batch_id, cutoff),
            ).fetchall()
            for row in rows:
                duration = max(0.0, timestamp - float(row["started_at"]))
                connection.execute(
                    """
                    UPDATE stages SET state = 'failed', completed_at = ?,
                        error = 'interrupted before atomic completion', updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND stage_name = ?
                    """,
                    (
                        timestamp,
                        timestamp,
                        row["batch_id"],
                        row["video_id"],
                        row["stage_name"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO timings(
                        batch_id, video_id, stage_name, attempt, started_at,
                        finished_at, duration_seconds, succeeded
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(batch_id, video_id, stage_name, attempt) DO NOTHING
                    """,
                    (
                        row["batch_id"],
                        row["video_id"],
                        row["stage_name"],
                        row["attempts"],
                        row["started_at"],
                        timestamp,
                        duration,
                    ),
                )
                connection.execute(
                    """
                    UPDATE videos SET state = 'failed',
                        last_error = 'interrupted before atomic completion', updated_at = ?
                    WHERE batch_id = ? AND video_id = ? AND active = 0
                    """,
                    (timestamp, row["batch_id"], row["video_id"]),
                )
        return len(rows)

    def status_summary(self, batch_id: str) -> BatchStatus:
        """Return batch readiness, failures, per-stage counts, and an ETA."""

        with closing(self._connection()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if exists is None:
                raise LedgerError(f"Unknown batch {batch_id!r}")
            videos = connection.execute(
                """
                SELECT COUNT(*) AS total,
                    COALESCE(SUM(active), 0) AS active,
                    COALESCE(SUM(state = 'failed'), 0) AS failed,
                    COALESCE(SUM(state = 'processing'), 0) AS processing
                FROM videos WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()
            stage_rows = connection.execute(
                """
                SELECT stage_name, state, COUNT(*) AS count
                FROM stages WHERE batch_id = ? GROUP BY stage_name, state
                """,
                (batch_id,),
            ).fetchall()
            timing_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM timings WHERE batch_id = ? AND succeeded = 1",
                    (batch_id,),
                ).fetchone()[0]
            )
            active_durations = connection.execute(
                """
                SELECT SUM(t.duration_seconds) AS duration
                FROM videos v JOIN timings t
                    ON t.batch_id = v.batch_id AND t.video_id = v.video_id
                WHERE v.batch_id = ? AND v.active = 1
                GROUP BY v.video_id
                """,
                (batch_id,),
            ).fetchall()

        total = int(videos["total"])
        active = int(videos["active"])
        failed = int(videos["failed"])
        processing = int(videos["processing"])
        pending = max(0, total - active - failed - processing)
        durations = [float(row["duration"]) for row in active_durations]
        average = sum(durations) / len(durations) if durations else None
        eta = average * (total - active) if average is not None else None
        stage_counts: dict[str, dict[str, int]] = {}
        for row in stage_rows:
            stage_counts.setdefault(str(row["stage_name"]), {})[str(row["state"])] = int(
                row["count"]
            )
        return BatchStatus(
            batch_id=batch_id,
            total_videos=total,
            active_videos=active,
            failed_videos=failed,
            processing_videos=processing,
            pending_videos=pending,
            readiness=active / total if total else 0.0,
            completed_stage_timings=timing_count,
            average_seconds_per_active_video=average,
            eta_seconds=eta,
            stage_counts=stage_counts,
        )

    def core_ingestion_projection(
        self,
        batch_id: str,
        *,
        total_videos: int,
        sample_videos: int = 10,
        core_stages: Iterable[str] = ("probe", "frames", "visual"),
        preload_window_seconds: float | None = None,
        deadline_fraction: float = 0.8,
    ) -> CoreIngestionProjection:
        """Project full core ingestion from the first activated sample.

        A projection is deliberately withheld until ``sample_videos`` complete
        videos exist.  Visual stages with an encoder suffix are included so the
        accelerated profile is measured rather than treated as free work.
        """

        if total_videos < 0:
            raise ValueError("total_videos must be non-negative")
        if sample_videos < 1:
            raise ValueError("sample_videos must be positive")
        stages = tuple(dict.fromkeys(str(value) for value in core_stages))
        if not stages or any(not value.strip() for value in stages):
            raise ValueError("core_stages must contain non-empty stage names")
        if not 0.0 < deadline_fraction <= 1.0:
            raise ValueError("deadline_fraction must be between 0 and 1")
        if preload_window_seconds is not None and preload_window_seconds <= 0:
            raise ValueError("preload_window_seconds must be positive when supplied")
        placeholders = ",".join("?" for _ in stages)
        with closing(self._connection()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if exists is None:
                raise LedgerError(f"Unknown batch {batch_id!r}")
            rows = connection.execute(
                f"""
                SELECT v.video_id, SUM(t.duration_seconds) AS duration
                FROM videos v JOIN timings t
                    ON t.batch_id = v.batch_id AND t.video_id = v.video_id
                WHERE v.batch_id = ? AND v.active = 1 AND t.succeeded = 1
                    AND (t.stage_name IN ({placeholders})
                        OR t.stage_name LIKE 'visual\\_%' ESCAPE '\\')
                GROUP BY v.video_id, v.activated_at
                HAVING COUNT(DISTINCT CASE
                    WHEN t.stage_name IN ({placeholders}) THEN t.stage_name
                END) >= ?
                ORDER BY v.activated_at, v.video_id
                LIMIT ?
                """,
                (batch_id, *stages, *stages, len(stages), sample_videos),
            ).fetchall()
        durations = [float(row["duration"]) for row in rows]
        ready = len(durations) >= sample_videos
        average = sum(durations) / len(durations) if durations else None
        projected = average * total_videos if ready and average is not None else None
        budget = (
            preload_window_seconds * deadline_fraction
            if preload_window_seconds is not None
            else None
        )
        fits = projected <= budget if projected is not None and budget is not None else None
        return CoreIngestionProjection(
            sample_target_videos=sample_videos,
            sampled_videos=len(durations),
            projection_ready=ready,
            average_core_seconds_per_video=average,
            projected_core_seconds=projected,
            preload_window_seconds=preload_window_seconds,
            preload_budget_seconds=budget,
            preload_deadline_fraction=deadline_fraction,
            fits_preload_budget=fits,
            accelerated_profile_mandatory=None if fits is None else not fits,
        )
