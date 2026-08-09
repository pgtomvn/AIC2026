"""Explicit SQLite sessions and reasoning cache for Visual Q&A."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SESSION_TTL_SECONDS = 24 * 60 * 60
_REASONING_TTL_SECONDS = 30 * 24 * 60 * 60
_CANDIDATE_TTL_SECONDS = 24 * 60 * 60
_IMAGE_FINGERPRINT_TTL_SECONDS = 30 * 24 * 60 * 60

_STABLE_FAILURES: dict[str, tuple[str, str, str]] = {
    "media_unavailable": ("media_failed", "media_unavailable", "media"),
    "media_window_too_large": (
        "media_failed",
        "media_window_too_large",
        "media",
    ),
    "provider_unavailable": (
        "provider_failed",
        "provider_unavailable",
        "provider",
    ),
    "provider_request_failed": (
        "provider_failed",
        "provider_request_failed",
        "provider",
    ),
    "provider_response_invalid": (
        "provider_failed",
        "provider_response_invalid",
        "provider",
    ),
    "internal_reasoning_error": (
        "internal_failed",
        "internal_reasoning_error",
        "internal",
    ),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_sessions (
    session_id TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS qa_session_targets (
    session_id TEXT NOT NULL,
    target_key TEXT NOT NULL,
    target_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, target_key),
    FOREIGN KEY (session_id) REFERENCES qa_sessions(session_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS qa_reasoning_cache (
    cache_key TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS qa_operator_edits (
    edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    window_id TEXT NOT NULL,
    edit_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES qa_sessions(session_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS qa_candidate_sets (
    cache_key TEXT PRIMARY KEY,
    windows_json TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS qa_session_windows (
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, position),
    UNIQUE (session_id, window_id),
    FOREIGN KEY (session_id) REFERENCES qa_sessions(session_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS qa_session_candidates (
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, position),
    UNIQUE (session_id, candidate_id),
    FOREIGN KEY (session_id) REFERENCES qa_sessions(session_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS qa_image_fingerprints (
    cache_key TEXT PRIMARY KEY,
    fingerprint_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS qa_cache_leases (
    cache_key TEXT PRIMARY KEY,
    owner_token TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qa_sessions_expiry ON qa_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_qa_reasoning_expiry ON qa_reasoning_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_qa_candidates_expiry ON qa_candidate_sets(expires_at);
CREATE INDEX IF NOT EXISTS idx_qa_images_expiry ON qa_image_fingerprints(expires_at);
CREATE INDEX IF NOT EXISTS idx_qa_windows_session
ON qa_session_windows(session_id, position);
CREATE INDEX IF NOT EXISTS idx_qa_candidates_session
ON qa_session_candidates(session_id, position);
CREATE INDEX IF NOT EXISTS idx_qa_operator_edits_session
ON qa_operator_edits(session_id, edit_id);
"""


class QASessionStoreError(RuntimeError):
    """Base error for explicit QA session persistence."""


@dataclass(frozen=True, slots=True)
class QASession:
    """One live QA request and its JSON state."""

    session_id: str
    request: dict[str, Any]
    state: dict[str, Any]
    created_at: float
    updated_at: float
    expires_at: float
    revision: int = 0


def reasoning_cache_key(
    *,
    provider_fingerprint: str,
    request_fingerprint: str,
    image_sha256s: Iterable[str] = (),
    frame_fingerprints: Iterable[Mapping[str, Any]] = (),
    answer_spec_id: str = "",
) -> str:
    """Build a stable reasoning-cache key without storing secret material.

    ``provider_fingerprint`` is expected to cover prompt/normalizer versions and
    generation settings. Ordered canonical identities and normalized image
    hashes prevent two frames with identical pixels from sharing provenance.
    """

    payload = {
        "version": "qa-reason-v2",
        "provider": provider_fingerprint,
        "request": request_fingerprint,
        "answer_spec_id": answer_spec_id,
        "frames": [dict(value) for value in frame_fingerprints],
        "legacy_image_hashes": list(image_sha256s),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class QASessionStore:
    """SQLite-backed QA state with 24-hour sessions and 30-day reasoning cache."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_ttl_seconds: int = _SESSION_TTL_SECONDS,
        reasoning_ttl_seconds: int = _REASONING_TTL_SECONDS,
        candidate_ttl_seconds: int = _CANDIDATE_TTL_SECONDS,
        image_fingerprint_ttl_seconds: int = _IMAGE_FINGERPRINT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if (
            session_ttl_seconds < 1
            or reasoning_ttl_seconds < 1
            or candidate_ttl_seconds < 1
            or image_fingerprint_ttl_seconds < 1
        ):
            raise ValueError("QA TTL values must be positive")
        self.path = Path(path)
        self.session_ttl_seconds = int(session_ttl_seconds)
        self.reasoning_ttl_seconds = int(reasoning_ttl_seconds)
        self.candidate_ttl_seconds = int(candidate_ttl_seconds)
        self.image_fingerprint_ttl_seconds = int(image_fingerprint_ttl_seconds)
        self.clock = clock
        self._initialized = False

    def initialize(self) -> None:
        """Explicitly create the QA schema; construction remains side-effect free."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(qa_sessions)")}
            if "revision" not in columns:
                connection.execute(
                    "ALTER TABLE qa_sessions ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
                )
            candidate_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(qa_candidate_sets)")
            }
            if "diagnostics_json" not in candidate_columns:
                connection.execute(
                    """
                    ALTER TABLE qa_candidate_sets
                    ADD COLUMN diagnostics_json TEXT NOT NULL DEFAULT '{}'
                    """
                )
            connection.commit()
        self._initialized = True

    def _connection(self) -> sqlite3.Connection:
        if not self._initialized:
            raise QASessionStoreError("QASessionStore.initialize() must be called first")
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        result = dict(value)
        try:
            json.dumps(result, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"{name} must be JSON serializable") from exc
        return result

    @staticmethod
    def _reject_media_locations(value: Any, name: str) -> None:
        forbidden = {"image_uri", "video_uri", "path", "member", "data", "bytes"}
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in forbidden:
                    raise ValueError(f"{name} contains forbidden media location or bytes")
                QASessionStore._reject_media_locations(item, name)
        elif isinstance(value, (list, tuple)):
            for item in value:
                QASessionStore._reject_media_locations(item, name)

    @staticmethod
    def _session(row: sqlite3.Row) -> QASession:
        return QASession(
            session_id=str(row["session_id"]),
            request=json.loads(row["request_json"]),
            state=json.loads(row["state_json"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            expires_at=float(row["expires_at"]),
            revision=int(row["revision"]),
        )

    def create_session(
        self,
        session_id: str,
        request: Mapping[str, Any],
        *,
        state: Mapping[str, Any] | None = None,
    ) -> QASession:
        """Create a session idempotently when the original request is identical."""

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        request_value = self._mapping(request, "request")
        state_value = self._mapping(state or {}, "state")
        request_json = json.dumps(request_value, ensure_ascii=False, sort_keys=True)
        state_json = json.dumps(state_value, ensure_ascii=False, sort_keys=True)
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM qa_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is not None and float(row["expires_at"]) > now:
                if row["request_json"] != request_json:
                    raise QASessionStoreError("session_id is already bound to another request")
                connection.commit()
                return self._session(row)
            if row is not None:
                connection.execute("DELETE FROM qa_sessions WHERE session_id = ?", (session_id,))
            expires = now + self.session_ttl_seconds
            connection.execute(
                """
                INSERT INTO qa_sessions
                    (session_id, request_json, state_json, created_at, updated_at, expires_at,
                     revision)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (session_id, request_json, state_json, now, now, expires),
            )
            connection.commit()
            return QASession(session_id, request_value, state_value, now, now, expires, 0)

    def get_session(self, session_id: str) -> QASession | None:
        """Return a non-expired session without extending its fixed 24-hour TTL."""

        now = self.clock()
        with closing(self._connection()) as connection:
            row = connection.execute(
                "SELECT * FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
        return None if row is None else self._session(row)

    def update_state(
        self,
        session_id: str,
        state: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> QASession:
        """Replace JSON session state without changing its expiry."""

        state_value = self._mapping(state, "state")
        now = self.clock()
        with closing(self._connection()) as connection:
            parameters: list[Any] = [
                json.dumps(state_value, ensure_ascii=False, sort_keys=True),
                now,
                session_id,
                now,
            ]
            revision_clause = ""
            if expected_revision is not None:
                if isinstance(expected_revision, bool) or expected_revision < 0:
                    raise ValueError("expected_revision must be a non-negative integer")
                revision_clause = " AND revision = ?"
                parameters.append(int(expected_revision))
            cursor = connection.execute(
                """
                UPDATE qa_sessions
                SET state_json = ?, updated_at = ?, revision = revision + 1
                WHERE session_id = ? AND expires_at > ?
                """
                + revision_clause,
                parameters,
            )
            if cursor.rowcount != 1:
                raise QASessionStoreError(
                    "QA session is missing, expired, or concurrently modified"
                )
            connection.commit()
        session = self.get_session(session_id)
        if session is None:  # defensive against a clock jump between transactions
            raise QASessionStoreError("QA session expired during update")
        return session

    @staticmethod
    def _target_key(target: Mapping[str, Any]) -> str:
        batch = target.get("batch_id")
        video = target.get("video_id")
        frame = target.get("source_frame_idx")
        if not isinstance(batch, str) or not batch or not isinstance(video, str) or not video:
            raise ValueError("QA target requires batch_id and video_id")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("QA target requires a non-negative source_frame_idx")
        return f"{batch}:{video}:{frame}"

    def expand_targets(
        self, session_id: str, targets: Iterable[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Add unseen canonical targets atomically and return only newly added values."""

        prepared: dict[str, dict[str, Any]] = {}
        for target in targets:
            value = self._mapping(target, "target")
            prepared[self._target_key(value)] = value
        now = self.clock()
        added: list[dict[str, Any]] = []
        with closing(self._connection()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            for key, value in prepared.items():
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO qa_session_targets
                        (session_id, target_key, target_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        key,
                        json.dumps(value, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
                if cursor.rowcount == 1:
                    added.append(value)
            connection.commit()
        return tuple(added)

    def get_targets(self, session_id: str) -> tuple[dict[str, Any], ...]:
        """Return cumulative canonical targets for a live session in insertion order."""

        now = self.clock()
        with closing(self._connection()) as connection:
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            rows = connection.execute(
                """
                SELECT target_json FROM qa_session_targets
                WHERE session_id = ? ORDER BY rowid
                """,
                (session_id,),
            ).fetchall()
        return tuple(json.loads(row["target_json"]) for row in rows)

    def bind_session_windows(
        self, session_id: str, windows: Iterable[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Freeze one ordered, URI-free candidate set for a live session."""

        prepared = [self._mapping(window, "window") for window in windows]
        for value in prepared:
            self._reject_media_locations(value, "window")
        window_ids: set[str] = set()
        for value in prepared:
            window_id = value.get("window_id")
            if not isinstance(window_id, str) or not window_id:
                raise ValueError("QA session window requires a non-empty window_id")
            if window_id in window_ids:
                raise ValueError("QA session windows must have unique window_id values")
            window_ids.add(window_id)
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            existing = connection.execute(
                """
                SELECT window_json FROM qa_session_windows
                WHERE session_id = ? ORDER BY position
                """,
                (session_id,),
            ).fetchall()
            if existing:
                frozen = tuple(json.loads(row["window_json"]) for row in existing)
                connection.commit()
                return frozen
            for position, value in enumerate(prepared):
                connection.execute(
                    """
                    INSERT INTO qa_session_windows
                        (session_id, position, window_id, window_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        position,
                        value["window_id"],
                        json.dumps(value, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
            connection.commit()
        return tuple(prepared)

    def get_session_windows(self, session_id: str) -> tuple[dict[str, Any], ...]:
        """Return ordered frozen windows without extending the session TTL."""

        now = self.clock()
        with closing(self._connection()) as connection:
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            rows = connection.execute(
                """
                SELECT window_json FROM qa_session_windows
                WHERE session_id = ? ORDER BY position
                """,
                (session_id,),
            ).fetchall()
        return tuple(json.loads(row["window_json"]) for row in rows)

    def bind_session_candidates(
        self, session_id: str, candidates: Iterable[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Freeze one ordered, URI-free single-frame candidate set."""

        prepared = [self._mapping(candidate, "candidate") for candidate in candidates]
        candidate_ids: set[str] = set()
        for value in prepared:
            self._reject_media_locations(value, "candidate")
            candidate_id = value.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("QA candidate requires a non-empty candidate_id")
            if candidate_id in candidate_ids:
                raise ValueError("QA candidates must have unique candidate_id values")
            candidate_ids.add(candidate_id)
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            existing = connection.execute(
                """
                SELECT candidate_json FROM qa_session_candidates
                WHERE session_id = ? ORDER BY position
                """,
                (session_id,),
            ).fetchall()
            if existing:
                frozen = tuple(json.loads(row["candidate_json"]) for row in existing)
                connection.commit()
                return frozen
            for position, value in enumerate(prepared):
                connection.execute(
                    """
                    INSERT INTO qa_session_candidates
                        (session_id, position, candidate_id, candidate_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        position,
                        value["candidate_id"],
                        json.dumps(value, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
            connection.commit()
        return tuple(prepared)

    def get_session_candidates(self, session_id: str) -> tuple[dict[str, Any], ...]:
        """Return frozen v3 candidates without extending the session TTL."""

        now = self.clock()
        with closing(self._connection()) as connection:
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            rows = connection.execute(
                """
                SELECT candidate_json FROM qa_session_candidates
                WHERE session_id = ? ORDER BY position
                """,
                (session_id,),
            ).fetchall()
        return tuple(json.loads(row["candidate_json"]) for row in rows)

    @staticmethod
    def _legacy_target_key(value: Mapping[str, Any]) -> str | None:
        batch_id = value.get("batch_id")
        video_id = value.get("video_id")
        source_frame_idx = value.get("source_frame_idx")
        if (
            not isinstance(batch_id, str)
            or not batch_id
            or not isinstance(video_id, str)
            or not video_id
            or isinstance(source_frame_idx, bool)
            or not isinstance(source_frame_idx, int)
            or source_frame_idx < 0
        ):
            return None
        return f"{batch_id}:{video_id}:{source_frame_idx}"

    @classmethod
    def _legacy_window_id(cls, value: Mapping[str, Any]) -> str | None:
        window_id = value.get("window_id")
        if isinstance(window_id, str) and window_id:
            return window_id
        return cls._legacy_target_key(value)

    @staticmethod
    def _legacy_failure(value: Any) -> tuple[str, str, str]:
        """Map old or unknown failure values to the stable public taxonomy."""

        return _STABLE_FAILURES.get(
            str(value),
            ("internal_failed", "internal_reasoning_error", "internal"),
        )

    @classmethod
    def _legacy_prediction_indexes(
        cls, state: Mapping[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        by_window: dict[str, dict[str, Any]] = {}
        by_target: dict[str, dict[str, Any]] = {}
        predictions = state.get("predictions", ())
        if not isinstance(predictions, (list, tuple)):
            return by_window, by_target
        ordered = sorted(
            (dict(value) for value in predictions if isinstance(value, Mapping)),
            key=lambda item: (
                int(item.get("rank", 1_000_000))
                if isinstance(item.get("rank"), int) and not isinstance(item.get("rank"), bool)
                else 1_000_000
            ),
        )
        for prediction in ordered:
            window_id = prediction.get("window_id")
            if isinstance(window_id, str) and window_id:
                by_window.setdefault(window_id, prediction)
            evidence = prediction.get("evidence")
            representative = (
                evidence.get("representative_source_frame_idx")
                if isinstance(evidence, Mapping)
                else None
            )
            identity = {
                "batch_id": prediction.get("batch_id"),
                "video_id": prediction.get("video_id"),
                "source_frame_idx": (
                    representative
                    if isinstance(representative, int)
                    and not isinstance(representative, bool)
                    and representative >= 0
                    else prediction.get("source_frame_idx")
                ),
            }
            target_key = cls._legacy_target_key(identity)
            if target_key is not None:
                by_target.setdefault(target_key, prediction)
        return by_window, by_target

    @classmethod
    def _legacy_frames(
        cls,
        identity: Mapping[str, Any],
        prediction: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Recover URI-free legacy window frames, falling back to one anchor."""

        batch_id = identity.get("batch_id")
        video_id = identity.get("video_id")
        if prediction is not None:
            batch_id = prediction.get("batch_id", batch_id)
            video_id = prediction.get("video_id", video_id)
            evidence = prediction.get("evidence")
            window_frames = evidence.get("window_frames") if isinstance(evidence, Mapping) else None
            recovered: list[dict[str, Any]] = []
            if isinstance(window_frames, (list, tuple)):
                for item in window_frames:
                    if not isinstance(item, Mapping):
                        continue
                    source_frame_idx = item.get("source_frame_idx")
                    timestamp_ms = item.get("timestamp_ms")
                    if (
                        not isinstance(batch_id, str)
                        or not batch_id
                        or not isinstance(video_id, str)
                        or not video_id
                        or isinstance(source_frame_idx, bool)
                        or not isinstance(source_frame_idx, int)
                        or source_frame_idx < 0
                        or isinstance(timestamp_ms, bool)
                        or not isinstance(timestamp_ms, int)
                        or timestamp_ms < 0
                    ):
                        continue
                    recovered.append(
                        {
                            "position": len(recovered) + 1,
                            "batch_id": batch_id,
                            "video_id": video_id,
                            "source_frame_idx": source_frame_idx,
                            "timestamp_ms": timestamp_ms,
                        }
                    )
                    if len(recovered) == 5:
                        break
            if recovered:
                return recovered

        source_frame_idx = identity.get("source_frame_idx")
        timestamp_ms = identity.get("timestamp_ms", 0)
        if (
            isinstance(batch_id, str)
            and batch_id
            and isinstance(video_id, str)
            and video_id
            and isinstance(source_frame_idx, int)
            and not isinstance(source_frame_idx, bool)
            and source_frame_idx >= 0
        ):
            if (
                isinstance(timestamp_ms, bool)
                or not isinstance(timestamp_ms, int)
                or timestamp_ms < 0
            ):
                timestamp_ms = 0
            return [
                {
                    "position": 1,
                    "batch_id": batch_id,
                    "video_id": video_id,
                    "source_frame_idx": source_frame_idx,
                    "timestamp_ms": timestamp_ms,
                }
            ]
        return []

    @classmethod
    def _legacy_identity(
        cls,
        target: Mapping[str, Any],
        prediction: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        identity = dict(target)
        if prediction is not None:
            for name in ("batch_id", "video_id", "source_frame_idx", "timestamp_ms"):
                identity.setdefault(name, prediction.get(name))
            evidence = prediction.get("evidence")
            if isinstance(evidence, Mapping):
                identity.setdefault(
                    "source_frame_idx",
                    evidence.get("representative_source_frame_idx"),
                )
                identity.setdefault("shot_id", evidence.get("shot_id"))
                identity.setdefault("retrieval_rank", evidence.get("retrieval_rank"))
                identity.setdefault("retrieval_score", evidence.get("retrieval_score"))
        identity.setdefault("timestamp_ms", 0)
        identity.setdefault("shot_id", "legacy")
        identity.setdefault("retrieval_rank", 1_000_000)
        identity.setdefault("retrieval_score", 0.0)
        identity["frames"] = cls._legacy_frames(identity, prediction)
        return identity

    @classmethod
    def _legacy_outcomes(
        cls, state: Mapping[str, Any], targets: Iterable[Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        target_values = [dict(value) for value in targets]
        predictions_by_window, predictions_by_target = cls._legacy_prediction_indexes(state)
        outcomes: dict[str, dict[str, Any]] = {}
        results = state.get("results", {})
        result_values = dict(results) if isinstance(results, Mapping) else {}
        failures = state.get("failures", {})
        failure_values = dict(failures) if isinstance(failures, Mapping) else {}

        for target in target_values:
            target_key = cls._legacy_target_key(target)
            window_id = cls._legacy_window_id(target)
            if target_key is None or window_id is None:
                continue
            prediction = predictions_by_window.get(window_id) or predictions_by_target.get(
                target_key
            )
            identity = cls._legacy_identity(target, prediction)
            result_value = result_values.get(target_key, result_values.get(window_id))
            if isinstance(result_value, Mapping):
                nested = result_value.get("reasoning")
                reasoning = nested if isinstance(nested, Mapping) else result_value
                answerable = bool(reasoning.get("answerable", result_value.get("answer")))
                answer = str(
                    reasoning.get("normalized_answer")
                    or reasoning.get("answer")
                    or result_value.get("answer")
                    or ""
                )
                outcomes[window_id] = {
                    "window_id": window_id,
                    **identity,
                    "status": "answered" if answerable else "unanswerable",
                    "answerable": answerable,
                    "answer": answer if answerable else "",
                    "raw_answer": str(reasoning.get("raw_answer") or ""),
                    "normalized_answer": answer if answerable else "",
                    "evidence_frame_indices": list(reasoning.get("evidence_frame_indices", ())),
                    "provider": reasoning.get("provider"),
                    "model_name": reasoning.get("model_name"),
                    "prompt_version": reasoning.get("prompt_version"),
                    "provider_latency_ms": reasoning.get("latency_ms"),
                    "attempts": reasoning.get("attempts"),
                    "cache_status": reasoning.get("cache_status"),
                }
                continue
            if prediction is not None and prediction.get("answer") not in (None, ""):
                evidence = prediction.get("evidence")
                evidence = evidence if isinstance(evidence, Mapping) else {}
                answer = str(prediction.get("answer"))
                outcomes[window_id] = {
                    "window_id": window_id,
                    **identity,
                    "status": "answered",
                    "answerable": True,
                    "answer": answer,
                    "raw_answer": str(prediction.get("raw_answer") or ""),
                    "normalized_answer": answer,
                    "evidence_frame_indices": list(evidence.get("evidence_frame_indices", ())),
                    "provider": evidence.get("provider"),
                    "model_name": prediction.get("model_name"),
                    "provider_latency_ms": evidence.get("latency_ms"),
                    "attempts": evidence.get("attempts"),
                    "cache_status": evidence.get("cache_status"),
                }
                continue
            failure_value = failure_values.get(target_key, failure_values.get(window_id))
            if failure_value is not None:
                status, code, stage = cls._legacy_failure(failure_value)
                outcomes[window_id] = {
                    "window_id": window_id,
                    **identity,
                    "status": status,
                    "failure_code": code,
                    "failure_stage": stage,
                }
        return outcomes

    @classmethod
    def _normalize_legacy_outcome(
        cls,
        outcome: Mapping[str, Any],
        *,
        target: Mapping[str, Any] | None,
        prediction: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        value = dict(outcome)
        identity = cls._legacy_identity(target or value, prediction)
        for name, item in identity.items():
            if name == "frames":
                if not isinstance(value.get("frames"), (list, tuple)) or not value.get("frames"):
                    value["frames"] = item
            else:
                value.setdefault(name, item)
        status = str(value.get("status", "pending"))
        if status.endswith("_failed") or value.get("failure_code") is not None:
            normalized_status, code, stage = cls._legacy_failure(value.get("failure_code"))
            value["status"] = normalized_status
            value["failure_code"] = code
            value["failure_stage"] = stage
        return value

    @classmethod
    def _snapshot_state(
        cls,
        state: Mapping[str, Any],
        targets: Iterable[Mapping[str, Any]],
        windows: Iterable[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int | str]]:
        value = dict(state)
        target_values = [dict(target) for target in targets]
        predictions_by_window, predictions_by_target = cls._legacy_prediction_indexes(value)
        targets_by_id: dict[str, dict[str, Any]] = {}
        target_order: list[str] = []
        for target in target_values:
            window_id = cls._legacy_window_id(target)
            if window_id is None:
                continue
            targets_by_id[window_id] = target
            target_order.append(window_id)
        raw_outcomes = value.get("window_outcomes")
        if isinstance(raw_outcomes, Mapping):
            raw_values = [
                (str(key), dict(outcome))
                for key, outcome in raw_outcomes.items()
                if isinstance(outcome, Mapping)
            ]
        elif isinstance(raw_outcomes, list):
            raw_values = [
                (str(outcome.get("window_id")), dict(outcome))
                for outcome in raw_outcomes
                if isinstance(outcome, Mapping) and outcome.get("window_id")
            ]
        else:
            raw_values = list(cls._legacy_outcomes(value, target_values).items())
        if not raw_values and any(
            value.get(name) for name in ("results", "failures", "predictions")
        ):
            raw_values = list(cls._legacy_outcomes(value, target_values).items())
        outcomes_by_id: dict[str, dict[str, Any]] = {}
        for raw_key, raw_outcome in raw_values:
            window_id = str(raw_outcome.get("window_id") or raw_key)
            target = targets_by_id.get(window_id)
            target_key = cls._legacy_target_key(target or {})
            prediction = predictions_by_window.get(window_id)
            if prediction is None and target_key is not None:
                prediction = predictions_by_target.get(target_key)
            normalized = cls._normalize_legacy_outcome(
                raw_outcome,
                target=target,
                prediction=prediction,
            )
            normalized["window_id"] = window_id
            outcomes_by_id[window_id] = normalized
        window_values = [dict(window) for window in windows]
        ordered_ids = (
            [str(window.get("window_id")) for window in window_values]
            if window_values
            else target_order
        )
        ordered = [outcomes_by_id.pop(key) for key in ordered_ids if key in outcomes_by_id]
        ordered.extend(
            sorted(
                outcomes_by_id.values(),
                key=lambda item: (
                    int(item.get("retrieval_rank", 1_000_000)),
                    str(item.get("window_id", "")),
                ),
            )
        )
        statuses = [str(outcome.get("status", "pending")) for outcome in ordered]
        answered = statuses.count("answered")
        unanswerable = statuses.count("unanswerable")
        failed = sum(status.endswith("_failed") for status in statuses)
        pending = statuses.count("pending")
        reasoned = answered + unanswerable
        retrieved = int(value.get("retrieved_clusters", len(window_values) or len(ordered)))
        if pending:
            status = "incomplete"
        elif answered and failed:
            status = "partial"
        elif answered:
            status = "completed"
        elif unanswerable and failed:
            status = "partial"
        elif unanswerable:
            status = "completed_no_answer"
        elif failed:
            status = "failed"
        elif retrieved == 0:
            status = "no_candidates"
        else:
            status = "incomplete"
        value["state_version"] = 2
        value["window_outcomes"] = {
            str(outcome["window_id"]): outcome for outcome in ordered if outcome.get("window_id")
        }
        raw_failures = value.get("failures")
        if isinstance(raw_failures, Mapping):
            value["failures"] = {
                str(key): cls._legacy_failure(code)[1] for key, code in raw_failures.items()
            }
        counts: dict[str, int | str] = {
            "status": status,
            "reasoned_count": reasoned,
            "answerable_count": answered,
            "unanswerable_count": unanswerable,
            "failed_count": failed,
            "pending_count": pending,
        }
        return value, ordered, counts

    def get_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Return request, state, expiry, and cumulative targets for API restoration."""

        session = self.get_session(session_id)
        if session is None:
            return None
        if int(session.state.get("state_version", 0)) >= 3:
            candidates = self.get_session_candidates(session_id)
            state = dict(session.state)
            predictions = [
                dict(value)
                for value in state.get("predictions", ())
                if isinstance(value, Mapping)
            ]
            answers = state.get("answers", {})
            answer_values = dict(answers) if isinstance(answers, Mapping) else {}
            failures = state.get("failures", {})
            failure_values = dict(failures) if isinstance(failures, Mapping) else {}
            status = str(state.get("status") or "candidates_ready")
            return {
                "session_id": session.session_id,
                "request": session.request,
                "state": state,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "expires_at": session.expires_at,
                "revision": session.revision,
                "state_version": 3,
                "status": status,
                "session_candidates": list(candidates),
                "candidate_count": len(candidates),
                "predictions": predictions,
                "answers": answer_values,
                "failures": failure_values,
                "selected_candidate_id": state.get("selected_candidate_id"),
                "candidate_retrieval_ms": state.get("candidate_retrieval_ms", 0.0),
                "channel_statuses": dict(state.get("channel_statuses", {})),
                "retrieval_errors": dict(state.get("retrieval_errors", {})),
                "query_processing": dict(state.get("query_processing", {})),
                "operator_edits": list(self.get_operator_edits(session_id)),
            }
        targets = self.get_targets(session_id)
        windows = self.get_session_windows(session_id)
        state, outcomes, counts = self._snapshot_state(session.state, targets, windows)
        target_count = min(
            int(state.get("requested_target_clusters", len(targets))),
            int(state.get("retrieved_clusters", len(windows) or len(targets))),
        )
        raw_retrieval_errors = state.get("retrieval_errors", {})
        if isinstance(raw_retrieval_errors, Mapping):
            retrieval_errors: dict[str, str] | list[str] = {
                str(channel): str(code) for channel, code in raw_retrieval_errors.items()
            }
        elif isinstance(raw_retrieval_errors, (list, tuple)):
            retrieval_errors = [str(code) for code in raw_retrieval_errors]
        else:
            retrieval_errors = []
        raw_channel_statuses = state.get("channel_statuses", {})
        allowed_channel_fields = {
            "applicability",
            "state",
            "reason_code",
            "coverage",
            "count",
            "collection_name",
            "index_name",
            "expected_dimension",
            "remediation_id",
        }
        channel_statuses: dict[str, Any] = {}
        if isinstance(raw_channel_statuses, Mapping):
            for channel, channel_value in raw_channel_statuses.items():
                if isinstance(channel_value, Mapping):
                    channel_statuses[str(channel)] = {
                        str(key): value
                        for key, value in channel_value.items()
                        if key in allowed_channel_fields
                        and isinstance(value, (str, int, float, bool, type(None)))
                    }
                elif isinstance(channel_value, str):
                    channel_statuses[str(channel)] = {"state": channel_value}
        return {
            "session_id": session.session_id,
            "request": session.request,
            "state": state,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "expires_at": session.expires_at,
            "targets": list(targets),
            "target_count": target_count,
            "session_windows": list(windows),
            "window_outcomes": outcomes,
            "revision": session.revision,
            **counts,
            "requested_target_clusters": int(state.get("requested_target_clusters", target_count)),
            "retrieved_clusters": int(state.get("retrieved_clusters", target_count)),
            "retrieval_exhausted": bool(state.get("retrieval_exhausted", False)),
            "channel_statuses": channel_statuses,
            "retrieval_errors": retrieval_errors,
            "operator_edits": list(self.get_operator_edits(session_id)),
        }

    def record_operator_edit(
        self,
        session_id: str,
        window_id: str,
        edit: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append one audited answer edit to a live QA session."""

        if not isinstance(window_id, str) or not window_id.strip():
            raise ValueError("window_id must be a non-empty string")
        value = self._mapping(edit, "edit")
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            cursor = connection.execute(
                """
                INSERT INTO qa_operator_edits(session_id, window_id, edit_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    window_id,
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
            edit_id = int(cursor.lastrowid)
            connection.commit()
        return {
            "edit_id": edit_id,
            "session_id": session_id,
            "window_id": window_id,
            "created_at": now,
            **value,
        }

    def get_operator_edits(self, session_id: str) -> tuple[dict[str, Any], ...]:
        """Return the ordered answer-edit audit for a live QA session."""

        now = self.clock()
        with closing(self._connection()) as connection:
            live = connection.execute(
                "SELECT 1 FROM qa_sessions WHERE session_id = ? AND expires_at > ?",
                (session_id, now),
            ).fetchone()
            if live is None:
                raise QASessionStoreError("QA session is missing or expired")
            rows = connection.execute(
                """
                SELECT edit_id, window_id, edit_json, created_at
                FROM qa_operator_edits WHERE session_id = ? ORDER BY edit_id
                """,
                (session_id,),
            ).fetchall()
        return tuple(
            {
                "edit_id": int(row["edit_id"]),
                "session_id": session_id,
                "window_id": str(row["window_id"]),
                "created_at": float(row["created_at"]),
                **json.loads(row["edit_json"]),
            }
            for row in rows
        )

    def get_candidate_set(self, cache_key: str) -> tuple[dict[str, Any], ...] | None:
        """Return one non-expired, URI-free candidate set."""

        entry = self.get_candidate_entry(cache_key)
        return None if entry is None else entry[0]

    def get_candidate_entry(
        self, cache_key: str
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]] | None:
        """Return cached windows plus safe retrieval diagnostics."""

        now = self.clock()
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT windows_json, diagnostics_json FROM qa_candidate_sets
                WHERE cache_key = ? AND expires_at > ?
                """,
                (cache_key, now),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(row["windows_json"])
        if not isinstance(value, list):
            return None
        diagnostics = json.loads(row["diagnostics_json"] or "{}")
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        return (
            tuple(dict(item) for item in value if isinstance(item, Mapping)),
            dict(diagnostics),
        )

    def put_candidate_set(
        self,
        cache_key: str,
        windows: Iterable[Mapping[str, Any]],
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        """Upsert one URI-free cross-session candidate set."""

        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError("cache_key must be non-empty")
        values = [self._mapping(window, "window") for window in windows]
        for value in values:
            self._reject_media_locations(value, "window")
        diagnostic_value = self._mapping(diagnostics or {}, "candidate diagnostics")
        self._reject_media_locations(diagnostic_value, "candidate diagnostics")
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute(
                """
                INSERT INTO qa_candidate_sets
                    (cache_key, windows_json, diagnostics_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    windows_json = excluded.windows_json,
                    diagnostics_json = excluded.diagnostics_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    json.dumps(values, ensure_ascii=False, sort_keys=True),
                    json.dumps(diagnostic_value, ensure_ascii=False, sort_keys=True),
                    now,
                    now + self.candidate_ttl_seconds,
                ),
            )
            connection.commit()

    def get_image_fingerprint(self, cache_key: str) -> dict[str, Any] | None:
        """Return normalized image metadata without reading image bytes."""

        now = self.clock()
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT fingerprint_json FROM qa_image_fingerprints
                WHERE cache_key = ? AND expires_at > ?
                """,
                (cache_key, now),
            ).fetchone()
        return None if row is None else json.loads(row["fingerprint_json"])

    def get_image_fingerprints(self, cache_keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Batch-read normalized image metadata for cheap reasoning-cache probes."""

        keys = tuple(dict.fromkeys(str(key) for key in cache_keys if key))
        if not keys:
            return {}
        now = self.clock()
        placeholders = ",".join("?" for _ in keys)
        with closing(self._connection()) as connection:
            rows = connection.execute(
                f"""
                SELECT cache_key, fingerprint_json FROM qa_image_fingerprints
                WHERE cache_key IN ({placeholders}) AND expires_at > ?
                """,
                (*keys, now),
            ).fetchall()
        return {str(row["cache_key"]): json.loads(row["fingerprint_json"]) for row in rows}

    def put_image_fingerprint(self, cache_key: str, value: Mapping[str, Any]) -> None:
        """Persist hashes/dimensions only; callers must never pass image bytes or URIs."""

        self.put_image_fingerprints({cache_key: value})

    def put_image_fingerprints(self, values: Mapping[str, Mapping[str, Any]]) -> None:
        """Batch-upsert normalized image metadata in one SQLite transaction."""

        if not isinstance(values, Mapping):
            raise TypeError("image fingerprints must be a mapping")
        forbidden = {"data", "bytes", "image_uri", "video_uri", "path", "member"}
        prepared: list[tuple[str, dict[str, Any]]] = []
        for cache_key, value in values.items():
            if not isinstance(cache_key, str) or not cache_key:
                raise ValueError("cache_key must be non-empty")
            fingerprint = self._mapping(value, "image fingerprint")
            if forbidden & set(fingerprint):
                raise ValueError("image fingerprint contains forbidden media location or bytes")
            prepared.append((cache_key, fingerprint))
        if not prepared:
            return
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.executemany(
                """
                INSERT INTO qa_image_fingerprints
                    (cache_key, fingerprint_json, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    fingerprint_json = excluded.fingerprint_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                [
                    (
                        cache_key,
                        json.dumps(fingerprint, ensure_ascii=False, sort_keys=True),
                        now,
                        now + self.image_fingerprint_ttl_seconds,
                    )
                    for cache_key, fingerprint in prepared
                ],
            )
            connection.commit()

    def acquire_cache_lease(self, cache_key: str, owner_token: str, *, ttl_seconds: float) -> bool:
        """Acquire or take over an expired cross-process cache lease."""

        if not cache_key or not owner_token:
            raise ValueError("cache_key and owner_token must be non-empty")
        if ttl_seconds <= 0 or ttl_seconds > 600:
            raise ValueError("cache lease TTL must be in (0, 600]")
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM qa_cache_leases WHERE cache_key = ? AND expires_at <= ?",
                (cache_key, now),
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO qa_cache_leases(cache_key, owner_token, expires_at)
                VALUES (?, ?, ?)
                """,
                (cache_key, owner_token, now + float(ttl_seconds)),
            )
            connection.commit()
        return cursor.rowcount == 1

    def release_cache_lease(self, cache_key: str, owner_token: str) -> None:
        """Release a cache lease only when owned by the caller."""

        with closing(self._connection()) as connection:
            connection.execute(
                "DELETE FROM qa_cache_leases WHERE cache_key = ? AND owner_token = ?",
                (cache_key, owner_token),
            )
            connection.commit()

    def get_reasoning(self, cache_key: str) -> dict[str, Any] | None:
        """Return a non-expired cached reasoning result."""

        now = self.clock()
        with closing(self._connection()) as connection:
            row = connection.execute(
                "SELECT result_json FROM qa_reasoning_cache WHERE cache_key = ? AND expires_at > ?",
                (cache_key, now),
            ).fetchone()
        return None if row is None else json.loads(row["result_json"])

    def get_reasoning_many(self, cache_keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Batch-read non-expired reasoning results."""

        keys = tuple(dict.fromkeys(str(key) for key in cache_keys if key))
        if not keys:
            return {}
        now = self.clock()
        placeholders = ",".join("?" for _ in keys)
        with closing(self._connection()) as connection:
            rows = connection.execute(
                f"""
                SELECT cache_key, result_json FROM qa_reasoning_cache
                WHERE cache_key IN ({placeholders}) AND expires_at > ?
                """,
                (*keys, now),
            ).fetchall()
        return {str(row["cache_key"]): json.loads(row["result_json"]) for row in rows}

    def put_reasoning(self, cache_key: str, result: Mapping[str, Any]) -> None:
        """Upsert one reasoning result with a fresh 30-day TTL."""

        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError("cache_key must be non-empty")
        value = self._mapping(result, "result")
        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute(
                """
                INSERT INTO qa_reasoning_cache(cache_key, result_json, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    result_json = excluded.result_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    now,
                    now + self.reasoning_ttl_seconds,
                ),
            )
            connection.commit()

    def purge_expired(self) -> dict[str, int]:
        """Explicitly remove expired sessions/targets and reasoning entries."""

        now = self.clock()
        with closing(self._connection()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            sessions = connection.execute(
                "DELETE FROM qa_sessions WHERE expires_at <= ?", (now,)
            ).rowcount
            reasoning = connection.execute(
                "DELETE FROM qa_reasoning_cache WHERE expires_at <= ?", (now,)
            ).rowcount
            connection.execute("DELETE FROM qa_candidate_sets WHERE expires_at <= ?", (now,))
            connection.execute("DELETE FROM qa_image_fingerprints WHERE expires_at <= ?", (now,))
            connection.execute("DELETE FROM qa_cache_leases WHERE expires_at <= ?", (now,))
            connection.commit()
        return {"sessions": sessions, "reasoning": reasoning}


__all__ = [
    "QASession",
    "QASessionStore",
    "QASessionStoreError",
    "reasoning_cache_key",
]
