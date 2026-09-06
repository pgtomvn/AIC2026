"""Atomic machine-local staging state for Codabench submission entries."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping, TypeAlias, Union

from src.codabench_query_id import (
    codabench_query_id_collision_key,
    parse_codabench_query_filename,
)
from src.export.preflight import BundleEntry
from src.ingest.batch_lock import BatchIngestLock, BatchIngestLockError
from src.utils.io_helpers import atomic_write_bytes

STAGING_SCHEMA_VERSION = 1
MAX_STAGING_ENTRIES = 256
MAX_ENTRY_BYTES = 1_000_000
MAX_TOTAL_BYTES = 8_000_000
MAX_DOCUMENT_BYTES = 16_000_000
MAX_STAGING_ROWS = 100

StagingRowAction = Literal["append", "update", "delete", "move"]


class StagingStoreError(RuntimeError):
    """Stable operational failure that does not expose a filesystem path."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class StagingRevisionConflict(StagingStoreError):
    """Raised when a browser tries to update an obsolete staging revision."""

    def __init__(self) -> None:
        super().__init__("staging_revision_conflict")


StagingEntryValidator: TypeAlias = Callable[[BundleEntry], Union[int, "StagedEntry"]]


def _utc_timestamp(clock: Callable[[], datetime]) -> str:
    value = clock().astimezone(timezone.utc).replace(microsecond=0)
    return value.isoformat().replace("+00:00", "Z")


def _require_timestamp(value: object) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise StagingStoreError("staging_store_invalid")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise StagingStoreError("staging_store_invalid") from None
    return value


def _csv_records(data: bytes) -> tuple[bytes, ...]:
    """Split UTF-8 CSV into exact logical records without changing their bytes."""

    if not data or data.startswith(b"\xef\xbb\xbf") or not data.endswith(b"\n"):
        raise StagingStoreError("staging_row_invalid")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise StagingStoreError("staging_row_invalid") from None
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    physical_lines: list[int] = []
    try:
        for row in reader:
            if not row:
                raise StagingStoreError("staging_row_invalid")
            physical_lines.append(reader.line_num)
    except csv.Error:
        raise StagingStoreError("staging_row_invalid") from None
    text_lines = text.splitlines(keepends=True)
    records: list[bytes] = []
    previous_line = 0
    for line_number in physical_lines:
        record = "".join(text_lines[previous_line:line_number]).encode("utf-8")
        if not record:
            raise StagingStoreError("staging_row_invalid")
        records.append(record)
        previous_line = line_number
    if previous_line != len(text_lines) or b"".join(records) != data:
        raise StagingStoreError("staging_row_invalid")
    return tuple(records)


def _rank(value: object, *, upper_bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > upper_bound:
        raise StagingStoreError("staging_rank_invalid")
    return value


@dataclass(frozen=True, slots=True)
class StagedEntry:
    """One validated immutable CSV persisted in the active staging store."""

    bundle_entry: BundleEntry
    row_count: int
    sha256: str
    updated_at_utc: str

    @classmethod
    def create(
        cls,
        entry: BundleEntry,
        *,
        row_count: int,
        clock: Callable[[], datetime],
    ) -> StagedEntry:
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise StagingStoreError("staging_entry_invalid")
        if len(entry.csv_bytes) > MAX_ENTRY_BYTES:
            raise StagingStoreError("staging_size_limit_exceeded")
        try:
            actual_row_count = len(_csv_records(entry.csv_bytes))
        except StagingStoreError:
            raise StagingStoreError("staging_entry_invalid") from None
        if actual_row_count != row_count or row_count > MAX_STAGING_ROWS:
            raise StagingStoreError("staging_entry_invalid")
        return cls(
            bundle_entry=entry,
            row_count=row_count,
            sha256=hashlib.sha256(entry.csv_bytes).hexdigest(),
            updated_at_utc=_utc_timestamp(clock),
        )

    @classmethod
    def from_disk(cls, filename: str, value: object) -> StagedEntry:
        if not isinstance(value, Mapping):
            raise StagingStoreError("staging_store_invalid")
        try:
            query_id, task = parse_codabench_query_filename(filename)
        except (TypeError, ValueError):
            raise StagingStoreError("staging_store_invalid") from None
        csv_content = value.get("csv_content")
        if not isinstance(csv_content, str):
            raise StagingStoreError("staging_store_invalid")
        try:
            csv_bytes = csv_content.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise StagingStoreError("staging_store_invalid") from None
        if not csv_bytes or len(csv_bytes) > MAX_ENTRY_BYTES:
            raise StagingStoreError("staging_store_invalid")
        ranks = value.get("source_ranks")
        try:
            entry = BundleEntry(
                task=task,
                query_id=query_id,
                csv_bytes=csv_bytes,
                batch_id=value.get("batch_id"),  # type: ignore[arg-type]
                source_fingerprint=value.get("source_fingerprint"),  # type: ignore[arg-type]
                stage_count=value.get("stage_count"),  # type: ignore[arg-type]
                source_ranks=None if ranks is None else tuple(ranks),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            raise StagingStoreError("staging_store_invalid") from None
        row_count = value.get("row_count")
        digest = value.get("sha256")
        try:
            actual_row_count = len(_csv_records(csv_bytes))
        except StagingStoreError:
            raise StagingStoreError("staging_store_invalid") from None
        if (
            value.get("task") != task
            or value.get("query_id") != query_id
            or value.get("status") != "READY"
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
            or row_count > MAX_STAGING_ROWS
            or row_count != actual_row_count
            or not isinstance(digest, str)
            or digest != hashlib.sha256(csv_bytes).hexdigest()
        ):
            raise StagingStoreError("staging_store_invalid")
        return cls(entry, row_count, digest, _require_timestamp(value.get("updated_at_utc")))

    @property
    def filename(self) -> str:
        return self.bundle_entry.filename

    def to_disk(self) -> dict[str, object]:
        entry = self.bundle_entry
        return {
            "task": entry.task,
            "query_id": entry.query_id,
            "batch_id": entry.batch_id,
            "status": "READY",
            "row_count": self.row_count,
            "stage_count": entry.stage_count,
            "source_fingerprint": entry.source_fingerprint,
            "source_ranks": None if entry.source_ranks is None else list(entry.source_ranks),
            "sha256": self.sha256,
            "csv_content": entry.csv_bytes.decode("utf-8", errors="strict"),
            "updated_at_utc": self.updated_at_utc,
        }

    def to_api(self) -> dict[str, object]:
        value = self.to_disk()
        value.pop("csv_content")
        value["filename"] = self.filename
        value["csv_base64"] = base64.b64encode(self.bundle_entry.csv_bytes).decode("ascii")
        return value


@dataclass(frozen=True, slots=True)
class StagingSnapshot:
    revision: int
    updated_at_utc: str | None
    entries: tuple[StagedEntry, ...]

    def to_api(self) -> dict[str, object]:
        return {
            "schema_version": STAGING_SCHEMA_VERSION,
            "revision": self.revision,
            "updated_at_utc": self.updated_at_utc,
            "entries": {entry.filename: entry.to_api() for entry in self.entries},
        }


class StagingStore:
    """Persist a single revisioned staging set without touching disk on construction."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        lock_timeout_seconds: float = 10.0,
    ) -> None:
        self.path = Path(path)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lock_timeout_seconds = float(lock_timeout_seconds)

    def _empty(self) -> StagingSnapshot:
        return StagingSnapshot(0, None, ())

    def _read_unlocked(self) -> StagingSnapshot:
        if not self.path.exists():
            return self._empty()
        try:
            if self.path.stat().st_size > MAX_DOCUMENT_BYTES:
                raise StagingStoreError("staging_store_invalid")
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except StagingStoreError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise StagingStoreError("staging_store_invalid") from None
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise StagingStoreError("staging_store_invalid")
        revision = value.get("revision")
        raw_entries = value.get("entries")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
            or not isinstance(raw_entries, Mapping)
            or len(raw_entries) > MAX_STAGING_ENTRIES
        ):
            raise StagingStoreError("staging_store_invalid")
        updated = value.get("updated_at_utc")
        if updated is not None:
            updated = _require_timestamp(updated)
        entries = tuple(
            StagedEntry.from_disk(filename, raw_entries[filename])
            for filename in sorted(raw_entries)
            if isinstance(filename, str)
        )
        if len(entries) != len(raw_entries):
            raise StagingStoreError("staging_store_invalid")
        total = sum(len(entry.bundle_entry.csv_bytes) for entry in entries)
        collision_keys = [
            codabench_query_id_collision_key(entry.bundle_entry.query_id) for entry in entries
        ]
        if total > MAX_TOTAL_BYTES or len(collision_keys) != len(set(collision_keys)):
            raise StagingStoreError("staging_store_invalid")
        return StagingSnapshot(revision, updated, entries)

    def load(self) -> StagingSnapshot:
        """Return the latest complete snapshot; atomic replacement makes reads lock-free."""

        return self._read_unlocked()

    def _write_unlocked(self, snapshot: StagingSnapshot) -> StagingSnapshot:
        document = {
            "schema_version": STAGING_SCHEMA_VERSION,
            "revision": snapshot.revision,
            "updated_at_utc": snapshot.updated_at_utc,
            "entries": {entry.filename: entry.to_disk() for entry in snapshot.entries},
        }
        data = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if len(data) > MAX_DOCUMENT_BYTES:
            raise StagingStoreError("staging_size_limit_exceeded")
        try:
            atomic_write_bytes(self.path, data)
        except OSError:
            raise StagingStoreError("staging_store_unavailable") from None
        return snapshot

    def _locked(self):
        return BatchIngestLock(
            self.path.parent / ".locks",
            "codabench-active-staging",
            timeout_seconds=self.lock_timeout_seconds,
        )

    @staticmethod
    def _assert_revision(snapshot: StagingSnapshot, expected_revision: int) -> None:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
            or snapshot.revision != expected_revision
        ):
            raise StagingRevisionConflict()

    def upsert(
        self,
        entry: StagedEntry,
        *,
        expected_revision: int,
    ) -> StagingSnapshot:
        if not isinstance(entry, StagedEntry):
            raise StagingStoreError("staging_entry_invalid")
        try:
            actual_row_count = len(_csv_records(entry.bundle_entry.csv_bytes))
        except StagingStoreError:
            raise StagingStoreError("staging_entry_invalid") from None
        if (
            actual_row_count != entry.row_count
            or entry.row_count > MAX_STAGING_ROWS
            or entry.sha256 != hashlib.sha256(entry.bundle_entry.csv_bytes).hexdigest()
        ):
            raise StagingStoreError("staging_entry_invalid")
        try:
            with self._locked():
                current = self._read_unlocked()
                self._assert_revision(current, expected_revision)
                by_name = {item.filename: item for item in current.entries}
                collision = codabench_query_id_collision_key(entry.bundle_entry.query_id)
                for filename, existing in by_name.items():
                    if filename == entry.filename:
                        continue
                    if (
                        filename.casefold() == entry.filename.casefold()
                        or codabench_query_id_collision_key(existing.bundle_entry.query_id)
                        == collision
                    ):
                        raise StagingStoreError("bundle_query_id_collision")
                by_name[entry.filename] = entry
                if len(by_name) > MAX_STAGING_ENTRIES:
                    raise StagingStoreError("staging_size_limit_exceeded")
                total = sum(len(item.bundle_entry.csv_bytes) for item in by_name.values())
                if total > MAX_TOTAL_BYTES:
                    raise StagingStoreError("staging_size_limit_exceeded")
                timestamp = _utc_timestamp(self.clock)
                updated_entry = StagedEntry(
                    entry.bundle_entry,
                    entry.row_count,
                    entry.sha256,
                    timestamp,
                )
                by_name[entry.filename] = updated_entry
                return self._write_unlocked(
                    StagingSnapshot(
                        current.revision + 1,
                        timestamp,
                        tuple(by_name[name] for name in sorted(by_name)),
                    )
                )
        except BatchIngestLockError:
            raise StagingStoreError("staging_store_unavailable") from None

    def _validate_mutated_entry(
        self,
        existing: StagedEntry,
        csv_bytes: bytes,
        *,
        validator: StagingEntryValidator,
        source_fingerprint: str | None,
        updated_at_utc: str,
    ) -> StagedEntry:
        records = _csv_records(csv_bytes)
        if len(records) > MAX_STAGING_ROWS:
            raise StagingStoreError("staging_row_limit_exceeded")
        if len(csv_bytes) > MAX_ENTRY_BYTES:
            raise StagingStoreError("staging_size_limit_exceeded")
        fingerprint = source_fingerprint or hashlib.sha256(csv_bytes).hexdigest()
        previous = existing.bundle_entry
        try:
            candidate = BundleEntry(
                task=previous.task,
                query_id=previous.query_id,
                csv_bytes=csv_bytes,
                batch_id=previous.batch_id,
                source_fingerprint=fingerprint,
                stage_count=previous.stage_count,
                source_ranks=None,
            )
        except (TypeError, ValueError):
            raise StagingStoreError("staging_entry_invalid") from None
        try:
            validation = validator(candidate)
        except StagingStoreError:
            raise
        if isinstance(validation, StagedEntry):
            if (
                validation.bundle_entry != candidate
                or validation.row_count != len(records)
                or validation.sha256 != hashlib.sha256(csv_bytes).hexdigest()
            ):
                raise StagingStoreError("staging_entry_validation_invalid")
            row_count = validation.row_count
        elif isinstance(validation, bool) or not isinstance(validation, int):
            raise StagingStoreError("staging_entry_validation_invalid")
        else:
            row_count = validation
        if row_count != len(records):
            raise StagingStoreError("staging_entry_validation_invalid")
        return StagedEntry(
            candidate,
            row_count,
            hashlib.sha256(csv_bytes).hexdigest(),
            updated_at_utc,
        )

    def mutate_rows(
        self,
        filename: str,
        *,
        action: StagingRowAction,
        expected_revision: int,
        validator: StagingEntryValidator,
        row_csv_bytes: bytes | None = None,
        rank: int | None = None,
        target_rank: int | None = None,
        source_fingerprint: str | None = None,
    ) -> StagingSnapshot:
        """Atomically append, update, delete, or move a QA/TRAKE CSV row.

        Ranks are one-based and exist only as physical CSV row order. The
        required validator runs while the cross-process lock is held and must
        return the validated row count (or a matching :class:`StagedEntry`).
        """

        if action not in {"append", "update", "delete", "move"}:
            raise StagingStoreError("staging_row_action_invalid")
        if not callable(validator):
            raise StagingStoreError("staging_entry_validation_invalid")
        try:
            _, task = parse_codabench_query_filename(filename)
        except (TypeError, ValueError):
            raise StagingStoreError("staging_entry_invalid") from None
        if task not in {"qa", "trake"}:
            raise StagingStoreError("staging_row_mutation_unsupported")

        try:
            with self._locked():
                current = self._read_unlocked()
                self._assert_revision(current, expected_revision)
                by_name = {item.filename: item for item in current.entries}
                existing = by_name.get(filename)
                if existing is None:
                    raise StagingStoreError("staging_entry_not_found")
                if existing.bundle_entry.task != task:
                    raise StagingStoreError("staging_store_invalid")
                records = list(_csv_records(existing.bundle_entry.csv_bytes))
                replacement: bytes | None = None
                if action in {"append", "update"}:
                    if not isinstance(row_csv_bytes, bytes):
                        raise StagingStoreError("staging_row_invalid")
                    incoming = _csv_records(row_csv_bytes)
                    if len(incoming) != 1:
                        raise StagingStoreError("staging_row_invalid")
                    replacement = incoming[0]
                elif row_csv_bytes is not None:
                    raise StagingStoreError("staging_row_invalid")

                if action == "append":
                    if rank is not None or target_rank is not None:
                        raise StagingStoreError("staging_rank_invalid")
                    if len(records) >= MAX_STAGING_ROWS:
                        raise StagingStoreError("staging_row_limit_exceeded")
                    records.append(replacement or b"")
                elif action == "update":
                    selected = _rank(rank, upper_bound=len(records))
                    if target_rank is not None:
                        raise StagingStoreError("staging_rank_invalid")
                    records[selected - 1] = replacement or b""
                elif action == "delete":
                    selected = _rank(rank, upper_bound=len(records))
                    if target_rank is not None:
                        raise StagingStoreError("staging_rank_invalid")
                    records.pop(selected - 1)
                else:
                    selected = _rank(rank, upper_bound=len(records))
                    destination = _rank(target_rank, upper_bound=len(records))
                    if selected == destination:
                        return current
                    record = records.pop(selected - 1)
                    records.insert(destination - 1, record)

                timestamp = _utc_timestamp(self.clock)
                if not records:
                    del by_name[filename]
                else:
                    csv_bytes = b"".join(records)
                    if csv_bytes == existing.bundle_entry.csv_bytes:
                        return current
                    by_name[filename] = self._validate_mutated_entry(
                        existing,
                        csv_bytes,
                        validator=validator,
                        source_fingerprint=source_fingerprint,
                        updated_at_utc=timestamp,
                    )
                    total = sum(len(item.bundle_entry.csv_bytes) for item in by_name.values())
                    if total > MAX_TOTAL_BYTES:
                        raise StagingStoreError("staging_size_limit_exceeded")
                return self._write_unlocked(
                    StagingSnapshot(
                        current.revision + 1,
                        timestamp,
                        tuple(by_name[name] for name in sorted(by_name)),
                    )
                )
        except BatchIngestLockError:
            raise StagingStoreError("staging_store_unavailable") from None

    def append_row(
        self,
        filename: str,
        row_csv_bytes: bytes,
        *,
        expected_revision: int,
        validator: StagingEntryValidator,
        source_fingerprint: str | None = None,
    ) -> StagingSnapshot:
        """Append one validated row as the next competition rank."""

        return self.mutate_rows(
            filename,
            action="append",
            expected_revision=expected_revision,
            validator=validator,
            row_csv_bytes=row_csv_bytes,
            source_fingerprint=source_fingerprint,
        )

    def update_row(
        self,
        filename: str,
        rank: int,
        row_csv_bytes: bytes,
        *,
        expected_revision: int,
        validator: StagingEntryValidator,
        source_fingerprint: str | None = None,
    ) -> StagingSnapshot:
        """Replace exactly one one-based rank after whole-entry validation."""

        return self.mutate_rows(
            filename,
            action="update",
            expected_revision=expected_revision,
            validator=validator,
            row_csv_bytes=row_csv_bytes,
            rank=rank,
            source_fingerprint=source_fingerprint,
        )

    def delete_row(
        self,
        filename: str,
        rank: int,
        *,
        expected_revision: int,
        validator: StagingEntryValidator,
    ) -> StagingSnapshot:
        """Delete one rank; deleting the last row removes the staged entry."""

        return self.mutate_rows(
            filename,
            action="delete",
            expected_revision=expected_revision,
            validator=validator,
            rank=rank,
        )

    def move_row(
        self,
        filename: str,
        rank: int,
        target_rank: int,
        *,
        expected_revision: int,
        validator: StagingEntryValidator,
    ) -> StagingSnapshot:
        """Move a row to a new one-based rank and shift intervening rows."""

        return self.mutate_rows(
            filename,
            action="move",
            expected_revision=expected_revision,
            validator=validator,
            rank=rank,
            target_rank=target_rank,
        )

    def delete(self, filename: str, *, expected_revision: int) -> StagingSnapshot:
        try:
            parse_codabench_query_filename(filename)
        except (TypeError, ValueError):
            raise StagingStoreError("staging_entry_invalid") from None
        try:
            with self._locked():
                current = self._read_unlocked()
                self._assert_revision(current, expected_revision)
                remaining = tuple(entry for entry in current.entries if entry.filename != filename)
                if len(remaining) == len(current.entries):
                    return current
                timestamp = _utc_timestamp(self.clock)
                return self._write_unlocked(
                    StagingSnapshot(current.revision + 1, timestamp, remaining)
                )
        except BatchIngestLockError:
            raise StagingStoreError("staging_store_unavailable") from None

    def clear(self, *, expected_revision: int) -> StagingSnapshot:
        try:
            with self._locked():
                current = self._read_unlocked()
                self._assert_revision(current, expected_revision)
                timestamp = _utc_timestamp(self.clock)
                return self._write_unlocked(StagingSnapshot(current.revision + 1, timestamp, ()))
        except BatchIngestLockError:
            raise StagingStoreError("staging_store_unavailable") from None


__all__ = [
    "MAX_ENTRY_BYTES",
    "MAX_STAGING_ENTRIES",
    "MAX_STAGING_ROWS",
    "MAX_TOTAL_BYTES",
    "STAGING_SCHEMA_VERSION",
    "StagedEntry",
    "StagingEntryValidator",
    "StagingRevisionConflict",
    "StagingRowAction",
    "StagingSnapshot",
    "StagingStore",
    "StagingStoreError",
]
