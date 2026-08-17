"""Cross-process lock for one native-ingestion batch."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from types import TracebackType


class BatchIngestLockError(RuntimeError):
    """Raised when another process owns the same batch lock."""


class BatchIngestLock:
    """Serialize writers for one batch without connecting during construction."""

    def __init__(
        self,
        root: str | Path,
        batch_id: str,
        *,
        timeout_seconds: float = 0.0,
        poll_seconds: float = 0.1,
    ) -> None:
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        if timeout_seconds < 0 or poll_seconds <= 0:
            raise ValueError("lock timing values must be non-negative")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", batch_id.strip()).strip("._-") or "batch"
        digest = hashlib.sha256(batch_id.strip().encode("utf-8")).hexdigest()[:10]
        self.path = Path(root) / f"{safe}-{digest}.lock"
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self._handle = None

    @staticmethod
    def _try_lock(handle) -> bool:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    @staticmethod
    def _unlock(handle) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __enter__(self) -> BatchIngestLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + self.timeout_seconds
        while not self._try_lock(handle):
            if time.monotonic() >= deadline:
                handle.close()
                raise BatchIngestLockError("batch_ingest_lock_unavailable")
            time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))
        self._handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                self._unlock(handle)
            finally:
                handle.close()


__all__ = ["BatchIngestLock", "BatchIngestLockError"]
