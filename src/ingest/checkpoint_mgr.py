"""Atomic, sanitized mirrors of authoritative ingestion ledger progress."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CheckpointError(RuntimeError):
    """Raised when a checkpoint mirror is malformed or cannot be published."""


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """Safe progress projection; never used as the source of resume decisions."""

    schema_version: int
    batch_id: str
    stage: str
    fingerprint: str
    completed_video_ids: tuple[str, ...]
    failed_video_ids: tuple[str, ...]
    source_missing_video_ids: tuple[str, ...]
    counts: Mapping[str, int]
    timings_ms: Mapping[str, float]
    resources: Mapping[str, int | float | None]
    state: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe representation."""

        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "stage": self.stage,
            "fingerprint": self.fingerprint,
            "state": self.state,
            "completed_video_ids": list(self.completed_video_ids),
            "failed_video_ids": list(self.failed_video_ids),
            "source_missing_video_ids": list(self.source_missing_video_ids),
            "counts": dict(self.counts),
            "timings_ms": dict(self.timings_ms),
            "resources": dict(self.resources),
        }


class CheckpointMirror:
    """Write one atomic JSON mirror below an operator-selected generated directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _filename(stage: str) -> str:
        if stage not in {"ocr", "siglip2", "asr"}:
            raise ValueError("unsupported provisioning checkpoint stage")
        return f"b01_{'siglip' if stage == 'siglip2' else stage}_progress.json"

    def path_for(self, stage: str) -> Path:
        """Return the generated-state destination for one stage."""

        return self.root / self._filename(stage)

    def publish(self, snapshot: CheckpointSnapshot) -> Path:
        """Atomically publish a mirror without exposing absolute source paths."""

        if snapshot.schema_version != 1:
            raise CheckpointError("unsupported checkpoint schema")
        if not snapshot.batch_id or not snapshot.fingerprint:
            raise CheckpointError("checkpoint identity is incomplete")
        payload = snapshot.to_dict()
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        target = self.path_for(snapshot.stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as error:
            raise CheckpointError("checkpoint publish failed") from error
        finally:
            temporary.unlink(missing_ok=True)
        return target


__all__ = ["CheckpointError", "CheckpointMirror", "CheckpointSnapshot"]
