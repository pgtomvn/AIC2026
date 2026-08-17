"""Atomic Codabench #10187 bundle creation and audit manifests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from src.export.preflight import (
    CODABENCH_SPEC_ID,
    BundleEntry,
    BundleMode,
    BundleValidationContext,
    ValidatedBundleEntry,
    validate_bundle_entries,
)
from src.ingest.batch_lock import BatchIngestLock, BatchIngestLockError
from src.utils.io_helpers import atomic_write_bytes


class BundleBuildError(RuntimeError):
    """Stable operational failure raised without exposing filesystem paths."""


@dataclass(frozen=True, slots=True)
class BundleArtifact:
    """Published bundle path plus its sanitized immutable manifest."""

    path: Path
    manifest: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.manifest)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _timestamp(now: datetime) -> str:
    value = now.astimezone(timezone.utc)
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_bytes(entries: Sequence[ValidatedBundleEntry]) -> bytes:
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            directory = zipfile.ZipInfo("submission/")
            directory.external_attr = (0o40755 & 0xFFFF) << 16
            archive.writestr(directory, b"")
            for validated in entries:
                archive.writestr(
                    f"submission/{validated.entry.filename}",
                    validated.entry.csv_bytes,
                )
        handle.seek(0)
        data = handle.read()
    expected = ["submission/", *(f"submission/{item.entry.filename}" for item in entries)]
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        if archive.namelist() != expected or archive.testzip() is not None:
            raise BundleBuildError("bundle_integrity_check_failed")
    return data


def _root_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"schema_version": 1, "bundles": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BundleBuildError("bundle_manifest_invalid") from None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or not isinstance(value.get("bundles"), list)
    ):
        raise BundleBuildError("bundle_manifest_invalid")
    return value


def build_codabench_bundle(
    entries: Sequence[BundleEntry],
    output_dir: str | Path,
    *,
    validation_context: BundleValidationContext,
    mode: BundleMode = "all",
    now: Callable[[], datetime] | None = None,
) -> BundleArtifact:
    """Validate, publish, checksum, and record one immutable Codabench bundle."""

    validated = validate_bundle_entries(entries, context=validation_context, mode=mode)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    clock = now or (lambda: datetime.now(timezone.utc))
    timestamp = _timestamp(clock())
    run_id = f"run_{timestamp}"
    final_run = root / run_id
    if final_run.exists():
        raise BundleBuildError("bundle_run_id_collision")

    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".tmp", dir=root))
    published = False
    try:
        submission_dir = temporary / "submission"
        submission_dir.mkdir()
        for item in validated:
            atomic_write_bytes(submission_dir / item.entry.filename, item.entry.csv_bytes)

        zip_name = f"AIC2026_submission_{timestamp}.zip"
        zip_data = _zip_bytes(validated)
        zip_path = atomic_write_bytes(temporary / zip_name, zip_data)
        files = [
            {
                **item.to_manifest(),
                "sha256": _sha256(item.entry.csv_bytes),
            }
            for item in validated
        ]
        manifest: dict[str, object] = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ")
            .replace(tzinfo=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "spec_id": CODABENCH_SPEC_ID,
            "mode": mode,
            "batch_id": validated[0].entry.batch_id,
            "filename": zip_name,
            "relative_path": f"{run_id}/{zip_name}",
            "sha256": _sha256(zip_data),
            "byte_count": len(zip_data),
            "file_count": len(validated),
            "status": "valid",
            "validation_status": "valid",
            "files": files,
        }
        atomic_write_bytes(temporary / "manifest.json", _json_bytes(manifest))

        lock_root = root / ".locks"
        try:
            with BatchIngestLock(
                lock_root,
                "codabench-submission-manifest",
                timeout_seconds=10.0,
            ):
                if final_run.exists():
                    raise BundleBuildError("bundle_run_id_collision")
                os.replace(temporary, final_run)
                published = True
                root_path = root / "manifest.json"
                index = _root_manifest(root_path)
                bundles = list(index["bundles"])  # type: ignore[arg-type]
                bundles.append(manifest)
                atomic_write_bytes(
                    root_path,
                    _json_bytes({"schema_version": 1, "bundles": bundles}),
                )
        except BatchIngestLockError:
            raise BundleBuildError("bundle_manifest_lock_timeout") from None
        except Exception:
            if published:
                shutil.rmtree(final_run, ignore_errors=True)
                published = False
            raise
        return BundleArtifact(path=final_run / zip_path.name, manifest=manifest)
    except Exception:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_bundle_manifest(output_dir: str | Path, run_id: str) -> dict[str, object] | None:
    """Read one published manifest without accepting path-like run identifiers."""

    if not isinstance(run_id, str) or not re_fullmatch_run_id(run_id):
        return None
    path = Path(output_dir) / run_id / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("run_id") == run_id else None


def re_fullmatch_run_id(value: str) -> bool:
    """Return whether a run ID has the exact UTC timestamp form produced above."""

    return re.fullmatch(r"run_[0-9]{8}T[0-9]{12}Z", value) is not None


__all__ = [
    "BundleArtifact",
    "BundleBuildError",
    "build_codabench_bundle",
    "load_bundle_manifest",
    "re_fullmatch_run_id",
]
