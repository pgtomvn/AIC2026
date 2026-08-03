"""SHA-256 freezing and verification for competition-day runtime assets."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_assets(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    destination: str | Path,
) -> int:
    """Atomically write a sorted SHA-256 manifest for explicit files/directories."""

    root_path = Path(root).resolve()
    destination_path = Path(destination)
    files: set[Path] = set()

    def add_file(candidate: Path) -> None:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError as error:
            raise ValueError(f"asset is outside manifest root: {resolved}") from error
        files.add(resolved)

    for raw_path in paths:
        unresolved = Path(raw_path)
        path = (
            (root_path / unresolved).resolve()
            if not unresolved.is_absolute()
            else unresolved.resolve()
        )
        try:
            path.relative_to(root_path)
        except ValueError as error:
            raise ValueError(f"asset is outside manifest root: {path}") from error
        if path.is_file():
            add_file(path)
        elif path.is_dir():
            for item in path.rglob("*"):
                if item.is_file():
                    add_file(item)
        else:
            raise FileNotFoundError(path)
    files.discard(destination_path.resolve())
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root_path).as_posix()}"
        for path in sorted(
            files,
            key=lambda item: (
                item.relative_to(root_path).as_posix().casefold(),
                item.relative_to(root_path).as_posix(),
            ),
        )
    ]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines) + ("\n" if lines else ""))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(lines)


def verify_asset_manifest(
    manifest: str | Path,
    *,
    root: str | Path,
    required_paths: Iterable[str | Path] = (),
) -> tuple[int, list[str]]:
    """Verify manifest entries and optional required files/directories.

    Required directories must exist, contain at least one regular file, and
    have every file represented in the manifest. This prevents an incomplete
    partial-model manifest from being mistaken for offline readiness.
    """

    manifest_path = Path(manifest)
    root_path = Path(root).resolve()
    if not manifest_path.is_file():
        return 0, [f"asset manifest is missing: {manifest_path}"]
    checked = 0
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.fullmatch(line)
        if match is None:
            errors.append(f"invalid manifest line {line_number}")
            continue
        expected, relative_value = match.groups()
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe manifest path on line {line_number}")
            continue
        key = relative.as_posix().casefold()
        if key in seen:
            errors.append(f"duplicate manifest path: {relative_value}")
            continue
        seen.add(key)
        target = (root_path / relative).resolve()
        try:
            target.relative_to(root_path)
        except ValueError:
            errors.append(f"manifest path escapes root: {relative_value}")
            continue
        if not target.is_file():
            errors.append(f"missing asset: {relative_value}")
            continue
        actual = sha256_file(target)
        checked += 1
        if actual != expected:
            errors.append(f"checksum mismatch: {relative_value}")

    checked_requirements: set[Path] = set()
    for raw_required in required_paths:
        unresolved = Path(raw_required)
        required = (
            (root_path / unresolved).resolve()
            if not unresolved.is_absolute()
            else unresolved.resolve()
        )
        if required in checked_requirements:
            continue
        checked_requirements.add(required)
        try:
            required.relative_to(root_path)
        except ValueError:
            errors.append(f"required asset path escapes root: {required}")
            continue
        if required.is_file():
            required_files = [required]
        elif required.is_dir():
            required_files = sorted(
                (item.resolve() for item in required.rglob("*") if item.is_file()),
                key=lambda item: item.as_posix().casefold(),
            )
            if not required_files:
                relative = required.relative_to(root_path).as_posix()
                errors.append(f"required asset directory has no files: {relative}")
                continue
        else:
            relative = required.relative_to(root_path).as_posix()
            errors.append(f"missing required asset path: {relative}")
            continue
        for required_file in required_files:
            try:
                relative = required_file.relative_to(root_path)
            except ValueError:
                errors.append(f"required asset resolves outside root: {required_file}")
                continue
            if relative.as_posix().casefold() not in seen:
                errors.append(f"unlisted required asset: {relative.as_posix()}")
    if checked == 0 and not errors:
        errors.append("asset manifest has no files")
    return checked, errors
