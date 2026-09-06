"""Build a credential-free source bundle for the Colab extraction notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "colab" / "aic2026-colab-source.zip"
INCLUDE_PATHS = (
    Path("src"),
    Path("scripts"),
    Path("config"),
    Path("checksums"),
    Path("notebooks"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".venv",
        ".venv-runtime",
        "__pycache__",
        "data",
        "docker_data",
        "logs",
        "models",
        "outputs",
    }
)
_FORBIDDEN_NAMES = frozenset({".env", ".env.local", ".env.production"})
_FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bin",
        ".db",
        ".gz",
        ".npy",
        ".onnx",
        ".pt",
        ".pyc",
        ".pyo",
        ".sqlite",
        ".sqlite3",
        ".zip",
    }
)


def _eligible(relative_path: Path) -> bool:
    lowered_parts = {part.lower() for part in relative_path.parts}
    return (
        not lowered_parts.intersection(_FORBIDDEN_PARTS)
        and relative_path.name.lower() not in _FORBIDDEN_NAMES
        and relative_path.suffix.lower() not in _FORBIDDEN_SUFFIXES
    )


def _source_files(project_root: Path, output_path: Path) -> Iterable[tuple[Path, Path]]:
    root = project_root.resolve()
    resolved_output = output_path.resolve()
    for relative_root in INCLUDE_PATHS:
        candidate = (root / relative_root).resolve()
        if candidate != root and root not in candidate.parents:
            raise RuntimeError(f"source path escapes project root: {relative_root}")
        if not candidate.exists():
            raise FileNotFoundError(f"required source path is missing: {relative_root}")
        paths = (candidate,) if candidate.is_file() else candidate.rglob("*")
        for path in paths:
            if not path.is_file() or path.resolve() == resolved_output:
                continue
            relative = path.resolve().relative_to(root)
            if _eligible(relative):
                yield path, relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_colab_source_bundle(project_root: Path, output_path: Path) -> dict[str, object]:
    """Atomically publish a deterministic allowlisted source ZIP and hash sidecar."""

    root = project_root.resolve()
    output = output_path.resolve()
    files = sorted(_source_files(root, output), key=lambda item: item[1].as_posix())
    if not files:
        raise RuntimeError("Colab source bundle would be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for source, relative in files:
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with source.open("rb") as input_stream, archive.open(info, "w") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    os.replace(temporary, output)
    digest = _sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    os.replace(sidecar_tmp, sidecar)
    return {
        "bundle": str(output),
        "sha256": digest,
        "sha256_sidecar": str(sidecar),
        "file_count": len(files),
        "byte_count": output.stat().st_size,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_colab_source_bundle(args.project_root, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
