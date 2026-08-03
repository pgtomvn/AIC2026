from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.assets import freeze_assets, sha256_file, verify_asset_manifest


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_freeze_assets_is_sorted_deterministic_and_excludes_destination(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "models"
    (assets / "nested").mkdir(parents=True)
    (assets / "z.bin").write_bytes(b"z")
    (assets / "nested" / "a.bin").write_bytes(b"a")
    manifest = assets / "runtime.sha256"

    assert freeze_assets(["models"], root=tmp_path, destination=manifest) == 2
    first = manifest.read_bytes()
    assert first.decode("utf-8").splitlines() == [
        f"{_digest(b'a')}  models/nested/a.bin",
        f"{_digest(b'z')}  models/z.bin",
    ]

    assert freeze_assets([assets], root=tmp_path, destination=manifest) == 2
    assert manifest.read_bytes() == first
    assert verify_asset_manifest(manifest, root=tmp_path) == (2, [])


def test_freeze_assets_rejects_missing_and_outside_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "runtime.sha256"

    with pytest.raises(FileNotFoundError):
        freeze_assets(["missing"], root=tmp_path, destination=manifest)

    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(ValueError, match="outside manifest root"):
            freeze_assets([outside], root=tmp_path, destination=manifest)
    finally:
        outside.unlink(missing_ok=True)


def test_freeze_assets_rejects_file_symlink_that_escapes_root(tmp_path: Path) -> None:
    assets = tmp_path / "models"
    assets.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-target.bin"
    outside.write_bytes(b"outside")
    link = assets / "escaped.bin"
    try:
        try:
            link.symlink_to(outside)
        except OSError as error:
            pytest.skip(f"file symlinks are unavailable: {error}")
        with pytest.raises(ValueError, match="outside manifest root"):
            freeze_assets([assets], root=tmp_path, destination=tmp_path / "runtime.sha256")
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


def test_verify_reports_changed_and_missing_assets(tmp_path: Path) -> None:
    present = tmp_path / "present.bin"
    missing = tmp_path / "missing.bin"
    present.write_bytes(b"changed")
    manifest = tmp_path / "runtime.sha256"
    manifest.write_text(
        f"{_digest(b'original')}  present.bin\n{_digest(b'missing')}  missing.bin\n",
        encoding="utf-8",
    )

    checked, errors = verify_asset_manifest(manifest, root=tmp_path)

    assert checked == 1
    assert errors == [
        "checksum mismatch: present.bin",
        "missing asset: missing.bin",
    ]


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("../escape.bin", "unsafe manifest path on line 1"),
        ("nested/../../escape.bin", "unsafe manifest path on line 1"),
        ("not-a-manifest-line", "invalid manifest line 1"),
    ],
)
def test_verify_rejects_unsafe_or_invalid_entries(
    tmp_path: Path, entry: str, expected: str
) -> None:
    manifest = tmp_path / "runtime.sha256"
    line = entry if entry.startswith("not-") else f"{'0' * 64}  {entry}"
    manifest.write_text(line + "\n", encoding="utf-8")

    assert verify_asset_manifest(manifest, root=tmp_path) == (0, [expected])


def test_verify_reports_missing_empty_and_duplicate_manifests(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sha256"
    assert verify_asset_manifest(missing, root=tmp_path) == (
        0,
        [f"asset manifest is missing: {missing}"],
    )

    empty = tmp_path / "empty.sha256"
    empty.write_text("# comments only\n", encoding="utf-8")
    assert verify_asset_manifest(empty, root=tmp_path) == (
        0,
        ["asset manifest has no files"],
    )

    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")
    digest = sha256_file(asset)
    duplicate = tmp_path / "duplicate.sha256"
    duplicate.write_text(f"{digest}  asset.bin\n{digest}  ASSET.BIN\n", encoding="utf-8")
    assert verify_asset_manifest(duplicate, root=tmp_path) == (
        1,
        ["duplicate manifest path: ASSET.BIN"],
    )


def test_verify_requires_complete_manifest_coverage(tmp_path: Path) -> None:
    assets = tmp_path / "models"
    empty = tmp_path / "empty-model"
    assets.mkdir()
    empty.mkdir()
    listed = assets / "listed.bin"
    unlisted = assets / "unlisted.bin"
    listed.write_bytes(b"listed")
    unlisted.write_bytes(b"unlisted")
    manifest = tmp_path / "runtime.sha256"
    manifest.write_text(f"{sha256_file(listed)}  models/listed.bin\n", encoding="utf-8")

    assert verify_asset_manifest(
        manifest,
        root=tmp_path,
        required_paths=(assets, empty, "missing-model"),
    ) == (
        1,
        [
            "unlisted required asset: models/unlisted.bin",
            "required asset directory has no files: empty-model",
            "missing required asset path: missing-model",
        ],
    )


def test_sha256_file_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sha256_file(asset, chunk_size=0)
