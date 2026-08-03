"""Offline tests for archive discovery, validation, and safe relocation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path

import pytest

from scripts.catalog import main as catalog_main
from src import PROJECT_ROOT
from src.ingest.archive_catalog import (
    ArchiveKind,
    ArchiveValidationError,
    ProtectedArchiveError,
    discover_archives,
    inspect_archive,
    is_stable_archive,
    observe_archive,
    relocate_archive,
    update_checksum_manifest,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


@pytest.mark.parametrize(
    ("member_name", "payload", "expected"),
    [
        ("incoming/L99_V001.mp4", b"video", ArchiveKind.VIDEOS),
        ("frames/L99_V001/000001.jpg", b"jpeg", ArchiveKind.KEYFRAMES),
        (
            "maps/L99_V001.csv",
            b"n,pts_time,fps,frame_idx\n1,0.0,25,0\n",
            ArchiveKind.FRAME_MAPS,
        ),
        ("vectors/L99_V001.npy", b"npy", ArchiveKind.FEATURES),
        (
            "records/L99_V001.json",
            b'{"format":{"duration":"2.0"},"streams":[{"codec_name":"h264"}]}',
            ArchiveKind.MEDIA_INFO,
        ),
        (
            "records/L99_V001.json",
            b'{"detection_class_entities":["Person"],"detection_scores":[0.9]}',
            ArchiveKind.OBJECTS,
        ),
    ],
)
def test_classifies_archive_from_members(
    tmp_path: Path,
    member_name: str,
    payload: bytes,
    expected: ArchiveKind,
) -> None:
    archive = tmp_path / "misleading-name.zip"
    _write_zip(archive, {member_name: payload})

    inspection = inspect_archive(archive, verify_crc=True, compute_sha256=True)

    assert inspection.kinds == (expected,)
    assert inspection.crc_ok is True
    assert inspection.sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert inspection.file_count == 1


def test_discovery_ignores_partial_and_recent_files(tmp_path: Path) -> None:
    stable = tmp_path / "stable.zip"
    recent = tmp_path / "recent.zip"
    partial = tmp_path / "partial.zip.crdownload"
    _write_zip(stable, {"v/L99_V001.mp4": b"old"})
    _write_zip(recent, {"v/L99_V002.mp4": b"new"})
    partial.write_bytes(b"not complete")
    now = time.time()
    os.utime(stable, (now - 121, now - 121))
    os.utime(recent, (now - 10, now - 10))

    found = discover_archives([tmp_path], min_age_seconds=120, now=now)

    assert [item.path.name for item in found] == ["stable.zip"]


def test_two_observation_stability_does_not_sleep(tmp_path: Path) -> None:
    archive = tmp_path / "arriving.zip"
    _write_zip(archive, {"v/L99_V001.mp4": b"video"})
    observation = observe_archive(archive, now=1_000.0)

    assert not is_stable_archive(archive, previous=observation, now=1_119.9, min_age_seconds=120)
    assert is_stable_archive(archive, previous=observation, now=1_120.0, min_age_seconds=120)

    archive.write_bytes(archive.read_bytes() + b"changed")
    assert not is_stable_archive(archive, previous=observation, now=1_200.0, min_age_seconds=120)


def test_crc_failure_is_reported(tmp_path: Path) -> None:
    archive = tmp_path / "corrupt.zip"
    unique_payload = b"payload-unique-for-crc-test"
    _write_zip(archive, {"v/L99_V001.mp4": unique_payload})
    damaged = bytearray(archive.read_bytes())
    offset = damaged.index(unique_payload)
    damaged[offset] ^= 0xFF
    archive.write_bytes(damaged)

    with pytest.raises(ArchiveValidationError, match="CRC validation failed"):
        inspect_archive(archive, verify_crc=True)


@pytest.mark.parametrize("name", ["Keyframes_L21.zip", "VIDEOS_L21_A.ZIP"])
def test_protected_archives_cannot_be_relocated(tmp_path: Path, name: str) -> None:
    archive = tmp_path / name
    _write_zip(archive, {"v/L21_V001.mp4": b"preserve"})

    with pytest.raises(ProtectedArchiveError):
        relocate_archive(
            archive,
            tmp_path / "archives",
            competition="aic25",
            batch_id="b1",
            kinds=(ArchiveKind.VIDEOS,),
        )

    assert archive.is_file()
    assert archive.read_bytes()


def test_relocation_is_bucketed_and_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "incoming" / "batch.zip"
    source.parent.mkdir()
    _write_zip(source, {"v/L99_V001.mp4": b"preserve"})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    destination = relocate_archive(
        source,
        tmp_path / "archives",
        competition="aic25",
        batch_id="b2",
        kinds=(ArchiveKind.VIDEOS,),
        expected_sha256=digest,
    )

    assert destination == tmp_path / "archives" / "aic25" / "b2" / "videos" / "batch.zip"
    assert destination.is_file()
    assert not source.exists()

    second = tmp_path / "incoming" / "batch.zip"
    _write_zip(second, {"v/L99_V001.mp4": b"new data"})
    with pytest.raises(FileExistsError):
        relocate_archive(
            second,
            tmp_path / "archives",
            competition="aic25",
            batch_id="b2",
            kinds=(ArchiveKind.VIDEOS,),
        )
    assert second.is_file()
    assert destination.read_bytes() != second.read_bytes()


def test_relocation_rejects_path_components_and_hash_mismatch_without_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch.zip"
    _write_zip(source, {"v/L99_V001.mp4": b"preserve"})
    archive_root = tmp_path / "archives"

    with pytest.raises(ValueError, match="competition"):
        relocate_archive(
            source,
            archive_root,
            competition="../outside",
            batch_id="b2",
            kinds=(ArchiveKind.VIDEOS,),
        )
    with pytest.raises(ArchiveValidationError, match="SHA-256 mismatch"):
        relocate_archive(
            source,
            archive_root,
            competition="aic25",
            batch_id="b2",
            kinds=(ArchiveKind.VIDEOS,),
            expected_sha256="0" * 64,
        )

    assert source.is_file()
    assert not archive_root.exists()


def test_checksum_manifest_update_preserves_comments_and_rewrites_path(tmp_path: Path) -> None:
    manifest = tmp_path / "checksums" / "archives.sha256"
    manifest.parent.mkdir()
    old_path = tmp_path / "incoming" / "batch.zip"
    new_path = tmp_path / "data" / "archives" / "aic25" / "b2" / "videos" / "batch.zip"
    digest = "a" * 64
    manifest.write_text(f"# keep this\n{'b' * 64}  incoming/batch.zip\n", encoding="utf-8")

    update_checksum_manifest(
        manifest,
        new_path,
        digest,
        previous_path=old_path,
        root=tmp_path,
    )

    assert manifest.read_text(encoding="utf-8") == (
        f"# keep this\n{digest}  data/archives/aic25/b2/videos/batch.zip\n"
    )
    assert list(manifest.parent.glob(".*.tmp")) == []


def test_catalog_cli_relocates_only_after_full_verification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "batch-two.zip"
    _write_zip(source, {"videos/L99_V001.mp4": b"verified-content"})
    now = time.time()
    os.utime(source, (now - 121, now - 121))
    archive_root = tmp_path / "data" / "archives"
    ledger = tmp_path / "data" / "state" / "jobs.sqlite3"
    manifest = tmp_path / "checksums" / "archives.sha256"

    result = catalog_main(
        [
            str(incoming),
            "--relocate",
            "--competition",
            "aic26",
            "--batch-id",
            "b2",
            "--archive-root",
            str(archive_root),
            "--ledger",
            str(ledger),
            "--checksum-manifest",
            str(manifest),
        ]
    )

    assert result == 0
    destination = archive_root / "aic26" / "b2" / "videos" / source.name
    assert destination.read_bytes()
    assert not source.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["crc_ok"] is True
    assert len(payload[0]["sha256"]) == 64
    assert payload[0]["relocated_to"] == str(destination)
    manifest_text = manifest.read_text(encoding="utf-8")
    manifest_value = manifest_text.strip().split(maxsplit=1)[1]
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    assert manifest_path.resolve() == destination.resolve()

    with sqlite3.connect(ledger) as connection:
        row = connection.execute(
            "SELECT state, relocated_to, crc_ok, sha256 FROM archives"
        ).fetchone()
    assert row == ("relocated", str(destination), 1, payload[0]["sha256"])


def test_catalog_cli_relocation_cannot_bypass_stability_gate(tmp_path: Path) -> None:
    source = tmp_path / "recent.zip"
    _write_zip(source, {"videos/L99_V001.mp4": b"still-arriving"})

    with pytest.raises(SystemExit):
        catalog_main(
            [
                str(source),
                "--relocate",
                "--competition",
                "aic26",
                "--batch-id",
                "b2",
                "--min-age",
                "0",
            ]
        )

    assert source.is_file()


def test_catalog_cli_does_not_relocate_a_recent_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "recent.zip"
    _write_zip(source, {"videos/L99_V001.mp4": b"still-arriving"})

    result = catalog_main(
        [
            str(source),
            "--relocate",
            "--competition",
            "aic26",
            "--batch-id",
            "b2",
            "--archive-root",
            str(tmp_path / "archives"),
            "--ledger",
            str(tmp_path / "jobs.sqlite3"),
            "--checksum-manifest",
            str(tmp_path / "archives.sha256"),
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == []
    assert source.is_file()
    assert not (tmp_path / "archives").exists()


def test_catalog_cli_protected_archive_is_never_moved(tmp_path: Path) -> None:
    source = tmp_path / "Keyframes_L21.zip"
    _write_zip(source, {"frames/L21_V001/000001.jpg": b"protected"})
    now = time.time()
    os.utime(source, (now - 121, now - 121))

    with pytest.raises(ProtectedArchiveError):
        catalog_main(
            [
                str(source),
                "--relocate",
                "--competition",
                "aic26",
                "--batch-id",
                "b2",
                "--archive-root",
                str(tmp_path / "archives"),
                "--ledger",
                str(tmp_path / "jobs.sqlite3"),
                "--checksum-manifest",
                str(tmp_path / "archives.sha256"),
            ]
        )

    assert source.is_file()
