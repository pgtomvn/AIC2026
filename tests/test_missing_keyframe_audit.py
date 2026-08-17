"""Offline audit tests for canonical identities and organizer ordinal aliases."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

from scripts.audit_missing_keyframes import audit_canonical_frames


def _database(path: Path, archive: Path, *, include_second_identity: bool = True) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE canonical_frames (
                batch_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                source_frame_idx INTEGER NOT NULL,
                keyframe_ordinal INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                image_uri TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE canonical_frame_mappings (
                batch_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                keyframe_ordinal INTEGER NOT NULL,
                source_frame_idx INTEGER NOT NULL
            )
            """
        )
        canonical = [
            (
                "b1",
                "V001",
                10,
                1,
                1_000,
                f"zip://{archive.resolve()}!/keyframes/V001/001.jpg",
            )
        ]
        if include_second_identity:
            canonical.append(
                (
                    "b1",
                    "V001",
                    20,
                    3,
                    2_000,
                    f"zip://{archive.resolve()}!/keyframes/V001/003.jpg",
                )
            )
        connection.executemany(
            "INSERT INTO canonical_frames VALUES (?, ?, ?, ?, ?, ?)", canonical
        )
        connection.executemany(
            "INSERT INTO canonical_frame_mappings VALUES (?, ?, ?, ?)",
            [
                ("b1", "V001", 1, 10),
                ("b1", "V001", 2, 10),
                ("b1", "V001", 3, 20),
            ],
        )


def test_audit_distinguishes_duplicate_ordinal_from_missing_identity(tmp_path: Path) -> None:
    archive = tmp_path / "keyframes.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for ordinal in (1, 2, 3):
            handle.writestr(f"keyframes/V001/{ordinal:03d}.jpg", b"jpeg")
    database = tmp_path / "artifacts.sqlite3"
    _database(database, archive)

    summary, missing, aliases = audit_canonical_frames(
        database, "b1", allowed_root=tmp_path
    )

    assert summary == {
        "schema_version": 1,
        "batch_id": "b1",
        "mapping_rows": 3,
        "canonical_identities": 2,
        "unique_mapping_identities": 2,
        "duplicate_alias_rows": 1,
        "duplicate_alias_groups": 1,
        "missing_identity_count": 0,
        "missing_canonical_media_count": 0,
        "missing_ordinal_media_count": 0,
        "missing_record_count": 0,
    }
    assert missing == []
    assert aliases == [
        {
            "video_id": "V001",
            "source_frame_idx": 10,
            "canonical_keyframe_ordinal": 1,
            "keyframe_ordinals": [1, 2],
        }
    ]
    assert str(tmp_path) not in str(summary) + str(aliases)


def test_audit_reports_true_missing_identity_and_media_without_local_paths(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "keyframes.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("keyframes/V001/001.jpg", b"jpeg")
        handle.writestr("keyframes/V001/002.jpg", b"jpeg")
    database = tmp_path / "artifacts.sqlite3"
    _database(database, archive, include_second_identity=False)

    summary, missing, _aliases = audit_canonical_frames(
        database, "b1", allowed_root=tmp_path
    )

    assert summary["missing_identity_count"] == 1
    assert summary["missing_record_count"] == 1
    assert missing[0]["reason_code"] == "canonical_record_missing"
    assert missing[0]["source_frame_idx"] == 20
    assert missing[0]["timestamp_ms"] is None
    assert missing[0]["target_output_path"].startswith(
        "data/cache/recovered-keyframes/b1/V001/"
    )
    assert str(tmp_path) not in str(missing)
