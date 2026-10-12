"""Offline tests for the read-only B02 structural audit."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_b02_p0 import audit_b02, inspect_pts_collision_evidence, main


def _archive(path: Path, members: dict[str, bytes | str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _npy(rows: int, dimension: int = 512, *, distinguish_last: bool = False) -> bytes:
    stream = io.BytesIO()
    matrix = np.zeros((rows, dimension), dtype=np.float16)
    if distinguish_last:
        matrix[-1, 0] = 1
    np.save(stream, matrix, allow_pickle=False)
    return stream.getvalue()


def _fixture(
    root: Path,
    *,
    omit_object: bool = False,
    bad_feature_rows: bool = False,
    bad_pts: bool = False,
    conflicting_duplicate: bool = False,
) -> None:
    n_last_pts = "0" if bad_pts else "0.0005"
    _archive(
        root / "aic26-b2-map-keyframes.zip",
        {
            "map-keyframes/M01_V001.csv": ("n,pts_time,fps,frame_idx\n1,0,25,0\n2,1.2,25,30\n"),
            "map-keyframes/N001-V001.csv": (
                f"n,pts_time,fps,frame_idx\n1,0,25,0\n2,{n_last_pts},25,0\n"
            ),
            "map-keyframes/S01-V001.csv": ("n,pts_time,fps,frame_idx\n1,0,30,0\n2,2,30,60\n"),
        },
    )
    _archive(
        root / "aic26-b2-clip-features.zip",
        {
            "clip-features/M01_V001.npy": _npy(2),
            "clip-features/N001-V001.npy": _npy(
                1 if bad_feature_rows else 2,
                distinguish_last=conflicting_duplicate,
            ),
            "clip-features/S01-V001.npy": _npy(2),
        },
    )
    _archive(
        root / "aic26-b2-media-info.zip",
        {
            "media-info/M01_V001.json": json.dumps({"title": "M"}),
            "media-info/S01_V001.json": json.dumps({"title": "S"}),
        },
    )
    _archive(
        root / "Objects_M01_M10.zip",
        {
            "M01_V001/001.json": "{}",
            "M01_V001/002.json": "{}",
        },
    )
    _archive(
        root / "Objects_N001_N100.zip",
        {
            "N001-V001/001.json": "{}",
            **(
                {}
                if omit_object
                else {
                    "N001-V001/002.json": '{"different": true}' if conflicting_duplicate else "{}"
                }
            ),
        },
    )
    _archive(
        root / "Objects_S01.zip",
        {
            "S01-V001/0001.json": "{}",
            "S01-V001/0002.json": "{}",
        },
    )
    for video_id, name in (
        ("M01_V001", "Keyframes_M01.zip"),
        ("N001-V001", "Keyframes_N001-N010.zip"),
        ("S01-V001", "Keyframes_S01.zip"),
    ):
        _archive(
            root / name,
            {
                f"keyframes/{video_id}/001.jpg": b"sample image",
                f"keyframes/{video_id}/002.jpg": (
                    b"different image"
                    if conflicting_duplicate and video_id == "N001-V001"
                    else b"sample image"
                ),
            },
        )


def test_p0_audit_aligns_canonical_ordinals_and_s_metadata_alias(tmp_path: Path) -> None:
    _fixture(tmp_path)

    report = audit_b02(tmp_path, expected_videos=3, expected_keyframes=6)

    assert report["status"] == "structurally_ready"
    assert report["summary"]["map_rows"] == 6
    assert report["summary"]["feature_rows"] == 6
    assert report["summary"]["object_members"] == 6
    assert report["summary"]["keyframe_members"] == 6
    assert report["summary"]["feature_dimension"] == 512
    assert report["videos"]["S01-V001"]["media_info_source_video_id"] == "S01_V001"
    assert report["videos"]["N001-V001"]["source_frame_idx_collisions"] == 1
    assert report["videos"]["N001-V001"]["rounded_timestamp_ms_collisions"] == 1
    assert report["videos"]["N001-V001"]["kis_qa_time_source"] == "pts_time"
    assert report["videos"]["N001-V001"]["is_vfr_traffic_video"] is True
    assert report["videos"]["N001-V001"]["trake_eligible"] is False
    assert report["videos"]["S01-V001"]["trake_eligible"] is True
    assert "media_info_member" not in report["videos"]["N001-V001"]
    assert {entry["code"] for entry in report["warnings"]} >= {
        "traffic_media_info_unavailable",
        "relative_frame_idx_collisions",
        "video_archives_not_found_local",
    }
    assert report["verification_level"].endswith("not-full-archive-hash")
    assert (
        audit_b02(tmp_path, expected_videos=3, expected_keyframes=6)["manifest_sha256"]
        == report["manifest_sha256"]
    )


def test_p0_audit_rejects_missing_object_ordinal(tmp_path: Path) -> None:
    _fixture(tmp_path, omit_object=True)

    report = audit_b02(tmp_path, expected_videos=3, expected_keyframes=6)

    assert report["status"] == "blocked"
    assert "object_ordinal_mismatch" in {entry["code"] for entry in report["errors"]}


def test_p0_audit_rejects_feature_row_mismatch(tmp_path: Path) -> None:
    _fixture(tmp_path, bad_feature_rows=True)

    report = audit_b02(tmp_path, expected_videos=3, expected_keyframes=6)

    assert report["status"] == "blocked"
    assert "feature_map_row_mismatch" in {entry["code"] for entry in report["errors"]}


def test_p0_audit_rejects_nonincreasing_pts(tmp_path: Path) -> None:
    _fixture(tmp_path, bad_pts=True)

    report = audit_b02(tmp_path, expected_videos=3, expected_keyframes=6)

    assert report["status"] == "blocked"
    assert "nonincreasing_pts" in {entry["code"] for entry in report["errors"]}
    assert any("ordinals 1,2" in entry["detail"] for entry in report["errors"])


def test_targeted_duplicate_pts_evidence_compares_all_three_modalities(tmp_path: Path) -> None:
    _fixture(tmp_path, bad_pts=True, conflicting_duplicate=True)

    report = inspect_pts_collision_evidence(
        tmp_path,
        "N001-V001",
        keyframe_archive="Keyframes_N001-N010.zip",
        object_archive="Objects_N001_N100.zip",
    )

    assert report["equal_pts_pair_count"] == 1
    assert report["distinct_evidence_pair_count"] == 1
    pair = report["pairs"][0]
    assert pair["ordinals"] == [1, 2]
    assert pair["frame_idx"] == [0, 0]
    assert pair["image_same_zip_crc_and_size"] is False
    assert pair["feature_vector_equal"] is False
    assert pair["object_same_zip_crc_and_size"] is False


def test_p0_cli_writes_only_with_explicit_output(tmp_path: Path) -> None:
    _fixture(tmp_path)
    output = tmp_path / "outputs" / "p0.json"

    assert main(["--root", str(tmp_path), "--expected-videos", "3", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "structurally_ready"


def test_p0_cli_never_overwrites_existing_manifest(tmp_path: Path) -> None:
    _fixture(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("keep this checkpoint", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(["--root", str(tmp_path), "--output", str(output)])

    assert output.read_text(encoding="utf-8") == "keep this checkpoint"
