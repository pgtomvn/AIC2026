"""Offline tests for the B02 source and encoder import gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.import_b02_reference import main
from src.ingest.b02_reference import (
    FEATURE_ARCHIVE,
    MAP_ARCHIVE,
    MEDIA_ARCHIVE,
    OBJECT_ARCHIVES,
    B02ReferencePlan,
    _archive_fingerprint,
    _canonical_json,
    encoder_receipt_matches,
    load_b02_reference_plan,
)


@pytest.fixture
def fixture_root() -> Iterator[Path]:
    root = Path.cwd() / ".cache" / f"b02-reference-test-{uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().parent != (Path.cwd() / ".cache").resolve():
            raise RuntimeError("test scratch path escaped the workspace cache")
        shutil.rmtree(root)


def _sealed(value: dict[str, object], field: str) -> dict[str, object]:
    value[field] = hashlib.sha256(_canonical_json(value)).hexdigest()
    return value


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    keyframes = (
        "Keyframes_M08.zip",
        "Keyframes_N001-N010.zip",
        "Keyframes_S01.zip",
    )
    names = (MAP_ARCHIVE, FEATURE_ARCHIVE, MEDIA_ARCHIVE, *OBJECT_ARCHIVES, *keyframes)
    inputs = {}
    for name in names:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("placeholder/member", name)
        size, digest = _archive_fingerprint(path)
        inputs[name] = {
            "archive_size_bytes": size,
            "central_directory_sha256": digest,
        }
    videos = {
        "M08_V001": {
            "keyframe_archive": keyframes[0],
            "object_archive": OBJECT_ARCHIVES[0],
            "keyframe_count": 2,
            "keyframe_member_count": 2,
            "object_member_count": 2,
            "feature_rows": 2,
            "kis_qa_time_source": "pts_time",
            "trake_eligible": True,
            "media_info_member": "media-info/M08_V001.json",
        },
        "N001-V001": {
            "keyframe_archive": keyframes[1],
            "object_archive": OBJECT_ARCHIVES[1],
            "keyframe_count": 1,
            "keyframe_member_count": 1,
            "object_member_count": 1,
            "feature_rows": 1,
            "kis_qa_time_source": "pts_time",
            "trake_eligible": False,
        },
        "S01-V001": {
            "keyframe_archive": keyframes[2],
            "object_archive": OBJECT_ARCHIVES[2],
            "keyframe_count": 1,
            "keyframe_member_count": 1,
            "object_member_count": 1,
            "feature_rows": 1,
            "kis_qa_time_source": "pts_time",
            "trake_eligible": True,
            "media_info_member": "media-info/S01_V001.json",
        },
    }
    evidence = _sealed(
        {
            "schema_version": "b02-p0-pts-collision-evidence-v1",
            "batch_id": "aic26-b2",
            "video_id": "M08_V001",
            "equal_pts_pair_count": 1,
            "distinct_evidence_pair_count": 1,
            "pairs": [
                {
                    "ordinals": [1, 2],
                    "pts_time": ["1.0", "1.0"],
                    "frame_idx": [10, 10],
                    "distinct_evidence": True,
                }
            ],
        },
        "report_sha256",
    )
    manifest = _sealed(
        {
            "schema_version": "b02-p0-audit-v1",
            "batch_id": "aic26-b2",
            "status": "blocked",
            "summary": {
                "videos": 3,
                "map_rows": 4,
                "feature_rows": 4,
                "keyframe_members": 4,
                "object_members": 4,
                "media_info_videos": 2,
                "feature_dimension": 512,
            },
            "videos": videos,
            "inputs": inputs,
            "errors": [
                {
                    "code": "nonincreasing_pts",
                    "detail": "M08_V001: ordinals 1,2; pts 1.0,1.0; frame_idx 10,10",
                }
            ],
        },
        "manifest_sha256",
    )
    return (
        tmp_path,
        _write_json(tmp_path / "manifest.json", manifest),
        _write_json(tmp_path / "evidence.json", evidence),
    )


def _load(root: Path, manifest: Path, evidence: Path) -> B02ReferencePlan:
    return load_b02_reference_plan(
        root,
        manifest,
        evidence,
        expected_videos=3,
        expected_frames=4,
        expected_keyframe_archives=3,
        expected_media_videos=2,
        expected_pts_collisions=1,
    )


def test_b02_plan_accepts_only_documented_equal_pts_pair(fixture_root: Path) -> None:
    root, manifest, evidence = _fixture(fixture_root)
    plan = _load(root, manifest, evidence)
    assert plan.batch_id == "aic26-b2"
    assert plan.frames == 4
    assert plan.quarantined_pts_pairs == ((1, 2),)
    assert len(plan.keyframe_archives) == 3
    assert len(plan.object_archives) == 3


def test_b02_plan_rejects_stale_archive_and_unreviewed_error(fixture_root: Path) -> None:
    root, manifest, evidence = _fixture(fixture_root)
    with zipfile.ZipFile(root / FEATURE_ARCHIVE, "a") as archive:
        archive.writestr("another/member", b"different")
    with pytest.raises(ValueError, match="differs from P0 manifest"):
        _load(root, manifest, evidence)

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["errors"].append({"code": "missing_feature", "detail": "M08_V001"})
    value.pop("manifest_sha256")
    _write_json(manifest, _sealed(value, "manifest_sha256"))
    with pytest.raises(ValueError, match="unreviewed errors"):
        _load(root, manifest, evidence)


def test_b02_encoder_receipt_is_batch_and_feature_bound(fixture_root: Path) -> None:
    plan = _load(*_fixture(fixture_root))
    receipt = {
        "schema_version": "b02-clip-encoder-verification-v1",
        "verified": True,
        "batch_id": "aic26-b2",
        "manifest_sha256": plan.manifest_sha256,
        "feature_archive": FEATURE_ARCHIVE,
        "feature_archive_sha256": "a" * 64,
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "encoder_fingerprint": "fingerprint",
        "dimension": 512,
        "sample_count": 3,
        "median_cosine": 0.9997,
        "minimum_median_cosine": 0.999,
        "minimum_sample_cosine": 0.9995,
        "minimum_sample_cosine_required": 0.999,
        "samples": [
            {"video_id": group + "01", "cosine": score}
            for group, score in (("M", 0.9995), ("N", 0.9997), ("S", 0.9999))
        ],
    }
    path = _write_json(fixture_root / "receipt.json", receipt)
    kwargs = {
        "plan": plan,
        "feature_sha256": "a" * 64,
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "encoder_fingerprint": "fingerprint",
        "minimum_median_cosine": 0.999,
        "minimum_sample_size": 3,
    }
    assert encoder_receipt_matches(path, **kwargs)
    assert not encoder_receipt_matches(path, **{**kwargs, "feature_sha256": "b" * 64})
    receipt["batch_id"] = "aic25-b1-reference"
    _write_json(path, receipt)
    assert not encoder_receipt_matches(path, **kwargs)


def test_b02_cli_requires_explicit_apply_before_any_write() -> None:
    with pytest.raises(SystemExit) as result:
        main(["--index-vectors"])
    assert result.value.code == 2
