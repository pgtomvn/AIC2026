"""Read-only source gate and isolated encoder receipt for B02 reference packs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

BATCH_ID = "aic26-b2"
MAP_ARCHIVE = "aic26-b2-map-keyframes.zip"
FEATURE_ARCHIVE = "aic26-b2-clip-features.zip"
MEDIA_ARCHIVE = "aic26-b2-media-info.zip"
OBJECT_ARCHIVES = ("Objects_M01_M10.zip", "Objects_N001_N100.zip", "Objects_S01.zip")
EXPECTED_VIDEOS = 614
EXPECTED_FRAMES = 220_270
EXPECTED_KEYFRAME_ARCHIVES = 21
EXPECTED_MEDIA_VIDEOS = 316
EXPECTED_PTS_COLLISIONS = 9
_MANIFEST_SCHEMA = "b02-p0-audit-v1"
_EVIDENCE_SCHEMA = "b02-p0-pts-collision-evidence-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _checked_sha256_record(value: Mapping[str, Any], field: str) -> str:
    recorded = value.get(field)
    if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
        raise ValueError(f"invalid {field}")
    body = {key: item for key, item in value.items() if key != field}
    if hashlib.sha256(_canonical_json(body)).hexdigest() != recorded:
        raise ValueError(f"{field} does not match content")
    return recorded


def _archive_fingerprint(path: Path) -> tuple[int, str]:
    """Match the P0 central-directory fingerprint without reading media payloads."""

    digest = hashlib.sha256()
    names: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda member: member.filename):
            name = info.filename
            parts = PurePosixPath(name).parts
            mode = (info.external_attr >> 16) & 0o170000
            if (
                name.startswith("/")
                or "\\" in name
                or ".." in parts
                or name in names
                or (info.create_system == 3 and mode == stat.S_IFLNK)
            ):
                raise ValueError(f"unsafe or duplicate archive member: {path.name}: {name}")
            names.add(name)
            digest.update(
                _canonical_json((name, info.file_size, info.compress_size, info.CRC, info.is_dir()))
                + b"\n"
            )
    return path.stat().st_size, digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one reference ZIP when binding an encoder receipt to its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class B02ReferencePlan:
    root: Path
    manifest_sha256: str
    batch_id: str
    videos: int
    frames: int
    feature_dimension: int
    map_archive: Path
    feature_archive: Path
    media_archive: Path
    keyframe_archives: tuple[Path, ...]
    object_archives: tuple[Path, ...]
    quarantined_pts_pairs: tuple[tuple[int, int], ...]


def load_b02_reference_plan(
    root: Path,
    manifest_path: Path,
    evidence_path: Path,
    *,
    expected_videos: int = EXPECTED_VIDEOS,
    expected_frames: int = EXPECTED_FRAMES,
    expected_keyframe_archives: int = EXPECTED_KEYFRAME_ARCHIVES,
    expected_media_videos: int = EXPECTED_MEDIA_VIDEOS,
    expected_pts_collisions: int = EXPECTED_PTS_COLLISIONS,
) -> B02ReferencePlan:
    """Validate the P0 source inventory and the documented M08 time exception."""

    root = root.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("B02 manifest must be a JSON object")
    manifest_hash = _checked_sha256_record(manifest, "manifest_sha256")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA or manifest.get("batch_id") != BATCH_ID:
        raise ValueError("not a B02 P0 manifest")
    summary = manifest.get("summary")
    videos = manifest.get("videos")
    inputs = manifest.get("inputs")
    if (
        not isinstance(summary, dict)
        or not isinstance(videos, dict)
        or not isinstance(inputs, dict)
    ):
        raise ValueError("incomplete B02 manifest")
    counts = ("map_rows", "feature_rows", "keyframe_members", "object_members")
    if (
        summary.get("videos") != expected_videos
        or len(videos) != expected_videos
        or any(summary.get(key) != expected_frames for key in counts)
        or summary.get("media_info_videos") != expected_media_videos
        or summary.get("feature_dimension") != 512
    ):
        raise ValueError("B02 manifest counts or feature shape do not match the source contract")
    if sum(int(item["keyframe_count"]) for item in videos.values()) != expected_frames:
        raise ValueError("B02 per-video frame counts do not add up")

    keyframe_names = sorted(
        name
        for name in inputs
        if re.fullmatch(r"Keyframes_(?:M\d{2}|N\d{3}-N\d{3}|S\d{2})\.zip", name)
    )
    if len(keyframe_names) != expected_keyframe_archives:
        raise ValueError("unexpected B02 keyframe ZIP inventory")
    required_names = (
        MAP_ARCHIVE,
        FEATURE_ARCHIVE,
        MEDIA_ARCHIVE,
        *OBJECT_ARCHIVES,
        *keyframe_names,
    )
    if any(name not in inputs for name in required_names):
        raise ValueError("B02 manifest omits a required archive")
    for video_id, item in videos.items():
        if not isinstance(item, dict) or item.get("keyframe_archive") not in keyframe_names:
            raise ValueError(f"no keyframe ZIP for {video_id}")
        if item.get("object_archive") not in OBJECT_ARCHIVES:
            raise ValueError(f"no object ZIP for {video_id}")
        count = item["keyframe_count"]
        if any(
            item.get(key) != count
            for key in ("keyframe_member_count", "object_member_count", "feature_rows")
        ):
            raise ValueError(f"per-video ordinal coverage mismatch: {video_id}")
        if item.get("kis_qa_time_source") != "pts_time":
            raise ValueError(f"B02 timestamp source is not PTS: {video_id}")
        if video_id.startswith("N"):
            if item.get("trake_eligible") is not False or "media_info_member" in item:
                raise ValueError(f"unexpected N-series timing or media-info policy: {video_id}")
        elif not item.get("media_info_member"):
            raise ValueError(f"missing M/S media-info: {video_id}")

    errors = manifest.get("errors")
    if not isinstance(errors, list):
        raise ValueError("B02 manifest errors must be a list")
    quarantined: tuple[tuple[int, int], ...] = ()
    if errors:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            raise ValueError("B02 collision evidence must be a JSON object")
        _checked_sha256_record(evidence, "report_sha256")
        pairs = evidence.get("pairs")
        if (
            evidence.get("schema_version") != _EVIDENCE_SCHEMA
            or evidence.get("batch_id") != BATCH_ID
            or evidence.get("video_id") != "M08_V001"
            or not isinstance(pairs, list)
            or len(pairs) != expected_pts_collisions
            or evidence.get("equal_pts_pair_count") != len(pairs)
            or evidence.get("distinct_evidence_pair_count") != len(pairs)
        ):
            raise ValueError("B02 PTS collision evidence does not match the known exception")
        expected_errors: list[dict[str, str]] = []
        ordinals: list[tuple[int, int]] = []
        for pair in pairs:
            left, right = pair["ordinals"]
            pts_left, pts_right = pair["pts_time"]
            idx_left, idx_right = pair["frame_idx"]
            if (
                not pair.get("distinct_evidence")
                or right != left + 1
                or pts_left != pts_right
                or idx_left != idx_right
            ):
                raise ValueError("B02 PTS exception must preserve distinct adjacent ordinals")
            ordinals.append((left, right))
            expected_errors.append(
                {
                    "code": "nonincreasing_pts",
                    "detail": (
                        f"M08_V001: ordinals {left},{right}; pts {pts_left},{pts_right}; "
                        f"frame_idx {idx_left},{idx_right}"
                    ),
                }
            )
        if errors != expected_errors or manifest.get("status") != "blocked":
            raise ValueError("B02 manifest has unreviewed errors")
        quarantined = tuple(ordinals)
    elif manifest.get("status") != "structurally_ready":
        raise ValueError("B02 manifest status does not match its errors")

    for name in required_names:
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing B02 archive: {path}")
        expected = inputs[name]
        size, fingerprint = _archive_fingerprint(path)
        if (
            expected.get("archive_size_bytes") != size
            or expected.get("central_directory_sha256") != fingerprint
        ):
            raise ValueError(f"B02 archive differs from P0 manifest: {name}")

    return B02ReferencePlan(
        root=root,
        manifest_sha256=manifest_hash,
        batch_id=BATCH_ID,
        videos=expected_videos,
        frames=expected_frames,
        feature_dimension=512,
        map_archive=root / MAP_ARCHIVE,
        feature_archive=root / FEATURE_ARCHIVE,
        media_archive=root / MEDIA_ARCHIVE,
        keyframe_archives=tuple(root / name for name in keyframe_names),
        object_archives=tuple(root / name for name in OBJECT_ARCHIVES),
        quarantined_pts_pairs=quarantined,
    )


def encoder_receipt_matches(
    path: Path,
    *,
    plan: B02ReferencePlan,
    feature_sha256: str,
    model_name: str,
    pretrained: str,
    encoder_fingerprint: str,
    minimum_median_cosine: float,
    minimum_sample_size: int,
) -> bool:
    """Only a B02-specific, source-bound success receipt permits writes."""

    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        samples = receipt.get("samples")
        if not isinstance(samples, list) or len(samples) < minimum_sample_size:
            return False
        if receipt.get("sample_count") != len(samples):
            return False
        if {str(item["video_id"])[0] for item in samples} != {"M", "N", "S"}:
            return False
        scores = [float(item["cosine"]) for item in samples]
        if any(
            not math.isfinite(score) or score < minimum_median_cosine or score > 1.00001
            for score in scores
        ):
            return False
        if not math.isclose(float(receipt.get("median_cosine", -1)), statistics.median(scores)):
            return False
        if not math.isclose(float(receipt.get("minimum_sample_cosine", -1)), min(scores)):
            return False
        return bool(receipt.get("verified")) and all(
            (
                receipt.get("schema_version") == "b02-clip-encoder-verification-v1",
                receipt.get("batch_id") == plan.batch_id,
                receipt.get("manifest_sha256") == plan.manifest_sha256,
                receipt.get("feature_archive") == plan.feature_archive.name,
                receipt.get("feature_archive_sha256") == feature_sha256,
                receipt.get("model_name") == model_name,
                receipt.get("pretrained") == pretrained,
                receipt.get("encoder_fingerprint") == encoder_fingerprint,
                receipt.get("dimension") == plan.feature_dimension,
                receipt.get("sample_count", 0) >= minimum_sample_size,
                float(receipt.get("median_cosine", -1)) >= minimum_median_cosine,
                float(receipt.get("minimum_median_cosine", -1)) == minimum_median_cosine,
                float(receipt.get("minimum_sample_cosine", -1)) >= minimum_median_cosine,
                float(receipt.get("minimum_sample_cosine_required", -1)) == minimum_median_cosine,
            )
        )
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return False
