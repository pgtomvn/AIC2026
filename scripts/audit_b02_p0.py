"""Read-only structural audit of the AIC 2026 B02 organizer archives.

This intentionally reads ZIP central directories, CSV maps, NPY headers, and
small media-info JSON documents. It does not extract members, hash complete
archives, read feature vectors, or alter any retrieval service.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import stat
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from src.ingest.b02_keyframe_mapping import (
    is_vfr_traffic_video,
    parse_b02_keyframe_row,
    trake_eligible,
)

MAP_ARCHIVE = "aic26-b2-map-keyframes.zip"
FEATURE_ARCHIVE = "aic26-b2-clip-features.zip"
MEDIA_ARCHIVE = "aic26-b2-media-info.zip"
OBJECT_ARCHIVES = ("Objects_M01_M10.zip", "Objects_N001_N100.zip", "Objects_S01.zip")
VIDEO_ID_PATTERN = re.compile(r"(?:M\d{2}_V\d{3}|N\d{3}-V\d{3}|S\d{2}-V\d{3})\Z")
MEDIA_S_ALIAS_PATTERN = re.compile(r"S(\d{2})_V(\d{3})\Z")
VERIFICATION_LEVEL = "zip-central-directory+csv-content+npy-header+media-json;not-full-archive-hash"


def _issue(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _catalog(archive: zipfile.ZipFile, path: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    """Fingerprint metadata only; ZIP CRCs are not full-content verification."""

    digest = hashlib.sha256()
    names: set[str] = set()
    member_count = 0
    uncompressed_bytes = 0
    for info in sorted(archive.infolist(), key=lambda item: item.filename):
        name = info.filename
        parts = PurePosixPath(name).parts
        if name.startswith("/") or "\\" in name or ".." in parts:
            errors.append(_issue("unsafe_member_path", f"{path.name}: {name}"))
        if name in names:
            errors.append(_issue("duplicate_zip_member", f"{path.name}: {name}"))
        names.add(name)
        mode = (info.external_attr >> 16) & 0o170000
        if info.create_system == 3 and mode == stat.S_IFLNK:
            errors.append(_issue("symlink_zip_member", f"{path.name}: {name}"))
        digest.update(
            _canonical_json((name, info.file_size, info.compress_size, info.CRC, info.is_dir()))
            + b"\n"
        )
        if not info.is_dir():
            member_count += 1
            uncompressed_bytes += info.file_size
    file_stat = path.stat()
    return {
        "archive_name": path.name,
        "archive_size_bytes": file_stat.st_size,
        "archive_mtime_utc": datetime.fromtimestamp(
            file_stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "member_count": member_count,
        "uncompressed_member_bytes": uncompressed_bytes,
        "central_directory_sha256": digest.hexdigest(),
        "verification_level": "central_directory_only;member_CRC_not_recomputed",
    }


def _video_id_from_member(
    member: str, suffix: str, errors: list[dict[str, str]], archive_name: str
) -> str | None:
    path = PurePosixPath(member)
    if path.suffix.lower() != suffix or len(path.parts) != 2:
        errors.append(_issue("invalid_member_layout", f"{archive_name}: {member}"))
        return None
    video_id = path.stem
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        errors.append(_issue("invalid_video_id", f"{archive_name}: {member}"))
        return None
    return video_id


def _read_maps(archive: zipfile.ZipFile, errors: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    members = sorted(
        (info for info in archive.infolist() if info.filename.lower().endswith(".csv")),
        key=lambda item: item.filename,
    )
    for info in members:
        video_id = _video_id_from_member(info.filename, ".csv", errors, MAP_ARCHIVE)
        if video_id is None:
            continue
        if video_id in maps:
            errors.append(_issue("duplicate_video_map", video_id))
            continue
        count = 0
        previous_pts: Decimal | None = None
        previous_ms: int | None = None
        previous_frame_idx: int | None = None
        first_ms: int | None = None
        source_indices: set[int] = set()
        frame_idx_collisions = 0
        rounded_ms_collisions = 0
        try:
            with archive.open(info) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
                    errors.append(_issue("invalid_map_header", f"{video_id}: {reader.fieldnames}"))
                    continue
                for row in reader:
                    count += 1
                    try:
                        mapped = parse_b02_keyframe_row("aic26-b2", video_id, row)
                        ordinal = mapped.identity.keyframe_ordinal
                        pts = Decimal(row["pts_time"])
                        fps = Decimal(row["fps"])
                        frame_idx = mapped.organizer_frame_idx
                    except (ValueError, TypeError, KeyError, InvalidOperation) as exc:
                        errors.append(_issue("invalid_map_row", f"{video_id}:{count}: {exc}"))
                        continue
                    if ordinal != count:
                        errors.append(
                            _issue("noncontiguous_ordinal", f"{video_id}:{count}: n={ordinal}")
                        )
                    if not pts.is_finite() or pts < 0 or not fps.is_finite() or fps <= 0:
                        errors.append(_issue("invalid_pts_or_fps", f"{video_id}:{count}"))
                        continue
                    if frame_idx < 0:
                        errors.append(_issue("negative_frame_idx", f"{video_id}:{count}"))
                    if previous_pts is not None and pts <= previous_pts:
                        errors.append(
                            _issue(
                                "nonincreasing_pts",
                                f"{video_id}: ordinals {count - 1},{count}; "
                                f"pts {previous_pts},{pts}; frame_idx "
                                f"{previous_frame_idx},{frame_idx}",
                            )
                        )
                    timestamp_ms = mapped.timestamp_ms
                    if previous_ms == timestamp_ms:
                        rounded_ms_collisions += 1
                    if frame_idx in source_indices:
                        frame_idx_collisions += 1
                    source_indices.add(frame_idx)
                    if first_ms is None:
                        first_ms = timestamp_ms
                    previous_pts = pts
                    previous_ms = timestamp_ms
                    previous_frame_idx = frame_idx
        except (UnicodeError, OSError, zipfile.BadZipFile) as exc:
            errors.append(_issue("unreadable_map", f"{video_id}: {exc}"))
        if count == 0:
            errors.append(_issue("empty_map", video_id))
        maps[video_id] = {
            "map_member": info.filename,
            "keyframe_count": count,
            "first_timestamp_ms": first_ms,
            "last_timestamp_ms": previous_ms,
            "source_frame_idx_collisions": frame_idx_collisions,
            "rounded_timestamp_ms_collisions": rounded_ms_collisions,
            "kis_qa_time_source": "pts_time",
            "is_vfr_traffic_video": is_vfr_traffic_video(video_id),
            "trake_eligible": trake_eligible(video_id),
        }
    if not maps:
        errors.append(_issue("no_maps", MAP_ARCHIVE))
    for info in archive.infolist():
        if not info.is_dir() and not info.filename.lower().endswith(".csv"):
            errors.append(_issue("unexpected_member_type", f"{MAP_ARCHIVE}: {info.filename}"))
    return maps


def _read_npy_header(raw: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    version = np.lib.format.read_magic(raw)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(raw)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(raw)
    else:
        raise ValueError(f"unsupported NPY version {version}")
    return shape, fortran, np.dtype(dtype), raw.tell()


def _read_features(
    archive: zipfile.ZipFile,
    maps: dict[str, dict[str, Any]],
    errors: list[dict[str, str]],
    expected_dimension: int,
) -> tuple[int, str | None, int]:
    seen: set[str] = set()
    dtypes: set[str] = set()
    dimensions: set[int] = set()
    rows_total = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        if not info.filename.lower().endswith(".npy"):
            errors.append(_issue("unexpected_member_type", f"{FEATURE_ARCHIVE}: {info.filename}"))
            continue
        video_id = _video_id_from_member(info.filename, ".npy", errors, FEATURE_ARCHIVE)
        if video_id is None:
            continue
        if video_id in seen:
            errors.append(_issue("duplicate_feature_video", video_id))
            continue
        seen.add(video_id)
        if video_id not in maps:
            errors.append(_issue("feature_without_map", video_id))
        try:
            with archive.open(info) as raw:
                shape, fortran, dtype, payload_offset = _read_npy_header(raw)
            if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
                raise ValueError(f"expected nonempty 2-D array, got {shape}")
            if fortran or not np.issubdtype(dtype, np.floating) or dtype.hasobject:
                raise ValueError(f"unsupported layout/dtype: fortran={fortran}, dtype={dtype}")
            expected_size = payload_offset + math.prod(shape) * dtype.itemsize
            if info.file_size != expected_size:
                raise ValueError(f"NPY member size {info.file_size} != {expected_size}")
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            errors.append(_issue("invalid_feature_header", f"{video_id}: {exc}"))
            continue
        rows_total += shape[0]
        dimensions.add(shape[1])
        dtypes.add(str(dtype))
        if video_id in maps:
            maps[video_id]["feature_member"] = info.filename
            maps[video_id]["feature_rows"] = shape[0]
            if shape[0] != maps[video_id]["keyframe_count"]:
                errors.append(
                    _issue(
                        "feature_map_row_mismatch",
                        f"{video_id}: {shape[0]} != {maps[video_id]['keyframe_count']}",
                    )
                )
    for video_id in sorted(set(maps) - seen):
        errors.append(_issue("map_without_feature", video_id))
    if dimensions != {expected_dimension}:
        errors.append(_issue("feature_dimension_mismatch", str(sorted(dimensions))))
    if len(dtypes) != 1:
        errors.append(_issue("inconsistent_feature_dtype", str(sorted(dtypes))))
    return (
        rows_total,
        next(iter(dtypes)) if len(dtypes) == 1 else None,
        next(iter(dimensions)) if len(dimensions) == 1 else 0,
    )


def _read_ordinal_members(
    archives: list[Path],
    *,
    kind: str,
    maps: dict[str, dict[str, Any]],
    inputs: dict[str, dict[str, Any]],
    errors: list[dict[str, str]],
) -> int:
    seen: dict[str, set[int]] = defaultdict(set)
    total = 0
    expected_suffixes = {".jpg", ".jpeg", ".png"} if kind == "keyframe" else {".json"}
    for path in archives:
        try:
            with zipfile.ZipFile(path) as archive:
                inputs[path.name] = _catalog(archive, path, errors)
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    member = PurePosixPath(info.filename)
                    if member.suffix.lower() not in expected_suffixes:
                        errors.append(
                            _issue("unexpected_member_type", f"{path.name}: {info.filename}")
                        )
                        continue
                    expected_parts = 3 if kind == "keyframe" else 2
                    if len(member.parts) != expected_parts or (
                        kind == "keyframe" and member.parts[0] != "keyframes"
                    ):
                        errors.append(
                            _issue("invalid_member_layout", f"{path.name}: {info.filename}")
                        )
                        continue
                    video_id = member.parts[-2]
                    if not VIDEO_ID_PATTERN.fullmatch(video_id):
                        errors.append(_issue("invalid_video_id", f"{path.name}: {video_id}"))
                        continue
                    try:
                        ordinal = int(member.stem)
                    except ValueError:
                        errors.append(
                            _issue("invalid_ordinal_filename", f"{path.name}: {info.filename}")
                        )
                        continue
                    if ordinal < 1:
                        errors.append(
                            _issue("invalid_ordinal_filename", f"{path.name}: {info.filename}")
                        )
                    if ordinal in seen[video_id]:
                        errors.append(
                            _issue("duplicate_ordinal_member", f"{kind}: {video_id}:{ordinal}")
                        )
                    seen[video_id].add(ordinal)
                    total += 1
                    if video_id in maps:
                        field = "keyframe_archive" if kind == "keyframe" else "object_archive"
                        existing = maps[video_id].get(field)
                        if existing is not None and existing != path.name:
                            errors.append(
                                _issue("video_split_across_archives", f"{kind}: {video_id}")
                            )
                        maps[video_id][field] = path.name
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(_issue("unreadable_archive", f"{path.name}: {exc}"))
    for video_id in sorted(set(maps) | set(seen)):
        actual = seen.get(video_id, set())
        expected_count = maps.get(video_id, {}).get("keyframe_count", 0)
        expected = set(range(1, expected_count + 1))
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            errors.append(
                _issue(
                    f"{kind}_ordinal_mismatch",
                    f"{video_id}: expected={expected_count}, actual={len(actual)}, "
                    f"missing_first={missing}, extra_first={extra}",
                )
            )
        if video_id in maps:
            maps[video_id][f"{kind}_member_count"] = len(actual)
    return total


def _read_media_info(
    archive: zipfile.ZipFile,
    maps: dict[str, dict[str, Any]],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> int:
    seen: set[str] = set()
    for info in archive.infolist():
        if info.is_dir():
            continue
        if not info.filename.lower().endswith(".json"):
            errors.append(_issue("unexpected_member_type", f"{MEDIA_ARCHIVE}: {info.filename}"))
            continue
        path = PurePosixPath(info.filename)
        if len(path.parts) != 2 or path.parts[0] != "media-info":
            errors.append(_issue("invalid_member_layout", f"{MEDIA_ARCHIVE}: {info.filename}"))
            continue
        raw_id = path.stem
        alias = MEDIA_S_ALIAS_PATTERN.fullmatch(raw_id)
        video_id = f"S{alias.group(1)}-V{alias.group(2)}" if alias else raw_id
        if video_id in seen:
            errors.append(_issue("duplicate_media_info", video_id))
            continue
        seen.add(video_id)
        if video_id not in maps:
            errors.append(_issue("media_info_without_map", raw_id))
            continue
        try:
            data = json.loads(archive.read(info))
            if not isinstance(data, dict):
                raise ValueError("media info must be a JSON object")
        except (ValueError, UnicodeError, zipfile.BadZipFile) as exc:
            errors.append(_issue("invalid_media_json", f"{raw_id}: {exc}"))
            continue
        maps[video_id]["media_info_member"] = info.filename
        if alias:
            maps[video_id]["media_info_source_video_id"] = raw_id
    missing_nontraffic = sorted(
        video_id for video_id in maps if not video_id.startswith("N") and video_id not in seen
    )
    for video_id in missing_nontraffic:
        errors.append(_issue("missing_media_info", video_id))
    missing_traffic = sorted(
        video_id for video_id in maps if video_id.startswith("N") and video_id not in seen
    )
    if missing_traffic:
        warnings.append(
            _issue(
                "traffic_media_info_unavailable",
                f"{len(missing_traffic)} N-series videos have no media-info; audio is unknown, not absent",
            )
        )
    return len(seen)


def audit_b02(
    root: Path,
    *,
    expected_videos: int | None = None,
    expected_keyframes: int | None = None,
    expected_feature_dimension: int = 512,
) -> dict[str, Any]:
    """Audit B02 archive structure without extraction or complete-payload hashing."""

    root = root.resolve(strict=True)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    inputs: dict[str, dict[str, Any]] = {}
    required = [MAP_ARCHIVE, FEATURE_ARCHIVE, MEDIA_ARCHIVE, *OBJECT_ARCHIVES]
    missing = [name for name in required if not (root / name).is_file()]
    for name in missing:
        errors.append(_issue("missing_required_archive", name))
    keyframe_paths = (
        sorted(root.glob("Keyframes_M*.zip"))
        + sorted(root.glob("Keyframes_N*.zip"))
        + sorted(root.glob("Keyframes_S*.zip"))
    )
    if not keyframe_paths:
        errors.append(_issue("missing_keyframe_archives", str(root)))
    if missing or not keyframe_paths:
        return {
            "schema_version": "b02-p0-audit-v1",
            "batch_id": "aic26-b2",
            "verification_level": VERIFICATION_LEVEL,
            "status": "blocked",
            "inputs": inputs,
            "errors": errors,
            "warnings": warnings,
        }
    with zipfile.ZipFile(root / MAP_ARCHIVE) as archive:
        inputs[MAP_ARCHIVE] = _catalog(archive, root / MAP_ARCHIVE, errors)
        maps = _read_maps(archive, errors)
    with zipfile.ZipFile(root / FEATURE_ARCHIVE) as archive:
        inputs[FEATURE_ARCHIVE] = _catalog(archive, root / FEATURE_ARCHIVE, errors)
        feature_rows, feature_dtype, feature_dimension = _read_features(
            archive, maps, errors, expected_feature_dimension
        )
    keyframe_count = _read_ordinal_members(
        keyframe_paths, kind="keyframe", maps=maps, inputs=inputs, errors=errors
    )
    object_count = _read_ordinal_members(
        [root / name for name in OBJECT_ARCHIVES],
        kind="object",
        maps=maps,
        inputs=inputs,
        errors=errors,
    )
    with zipfile.ZipFile(root / MEDIA_ARCHIVE) as archive:
        inputs[MEDIA_ARCHIVE] = _catalog(archive, root / MEDIA_ARCHIVE, errors)
        media_count = _read_media_info(archive, maps, errors, warnings)
    video_archives = (
        sorted(root.glob("Videos_M*.zip"))
        + sorted(root.glob("Videos_N*.zip"))
        + sorted(root.glob("Videos_S*.zip"))
    )
    for path in video_archives:
        with zipfile.ZipFile(path) as archive:
            inputs[path.name] = _catalog(archive, path, errors)
    if not video_archives:
        warnings.append(
            _issue(
                "video_archives_not_found_local",
                "B02 video ZIPs not found in audit root; external source not audited",
            )
        )
    if expected_videos is not None and len(maps) != expected_videos:
        errors.append(_issue("video_count_mismatch", f"{len(maps)} != {expected_videos}"))
    map_rows = sum(item["keyframe_count"] for item in maps.values())
    if expected_keyframes is not None and map_rows != expected_keyframes:
        errors.append(_issue("map_row_count_mismatch", f"{map_rows} != {expected_keyframes}"))
    if map_rows != keyframe_count or map_rows != object_count or map_rows != feature_rows:
        errors.append(
            _issue(
                "cross_artifact_count_mismatch",
                f"map={map_rows}, keyframes={keyframe_count}, objects={object_count}, features={feature_rows}",
            )
        )
    collision_videos = [
        video_id for video_id, item in maps.items() if item["source_frame_idx_collisions"]
    ]
    if collision_videos:
        warnings.append(
            _issue(
                "relative_frame_idx_collisions",
                f"{len(collision_videos)} videos; use ordinal identity and PTS-based KIS/QA time",
            )
        )
    rounded_collision_videos = [
        video_id for video_id, item in maps.items() if item["rounded_timestamp_ms_collisions"]
    ]
    if rounded_collision_videos:
        warnings.append(
            _issue(
                "rounded_timestamp_ms_collisions",
                f"{len(rounded_collision_videos)} videos; retain original PTS and ordinal identity",
            )
        )
    groups = {
        group: {
            "videos": sum(video_id.startswith(group) for video_id in maps),
            "keyframes": sum(
                item["keyframe_count"]
                for video_id, item in maps.items()
                if video_id.startswith(group)
            ),
        }
        for group in ("M", "N", "S")
    }
    result: dict[str, Any] = {
        "schema_version": "b02-p0-audit-v1",
        "batch_id": "aic26-b2",
        "verification_level": VERIFICATION_LEVEL,
        "status": "blocked" if errors else "structurally_ready",
        "inputs": dict(sorted(inputs.items())),
        "summary": {
            "videos": len(maps),
            "map_rows": map_rows,
            "keyframe_members": keyframe_count,
            "feature_rows": feature_rows,
            "object_members": object_count,
            "media_info_videos": media_count,
            "feature_dimension": feature_dimension,
            "feature_dtype": feature_dtype,
            "feature_model_provenance": "unverified_by_archive_header",
            "local_video_zip_count": len(video_archives),
            "relative_frame_idx_collision_videos": len(collision_videos),
            "rounded_timestamp_ms_collision_videos": len(rounded_collision_videos),
        },
        "groups": groups,
        "videos": dict(sorted(maps.items())),
        "errors": errors,
        "warnings": warnings,
    }
    result["manifest_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def inspect_pts_collision_evidence(
    root: Path,
    video_id: str,
    *,
    keyframe_archive: str,
    object_archive: str,
) -> dict[str, Any]:
    """Inspect one video's equal-PTS pairs without extracting image/object payloads.

    ZIP central-directory CRC/size values are compared for JPG/JSON; only the
    target video's NPY matrix is read to compare vector rows exactly.
    """

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"invalid video ID: {video_id}")
    root = root.resolve(strict=True)
    with zipfile.ZipFile(root / MAP_ARCHIVE) as archive:
        map_member = f"map-keyframes/{video_id}.csv"
        with archive.open(map_member) as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")))
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for index in range(1, len(rows)):
        if Decimal(rows[index - 1]["pts_time"]) == Decimal(rows[index]["pts_time"]):
            pairs.append((rows[index - 1], rows[index]))

    def ordinal_infos(archive: zipfile.ZipFile) -> dict[int, zipfile.ZipInfo]:
        result: dict[int, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if len(path.parts) < 2 or path.parts[-2] != video_id or not path.stem.isdigit():
                continue
            ordinal = int(path.stem)
            if ordinal in result:
                raise ValueError(f"duplicate member ordinal {ordinal} in {archive.filename}")
            result[ordinal] = info
        return result

    with (
        zipfile.ZipFile(root / FEATURE_ARCHIVE) as feature_zip,
        zipfile.ZipFile(root / keyframe_archive) as image_zip,
        zipfile.ZipFile(root / object_archive) as object_zip,
    ):
        matrix = np.load(
            io.BytesIO(feature_zip.read(f"clip-features/{video_id}.npy")),
            allow_pickle=False,
        )
        image_infos = ordinal_infos(image_zip)
        object_infos = ordinal_infos(object_zip)
        evidence_pairs: list[dict[str, Any]] = []
        for left, right in pairs:
            left_ordinal, right_ordinal = int(left["n"]), int(right["n"])
            left_image, right_image = image_infos[left_ordinal], image_infos[right_ordinal]
            left_object, right_object = object_infos[left_ordinal], object_infos[right_ordinal]
            image_equal = (left_image.CRC, left_image.file_size) == (
                right_image.CRC,
                right_image.file_size,
            )
            object_equal = (left_object.CRC, left_object.file_size) == (
                right_object.CRC,
                right_object.file_size,
            )
            vector_equal = bool(np.array_equal(matrix[left_ordinal - 1], matrix[right_ordinal - 1]))
            evidence_pairs.append(
                {
                    "ordinals": [left_ordinal, right_ordinal],
                    "pts_time": [left["pts_time"], right["pts_time"]],
                    "frame_idx": [int(left["frame_idx"]), int(right["frame_idx"])],
                    "image_same_zip_crc_and_size": image_equal,
                    "feature_vector_equal": vector_equal,
                    "object_same_zip_crc_and_size": object_equal,
                    "distinct_evidence": not (image_equal and vector_equal and object_equal),
                }
            )
    report: dict[str, Any] = {
        "schema_version": "b02-p0-pts-collision-evidence-v1",
        "batch_id": "aic26-b2",
        "video_id": video_id,
        "verification_level": "map+targeted_NPY_payload+image_object_zip_CRC_and_size;not_full_JPEG_JSON_hash",
        "source_archives": [MAP_ARCHIVE, FEATURE_ARCHIVE, keyframe_archive, object_archive],
        "equal_pts_pair_count": len(evidence_pairs),
        "distinct_evidence_pair_count": sum(item["distinct_evidence"] for item in evidence_pairs),
        "pairs": evidence_pairs,
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-videos", type=int)
    parser.add_argument("--expected-keyframes", type=int)
    parser.add_argument("--expected-feature-dimension", type=int, default=512)
    parser.add_argument(
        "--output", type=Path, help="Explicit manifest JSON path; no output file otherwise"
    )
    parser.add_argument("--collision-video", help="Target one video for equal-PTS evidence audit")
    parser.add_argument("--collision-keyframes-archive", help="ZIP name for target keyframes")
    parser.add_argument("--collision-objects-archive", help="ZIP name for target object JSON")
    parser.add_argument(
        "--collision-output", type=Path, help="Explicit targeted collision report path"
    )
    args = parser.parse_args(argv)
    destinations = [path.resolve() for path in (args.output, args.collision_output) if path]
    if len(destinations) != len(set(destinations)):
        parser.error("manifest and collision report must have different output paths")
    for destination in destinations:
        if destination.exists():
            parser.error(f"refusing to overwrite existing audit artifact: {destination}")
    result = audit_b02(
        args.root,
        expected_videos=args.expected_videos,
        expected_keyframes=args.expected_keyframes,
        expected_feature_dimension=args.expected_feature_dimension,
    )
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                + b"\n"
            )
    collision_summary = None
    if args.collision_video:
        if not args.collision_keyframes_archive or not args.collision_objects_archive:
            parser.error("collision inspection requires both archive names")
        evidence = inspect_pts_collision_evidence(
            args.root,
            args.collision_video,
            keyframe_archive=args.collision_keyframes_archive,
            object_archive=args.collision_objects_archive,
        )
        if args.collision_output is not None:
            output = args.collision_output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(
                    json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True).encode(
                        "utf-8"
                    )
                    + b"\n"
                )
        collision_summary = {
            "video_id": evidence["video_id"],
            "equal_pts_pair_count": evidence["equal_pts_pair_count"],
            "distinct_evidence_pair_count": evidence["distinct_evidence_pair_count"],
            "report_sha256": evidence["report_sha256"],
        }
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result.get("summary"),
                "manifest_sha256": result.get("manifest_sha256"),
                "errors": result["errors"][:10],
                "warnings": result["warnings"],
                "collision_evidence": collision_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "structurally_ready" else 2


if __name__ == "__main__":
    sys.exit(main())
