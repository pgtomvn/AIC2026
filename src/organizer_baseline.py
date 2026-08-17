"""Evidence-strict inspection and exact search for organizer reference artifacts."""

from __future__ import annotations

import csv
import heapq
import io
import json
import math
import statistics
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


class OrganizerBaselineError(ValueError):
    """Raised when organizer artifacts violate the published baseline layout."""


@dataclass(frozen=True, slots=True)
class MapRow:
    """One organizer keyframe-map row."""

    ordinal: int
    pts_time: float
    fps: float
    source_frame_idx: int


@dataclass(frozen=True, slots=True)
class ReferenceArchiveStats:
    """Validated aggregate facts for one organizer reference pack."""

    videos: int
    map_rows: int
    feature_rows: int
    object_json_files: int | None
    feature_dimension: int
    feature_dtype: str
    norm_min: float
    norm_median: float
    norm_max: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BaselineHit:
    """One deterministic exact-cosine result."""

    video_id: str
    source_frame_idx: int
    keyframe_ordinal: int
    score: float

    @property
    def identity(self) -> tuple[str, int]:
        return self.video_id, self.source_frame_idx

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _video_id(member: str, suffix: str) -> str:
    path = PurePosixPath(member)
    if path.suffix.casefold() != suffix.casefold() or len(path.parts) < 2:
        raise OrganizerBaselineError(f"invalid organizer member path: {member}")
    value = path.stem
    if not value:
        raise OrganizerBaselineError(f"empty video ID in organizer member: {member}")
    return value


def read_frame_maps(map_archive: str | Path) -> dict[str, tuple[MapRow, ...]]:
    """Stream and validate all per-video organizer frame maps."""

    result: dict[str, tuple[MapRow, ...]] = {}
    with zipfile.ZipFile(map_archive) as archive:
        members = sorted(name for name in archive.namelist() if name.casefold().endswith(".csv"))
        if not members:
            raise OrganizerBaselineError("map archive contains no CSV files")
        for member in members:
            video_id = _video_id(member, ".csv")
            if video_id in result:
                raise OrganizerBaselineError(f"duplicate map for video {video_id}")
            with archive.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
                    raise OrganizerBaselineError(
                        f"unexpected map header for {video_id}: {reader.fieldnames}"
                    )
                rows: list[MapRow] = []
                for expected, value in enumerate(reader, start=1):
                    try:
                        row = MapRow(
                            ordinal=int(value["n"]),
                            pts_time=float(value["pts_time"]),
                            fps=float(value["fps"]),
                            source_frame_idx=int(value["frame_idx"]),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise OrganizerBaselineError(
                            f"invalid frame-map row for {video_id}"
                        ) from exc
                    if row.ordinal != expected:
                        raise OrganizerBaselineError(
                            f"non-contiguous keyframe ordinals for {video_id}"
                        )
                    if (
                        not math.isfinite(row.pts_time)
                        or not math.isfinite(row.fps)
                        or row.pts_time < 0
                        or row.fps <= 0
                        or row.source_frame_idx < 0
                    ):
                        raise OrganizerBaselineError(f"invalid frame-map values for {video_id}")
                    rows.append(row)
            if not rows:
                raise OrganizerBaselineError(f"empty frame map for {video_id}")
            result[video_id] = tuple(rows)
    return result


def _load_feature(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise OrganizerBaselineError(f"invalid numpy feature member: {member}") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise OrganizerBaselineError(f"feature matrix must be non-empty 2D: {member}")
    if not np.issubdtype(array.dtype, np.floating):
        raise OrganizerBaselineError(f"feature matrix must be floating point: {member}")
    if not np.all(np.isfinite(array)):
        raise OrganizerBaselineError(f"feature matrix contains NaN or infinity: {member}")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if np.any(norms == 0) or np.any(~np.isfinite(norms)):
        raise OrganizerBaselineError(f"feature matrix contains invalid vectors: {member}")
    return array


def inspect_reference_archives(
    map_archive: str | Path,
    feature_archive: str | Path,
    *,
    object_archive: str | Path | None = None,
    expected_dimension: int = 512,
    unit_norm_tolerance: float = 0.01,
) -> ReferenceArchiveStats:
    """Validate organizer maps/features/objects without extracting them."""

    maps = read_frame_maps(map_archive)
    feature_rows = 0
    dimension: int | None = None
    dtype: str | None = None
    all_norms: list[np.ndarray] = []
    seen: set[str] = set()
    with zipfile.ZipFile(feature_archive) as archive:
        members = sorted(name for name in archive.namelist() if name.casefold().endswith(".npy"))
        for member in members:
            video_id = _video_id(member, ".npy")
            if video_id in seen:
                raise OrganizerBaselineError(f"duplicate feature matrix for video {video_id}")
            seen.add(video_id)
            if video_id not in maps:
                raise OrganizerBaselineError(f"feature matrix has no frame map: {video_id}")
            array = _load_feature(archive, member)
            if array.shape[0] != len(maps[video_id]):
                raise OrganizerBaselineError(
                    f"feature/map row mismatch for {video_id}: {array.shape[0]} != "
                    f"{len(maps[video_id])}"
                )
            if dimension is None:
                dimension = int(array.shape[1])
                dtype = str(array.dtype)
            elif array.shape[1] != dimension or str(array.dtype) != dtype:
                raise OrganizerBaselineError(f"inconsistent feature schema for {video_id}")
            feature_rows += int(array.shape[0])
            all_norms.append(np.linalg.norm(array.astype(np.float64), axis=1))
    if seen != set(maps):
        missing = sorted(set(maps) - seen)[:3]
        raise OrganizerBaselineError(f"frame maps missing feature matrices: {missing}")
    if dimension != expected_dimension:
        raise OrganizerBaselineError(
            f"feature dimension {dimension} does not match expected {expected_dimension}"
        )
    norms = np.concatenate(all_norms)
    if np.max(np.abs(norms - 1.0)) > unit_norm_tolerance:
        raise OrganizerBaselineError("organizer features are outside the unit-norm tolerance")

    object_count: int | None = None
    if object_archive is not None:
        with zipfile.ZipFile(object_archive) as archive:
            object_count = sum(
                1 for name in archive.namelist() if name.casefold().endswith(".json")
            )
        if object_count != feature_rows:
            raise OrganizerBaselineError(
                f"object/feature count mismatch: {object_count} != {feature_rows}"
            )

    return ReferenceArchiveStats(
        videos=len(maps),
        map_rows=sum(len(rows) for rows in maps.values()),
        feature_rows=feature_rows,
        object_json_files=object_count,
        feature_dimension=int(dimension),
        feature_dtype=str(dtype),
        norm_min=float(norms.min()),
        norm_median=float(np.median(norms)),
        norm_max=float(norms.max()),
    )


def exact_archive_search(
    query_vector: np.ndarray,
    map_archive: str | Path,
    feature_archive: str | Path,
    *,
    top_k: int = 100,
) -> list[BaselineHit]:
    """Return deterministic exact cosine results streamed per video."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError("query_vector must be one finite vector")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("query_vector must have a non-zero finite norm")
    query = query / norm
    maps = read_frame_maps(map_archive)
    candidates: list[BaselineHit] = []
    with zipfile.ZipFile(feature_archive) as archive:
        members = sorted(name for name in archive.namelist() if name.casefold().endswith(".npy"))
        for member in members:
            video_id = _video_id(member, ".npy")
            rows = maps.get(video_id)
            if rows is None:
                raise OrganizerBaselineError(f"feature matrix has no frame map: {video_id}")
            array = _load_feature(archive, member)
            if array.shape != (len(rows), query.shape[0]):
                raise OrganizerBaselineError(f"feature/query schema mismatch for {video_id}")
            values = array.astype(np.float32, copy=False)
            row_norms = np.linalg.norm(values, axis=1, keepdims=True)
            scores = (values / row_norms) @ query
            count = min(top_k, len(rows))
            indices = np.argpartition(scores, -count)[-count:] if count < len(rows) else range(count)
            for index in indices:
                row = rows[int(index)]
                candidates.append(
                    BaselineHit(
                        video_id=video_id,
                        source_frame_idx=row.source_frame_idx,
                        keyframe_ordinal=row.ordinal,
                        score=float(scores[int(index)]),
                    )
                )
    return heapq.nsmallest(
        top_k,
        candidates,
        key=lambda hit: (-hit.score, hit.video_id, hit.source_frame_idx),
    )


def compare_rankings(
    baseline: Sequence[BaselineHit],
    candidate_hits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare candidate search results to an exact archive ranking."""

    baseline_ids = [hit.identity for hit in baseline]
    candidate_ids: list[tuple[str, int]] = []
    candidate_scores: dict[tuple[str, int], float] = {}
    for hit in candidate_hits:
        payload = hit.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("candidate hit is missing payload")
        identity = (str(payload["video_id"]), int(payload["source_frame_idx"]))
        candidate_ids.append(identity)
        candidate_scores[identity] = float(hit.get("score", 0.0))
    expected = set(baseline_ids)
    actual = set(candidate_ids)
    common = expected & actual
    baseline_scores = {hit.identity: hit.score for hit in baseline}
    score_delta = max(
        (abs(baseline_scores[item] - candidate_scores[item]) for item in common),
        default=None,
    )
    return {
        "top_1_agreement": bool(baseline_ids and candidate_ids and baseline_ids[0] == candidate_ids[0]),
        "identity_recall": len(common) / len(expected) if expected else 1.0,
        "jaccard": len(common) / len(expected | actual) if expected or actual else 1.0,
        "max_common_score_delta": score_delta,
        "baseline_count": len(baseline_ids),
        "candidate_count": len(candidate_ids),
    }


def load_query_file(path: str | Path) -> list[dict[str, str]]:
    """Load a JSON/JSONL/plain-text parity query file without inventing labels."""

    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    suffix = source.suffix.casefold()
    if suffix == ".json":
        payload = json.loads(text)
        values = payload.get("queries") if isinstance(payload, Mapping) else payload
        if not isinstance(values, list):
            raise ValueError("query JSON must be a list or contain a queries list")
    elif suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]

    result: list[dict[str, str]] = []
    for index, value in enumerate(values, start=1):
        if isinstance(value, str):
            query_id, query_text = f"q{index:03d}", value
        elif isinstance(value, Mapping):
            query_id = str(value.get("query_id", f"q{index:03d}"))
            query_text = str(value.get("text", ""))
        else:
            raise ValueError("query entries must be strings or objects")
        if not query_id.strip() or not query_text.strip():
            raise ValueError("query ID and text must be non-empty")
        result.append({"query_id": query_id.strip(), "text": query_text.strip()})
    if not result:
        raise ValueError("query file is empty")
    return result


def latency_summary(values: Iterable[float]) -> dict[str, float]:
    """Return stable latency summary values in milliseconds."""

    data = sorted(float(value) for value in values)
    if not data:
        raise ValueError("latency values are empty")
    return {
        "mean_ms": round(statistics.fmean(data), 3),
        "p50_ms": round(float(np.percentile(data, 50)), 3),
        "p95_ms": round(float(np.percentile(data, 95)), 3),
        "max_ms": round(max(data), 3),
    }
