"""Deterministic OCR quality sidecars and bounded lexical calibration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from src.retrieval.candidate_rescue import validate_rollout_mode

OCR_QUALITY_FEATURE_VERSION = "ocr-quality-v1"
_TOKEN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def normalize_ocr_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).casefold()
    return " ".join(_TOKEN.findall(text))


def _noise_ratio(value: object) -> float:
    text = str(value or "")
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 1.0
    noisy = sum(
        not (character.isalnum() or character in "-.,:%+/&'’")
        for character in visible
    )
    return noisy / len(visible)


def _bbox(polygons: object) -> tuple[float, float, float, float] | None:
    if not isinstance(polygons, (list, tuple)):
        return None
    points: list[tuple[float, float]] = []
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)):
            continue
        for point in polygon:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                points.append((float(point[0]), float(point[1])))
    if not points:
        return None
    xs, ys = zip(*points, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True, slots=True)
class OCRQualityFeature:
    segment_id: str
    video_id: str
    source_frame_idx: int
    confidence: float
    noise_ratio: float
    token_count: int
    character_count: int
    persistence: float
    position_stability: float | None
    normalized_area: float | None
    roi_available: bool
    normalized_text_sha256: str

    @property
    def quality(self) -> float:
        components = [
            (0.35, max(0.0, min(1.0, self.confidence))),
            (0.20, 1.0 - max(0.0, min(1.0, self.noise_ratio))),
            (0.10, min(1.0, self.token_count / 4.0)),
            (0.20, max(0.0, min(1.0, self.persistence))),
        ]
        if self.position_stability is not None:
            components.append((0.15, max(0.0, min(1.0, self.position_stability))))
        total = sum(weight for weight, _value in components)
        return sum(weight * value for weight, value in components) / total


@dataclass(frozen=True, slots=True)
class OCRQualityConfig:
    mode: str = "off"
    sidecar_path: str = "data/state/ocr-quality.sqlite3"
    min_multiplier: float = 0.65
    max_multiplier: float = 1.10
    candidate_limit: int = 200

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "OCRQualityConfig":
        raw = value or {}
        config = cls(
            mode=validate_rollout_mode(raw.get("mode", "off"), name="ocr_quality"),
            sidecar_path=str(raw.get("sidecar_path", "data/state/ocr-quality.sqlite3")),
            min_multiplier=float(raw.get("min_multiplier", 0.65)),
            max_multiplier=float(raw.get("max_multiplier", 1.10)),
            candidate_limit=int(raw.get("candidate_limit", 200)),
        )
        if not 0.0 < config.min_multiplier <= config.max_multiplier:
            raise ValueError("ocr_quality multiplier bounds are invalid")
        if config.candidate_limit < 1:
            raise ValueError("ocr_quality candidate_limit must be positive")
        return config


def build_ocr_quality_features(
    segments: Iterable[Mapping[str, Any]],
    *,
    image_sizes: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[OCRQualityFeature, ...]:
    """Build deterministic features without changing or filtering OCR records."""

    rows: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = segment.get("segment_id", segment.get("id"))
        video_id = segment.get("video_id")
        frame = segment.get("source_frame_idx")
        if (
            not isinstance(segment_id, str)
            or not segment_id
            or not isinstance(video_id, str)
            or not video_id
            or isinstance(frame, bool)
        ):
            raise ValueError("OCR quality input has an invalid deterministic identity")
        frame_idx = int(frame)
        text = segment.get("text_raw", segment.get("text", segment.get("ocr_text", "")))
        normalized = normalize_ocr_text(text)
        confidence = float(segment.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("OCR confidence must be in [0, 1]")
        rows.append(
            {
                "segment_id": segment_id,
                "video_id": video_id,
                "frame": frame_idx,
                "text": str(text or ""),
                "normalized": normalized,
                "confidence": confidence,
                "bbox": _bbox(segment.get("polygons")),
            }
        )
    rows.sort(key=lambda item: (item["video_id"], item["frame"], item["segment_id"]))
    frame_ordinals: dict[tuple[str, int], int] = {}
    for video_id in sorted({str(item["video_id"]) for item in rows}):
        frames = sorted(
            {int(item["frame"]) for item in rows if item["video_id"] == video_id}
        )
        frame_ordinals.update(
            ((video_id, frame), ordinal) for ordinal, frame in enumerate(frames)
        )
    text_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["normalized"]:
            text_groups.setdefault((row["video_id"], row["normalized"]), []).append(row)
    text_group_ordinals = {
        key: [frame_ordinals[(item["video_id"], item["frame"])] for item in group]
        for key, group in text_groups.items()
    }
    features: list[OCRQualityFeature] = []
    for row in rows:
        ordinal = frame_ordinals[(row["video_id"], row["frame"])]
        group_key = (row["video_id"], row["normalized"])
        group = text_groups.get(group_key, [])
        ordinals = text_group_ordinals.get(group_key, [])
        start = bisect_left(ordinals, ordinal - 2)
        stop = bisect_right(ordinals, ordinal + 2)
        neighbours = group[start:stop]
        persistent_frames = {int(other["frame"]) for other in neighbours}
        persistence = (
            min(1.0, len(persistent_frames) / 3.0) if row["normalized"] else 0.0
        )
        bbox = row["bbox"]
        size = (image_sizes or {}).get(row["video_id"])
        normalized_area: float | None = None
        if bbox is not None and size is not None and size[0] > 0 and size[1] > 0:
            normalized_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
            normalized_area /= float(size[0] * size[1])
            normalized_area = max(0.0, min(1.0, normalized_area))
        stability_values: list[float] = []
        if bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            scale = max(1.0, bbox[2] - bbox[0], bbox[3] - bbox[1])
            for other in neighbours:
                other_box = other["bbox"]
                if other is row or other_box is None:
                    continue
                other_cx = (other_box[0] + other_box[2]) / 2.0
                other_cy = (other_box[1] + other_box[3]) / 2.0
                distance = ((cx - other_cx) ** 2 + (cy - other_cy) ** 2) ** 0.5
                stability_values.append(max(0.0, 1.0 - distance / (2.0 * scale)))
        position_stability = (
            sum(stability_values) / len(stability_values) if stability_values else None
        )
        normalized = row["normalized"]
        features.append(
            OCRQualityFeature(
                segment_id=row["segment_id"],
                video_id=row["video_id"],
                source_frame_idx=row["frame"],
                confidence=row["confidence"],
                noise_ratio=_noise_ratio(row["text"]),
                token_count=len(_TOKEN.findall(normalized)),
                character_count=len(normalized.replace(" ", "")),
                persistence=persistence,
                position_stability=position_stability,
                normalized_area=normalized_area,
                roi_available=normalized_area is not None,
                normalized_text_sha256=hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
            )
        )
    return tuple(features)


def write_ocr_quality_sidecar(
    path: str | Path,
    features: Sequence[OCRQualityFeature],
    *,
    source_fingerprint: str,
) -> str:
    """Atomically publish an idempotent SQLite sidecar and return its fingerprint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = tuple(sorted(features, key=lambda item: item.segment_id))
    if len({item.segment_id for item in ordered}) != len(ordered):
        raise ValueError("OCR quality sidecar contains duplicate segment IDs")
    digest = hashlib.sha256(
        f"{OCR_QUALITY_FEATURE_VERSION}:{source_fingerprint}\n".encode("utf-8")
    )
    for item in ordered:
        digest.update(
            json.dumps(asdict(item), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    fingerprint = digest.hexdigest()
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE features (
                segment_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("feature_version", OCR_QUALITY_FEATURE_VERSION),
                ("source_fingerprint", source_fingerprint),
                ("fingerprint", fingerprint),
            ),
        )
        connection.executemany(
            "INSERT INTO features(segment_id, payload_json) VALUES (?, ?)",
            (
                (
                    item.segment_id,
                    json.dumps(asdict(item), sort_keys=True, separators=(",", ":")),
                )
                for item in ordered
            ),
        )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, target)
    return fingerprint


def write_ocr_quality_sidecar_streaming(
    path: str | Path,
    feature_groups: Iterable[Sequence[OCRQualityFeature]],
    *,
    source_fingerprint: str,
) -> tuple[str, int]:
    """Publish a sidecar from globally ordered groups without retaining the corpus."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        f"{OCR_QUALITY_FEATURE_VERSION}:{source_fingerprint}\n".encode("utf-8")
    )
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    connection = sqlite3.connect(temporary)
    count = 0
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE features (
                segment_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            """
        )
        for group in feature_groups:
            rows: list[tuple[str, str]] = []
            group_ids: set[str] = set()
            for item in sorted(group, key=lambda value: value.segment_id):
                if item.segment_id in group_ids:
                    raise ValueError("OCR quality sidecar contains duplicate segment IDs")
                group_ids.add(item.segment_id)
                payload = json.dumps(
                    asdict(item), sort_keys=True, separators=(",", ":")
                )
                digest.update(f"{item.video_id}\0{item.segment_id}\0".encode("utf-8"))
                digest.update(payload.encode("utf-8"))
                digest.update(b"\n")
                rows.append((item.segment_id, payload))
                count += 1
            try:
                connection.executemany(
                    "INSERT INTO features(segment_id, payload_json) VALUES (?, ?)", rows
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "OCR quality sidecar contains duplicate segment IDs"
                ) from error
            connection.commit()
        fingerprint = digest.hexdigest()
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("feature_version", OCR_QUALITY_FEATURE_VERSION),
                ("source_fingerprint", source_fingerprint),
                ("fingerprint", fingerprint),
                ("feature_count", str(count)),
            ),
        )
        connection.commit()
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    os.replace(temporary, target)
    return fingerprint, count


class OCRQualityStore:
    """Read-only, batch-loaded feature access with no constructor I/O."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, segment_ids: Sequence[str]) -> dict[str, OCRQualityFeature]:
        ordered = tuple(dict.fromkeys(value for value in segment_ids if value))
        if not ordered or not self.path.is_file():
            return {}
        uri = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            values: dict[str, OCRQualityFeature] = {}
            for start in range(0, len(ordered), 500):
                batch = ordered[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                for segment_id, payload in connection.execute(
                    f"SELECT segment_id, payload_json FROM features WHERE segment_id IN ({placeholders})",
                    batch,
                ):
                    values[str(segment_id)] = OCRQualityFeature(**json.loads(payload))
            return values
        finally:
            connection.close()


def calibrate_ocr_results(
    results: Sequence[Mapping[str, Any]],
    *,
    config: OCRQualityConfig,
    store: OCRQualityStore | None,
    exact_terms: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply bounded within-branch multipliers; never filter an OCR result."""

    diagnostics: dict[str, Any] = {
        "mode": config.mode,
        "candidate_count": len(results),
        "evaluated_candidate_count": min(len(results), config.candidate_limit),
        "feature_count": 0,
        "quoted_text_exemptions": 0,
        "applied": False,
    }
    if config.mode == "off" or store is None or not results:
        return [dict(item) for item in results], diagnostics
    selected = results[: config.candidate_limit]
    segment_ids = [
        str((item.get("payload") or item).get("segment_id", "")) for item in selected
    ]
    features = store.load(segment_ids)
    diagnostics["feature_count"] = len(features)
    normalized_terms = tuple(
        value for term in exact_terms if (value := normalize_ocr_text(term))
    )
    calibrated: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        value = dict(item)
        source = value.get("payload") if isinstance(value.get("payload"), Mapping) else value
        segment_id = str(source.get("segment_id", ""))
        feature = features.get(segment_id)
        multiplier = 1.0
        reason = "feature_unavailable"
        if feature is not None:
            quality = feature.quality
            text = normalize_ocr_text(
                source.get("text_raw", source.get("text", source.get("ocr_text", "")))
            )
            exempt = bool(text and any(term in text for term in normalized_terms))
            if exempt and feature.persistence < 1.0:
                quality = min(1.0, quality + 0.20 * (1.0 - feature.persistence))
                diagnostics["quoted_text_exemptions"] += 1
            multiplier = config.min_multiplier + quality * (
                config.max_multiplier - config.min_multiplier
            )
            multiplier = max(config.min_multiplier, min(config.max_multiplier, multiplier))
            reason = "quoted_text_exemption" if exempt else "bounded_quality"
        base_score = float(value.get("score", value.get("_rankingScore", 1.0 / rank)))
        value["ocr_quality_multiplier"] = multiplier
        value["ocr_quality_reason"] = reason
        value["_ocr_calibrated_score"] = base_score * multiplier
        calibrated.append(value)
    calibrated.sort(
        key=lambda item: (
            -float(item["_ocr_calibrated_score"]),
            str((item.get("payload") or item).get("video_id", "")),
            str((item.get("payload") or item).get("segment_id", "")),
        )
    )
    for item in calibrated:
        item.pop("_ocr_calibrated_score", None)
    diagnostics["applied"] = config.mode == "active" and bool(features)
    if config.mode != "active":
        return [dict(item) for item in results], diagnostics
    return calibrated, diagnostics


__all__ = [
    "OCR_QUALITY_FEATURE_VERSION",
    "OCRQualityConfig",
    "OCRQualityFeature",
    "OCRQualityStore",
    "build_ocr_quality_features",
    "calibrate_ocr_results",
    "normalize_ocr_text",
    "write_ocr_quality_sidecar",
    "write_ocr_quality_sidecar_streaming",
]
