"""Lazy, bounded organizer object evidence and 7x7 spatial scoring."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

from src.contracts import FrameRecord
from src.query.entity_extractor import fold_entity_text
from src.query.query_decomposer import SpatialConstraint

ObjectState = Literal[
    "ready",
    "not_applicable",
    "archive_missing",
    "member_missing",
    "invalid",
]


class FrameRepository(Protocol):
    def get_frame(
        self,
        batch_id: str,
        video_id: str,
        source_frame_idx: int,
    ) -> FrameRecord | None: ...

    def get_frames(
        self,
        batch_id: str,
        identities: Sequence[tuple[str, int]],
    ) -> Mapping[tuple[str, int], FrameRecord]: ...


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ObjectEvidence:
    state: ObjectState
    detections: tuple[ObjectDetection, ...] = ()
    cache_hit: bool = False
    read_ms: float = 0.0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class OrganizerObjectArchive:
    """Read organizer JSON members lazily without exposing archive paths."""

    def __init__(
        self,
        archive_path: str | Path,
        *,
        allowed_root: str | Path,
        batch_id: str = "aic25-b1-reference",
        confidence_threshold: float = 0.50,
        max_member_bytes: int = 1024 * 1024,
        max_entries: int = 2_048,
        max_cache_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        resolved_root = Path(allowed_root).resolve()
        resolved_archive = Path(archive_path).resolve()
        if not _is_relative_to(resolved_archive, resolved_root):
            raise ValueError("object archive must stay under the approved root")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if min(max_member_bytes, max_entries, max_cache_bytes) < 1:
            raise ValueError("object archive bounds must be positive")
        self.archive_path = resolved_archive
        self.batch_id = batch_id
        self.confidence_threshold = float(confidence_threshold)
        self.max_member_bytes = int(max_member_bytes)
        self.max_entries = int(max_entries)
        self.max_cache_bytes = int(max_cache_bytes)
        self._guard = threading.RLock()
        self._archive: zipfile.ZipFile | None = None
        self._members: frozenset[str] = frozenset()
        self._cache: OrderedDict[
            tuple[str, int], tuple[ObjectEvidence, int]
        ] = OrderedDict()
        self._cache_bytes = 0
        self._open_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def fingerprint(self) -> str:
        payload = (
            f"organizer-objects-v1:{self.batch_id}:{self.confidence_threshold}:"
            f"{self.max_member_bytes}:{self.max_entries}:{self.max_cache_bytes}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def diagnostics(self) -> dict[str, Any]:
        with self._guard:
            return {
                "archive_state": (
                    "ready"
                    if self._archive is not None
                    else "available"
                    if self.archive_path.is_file()
                    else "archive_missing"
                ),
                "archive_open_ms": round(self._open_ms, 3),
                "cache_entries": len(self._cache),
                "cache_bytes": self._cache_bytes,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

    def _open_locked(self) -> bool:
        if self._archive is not None:
            return True
        if not self.archive_path.is_file():
            return False
        started = time.monotonic()
        archive = zipfile.ZipFile(self.archive_path)
        members = frozenset(
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.endswith(".json")
        )
        self._archive = archive
        self._members = members
        self._open_ms = (time.monotonic() - started) * 1_000
        return True

    @staticmethod
    def _labels(payload: dict[str, Any]) -> Sequence[Any]:
        entities = payload.get("detection_class_entities")
        if isinstance(entities, list):
            return entities
        names = payload.get("detection_class_names")
        return names if isinstance(names, list) else ()

    def _parse(self, raw: bytes) -> tuple[ObjectDetection, ...]:
        try:
            payload = json.loads(raw.decode("utf-8"))
            scores = payload["detection_scores"]
            boxes = payload["detection_boxes"]
            labels = self._labels(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("object_evidence_invalid") from error
        if not isinstance(scores, list) or not isinstance(boxes, list):
            raise ValueError("object_evidence_invalid")
        detections: list[ObjectDetection] = []
        for score_value, label_value, box_value in zip(scores, labels, boxes):
            try:
                score = float(score_value)
                label = str(label_value).strip()
                if not isinstance(box_value, list) or len(box_value) != 4:
                    continue
                ymin, xmin, ymax, xmax = (float(value) for value in box_value)
            except (TypeError, ValueError):
                continue
            if (
                score < self.confidence_threshold
                or not math.isfinite(score)
                or not label
                or label.startswith("/m/")
                or not all(math.isfinite(value) for value in (ymin, xmin, ymax, xmax))
                or not (0.0 <= ymin < ymax <= 1.0)
                or not (0.0 <= xmin < xmax <= 1.0)
            ):
                continue
            detections.append(ObjectDetection(label, score, (ymin, xmin, ymax, xmax)))
        return tuple(detections)

    def get(self, frame: FrameRecord) -> ObjectEvidence:
        if frame.batch_id != self.batch_id:
            return ObjectEvidence("not_applicable")
        key = (frame.video_id, frame.keyframe_ordinal)
        with self._guard:
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
                self._cache_hits += 1
                value, _size = cached
                return ObjectEvidence(value.state, value.detections, True, value.read_ms)
            self._cache_misses += 1
            if not self._open_locked():
                return ObjectEvidence("archive_missing")
            member = f"objects/{frame.video_id}/{frame.keyframe_ordinal:03d}.json"
            if member not in self._members:
                return ObjectEvidence("member_missing")
            assert self._archive is not None
            try:
                info = self._archive.getinfo(member)
                if info.file_size < 1 or info.file_size > self.max_member_bytes:
                    return ObjectEvidence("invalid")
                started = time.monotonic()
                raw = self._archive.read(info)
                detections = self._parse(raw)
                value = ObjectEvidence(
                    "ready",
                    detections,
                    False,
                    (time.monotonic() - started) * 1_000,
                )
            except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, ValueError):
                return ObjectEvidence("invalid")
            size = len(raw)
            self._cache[key] = (value, size)
            self._cache_bytes += size
            while self._cache and (
                len(self._cache) > self.max_entries
                or self._cache_bytes > self.max_cache_bytes
            ):
                _old_key, (_old_value, old_size) = self._cache.popitem(last=False)
                self._cache_bytes -= old_size
            return value

    def warmup(self) -> dict[str, Any]:
        """Open only the central directory; no JSON member is decoded."""

        with self._guard:
            self._open_locked()
            return self.diagnostics

    def close(self) -> None:
        with self._guard:
            if self._archive is not None:
                self._archive.close()
            self._archive = None
            self._members = frozenset()
            self._cache.clear()
            self._cache_bytes = 0


_ZONE_CELLS: dict[str, frozenset[tuple[int, int]]] = {
    "left": frozenset((row, col) for row in range(7) for col in range(3)),
    "right": frozenset((row, col) for row in range(7) for col in range(4, 7)),
    "center": frozenset((row, col) for row in range(2, 5) for col in range(2, 5)),
    "top": frozenset((row, col) for row in range(3) for col in range(7)),
    "bottom": frozenset((row, col) for row in range(4, 7) for col in range(7)),
}
_LABEL_ALIAS_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("person", frozenset({"person", "people", "man", "woman", "driver", "rider", "chef", "người", "đàn ông", "phụ nữ", "tài xế", "đầu bếp"})),
    ("car", frozenset({"car", "automobile", "ô tô", "xe hơi"})),
    ("motorbike", frozenset({"motorbike", "motorcycle", "scooter", "xe máy"})),
    ("bicycle", frozenset({"bicycle", "bike", "cyclist", "xe đạp", "tay đua"})),
    ("hat", frozenset({"hat", "cap", "nón", "mũ"})),
    ("shirt", frozenset({"shirt", "t-shirt", "áo"})),
    ("trousers", frozenset({"trousers", "pants", "quần"})),
    ("dog", frozenset({"dog", "chó"})),
    ("cat", frozenset({"cat", "mèo"})),
)
_POSITIVE_LABEL_ALIAS_GROUPS: dict[str, frozenset[str]] = {
    "vehicle": frozenset(
        {
            "vehicle",
            "land vehicle",
            "car",
            "automobile",
            "boat",
            "tank",
            "motorbike",
            "motorcycle",
            "bicycle",
            "bus",
            "truck",
        }
    ),
}


@lru_cache(maxsize=1)
def _folded_alias_groups() -> tuple[tuple[str, frozenset[str]], ...]:
    """Normalize the fixed vocabulary once, not for each cold detector label."""
    return tuple(
        (canonical, frozenset(fold_entity_text(alias) for alias in aliases))
        for canonical, aliases in _LABEL_ALIAS_GROUPS
    )


@lru_cache(maxsize=4096)
def _canonical_label_tokens(value: str) -> frozenset[str]:
    folded = fold_entity_text(value)
    tokens = {token for token in folded.split() if len(token) > 2}
    for canonical, aliases in _folded_alias_groups():
        padded = f" {folded} "
        matched = False
        for alias_folded in aliases:
            if alias_folded == folded or f" {alias_folded} " in padded:
                matched = True
                break
        if matched:
            tokens.add(canonical)
    return tokens





@lru_cache(maxsize=4096)
def labels_compatible(subject: str, label: str) -> bool:
    subject_folded = fold_entity_text(subject)
    label_folded = fold_entity_text(label)
    if not subject_folded or not label_folded:
        return False
    subject_padded = f" {subject_folded} "
    label_padded = f" {label_folded} "
    if label_padded in subject_padded or subject_padded in label_padded:
        return True
    subject_tokens = _canonical_label_tokens(subject)
    label_tokens = _canonical_label_tokens(label)
    return bool(subject_tokens & label_tokens)


@lru_cache(maxsize=4096)
def labels_positive_compatible(subject: str, label: str) -> bool:
    """Allow broad taxonomy matches only for positive localization evidence."""

    if labels_compatible(subject, label):
        return True
    subject_folded = fold_entity_text(subject)
    label_folded = fold_entity_text(label)
    for canonical, aliases in _POSITIVE_LABEL_ALIAS_GROUPS.items():
        if subject_folded == canonical and label_folded in aliases:
            return True
    return False


def occupied_cells(box: tuple[float, float, float, float]) -> frozenset[tuple[int, int]]:
    ymin, xmin, ymax, xmax = box
    row_start = max(0, min(6, int(math.floor(ymin * 7))))
    row_end = max(0, min(6, int(math.ceil(ymax * 7) - 1)))
    col_start = max(0, min(6, int(math.floor(xmin * 7))))
    col_end = max(0, min(6, int(math.ceil(xmax * 7) - 1)))
    return frozenset(
        (row, col)
        for row in range(row_start, row_end + 1)
        for col in range(col_start, col_end + 1)
    )


def spatial_constraint_score(
    constraint: SpatialConstraint,
    detections: Sequence[ObjectDetection],
) -> tuple[float | None, tuple[ObjectDetection, ...]]:
    if not constraint.bound or constraint.subject is None:
        return None, ()
    matching = tuple(
        value for value in detections if labels_compatible(constraint.subject, value.label)
    )
    if not matching:
        return None, ()
    occupancy: set[tuple[int, int]] = set()
    for detection in matching:
        occupancy.update(occupied_cells(detection.box))
    zone = _ZONE_CELLS[constraint.zone]
    return len(occupancy & zone) / len(zone), matching
