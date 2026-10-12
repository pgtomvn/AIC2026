"""Streaming importer for organizer keyframes, maps, vectors, and text metadata."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Literal

import numpy as np

from src.contracts import FrameRecord
from src.ingest.b02_keyframe_mapping import is_vfr_traffic_video, parse_b02_keyframe_row
from src.ingest.ledger import ArtifactSpec, JobLedger, StageState
from src.retrieval.query_plan import accent_fold, normalize_nfc


@dataclass(frozen=True)
class ReferenceStats:
    videos: int
    frames: int
    feature_rows: int
    object_documents: int
    metadata_documents: int


@dataclass(frozen=True)
class CanonicalReferenceStats:
    videos: int
    mapping_rows: int
    canonical_frames: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile, suffix: str) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {info.filename!r}")
        if not info.is_dir() and path.suffix.casefold() == suffix:
            members.append(info)
    return members


def _video_id(info: zipfile.ZipInfo) -> str:
    return PurePosixPath(info.filename).stem


def _frame_rows(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> list[dict[str, str]]:
    with archive.open(info) as raw:
        wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        rows = list(csv.DictReader(wrapper))
    required = {"n", "pts_time", "fps", "frame_idx"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid organizer frame map: {info.filename}")
    return rows


class OrganizerReferenceImporter:
    """Import reference packs per video without expanding them to disk."""

    def __init__(
        self,
        *,
        map_archive: str | Path,
        feature_archive: str | Path,
        keyframe_archives: Iterable[str | Path] = (),
        object_archive: str | Path | None = None,
        object_archives: Iterable[str | Path] = (),
        metadata_archive: str | Path | None = None,
        batch_id: str = "aic25-b1-reference",
        object_confidence: float = 0.50,
        identity_mode: Literal["source_frame_idx", "ordinal"] = "source_frame_idx",
        metadata_policy: Literal["exact", "partial"] = "exact",
        reference_manifest_sha256: str | None = None,
    ) -> None:
        if identity_mode not in ("source_frame_idx", "ordinal"):
            raise ValueError("identity_mode must be 'source_frame_idx' or 'ordinal'")
        if metadata_policy not in ("exact", "partial"):
            raise ValueError("metadata_policy must be 'exact' or 'partial'")
        supplied_object_archives = tuple(Path(path) for path in object_archives)
        if object_archive is not None and supplied_object_archives:
            raise ValueError("pass object_archive or object_archives, not both")
        self.map_archive = Path(map_archive)
        self.feature_archive = Path(feature_archive)
        self.keyframe_archives = tuple(Path(path) for path in keyframe_archives)
        self.object_archive = Path(object_archive) if object_archive else None
        self.object_archives = (
            (self.object_archive,) if self.object_archive else supplied_object_archives
        )
        self.metadata_archive = Path(metadata_archive) if metadata_archive else None
        self.batch_id = batch_id
        self.object_confidence = object_confidence
        self.identity_mode = identity_mode
        self.metadata_policy = metadata_policy
        if reference_manifest_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", reference_manifest_sha256
        ):
            raise ValueError("reference_manifest_sha256 must be a SHA-256 digest")
        self.reference_manifest_sha256 = reference_manifest_sha256

    def _indexes(self) -> tuple[dict[str, zipfile.ZipInfo], dict[str, zipfile.ZipInfo]]:
        with zipfile.ZipFile(self.map_archive) as maps:
            map_members = _safe_members(maps, ".csv")
            map_index = {_video_id(info): info for info in map_members}
        with zipfile.ZipFile(self.feature_archive) as features:
            feature_members = _safe_members(features, ".npy")
            feature_index = {_video_id(info): info for info in feature_members}
        if len(map_index) != len(map_members):
            raise ValueError("duplicate video IDs occur in the frame-map archive")
        if len(feature_index) != len(feature_members):
            raise ValueError("duplicate video IDs occur in the feature archive")
        if set(map_index) != set(feature_index):
            missing_features = sorted(set(map_index) - set(feature_index))[:5]
            missing_maps = sorted(set(feature_index) - set(map_index))[:5]
            raise ValueError(
                f"reference video mismatch; missing features={missing_features}, missing maps={missing_maps}"
            )
        return map_index, feature_index

    @staticmethod
    def _select_videos(
        map_index: dict[str, zipfile.ZipInfo], video_ids: Iterable[str] | None
    ) -> dict[str, zipfile.ZipInfo]:
        if video_ids is None:
            return map_index
        selected = set(video_ids)
        if not selected or selected - set(map_index):
            raise ValueError("selected videos must be a nonempty subset of the frame map")
        return {video_id: map_index[video_id] for video_id in sorted(selected)}

    def _payload(self, video_id: str, row: dict[str, str], image_uri: str) -> dict[str, Any]:
        ordinal = int(row["n"])
        source_frame_idx = int(row["frame_idx"])
        timestamp_ms = (
            parse_b02_keyframe_row(self.batch_id, video_id, row).timestamp_ms
            if self.identity_mode == "ordinal"
            else round(float(row["pts_time"]) * 1000)
        )
        fps = Fraction(str(row["fps"])).limit_denominator(100_000)
        payload = {
            "video_id": video_id,
            "keyframe_ordinal": ordinal,
            "source_frame_idx": source_frame_idx,
            "timestamp_ms": timestamp_ms,
            "fps_num": fps.numerator,
            "fps_den": fps.denominator,
            "shot_id": f"reference:{ordinal}",
            "image_uri": image_uri,
        }
        if self.identity_mode == "ordinal":
            payload["pts_time"] = row["pts_time"]
            if self.reference_manifest_sha256 is not None:
                payload["reference_manifest_sha256"] = self.reference_manifest_sha256
            payload["trake_eligible"] = (
                not is_vfr_traffic_video(video_id) and video_id != "M08_V001"
            )
        return payload

    def _metadata_video_id(self, raw_video_id: str) -> str:
        if self.identity_mode == "ordinal":
            match = re.fullmatch(r"S(\d{2})_V(\d{3})", raw_video_id)
            if match is not None:
                return f"S{match.group(1)}-V{match.group(2)}"
        return raw_video_id

    def _keyframe_archives_by_video(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for archive_path in self.keyframe_archives:
            with zipfile.ZipFile(archive_path) as archive:
                for info in _safe_members(archive, ".jpg"):
                    path = PurePosixPath(info.filename)
                    if len(path.parts) >= 2:
                        video_id = path.parts[-2]
                        previous = result.setdefault(video_id, archive_path)
                        if previous != archive_path:
                            raise ValueError(f"keyframes for {video_id} occur in multiple archives")
        return result

    def validate(
        self, *, expected_videos: int | None = None, expected_frames: int | None = None
    ) -> ReferenceStats:
        """Validate per-video row alignment without persisting derived data."""

        map_index, feature_index = self._indexes()
        frame_count = 0
        feature_count = 0
        feature_dimension: int | None = None
        with (
            zipfile.ZipFile(self.map_archive) as maps,
            zipfile.ZipFile(self.feature_archive) as features,
        ):
            for video_id in sorted(map_index):
                rows = _frame_rows(maps, map_index[video_id])
                array = np.load(
                    io.BytesIO(features.read(feature_index[video_id])), allow_pickle=False
                )
                if array.ndim != 2 or array.shape[0] != len(rows):
                    raise ValueError(
                        f"feature/map row mismatch for {video_id}: {array.shape} versus {len(rows)}"
                    )
                if feature_dimension is None:
                    feature_dimension = int(array.shape[1])
                elif int(array.shape[1]) != feature_dimension:
                    raise ValueError(
                        f"inconsistent feature dimension for {video_id}: {array.shape[1]} "
                        f"versus {feature_dimension}"
                    )
                ordinals = [int(row["n"]) for row in rows]
                if ordinals != list(range(1, len(rows) + 1)):
                    raise ValueError(f"non-contiguous keyframe ordinals for {video_id}")
                if self.identity_mode == "ordinal":
                    times = [
                        parse_b02_keyframe_row(self.batch_id, video_id, row).timestamp_ms
                        for row in rows
                    ]
                    if times != sorted(times):
                        raise ValueError(f"decreasing keyframe PTS for {video_id}")
                frame_count += len(rows)
                feature_count += int(array.shape[0])
        keyframe_count = 0
        if self.keyframe_archives:
            keyframes: dict[str, set[int]] = {}
            for archive_path in self.keyframe_archives:
                with zipfile.ZipFile(archive_path) as archive:
                    for info in _safe_members(archive, ".jpg"):
                        path = PurePosixPath(info.filename)
                        if len(path.parts) < 3 or not path.stem.isdigit():
                            continue
                        keyframes.setdefault(path.parts[-2], set()).add(int(path.stem))
            if set(keyframes) != set(map_index):
                raise ValueError("keyframe videos do not exactly match frame-map videos")
            with zipfile.ZipFile(self.map_archive) as maps:
                for video_id, info in map_index.items():
                    expected_ordinals = {int(row["n"]) for row in _frame_rows(maps, info)}
                    if keyframes[video_id] != expected_ordinals:
                        raise ValueError(f"keyframe ordinals do not match map for {video_id}")
                    keyframe_count += len(expected_ordinals)

        object_count = 0
        if self.object_archives:
            objects: dict[str, set[int]] = {}
            for archive_path in self.object_archives:
                with zipfile.ZipFile(archive_path) as archive:
                    for info in _safe_members(archive, ".json"):
                        path = PurePosixPath(info.filename)
                        if len(path.parts) < 2 or not path.stem.isdigit():
                            continue
                        video_id = path.parts[-2]
                        ordinal = int(path.stem)
                        ordinals = objects.setdefault(video_id, set())
                        if ordinal in ordinals:
                            raise ValueError(f"duplicate object ordinal {video_id}/{ordinal}")
                        ordinals.add(ordinal)
            if set(objects) != set(map_index):
                raise ValueError("object videos do not exactly match frame-map videos")
            with zipfile.ZipFile(self.map_archive) as maps:
                for video_id, info in map_index.items():
                    expected_ordinals = {int(row["n"]) for row in _frame_rows(maps, info)}
                    if objects[video_id] != expected_ordinals:
                        raise ValueError(f"object ordinals do not match map for {video_id}")
                    object_count += len(expected_ordinals)

        metadata_count = 0
        if self.metadata_archive is not None:
            with zipfile.ZipFile(self.metadata_archive) as archive:
                members = _safe_members(archive, ".json")
                metadata_videos = {self._metadata_video_id(_video_id(info)) for info in members}
            if len(metadata_videos) != len(members):
                raise ValueError("duplicate metadata videos after ID normalization")
            if self.metadata_policy == "exact":
                if metadata_videos != set(map_index):
                    raise ValueError("metadata videos do not exactly match frame-map videos")
            elif metadata_videos - set(map_index):
                raise ValueError("metadata videos have no frame map")
            elif self.identity_mode == "ordinal" and any(
                video_id not in metadata_videos and not is_vfr_traffic_video(video_id)
                for video_id in map_index
            ):
                raise ValueError("nontraffic videos are missing metadata")
            metadata_count = len(metadata_videos)

        if self.keyframe_archives and keyframe_count != frame_count:
            raise ValueError("keyframe count does not match frame-map count")
        if expected_videos is not None and len(map_index) != expected_videos:
            raise ValueError(f"expected {expected_videos} videos, found {len(map_index)}")
        if expected_frames is not None and frame_count != expected_frames:
            raise ValueError(f"expected {expected_frames} frames, found {frame_count}")
        return ReferenceStats(
            len(map_index), frame_count, feature_count, object_count, metadata_count
        )

    def import_vectors(
        self,
        store: Any,
        *,
        recreate: bool = False,
        expected_videos: int | None = None,
        expected_frames: int | None = None,
        video_ids: Iterable[str] | None = None,
        before_video: Callable[[str], None] | None = None,
    ) -> ReferenceStats:
        """Stream one video's feature matrix at a time into a vector store."""

        map_index, feature_index = self._indexes()
        map_index = self._select_videos(map_index, video_ids)
        keyframe_locations = self._keyframe_archives_by_video()
        store.create_collection(recreate=recreate)
        frame_count = 0
        with (
            zipfile.ZipFile(self.map_archive) as maps,
            zipfile.ZipFile(self.feature_archive) as features,
        ):
            for video_id in sorted(map_index):
                if before_video is not None:
                    before_video(video_id)
                rows = _frame_rows(maps, map_index[video_id])
                vectors = np.load(
                    io.BytesIO(features.read(feature_index[video_id])), allow_pickle=False
                )
                if vectors.ndim != 2 or vectors.shape[0] != len(rows):
                    raise ValueError(f"feature/map row mismatch for {video_id}")
                archive_path = keyframe_locations.get(video_id)
                ids: list[str] = []
                payloads: list[dict[str, Any]] = []
                for row in rows:
                    ordinal = int(row["n"])
                    image_uri = (
                        f"zip://{archive_path.resolve()}!/keyframes/{video_id}/{ordinal:03d}.jpg"
                        if archive_path is not None
                        else ""
                    )
                    payload = self._payload(video_id, row, image_uri)
                    payload["batch_id"] = self.batch_id
                    payload["feature_source"] = "organizer-clip-32"
                    # Some organizer maps assign the same submitted source
                    # frame to adjacent JPEG ordinals (614 rows in Batch 1).
                    # Preserve every supplied vector with an ordinal-qualified
                    # point ID; retrieval still deduplicates on the canonical
                    # (video_id, source_frame_idx) payload identity.
                    ids.append(f"{self.batch_id}:{video_id}:ordinal:{ordinal}")
                    payloads.append(payload)
                store.upsert(ids, vectors, payloads)
                frame_count += len(rows)
        stats = ReferenceStats(len(map_index), frame_count, frame_count, 0, 0)
        if expected_videos is not None and stats.videos != expected_videos:
            raise ValueError(f"expected {expected_videos} videos, found {stats.videos}")
        if expected_frames is not None and stats.frames != expected_frames:
            raise ValueError(f"expected {expected_frames} frames, found {stats.frames}")
        filters = {"batch_id": self.batch_id, "video_id": list(map_index)} if video_ids is not None else None
        count = store.count(filters) if filters else store.count()
        if count != frame_count:
            raise ValueError(f"vector index count {count} does not match {frame_count}")
        return stats

    def verify_vector_store(
        self,
        store: Any,
        *,
        expected_videos: int | None = None,
        expected_frames: int | None = None,
        video_ids: Iterable[str] | None = None,
    ) -> ReferenceStats:
        """Verify existing vector payload identities against every organizer map row."""

        map_index, _ = self._indexes()
        map_index = self._select_videos(map_index, video_ids)
        store.create_collection(recreate=False)
        total = 0
        with zipfile.ZipFile(self.map_archive) as maps:
            for video_id in sorted(map_index):
                rows = _frame_rows(maps, map_index[video_id])
                expected = sorted((int(row["n"]), int(row["frame_idx"])) for row in rows)
                payloads = list(
                    store.iter_payloads(filters={"batch_id": self.batch_id, "video_id": video_id})
                )
                actual: list[tuple[int, int]] = []
                for payload in payloads:
                    if payload.get("batch_id") != self.batch_id:
                        raise ValueError(f"wrong vector batch payload for {video_id}")
                    if payload.get("video_id") != video_id:
                        raise ValueError(f"wrong vector video payload for {video_id}")
                    try:
                        actual.append(
                            (
                                int(payload["keyframe_ordinal"]),
                                int(payload["source_frame_idx"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid canonical vector payload for {video_id}"
                        ) from error
                if sorted(actual) != expected:
                    raise ValueError(
                        f"vector payload identities do not match frame map for {video_id}"
                    )
                if self.identity_mode == "ordinal":
                    expected_payloads = {
                        int(row["n"]): self._payload(video_id, row, "") for row in rows
                    }
                    for payload in payloads:
                        ordinal = int(payload["keyframe_ordinal"])
                        expected_payload = expected_payloads[ordinal]
                        fields = ["timestamp_ms", "pts_time", "trake_eligible", "fps_num", "fps_den"]
                        if self.reference_manifest_sha256 is not None:
                            fields.append("reference_manifest_sha256")
                        for field in fields:
                            if payload.get(field) != expected_payload[field]:
                                raise ValueError(f"B02 vector {field} mismatch: {video_id}/{ordinal}")
                total += len(actual)
        stats = ReferenceStats(len(map_index), total, total, 0, 0)
        if expected_videos is not None and stats.videos != expected_videos:
            raise ValueError(f"expected {expected_videos} videos, found {stats.videos}")
        if expected_frames is not None and stats.frames != expected_frames:
            raise ValueError(f"expected {expected_frames} frames, found {stats.frames}")
        filters = {"batch_id": self.batch_id}
        if video_ids is not None:
            filters["video_id"] = list(map_index)
        if store.count(filters=filters) != total:
            raise ValueError(
                "organizer collection contains an unexpected number of points for the batch"
            )
        return stats

    def import_canonical_frames(
        self,
        repository: Any,
        *,
        ledger: JobLedger | None = None,
        vectors_ready: bool = False,
        competition: str = "aic25",
        video_ids: Iterable[str] | None = None,
        before_video: Callable[[str], None] | None = None,
    ) -> CanonicalReferenceStats:
        """Persist organizer maps as canonical frames without expanding JPEG ZIPs.

        The default Batch 1 path retains one row per source frame plus ordinal
        aliases. Explicit ordinal mode retains every Batch 2 keyframe, including
        rows with the same source frame or presentation timestamp.
        """

        map_index, _ = self._indexes()
        map_index = self._select_videos(map_index, video_ids)
        keyframe_locations = self._keyframe_archives_by_video()
        source_digest = _sha256(self.map_archive)
        frame_version = (
            "organizer-reference-ordinal-map-v1"
            if self.identity_mode == "ordinal"
            else "organizer-reference-map-v1"
        )
        frame_fingerprint = f"{frame_version}:{source_digest[:24]}"
        vector_fingerprint = f"organizer-reference-vectors-v1:{source_digest[:24]}"
        if ledger is not None:
            ledger.register_batch(
                self.batch_id,
                competition,
                source_label=str(self.map_archive.resolve()),
            )
        retrieval_changed = False

        mapping_rows = 0
        canonical_count = 0
        with zipfile.ZipFile(self.map_archive) as maps:
            for video_id in sorted(map_index):
                if before_video is not None:
                    before_video(video_id)
                rows = _frame_rows(maps, map_index[video_id])
                archive_path = keyframe_locations.get(video_id)
                canonical: dict[int, FrameRecord] = {}
                ordinal_to_source: dict[int, int] = {}
                previous_timestamp_ms = -1
                for row in rows:
                    ordinal = int(row["n"])
                    source_frame_idx = int(row["frame_idx"])
                    ordinal_to_source[ordinal] = source_frame_idx
                    if self.identity_mode == "source_frame_idx" and source_frame_idx in canonical:
                        continue
                    timestamp_ms = self._payload(video_id, row, "")["timestamp_ms"]
                    if self.identity_mode == "ordinal":
                        if ordinal != len(canonical) + 1:
                            raise ValueError(f"non-contiguous keyframe ordinals for {video_id}")
                        if timestamp_ms < previous_timestamp_ms:
                            raise ValueError(f"decreasing keyframe PTS for {video_id}")
                        previous_timestamp_ms = timestamp_ms
                    fps = Fraction(str(row["fps"])).limit_denominator(100_000)
                    image_uri = (
                        f"zip://{archive_path.resolve()}!/keyframes/{video_id}/{ordinal:03d}.jpg"
                        if archive_path is not None
                        else f"reference://{self.batch_id}/{video_id}/ordinal/{ordinal}"
                    )
                    identity = ordinal if self.identity_mode == "ordinal" else source_frame_idx
                    exact_pts = Fraction(str(row["pts_time"]))
                    canonical[identity] = FrameRecord(
                        batch_id=self.batch_id,
                        video_id=video_id,
                        keyframe_ordinal=ordinal,
                        source_frame_idx=source_frame_idx,
                        pts=(exact_pts.numerator if self.identity_mode == "ordinal" else timestamp_ms),
                        time_base=(f"1/{exact_pts.denominator}" if self.identity_mode == "ordinal" else "1/1000"),
                        timestamp_ms=timestamp_ms,
                        fps_num=fps.numerator,
                        fps_den=fps.denominator,
                        shot_id=f"reference:{ordinal}",
                        sampling_reasons=("organizer_reference",),
                        image_uri=image_uri,
                        video_uri=f"reference://{self.batch_id}/{video_id}",
                        source_sha256=source_digest,
                        extractor_version=frame_version,
                    )

                records = tuple(canonical.values())
                if self.identity_mode == "ordinal":
                    repository.upsert_ordinal_frames(
                        records, fingerprint=frame_fingerprint, index_ready=vectors_ready
                    )
                else:
                    repository.upsert_frames(records, fingerprint=frame_fingerprint)
                    repository.upsert_frame_mappings(self.batch_id, video_id, ordinal_to_source)
                mapping_rows += len(rows)
                canonical_count += len(records)

                if ledger is None:
                    continue
                source_uri = f"reference://{self.batch_id}/{video_id}"
                ledger.register_video(
                    self.batch_id,
                    video_id,
                    source_uri,
                    source_sha256=source_digest,
                )
                frame_stage = ledger.start_stage(
                    self.batch_id, video_id, "reference_frames", frame_fingerprint
                )
                if frame_stage.changed:
                    ledger.complete_stage(
                        self.batch_id,
                        video_id,
                        "reference_frames",
                        artifacts=(
                            ArtifactSpec(
                                "reference_frames",
                                f"sqlite://canonical/{self.batch_id}/{video_id}",
                                frame_fingerprint,
                                source_digest,
                                None,
                            ),
                        ),
                    )
                elif frame_stage.state is not StageState.COMPLETED:
                    raise RuntimeError(f"reference frame stage is already running for {video_id}")
                if vectors_ready:
                    vector_stage = ledger.start_stage(
                        self.batch_id, video_id, "reference_vectors", vector_fingerprint
                    )
                    if vector_stage.changed:
                        ledger.complete_stage(
                            self.batch_id,
                            video_id,
                            "reference_vectors",
                            artifacts=(
                                ArtifactSpec(
                                    "reference_vectors",
                                    f"qdrant://organizer/{self.batch_id}/{video_id}",
                                    vector_fingerprint,
                                    None,
                                    None,
                                ),
                            ),
                        )
                    elif vector_stage.state is not StageState.COMPLETED:
                        raise RuntimeError(
                            f"reference vector stage is already running for {video_id}"
                        )
                    activated = ledger.activate_video(
                        self.batch_id,
                        video_id,
                        required_stages=("reference_frames", "reference_vectors"),
                        required_artifacts={
                            "reference_frames": 1,
                            "reference_vectors": 1,
                        },
                    )
                    retrieval_changed = retrieval_changed or activated
        if ledger is not None and vectors_ready:
            ledger.activate_batch_if_ready(self.batch_id)
            if retrieval_changed:
                ledger.bump_retrieval_generation(self.batch_id)
        return CanonicalReferenceStats(len(map_index), mapping_rows, canonical_count)

    def _map_lookup(self) -> dict[tuple[str, int], dict[str, Any]]:
        lookup: dict[tuple[str, int], dict[str, Any]] = {}
        with zipfile.ZipFile(self.map_archive) as maps:
            for info in _safe_members(maps, ".csv"):
                video_id = _video_id(info)
                for row in _frame_rows(maps, info):
                    lookup[(video_id, int(row["n"]))] = self._payload(video_id, row, "")
        return lookup

    def import_objects(
        self, store: Any, *, create_index: bool = True,
        before_batch: Callable[[], None] | None = None,
    ) -> int:
        """Stream high-confidence organizer object labels into a lexical index."""

        if not self.object_archives:
            return 0
        mapping = self._map_lookup()
        if create_index:
            store.create_index()
        count = 0
        pending: list[dict[str, Any]] = []
        for archive_path in self.object_archives:
            with zipfile.ZipFile(archive_path) as archive:
                for info in _safe_members(archive, ".json"):
                    path = PurePosixPath(info.filename)
                    if len(path.parts) < 2 or not path.stem.isdigit():
                        continue
                    video_id = path.parts[-2]
                    ordinal = int(path.stem)
                    frame = mapping.get((video_id, ordinal))
                    if frame is None:
                        raise ValueError(f"object file has no frame map: {info.filename}")
                    payload = json.loads(archive.read(info))
                    scores = payload.get("detection_scores", [])
                    names = payload.get("detection_class_names", [])
                    entities = payload.get("detection_class_entities", [])
                    if len(names) < len(scores) and len(entities) < len(scores):
                        raise ValueError(f"misaligned object detections: {info.filename}")
                    labels: list[str] = []
                    for index, score in enumerate(scores):
                        if float(score) < self.object_confidence:
                            continue
                        entity = str(entities[index]).strip() if index < len(entities) else ""
                        name = str(names[index]).strip() if index < len(names) else ""
                        # Organizer names are often OpenImages MIDs. Keep readable
                        # class names only; never pollute lexical search with /m/*.
                        if name and not name.startswith("/m/"):
                            labels.append(name)
                        if entity:
                            labels.append(entity)
                    text = normalize_nfc(" ".join(dict.fromkeys(labels)))
                    if not text:
                        continue
                    pending.append(
                        {
                            "id": f"object:{self.batch_id}:{video_id}:ordinal:{ordinal}",
                            "batch_id": self.batch_id,
                            **frame,
                            "kind": "objects",
                            "text_raw": text,
                            "text_nfc": text,
                            "text_accent_folded": accent_fold(text),
                        }
                    )
                    if len(pending) >= 1_000:
                        if before_batch is not None:
                            before_batch()
                        store.add_documents(pending)
                        count += len(pending)
                        pending.clear()
        if pending:
            if before_batch is not None:
                before_batch()
            store.add_documents(pending)
            count += len(pending)
        return count

    def import_metadata(self, store: Any, *, create_index: bool = True) -> int:
        """Stream video-level title/description/keywords into lexical search."""

        if self.metadata_archive is None:
            return 0
        map_index, _ = self._indexes()
        if create_index:
            store.create_index()
        first_frames: dict[str, dict[str, str]] = {}
        with zipfile.ZipFile(self.map_archive) as maps:
            for video_id, info in map_index.items():
                first_frames[video_id] = _frame_rows(maps, info)[0]
        documents: list[dict[str, Any]] = []
        with zipfile.ZipFile(self.metadata_archive) as archive:
            for info in _safe_members(archive, ".json"):
                video_id = self._metadata_video_id(_video_id(info))
                if video_id not in first_frames:
                    raise ValueError(f"metadata has no frame map: {video_id}")
                value = json.loads(archive.read(info))
                keywords = value.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [keywords]
                parts = [value.get("title", ""), value.get("description", ""), *keywords]
                text = normalize_nfc(" ".join(str(part) for part in parts if part))
                if not text:
                    continue
                frame = self._payload(video_id, first_frames[video_id], "")
                documents.append(
                    {
                        "id": f"metadata:{self.batch_id}:{video_id}",
                        "batch_id": self.batch_id,
                        **frame,
                        "kind": "metadata",
                        "text_raw": text,
                        "text_nfc": text,
                        "text_accent_folded": accent_fold(text),
                        "title": str(value.get("title", "")),
                        "description": str(value.get("description", "")),
                    }
                )
        store.add_documents(documents)
        return len(documents)

    def verify_encoder(
        self,
        encoder: Any,
        *,
        sample_size: int = 16,
        minimum_median_cosine: float = 0.999,
    ) -> float:
        """Verify image preprocessing/model provenance against reference vectors."""

        if not self.keyframe_archives:
            raise ValueError("keyframe archives are required for encoder verification")
        _, feature_index = self._indexes()
        samples: list[tuple[Path, str, int, bytes]] = []
        for archive_path in self.keyframe_archives:
            with zipfile.ZipFile(archive_path) as archive:
                for info in _safe_members(archive, ".jpg"):
                    path = PurePosixPath(info.filename)
                    if len(path.parts) >= 3 and path.stem.isdigit():
                        samples.append(
                            (archive_path, path.parts[-2], int(path.stem), archive.read(info))
                        )
                        if len(samples) >= sample_size:
                            break
            if len(samples) >= sample_size:
                break
        if len(samples) < sample_size:
            raise ValueError(f"only {len(samples)} verification images were available")
        expected: list[np.ndarray] = []
        with zipfile.ZipFile(self.feature_archive) as features:
            arrays: dict[str, np.ndarray] = {}
            for _, video_id, ordinal, _ in samples:
                if video_id not in arrays:
                    arrays[video_id] = np.load(
                        io.BytesIO(features.read(feature_index[video_id])), allow_pickle=False
                    )
                expected.append(np.asarray(arrays[video_id][ordinal - 1], dtype=np.float32))
        with tempfile.TemporaryDirectory(prefix="aic-reference-verify-") as temporary_dir:
            paths: list[Path] = []
            for index, (_, _, _, contents) in enumerate(samples):
                target = Path(temporary_dir) / f"{index:03d}.jpg"
                target.write_bytes(contents)
                paths.append(target)
            actual = np.asarray(encoder.encode_images(paths), dtype=np.float32)
        expected_array = np.asarray(expected, dtype=np.float32)
        expected_array /= np.maximum(np.linalg.norm(expected_array, axis=1, keepdims=True), 1e-12)
        actual /= np.maximum(np.linalg.norm(actual, axis=1, keepdims=True), 1e-12)
        if actual.shape != expected_array.shape:
            raise ValueError(f"encoder shape {actual.shape} does not match {expected_array.shape}")
        median = float(np.median(np.sum(actual * expected_array, axis=1)))
        if median < minimum_median_cosine:
            raise ValueError(
                f"reference encoder verification failed: median cosine {median:.6f} "
                f"is below {minimum_median_cosine:.6f}"
            )
        return median
