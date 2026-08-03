"""Resumable raw-video ingestion for the Textual KIS core index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from src.contracts import FrameRecord, TextSegment
from src.extraction.asr_engine import ASREngine
from src.extraction.audio import AudioExtractor
from src.extraction.keyframe_extractor import KeyframeExtractor
from src.extraction.media_probe import MediaProbe, probe_media
from src.extraction.ocr_engine import OCREngine
from src.ingest.ledger import ArtifactSpec, JobLedger, StageState

_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"})


def _safe_zip_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name.replace("\\", "/"))
    if (
        member.is_absolute()
        or ".." in member.parts
        or (member.parts and re.fullmatch(r"[A-Za-z]:", member.parts[0]))
    ):
        raise ValueError(f"unsafe archive member: {name!r}")
    return member


def fingerprint(value: Any) -> str:
    """Return a short deterministic JSON fingerprint for stage invalidation."""

    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:24]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: Any, path: Path) -> ArtifactSpec:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return ArtifactSpec("manifest", str(path), "", _sha256(path), path.stat().st_size)


@dataclass(frozen=True)
class VideoSource:
    """A loose video or one safe video member inside an archive."""

    video_id: str
    path: Path
    member: str | None = None
    size_bytes: int | None = None

    @property
    def uri(self) -> str:
        if self.member is None:
            return str(self.path.resolve())
        return f"zip://{self.path.resolve()}!/{self.member}"


def discover_video_sources(paths: Iterable[str | Path]) -> list[VideoSource]:
    """Discover loose videos and safe ZIP members without extracting them."""

    sources: list[VideoSource] = []
    for raw_path in paths:
        path = Path(raw_path)
        candidates = [path]
        if path.is_dir():
            candidates = sorted(item for item in path.rglob("*") if item.is_file())
        for candidate in candidates:
            if candidate.name.casefold().endswith(".crdownload"):
                continue
            if candidate.suffix.casefold() in _VIDEO_SUFFIXES:
                sources.append(VideoSource(candidate.stem, candidate))
            elif candidate.suffix.casefold() == ".zip":
                with zipfile.ZipFile(candidate) as archive:
                    for info in archive.infolist():
                        member = _safe_zip_member(info.filename)
                        if not info.is_dir() and member.suffix.casefold() in _VIDEO_SUFFIXES:
                            sources.append(
                                VideoSource(member.stem, candidate, info.filename, info.file_size)
                            )
    identities = [source.video_id for source in sources]
    if len(identities) != len(set(identities)):
        duplicates = sorted({value for value in identities if identities.count(value) > 1})
        raise ValueError(f"duplicate video IDs across inputs: {duplicates[:5]}")
    return sorted(sources, key=lambda source: source.video_id)


class ScratchManager:
    """Materialize at most one archive member under a bounded scratch root."""

    def __init__(
        self,
        root: str | Path,
        *,
        limit_bytes: int = 12 * 1024**3,
        reserve_bytes: int = 30 * 1024**3,
    ) -> None:
        self.root = Path(root)
        self.limit_bytes = limit_bytes
        self.reserve_bytes = reserve_bytes
        self._archive_lock = threading.Lock()

    def _available(self) -> int:
        probe = self.root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free

    @staticmethod
    def _safe_component(value: str) -> str:
        if not value or value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(f"unsafe scratch path component: {value!r}")
        return value

    def _usage(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    @contextmanager
    def workspace(
        self,
        video_id: str,
        purpose: str,
        *,
        required_bytes: int = 0,
    ) -> Iterator[Path]:
        """Create and safely clean one bounded per-stage scratch workspace."""

        video_id = self._safe_component(video_id)
        purpose = self._safe_component(purpose)
        if required_bytes < 0:
            raise ValueError("required scratch bytes cannot be negative")
        if self._usage() + required_bytes > self.limit_bytes:
            raise OSError("scratch workspace would violate the configured scratch limit")
        if self._available() - required_bytes < self.reserve_bytes:
            raise OSError("scratch workspace would violate the free-space reserve")
        parent = self.root / video_id
        parent.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=f"{purpose}-", dir=parent))
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
            try:
                parent.rmdir()
            except OSError:
                pass

    @contextmanager
    def materialize(self, source: VideoSource) -> Iterator[Path]:
        if source.member is None:
            yield source.path
            return
        with self._archive_lock:
            video_id = self._safe_component(source.video_id)
            size = int(source.size_bytes or 0)
            if size <= 0 or size > self.limit_bytes:
                raise OSError(f"archive member size {size} violates the scratch limit")
            free = self._available()
            if free - size < self.reserve_bytes:
                raise OSError(
                    f"materializing {source.video_id} would violate the "
                    f"{self.reserve_bytes}-byte reserve"
                )
            target_dir = self.root / video_id
            target_dir.mkdir(parents=True, exist_ok=True)
            suffix = PurePosixPath(source.member).suffix
            target = target_dir / f"source{suffix}"
            temporary = target_dir / f".{target.name}.tmp"
            try:
                with zipfile.ZipFile(source.path) as archive:
                    info = archive.getinfo(source.member)
                    _safe_zip_member(info.filename)
                    with archive.open(info) as raw, temporary.open("wb") as output:
                        shutil.copyfileobj(raw, output, length=8 * 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                if temporary.stat().st_size != size:
                    raise OSError(f"scratch size mismatch for {source.video_id}")
                os.replace(temporary, target)
                yield target
            finally:
                temporary.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                try:
                    target_dir.rmdir()
                except OSError:
                    pass


@dataclass(frozen=True)
class PreparedVideo:
    source: VideoSource
    probe: MediaProbe
    frames: tuple[FrameRecord, ...]


class NativeIngestor:
    """Coordinate CPU preparation, one GPU visual worker, then enrichment."""

    def __init__(
        self,
        *,
        batch_id: str,
        competition: str,
        ledger: JobLedger,
        artifacts: Any,
        extractor: KeyframeExtractor,
        visual_encoder: Any,
        visual_store: Any,
        scratch: ScratchManager,
        state_root: str | Path = "data/state/manifests",
        ffprobe_path: str | Path | None = None,
        ocr_engine: OCREngine | None = None,
        asr_engine: ASREngine | None = None,
        audio_extractor: AudioExtractor | None = None,
        ocr_store: Any = None,
        asr_store: Any = None,
        text_encoder: Any = None,
        text_vector_store: Any = None,
        additional_visual_channels: Iterable[tuple[str, Any, Any]] = (),
    ) -> None:
        self.batch_id = batch_id
        self.competition = competition
        self.ledger = ledger
        self.artifacts = artifacts
        self.extractor = extractor
        self.visual_encoder = visual_encoder
        self.visual_store = visual_store
        self.scratch = scratch
        self.state_root = Path(state_root)
        self.ffprobe_path = Path(ffprobe_path) if ffprobe_path is not None else None
        self.ocr_engine = ocr_engine
        self.asr_engine = asr_engine
        self.audio_extractor = audio_extractor
        self.ocr_store = ocr_store
        self.asr_store = asr_store
        self.text_encoder = text_encoder
        self.text_vector_store = text_vector_store
        self.additional_visual_channels = tuple(additional_visual_channels)

    @staticmethod
    def _unload_model(model: Any) -> None:
        unload = getattr(model, "unload", None)
        if callable(unload):
            unload()

    def _unload_visual_models(self) -> None:
        """Hand GPU ownership from all visual encoders to the ASR stage."""

        self._unload_model(self.visual_encoder)
        for _, encoder, _ in self.additional_visual_channels:
            self._unload_model(encoder)

    def initialize(
        self,
        *,
        recreate_indexes: bool = False,
        recover_interrupted: bool = True,
    ) -> None:
        """Explicitly initialize state and index schemas."""

        self.ledger.initialize()
        self.artifacts.initialize()
        self.ledger.register_batch(self.batch_id, self.competition)
        if recover_interrupted:
            self.ledger.recover_interrupted(self.batch_id)
        self.visual_store.create_collection(recreate=recreate_indexes)
        for _, _, store in self.additional_visual_channels:
            store.create_collection(recreate=recreate_indexes)
        if self.ocr_store is not None:
            self.ocr_store.create_index()
        if self.asr_store is not None:
            self.asr_store.create_index()
        if self.text_vector_store is not None:
            self.text_vector_store.create_collection(recreate=recreate_indexes)

    def _stage(
        self,
        source: VideoSource,
        stage_name: str,
        stage_fingerprint: str,
        operation: Any,
        *,
        allow_reprocess: bool = False,
    ) -> Any:
        transition = self.ledger.start_stage(
            self.batch_id,
            source.video_id,
            stage_name,
            stage_fingerprint,
            allow_reprocess=allow_reprocess,
        )
        if not transition.changed:
            if transition.state is StageState.COMPLETED:
                return None
            raise RuntimeError(
                f"stage {stage_name} for {source.video_id} is already running; "
                "recover interrupted stages before retrying"
            )
        try:
            result, artifacts = operation()
            normalized = [
                replace(artifact, fingerprint=stage_fingerprint) for artifact in artifacts
            ]
            self.ledger.complete_stage(
                self.batch_id,
                source.video_id,
                stage_name,
                artifacts=normalized,
            )
            return result
        except Exception as error:
            self.ledger.fail_stage(
                self.batch_id, source.video_id, stage_name, f"{type(error).__name__}: {error}"
            )
            raise

    def _prepare(self, source: VideoSource) -> PreparedVideo:
        self.ledger.register_video(self.batch_id, source.video_id, source.uri)
        source_stat = source.path.stat()
        probe_fingerprint = fingerprint(
            {
                "probe": "v1",
                "source": source.uri,
                "archive_size": source_stat.st_size,
                "archive_mtime_ns": source_stat.st_mtime_ns,
            }
        )
        probe_path = self.state_root / self.batch_id / f"{source.video_id}.probe.json"
        with self.scratch.materialize(source) as media_path:

            def run_probe() -> tuple[MediaProbe, list[ArtifactSpec]]:
                value = (
                    probe_media(media_path)
                    if self.ffprobe_path is None
                    else probe_media(media_path, ffprobe_path=self.ffprobe_path)
                )
                artifact = _atomic_json(value.to_dict(), probe_path)
                return value, [replace(artifact, kind="media_probe")]

            probe = self._stage(source, "probe", probe_fingerprint, run_probe)
            if probe is None:
                probe = MediaProbe(**json.loads(probe_path.read_text(encoding="utf-8")))

            frame_fingerprint = fingerprint(
                {
                    "extractor": self.extractor.extractor_version,
                    "scene": self.extractor.min_scene_change,
                    "max_gap": self.extractor.max_gap_seconds,
                    "min_separation": self.extractor.min_separation_seconds,
                    "max_dim": self.extractor.resize_max_dim,
                }
            )

            def run_frames() -> tuple[tuple[FrameRecord, ...], list[ArtifactSpec]]:
                raw_records = self.extractor.extract(media_path, video_id=source.video_id)
                frame_fields = set(FrameRecord.__dataclass_fields__)
                frames = tuple(
                    FrameRecord.from_dict(
                        {
                            key: value
                            for key, value in {
                                **record,
                                "batch_id": self.batch_id,
                                "video_id": source.video_id,
                                "video_uri": source.uri,
                                "sampling_reasons": tuple(record["sampling_reasons"]),
                            }.items()
                            if key in frame_fields
                        }
                    )
                    for record in raw_records
                )
                self.ledger.register_video(
                    self.batch_id,
                    source.video_id,
                    source.uri,
                    source_sha256=frames[0].source_sha256,
                )
                self.artifacts.upsert_frames(frames, fingerprint=frame_fingerprint)
                manifest_path = self.state_root / self.batch_id / f"{source.video_id}.frames.json"
                artifact = _atomic_json([frame.to_dict() for frame in frames], manifest_path)
                return frames, [replace(artifact, kind="frames_manifest")]

            frames = self._stage(source, "frames", frame_fingerprint, run_frames)
        if frames is None:
            frames = tuple(self.artifacts.list_frames(self.batch_id, source.video_id))
        return PreparedVideo(source, probe, frames)

    def _index_visual(self, prepared: PreparedVideo) -> None:
        stage_fingerprint = fingerprint(
            {
                "encoder": self.visual_encoder.fingerprint,
                "collection": self.visual_store.collection_name,
                "frame_count": len(prepared.frames),
            }
        )

        def operation() -> tuple[None, list[ArtifactSpec]]:
            paths = [Path(frame.image_uri) for frame in prepared.frames]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} retained thumbnails are missing; first={missing[0]}"
                )
            if len(set(paths)) != len(paths):
                raise ValueError("retained frames must have unique thumbnail paths")
            vectors = self.visual_encoder.encode_images(paths)
            if vectors.shape[0] != len(prepared.frames):
                raise ValueError("visual encoder returned the wrong number of vectors")
            ids = [
                f"{frame.batch_id}:{frame.video_id}:{frame.source_frame_idx}"
                for frame in prepared.frames
            ]
            payloads = [frame.to_dict() for frame in prepared.frames]
            for payload in payloads:
                payload["model_fingerprint"] = self.visual_encoder.fingerprint
            self.visual_store.upsert(ids, vectors, payloads)
            indexed_count = self.visual_store.count(
                filters={
                    "batch_id": self.batch_id,
                    "video_id": prepared.source.video_id,
                }
            )
            if indexed_count != len(prepared.frames):
                raise ValueError(
                    f"visual index count {indexed_count} does not match retained frame "
                    f"count {len(prepared.frames)} for {self.batch_id}/{prepared.source.video_id}"
                )
            artifact = ArtifactSpec(
                "visual_vectors",
                f"qdrant://{self.visual_store.collection_name}/{prepared.source.video_id}",
                stage_fingerprint,
                None,
                int(vectors.nbytes),
            )
            return None, [artifact]

        allow_reprocess = False
        existing = self.ledger.get_stage(self.batch_id, prepared.source.video_id, "visual")
        if existing is not None and existing["state"] == StageState.COMPLETED.value:
            try:
                paths = [Path(frame.image_uri) for frame in prepared.frames]
                if any(not path.is_file() for path in paths):
                    raise FileNotFoundError("one or more retained thumbnails are missing")
                indexed_count = self.visual_store.count(
                    filters={
                        "batch_id": self.batch_id,
                        "video_id": prepared.source.video_id,
                    }
                )
                if indexed_count != len(prepared.frames):
                    raise ValueError(
                        f"visual index count {indexed_count} does not match retained frame "
                        f"count {len(prepared.frames)}"
                    )
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                self.ledger.deactivate_video(
                    self.batch_id,
                    prepared.source.video_id,
                    f"visual activation revalidation failed: {error}",
                )
                allow_reprocess = True
        self._stage(
            prepared.source,
            "visual",
            stage_fingerprint,
            operation,
            allow_reprocess=allow_reprocess,
        )
        # Always validate the durable state, including the completed-stage
        # resume path where the operation itself was intentionally skipped.
        paths = [Path(frame.image_uri) for frame in prepared.frames]
        if any(not path.is_file() for path in paths):
            self.ledger.deactivate_video(
                self.batch_id,
                prepared.source.video_id,
                "visual activation revalidation failed: retained thumbnail missing",
            )
            raise FileNotFoundError("one or more retained thumbnails are missing")
        indexed_count = self.visual_store.count(
            filters={"batch_id": self.batch_id, "video_id": prepared.source.video_id}
        )
        if indexed_count != len(prepared.frames):
            self.ledger.deactivate_video(
                self.batch_id,
                prepared.source.video_id,
                "visual activation revalidation failed: index count mismatch",
            )
            raise ValueError(
                f"visual index count {indexed_count} does not match retained frame "
                f"count {len(prepared.frames)}"
            )
        activated = self.ledger.activate_video(
            self.batch_id,
            prepared.source.video_id,
            required_stages=("probe", "frames", "visual"),
            required_artifacts={"frames_manifest": 1, "visual_vectors": 1},
        )
        if activated:
            self.ledger.bump_retrieval_generation(self.batch_id)

    def _index_additional_visuals(self, prepared: PreparedVideo) -> None:
        for name, encoder, store in self.additional_visual_channels:
            stage_name = f"visual_{name}"
            stage_fingerprint = fingerprint(
                {
                    "encoder": encoder.fingerprint,
                    "collection": store.collection_name,
                    "frame_count": len(prepared.frames),
                }
            )

            def operation(
                encoder: Any = encoder,
                store: Any = store,
                stage_fingerprint: str = stage_fingerprint,
            ) -> tuple[None, list[ArtifactSpec]]:
                vectors = encoder.encode_images([frame.image_uri for frame in prepared.frames])
                ids = [
                    f"{frame.batch_id}:{frame.video_id}:{frame.source_frame_idx}"
                    for frame in prepared.frames
                ]
                payloads = [frame.to_dict() for frame in prepared.frames]
                for payload in payloads:
                    payload["model_fingerprint"] = encoder.fingerprint
                store.upsert(ids, vectors, payloads)
                return None, [
                    ArtifactSpec(
                        f"{name}_vectors",
                        f"qdrant://{store.collection_name}/{prepared.source.video_id}",
                        stage_fingerprint,
                        None,
                        int(vectors.nbytes),
                    )
                ]

            self._stage(prepared.source, stage_name, stage_fingerprint, operation)

    def ingest_many(
        self,
        sources: Iterable[VideoSource],
        *,
        cpu_workers: int = 2,
        queue_size: int = 4,
        enrich: bool = True,
    ) -> dict[str, str]:
        """Run bounded CPU preparation with one serialized visual GPU consumer."""

        if cpu_workers < 1 or queue_size < 1:
            raise ValueError("cpu_workers and queue_size must be positive")
        # A reused ingestor may still own a Whisper model from an earlier run.
        # Release it before any visual encoder is allowed to load on CUDA.
        if self.asr_engine is not None:
            self._unload_model(self.asr_engine)
        source_iter = iter(sources)
        outcomes: dict[str, str] = {}
        prepared_videos: list[PreparedVideo] = []
        with ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="aic-decode") as pool:
            pending: dict[Future[PreparedVideo], VideoSource] = {}
            while True:
                while len(pending) < queue_size:
                    try:
                        source = next(source_iter)
                    except StopIteration:
                        break
                    pending[pool.submit(self._prepare, source)] = source
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    source = pending.pop(future)
                    try:
                        prepared = future.result()
                        self._index_visual(prepared)
                        self._index_additional_visuals(prepared)
                        prepared_videos.append(prepared)
                        outcomes[source.video_id] = "active"
                    except Exception as error:
                        outcomes[source.video_id] = f"failed: {type(error).__name__}: {error}"
        # Core search is ready before GPU-heavy ASR begins.  Explicit unloading
        # is required on the 4 GiB profile; garbage collection timing is not a
        # safe GPU scheduling primitive.
        self._unload_visual_models()
        if enrich:
            try:
                for prepared in prepared_videos:
                    try:
                        failures = self.enrich(prepared)
                        if failures:
                            outcomes[prepared.source.video_id] = (
                                "active; enrichment failed: " + ", ".join(failures)
                            )
                    except Exception as error:
                        outcomes[prepared.source.video_id] = (
                            f"active; enrichment failed: {type(error).__name__}: {error}"
                        )
            finally:
                if self.asr_engine is not None:
                    self._unload_model(self.asr_engine)
        # Batch readiness is an explicit persisted state, not an inference made
        # later by the exporter from video counts.
        self.ledger.activate_batch_if_ready(self.batch_id)
        return outcomes

    def _ocr_segments(self, prepared: PreparedVideo) -> tuple[TextSegment, ...]:
        if self.ocr_engine is None:
            return ()
        segments: list[TextSegment] = []
        for frame in prepared.frames:
            for index, detection in enumerate(self.ocr_engine.extract_text(frame.image_uri)):
                segments.append(
                    TextSegment(
                        segment_id=(
                            f"ocr:{frame.batch_id}:{frame.video_id}:"
                            f"{frame.source_frame_idx}:{index}"
                        ),
                        batch_id=frame.batch_id,
                        video_id=frame.video_id,
                        kind="ocr",
                        start_ms=frame.timestamp_ms,
                        end_ms=frame.timestamp_ms,
                        source_frame_idx=frame.source_frame_idx,
                        text_raw=detection["text_raw"],
                        text_nfc=detection["text_nfc"],
                        text_accent_folded=detection["text_accent_folded"],
                        confidence=float(detection["confidence"]),
                        model_version=self.ocr_engine.model_name,
                        polygons=(
                            tuple(
                                (float(point[0]), float(point[1]))
                                for point in detection["polygon"]
                            ),
                        ),
                    )
                )
        return tuple(segments)

    def _asr_segments(self, prepared: PreparedVideo) -> tuple[TextSegment, ...]:
        if self.asr_engine is None:
            return ()
        with self.scratch.materialize(prepared.source) as media_path:
            if self.audio_extractor is None:
                transcript = self.asr_engine.transcribe(media_path)
            else:
                expected_size = self.audio_extractor.estimated_size_bytes(
                    prepared.probe.duration_ms
                )
                with self.scratch.workspace(
                    prepared.source.video_id,
                    "asr-audio",
                    required_bytes=expected_size,
                ) as workspace:
                    audio_path = self.audio_extractor.extract_16k_mono(
                        media_path,
                        workspace / "audio-16khz-mono.wav",
                    )
                    transcript = self.asr_engine.transcribe(audio_path)
        segments: list[TextSegment] = []
        for index, value in enumerate(transcript["segments"]):
            nearest = self.artifacts.nearest_frame(
                self.batch_id,
                prepared.source.video_id,
                (int(value["start_ms"]) + int(value["end_ms"])) // 2,
            )
            confidence = max(0.0, min(1.0, math.exp(float(value["avg_logprob"]))))
            segments.append(
                TextSegment(
                    segment_id=f"asr:{self.batch_id}:{prepared.source.video_id}:{index}",
                    batch_id=self.batch_id,
                    video_id=prepared.source.video_id,
                    kind="asr",
                    start_ms=int(value["start_ms"]),
                    end_ms=int(value["end_ms"]),
                    source_frame_idx=(None if nearest is None else nearest.source_frame_idx),
                    text_raw=value["text_raw"],
                    text_nfc=value["text_nfc"],
                    text_accent_folded=value["text_accent_folded"],
                    confidence=confidence,
                    model_version=self.asr_engine.model_size,
                )
            )
        return tuple(segments)

    @staticmethod
    def _text_document(segment: TextSegment) -> dict[str, Any]:
        return {
            "id": segment.segment_id,
            **segment.to_dict(),
            "text": segment.text_nfc,
        }

    def _persisted_segments(
        self, prepared: PreparedVideo, kind: str
    ) -> tuple[TextSegment, ...]:
        return tuple(
            self.artifacts.iter_text_segments(
                batch_id=self.batch_id,
                video_id=prepared.source.video_id,
                kind=kind,
            )
        )

    def _enrich_ocr(self, prepared: PreparedVideo) -> tuple[TextSegment, ...]:
        if self.ocr_engine is None:
            return ()
        stage_fp = fingerprint(
            {"ocr": self.ocr_engine.model_name, "frames": len(prepared.frames)}
        )

        def run_ocr() -> tuple[tuple[TextSegment, ...], list[ArtifactSpec]]:
            values = self._ocr_segments(prepared)
            self.artifacts.upsert_text_segments(values, fingerprint=stage_fp)
            if self.ocr_store is not None:
                self.ocr_store.add_documents([self._text_document(item) for item in values])
            return values, [
                ArtifactSpec(
                    "ocr_segments",
                    f"sqlite://{prepared.source.video_id}/ocr",
                    stage_fp,
                    None,
                    len(values),
                )
            ]

        values = self._stage(prepared.source, "ocr", stage_fp, run_ocr)
        return self._persisted_segments(prepared, "ocr") if values is None else values

    def _enrich_asr(self, prepared: PreparedVideo) -> tuple[TextSegment, ...]:
        if self.asr_engine is None or not prepared.probe.has_audio:
            return ()
        stage_fp = fingerprint(
            {
                "asr": self.asr_engine.model_size,
                "compute": self.asr_engine.compute_type,
                "beam": self.asr_engine.beam_size,
                "audio_extractor": (
                    "direct"
                    if self.audio_extractor is None
                    else self.audio_extractor.fingerprint
                ),
            }
        )

        def run_asr() -> tuple[tuple[TextSegment, ...], list[ArtifactSpec]]:
            values = self._asr_segments(prepared)
            self.artifacts.upsert_text_segments(values, fingerprint=stage_fp)
            if self.asr_store is not None:
                self.asr_store.add_documents([self._text_document(item) for item in values])
            return values, [
                ArtifactSpec(
                    "asr_segments",
                    f"sqlite://{prepared.source.video_id}/asr",
                    stage_fp,
                    None,
                    len(values),
                )
            ]

        values = self._stage(prepared.source, "asr", stage_fp, run_asr)
        return self._persisted_segments(prepared, "asr") if values is None else values

    def _enrich_text(
        self, prepared: PreparedVideo, segments: list[TextSegment]
    ) -> None:
        if self.text_encoder is None or self.text_vector_store is None or not segments:
            return
        stage_fp = fingerprint(
            {"text_encoder": self.text_encoder.fingerprint, "segments": len(segments)}
        )

        def run_text() -> tuple[None, list[ArtifactSpec]]:
            vectors = self.text_encoder.encode_text([item.text_nfc for item in segments])
            self.text_vector_store.upsert(
                [item.segment_id for item in segments],
                vectors,
                [item.to_dict() for item in segments],
            )
            return None, [
                ArtifactSpec(
                    "text_vectors",
                    f"qdrant://{self.text_vector_store.collection_name}/{prepared.source.video_id}",
                    stage_fp,
                    None,
                    int(vectors.nbytes),
                )
            ]

        self._stage(prepared.source, "text", stage_fp, run_text)

    def enrich(self, prepared: PreparedVideo) -> tuple[str, ...]:
        """Run OCR/ASR independently, then index every persisted successful segment."""

        failures: list[str] = []
        all_segments: list[TextSegment] = []
        try:
            all_segments.extend(self._enrich_ocr(prepared))
        except Exception:
            failures.append("ocr")
            try:
                recovered_ocr = self._persisted_segments(prepared, "ocr")
            except Exception:
                recovered_ocr = ()
            all_segments.extend(recovered_ocr)
        try:
            all_segments.extend(self._enrich_asr(prepared))
        except Exception:
            failures.append("asr")
            try:
                recovered_asr = self._persisted_segments(prepared, "asr")
            except Exception:
                recovered_asr = ()
            all_segments.extend(recovered_asr)
        try:
            self._enrich_text(prepared, all_segments)
        except Exception:
            failures.append("semantic_text")
        return tuple(failures)


def _compatibility_main(argv: list[str] | None = None) -> int:
    """Delegate the legacy module command to the canonical ingestion CLI."""

    from src.ingest.cli import main

    return main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess test
    raise SystemExit(_compatibility_main())
