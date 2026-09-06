"""Resumable multimodal provisioning for the immutable Batch 1 reference corpus."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.contracts import FrameRecord, TextSegment
from src.extraction.media_probe import probe_media
from src.ingest.checkpoint_mgr import CheckpointMirror, CheckpointSnapshot
from src.ingest.ledger import ArtifactSpec, JobLedger, StageState
from src.ingest.native_pipeline import ScratchManager, VideoSource
from src.ingest.resource_watchdog import ResourceBudgetExceeded, ResourceBudgetGuard

REFERENCE_BATCH_ID = "aic25-b1-reference"
REFERENCE_BATCH_ALIAS = "B01"
_STAGE_NAMES = {
    "ocr": "reference_ocr",
    "siglip2": "reference_siglip2",
    "asr": "reference_asr",
}


class ReferenceProvisioningError(RuntimeError):
    """Stable, sanitized provisioning failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class StageReport:
    """Safe report returned by one isolated stage subprocess."""

    batch_id: str
    stage: str
    state: str
    selected_videos: int
    completed_videos: int
    failed_videos: int
    source_missing_videos: int
    canonical_frames: int
    records_written: int
    elapsed_ms: float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    frames_per_hour: float
    videos_per_hour: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "stage": self.stage,
            "state": self.state,
            "selected_videos": self.selected_videos,
            "completed_videos": self.completed_videos,
            "failed_videos": self.failed_videos,
            "source_missing_videos": self.source_missing_videos,
            "canonical_frames": self.canonical_frames,
            "records_written": self.records_written,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "frames_per_hour": round(self.frames_per_hour, 3),
            "videos_per_hour": round(self.videos_per_hour, 3),
        }


def normalize_reference_batch(value: str) -> str:
    """Resolve the single supported CLI alias without accepting arbitrary batches."""

    if not isinstance(value, str):
        raise TypeError("batch must be a string")
    normalized = value.strip()
    if normalized.casefold() in {
        REFERENCE_BATCH_ALIAS.casefold(),
        REFERENCE_BATCH_ID.casefold(),
    }:
        return REFERENCE_BATCH_ID
    raise ValueError("reference_batch_unsupported")


def _stable_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _safe_error_code(error: BaseException) -> str:
    if isinstance(error, ResourceBudgetExceeded):
        return error.code
    if isinstance(error, ReferenceProvisioningError):
        return error.code
    if isinstance(error, (FileNotFoundError, KeyError)):
        return "provisioning_asset_missing"
    if isinstance(error, (MemoryError,)):
        return "provisioning_resource_exhausted"
    if isinstance(error, (OSError, RuntimeError, ValueError)):
        return f"provisioning_{type(error).__name__.casefold()}"
    return "provisioning_internal_error"


class ReferenceProvisioner:
    """Provision OCR, SigLIP2, and ASR without mutating canonical media."""

    def __init__(
        self,
        *,
        ledger: JobLedger,
        artifacts: Any,
        image_resolver: Any,
        resource_guard: ResourceBudgetGuard,
        checkpoint_mirror: CheckpointMirror,
        visual_encoder: Any | None = None,
        visual_store: Any | None = None,
        ocr_engine: Any | None = None,
        ocr_store: Any | None = None,
        asr_engine: Any | None = None,
        asr_store: Any | None = None,
        scratch: ScratchManager | None = None,
        audio_extractor: Any | None = None,
        ffprobe_path: str | Path | None = None,
        video_sources: Mapping[str, VideoSource] | None = None,
        batch_id: str = REFERENCE_BATCH_ID,
        expected_videos: int = 873,
        expected_frames: int = 176_707,
        ocr_upsert_batch_size: int = 500,
        vector_upsert_batch_size: int = 256,
    ) -> None:
        self.batch_id = normalize_reference_batch(batch_id)
        self.ledger = ledger
        self.artifacts = artifacts
        self.image_resolver = image_resolver
        self.resource_guard = resource_guard
        self.checkpoint_mirror = checkpoint_mirror
        self.visual_encoder = visual_encoder
        self.visual_store = visual_store
        self.ocr_engine = ocr_engine
        self.ocr_store = ocr_store
        self.asr_engine = asr_engine
        self.asr_store = asr_store
        self.scratch = scratch
        self.audio_extractor = audio_extractor
        self.ffprobe_path = None if ffprobe_path is None else Path(ffprobe_path)
        self.video_sources = dict(video_sources or {})
        self.expected_videos = int(expected_videos)
        self.expected_frames = int(expected_frames)
        self.ocr_upsert_batch_size = int(ocr_upsert_batch_size)
        self.vector_upsert_batch_size = int(vector_upsert_batch_size)
        if (
            min(
                self.expected_videos,
                self.expected_frames,
                self.ocr_upsert_batch_size,
                self.vector_upsert_batch_size,
            )
            < 1
        ):
            raise ValueError("reference provisioning counts must be positive")

    @staticmethod
    def _text_document(segment: TextSegment) -> dict[str, Any]:
        return {"id": segment.segment_id, **segment.to_dict(), "text": segment.text_nfc}

    def _stage_fingerprint(
        self,
        stage: str,
        *,
        frame_count: int | None = None,
        source: VideoSource | None = None,
    ) -> str:
        if stage == "ocr":
            engine = getattr(self.ocr_engine, "fingerprint", "missing")
            destination = getattr(self.ocr_store, "index_name", "missing")
        elif stage == "siglip2":
            engine = getattr(self.visual_encoder, "fingerprint", "missing")
            destination = getattr(self.visual_store, "collection_name", "missing")
        elif stage == "asr":
            engine = {
                "model": getattr(self.asr_engine, "model_size", "missing"),
                "revision": getattr(self.asr_engine, "model_revision", None),
                "compute": getattr(self.asr_engine, "compute_type", None),
                "beam": getattr(self.asr_engine, "beam_size", None),
                "audio": getattr(self.audio_extractor, "fingerprint", "direct"),
            }
            destination = getattr(self.asr_store, "index_name", "missing")
        else:
            raise ValueError("unsupported provisioning stage")
        return _stable_fingerprint(
            {
                "contract": "reference-provisioning-v1",
                "batch": self.batch_id,
                "stage": stage,
                "engine": engine,
                "destination": destination,
                "frame_count": frame_count,
                "source_member": None if source is None else source.member,
                "source_size": None if source is None else source.size_bytes,
            }
        )

    def _video_ids(self, limit_videos: int | None) -> tuple[str, ...]:
        values = self.ledger.list_video_ids(self.batch_id)
        if limit_videos is not None:
            if isinstance(limit_videos, bool) or limit_videos < 1:
                raise ValueError("limit_videos must be a positive integer")
            values = values[:limit_videos]
        return values

    def _start(
        self,
        video_id: str,
        stage: str,
        fingerprint: str,
        *,
        reprocess: bool,
    ) -> bool:
        transition = self.ledger.start_stage(
            self.batch_id,
            video_id,
            _STAGE_NAMES[stage],
            fingerprint,
            allow_reprocess=reprocess,
        )
        return transition.changed

    def _complete(
        self,
        video_id: str,
        stage: str,
        fingerprint: str,
        *,
        record_count: int,
        artifact_kind: str | None = None,
    ) -> None:
        kind = artifact_kind or f"{stage}_records"
        self.ledger.complete_stage(
            self.batch_id,
            video_id,
            _STAGE_NAMES[stage],
            artifacts=(
                ArtifactSpec(
                    kind,
                    f"state://reference/{stage}/{video_id}",
                    fingerprint,
                    None,
                    int(record_count),
                ),
            ),
        )

    def _persist_ocr(
        self,
        video_id: str,
        frames: tuple[FrameRecord, ...],
        fingerprint: str,
        *,
        reprocess: bool,
    ) -> tuple[int, int]:
        if self.ocr_engine is None or self.ocr_store is None:
            raise ReferenceProvisioningError("rapidocr_not_provisioned")
        segments: list[TextSegment] = []
        scanned = 0
        for frame in frames:
            image = self.image_resolver.resolve(frame)
            for index, detection in enumerate(self.ocr_engine.extract_text(image.data)):
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
                        polygons=(tuple(tuple(point) for point in detection["polygon"]),),
                    )
                )
            scanned += 1
            if scanned % 16 == 0:
                self.resource_guard.check()
        if reprocess:
            self.artifacts.replace_text_segments_for_video(
                self.batch_id,
                video_id,
                "ocr",
                segments,
                fingerprint=fingerprint,
            )
        else:
            self.artifacts.upsert_text_segments(segments, fingerprint=fingerprint)
        self.ocr_store.add_documents(
            [self._text_document(value) for value in segments],
            batch_size=self.ocr_upsert_batch_size,
        )
        return scanned, len(segments)

    def _persist_siglip2(
        self,
        video_id: str,
        frames: tuple[FrameRecord, ...],
        fingerprint: str,
    ) -> tuple[int, int]:
        if self.visual_encoder is None or self.visual_store is None:
            raise ReferenceProvisioningError("siglip2_not_provisioned")
        self.visual_store.create_collection()
        records = 0
        for start in range(0, len(frames), self.vector_upsert_batch_size):
            chunk = frames[start : start + self.vector_upsert_batch_size]
            images = [self.image_resolver.resolve(frame).data for frame in chunk]
            vectors = np.asarray(self.visual_encoder.encode_images(images), dtype=np.float32)
            if vectors.shape != (len(chunk), 768) or not np.all(np.isfinite(vectors)):
                raise ReferenceProvisioningError("siglip2_vector_contract_invalid")
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
                raise ReferenceProvisioningError("siglip2_vector_norm_invalid")
            identifiers = [
                f"{frame.batch_id}:{frame.video_id}:{frame.source_frame_idx}" for frame in chunk
            ]
            payloads = [frame.to_dict() for frame in chunk]
            for payload in payloads:
                payload["model_fingerprint"] = self.visual_encoder.fingerprint
            self.visual_store.upsert(
                identifiers,
                vectors,
                payloads,
                batch_size=self.vector_upsert_batch_size,
            )
            records += len(chunk)
            self.resource_guard.check()
        indexed = self.visual_store.count(filters={"batch_id": self.batch_id, "video_id": video_id})
        if indexed != len(frames):
            raise ReferenceProvisioningError("siglip2_external_count_mismatch")
        return len(frames), records

    def _asr_segments(
        self,
        video_id: str,
        transcript: Mapping[str, Any],
    ) -> tuple[TextSegment, ...]:
        segments: list[TextSegment] = []
        for index, value in enumerate(transcript.get("segments", ())):
            text_raw = str(value.get("text_raw", ""))
            if not text_raw.strip():
                continue
            start_ms = int(value["start_ms"])
            end_ms = int(value["end_ms"])
            nearest = self.artifacts.nearest_frame(
                self.batch_id,
                video_id,
                start_ms + (end_ms - start_ms) // 2,
            )
            confidence = max(0.0, min(1.0, math.exp(float(value.get("avg_logprob", 0.0)))))
            segments.append(
                TextSegment(
                    segment_id=f"asr:{self.batch_id}:{video_id}:{index}",
                    batch_id=self.batch_id,
                    video_id=video_id,
                    kind="asr",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    source_frame_idx=None if nearest is None else nearest.source_frame_idx,
                    text_raw=text_raw,
                    text_nfc=str(value["text_nfc"]),
                    text_accent_folded=str(value["text_accent_folded"]),
                    confidence=confidence,
                    model_version=str(getattr(self.asr_engine, "model_size", "small")),
                )
            )
        return tuple(segments)

    @staticmethod
    def _pyav_has_audio(media_path: Path) -> bool:
        """Inspect streams locally without logging container metadata or paths."""

        try:
            import av
        except ImportError as error:
            raise ReferenceProvisioningError("asr_media_probe_unavailable") from error
        try:
            with av.open(str(media_path), mode="r") as container:
                return any(stream.type == "audio" for stream in container.streams)
        except Exception as error:
            raise ReferenceProvisioningError("asr_media_probe_failed") from error

    def _persist_asr(
        self,
        video_id: str,
        source: VideoSource,
        fingerprint: str,
        *,
        reprocess: bool,
    ) -> tuple[int, int, str]:
        if self.asr_engine is None or self.asr_store is None or self.scratch is None:
            raise ReferenceProvisioningError("asr_not_provisioned")
        with self.scratch.materialize(source) as media_path:
            if self.ffprobe_path is None:
                has_audio = self._pyav_has_audio(media_path)
                duration_ms = 0
            else:
                media = probe_media(media_path, ffprobe_path=self.ffprobe_path)
                has_audio = media.has_audio
                duration_ms = media.duration_ms
            if not has_audio:
                if reprocess:
                    self.artifacts.replace_text_segments_for_video(
                        self.batch_id,
                        video_id,
                        "asr",
                        (),
                        fingerprint=fingerprint,
                    )
                return 1, 0, "asr_no_audio"
            if self.audio_extractor is None:
                transcript = self.asr_engine.transcribe(media_path)
            else:
                required = self.audio_extractor.estimated_size_bytes(duration_ms)
                with self.scratch.workspace(
                    video_id,
                    "reference-asr",
                    required_bytes=required,
                ) as workspace:
                    audio_path = self.audio_extractor.extract_16k_mono(
                        media_path,
                        workspace / "audio.wav",
                    )
                    transcript = self.asr_engine.transcribe(audio_path)
        segments = self._asr_segments(video_id, transcript)
        if reprocess:
            self.artifacts.replace_text_segments_for_video(
                self.batch_id,
                video_id,
                "asr",
                segments,
                fingerprint=fingerprint,
            )
        else:
            self.artifacts.upsert_text_segments(segments, fingerprint=fingerprint)
        self.asr_store.add_documents(
            [self._text_document(value) for value in segments],
            batch_size=self.ocr_upsert_batch_size,
        )
        self.resource_guard.check()
        return 1, len(segments), "asr_records"

    def _reconcile_completed(
        self,
        video_id: str,
        stage: str,
        frames: tuple[FrameRecord, ...],
    ) -> int:
        if stage == "siglip2":
            actual = self.visual_store.count(
                filters={"batch_id": self.batch_id, "video_id": video_id}
            )
            if actual != len(frames):
                raise ReferenceProvisioningError("siglip2_external_count_mismatch")
            return actual
        segments = tuple(
            self.artifacts.iter_text_segments(
                batch_id=self.batch_id,
                video_id=video_id,
                kind=stage,
            )
        )
        store = self.ocr_store if stage == "ocr" else self.asr_store
        if store is None:
            raise ReferenceProvisioningError(f"{stage}_store_unavailable")
        store.add_documents(
            [self._text_document(value) for value in segments],
            batch_size=self.ocr_upsert_batch_size,
        )
        return len(segments)

    def _checkpoint(
        self,
        stage: str,
        fingerprint: str,
        *,
        canonical_frames: int,
        records_written: int,
        elapsed_ms: float,
        source_missing: Iterable[str],
    ) -> None:
        rows = self.ledger.stage_records(self.batch_id, _STAGE_NAMES[stage])
        completed = tuple(
            str(row["video_id"]) for row in rows if row["state"] == StageState.COMPLETED.value
        )
        failed = tuple(
            str(row["video_id"]) for row in rows if row["state"] == StageState.FAILED.value
        )
        missing = tuple(sorted(set(source_missing)))
        expected = self.expected_videos
        state = (
            "complete" if len(completed) == expected and not failed and not missing else "partial"
        )
        peaks = self.resource_guard.peaks
        self.checkpoint_mirror.publish(
            CheckpointSnapshot(
                schema_version=1,
                batch_id=self.batch_id,
                stage=stage,
                fingerprint=fingerprint,
                completed_video_ids=completed,
                failed_video_ids=failed,
                source_missing_video_ids=missing,
                counts={
                    "expected_videos": expected,
                    "expected_canonical_frames": self.expected_frames,
                    "completed_videos": len(completed),
                    "failed_videos": len(failed),
                    "canonical_frames": canonical_frames,
                    "records_written": records_written,
                },
                timings_ms={"elapsed": elapsed_ms},
                resources=peaks.to_dict(),
                state=state,
            )
        )

    def run_stage(
        self,
        stage: str,
        *,
        limit_videos: int | None = None,
        dry_run: bool = False,
        reprocess_videos: Iterable[str] = (),
    ) -> StageReport:
        """Run one isolated stage in canonical video order."""

        if stage not in _STAGE_NAMES:
            raise ValueError("stage must be ocr, siglip2, or asr")
        video_ids = self._video_ids(limit_videos)
        requested_reprocess = frozenset(reprocess_videos)
        unknown = requested_reprocess.difference(video_ids)
        if unknown:
            raise ValueError("reprocess video is outside the selected reference scope")
        if dry_run:
            return StageReport(
                self.batch_id,
                stage,
                "dry_run",
                len(video_ids),
                0,
                0,
                sum(video_id not in self.video_sources for video_id in video_ids)
                if stage == "asr"
                else 0,
                sum(
                    self.artifacts.count_frames(self.batch_id, video_id=video_id)
                    for video_id in video_ids
                ),
                0,
                0.0,
                0,
                0,
                0.0,
                0.0,
            )
        if stage in {"ocr", "asr"}:
            store = self.ocr_store if stage == "ocr" else self.asr_store
            if store is None:
                raise ReferenceProvisioningError(f"{stage}_store_unavailable")
            store.create_index()
        if stage == "siglip2" and self.visual_store is not None:
            self.visual_store.create_collection()

        started = time.perf_counter()
        canonical_frames = records_written = 0
        completed = failed = 0
        missing_sources: list[str] = []
        stage_fingerprint = self._stage_fingerprint(stage)
        try:
            for video_id in video_ids:
                frames = tuple(
                    self.artifacts.iter_frames(
                        batch_id=self.batch_id,
                        video_id=video_id,
                    )
                )
                canonical_frames += len(frames)
                source = self.video_sources.get(video_id)
                fingerprint = self._stage_fingerprint(
                    stage,
                    frame_count=len(frames),
                    source=source,
                )
                if stage == "asr" and source is None:
                    missing_sources.append(video_id)
                    failed += 1
                    continue
                reprocess = video_id in requested_reprocess
                try:
                    changed = self._start(
                        video_id,
                        stage,
                        fingerprint,
                        reprocess=reprocess,
                    )
                    if not changed:
                        records_written += self._reconcile_completed(
                            video_id,
                            stage,
                            frames,
                        )
                        completed += 1
                        continue
                    if stage == "ocr":
                        _, count = self._persist_ocr(
                            video_id,
                            frames,
                            fingerprint,
                            reprocess=reprocess,
                        )
                        artifact_kind = "ocr_records"
                    elif stage == "siglip2":
                        _, count = self._persist_siglip2(
                            video_id,
                            frames,
                            fingerprint,
                        )
                        artifact_kind = "siglip2_vectors"
                    else:
                        _, count, artifact_kind = self._persist_asr(
                            video_id,
                            source,
                            fingerprint,
                            reprocess=reprocess,
                        )
                    self.resource_guard.check()
                    self._complete(
                        video_id,
                        stage,
                        fingerprint,
                        record_count=count,
                        artifact_kind=artifact_kind,
                    )
                    records_written += count
                    completed += 1
                except Exception as error:
                    failed += 1
                    record = self.ledger.get_stage(
                        self.batch_id,
                        video_id,
                        _STAGE_NAMES[stage],
                    )
                    if record is not None and record.get("state") == StageState.RUNNING.value:
                        self.ledger.fail_stage(
                            self.batch_id,
                            video_id,
                            _STAGE_NAMES[stage],
                            _safe_error_code(error),
                        )
                    if isinstance(error, ResourceBudgetExceeded):
                        raise
        finally:
            if stage == "ocr" and self.ocr_engine is not None:
                self.ocr_engine.unload()
            elif stage == "siglip2" and self.visual_encoder is not None:
                unload = getattr(self.visual_encoder, "unload", None)
                if callable(unload):
                    unload()
            elif stage == "asr" and self.asr_engine is not None:
                self.asr_engine.unload()
            ResourceBudgetGuard.release_cuda()
            close = getattr(self.image_resolver, "close", None)
            if callable(close):
                close()

        elapsed_ms = (time.perf_counter() - started) * 1000
        self._checkpoint(
            stage,
            stage_fingerprint,
            canonical_frames=canonical_frames,
            records_written=records_written,
            elapsed_ms=elapsed_ms,
            source_missing=missing_sources,
        )
        elapsed_hours = max(elapsed_ms / 3_600_000, 1e-12)
        rows = self.ledger.stage_records(self.batch_id, _STAGE_NAMES[stage])
        all_completed = sum(row["state"] == StageState.COMPLETED.value for row in rows)
        all_failed = sum(row["state"] == StageState.FAILED.value for row in rows)
        state = (
            "complete" if all_completed == self.expected_videos and all_failed == 0 else "partial"
        )
        peaks = self.resource_guard.peaks
        return StageReport(
            self.batch_id,
            stage,
            state,
            len(video_ids),
            completed,
            failed,
            len(missing_sources),
            canonical_frames,
            records_written,
            elapsed_ms,
            peaks.process_rss_bytes,
            peaks.gpu_process_bytes,
            canonical_frames / elapsed_hours,
            len(video_ids) / elapsed_hours,
        )


__all__ = [
    "REFERENCE_BATCH_ALIAS",
    "REFERENCE_BATCH_ID",
    "ReferenceProvisioner",
    "ReferenceProvisioningError",
    "StageReport",
    "normalize_reference_batch",
]
