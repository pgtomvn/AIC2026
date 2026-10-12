"""Local coarse-to-fine frame selection from raw candidate windows."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.contracts import FrameRecord, RefinedFrameRecord
from src.embedding.text_chunks import encode_query_mean
from src.ingest.artifacts import ArtifactRepository
from src.ingest.native_pipeline import ScratchManager, VideoSource
from src.retrieval.query_plan import QueryPlan

_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RefinedFramePersistenceError(RuntimeError):
    """Raised when an exact frame cannot be made durable within safe bounds."""


class _RefinementEmbeddingError(ValueError):
    """An invalid embedding must not be mistaken for unavailable raw media."""


class RefinedFrameSink(Protocol):
    """Persistence boundary used before a temporary decoded frame is released."""

    def persist(
        self,
        candidate: Mapping[str, Any],
        sample: Mapping[str, Any],
        *,
        refiner_fingerprint: str,
    ) -> FrameRecord | RefinedFrameRecord: ...


def _source_from_uri(video_id: str, uri: str) -> VideoSource:
    if uri.startswith("reference://"):
        raise FileNotFoundError("reference-only identity has no local raw video")
    if uri.startswith("zip://"):
        value = uri[len("zip://") :]
        archive_value, separator, member = value.partition("!/")
        if not separator or not member:
            raise ValueError(f"invalid ZIP video URI: {uri}")
        archive = Path(archive_value)
        import zipfile

        with zipfile.ZipFile(archive) as handle:
            try:
                size = handle.getinfo(member).file_size
            except KeyError as error:
                raise FileNotFoundError("raw video member is unavailable") from error
        return VideoSource(video_id, archive, member, size)
    return VideoSource(video_id, Path(uri))


class PersistentRefinedFrameSink:
    """Persist one bounded WebP and its exact source-frame provenance."""

    def __init__(
        self,
        repository: ArtifactRepository,
        root: str | Path,
        *,
        max_dimension: int = 512,
        webp_quality: int = 85,
        max_image_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 20_000,
        max_total_bytes: int = 2 * 1024**3,
        reserve_bytes: int = 30 * 1024**3,
        max_input_pixels: int = 40_000_000,
    ) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must be an ArtifactRepository")
        for name, value in (
            ("max_dimension", max_dimension),
            ("max_image_bytes", max_image_bytes),
            ("max_entries", max_entries),
            ("max_total_bytes", max_total_bytes),
            ("max_input_pixels", max_input_pixels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int):
            raise TypeError("reserve_bytes must be an integer")
        if reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")
        if not 1 <= webp_quality <= 100:
            raise ValueError("webp_quality must be between 1 and 100")
        self.repository = repository
        self.root = Path(root).resolve()
        self.max_dimension = max_dimension
        self.webp_quality = webp_quality
        self.max_image_bytes = max_image_bytes
        self.max_entries = max_entries
        self.max_total_bytes = max_total_bytes
        self.reserve_bytes = reserve_bytes
        self.max_input_pixels = max_input_pixels
        self._guard = threading.RLock()

    @staticmethod
    def _component(value: object, name: str) -> str:
        if not isinstance(value, str) or not _SAFE_PATH_COMPONENT.fullmatch(value):
            raise RefinedFramePersistenceError(f"{name} is not a safe identifier")
        if value in {".", ".."}:
            raise RefinedFramePersistenceError(f"{name} is not a safe identifier")
        return value

    @staticmethod
    def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RefinedFramePersistenceError(f"{name} must be an integer")
        if minimum is not None and value < minimum:
            raise RefinedFramePersistenceError(f"{name} is below its minimum")
        return value

    @staticmethod
    def _text(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RefinedFramePersistenceError(f"{name} must be non-empty")
        return value

    def _available_bytes(self) -> int:
        probe = self.root
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        return int(shutil.disk_usage(probe).free)

    def _normalize(self, path: Path) -> tuple[bytes, int, int, str]:
        try:
            with Image.open(path) as probe:
                width, height = probe.size
                if width < 1 or height < 1 or width * height > self.max_input_pixels:
                    raise RefinedFramePersistenceError(
                        "decoded refined image exceeds configured bounds"
                    )
                probe.verify()
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                image.load()
                output = io.BytesIO()
                image.save(
                    output,
                    format="WEBP",
                    quality=self.webp_quality,
                    method=4,
                )
                encoded = output.getvalue()
        except RefinedFramePersistenceError:
            raise
        except (OSError, SyntaxError, UnidentifiedImageError) as exc:
            raise RefinedFramePersistenceError("refined image is invalid") from exc
        if not encoded or len(encoded) > self.max_image_bytes:
            raise RefinedFramePersistenceError("encoded refined image exceeds configured bounds")
        return encoded, image.width, image.height, hashlib.sha256(encoded).hexdigest()

    def _target(
        self,
        batch_id: str,
        video_id: str,
        source_sha256: str,
        source_frame_idx: int,
    ) -> Path:
        target = (
            self.root / batch_id / video_id / source_sha256[:16] / f"{source_frame_idx:09d}.webp"
        ).resolve()
        if self.root != target and self.root not in target.parents:
            raise RefinedFramePersistenceError("refined image target is outside its root")
        return target

    @staticmethod
    def _write_atomic(target: Path, data: bytes) -> bool:
        if (
            target.is_file()
            and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(data).digest()
        ):
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            return True
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def persist(
        self,
        candidate: Mapping[str, Any],
        sample: Mapping[str, Any],
        *,
        refiner_fingerprint: str,
    ) -> FrameRecord | RefinedFrameRecord:
        """Make a selected temporary sample durable before returning its identity."""

        try:
            batch_id = self._component(candidate.get("batch_id"), "batch_id")
            video_id = self._component(candidate.get("video_id"), "video_id")
            source_frame_idx = self._integer(
                sample.get("source_frame_idx"), "source_frame_idx", minimum=0
            )
            coarse_source_frame_idx = self._integer(
                candidate.get("source_frame_idx"),
                "coarse_source_frame_idx",
                minimum=0,
            )
            source_sha256 = self._text(candidate.get("source_sha256"), "source_sha256")
            if not re.fullmatch(r"[0-9A-Fa-f]{64}", source_sha256):
                raise RefinedFramePersistenceError("source_sha256 is invalid")
            refiner_fingerprint = self._text(refiner_fingerprint, "refiner_fingerprint")
            canonical = self.repository.get_frame(batch_id, video_id, source_frame_idx)
            if canonical is not None:
                if canonical.source_sha256 != source_sha256.casefold():
                    raise RefinedFramePersistenceError(
                        "retained source frame belongs to a different source revision"
                    )
                return canonical
            path_value = sample.get("path")
            if not isinstance(path_value, (str, Path)):
                raise RefinedFramePersistenceError("selected sample path is invalid")
            encoded, width, height, image_sha256 = self._normalize(Path(path_value))
            target = self._target(
                batch_id,
                video_id,
                source_sha256.casefold(),
                source_frame_idx,
            )
            record = RefinedFrameRecord(
                batch_id=batch_id,
                video_id=video_id,
                source_frame_idx=source_frame_idx,
                coarse_source_frame_idx=coarse_source_frame_idx,
                pts=self._integer(sample.get("pts"), "pts"),
                time_base=self._text(sample.get("time_base"), "time_base"),
                timestamp_ms=self._integer(sample.get("timestamp_ms"), "timestamp_ms", minimum=0),
                fps_num=self._integer(candidate.get("fps_num"), "fps_num", minimum=1),
                fps_den=self._integer(candidate.get("fps_den"), "fps_den", minimum=1),
                shot_id=self._text(candidate.get("shot_id"), "shot_id"),
                image_uri=str(target),
                video_uri=self._text(candidate.get("video_uri"), "video_uri"),
                source_sha256=source_sha256,
                image_sha256=image_sha256,
                image_width=width,
                image_height=height,
                image_bytes=len(encoded),
                refiner_fingerprint=refiner_fingerprint,
            )
            with self._guard:
                existing = self.repository.get_refined_frame(batch_id, video_id, source_frame_idx)
                count, total_bytes = self.repository.refined_frame_usage()
                if existing is None and count >= self.max_entries:
                    raise RefinedFramePersistenceError("refined frame entry limit reached")
                if existing is None and total_bytes + len(encoded) > self.max_total_bytes:
                    raise RefinedFramePersistenceError("refined frame byte limit reached")
                if self._available_bytes() - len(encoded) < self.reserve_bytes:
                    raise RefinedFramePersistenceError("disk reserve would be violated")
                wrote = self._write_atomic(target, encoded)
                try:
                    self.repository.upsert_refined_frame(record)
                except Exception:
                    committed = None
                    state_checked = False
                    try:
                        committed = self.repository.get_refined_frame(
                            batch_id, video_id, source_frame_idx
                        )
                        state_checked = True
                    except Exception:
                        # Preserve the file when database state cannot be
                        # checked; deleting another process's committed image
                        # would be worse than leaving a bounded orphan.
                        pass
                    if wrote and existing is None and state_checked and committed is None:
                        target.unlink(missing_ok=True)
                    raise
                resolved = self.repository.get_source_frame(batch_id, video_id, source_frame_idx)
                if resolved is None:  # pragma: no cover - guarded by committed upsert
                    raise RefinedFramePersistenceError("refined frame did not become durable")
                return resolved
        except RefinedFramePersistenceError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RefinedFramePersistenceError("refined frame persistence failed") from exc


class LocalFrameRefiner:
    """Decode ±radius at a fixed rate and select with the same visual encoder."""

    def __init__(
        self,
        encoder: Any,
        scratch: ScratchManager,
        *,
        radius_seconds: float = 4.0,
        sample_fps: float = 6.0,
        top_windows: int = 20,
        frame_sink: RefinedFrameSink | None = None,
    ) -> None:
        if radius_seconds <= 0 or sample_fps <= 0 or top_windows <= 0:
            raise ValueError("raw refinement settings must be positive")
        self.encoder = encoder
        self.scratch = scratch
        self.radius_seconds = radius_seconds
        self.sample_fps = sample_fps
        self.top_windows = top_windows
        self.frame_sink = frame_sink
        self._state = threading.local()

    @property
    def last_diagnostics(self) -> dict[str, Any]:
        """Per-request outcome, including coarse-only resource fallbacks."""
        return dict(getattr(self._state, "diagnostics", {}))

    @property
    def fingerprint(self) -> str:
        """Return the exact-decoding and selection policy fingerprint."""

        payload = {
            "version": "local-raw-frame-refiner-v2",
            "encoder": str(getattr(self.encoder, "fingerprint", type(self.encoder).__name__)),
            "radius_seconds": self.radius_seconds,
            "sample_fps": self.sample_fps,
            "top_windows": self.top_windows,
            "identity_method": "canonical-anchor-relative-pyav-decode",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _decode(
        self,
        media_path: Path,
        center_ms: int,
        output_dir: Path,
        *,
        anchor_source_frame_idx: int,
        anchor_pts: int | None = None,
        anchor_time_base: str | None = None,
        fps_num: int | None = None,
        fps_den: int | None = None,
    ) -> list[dict[str, Any]]:
        """Decode a window and infer exact ordinals from a known canonical anchor.

        Seeking itself does not provide a global decoded-frame ordinal.  PyAV
        therefore decodes every frame from the preceding keyframe and aligns
        that local sequence to the coarse frame's persisted PTS/source ordinal.
        This remains correct for VFR video and avoids trusting OpenCV's
        backend-dependent ``CAP_PROP_POS_FRAMES`` after a seek.
        """

        try:
            import av
        except ImportError as error:  # pragma: no cover - competition extra is locked
            raise ValueError("PyAV is required for exact raw-frame refinement") from error

        start_ms = max(0, center_ms - round(self.radius_seconds * 1000))
        end_ms = center_ms + round(self.radius_seconds * 1000)
        interval_ms = 1000 / self.sample_fps
        next_sample = float(start_ms)
        samples: list[dict[str, Any]] = []
        anchor_position: int | None = None
        if anchor_pts is None or anchor_time_base is None:
            raise ValueError("exact refinement requires a persisted PTS/time-base anchor")
        requested_anchor_time = Fraction(anchor_pts) * Fraction(anchor_time_base)
        container = av.open(str(media_path))
        try:
            if not container.streams.video:
                raise ValueError(f"raw video has no video stream: {media_path}")
            stream = container.streams.video[0]
            stream_time_base = Fraction(stream.time_base)
            seek_offset = int(Fraction(start_ms, 1000) / stream_time_base)
            container.seek(max(0, seek_offset), stream=stream, backward=True, any_frame=False)
            decode_position = -1
            for frame in container.decode(stream):
                decode_position += 1
                if frame.pts is None:
                    continue
                frame_time = Fraction(frame.pts) * stream_time_base
                timestamp_ms = float(frame_time * 1000)
                if timestamp_ms > end_ms:
                    break
                if frame_time == requested_anchor_time:
                    anchor_position = decode_position
                if timestamp_ms + 1e-6 < next_sample:
                    continue
                target = output_dir / f"sample-{decode_position:09d}.jpg"
                frame.to_image().save(target, format="JPEG", quality=92)
                samples.append(
                    {
                        "pts": int(frame.pts),
                        "time_base": (
                            f"{stream_time_base.numerator}/{stream_time_base.denominator}"
                        ),
                        "timestamp_ms": round(timestamp_ms),
                        "path": target,
                        "decode_position": decode_position,
                    }
                )
                while next_sample <= timestamp_ms + 1e-6:
                    next_sample += interval_ms
        finally:
            container.close()

        if anchor_position is None:
            raise ValueError(
                "cannot exactly align canonical PTS after rescaling to the decoded stream"
            )
        frames: list[dict[str, Any]] = []
        for sample in samples:
            source_frame_idx = anchor_source_frame_idx + (
                int(sample.pop("decode_position")) - anchor_position
            )
            if source_frame_idx < 0:
                continue
            sample["source_frame_idx"] = source_frame_idx
            frames.append(sample)
        return frames

    @staticmethod
    def is_eligible(candidate: dict[str, Any]) -> bool:
        """Return whether a canonical hit can be decoded around its exact anchor."""

        sampling_reasons = candidate.get("sampling_reasons") or ()
        return (
            isinstance(candidate.get("video_uri"), str)
            and not candidate["video_uri"].startswith("reference://")
            and candidate.get("timestamp_ms") is not None
            and candidate.get("pts") is not None
            and candidate.get("time_base") is not None
            and "estimated_pts" not in sampling_reasons
        )

    def __call__(
        self,
        plan: QueryPlan,
        candidates: list[dict[str, Any]],
        *,
        query_texts: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        active = candidates[: self.top_windows]
        self._state.diagnostics = {
            "state": "not_applicable",
            "reason_code": (
                "reference_only_source" if active and all(
                    str(candidate.get("video_uri", "")).startswith("reference://")
                    for candidate in active
                ) else "no_exact_raw_anchor"
            ),
            "refined_count": 0,
        }
        if self.frame_sink is None or not any(self.is_eligible(candidate) for candidate in active):
            return [dict(candidate) for candidate in candidates]
        texts = query_texts or (plan.original_text,)
        query_vector: np.ndarray | None = None
        reasons: Counter[str] = Counter()
        refined_count = 0
        refined: list[dict[str, Any]] = []
        for candidate in candidates[: self.top_windows]:
            video_uri = candidate.get("video_uri")
            center_ms = candidate.get("timestamp_ms")
            if not self.is_eligible(candidate):
                refined.append(dict(candidate))
                continue
            try:
                source = _source_from_uri(str(candidate["video_id"]), video_uri)
                with self.scratch.materialize(source) as media_path:
                    self.scratch.root.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(
                        prefix="refine-", dir=self.scratch.root
                    ) as temporary_dir:
                        frames = self._decode(
                            media_path,
                            int(center_ms),
                            Path(temporary_dir),
                            anchor_source_frame_idx=int(candidate["source_frame_idx"]),
                            anchor_pts=(
                                int(candidate["pts"]) if candidate.get("pts") is not None else None
                            ),
                            anchor_time_base=(
                                str(candidate["time_base"])
                                if candidate.get("time_base") is not None
                                else None
                            ),
                            fps_num=(
                                int(candidate["fps_num"])
                                if candidate.get("fps_num") is not None
                                else None
                            ),
                            fps_den=(
                                int(candidate["fps_den"])
                                if candidate.get("fps_den") is not None
                                else None
                            ),
                        )
                        if not frames:
                            reasons["raw_window_empty"] += 1
                            refined.append(dict(candidate))
                            continue
                        # Do not run the GPU encoder when every raw source is
                        # absent. Still encode exactly once for usable windows.
                        if query_vector is None:
                            try:
                                query_vectors = np.stack([
                                    encode_query_mean(self.encoder, text) for text in texts
                                ])
                            except ValueError as error:
                                raise _RefinementEmbeddingError(str(error)) from error
                            query_vector = query_vectors.mean(axis=0)
                            norm = float(np.linalg.norm(query_vector))
                            if not np.isfinite(norm) or norm <= 0:
                                raise _RefinementEmbeddingError(
                                    "refinement query embedding must have a finite non-zero norm"
                                )
                            query_vector = (query_vector / norm).astype(np.float32, copy=False)
                        vectors = np.asarray(
                            self.encoder.encode_images([frame["path"] for frame in frames]),
                            dtype=np.float32,
                        )
                        similarities = vectors @ query_vector
                        best_index = int(np.argmax(similarities))
                        best = frames[best_index]
                        try:
                            stored = self.frame_sink.persist(
                                candidate,
                                best,
                                refiner_fingerprint=self.fingerprint,
                            )
                        except RefinedFramePersistenceError:
                            reasons["raw_persistence_unavailable"] += 1
                            refined.append(dict(candidate))
                            continue
                        item = dict(candidate)
                        item["coarse_source_frame_idx"] = item["source_frame_idx"]
                        item.pop("_object_assignments", None)
                        item["source_frame_idx"] = stored.source_frame_idx
                        item["timestamp_ms"] = stored.timestamp_ms
                        item["pts"] = stored.pts
                        item["time_base"] = stored.time_base
                        item["image_uri"] = stored.image_uri
                        item["refiner_fingerprint"] = self.fingerprint
                        item["refinement_score"] = float(similarities[best_index])
                        item["evidence"] = list(item.get("evidence", []))
                        item["evidence"].append(
                            {
                                "channel": "raw_refinement",
                                "sample_fps": self.sample_fps,
                                "radius_seconds": self.radius_seconds,
                                "identity_method": "canonical-anchor-relative-pyav-decode",
                            }
                        )
                        refined.append(item)
                        refined_count += 1
            except (FileNotFoundError, OSError, ValueError) as error:
                if isinstance(error, _RefinementEmbeddingError):
                    raise
                # Reference-only videos and unavailable raw archives retain the
                # coarse result instead of causing the whole query to fail.
                refined.append(dict(candidate))
                reason = (
                    "raw_source_unavailable" if isinstance(error, FileNotFoundError)
                    else "raw_storage_unavailable" if isinstance(error, OSError)
                    else "raw_window_invalid"
                )
                reasons[reason] += 1
        refined.extend(dict(candidate) for candidate in candidates[self.top_windows :])
        self._state.diagnostics = {
            "state": "degraded" if reasons else "ready",
            "reason_code": (
                "raw_partial_refinement" if reasons and refined_count
                else next(iter(reasons)) if reasons else "query_completed"
            ),
            "refined_count": refined_count,
            "fallback_counts": dict(reasons),
        }
        return refined
