"""Safe, bounded resolution of canonical frame images for Visual Q&A."""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

from src.contracts import FrameRecord, RefinedFrameRecord

_ALLOWED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:$")


class CanonicalImageError(ValueError):
    """Raised when canonical image provenance or image content is unsafe."""


class CanonicalImageTooLargeError(CanonicalImageError):
    """Raised when a canonical image exceeds a configured safety bound."""


@dataclass(frozen=True, slots=True)
class CanonicalImageProbe:
    """Cheap, secret-free revision identity used before reading image bytes."""

    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    source_revision: str
    resolver_fingerprint: str

    @property
    def cache_key(self) -> str:
        """Return the stable key used by the persistent fingerprint cache."""

        payload = {
            "batch_id": self.batch_id,
            "video_id": self.video_id,
            "source_frame_idx": self.source_frame_idx,
            "source_revision": self.source_revision,
            "resolver_fingerprint": self.resolver_fingerprint,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedCanonicalImage:
    """Validated RGB image bytes with canonical frame provenance."""

    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    content_sha256: str
    width: int
    height: int
    mime_type: str
    data: bytes
    raw_sha256: str = ""
    source_revision: str = ""
    resolver_fingerprint: str = ""


class ImageResolver(Protocol):
    """Reasoner-facing image resolver protocol."""

    def resolve(
        self,
        frame: FrameRecord | RefinedFrameRecord,
    ) -> ResolvedCanonicalImage: ...


class CanonicalImageResolver:
    """Resolve local or ``zip://`` FrameRecord images under approved roots.

    Compressed input size, decoded pixels, and resized output size are bounded.
    EXIF orientation is applied before conversion to RGB and resizing.
    """

    def __init__(
        self,
        allowed_roots: tuple[str | Path, ...],
        *,
        max_input_bytes: int = 32 * 1024 * 1024,
        max_pixels: int = 40_000_000,
        max_dimension: int = 1280,
        jpeg_quality: int = 90,
        max_open_archives: int = 8,
        normalized_cache_entries: int = 256,
        normalized_cache_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if not allowed_roots:
            raise ValueError("allowed_roots cannot be empty")
        if max_input_bytes < 1 or max_pixels < 1 or max_dimension < 1:
            raise ValueError("image bounds must be positive")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")
        if max_open_archives < 1:
            raise ValueError("max_open_archives must be positive")
        if normalized_cache_entries < 1 or normalized_cache_bytes < 1:
            raise ValueError("normalized image cache bounds must be positive")
        self.allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)
        self.max_input_bytes = int(max_input_bytes)
        self.max_pixels = int(max_pixels)
        self.max_dimension = int(max_dimension)
        self.jpeg_quality = int(jpeg_quality)
        self.max_open_archives = int(max_open_archives)
        self.normalized_cache_entries = int(normalized_cache_entries)
        self.normalized_cache_bytes = int(normalized_cache_bytes)
        self._archive_guard = threading.RLock()
        self._archives: OrderedDict[
            Path,
            tuple[
                tuple[int, int],
                zipfile.ZipFile,
                frozenset[str],
                threading.RLock,
            ],
        ] = OrderedDict()
        self._image_guard = threading.RLock()
        self._images: OrderedDict[str, ResolvedCanonicalImage] = OrderedDict()
        self._image_bytes = 0
        # Fixed stripes provide same-key single-flight without retaining one
        # lock for every frame ever seen by a long-lived competition process.
        self._normalization_locks = tuple(threading.RLock() for _ in range(64))

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for normalized output semantics."""

        payload = {
            "version": "canonical-image-resolver-v3",
            "max_input_bytes": self.max_input_bytes,
            "max_pixels": self.max_pixels,
            "max_dimension": self.max_dimension,
            "format": "jpeg",
            "jpeg_quality": self.jpeg_quality,
            "exif_transpose": True,
            "colorspace": "RGB",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _approved_path(self, value: str) -> Path:
        path = Path(value).resolve()
        if not any(path == root or root in path.parents for root in self.allowed_roots):
            raise CanonicalImageError(f"canonical image path is outside approved roots: {path}")
        return path

    @staticmethod
    def _member(value: str) -> PurePosixPath:
        member = PurePosixPath(value.replace("\\", "/"))
        if (
            member.is_absolute()
            or not member.parts
            or ".." in member.parts
            or _DRIVE_PREFIX.fullmatch(member.parts[0])
        ):
            raise CanonicalImageError(f"unsafe ZIP image member: {value!r}")
        if member.suffix.casefold() not in _ALLOWED_SUFFIXES:
            raise CanonicalImageError(f"unsupported canonical image type: {member.suffix}")
        return member

    def _read_local(self, uri: str) -> bytes:
        path = self._approved_path(uri)
        if not path.is_file() or path.suffix.casefold() not in _ALLOWED_SUFFIXES:
            raise CanonicalImageError(f"canonical image is missing or unsupported: {path}")
        size = path.stat().st_size
        if size < 1 or size > self.max_input_bytes:
            raise CanonicalImageTooLargeError(
                f"canonical image size {size} exceeds configured bounds"
            )
        with path.open("rb") as handle:
            data = handle.read(self.max_input_bytes + 1)
        if len(data) != size:
            raise CanonicalImageError("canonical image changed while it was being read")
        return data

    def _zip_parts(self, uri: str) -> tuple[Path, str]:
        archive_value, separator, member_value = uri[len("zip://") :].partition("!/")
        if not separator or not archive_value or not member_value:
            raise CanonicalImageError("invalid canonical zip URI")
        archive_path = self._approved_path(archive_value)
        if not archive_path.is_file() or archive_path.suffix.casefold() != ".zip":
            raise CanonicalImageError(f"canonical ZIP is missing: {archive_path}")
        self._member(member_value)
        return archive_path, member_value

    def _archive_entry(
        self, archive_path: Path
    ) -> tuple[zipfile.ZipFile, frozenset[str], threading.RLock]:
        stat = archive_path.stat()
        revision = (int(stat.st_size), int(stat.st_mtime_ns))
        with self._archive_guard:
            cached = self._archives.get(archive_path)
            if cached is not None and cached[0] == revision:
                self._archives.move_to_end(archive_path)
                return cached[1], cached[2], cached[3]
            if cached is not None:
                with cached[3]:
                    cached[1].close()
                del self._archives[archive_path]
            archive = zipfile.ZipFile(archive_path)
            duplicate_members = self._duplicate_member_names(archive)
            lock = threading.RLock()
            self._archives[archive_path] = (revision, archive, duplicate_members, lock)
            while len(self._archives) > self.max_open_archives:
                evicted_key = next(
                    (key for key in self._archives if key != archive_path), None
                )
                if evicted_key is None:
                    break
                evicted = self._archives[evicted_key]
                if not evicted[3].acquire(blocking=False):
                    # A concurrent member read owns this archive. Temporarily
                    # exceed the handle target rather than closing it in use.
                    break
                try:
                    del self._archives[evicted_key]
                    evicted[1].close()
                finally:
                    evicted[3].release()
            return archive, duplicate_members, lock

    @staticmethod
    def _duplicate_member_names(archive: zipfile.ZipFile) -> frozenset[str]:
        """Return only duplicated names instead of copying the ZIP member index."""

        # ZipFile already owns a compact name-to-info lookup. The overwhelmingly
        # common case has no duplicate names and therefore needs no second scan
        # or per-member container at all.
        if len(archive.filelist) == len(archive.NameToInfo):
            return frozenset()
        seen: set[str] = set()
        duplicates: set[str] = set()
        for info in archive.filelist:
            if info.filename in seen:
                duplicates.add(info.filename)
            else:
                seen.add(info.filename)
        return frozenset(duplicates)

    def _validated_zip_info(
        self,
        archive: zipfile.ZipFile,
        member_value: str,
        duplicate_members: frozenset[str],
    ) -> zipfile.ZipInfo:
        """Return a safe unique image member from an already-open archive."""

        self._member(member_value)
        if member_value in duplicate_members:
            raise CanonicalImageError("canonical ZIP member is missing or duplicated")
        try:
            info = archive.getinfo(member_value)
        except KeyError as exc:
            raise CanonicalImageError(
                "canonical ZIP member is missing or duplicated"
            ) from exc
        self._member(info.filename)
        if info.is_dir() or info.flag_bits & 0x1:
            raise CanonicalImageError("canonical ZIP member is not a readable image")
        if info.file_size < 1 or info.file_size > self.max_input_bytes:
            raise CanonicalImageTooLargeError(
                "canonical ZIP image exceeds configured size"
            )
        return info

    def _read_zip(self, uri: str) -> bytes:
        archive_path, member_value = self._zip_parts(uri)
        try:
            with self._archive_guard:
                archive, duplicate_members, lock = self._archive_entry(archive_path)
                lock.acquire()
            try:
                info = self._validated_zip_info(
                    archive, member_value, duplicate_members
                )
                with archive.open(info) as handle:
                    data = handle.read(self.max_input_bytes + 1)
                if len(data) != info.file_size:
                    raise CanonicalImageError("canonical ZIP image size mismatch")
                return data
            finally:
                lock.release()
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise CanonicalImageError(f"cannot read canonical ZIP image: {exc}") from exc

    def source_uri_resolvable(self, uri: str) -> bool:
        """Return whether a canonical image source is safely addressable.

        This is the cheap readiness counterpart to :meth:`resolve`: it applies
        the same root, scheme, suffix, size, archive-member, duplication, and
        encryption checks without reading or decoding image bytes.
        """

        return self.source_uris_resolvable((uri,))

    def _local_source_uri_resolvable(self, uri: str) -> bool:
        """Check one local source without opening or decoding it."""

        if "://" in uri:
            return False
        path = self._approved_path(uri)
        if not path.is_file() or path.suffix.casefold() not in _ALLOWED_SUFFIXES:
            return False
        return 1 <= path.stat().st_size <= self.max_input_bytes

    def source_uris_resolvable(self, uris: Iterable[str]) -> bool:
        """Check all sources while retaining at most one temporary ZIP handle."""

        active_archive_value: str | None = None
        active_archive: zipfile.ZipFile | None = None
        active_duplicate_members: frozenset[str] = frozenset()
        try:
            for uri in uris:
                if not isinstance(uri, str) or not uri.strip():
                    return False
                if not uri.startswith("zip://"):
                    if not self._local_source_uri_resolvable(uri):
                        return False
                    continue
                archive_value, separator, member_value = uri[len("zip://") :].partition(
                    "!/"
                )
                if not separator or not archive_value or not member_value:
                    return False
                if active_archive_value != archive_value:
                    if active_archive is not None:
                        active_archive.close()
                    archive_path = self._approved_path(archive_value)
                    if (
                        not archive_path.is_file()
                        or archive_path.suffix.casefold() != ".zip"
                    ):
                        return False
                    active_archive = zipfile.ZipFile(archive_path)
                    active_archive_value = archive_value
                    active_duplicate_members = self._duplicate_member_names(active_archive)
                if active_archive is None:  # pragma: no cover - guarded above
                    return False
                self._validated_zip_info(
                    active_archive, member_value, active_duplicate_members
                )
            return True
        except (CanonicalImageError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
            return False
        finally:
            if active_archive is not None:
                active_archive.close()

    def probe(self, frame: FrameRecord | RefinedFrameRecord) -> CanonicalImageProbe:
        """Validate provenance and compute a cheap source revision without decoding."""

        if not isinstance(frame, (FrameRecord, RefinedFrameRecord)):
            raise TypeError("frame must be a persisted source-frame record")
        if "://" in frame.image_uri and not frame.image_uri.startswith("zip://"):
            raise CanonicalImageError("unsupported canonical image URI")
        if frame.image_uri.startswith("zip://"):
            archive_path, member_value = self._zip_parts(frame.image_uri)
            stat = archive_path.stat()
            source = {
                "kind": "zip",
                "location": hashlib.sha256(str(archive_path).encode()).hexdigest(),
                "member": hashlib.sha256(member_value.encode()).hexdigest(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        else:
            path = self._approved_path(frame.image_uri)
            if not path.is_file() or path.suffix.casefold() not in _ALLOWED_SUFFIXES:
                raise CanonicalImageError("canonical image is missing or unsupported")
            stat = path.stat()
            if stat.st_size < 1 or stat.st_size > self.max_input_bytes:
                raise CanonicalImageTooLargeError(
                    "canonical image exceeds configured bounds"
                )
            source = {
                "kind": "local",
                "location": hashlib.sha256(str(path).encode()).hexdigest(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        source.update(
            {
                "source_sha256": frame.source_sha256,
                "image_sha256": getattr(frame, "image_sha256", None),
                "producer_fingerprint": getattr(
                    frame,
                    "extractor_version",
                    getattr(frame, "refiner_fingerprint", None),
                ),
                "batch_id": frame.batch_id,
                "video_id": frame.video_id,
                "source_frame_idx": frame.source_frame_idx,
            }
        )
        revision = hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return CanonicalImageProbe(
            batch_id=frame.batch_id,
            video_id=frame.video_id,
            source_frame_idx=frame.source_frame_idx,
            timestamp_ms=frame.timestamp_ms,
            source_revision=revision,
            resolver_fingerprint=self.fingerprint,
        )

    def _cached_image(self, key: str) -> ResolvedCanonicalImage | None:
        with self._image_guard:
            value = self._images.get(key)
            if value is not None:
                self._images.move_to_end(key)
            return value

    def _cache_image(self, key: str, value: ResolvedCanonicalImage) -> None:
        with self._image_guard:
            prior = self._images.pop(key, None)
            if prior is not None:
                self._image_bytes -= len(prior.data)
            self._images[key] = value
            self._image_bytes += len(value.data)
            while (
                len(self._images) > self.normalized_cache_entries
                or self._image_bytes > self.normalized_cache_bytes
            ):
                _, evicted = self._images.popitem(last=False)
                self._image_bytes -= len(evicted.data)

    def close(self) -> None:
        """Close cached archive handles and clear bounded in-memory image data."""

        with self._archive_guard:
            for _, archive, _, lock in self._archives.values():
                with lock:
                    archive.close()
            self._archives.clear()
        with self._image_guard:
            self._images.clear()
            self._image_bytes = 0

    def resolve(
        self,
        frame: FrameRecord | RefinedFrameRecord,
    ) -> ResolvedCanonicalImage:
        """Validate, orient, resize, and encode one canonical image as JPEG."""

        probe_value = self.probe(frame)
        cached = self._cached_image(probe_value.cache_key)
        if cached is not None:
            return cached
        stripe = self._normalization_locks[
            int(probe_value.cache_key[:8], 16) % len(self._normalization_locks)
        ]
        with stripe:
            cached = self._cached_image(probe_value.cache_key)
            if cached is not None:
                return cached
            raw = (
                self._read_zip(frame.image_uri)
                if frame.image_uri.startswith("zip://")
                else self._read_local(frame.image_uri)
            )
            expected_image_hash = getattr(frame, "image_sha256", None)
            raw_sha256 = hashlib.sha256(raw).hexdigest()
            if expected_image_hash is not None and raw_sha256 != expected_image_hash:
                raise CanonicalImageError("persisted refined image checksum does not match")
            try:
                with Image.open(io.BytesIO(raw)) as probe:
                    width, height = probe.size
                    if width < 1 or height < 1 or width * height > self.max_pixels:
                        raise CanonicalImageTooLargeError(
                            "decoded canonical image exceeds pixel bounds"
                        )
                    probe.verify()
                with Image.open(io.BytesIO(raw)) as source:
                    oriented = ImageOps.exif_transpose(source)
                    image = oriented.convert("RGB")
                    image.thumbnail(
                        (self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS
                    )
                    image.load()
                    output = io.BytesIO()
                    image.save(
                        output,
                        format="JPEG",
                        quality=self.jpeg_quality,
                        optimize=False,
                        progressive=False,
                    )
                    encoded = output.getvalue()
                    # Cache identity is based on deterministic normalized pixels,
                    # not container metadata or the organizer's source encoding.
                    digest = hashlib.sha256(encoded).hexdigest()
                    resolved = ResolvedCanonicalImage(
                        batch_id=frame.batch_id,
                        video_id=frame.video_id,
                        source_frame_idx=frame.source_frame_idx,
                        timestamp_ms=frame.timestamp_ms,
                        content_sha256=digest,
                        width=image.width,
                        height=image.height,
                        mime_type="image/jpeg",
                        data=encoded,
                        raw_sha256=raw_sha256,
                        source_revision=probe_value.source_revision,
                        resolver_fingerprint=probe_value.resolver_fingerprint,
                    )
                    self._cache_image(probe_value.cache_key, resolved)
                    return resolved
            except (UnidentifiedImageError, OSError, SyntaxError) as exc:
                raise CanonicalImageError(f"invalid canonical image content: {exc}") from exc


__all__ = [
    "CanonicalImageError",
    "CanonicalImageProbe",
    "CanonicalImageResolver",
    "CanonicalImageTooLargeError",
    "ImageResolver",
    "ResolvedCanonicalImage",
]
