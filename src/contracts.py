"""Validated, JSON-friendly domain contracts for Textual KIS.

The contracts deliberately keep ``keyframe_ordinal`` (the one-based position of
an extracted image) separate from ``source_frame_idx`` (the zero-based decoded
video-frame index accepted by the historical competition adapter).
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, ClassVar, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from src.query.entity_extractor import (
    MAX_ENTITIES_PER_TRAKE_QUERY,
    EntityConstraint,
    normalize_entity_constraints,
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_QUERY_MODES = frozenset({"automatic", "interactive"})
_KIS_QUERY_MODES = frozenset(
    {"automatic", "interactive", "visual_only", "asr_only"}
)
_KIS_PRECISION_PROFILES = frozenset(
    {"standard", "advanced", "experimental_cross_modal"}
)
_QA_ANSWER_NORMALIZATIONS = frozenset({"numeric", "general_text"})
QA_HISTORICAL_NUMERIC_SPEC_ID = "aic-2024-visual-qa-numeric-v1"
QA_GENERAL_TEXT_SPEC_ID = "aic-2026-visual-qa-general-text-provisional-v1"
_QA_SPEC_IDS = frozenset({QA_HISTORICAL_NUMERIC_SPEC_ID, QA_GENERAL_TEXT_SPEC_ID})


@dataclass(frozen=True, slots=True)
class TextEncodingDiagnostics:
    """Sanitized accounting for tokenizer-aware dense-query chunking."""

    token_count: int
    context_limit: int
    content_limit: int
    chunk_count: int
    processed_tokens: int
    dropped_tokens: int
    query_chunked: bool
    query_truncated: bool

    def __post_init__(self) -> None:
        for name in (
            "token_count",
            "context_limit",
            "content_limit",
            "chunk_count",
            "processed_tokens",
            "dropped_tokens",
        ):
            _require_int(getattr(self, name), name, minimum=0)
        if self.context_limit < 1 or self.content_limit < 1 or self.chunk_count < 1:
            raise ValueError("text encoding limits and chunk_count must be positive")
        if self.processed_tokens + self.dropped_tokens != self.token_count:
            raise ValueError("processed_tokens and dropped_tokens must account for all tokens")
        if self.query_truncated != (self.dropped_tokens > 0):
            raise ValueError("query_truncated must match dropped_tokens")

    def to_dict(self) -> dict[str, int | bool]:
        """Return a JSON-safe diagnostic payload."""

        return asdict(self)


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_int(value: int, name: str, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _as_string_tuple(value: tuple[str, ...] | list[str], name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a list or tuple of strings")
    result = tuple(value)
    for item in result:
        _require_text(item, f"{name} item")
    return result


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    for key in result:
        _require_text(key, f"{name} key")
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{name} must contain only JSON-serializable values") from exc
    return result


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Canonical identity and timing metadata for one retained video frame."""

    batch_id: str
    video_id: str
    keyframe_ordinal: int
    source_frame_idx: int
    pts: int
    time_base: str
    timestamp_ms: int
    fps_num: int
    fps_den: int
    shot_id: str
    sampling_reasons: tuple[str, ...]
    image_uri: str
    video_uri: str
    source_sha256: str
    extractor_version: str

    def __post_init__(self) -> None:
        for name in (
            "batch_id",
            "video_id",
            "time_base",
            "shot_id",
            "image_uri",
            "video_uri",
            "extractor_version",
        ):
            _require_text(getattr(self, name), name)
        _require_int(self.keyframe_ordinal, "keyframe_ordinal", minimum=1)
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        _require_int(self.pts, "pts")
        _require_int(self.timestamp_ms, "timestamp_ms", minimum=0)
        _require_int(self.fps_num, "fps_num", minimum=1)
        _require_int(self.fps_den, "fps_den", minimum=1)
        try:
            time_base = Fraction(self.time_base)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("time_base must be a positive rational such as '1/90000'") from exc
        if time_base <= 0:
            raise ValueError("time_base must be positive")

        reasons = _as_string_tuple(self.sampling_reasons, "sampling_reasons")
        if not reasons:
            raise ValueError("sampling_reasons must contain at least one reason")
        if len(set(reasons)) != len(reasons):
            raise ValueError("sampling_reasons must not contain duplicates")
        object.__setattr__(self, "sampling_reasons", reasons)

        if not isinstance(self.source_sha256, str) or not _SHA256_RE.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "source_sha256", self.source_sha256.lower())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["sampling_reasons"] = list(self.sampling_reasons)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrameRecord:
        """Validate and construct a frame record from serialized data."""
        data = dict(value)
        if "frame_id" in data:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RefinedFrameRecord:
    """Durable exact source frame selected from a retained raw-video anchor.

    A refined source frame is not a retained keyframe and therefore deliberately
    has no ``keyframe_ordinal``.  ``coarse_source_frame_idx`` identifies the
    canonical anchor used to prove its decoded source-frame identity.
    """

    batch_id: str
    video_id: str
    source_frame_idx: int
    coarse_source_frame_idx: int
    pts: int
    time_base: str
    timestamp_ms: int
    fps_num: int
    fps_den: int
    shot_id: str
    image_uri: str
    video_uri: str
    source_sha256: str
    image_sha256: str
    image_width: int
    image_height: int
    image_bytes: int
    refiner_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "batch_id",
            "video_id",
            "time_base",
            "shot_id",
            "image_uri",
            "video_uri",
            "refiner_fingerprint",
        ):
            _require_text(getattr(self, name), name)
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        _require_int(
            self.coarse_source_frame_idx,
            "coarse_source_frame_idx",
            minimum=0,
        )
        _require_int(self.pts, "pts")
        _require_int(self.timestamp_ms, "timestamp_ms", minimum=0)
        _require_int(self.fps_num, "fps_num", minimum=1)
        _require_int(self.fps_den, "fps_den", minimum=1)
        _require_int(self.image_width, "image_width", minimum=1)
        _require_int(self.image_height, "image_height", minimum=1)
        _require_int(self.image_bytes, "image_bytes", minimum=1)
        try:
            time_base = Fraction(self.time_base)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("time_base must be a positive rational such as '1/90000'") from exc
        if time_base <= 0:
            raise ValueError("time_base must be positive")
        for name in ("source_sha256", "image_sha256"):
            digest = getattr(self, name)
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
            object.__setattr__(self, name, digest.lower())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe source-frame metadata without an ordinal alias."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RefinedFrameRecord:
        """Validate and construct a durable refined-frame record."""

        data = dict(value)
        if "frame_id" in data:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        if "keyframe_ordinal" in data:
            raise ValueError("refined source frames do not have a keyframe_ordinal")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TextSegment:
    """Timestamped OCR, ASR, caption, object, or metadata text."""

    segment_id: str
    batch_id: str
    video_id: str
    kind: str
    start_ms: int
    end_ms: int
    source_frame_idx: int | None
    text_raw: str
    text_nfc: str
    text_accent_folded: str
    confidence: float
    model_version: str
    polygons: tuple[tuple[tuple[float, float], ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("segment_id", "batch_id", "video_id", "kind", "model_version"):
            _require_text(getattr(self, name), name)
        if not _KIND_RE.fullmatch(self.kind):
            raise ValueError("kind must be a lowercase identifier")
        _require_int(self.start_ms, "start_ms", minimum=0)
        _require_int(self.end_ms, "end_ms", minimum=0)
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if self.source_frame_idx is not None:
            _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        for name in ("text_raw", "text_nfc", "text_accent_folded"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if not unicodedata.is_normalized("NFC", self.text_nfc):
            raise ValueError("text_nfc must be Unicode NFC-normalized")
        _require_finite_number(self.confidence, "confidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        normalized_polygons: list[tuple[tuple[float, float], ...]] = []
        for polygon in self.polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                raise ValueError("each polygon must contain at least three points")
            normalized_points: list[tuple[float, float]] = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("polygon points must be x/y pairs")
                _require_finite_number(point[0], "polygon x")
                _require_finite_number(point[1], "polygon y")
                normalized_points.append((float(point[0]), float(point[1])))
            normalized_polygons.append(tuple(normalized_points))
        object.__setattr__(self, "polygons", tuple(normalized_polygons))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TextSegment:
        """Validate and construct a text segment from serialized data."""
        data = dict(value)
        if "frame_id" in data:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class FrameReference:
    """Batch-scoped image identity; frame zero is valid."""

    batch_id: str
    video_id: str
    source_frame_idx: int
    keyframe_ordinal: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.batch_id, "batch_id")
        _require_text(self.video_id, "video_id")
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        if self.batch_id == "aic26-b2":
            _require_int(self.keyframe_ordinal, "keyframe_ordinal", minimum=1)


@dataclass(frozen=True, slots=True)
class QueryRequest:
    """A Textual KIS request, including optional temporal clauses and feedback."""

    query_id: str
    original_text: str
    mode: str = "automatic"
    batch_filters: tuple[str, ...] = ()
    limit: int = 100
    before_clause: str | None = None
    current_clause: str | None = None
    after_clause: str | None = None
    quoted_text: tuple[str, ...] | None = None
    named_entities: tuple[str, ...] | None = None
    objects: tuple[str, ...] | None = None
    actions: tuple[str, ...] | None = None
    attributes: tuple[str, ...] | None = None
    positive_feedback_ids: tuple[str, ...] = ()
    negative_feedback_ids: tuple[str, ...] = ()
    query_language: str = "auto"
    english_text_override: str | None = None
    entity_constraints: tuple[EntityConstraint, ...] = ()
    refine_raw: bool = True
    precision_profile: str = "standard"
    reference_frame: FrameReference | None = None
    positive_frames: tuple[FrameReference, ...] = ()
    negative_frames: tuple[FrameReference, ...] = ()
    feedback_text: str | None = None
    image_only: bool = False
    temporal_steps: tuple[str, ...] = ()
    temporal_target: int | None = None
    use_temporal_sequence: bool = False
    temporal_mode: str = "auto"
    interaction_session_id: str | None = None
    interaction_turn: int | None = None
    event_source: str = "unknown"

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.original_text, "original_text")
        if len(self.original_text) > 8_192:
            raise ValueError("original_text cannot exceed 8192 Unicode code points")
        if self.temporal_mode not in {"auto", "manual", "off"}:
            raise ValueError("temporal_mode must be auto, manual or off")
        if self.event_source not in {"unknown", "user", "benchmark", "agent"}:
            raise ValueError("invalid event_source")
        if self.interaction_session_id is not None:
            if not isinstance(self.interaction_session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.interaction_session_id):
                raise ValueError("invalid interaction_session_id")
        if self.interaction_turn is not None:
            _require_int(self.interaction_turn, "interaction_turn", minimum=1)
        for name in ("image_only", "use_temporal_sequence"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.feedback_text is not None:
            _require_text(self.feedback_text, "feedback_text")
            if len(self.feedback_text) > 8192:
                raise ValueError("feedback_text exceeds budget")
        reference = self.reference_frame
        if isinstance(reference, Mapping):
            reference = FrameReference(**reference)
            object.__setattr__(self, "reference_frame", reference)
        if reference is not None and not isinstance(reference, FrameReference):
            raise TypeError("reference_frame must be a frame identity")
        for name in ("positive_frames", "negative_frames"):
            frames = tuple(
                FrameReference(**v) if isinstance(v, Mapping) else v
                for v in getattr(self, name)
            )
            if len(frames) > 20 or any(not isinstance(v, FrameReference) for v in frames):
                raise ValueError("feedback requires at most 20 canonical identities")
            object.__setattr__(self, name, tuple(dict.fromkeys(frames)))
        if set(self.positive_frames) & set(self.negative_frames):
            raise ValueError("positive and negative frames must be disjoint")
        if reference in self.negative_frames:
            raise ValueError("reference cannot also be negative")
        if self.image_only and reference is None:
            raise ValueError("image_only requires reference_frame")
        if self.mode == "asr_only" and (
            reference is not None or self.positive_frames or self.negative_frames
            or self.use_temporal_sequence
        ):
            raise ValueError("image feedback and temporal sequence require visual retrieval")
        steps = _as_string_tuple(self.temporal_steps, "temporal_steps")
        if steps and (not 2 <= len(steps) <= 6 or sum(map(len, steps)) > 8192):
            raise ValueError("temporal_steps requires 2 to 6 bounded events")
        object.__setattr__(self, "temporal_steps", steps)
        if self.temporal_target is not None:
            _require_int(self.temporal_target, "temporal_target", minimum=0)
            if not steps or self.temporal_target >= len(steps):
                raise ValueError("temporal_target must identify an explicit step")
        if self.mode not in _KIS_QUERY_MODES:
            raise ValueError(f"mode must be one of {sorted(_KIS_QUERY_MODES)}")
        if self.query_language not in {"auto", "vi", "en"}:
            raise ValueError("query_language must be auto, vi, or en")
        if self.english_text_override is not None:
            _require_text(self.english_text_override, "english_text_override")
        object.__setattr__(
            self,
            "entity_constraints",
            normalize_entity_constraints(self.entity_constraints),
        )
        if not isinstance(self.refine_raw, bool):
            raise TypeError("refine_raw must be a boolean")
        if not isinstance(self.precision_profile, str):
            raise TypeError("precision_profile must be a string")
        if self.precision_profile not in _KIS_PRECISION_PROFILES:
            raise ValueError(
                "precision_profile must be standard, advanced, or "
                "experimental_cross_modal"
            )
        _require_int(self.limit, "limit", minimum=1)
        if self.limit > 100:
            raise ValueError("limit cannot exceed the competition maximum of 100")
        for name in ("before_clause", "current_clause", "after_clause"):
            clause = getattr(self, name)
            if clause is not None:
                _require_text(clause, name)

        for name in ("quoted_text", "named_entities", "objects", "actions", "attributes"):
            values = getattr(self, name)
            if values is None:
                continue
            normalized = _as_string_tuple(values, name)
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, normalized)

        for name in ("batch_filters", "positive_feedback_ids", "negative_feedback_ids"):
            values = _as_string_tuple(getattr(self, name), name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        overlap = set(self.positive_feedback_ids) & set(self.negative_feedback_ids)
        if overlap:
            raise ValueError("positive and negative feedback IDs must be disjoint")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        for name in (
            "batch_filters",
            "positive_feedback_ids",
            "negative_feedback_ids",
            "quoted_text",
            "named_entities",
            "objects",
            "actions",
            "attributes",
        ):
            if getattr(self, name) is None:
                continue
            result[name] = list(getattr(self, name))
        result["entity_constraints"] = [asdict(value) for value in self.entity_constraints]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QueryRequest:
        """Validate and construct a query request from serialized data."""
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class QAQueryRequest:
    """A Visual Q&A request with distinct retrieval and reasoning text."""

    query_id: str
    event_description: str
    question: str
    mode: str = "automatic"
    answer_spec_id: str = QA_HISTORICAL_NUMERIC_SPEC_ID
    batch_filters: tuple[str, ...] = ()
    limit: int = 100
    reasoning_policy: str = "gemini_then_local"
    allow_external_frame_transfer: bool = False
    query_language: str = "auto"
    event_description_en: str | None = None
    transcript_context: tuple[str, ...] = ()
    entity_constraints: tuple[EntityConstraint, ...] = ()

    def __post_init__(self) -> None:
        for name in ("query_id", "event_description", "question", "answer_spec_id"):
            _require_text(getattr(self, name), name)
        if self.mode not in _QUERY_MODES:
            raise ValueError(f"mode must be one of {sorted(_QUERY_MODES)}")
        if self.query_language not in {"auto", "vi", "en"}:
            raise ValueError("query_language must be auto, vi, or en")
        if self.event_description_en is not None:
            _require_text(self.event_description_en, "event_description_en")
        transcript_context = _as_string_tuple(
            self.transcript_context, "transcript_context"
        )
        if len(transcript_context) > 200 or sum(map(len, transcript_context)) > 8_192:
            raise ValueError("transcript_context exceeds its bounded prompt budget")
        object.__setattr__(self, "transcript_context", transcript_context)
        object.__setattr__(
            self,
            "entity_constraints",
            normalize_entity_constraints(self.entity_constraints),
        )
        if self.answer_spec_id not in _QA_SPEC_IDS:
            raise ValueError(f"unknown Visual Q&A answer_spec_id: {self.answer_spec_id!r}")
        if self.reasoning_policy not in {"gemini_then_local", "local_only"}:
            raise ValueError("reasoning_policy must be gemini_then_local or local_only")
        if not isinstance(self.allow_external_frame_transfer, bool):
            raise TypeError("allow_external_frame_transfer must be a boolean")
        # QAQueryRequest is also the compatibility envelope for deprecated
        # retrieval-only calls, which never transfer a frame. Provider routing
        # and QASingleFrameAnswerRequest enforce consent at the actual Stage-2
        # trust boundary.
        filters = _as_string_tuple(self.batch_filters, "batch_filters")
        if len(filters) != len(set(filters)):
            raise ValueError("batch_filters must not contain duplicates")
        object.__setattr__(self, "batch_filters", filters)
        _require_int(self.limit, "limit", minimum=1)
        if self.limit > 100:
            raise ValueError("limit cannot exceed the competition maximum of 100")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["batch_filters"] = list(self.batch_filters)
        result["transcript_context"] = list(self.transcript_context)
        result["entity_constraints"] = [asdict(value) for value in self.entity_constraints]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QAQueryRequest:
        """Validate and construct a Visual Q&A request."""

        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class QACandidateRequest:
    """Pure retrieval request for the first Visual Q&A stage."""

    query_id: str
    event_description: str
    batch_filters: tuple[str, ...] = ()
    top_k: int = 5
    query_language: str = "auto"
    event_description_en: str | None = None
    entity_constraints: tuple[EntityConstraint, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.event_description, "event_description")
        if self.query_language not in {"auto", "vi", "en"}:
            raise ValueError("query_language must be auto, vi, or en")
        if self.event_description_en is not None:
            _require_text(self.event_description_en, "event_description_en")
        object.__setattr__(
            self,
            "entity_constraints",
            normalize_entity_constraints(self.entity_constraints),
        )
        filters = _as_string_tuple(self.batch_filters, "batch_filters")
        if len(filters) != len(set(filters)):
            raise ValueError("batch_filters must not contain duplicates")
        if len(filters) > 1:
            from src.retrieval.batch_scope import validate_scope

            validate_scope(filters)
        object.__setattr__(self, "batch_filters", filters)
        _require_int(self.top_k, "top_k", minimum=1)
        if self.top_k > 50:
            raise ValueError("top_k cannot exceed 50")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["batch_filters"] = list(self.batch_filters)
        result["entity_constraints"] = [asdict(value) for value in self.entity_constraints]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QACandidateRequest:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class QAFrameInjectionRequest:
    """Create a one-candidate QA session from an operator-selected canonical frame."""

    query_id: str
    batch_id: str
    video_id: str
    source_frame_idx: int
    source: str
    event_description: str
    keyframe_ordinal: int | None = None

    def __post_init__(self) -> None:
        for name in ("query_id", "batch_id", "video_id", "event_description"):
            _require_text(getattr(self, name), name)
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        if self.batch_id == "aic26-b2":
            _require_int(self.keyframe_ordinal, "keyframe_ordinal", minimum=1)
        if self.source not in {"kis_bridge", "manual"}:
            raise ValueError("source must be kis_bridge or manual")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QAFrameInjectionRequest:
        data = dict(value)
        if "frame_idx" in data or "frame_id" in data:
            raise ValueError(
                "ambiguous frame identifier is forbidden; use source_frame_idx"
            )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class QASingleFrameAnswerRequest:
    """On-demand reasoning request bound to one persisted QA candidate."""

    candidate_id: str
    question: str
    answer_spec_id: str = QA_GENERAL_TEXT_SPEC_ID
    reasoning_policy: str = "gemini_then_local"
    allow_external_frame_transfer: bool = False
    answer_engine: str = "vlm"
    florence_task: str | None = None
    florence_target_label: str | None = None
    manual_answer: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.question, "question")
        if self.answer_spec_id not in _QA_SPEC_IDS:
            raise ValueError(f"unknown Visual Q&A answer_spec_id: {self.answer_spec_id!r}")
        if self.reasoning_policy not in {"gemini_then_local", "local_only"}:
            raise ValueError("unsupported Visual Q&A reasoning_policy")
        if not isinstance(self.allow_external_frame_transfer, bool):
            raise TypeError("allow_external_frame_transfer must be a boolean")
        if (
            self.reasoning_policy == "gemini_then_local"
            and not self.allow_external_frame_transfer
        ):
            raise ValueError("Gemini requires external frame transfer consent")
        if self.answer_engine not in {"vlm", "florence2", "manual"}:
            raise ValueError("answer_engine must be vlm, florence2, or manual")
        if self.answer_engine == "manual":
            if self.reasoning_policy != "local_only" or self.allow_external_frame_transfer:
                raise ValueError("manual answers must be local-only without frame transfer")
            _require_text(self.manual_answer, "manual_answer")
            if self.florence_task is not None or self.florence_target_label is not None:
                raise ValueError("manual answers cannot include Florence task fields")
        elif self.manual_answer is not None:
            raise ValueError("manual_answer requires answer_engine=manual")
        elif self.answer_engine == "florence2":
            if self.reasoning_policy != "local_only" or self.allow_external_frame_transfer:
                raise ValueError("Florence-2 is an offline local-only answer engine")
            if self.florence_task not in {"ocr", "object_count"}:
                raise ValueError("Florence-2 supports only ocr or object_count")
            if self.florence_task == "object_count":
                _require_text(self.florence_target_label, "florence_target_label")
            normalized_question = " ".join(self.question.casefold().split())
            unsupported_prefixes = (
                "is ",
                "are ",
                "was ",
                "were ",
                "do ",
                "does ",
                "did ",
                "can ",
                "could ",
                "has ",
                "have ",
                "will ",
                "would ",
                "should ",
            )
            if normalized_question.startswith(unsupported_prefixes) or any(
                marker in normalized_question for marker in (" color", "colour", "màu ")
            ):
                raise ValueError("florence_task_unsupported")
        elif self.florence_task is not None or self.florence_target_label is not None:
            raise ValueError("Florence task fields require answer_engine=florence2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QASingleFrameAnswerRequest:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class QACandidateFrame:
    """One canonical frame returned by pure Visual Q&A retrieval."""

    candidate_id: str
    retrieval_rank: int
    retrieval_score: float
    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    shot_id: str
    channel_ranks: Mapping[str, int] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    keyframe_ordinal: int | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_id", "batch_id", "video_id", "shot_id"):
            _require_text(getattr(self, name), name)
        from src.ingest.frame_identity import candidate_identity

        expected_id = candidate_identity(self)
        if self.candidate_id != expected_id:
            raise ValueError("candidate_id must match the canonical frame identity")
        _require_int(self.retrieval_rank, "retrieval_rank", minimum=1)
        _require_finite_number(self.retrieval_score, "retrieval_score")
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        _require_int(self.timestamp_ms, "timestamp_ms", minimum=0)
        ranks = _json_mapping(self.channel_ranks, "channel_ranks")
        for channel, rank in ranks.items():
            _require_int(rank, f"channel_ranks[{channel!r}]", minimum=1)
        object.__setattr__(self, "retrieval_score", float(self.retrieval_score))
        object.__setattr__(self, "channel_ranks", ranks)
        object.__setattr__(self, "evidence", _json_mapping(self.evidence, "evidence"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QACandidateFrame:
        data = dict(value)
        if "frame_id" in data:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class QACandidateWindow:
    """A retrieval anchor and its ordered canonical evidence frames."""

    cluster_id: str
    retrieval_rank: int
    retrieval_score: float
    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    shot_id: str
    frames: tuple[FrameRecord, ...]
    frame_selection_scores: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("cluster_id", "batch_id", "video_id", "shot_id"):
            _require_text(getattr(self, name), name)
        _require_int(self.retrieval_rank, "retrieval_rank", minimum=1)
        _require_finite_number(self.retrieval_score, "retrieval_score")
        object.__setattr__(self, "retrieval_score", float(self.retrieval_score))
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        _require_int(self.timestamp_ms, "timestamp_ms", minimum=0)

        if not isinstance(self.frames, (tuple, list)):
            raise TypeError("frames must be a list or tuple of FrameRecord values")
        frames = tuple(self.frames)
        if not 3 <= len(frames) <= 5:
            raise ValueError("frames must contain between 3 and 5 canonical frames")
        if not all(isinstance(frame, FrameRecord) for frame in frames):
            raise TypeError("frames must contain only FrameRecord values")
        identities = [(frame.batch_id, frame.video_id, frame.source_frame_idx) for frame in frames]
        if len(identities) != len(set(identities)):
            raise ValueError("frames must not contain duplicate canonical identities")
        if any(
            frame.batch_id != self.batch_id or frame.video_id != self.video_id for frame in frames
        ):
            raise ValueError("all frames must belong to the candidate batch and video")
        if list(frames) != sorted(
            frames,
            key=lambda frame: (frame.timestamp_ms, frame.source_frame_idx),
        ):
            raise ValueError("frames must be ordered chronologically")
        anchor = next(
            (frame for frame in frames if frame.source_frame_idx == self.source_frame_idx),
            None,
        )
        if anchor is None:
            raise ValueError("the retrieval anchor must be present in frames")
        if anchor.timestamp_ms != self.timestamp_ms:
            raise ValueError("timestamp_ms must match the retrieval anchor frame")
        object.__setattr__(self, "frames", frames)
        scores = dict(self.frame_selection_scores)
        valid_frames = {frame.source_frame_idx for frame in frames}
        for source_frame_idx, score in scores.items():
            _require_int(source_frame_idx, "frame_selection_scores key", minimum=0)
            if source_frame_idx not in valid_frames:
                raise ValueError("frame_selection_scores must reference selected frames")
            _require_finite_number(score, "frame_selection_scores value")
            scores[source_frame_idx] = float(score)
        object.__setattr__(self, "frame_selection_scores", scores)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["frames"] = [frame.to_dict() for frame in self.frames]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QACandidateWindow:
        """Validate and construct a canonical Visual Q&A candidate window."""

        data = dict(value)
        if "frame_id" in data:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        data["frames"] = tuple(
            frame if isinstance(frame, FrameRecord) else FrameRecord.from_dict(frame)
            for frame in data.get("frames", ())
        )
        scores = data.get("frame_selection_scores")
        if isinstance(scores, Mapping):
            converted: dict[int, Any] = {}
            for raw_frame, score in scores.items():
                if isinstance(raw_frame, bool):
                    raise TypeError("frame_selection_scores key must be an integer")
                try:
                    source_frame_idx = int(raw_frame)
                except (TypeError, ValueError):
                    raise TypeError("frame_selection_scores key must be an integer") from None
                converted[source_frame_idx] = score
            data["frame_selection_scores"] = converted
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One canonical ranked retrieval result."""

    rank: int
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    shot_id: str
    fused_score: float
    thumbnail_uri: str
    channel_ranks: Mapping[str, int] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_int(self.rank, "rank", minimum=1)
        _require_text(self.video_id, "video_id")
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        _require_int(self.timestamp_ms, "timestamp_ms", minimum=0)
        _require_text(self.shot_id, "shot_id")
        _require_finite_number(self.fused_score, "fused_score")
        _require_text(self.thumbnail_uri, "thumbnail_uri")
        ranks = _json_mapping(self.channel_ranks, "channel_ranks")
        for channel, rank in ranks.items():
            _require_int(rank, f"channel_ranks[{channel!r}]", minimum=1)
        object.__setattr__(self, "channel_ranks", ranks)
        object.__setattr__(self, "evidence", _json_mapping(self.evidence, "evidence"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SearchHit:
        """Validate and construct a search hit from serialized data."""
        data = dict(value)
        if "frame_id" in data:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class QAPrediction:
    """One ranked Visual Q&A answer tied to a canonical source frame."""

    rank: int
    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    answer: str
    confidence_score: float
    answer_source: str
    window_id: str
    evidence: Mapping[str, Any]
    thumbnail_uri: str | None = None
    raw_answer: str | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        _require_int(self.rank, "rank", minimum=1)
        _require_text(self.batch_id, "batch_id")
        _require_text(self.video_id, "video_id")
        _require_int(self.source_frame_idx, "source_frame_idx", minimum=0)
        _require_int(self.timestamp_ms, "timestamp_ms", minimum=0)
        _require_text(self.answer, "answer")
        _require_text(self.answer_source, "answer_source")
        _require_text(self.window_id, "window_id")
        _require_finite_number(self.confidence_score, "confidence_score")
        if not 0.0 <= float(self.confidence_score) <= 1.0:
            raise ValueError("confidence_score must be between 0 and 1")
        object.__setattr__(self, "confidence_score", float(self.confidence_score))
        for name in ("thumbnail_uri", "raw_answer", "model_name"):
            value = getattr(self, name)
            if value is not None:
                _require_text(value, name)
        object.__setattr__(self, "evidence", _json_mapping(self.evidence, "evidence"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QAPrediction:
        """Validate a Visual Q&A prediction without accepting ambiguous frame IDs."""

        data = dict(value)
        if "frame_id" in data:
            raise ValueError("ambiguous frame_id is forbidden; use source_frame_idx")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class QAReasoningResult:
    """Auditable output from one VLM call over a canonical candidate window."""

    raw_answer: str
    normalized_answer: str
    answerable: bool
    evidence_frame_indices: tuple[int, ...]
    provider: str
    model_name: str
    prompt_version: str
    latency_ms: float
    attempts: int = 1
    cache_status: str = "miss"
    failure: Mapping[str, Any] | None = None
    requested_policy: str | None = None
    fallback_used: bool = False
    fallback_reason_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answerable, bool):
            raise TypeError("answerable must be a boolean")
        if not isinstance(self.raw_answer, str) or not isinstance(self.normalized_answer, str):
            raise TypeError("raw_answer and normalized_answer must be strings")
        if self.answerable:
            _require_text(self.raw_answer, "raw_answer")
            _require_text(self.normalized_answer, "normalized_answer")
        indices = tuple(self.evidence_frame_indices)
        for index in indices:
            _require_int(index, "evidence_frame_idx", minimum=1)
        if len(indices) != len(set(indices)):
            raise ValueError("evidence_frame_indices must not contain duplicates")
        if self.answerable and not indices:
            raise ValueError("an answerable result requires at least one evidence frame")
        object.__setattr__(self, "evidence_frame_indices", indices)
        for name in ("provider", "model_name", "prompt_version", "cache_status"):
            _require_text(getattr(self, name), name)
        _require_finite_number(self.latency_ms, "latency_ms")
        if float(self.latency_ms) < 0:
            raise ValueError("latency_ms cannot be negative")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        _require_int(self.attempts, "attempts", minimum=1)
        if self.requested_policy is not None:
            _require_text(self.requested_policy, "requested_policy")
        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used must be a boolean")
        if self.fallback_reason_code is not None:
            _require_text(self.fallback_reason_code, "fallback_reason_code")
        if not self.fallback_used and self.fallback_reason_code is not None:
            raise ValueError("fallback_reason_code requires fallback_used=true")
        if self.failure is not None:
            object.__setattr__(self, "failure", _json_mapping(self.failure, "failure"))

    @property
    def answer(self) -> str:
        """Compatibility alias for reasoners that historically returned ``answer``."""

        return self.normalized_answer

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["evidence_frame_indices"] = list(self.evidence_frame_indices)
        result["answer"] = self.normalized_answer
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QAReasoningResult:
        """Validate and construct an auditable reasoning result."""

        data = dict(value)
        answer = data.pop("answer", None)
        if "normalized_answer" not in data and answer is not None:
            data["normalized_answer"] = answer
        if answer is not None and data.get("normalized_answer") != answer:
            raise ValueError("answer must match normalized_answer")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CompetitionSpec:
    """Versioned rules for validation, export, and interval evaluation."""

    spec_id: str = "aic-2025-textual-kis"
    output_columns: tuple[str, ...] = ("video_name", "frame_idx")
    video_id_pattern: str = r"^L\d{2}_V\d{3}$"
    max_results: int = 100
    cutoffs: tuple[int, ...] = (1, 5, 20, 50, 100)
    interval_start_inclusive: bool = True
    interval_end_inclusive: bool = True
    source_frame_zero_based: bool = True
    include_header: bool = False
    uppercase_video_names: bool = True
    strip_video_extension: bool = True

    ALLOWED_OUTPUT_COLUMNS: ClassVar[frozenset[str]] = frozenset({"video_name", "frame_idx"})

    def __post_init__(self) -> None:
        _require_text(self.spec_id, "spec_id")
        columns = _as_string_tuple(self.output_columns, "output_columns")
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("output_columns must be non-empty and unique")
        unsupported = set(columns) - self.ALLOWED_OUTPUT_COLUMNS
        if unsupported:
            raise ValueError(f"unsupported output columns: {sorted(unsupported)}")
        if set(columns) != self.ALLOWED_OUTPUT_COLUMNS:
            raise ValueError("Textual KIS output must include video_name and frame_idx")
        object.__setattr__(self, "output_columns", columns)
        try:
            re.compile(self.video_id_pattern)
        except re.error as exc:
            raise ValueError("video_id_pattern must be a valid regular expression") from exc
        _require_int(self.max_results, "max_results", minimum=1)
        cutoffs = tuple(self.cutoffs)
        if not cutoffs:
            raise ValueError("cutoffs must not be empty")
        for cutoff in cutoffs:
            _require_int(cutoff, "cutoff", minimum=1)
        if tuple(sorted(set(cutoffs))) != cutoffs:
            raise ValueError("cutoffs must be strictly increasing")
        if cutoffs[-1] > self.max_results:
            raise ValueError("the largest cutoff cannot exceed max_results")
        object.__setattr__(self, "cutoffs", cutoffs)
        for name in (
            "interval_start_inclusive",
            "interval_end_inclusive",
            "source_frame_zero_based",
            "include_header",
            "uppercase_video_names",
            "strip_video_extension",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    def normalize_video_id(self, value: str) -> str:
        """Normalize and validate a competition video name."""
        _require_text(value, "video_id")
        normalized = value.strip()
        if "/" in normalized or "\\" in normalized:
            raise ValueError("video_id must be a name, not a path")
        if self.strip_video_extension and normalized.lower().endswith(".mp4"):
            normalized = normalized[:-4]
        if self.uppercase_video_names:
            normalized = normalized.upper()
        if not re.fullmatch(self.video_id_pattern, normalized):
            raise ValueError(f"video_id {value!r} does not match {self.video_id_pattern!r}")
        return normalized

    def contains_frame(self, frame_idx: int, start_frame_idx: int, end_frame_idx: int) -> bool:
        """Return whether a source-frame index satisfies this spec's interval rules."""
        minimum = 0 if self.source_frame_zero_based else 1
        _require_int(frame_idx, "frame_idx", minimum=minimum)
        _require_int(start_frame_idx, "start_frame_idx", minimum=minimum)
        _require_int(end_frame_idx, "end_frame_idx", minimum=minimum)
        if end_frame_idx < start_frame_idx:
            raise ValueError("end_frame_idx must be at least start_frame_idx")
        starts = (
            frame_idx >= start_frame_idx
            if self.interval_start_inclusive
            else frame_idx > start_frame_idx
        )
        ends = (
            frame_idx <= end_frame_idx if self.interval_end_inclusive else frame_idx < end_frame_idx
        )
        return starts and ends

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["output_columns"] = list(self.output_columns)
        result["cutoffs"] = list(self.cutoffs)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompetitionSpec:
        """Validate and construct a competition spec from serialized data."""
        return cls(**dict(value))

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> CompetitionSpec:
        """Build a spec from a larger application configuration section.

        Runtime-only keys such as ``output_dir`` are intentionally ignored so
        the same versioned section can drive validation and file placement.
        """

        fields = {
            name
            for name, field_definition in cls.__dataclass_fields__.items()
            if field_definition.init
        }
        return cls.from_dict({name: item for name, item in value.items() if name in fields})

    @classmethod
    def historical_2025(cls) -> CompetitionSpec:
        """Return the verified historical AIC Textual KIS defaults."""
        return cls()


@dataclass(frozen=True, slots=True)
class QACompetitionSpec:
    """Versioned Visual Q&A rules kept separate from Textual KIS export rules."""

    spec_id: str = QA_HISTORICAL_NUMERIC_SPEC_ID
    output_columns: tuple[str, ...] = ("video_name", "frame_idx", "answer")
    video_id_pattern: str = r"^L\d{2}_V\d{3}$"
    max_results: int = 100
    cutoffs: tuple[int, ...] = (1, 5, 20, 50, 100)
    interval_start_inclusive: bool = True
    interval_end_inclusive: bool = True
    source_frame_zero_based: bool = True
    include_header: bool = False
    uppercase_video_names: bool = True
    strip_video_extension: bool = True
    answer_normalization: str = "numeric"

    ALLOWED_OUTPUT_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {"video_name", "frame_idx", "answer"}
    )

    def __post_init__(self) -> None:
        _require_text(self.spec_id, "spec_id")
        columns = _as_string_tuple(self.output_columns, "output_columns")
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("output_columns must be non-empty and unique")
        unsupported = set(columns) - self.ALLOWED_OUTPUT_COLUMNS
        if unsupported:
            raise ValueError(f"unsupported output columns: {sorted(unsupported)}")
        if set(columns) != self.ALLOWED_OUTPUT_COLUMNS:
            raise ValueError("Visual Q&A output must include video_name, frame_idx, and answer")
        object.__setattr__(self, "output_columns", columns)
        try:
            re.compile(self.video_id_pattern)
        except re.error as exc:
            raise ValueError("video_id_pattern must be a valid regular expression") from exc
        _require_int(self.max_results, "max_results", minimum=1)
        cutoffs = tuple(self.cutoffs)
        if not cutoffs:
            raise ValueError("cutoffs must not be empty")
        for cutoff in cutoffs:
            _require_int(cutoff, "cutoff", minimum=1)
        if tuple(sorted(set(cutoffs))) != cutoffs:
            raise ValueError("cutoffs must be strictly increasing")
        if cutoffs[-1] > self.max_results:
            raise ValueError("the largest cutoff cannot exceed max_results")
        object.__setattr__(self, "cutoffs", cutoffs)
        for name in (
            "interval_start_inclusive",
            "interval_end_inclusive",
            "source_frame_zero_based",
            "include_header",
            "uppercase_video_names",
            "strip_video_extension",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.answer_normalization not in _QA_ANSWER_NORMALIZATIONS:
            raise ValueError(
                f"answer_normalization must be one of {sorted(_QA_ANSWER_NORMALIZATIONS)}"
            )

    def normalize_video_id(self, value: str) -> str:
        """Normalize a competition video name without changing frame semantics."""

        _require_text(value, "video_id")
        normalized = value.strip()
        if "/" in normalized or "\\" in normalized:
            raise ValueError("video_id must be a name, not a path")
        if self.strip_video_extension and normalized.lower().endswith(".mp4"):
            normalized = normalized[:-4]
        if self.uppercase_video_names:
            normalized = normalized.upper()
        if not re.fullmatch(self.video_id_pattern, normalized):
            raise ValueError(f"video_id {value!r} does not match {self.video_id_pattern!r}")
        return normalized

    def contains_frame(self, frame_idx: int, start_frame_idx: int, end_frame_idx: int) -> bool:
        """Return whether a source-frame index satisfies the configured interval rules."""

        minimum = 0 if self.source_frame_zero_based else 1
        _require_int(frame_idx, "frame_idx", minimum=minimum)
        _require_int(start_frame_idx, "start_frame_idx", minimum=minimum)
        _require_int(end_frame_idx, "end_frame_idx", minimum=minimum)
        if end_frame_idx < start_frame_idx:
            raise ValueError("end_frame_idx must be at least start_frame_idx")
        starts = (
            frame_idx >= start_frame_idx
            if self.interval_start_inclusive
            else frame_idx > start_frame_idx
        )
        ends = (
            frame_idx <= end_frame_idx if self.interval_end_inclusive else frame_idx < end_frame_idx
        )
        return starts and ends

    def normalize_answer(self, value: str, question: str | None = None) -> str:
        """Apply this spec's deterministic answer-normalization policy."""

        from src.vlm.answer_normalizer import normalize_answer

        return normalize_answer(value, mode=self.answer_normalization, question=question)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["output_columns"] = list(self.output_columns)
        result["cutoffs"] = list(self.cutoffs)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QACompetitionSpec:
        """Validate and construct a Visual Q&A competition spec."""

        return cls(**dict(value))

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> QACompetitionSpec:
        """Build a spec while ignoring unrelated runtime configuration keys."""

        fields = {
            name
            for name, field_definition in cls.__dataclass_fields__.items()
            if field_definition.init
        }
        return cls.from_dict({name: item for name, item in value.items() if name in fields})

    @classmethod
    def historical_numeric(cls) -> QACompetitionSpec:
        """Return the provisional historical numeric-answer adapter."""

        return cls()

    @classmethod
    def conservative_general_text(cls) -> QACompetitionSpec:
        """Return an adapter that preserves semantic words and Vietnamese accents."""

        return cls(
            spec_id=QA_GENERAL_TEXT_SPEC_ID,
            answer_normalization="general_text",
        )

    @classmethod
    def for_spec_id(cls, spec_id: str) -> QACompetitionSpec:
        """Resolve a supported answer spec without silently changing policies."""

        _require_text(spec_id, "answer_spec_id")
        if spec_id == QA_HISTORICAL_NUMERIC_SPEC_ID:
            return cls.historical_numeric()
        if spec_id == QA_GENERAL_TEXT_SPEC_ID:
            return cls.conservative_general_text()
        raise ValueError(f"unknown Visual Q&A answer_spec_id: {spec_id!r}")


def resolve_qa_competition_spec(spec_id: str) -> QACompetitionSpec:
    """Return the registered Visual Q&A competition/normalization adapter."""

    return QACompetitionSpec.for_spec_id(spec_id)


class _TRAKEModel(BaseModel):
    """Strict immutable base for public TRAKE request and response contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class TRAKEStage(_TRAKEModel):
    """One ordered natural-language stage in a TRAKE event sequence."""

    stage_index: int = Field(ge=1, le=20, strict=True)
    description: str = Field(min_length=1)
    description_en: str | None = Field(default=None, min_length=1)
    entity_constraints: tuple[EntityConstraint, ...] = ()

    @field_validator("entity_constraints", mode="before")
    @classmethod
    def _validate_entity_constraints(cls, value: Any) -> tuple[EntityConstraint, ...]:
        return normalize_entity_constraints(value)


class TRAKEQueryRequest(_TRAKEModel):
    """A validated TRAKE retrieval request before runtime batch resolution."""

    query_id: str = Field(min_length=1)
    global_description: str = Field(min_length=1)
    stages: tuple[TRAKEStage, ...] = Field(min_length=2, max_length=20)
    batch_filters: tuple[str, ...] = ()
    top_k: int = Field(default=100, ge=1, le=100, strict=True)
    query_language: Literal["auto", "vi", "en"] = "auto"
    global_description_en: str | None = Field(default=None, min_length=1)
    global_entity_constraints: tuple[EntityConstraint, ...] = ()

    @field_validator("global_entity_constraints", mode="before")
    @classmethod
    def _validate_global_entity_constraints(cls, value: Any) -> tuple[EntityConstraint, ...]:
        return normalize_entity_constraints(value)

    @field_validator("batch_filters")
    @classmethod
    def _validate_batch_filters(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("batch_filters must contain only non-empty strings")
        if len(value) > 1:
            from src.retrieval.batch_scope import validate_scope

            try:
                validate_scope(value)
            except ValueError:
                raise ValueError("TRAKE requires exactly one batch outside audited B01 + B02 scope") from None
        return value

    @model_validator(mode="after")
    def _validate_stage_order(self) -> TRAKEQueryRequest:
        expected = tuple(range(1, len(self.stages) + 1))
        actual = tuple(stage.stage_index for stage in self.stages)
        if actual != expected:
            raise ValueError("stage_index values must be contiguous from 1 in input order")
        total_entities = len(self.global_entity_constraints) + sum(
            len(stage.entity_constraints) for stage in self.stages
        )
        if total_entities > MAX_ENTITIES_PER_TRAKE_QUERY:
            raise ValueError(
                f"TRAKE entity constraints cannot exceed {MAX_ENTITIES_PER_TRAKE_QUERY} entries"
            )
        return self

    def resolve_batch(self, default_batch_id: str | None = None) -> str:
        """Return the sole requested batch after applying a runtime default."""

        if len(self.batch_filters) > 1:
            raise ValueError("alignment cores require exactly one batch")
        if self.batch_filters:
            return self.batch_filters[0]
        if not isinstance(default_batch_id, str) or not default_batch_id.strip():
            raise ValueError("TRAKE requires exactly one batch")
        return default_batch_id.strip()


class TRAKEStageMatch(_TRAKEModel):
    """One stage aligned to a canonical source frame."""

    stage_index: int = Field(ge=1, le=20, strict=True)
    source_frame_idx: int = Field(ge=0, strict=True)
    timestamp_ms: int = Field(ge=0, strict=True)
    similarity: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False, strict=True)
    keyframe_ordinal: int | None = Field(default=None, ge=1, strict=True, exclude_if=lambda value: value is None)


class TRAKEResponseItem(_TRAKEModel):
    """The best strict-monotonic alignment retained for one video."""

    rank: int = Field(ge=1, le=100, strict=True)
    batch_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    stage_matches: tuple[TRAKEStageMatch, ...] = Field(min_length=2, max_length=20)
    confidence_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def _validate_alignment(self) -> TRAKEResponseItem:
        expected = tuple(range(1, len(self.stage_matches) + 1))
        actual = tuple(item.stage_index for item in self.stage_matches)
        if actual != expected:
            raise ValueError("stage matches must be contiguous from 1")
        frames = tuple(item.source_frame_idx for item in self.stage_matches)
        if any(left >= right for left, right in zip(frames, frames[1:])):
            raise ValueError("source_frame_idx values must be strictly increasing")
        timestamps = tuple(item.timestamp_ms for item in self.stage_matches)
        if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamp_ms values must be strictly increasing")
        return self

    @computed_field(alias="frame_ids", return_type=tuple[int, ...])
    @property
    def source_frame_indices(self) -> tuple[int, ...]:
        """Return canonical frame indices, serialized to the public ``frame_ids`` alias."""

        return tuple(item.source_frame_idx for item in self.stage_matches)


class TRAKEResponse(_TRAKEModel):
    """Ranked TRAKE results plus sanitized retrieval-channel status."""

    query_id: str = Field(min_length=1)
    results: tuple[TRAKEResponseItem, ...] = Field(default=(), max_length=100)
    channel_statuses: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ranking(self) -> TRAKEResponse:
        ranks = tuple(item.rank for item in self.results)
        if ranks != tuple(range(1, len(self.results) + 1)):
            raise ValueError("TRAKE result ranks must be contiguous from 1")
        videos = tuple((item.batch_id, item.video_id) for item in self.results)
        if len(videos) != len(set(videos)):
            raise ValueError("TRAKE results must contain at most one alignment per video")
        return self


class TRAKECompetitionSpec(_TRAKEModel):
    """Versioned provisional TRAKE export and interval-scoring semantics."""

    spec_id: str = "historical_2025_provisional"
    video_id_pattern: str = r"^L\d{2}_V\d{3}$"
    max_results: int = Field(default=100, ge=1, le=100, strict=True)
    cutoffs: tuple[int, ...] = (1, 5, 20, 50, 100)
    interval_start_inclusive: bool = True
    interval_end_inclusive: bool = True
    source_frame_zero_based: bool = True
    include_header: bool = False
    uppercase_video_names: bool = True
    strip_video_extension: bool = True

    @field_validator("video_id_pattern")
    @classmethod
    def _validate_video_pattern(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError("video_id_pattern must be a valid regular expression") from exc
        return value

    @field_validator("cutoffs", mode="before")
    @classmethod
    def _validate_cutoff_types(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
            raise ValueError("cutoffs must be a list or tuple of integers")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("cutoffs must contain only integers")
        return value

    @model_validator(mode="after")
    def _validate_cutoffs(self) -> TRAKECompetitionSpec:
        if not self.cutoffs:
            raise ValueError("cutoffs must not be empty")
        if any(isinstance(value, bool) or value < 1 for value in self.cutoffs):
            raise ValueError("cutoffs must contain positive integers")
        if tuple(sorted(set(self.cutoffs))) != self.cutoffs:
            raise ValueError("cutoffs must be strictly increasing")
        if self.cutoffs[-1] > self.max_results:
            raise ValueError("the largest cutoff cannot exceed max_results")
        if self.include_header:
            raise ValueError("the provisional TRAKE format has no header")
        return self

    def normalize_video_id(self, value: str) -> str:
        """Normalize and validate one organizer video name."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("video_id must be a non-empty string")
        normalized = value.strip()
        if "/" in normalized or "\\" in normalized:
            raise ValueError("video_id must be a name, not a path")
        if self.strip_video_extension and normalized.casefold().endswith(".mp4"):
            normalized = normalized[:-4]
        if self.uppercase_video_names:
            normalized = normalized.upper()
        if not re.fullmatch(self.video_id_pattern, normalized):
            raise ValueError(f"video_id {value!r} does not match {self.video_id_pattern!r}")
        return normalized

    def contains_frame(self, frame_idx: int, start_frame_idx: int, end_frame_idx: int) -> bool:
        """Return whether a canonical frame falls inside an allowed interval."""

        minimum = 0 if self.source_frame_zero_based else 1
        for name, value in (
            ("frame_idx", frame_idx),
            ("start_frame_idx", start_frame_idx),
            ("end_frame_idx", end_frame_idx),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if end_frame_idx < start_frame_idx:
            raise ValueError("end_frame_idx must be at least start_frame_idx")
        starts = (
            frame_idx >= start_frame_idx
            if self.interval_start_inclusive
            else frame_idx > start_frame_idx
        )
        ends = (
            frame_idx <= end_frame_idx if self.interval_end_inclusive else frame_idx < end_frame_idx
        )
        return starts and ends

    @classmethod
    def historical_2025_provisional(cls) -> TRAKECompetitionSpec:
        """Return the locked provisional adapter without asserting AIC 2026 rules."""

        return cls()
