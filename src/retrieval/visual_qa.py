"""Visual Q&A retrieval, bounded media resolution, reasoning, and session cascade."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

from src.contracts import (
    FrameRecord,
    QACandidateFrame,
    QACandidateRequest,
    QACompetitionSpec,
    QAPrediction,
    QAQueryRequest,
    QASingleFrameAnswerRequest,
)
from src.retrieval.qa_media import (
    CanonicalImageError,
    CanonicalImageProbe,
    CanonicalImageTooLargeError,
    ImageResolver,
    ResolvedCanonicalImage,
)
from src.retrieval.qa_retrieval import QACandidateCluster, VisualQARetriever
from src.retrieval.qa_session import QASessionStore, QASessionStoreError, reasoning_cache_key


class QAReasoner(Protocol):
    """Provider-neutral reasoner protocol implemented under ``src.vlm``."""

    def reason(self, request: Any, images: Sequence[Any], spec: Any) -> Any: ...


_OUTCOME_STATUSES = frozenset(
    {
        "pending",
        "answered",
        "unanswerable",
        "media_failed",
        "provider_failed",
        "internal_failed",
    }
)
_FAILURE_CODES = frozenset(
    {
        "media_unavailable",
        "media_window_too_large",
        "provider_unavailable",
        "provider_request_failed",
        "provider_response_invalid",
        "internal_reasoning_error",
    }
)
_PROVIDER_ERROR_CATEGORIES = frozenset(
    {
        "auth_or_permission",
        "client_initialization",
        "client_unavailable",
        "invalid_request",
        "model_unavailable",
        "model_missing",
        "network_unreachable",
        "not_configured",
        "ollama_context_exceeded",
        "ollama_timeout",
        "ollama_response_invalid",
        "provider_chain_exhausted",
        "provider_request_failed",
        "rate_limited",
        "request_rejected",
        "response_invalid",
        "resource_exhausted",
        "server_unavailable",
    }
)


@dataclass(frozen=True, slots=True)
class QAWindowFrameRef:
    """Safe canonical frame reference persisted without a media URI."""

    position: int
    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or self.position < 1:
            raise ValueError("position must be a positive integer")
        if not self.batch_id or not self.video_id:
            raise ValueError("window frame requires batch_id and video_id")
        if isinstance(self.source_frame_idx, bool) or self.source_frame_idx < 0:
            raise ValueError("source_frame_idx must be non-negative")
        if isinstance(self.timestamp_ms, bool) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QAWindowOutcome:
    """One safe semantic or technical result for a candidate window."""

    window_id: str
    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    shot_id: str
    retrieval_rank: int
    retrieval_score: float
    frames: tuple[QAWindowFrameRef, ...]
    status: str
    answer: str = ""
    raw_answer: str = ""
    normalized_answer: str = ""
    answerable: bool | None = None
    evidence_frame_indices: tuple[int, ...] = ()
    provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    frame_prepare_ms: float | None = None
    provider_latency_ms: float | None = None
    attempts: int | None = None
    cache_status: str | None = None
    served_latency_ms: float | None = None
    cache_layers: Mapping[str, str] | None = None
    failure_code: str | None = None
    failure_stage: str | None = None
    provider_error_category: str | None = None
    provider_status_code: int | None = None

    def __post_init__(self) -> None:
        if not self.window_id or not self.batch_id or not self.video_id or not self.shot_id:
            raise ValueError("QA outcome identity fields must be non-empty")
        if self.status not in _OUTCOME_STATUSES:
            raise ValueError(f"unsupported QA outcome status: {self.status!r}")
        if not 3 <= len(self.frames) <= 5:
            raise ValueError("QA outcome requires three to five frame references")
        if tuple(frame.position for frame in self.frames) != tuple(range(1, len(self.frames) + 1)):
            raise ValueError("QA outcome frame positions must be contiguous from one")
        if self.failure_code is not None and self.failure_code not in _FAILURE_CODES:
            raise ValueError(f"unsupported QA failure code: {self.failure_code!r}")
        if self.status.endswith("_failed") and (
            self.failure_code is None or not self.failure_stage
        ):
            raise ValueError("failed QA outcomes require a safe code and stage")
        if (
            self.provider_error_category is not None
            and self.provider_error_category not in _PROVIDER_ERROR_CATEGORIES
        ):
            raise ValueError("unsupported provider error category")
        if self.provider_status_code is not None and not (
            isinstance(self.provider_status_code, int)
            and not isinstance(self.provider_status_code, bool)
            and 100 <= self.provider_status_code <= 599
        ):
            raise ValueError("provider_status_code must be an HTTP status code")
        if self.status in {"answered", "unanswerable"}:
            if self.answerable is not (self.status == "answered"):
                raise ValueError("QA semantic status and answerable must agree")
        if self.status == "answered" and not self.answer:
            raise ValueError("answered QA outcomes require a normalized answer")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["frames"] = [frame.to_dict() for frame in self.frames]
        value["evidence_frame_indices"] = list(self.evidence_frame_indices)
        value["cache_layers"] = dict(self.cache_layers or {})
        return value


class VisualQATotalFailureError(RuntimeError):
    """Sanitized total technical failure with a restorable search identity."""

    def __init__(self, search_id: str, failure_codes: Sequence[str]) -> None:
        self.search_id = search_id
        self.status = "failed"
        self.failure_codes = tuple(sorted(set(failure_codes)))
        super().__init__("Visual Q&A failed technically for every candidate")


class _ImageWindowTooLargeError(CanonicalImageTooLargeError):
    pass


@dataclass(frozen=True, slots=True)
class _FallbackQAImage:
    data: bytes
    mime_type: str
    index: int
    source_frame_idx: int
    timestamp_ms: int
    sha256: str


def _request_value(request: Any, name: str, default: Any = None) -> Any:
    if isinstance(request, Mapping):
        return request.get(name, default)
    return getattr(request, name, default)


def _json_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif hasattr(value, "to_dict"):
        result = dict(value.to_dict())
    elif is_dataclass(value):
        result = asdict(value)
    else:
        result = {name: item for name, item in vars(value).items() if not name.startswith("_")}
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _qa_image(value: ResolvedCanonicalImage, index: int) -> Any:
    fields = {
        "data": value.data,
        "mime_type": value.mime_type,
        "index": index,
        "source_frame_idx": value.source_frame_idx,
        "timestamp_ms": value.timestamp_ms,
        "sha256": value.content_sha256,
    }
    try:
        from src.vlm.qa_reasoner import QAImage

        return QAImage(**fields)
    except ImportError:
        return _FallbackQAImage(**fields)


class VisualQAPipeline:
    """Cascade from event retrieval to per-window visual reasoning.

    Retrieval receives only ``event_description`` and always uses
    ``refine_raw=False`` through :class:`VisualQARetriever`. Provider calls are
    injected and cached by request/provider/image fingerprints.
    """

    def __init__(
        self,
        clip_encoder: Any = None,
        vector_store: Any = None,
        vlm_engine: Any = None,
        top_k_retrieve: int = 20,
        top_k_final: int = 100,
        *,
        retriever: VisualQARetriever | None = None,
        reasoner: QAReasoner | None = None,
        florence_reasoner: Any = None,
        image_resolver: ImageResolver | None = None,
        session_store: QASessionStore | None = None,
        competition_spec: Any = None,
        competition_specs: Mapping[str, QACompetitionSpec] | None = None,
        max_inline_image_bytes: int = 20 * 1024 * 1024,
        reasoning_workers: int = 2,
        retrieval_fingerprint: str | Callable[[tuple[str, ...]], str] | None = None,
    ) -> None:
        if not 1 <= top_k_retrieve <= 20:
            raise ValueError("top_k_retrieve must be between 1 and 20")
        if not 1 <= top_k_final <= 100:
            raise ValueError("top_k_final must be between 1 and 100")
        if max_inline_image_bytes < 1:
            raise ValueError("max_inline_image_bytes must be positive")
        if isinstance(reasoning_workers, bool) or not 1 <= reasoning_workers <= 2:
            raise ValueError("reasoning_workers must be one or two")
        self.clip_encoder = clip_encoder
        self.vector_store = vector_store
        self.vlm_engine = vlm_engine
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final
        self.retriever = retriever
        self.reasoner = reasoner or vlm_engine
        self.florence_reasoner = florence_reasoner
        self.image_resolver = image_resolver
        self.session_store = session_store
        numeric_spec = QACompetitionSpec.historical_numeric()
        general_spec = QACompetitionSpec.conservative_general_text()
        specs = {
            numeric_spec.spec_id: numeric_spec,
            general_spec.spec_id: general_spec,
        }
        if competition_specs is not None:
            specs.update(dict(competition_specs))
        if competition_spec is not None:
            spec_id = getattr(competition_spec, "spec_id", None)
            if not isinstance(spec_id, str) or not spec_id:
                raise ValueError("competition_spec must expose a non-empty spec_id")
            specs[spec_id] = competition_spec
            self.default_competition_spec_id = spec_id
        else:
            self.default_competition_spec_id = numeric_spec.spec_id
        if any(key != value.spec_id for key, value in specs.items()):
            raise ValueError("competition_specs keys must match each spec_id")
        self.competition_specs = specs
        # Compatibility for existing consumers that inspect this attribute.
        self.competition_spec = specs[self.default_competition_spec_id]
        self.max_inline_image_bytes = int(max_inline_image_bytes)
        self.reasoning_workers = int(reasoning_workers)
        if retrieval_fingerprint is not None and not (
            isinstance(retrieval_fingerprint, str) or callable(retrieval_fingerprint)
        ):
            raise TypeError("retrieval_fingerprint must be a string, callable, or None")
        self.retrieval_fingerprint = retrieval_fingerprint
        self._session_locks: dict[str, threading.RLock] = {}
        self._session_locks_guard = threading.Lock()
        self._cache_locks: dict[str, threading.RLock] = {}
        self._cache_locks_guard = threading.Lock()

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._session_locks_guard:
            return self._session_locks.setdefault(session_id, threading.RLock())

    def _cache_lock(self, cache_key: str) -> threading.RLock:
        with self._cache_locks_guard:
            return self._cache_locks.setdefault(cache_key, threading.RLock())

    @staticmethod
    def _request_payload(request: Any, question: str, event_description: str) -> dict[str, Any]:
        if isinstance(request, str):
            return {"question": question, "event_description": event_description}
        result = _json_value(request)
        # Session identity is persistence metadata, not semantic request input;
        # excluding it allows the 30-day reasoning cache to serve other sessions.
        result.pop("session_id", None)
        result.setdefault("question", question)
        result.setdefault("event_description", event_description)
        return result

    @staticmethod
    def _reasoning_result(value: Any, spec: QACompetitionSpec) -> dict[str, Any]:
        result = _json_value(value)
        answer = result.get("normalized_answer", result.get("answer"))
        answerable = result.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError("QA reasoner result must include boolean answerable")
        if not isinstance(answer, str) or (answerable and not answer.strip()):
            raise ValueError("answerable QA result must include a non-empty answer")
        evidence = result.get("evidence_frame_indices", ())
        if not isinstance(evidence, (list, tuple)) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 1 for index in evidence
        ):
            raise ValueError("evidence_frame_indices must contain positive integers")
        result["answer"] = spec.normalize_answer(answer) if answerable else ""
        result["answerable"] = answerable
        result["evidence_frame_indices"] = list(evidence)
        return result

    def _dependencies(self) -> tuple[VisualQARetriever, QAReasoner, ImageResolver, QASessionStore]:
        missing = [
            name
            for name, value in (
                ("retriever", self.retriever),
                ("reasoner", self.reasoner),
                ("image_resolver", self.image_resolver),
                ("session_store", self.session_store),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(f"Visual Q&A dependencies are unavailable: {', '.join(missing)}")
        return self.retriever, self.reasoner, self.image_resolver, self.session_store

    @staticmethod
    def _provider_fingerprint(reasoner: QAReasoner) -> str:
        value = getattr(reasoner, "fingerprint", None)
        if callable(value):
            value = value()
        if isinstance(value, str) and value:
            return value
        prompt_version = normalizer_version = response_schema_version = "unknown"
        try:
            from src.vlm.qa_reasoner import (
                NORMALIZER_VERSION,
                PROMPT_VERSION,
                RESPONSE_SCHEMA_VERSION,
            )

            prompt_version = PROMPT_VERSION
            normalizer_version = NORMALIZER_VERSION
            response_schema_version = RESPONSE_SCHEMA_VERSION
        except ImportError:
            pass
        safe_settings = {
            "type": f"{type(reasoner).__module__}.{type(reasoner).__qualname__}",
            "prompt_version": prompt_version,
            "normalizer_version": normalizer_version,
            "response_schema_version": response_schema_version,
        }
        for name in (
            "provider_name",
            "model_name",
            "revision",
            "device",
            "dtype",
            "max_new_tokens",
            "local_files_only",
            "_max_images",
            "_max_total_image_bytes",
            "_max_attempts",
        ):
            setting = getattr(reasoner, name, None)
            if isinstance(setting, (str, int, float, bool)) or setting is None:
                safe_settings[name] = setting
        return hashlib.sha256(
            json.dumps(safe_settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _provider_cache_identities(
        cls,
        reasoner: QAReasoner,
        request: QAQueryRequest | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Return provider name/fingerprint pairs in fallback execution order."""

        selector = getattr(reasoner, "providers_for", None)
        configured = selector(request) if request is not None and callable(selector) else None
        if configured is None:
            configured = getattr(reasoner, "providers", None)
        providers = tuple(configured) if isinstance(configured, (list, tuple)) else (reasoner,)
        identities: list[tuple[str, str]] = []
        for provider in providers:
            name = getattr(provider, "provider_name", None)
            if not isinstance(name, str) or not name:
                name = f"{type(provider).__module__}.{type(provider).__qualname__}"
            identities.append((name, cls._provider_fingerprint(provider)))
        return tuple(identities)

    @staticmethod
    def _actual_provider_cache_key(
        reasoning: Mapping[str, Any],
        identities: Sequence[tuple[str, str, str]],
    ) -> str:
        """Select the cache key belonging to the provider that actually answered."""

        actual = reasoning.get("provider")
        if isinstance(actual, str):
            actual_folded = actual.casefold()
            for name, _, cache_key in identities:
                if name.casefold() == actual_folded:
                    return cache_key
        if len(identities) == 1:
            return identities[0][2]
        # Provider chains are required to return their provider identity. A
        # mismatch must not poison an individual provider's cache namespace.
        raise ValueError("QA reasoner result provider does not match configured providers")

    @staticmethod
    def _reasoning_lease_ttl(
        reasoner: QAReasoner,
        request: QAQueryRequest | None = None,
    ) -> float:
        """Derive a bounded lease from ordered provider retry/timeout budgets."""

        selector = getattr(reasoner, "providers_for", None)
        configured = selector(request) if request is not None and callable(selector) else None
        if configured is None:
            configured = getattr(reasoner, "providers", None)
        providers = tuple(configured) if isinstance(configured, (list, tuple)) else (reasoner,)
        budget = 0.0
        for provider in providers:
            timeout = getattr(provider, "_timeout_seconds", None)
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
                timeout = 30.0
            attempts = getattr(provider, "_max_attempts", 1)
            if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
                attempts = 1
            retry_delay = getattr(provider, "_retry_max_seconds", 0.0)
            if (
                not isinstance(retry_delay, (int, float))
                or isinstance(retry_delay, bool)
                or retry_delay < 0
            ):
                retry_delay = 0.0
            budget += float(timeout) * attempts + float(retry_delay) * max(0, attempts - 1)
        return min(600.0, max(10.0, budget + 10.0))

    @staticmethod
    def _confidence_score(
        cluster: QACandidateCluster,
        source_frame_idx: int,
        reasoning: Mapping[str, Any],
    ) -> float:
        """Bounded deterministic proxy from retrieval and visual evidence."""

        rank_component = 1.0 / max(1, cluster.retrieval_rank)
        raw_score = max(0.0, float(cluster.retrieval_score))
        retrieval_component = raw_score / (1.0 + raw_score)
        evidence_indices = {
            int(index)
            for index in reasoning.get("evidence_frame_indices", ())
            if isinstance(index, int) and not isinstance(index, bool)
        }
        evidence_frames = {
            frame.source_frame_idx
            for index, frame in enumerate(cluster.frames, 1)
            if index in evidence_indices
        }
        evidence_component = 1.0 if source_frame_idx in evidence_frames else 0.0
        frame = next(
            value for value in cluster.frames if value.source_frame_idx == source_frame_idx
        )
        temporal_component = 1.0 - min(1.0, abs(frame.timestamp_ms - cluster.timestamp_ms) / 4_000)
        value = (
            0.40 * rank_component
            + 0.20 * retrieval_component
            + 0.25 * evidence_component
            + 0.15 * temporal_component
        )
        return max(0.0, min(1.0, value))

    @staticmethod
    def _frame_refs(cluster: QACandidateCluster) -> tuple[QAWindowFrameRef, ...]:
        return tuple(
            QAWindowFrameRef(
                position=position,
                batch_id=frame.batch_id,
                video_id=frame.video_id,
                source_frame_idx=frame.source_frame_idx,
                timestamp_ms=frame.timestamp_ms,
            )
            for position, frame in enumerate(cluster.frames, 1)
        )

    @classmethod
    def _safe_window(cls, cluster: QACandidateCluster) -> dict[str, Any]:
        """Serialize a candidate without media locations or source secrets."""

        return {
            "window_id": cluster.cluster_id,
            "retrieval_rank": cluster.retrieval_rank,
            "retrieval_score": cluster.retrieval_score,
            "batch_id": cluster.batch_id,
            "video_id": cluster.video_id,
            "source_frame_idx": cluster.source_frame_idx,
            "timestamp_ms": cluster.timestamp_ms,
            "shot_id": cluster.shot_id,
            "frames": [frame.to_dict() for frame in cls._frame_refs(cluster)],
            "frame_selection_scores": {
                str(source_frame_idx): score
                for source_frame_idx, score in cluster.frame_selection_scores.items()
            },
        }

    @staticmethod
    def _hydrate_windows(
        retriever: VisualQARetriever, windows: Sequence[Mapping[str, Any]]
    ) -> tuple[QACandidateCluster, ...] | None:
        """Resolve safe persisted references through the canonical repository."""

        repository = getattr(retriever, "frame_repository", None)
        fixture_clusters = {
            str(cluster.cluster_id): cluster
            for cluster in getattr(retriever, "clusters", ())
            if isinstance(cluster, QACandidateCluster)
        }
        hydrated: list[QACandidateCluster] = []
        for value in windows:
            window_id = str(value.get("window_id") or "")
            fixture = fixture_clusters.get(window_id)
            if fixture is not None:
                expected = [
                    int(frame.get("source_frame_idx", -1))
                    for frame in value.get("frames", ())
                    if isinstance(frame, Mapping)
                ]
                if expected == [frame.source_frame_idx for frame in fixture.frames]:
                    hydrated.append(fixture)
                    continue
            if repository is None:
                return None
            frames = []
            for frame_value in value.get("frames", ()):
                if not isinstance(frame_value, Mapping):
                    return None
                frame = repository.get_frame(
                    str(frame_value.get("batch_id") or value.get("batch_id") or ""),
                    str(frame_value.get("video_id") or value.get("video_id") or ""),
                    int(frame_value.get("source_frame_idx", -1)),
                )
                if frame is None or frame.timestamp_ms != int(frame_value.get("timestamp_ms", -1)):
                    return None
                frames.append(frame)
            try:
                hydrated.append(
                    QACandidateCluster(
                        cluster_id=window_id,
                        retrieval_rank=int(value["retrieval_rank"]),
                        retrieval_score=float(value["retrieval_score"]),
                        batch_id=str(value["batch_id"]),
                        video_id=str(value["video_id"]),
                        source_frame_idx=int(value["source_frame_idx"]),
                        timestamp_ms=int(value["timestamp_ms"]),
                        shot_id=str(value["shot_id"]),
                        frames=tuple(frames),
                        frame_selection_scores={
                            int(source_frame_idx): float(score)
                            for source_frame_idx, score in dict(
                                value.get("frame_selection_scores", {})
                            ).items()
                        },
                    )
                )
            except (KeyError, TypeError, ValueError):
                return None
        return tuple(hydrated)

    def _retrieval_cache_fingerprint(
        self, retriever: VisualQARetriever, filters: tuple[str, ...]
    ) -> str:
        configured = self.retrieval_fingerprint
        if callable(configured):
            configured = configured(tuple(sorted(set(filters))))
        if isinstance(configured, str) and configured:
            return configured
        exposed = getattr(retriever, "fingerprint", None)
        if callable(exposed):
            exposed = exposed()
        textual = getattr(retriever, "textual_pipeline", None)
        textual_fingerprint = getattr(textual, "fingerprint", None)
        if callable(textual_fingerprint):
            textual_fingerprint = textual_fingerprint()
        payload = {
            "type": f"{type(retriever).__module__}.{type(retriever).__qualname__}",
            "exposed": exposed if isinstance(exposed, str) else None,
            "textual": (textual_fingerprint if isinstance(textual_fingerprint, str) else None),
        }
        for name in (
            "retrieval_depth",
            "max_clusters",
            "radius_ms",
            "frames_per_cluster",
            "pseudo_shot_separation_ms",
        ):
            value = getattr(retriever, name, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[name] = value
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _candidate_cache_key(
        self,
        retriever: VisualQARetriever,
        event: str,
        filters: tuple[str, ...],
    ) -> str:
        payload = {
            "version": "qa-candidates-v1",
            "event_description": unicodedata.normalize("NFC", event.strip()),
            "batch_filters": sorted(set(filters)),
            "retrieval": self._retrieval_cache_fingerprint(retriever, filters),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _semantic_request_fingerprint(event: str, question: str) -> str:
        payload = {
            "version": "qa-reason-request-v2",
            "event_description": unicodedata.normalize("NFC", event.strip()),
            "question": unicodedata.normalize("NFC", question.strip()),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _probe(resolver: ImageResolver, frame: Any) -> CanonicalImageProbe | None:
        probe = getattr(resolver, "probe", None)
        if not callable(probe):
            return None
        value = probe(frame)
        if not isinstance(value, CanonicalImageProbe):
            raise TypeError("image resolver probe must return CanonicalImageProbe")
        return value

    @staticmethod
    def _failure(exc: Exception, stage: str) -> tuple[str, str, str]:
        if isinstance(exc, CanonicalImageTooLargeError):
            return "media_failed", "media_window_too_large", "media"
        if isinstance(exc, CanonicalImageError):
            return "media_failed", "media_unavailable", "media"
        try:
            from src.vlm.qa_reasoner import (
                QAProviderRequestError,
                QAProviderResponseError,
                QAProviderUnavailableError,
            )

            if isinstance(exc, QAProviderUnavailableError):
                return "provider_failed", "provider_unavailable", "provider"
            if isinstance(exc, QAProviderRequestError):
                return "provider_failed", "provider_request_failed", "provider"
            if isinstance(exc, QAProviderResponseError):
                return "provider_failed", "provider_response_invalid", "provider"
        except ImportError:
            pass
        if stage == "response":
            return "provider_failed", "provider_response_invalid", "provider"
        return "internal_failed", "internal_reasoning_error", stage

    @staticmethod
    def _provider_failure_metadata(exc: Exception) -> dict[str, Any]:
        """Copy only allowlisted metadata from a sanitized reasoner error."""

        try:
            from src.vlm.qa_reasoner import QAReasonerError
        except ImportError:
            return {}
        if not isinstance(exc, QAReasonerError):
            return {}
        metadata = exc.safe_metadata()
        category = metadata.get("category")
        status_code = metadata.get("status_code")
        return {
            "provider": metadata.get("provider"),
            "model_name": metadata.get("model"),
            "attempts": metadata.get("attempts"),
            "provider_error_category": (
                category if category in _PROVIDER_ERROR_CATEGORIES else None
            ),
            "provider_status_code": (
                status_code
                if isinstance(status_code, int)
                and not isinstance(status_code, bool)
                and 100 <= status_code <= 599
                else None
            ),
        }

    @classmethod
    def _outcome(
        cls,
        cluster: QACandidateCluster,
        status: str,
        **values: Any,
    ) -> dict[str, Any]:
        return QAWindowOutcome(
            window_id=cluster.cluster_id,
            batch_id=cluster.batch_id,
            video_id=cluster.video_id,
            source_frame_idx=cluster.source_frame_idx,
            timestamp_ms=cluster.timestamp_ms,
            shot_id=cluster.shot_id,
            retrieval_rank=cluster.retrieval_rank,
            retrieval_score=cluster.retrieval_score,
            frames=cls._frame_refs(cluster),
            status=status,
            **values,
        ).to_dict()

    def check_one_window_live(
        self,
        request: QAQueryRequest,
        *,
        reasoner: QAReasoner,
        batch_filters: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Reason over one real retrieved window without sessions or cache writes.

        This deliberately narrow public integration boundary is used only by
        the explicit live checker.  The supplied provider must expose a hard
        one-attempt cap, so one invocation cannot fan out into retries or a
        provider fallback chain.
        """

        if not isinstance(request, QAQueryRequest):
            raise TypeError("request must be a QAQueryRequest")
        if self.retriever is None or self.image_resolver is None:
            raise RuntimeError("Visual Q&A retrieval and image resolution are unavailable")
        if getattr(reasoner, "max_attempts", None) != 1:
            raise ValueError("live one-window reasoner must have max_attempts=1")
        filters = tuple(batch_filters or request.batch_filters)
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        clusters = self.retriever.retrieve(
            request.event_description,
            batch_filters=filters,
            max_clusters=1,
        )
        candidate_retrieval_ms = (time.perf_counter() - retrieval_started) * 1_000
        if not clusters:
            raise LookupError("Visual Q&A live check found no canonical candidate window")
        cluster = clusters[0]
        frame_started = time.perf_counter()
        resolved = tuple(self.image_resolver.resolve(frame) for frame in cluster.frames)
        frame_prepare_ms = (time.perf_counter() - frame_started) * 1_000
        if sum(len(image.data) for image in resolved) > self.max_inline_image_bytes:
            raise _ImageWindowTooLargeError(
                "canonical image window exceeds the configured inline byte limit"
            )
        images = tuple(_qa_image(image, index) for index, image in enumerate(resolved, 1))
        active_spec = self.competition_specs[request.answer_spec_id]
        inference_started = time.perf_counter()
        response = reasoner.reason(request, images, active_spec)
        ollama_inference_ms = (time.perf_counter() - inference_started) * 1_000
        reasoning = self._reasoning_result(response, active_spec)
        status = "answered" if reasoning["answerable"] else "unanswerable"
        outcome = self._outcome(
            cluster,
            status,
            answer=reasoning["answer"] if reasoning["answerable"] else "",
            raw_answer=str(reasoning.get("raw_answer") or ""),
            normalized_answer=reasoning["answer"] if reasoning["answerable"] else "",
            answerable=bool(reasoning["answerable"]),
            evidence_frame_indices=tuple(reasoning["evidence_frame_indices"]),
            provider=reasoning.get("provider"),
            model_name=reasoning.get("model_name"),
            prompt_version=reasoning.get("prompt_version"),
            frame_prepare_ms=frame_prepare_ms,
            provider_latency_ms=reasoning.get("latency_ms"),
            attempts=reasoning.get("attempts"),
            cache_status=reasoning.get("cache_status"),
            served_latency_ms=(time.perf_counter() - total_started) * 1_000,
            cache_layers={"candidate": "bypass", "reasoning": "bypass"},
        )
        outcome["stage_timings"] = {
            "candidate_retrieval_ms": round(candidate_retrieval_ms, 3),
            "frame_prepare_ms": round(frame_prepare_ms, 3),
            "ollama_inference_ms": round(ollama_inference_ms, 3),
            "total_ms": round((time.perf_counter() - total_started) * 1_000, 3),
        }
        metrics = getattr(reasoner, "last_metrics", None)
        outcome["ollama_metrics"] = dict(metrics) if isinstance(metrics, Mapping) else {}
        outcome["frame_selection_scores"] = {
            str(source_frame_idx): score
            for source_frame_idx, score in cluster.frame_selection_scores.items()
        }
        return outcome

    def retrieve_vqa_candidates(
        self,
        event_description: str,
        top_k: int = 5,
        *,
        batch_filters: Sequence[str] = (),
        query_language: str = "auto",
        event_description_en: str | None = None,
    ) -> tuple[QACandidateFrame, ...]:
        """Run pure KIS retrieval and return canonical frames without media/VLM work."""

        if self.retriever is None:
            raise RuntimeError("Visual Q&A retriever is unavailable")
        retrieve_frames = getattr(self.retriever, "retrieve_frames", None)
        if not callable(retrieve_frames):
            raise RuntimeError("Visual Q&A single-frame retrieval is unavailable")
        return tuple(
            retrieve_frames(
                event_description,
                batch_filters=tuple(batch_filters),
                top_k=top_k,
                query_language=query_language,
                event_description_en=event_description_en,
            )
        )

    @staticmethod
    def _candidate_cache_key_v3(
        retriever: VisualQARetriever,
        event_description: str,
        batch_filters: tuple[str, ...],
        top_k: int,
        retrieval_fingerprint: str,
        query_language: str,
        event_description_en: str | None,
    ) -> str:
        payload = {
            "version": "qa-single-frame-candidates-v3",
            "event_description": unicodedata.normalize("NFC", event_description.strip()),
            "batch_filters": sorted(set(batch_filters)),
            "top_k": top_k,
            "retrieval": retrieval_fingerprint,
            "retriever_type": f"{type(retriever).__module__}.{type(retriever).__qualname__}",
            "query_language": query_language,
            "event_description_en": event_description_en,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def start_candidate_search(
        self,
        request: QACandidateRequest,
        *,
        session_id: str | None = None,
        legacy_request: Mapping[str, Any] | None = None,
    ) -> tuple[QACandidateFrame, ...]:
        """Persist a v3 retrieval-only QA session and its immutable candidates."""

        if not isinstance(request, QACandidateRequest):
            raise TypeError("request must be a QACandidateRequest")
        if self.retriever is None or self.session_store is None:
            raise RuntimeError("Visual Q&A retrieval/session dependencies are unavailable")
        search_id = str(session_id or uuid.uuid4().hex)
        request_payload = request.to_dict()
        if legacy_request:
            request_payload["legacy_request"] = dict(legacy_request)
        session = self.session_store.create_session(
            search_id,
            request_payload,
            state={
                "state_version": 3,
                "status": "retrieving",
                "answers": {},
                "failures": {},
                "predictions": [],
                "channel_statuses": {},
                "retrieval_errors": {},
            },
        )
        existing = self.session_store.get_session_candidates(search_id)
        if existing:
            return tuple(QACandidateFrame.from_dict(value) for value in existing)

        started = time.perf_counter()
        filters = tuple(request.batch_filters)
        retrieval_fingerprint = self._retrieval_cache_fingerprint(self.retriever, filters)
        cache_key = self._candidate_cache_key_v3(
            self.retriever,
            request.event_description,
            filters,
            request.top_k,
            retrieval_fingerprint,
            request.query_language,
            request.event_description_en,
        )
        cached = self.session_store.get_candidate_entry(cache_key)
        candidate_values = cached[0] if cached is not None else None
        candidates: tuple[QACandidateFrame, ...] = ()
        if candidate_values is not None:
            try:
                candidates = tuple(QACandidateFrame.from_dict(value) for value in candidate_values)
            except (TypeError, ValueError):
                candidates = ()
        cache_status = "hit" if candidates else "miss"
        if not candidates:
            candidates = self.retrieve_vqa_candidates(
                request.event_description,
                request.top_k,
                batch_filters=filters,
                query_language=request.query_language,
                event_description_en=request.event_description_en,
            )
            textual = getattr(self.retriever, "textual_pipeline", None)
            raw_statuses = getattr(textual, "last_channel_statuses", {})
            raw_errors = getattr(textual, "last_channel_errors", {})
            diagnostics = {
                "channel_statuses": dict(raw_statuses) if isinstance(raw_statuses, Mapping) else {},
                "retrieval_errors": {
                    str(name): "channel_unavailable"
                    for name in raw_errors
                }
                if isinstance(raw_errors, Mapping)
                else {},
                "query_processing": dict(
                    getattr(textual, "last_query_processing", {})
                )
                if isinstance(getattr(textual, "last_query_processing", {}), Mapping)
                else {},
            }
            self.session_store.put_candidate_set(
                cache_key,
                tuple(candidate.to_dict() for candidate in candidates),
                diagnostics=diagnostics,
            )
        else:
            diagnostics = cached[1] if cached is not None else {}
        frozen = self.session_store.bind_session_candidates(
            search_id, (candidate.to_dict() for candidate in candidates)
        )
        candidates = tuple(QACandidateFrame.from_dict(value) for value in frozen)
        state = dict(session.state)
        state.update(
            {
                "state_version": 3,
                "status": "candidates_ready" if candidates else "no_candidates",
                "candidate_retrieval_ms": round(
                    (time.perf_counter() - started) * 1_000, 3
                ),
                "candidate_cache_status": cache_status,
                "channel_statuses": dict(diagnostics.get("channel_statuses", {})),
                "retrieval_errors": dict(diagnostics.get("retrieval_errors", {})),
                "query_processing": dict(diagnostics.get("query_processing", {})),
            }
        )
        self.session_store.update_state(
            search_id, state, expected_revision=session.revision
        )
        return candidates

    def answer_vqa_single_frame(
        self,
        selected_frame: FrameRecord,
        question: str,
        *,
        event_description: str = "selected event frame",
        answer_spec_id: str | None = None,
        reasoning_policy: str = "gemini_then_local",
        allow_external_frame_transfer: bool = False,
        query_id: str | None = None,
        reasoner: QAReasoner | None = None,
        florence_task: str | None = None,
        florence_target_label: str | None = None,
        _resolved_image: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Resolve and reason over exactly one canonical frame."""

        if not isinstance(selected_frame, FrameRecord):
            raise TypeError("selected_frame must be a FrameRecord")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if self.image_resolver is None:
            raise RuntimeError("Visual Q&A image resolver is unavailable")
        active_reasoner = reasoner or self.reasoner
        if active_reasoner is None:
            raise RuntimeError("Visual Q&A reasoner is unavailable")
        spec_id = answer_spec_id or self.default_competition_spec_id
        active_spec = self.competition_specs[spec_id]
        provider_request = QAQueryRequest(
            query_id=query_id or uuid.uuid4().hex,
            event_description=event_description,
            question=question,
            mode="interactive",
            answer_spec_id=spec_id,
            batch_filters=(selected_frame.batch_id,),
            limit=1,
            reasoning_policy=reasoning_policy,
            allow_external_frame_transfer=allow_external_frame_transfer,
        )
        frame_started = time.perf_counter()
        resolved = (
            _resolved_image
            if _resolved_image is not None
            else self.image_resolver.resolve(selected_frame)
        )
        frame_prepare_ms = (time.perf_counter() - frame_started) * 1_000
        if len(resolved.data) > self.max_inline_image_bytes:
            raise _ImageWindowTooLargeError("canonical image exceeds the inline-image limit")
        image = _qa_image(resolved, 1)
        inference_started = time.perf_counter()
        reason_task = getattr(active_reasoner, "reason_task", None)
        if florence_task is not None and callable(reason_task):
            response = reason_task(
                provider_request,
                (image,),
                active_spec,
                task=florence_task,
                target_label=florence_target_label,
            )
        else:
            response = active_reasoner.reason(provider_request, (image,), active_spec)
        answer_call_ms = (time.perf_counter() - inference_started) * 1_000
        reasoning = self._reasoning_result(response, active_spec)
        evidence = reasoning.get("evidence_frame_indices", ())
        if reasoning["answerable"] and tuple(evidence) != (1,):
            raise ValueError("single-frame reasoning evidence must be exactly [1]")
        provider_latency = reasoning.get("latency_ms")
        inference_ms = (
            float(provider_latency)
            if isinstance(provider_latency, (int, float))
            and not isinstance(provider_latency, bool)
            and float(provider_latency) >= 0
            else answer_call_ms
        )
        metrics = {
            "frame_prepare_ms": round(frame_prepare_ms, 3),
            "vlm_inference_ms": round(inference_ms, 3),
            "answer_call_ms": round(answer_call_ms, 3),
        }
        provider_metrics = getattr(active_reasoner, "last_metrics", None)
        if isinstance(provider_metrics, Mapping):
            handoff = provider_metrics.get("handoff_ms")
            if isinstance(handoff, (int, float)) and not isinstance(handoff, bool):
                metrics["gpu_handoff_ms"] = round(max(0.0, float(handoff)), 3)
        return reasoning, metrics

    def answer_candidate(
        self,
        search_id: str,
        request: QASingleFrameAnswerRequest,
    ) -> dict[str, Any]:
        """Answer one persisted v3 candidate and update the restorable session."""

        if self.session_store is None or self.retriever is None:
            raise RuntimeError("Visual Q&A session dependencies are unavailable")
        snapshot = self.session_store.get_snapshot(search_id)
        if snapshot is None:
            raise KeyError(search_id)
        if int(snapshot.get("state_version", 0)) != 3:
            raise RuntimeError("legacy_qa_session_read_only")
        candidates = tuple(
            QACandidateFrame.from_dict(value)
            for value in snapshot.get("session_candidates", ())
        )
        candidate = next(
            (value for value in candidates if value.candidate_id == request.candidate_id),
            None,
        )
        if candidate is None:
            raise ValueError("candidate_id is not part of this QA session")
        repository = getattr(self.retriever, "frame_repository", None)
        frame = repository.get_frame(
            candidate.batch_id, candidate.video_id, candidate.source_frame_idx
        ) if repository is not None else None
        if frame is None or frame.timestamp_ms != candidate.timestamp_ms:
            raise ValueError("candidate frame is no longer canonical")
        stored_request = snapshot.get("request", {})
        event_description = str(stored_request.get("event_description") or "")
        active_reasoner = (
            self.florence_reasoner
            if request.answer_engine == "florence2"
            else self.reasoner
        )
        if active_reasoner is None:
            raise RuntimeError(
                "florence_not_provisioned"
                if request.answer_engine == "florence2"
                else "qa_reasoner_unavailable"
            )
        resolved = self.image_resolver.resolve(frame) if self.image_resolver is not None else None
        if resolved is None:
            raise RuntimeError("media_unavailable")
        request_payload = {
            "version": "qa-single-frame-answer-v3",
            "event_description": unicodedata.normalize("NFC", event_description.strip()),
            "question": unicodedata.normalize("NFC", request.question.strip()),
            "answer_spec_id": request.answer_spec_id,
            "reasoning_policy": request.reasoning_policy,
            "answer_engine": request.answer_engine,
            "florence_task": request.florence_task,
            "florence_target_label": request.florence_target_label,
        }
        request_fingerprint = hashlib.sha256(
            json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        frame_fingerprint = {
            "batch_id": frame.batch_id,
            "video_id": frame.video_id,
            "source_frame_idx": frame.source_frame_idx,
            "sha256": resolved.content_sha256,
        }
        provider_request = QAQueryRequest(
            query_id=search_id,
            event_description=event_description,
            question=request.question,
            mode="interactive",
            answer_spec_id=request.answer_spec_id,
            batch_filters=(frame.batch_id,),
            limit=1,
            reasoning_policy=request.reasoning_policy,
            allow_external_frame_transfer=request.allow_external_frame_transfer,
        )
        identities = tuple(
            (
                provider_name,
                provider_fingerprint,
                reasoning_cache_key(
                    provider_fingerprint=provider_fingerprint,
                    request_fingerprint=request_fingerprint,
                    answer_spec_id=request.answer_spec_id,
                    frame_fingerprints=(frame_fingerprint,),
                ),
            )
            for provider_name, provider_fingerprint in self._provider_cache_identities(
                active_reasoner, provider_request
            )
        )
        cache_key = identities[0][2]
        reasoning = self.session_store.get_reasoning(cache_key)
        timings = {"frame_prepare_ms": 0.0, "vlm_inference_ms": 0.0}
        cache_status = "hit" if reasoning is not None else "miss"
        try:
            if reasoning is None:
                reasoning, timings = self.answer_vqa_single_frame(
                    frame,
                    request.question,
                    event_description=event_description,
                    answer_spec_id=request.answer_spec_id,
                    reasoning_policy=request.reasoning_policy,
                    allow_external_frame_transfer=request.allow_external_frame_transfer,
                    query_id=search_id,
                    reasoner=active_reasoner,
                    florence_task=request.florence_task,
                    florence_target_label=request.florence_target_label,
                    _resolved_image=resolved,
                )
                actual_cache_key = self._actual_provider_cache_key(reasoning, identities)
                if not bool(reasoning.get("fallback_used", False)):
                    self.session_store.put_reasoning(actual_cache_key, reasoning)
            reasoning = dict(reasoning)
            reasoning["cache_status"] = cache_status
            prediction = None
            if reasoning.get("answerable"):
                prediction = QAPrediction(
                    rank=1,
                    batch_id=frame.batch_id,
                    video_id=frame.video_id,
                    source_frame_idx=frame.source_frame_idx,
                    timestamp_ms=frame.timestamp_ms,
                    answer=str(reasoning["answer"]),
                    raw_answer=str(reasoning.get("raw_answer") or reasoning["answer"]),
                    model_name=reasoning.get("model_name"),
                    confidence_score=max(
                        0.0,
                        min(1.0, 1.0 / max(1, candidate.retrieval_rank)),
                    ),
                    answer_source="/".join(
                        value
                        for value in (
                            reasoning.get("provider"),
                            reasoning.get("model_name"),
                        )
                        if isinstance(value, str) and value
                    )
                    or "qa_reasoner",
                    window_id=candidate.candidate_id,
                    evidence={
                        "candidate_id": candidate.candidate_id,
                        "retrieval_rank": candidate.retrieval_rank,
                        "retrieval_score": candidate.retrieval_score,
                        "evidence_frame_indices": [1],
                        "evidence_source_frame_indices": [frame.source_frame_idx],
                        "provider": reasoning.get("provider"),
                        "requested_policy": reasoning.get("requested_policy"),
                        "fallback_used": bool(reasoning.get("fallback_used", False)),
                        "fallback_reason_code": reasoning.get("fallback_reason_code"),
                        "latency_ms": reasoning.get("latency_ms"),
                        "answer_call_ms": timings.get("answer_call_ms"),
                        "cache_status": cache_status,
                    },
                ).to_dict()
            session = self.session_store.get_session(search_id)
            if session is None:
                raise QASessionStoreError("QA session expired during answering")
            state = dict(session.state)
            answers = dict(state.get("answers", {}))
            failures = dict(state.get("failures", {}))
            failures.pop(candidate.candidate_id, None)
            answers[candidate.candidate_id] = {
                "candidate_id": candidate.candidate_id,
                "status": "answered" if prediction is not None else "unanswerable",
                "reasoning": reasoning,
                "prediction": prediction,
                "stage_timings": timings,
            }
            predictions = [
                value.get("prediction")
                for _, value in sorted(
                    answers.items(),
                    key=lambda item: next(
                        candidate.retrieval_rank
                        for candidate in candidates
                        if candidate.candidate_id == item[0]
                    ),
                )
                if isinstance(value, Mapping) and value.get("prediction") is not None
            ]
            for rank, value in enumerate(predictions, 1):
                value["rank"] = rank
            state.update(
                {
                    "state_version": 3,
                    "status": "answered" if predictions else "completed_no_answer",
                    "selected_candidate_id": candidate.candidate_id,
                    "answers": answers,
                    "failures": failures,
                    "predictions": predictions,
                }
            )
            self.session_store.update_state(
                search_id, state, expected_revision=session.revision
            )
            return answers[candidate.candidate_id]
        except Exception as error:
            session = self.session_store.get_session(search_id)
            if session is not None:
                state = dict(session.state)
                failures = dict(state.get("failures", {}))
                failures[candidate.candidate_id] = self._failure(error, "provider")[1]
                state.update(
                    {
                        "state_version": 3,
                        "status": "failed",
                        "selected_candidate_id": candidate.candidate_id,
                        "failures": failures,
                    }
                )
                self.session_store.update_state(
                    search_id, state, expected_revision=session.revision
                )
            raise

    def answer(
        self,
        request: QAQueryRequest | Any,
        *,
        event_description: str | None = None,
        session_id: str | None = None,
        batch_filters: Sequence[str] | None = None,
        max_clusters: int | None = None,
    ) -> list[dict[str, Any]]:
        """Serialize a session cascade so concurrent expansions remain idempotent."""

        effective_session_id = (
            session_id
            or _request_value(request, "session_id")
            or _request_value(request, "query_id")
            or uuid.uuid4().hex
        )
        with self._session_lock(str(effective_session_id)):
            for attempt in range(10):
                try:
                    return self._answer_locked(
                        request,
                        event_description=event_description,
                        session_id=str(effective_session_id),
                        batch_filters=batch_filters,
                        max_clusters=max_clusters,
                    )
                except QASessionStoreError as exc:
                    if "concurrently modified" not in str(exc) or attempt == 9:
                        raise
                    time.sleep(0.05 * (attempt + 1))
            raise AssertionError("unreachable QA session retry state")

    def _answer_locked(
        self,
        request: QAQueryRequest | Any,
        *,
        event_description: str | None = None,
        session_id: str | None = None,
        batch_filters: Sequence[str] | None = None,
        max_clusters: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run or resume a Visual Q&A cascade and return unique ranked answers."""

        retriever, reasoner, resolver, sessions = self._dependencies()
        candidate_started = time.perf_counter()
        question = request if isinstance(request, str) else _request_value(request, "question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Visual Q&A request requires a non-empty question")
        event = event_description or _request_value(request, "event_description")
        if not isinstance(event, str) or not event.strip():
            raise ValueError("Visual Q&A request requires a non-empty event_description")
        filters_value = (
            batch_filters
            if batch_filters is not None
            else _request_value(request, "batch_filters", ())
        )
        filters = tuple(str(value) for value in (filters_value or ()))
        answer_spec_id = _request_value(request, "answer_spec_id", self.default_competition_spec_id)
        if answer_spec_id not in self.competition_specs:
            raise ValueError(f"unknown Visual Q&A answer_spec_id: {answer_spec_id!r}")
        active_spec = self.competition_specs[answer_spec_id]
        session_value = (
            session_id
            or _request_value(request, "session_id")
            or _request_value(request, "query_id")
            or uuid.uuid4().hex
        )
        payload = self._request_payload(request, question.strip(), event.strip())
        if isinstance(request, QAQueryRequest):
            provider_request = request
        elif isinstance(request, str):
            legacy_id = hashlib.sha256(
                f"{event.strip()}\0{question.strip()}".encode("utf-8")
            ).hexdigest()[:16]
            provider_request = QAQueryRequest(
                query_id=f"legacy-{legacy_id}",
                event_description=event.strip(),
                question=question.strip(),
                batch_filters=filters,
                limit=self.top_k_final,
                reasoning_policy="local_only",
            )
            payload = provider_request.to_dict()
        else:
            provider_payload = dict(payload)
            provider_payload.setdefault("query_id", str(session_value))
            provider_payload.setdefault("answer_spec_id", active_spec.spec_id)
            provider_request = QAQueryRequest.from_dict(provider_payload)
            payload = provider_request.to_dict()
        session = sessions.create_session(
            str(session_value),
            payload,
            state={
                "state_version": 2,
                "results": {},
                "failures": {},
                "predictions": [],
                "window_outcomes": {},
            },
        )
        state = dict(session.state)
        state["state_version"] = 2
        stored_results = dict(state.get("results", {}))
        failures = {
            str(key): (str(value) if str(value) in _FAILURE_CODES else "internal_reasoning_error")
            for key, value in dict(state.get("failures", {})).items()
        }
        # ``get_snapshot`` performs a read-only legacy projection.  Seeding its
        # terminal outcomes here prevents an expansion from retrying old
        # answered, unanswerable, or failed windows merely because the original
        # session predates state v2.
        restored = sessions.get_snapshot(str(session_value))
        restored_outcomes = restored.get("window_outcomes", ()) if restored else ()
        outcomes = {
            str(value["window_id"]): dict(value)
            for value in restored_outcomes
            if isinstance(value, Mapping) and value.get("window_id")
        }

        frozen_windows = sessions.get_session_windows(str(session_value))
        candidate_cache_status = "session"
        clusters = self._hydrate_windows(retriever, frozen_windows) if frozen_windows else None
        if clusters is None:
            candidate_key = self._candidate_cache_key(retriever, event.strip(), filters)
            with self._cache_lock(f"candidate:{candidate_key}"):
                candidate_entry = sessions.get_candidate_entry(candidate_key)
                cached_windows = candidate_entry[0] if candidate_entry is not None else None
                clusters = (
                    self._hydrate_windows(retriever, cached_windows)
                    if cached_windows is not None
                    else None
                )
                if clusters is not None:
                    candidate_cache_status = "hit"
                    frozen_windows = cached_windows or ()
                    cached_diagnostics = candidate_entry[1] if candidate_entry else {}
                    if isinstance(cached_diagnostics.get("channel_statuses"), Mapping):
                        state["channel_statuses"] = dict(cached_diagnostics["channel_statuses"])
                    if isinstance(cached_diagnostics.get("retrieval_errors"), Mapping):
                        state["retrieval_errors"] = dict(cached_diagnostics["retrieval_errors"])
                else:
                    owner_token = uuid.uuid4().hex
                    lease_key = f"candidate:{candidate_key}"
                    acquired = sessions.acquire_cache_lease(
                        lease_key, owner_token, ttl_seconds=30.0
                    )
                    if not acquired:
                        deadline = time.monotonic() + 30.0
                        while time.monotonic() < deadline:
                            candidate_entry = sessions.get_candidate_entry(candidate_key)
                            cached_windows = (
                                candidate_entry[0] if candidate_entry is not None else None
                            )
                            clusters = (
                                self._hydrate_windows(retriever, cached_windows)
                                if cached_windows is not None
                                else None
                            )
                            if clusters is not None:
                                frozen_windows = cached_windows or ()
                                candidate_cache_status = "hit"
                                cached_diagnostics = candidate_entry[1] if candidate_entry else {}
                                if isinstance(cached_diagnostics.get("channel_statuses"), Mapping):
                                    state["channel_statuses"] = dict(
                                        cached_diagnostics["channel_statuses"]
                                    )
                                if isinstance(cached_diagnostics.get("retrieval_errors"), Mapping):
                                    state["retrieval_errors"] = dict(
                                        cached_diagnostics["retrieval_errors"]
                                    )
                                break
                            time.sleep(0.05)
                        if clusters is None:
                            acquired = sessions.acquire_cache_lease(
                                lease_key, owner_token, ttl_seconds=30.0
                            )
                    if clusters is None:
                        if not acquired:
                            raise RuntimeError("candidate cache lease unavailable")
                        try:
                            requested_retrieval = min(20, int(retriever.max_clusters))
                            clusters = retriever.retrieve(
                                event.strip(),
                                batch_filters=filters,
                                max_clusters=requested_retrieval,
                            )
                            textual = getattr(retriever, "textual_pipeline", None)
                            raw_statuses = getattr(textual, "last_channel_statuses", {})
                            if isinstance(raw_statuses, Mapping):
                                state["channel_statuses"] = {
                                    str(name): {
                                        str(key): value
                                        for key, value in dict(status).items()
                                        if key in {"state", "reason_code"}
                                        and isinstance(value, (str, int, float, bool, type(None)))
                                    }
                                    for name, status in raw_statuses.items()
                                    if isinstance(status, Mapping)
                                }
                            raw_errors = getattr(textual, "last_channel_errors", {})
                            if isinstance(raw_errors, Mapping):
                                state["retrieval_errors"] = {
                                    str(name): "channel_unavailable" for name in raw_errors
                                }
                            frozen_windows = tuple(
                                self._safe_window(cluster) for cluster in clusters
                            )
                            sessions.put_candidate_set(
                                candidate_key,
                                frozen_windows,
                                diagnostics={
                                    "channel_statuses": dict(state.get("channel_statuses", {})),
                                    "retrieval_errors": dict(state.get("retrieval_errors", {})),
                                },
                            )
                            candidate_cache_status = "miss"
                        finally:
                            sessions.release_cache_lease(lease_key, owner_token)
            frozen_windows = sessions.bind_session_windows(str(session_value), frozen_windows)
            bound_clusters = self._hydrate_windows(retriever, frozen_windows)
            if bound_clusters is not None:
                clusters = bound_clusters
        clusters = tuple(clusters or ())
        state["candidate_retrieval_ms"] = round(
            (time.perf_counter() - candidate_started) * 1_000,
            3,
        )

        prior_requested = int(state.get("requested_target_clusters", 0))
        requested_target = int(max_clusters if max_clusters is not None else self.top_k_retrieve)
        requested_target = min(20, max(prior_requested, requested_target))
        active_clusters = clusters[:requested_target]
        targets = [
            {
                "batch_id": cluster.batch_id,
                "video_id": cluster.video_id,
                "source_frame_idx": cluster.source_frame_idx,
                "timestamp_ms": cluster.timestamp_ms,
                "shot_id": cluster.shot_id,
                "window_id": cluster.cluster_id,
            }
            for cluster in active_clusters
        ]
        sessions.expand_targets(str(session_value), targets)
        request_fingerprint = self._semantic_request_fingerprint(event.strip(), question.strip())
        provider_identities = self._provider_cache_identities(reasoner, provider_request)
        chain_fingerprint = self._provider_fingerprint(reasoner)
        reasoning_lease_ttl = self._reasoning_lease_ttl(reasoner, provider_request)
        state["requested_target_clusters"] = requested_target
        state["retrieved_clusters"] = len(clusters)
        state["retrieval_exhausted"] = len(clusters) < requested_target
        state["session_windows"] = list(frozen_windows)

        pending_clusters: list[QACandidateCluster] = []
        for cluster in active_clusters:
            target_key = f"{cluster.batch_id}:{cluster.video_id}:{cluster.source_frame_idx}"
            outcome = outcomes.get(cluster.cluster_id, {})
            if outcome.get("status") not in _OUTCOME_STATUSES - {"pending"}:
                outcomes[cluster.cluster_id] = self._outcome(
                    cluster,
                    "pending",
                    cache_layers={"candidate": candidate_cache_status},
                )
                pending_clusters.append(cluster)

        # Persist safe frame references before any media or provider operation.
        state["window_outcomes"] = outcomes
        state["results"] = stored_results
        state["failures"] = failures
        session = sessions.update_state(
            str(session_value), state, expected_revision=session.revision
        )

        def reason_cluster(
            cluster: QACandidateCluster,
        ) -> tuple[str, dict[str, Any], dict[str, Any] | None, str | None]:
            target_key = f"{cluster.batch_id}:{cluster.video_id}:{cluster.source_frame_idx}"
            served_started = time.perf_counter()
            frame_prepare_ms = 0.0
            stage = "media"
            cache_layers = {
                "candidate": candidate_cache_status,
                "image_fingerprint": "miss",
                "reasoning": "miss",
            }
            window_lease_key: str | None = None
            window_owner_token: str | None = None
            window_lease_acquired = False

            def provider_cache_keys(
                frame_fingerprints: Sequence[Mapping[str, Any]],
            ) -> tuple[tuple[str, str, str], ...]:
                return tuple(
                    (
                        provider_name,
                        provider_fingerprint,
                        reasoning_cache_key(
                            provider_fingerprint=provider_fingerprint,
                            request_fingerprint=request_fingerprint,
                            answer_spec_id=active_spec.spec_id,
                            frame_fingerprints=frame_fingerprints,
                        ),
                    )
                    for provider_name, provider_fingerprint in provider_identities
                )

            def first_cached_reasoning(
                identities: Sequence[tuple[str, str, str]],
            ) -> tuple[dict[str, Any] | None, str | None]:
                cached = sessions.get_reasoning_many(cache_key for _, _, cache_key in identities)
                for _, _, candidate_key in identities:
                    value = cached.get(candidate_key)
                    if value is not None:
                        return value, candidate_key
                return None, None

            try:
                probes = tuple(self._probe(resolver, frame) for frame in cluster.frames)
                typed_probes: tuple[CanonicalImageProbe, ...] = ()
                fingerprint_values: list[dict[str, Any]] = []
                reasoning: dict[str, Any] | None = None
                cache_key: str | None = None
                if all(probe is not None for probe in probes):
                    typed_probes = tuple(
                        probe for probe in probes if isinstance(probe, CanonicalImageProbe)
                    )
                    persisted = sessions.get_image_fingerprints(
                        probe.cache_key for probe in typed_probes
                    )
                    for probe in typed_probes:
                        value = persisted.get(probe.cache_key)
                        if value is None:
                            fingerprint_values = []
                            break
                        fingerprint_values.append(
                            {
                                "batch_id": value.get("batch_id"),
                                "video_id": value.get("video_id"),
                                "source_frame_idx": value.get("source_frame_idx"),
                                "sha256": value.get("sha256"),
                            }
                        )
                    if len(fingerprint_values) == len(cluster.frames):
                        cache_layers["image_fingerprint"] = "hit"
                        cache_identities = provider_cache_keys(fingerprint_values)
                        reasoning, cache_key = first_cached_reasoning(cache_identities)
                        if reasoning is not None:
                            cache_layers["reasoning"] = "hit"

                if reasoning is None and len(typed_probes) == len(cluster.frames):
                    window_payload = {
                        "version": "qa-window-work-v1",
                        "provider_chain": chain_fingerprint,
                        "request": request_fingerprint,
                        "answer_spec_id": active_spec.spec_id,
                        "probes": [probe.cache_key for probe in typed_probes],
                    }
                    window_digest = hashlib.sha256(
                        json.dumps(window_payload, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    window_lease_key = f"window-work:{window_digest}"
                    window_owner_token = uuid.uuid4().hex
                    window_lease_acquired = sessions.acquire_cache_lease(
                        window_lease_key,
                        window_owner_token,
                        ttl_seconds=reasoning_lease_ttl,
                    )
                    if window_lease_acquired:
                        cache_layers["window_work"] = "owner"
                    if not window_lease_acquired:
                        deadline = time.monotonic() + reasoning_lease_ttl
                        while time.monotonic() < deadline:
                            persisted = sessions.get_image_fingerprints(
                                probe.cache_key for probe in typed_probes
                            )
                            fingerprint_values = []
                            for probe in typed_probes:
                                value = persisted.get(probe.cache_key)
                                if value is None:
                                    fingerprint_values = []
                                    break
                                fingerprint_values.append(
                                    {
                                        "batch_id": value.get("batch_id"),
                                        "video_id": value.get("video_id"),
                                        "source_frame_idx": value.get("source_frame_idx"),
                                        "sha256": value.get("sha256"),
                                    }
                                )
                            if len(fingerprint_values) == len(cluster.frames):
                                cache_layers["image_fingerprint"] = "hit"
                                cache_identities = provider_cache_keys(fingerprint_values)
                                reasoning, cache_key = first_cached_reasoning(cache_identities)
                                if reasoning is not None:
                                    cache_layers["reasoning"] = "hit"
                                    cache_layers["window_work"] = "wait_hit"
                                    break
                            window_lease_acquired = sessions.acquire_cache_lease(
                                window_lease_key,
                                window_owner_token,
                                ttl_seconds=reasoning_lease_ttl,
                            )
                            if window_lease_acquired:
                                cache_layers["window_work"] = "takeover"
                                break
                            time.sleep(0.05)
                        if reasoning is None and not window_lease_acquired:
                            raise RuntimeError("window work cache lease unavailable")

                if reasoning is None:
                    frame_prepare_started = time.perf_counter()
                    resolved = tuple(resolver.resolve(frame) for frame in cluster.frames)
                    frame_prepare_ms = (time.perf_counter() - frame_prepare_started) * 1_000
                    if sum(len(image.data) for image in resolved) > self.max_inline_image_bytes:
                        raise _ImageWindowTooLargeError(
                            "QA image window exceeds aggregate inline-image limit"
                        )
                    fingerprint_values = []
                    fingerprints_to_store: dict[str, dict[str, Any]] = {}
                    for probe, image in zip(probes, resolved, strict=True):
                        fingerprint = {
                            "batch_id": image.batch_id,
                            "video_id": image.video_id,
                            "source_frame_idx": image.source_frame_idx,
                            "sha256": image.content_sha256,
                        }
                        fingerprint_values.append(fingerprint)
                        if isinstance(probe, CanonicalImageProbe):
                            fingerprints_to_store[probe.cache_key] = {
                                **fingerprint,
                                "source_revision": probe.source_revision,
                                "resolver_fingerprint": probe.resolver_fingerprint,
                                "raw_sha256": image.raw_sha256,
                                "mime_type": image.mime_type,
                                "width": image.width,
                                "height": image.height,
                            }
                    sessions.put_image_fingerprints(fingerprints_to_store)
                    cache_identities = provider_cache_keys(fingerprint_values)
                    lease_cache_key = reasoning_cache_key(
                        provider_fingerprint=chain_fingerprint,
                        request_fingerprint=request_fingerprint,
                        answer_spec_id=active_spec.spec_id,
                        frame_fingerprints=fingerprint_values,
                    )
                    with self._cache_lock(f"reasoning:{lease_cache_key}"):
                        reasoning, cache_key = first_cached_reasoning(cache_identities)
                        if reasoning is not None:
                            cache_layers["reasoning"] = "hit"
                        else:
                            images = tuple(
                                _qa_image(image, index) for index, image in enumerate(resolved, 1)
                            )
                            owner_token: str | None = None
                            acquired = window_lease_acquired
                            if not acquired:
                                owner_token = uuid.uuid4().hex
                                acquired = sessions.acquire_cache_lease(
                                    f"reasoning:{lease_cache_key}",
                                    owner_token,
                                    ttl_seconds=reasoning_lease_ttl,
                                )
                            if not acquired:
                                deadline = time.monotonic() + reasoning_lease_ttl
                                while time.monotonic() < deadline:
                                    reasoning, cache_key = first_cached_reasoning(cache_identities)
                                    if reasoning is not None:
                                        cache_layers["reasoning"] = "hit"
                                        break
                                    time.sleep(0.05)
                                if reasoning is None:
                                    acquired = sessions.acquire_cache_lease(
                                        f"reasoning:{lease_cache_key}",
                                        str(owner_token),
                                        ttl_seconds=reasoning_lease_ttl,
                                    )
                            if reasoning is None:
                                if not acquired:
                                    raise RuntimeError("reasoning cache lease unavailable")
                                try:
                                    # The previous owner may commit and release
                                    # between our pre-lease cache read and this
                                    # successful acquisition. Recheck while we
                                    # own the lease before invoking a provider.
                                    reasoning, cache_key = first_cached_reasoning(
                                        cache_identities
                                    )
                                    if reasoning is not None:
                                        cache_layers["reasoning"] = "hit"
                                    else:
                                        stage = "provider"
                                        response = reasoner.reason(
                                            provider_request, images, active_spec
                                        )
                                        stage = "response"
                                        reasoning = self._reasoning_result(
                                            response, active_spec
                                        )
                                        cache_key = self._actual_provider_cache_key(
                                            reasoning, cache_identities
                                        )
                                        sessions.put_reasoning(cache_key, reasoning)
                                finally:
                                    if owner_token is not None:
                                        sessions.release_cache_lease(
                                            f"reasoning:{lease_cache_key}", owner_token
                                        )
                reasoning = dict(reasoning)
                if cache_layers["reasoning"] == "hit":
                    reasoning["cache_status"] = "hit"
                status = "answered" if reasoning["answerable"] else "unanswerable"
                served_latency_ms = (time.perf_counter() - served_started) * 1_000
                outcome = self._outcome(
                    cluster,
                    status,
                    answer=reasoning["answer"] if reasoning["answerable"] else "",
                    raw_answer=str(reasoning.get("raw_answer") or ""),
                    normalized_answer=(reasoning["answer"] if reasoning["answerable"] else ""),
                    answerable=bool(reasoning["answerable"]),
                    evidence_frame_indices=tuple(reasoning["evidence_frame_indices"]),
                    provider=reasoning.get("provider"),
                    model_name=reasoning.get("model_name"),
                    prompt_version=reasoning.get("prompt_version"),
                    frame_prepare_ms=frame_prepare_ms,
                    provider_latency_ms=reasoning.get("latency_ms"),
                    attempts=reasoning.get("attempts"),
                    cache_status=reasoning.get("cache_status"),
                    served_latency_ms=served_latency_ms,
                    cache_layers=cache_layers,
                )
                return (
                    target_key,
                    outcome,
                    {
                        "answer": reasoning["answer"],
                        "reasoning": reasoning,
                    },
                    None,
                )
            except Exception as exc:
                status, code, failure_stage = self._failure(exc, stage)
                provider_metadata = self._provider_failure_metadata(exc)
                outcome = self._outcome(
                    cluster,
                    status,
                    frame_prepare_ms=frame_prepare_ms,
                    served_latency_ms=(time.perf_counter() - served_started) * 1_000,
                    cache_layers=cache_layers,
                    failure_code=code,
                    failure_stage=failure_stage,
                    **provider_metadata,
                )
                return target_key, outcome, None, code
            finally:
                if window_lease_acquired and window_lease_key and window_owner_token:
                    sessions.release_cache_lease(window_lease_key, window_owner_token)

        if pending_clusters:
            worker_count = min(self.reasoning_workers, len(pending_clusters))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for target_key, outcome, result, failure_code in executor.map(
                    reason_cluster, pending_clusters
                ):
                    outcomes[str(outcome["window_id"])] = outcome
                    if result is None:
                        failures[target_key] = str(failure_code)
                    else:
                        stored_results[target_key] = result
                        failures.pop(target_key, None)

        # Provider lifetime is coordinated at the API workflow boundary.  In
        # particular Ollama remains warm across progressive QA expansions and
        # is explicitly released before a dense KIS/TRAKE workflow takes over.

        reasoned_clusters = [
            cluster
            for cluster in active_clusters
            if (
                stored_results.get(
                    f"{cluster.batch_id}:{cluster.video_id}:{cluster.source_frame_idx}", {}
                )
                .get("reasoning", {})
                .get("answerable")
            )
        ]
        frame_candidates: list[tuple[QACandidateCluster, Any]] = []
        # Put every representative first, preserving the video round-robin
        # order established by VisualQARetriever.
        for cluster in reasoned_clusters:
            representative = next(
                frame
                for frame in cluster.frames
                if frame.source_frame_idx == cluster.source_frame_idx
            )
            frame_candidates.append((cluster, representative))
        neighbor_queues = {
            cluster.cluster_id: [
                frame
                for frame in sorted(
                    cluster.frames,
                    key=lambda value: (
                        abs(value.timestamp_ms - cluster.timestamp_ms),
                        value.timestamp_ms,
                        value.source_frame_idx,
                    ),
                )
                if frame.source_frame_idx != cluster.source_frame_idx
            ]
            for cluster in reasoned_clusters
        }
        while any(neighbor_queues.values()):
            for cluster in reasoned_clusters:
                queue = neighbor_queues[cluster.cluster_id]
                if queue:
                    frame_candidates.append((cluster, queue.pop(0)))

        requested_limit = int(_request_value(request, "limit", self.top_k_final))
        requested_limit = min(requested_limit, self.top_k_final, 100)
        ordered: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for cluster, frame in frame_candidates:
            key = (cluster.video_id, frame.source_frame_idx)
            target_key = f"{cluster.batch_id}:{cluster.video_id}:{cluster.source_frame_idx}"
            result = stored_results.get(target_key)
            if result is None or key in seen:
                continue
            reasoning = result["reasoning"]
            confidence_score = self._confidence_score(cluster, frame.source_frame_idx, reasoning)
            answer_source = (
                "/".join(
                    value
                    for value in (reasoning.get("provider"), reasoning.get("model_name"))
                    if isinstance(value, str) and value
                )
                or "qa_reasoner"
            )
            evidence_positions = {
                int(value)
                for value in reasoning.get("evidence_frame_indices", ())
                if isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= len(cluster.frames)
            }
            window_frames = [
                {
                    "position": position,
                    "source_frame_idx": window_frame.source_frame_idx,
                    "timestamp_ms": window_frame.timestamp_ms,
                }
                for position, window_frame in enumerate(cluster.frames, 1)
            ]
            output = QAPrediction(
                rank=len(ordered) + 1,
                batch_id=cluster.batch_id,
                video_id=cluster.video_id,
                source_frame_idx=frame.source_frame_idx,
                timestamp_ms=frame.timestamp_ms,
                answer=result["answer"],
                raw_answer=reasoning.get("raw_answer"),
                model_name=reasoning.get("model_name"),
                confidence_score=confidence_score,
                answer_source=answer_source,
                window_id=cluster.cluster_id,
                evidence={
                    "representative_source_frame_idx": cluster.source_frame_idx,
                    "retrieval_rank": cluster.retrieval_rank,
                    "retrieval_score": cluster.retrieval_score,
                    "shot_id": cluster.shot_id,
                    "evidence_frame_indices": reasoning.get("evidence_frame_indices", []),
                    "evidence_source_frame_indices": [
                        item["source_frame_idx"]
                        for item in window_frames
                        if item["position"] in evidence_positions
                    ],
                    "window_frames": window_frames,
                    "provider": reasoning.get("provider"),
                    "latency_ms": reasoning.get("latency_ms"),
                    "attempts": reasoning.get("attempts"),
                    "cache_status": reasoning.get("cache_status"),
                },
            ).to_dict()
            ordered.append(output)
            seen.add(key)
            if len(ordered) >= requested_limit:
                break
        state["results"] = stored_results
        state["failures"] = failures
        state["predictions"] = ordered
        state["window_outcomes"] = outcomes
        state["cluster_count"] = len(active_clusters)
        sessions.update_state(str(session_value), state, expected_revision=session.revision)
        active_statuses = {
            str(outcomes.get(cluster.cluster_id, {}).get("status", "pending"))
            for cluster in active_clusters
        }
        semantic_success = bool(active_statuses & {"answered", "unanswerable"})
        technical_failure = any(status.endswith("_failed") for status in active_statuses)
        if active_clusters and technical_failure and not semantic_success:
            codes = [
                str(outcomes[cluster.cluster_id].get("failure_code"))
                for cluster in active_clusters
                if outcomes.get(cluster.cluster_id, {}).get("failure_code")
            ]
            raise VisualQATotalFailureError(str(session_value), codes)
        return ordered

    def search(
        self, request: QAQueryRequest | Any, *, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Start an initial QA search; alias retained for API composition."""

        return self.answer(request, session_id=session_id)

    def expand(self, search_id: str, target_clusters: int) -> list[dict[str, Any]]:
        """Idempotently expand a live search to a larger cumulative cluster target."""

        if self.session_store is None:
            raise RuntimeError("Visual Q&A session_store is unavailable")
        if not 1 <= target_clusters <= 20:
            raise ValueError("target_clusters must be between 1 and 20")
        snapshot = self.session_store.get_snapshot(search_id)
        if snapshot is None:
            raise KeyError(f"Visual Q&A search is missing or expired: {search_id}")
        cumulative_target = max(int(snapshot["target_count"]), target_clusters)
        return self.answer(
            snapshot["request"],
            session_id=search_id,
            max_clusters=cumulative_target,
        )

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        """Return a restorable search/session snapshot without rerunning retrieval."""

        if self.session_store is None:
            raise RuntimeError("Visual Q&A session_store is unavailable")
        return self.session_store.get_snapshot(search_id)


# Compatibility with contract terminology used by some consumers.
QACandidateWindow = QACandidateCluster

__all__ = [
    "QACandidateWindow",
    "QAReasoner",
    "QAWindowFrameRef",
    "QAWindowOutcome",
    "VisualQAPipeline",
    "VisualQATotalFailureError",
]
