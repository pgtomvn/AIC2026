"""Concurrency regression for idempotent Visual Q&A expansion."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.contracts import FrameRecord, QACompetitionSpec, QAQueryRequest, QAReasoningResult
from src.retrieval.qa_media import CanonicalImageProbe, ResolvedCanonicalImage
from src.retrieval.qa_retrieval import QACandidateCluster
from src.retrieval.qa_session import QASessionStore, reasoning_cache_key
from src.retrieval.visual_qa import VisualQAPipeline, VisualQATotalFailureError
from src.vlm.qa_reasoner import (
    ProviderChainReasoner,
    QAProviderRequestError,
)


def _frame(video_id: str, source_frame_idx: int) -> FrameRecord:
    return FrameRecord(
        batch_id="b1",
        video_id=video_id,
        keyframe_ordinal=source_frame_idx + 1,
        source_frame_idx=source_frame_idx,
        pts=source_frame_idx,
        time_base="1/10",
        timestamp_ms=source_frame_idx * 100,
        fps_num=10,
        fps_den=1,
        shot_id=f"shot:{video_id}",
        sampling_reasons=("fixture",),
        image_uri="fixture.jpg",
        video_uri="fixture.mp4",
        source_sha256="a" * 64,
        extractor_version="fixture-v1",
    )


class _Retriever:
    def __init__(self, count: int = 2) -> None:
        self.max_clusters = count
        self.calls = 0
        self.clusters = tuple(
            QACandidateCluster(
                cluster_id=f"b1:{video}:10",
                retrieval_rank=index,
                retrieval_score=1.0 / index,
                batch_id="b1",
                video_id=video,
                source_frame_idx=10,
                timestamp_ms=1_000,
                shot_id=f"shot:{video}",
                frames=tuple(_frame(video, value) for value in (9, 10, 11)),
            )
            for index, video in enumerate((f"L21_V{value:03d}" for value in range(1, count + 1)), 1)
        )

    def retrieve(self, _event, *, batch_filters=(), max_clusters=None):
        self.calls += 1
        return self.clusters[:max_clusters]


class _BlockingRetriever(_Retriever):
    def __init__(self, started: threading.Event, release: threading.Event, count: int = 1) -> None:
        super().__init__(count)
        self.started = started
        self.release = release

    def retrieve(self, event, *, batch_filters=(), max_clusters=None):
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().retrieve(event, batch_filters=batch_filters, max_clusters=max_clusters)


class _Resolver:
    def resolve(self, frame: FrameRecord) -> ResolvedCanonicalImage:
        data = f"jpeg-{frame.video_id}-{frame.source_frame_idx}".encode()
        return ResolvedCanonicalImage(
            frame.batch_id,
            frame.video_id,
            frame.source_frame_idx,
            frame.timestamp_ms,
            hashlib.sha256(data).hexdigest(),
            8,
            8,
            "image/jpeg",
            data,
        )


class _ProbeResolver(_Resolver):
    fingerprint = "probe-resolver-v1"

    def __init__(self) -> None:
        self.resolve_calls = 0
        self.probe_calls = 0

    def probe(self, frame: FrameRecord) -> CanonicalImageProbe:
        self.probe_calls += 1
        revision = hashlib.sha256(
            f"{frame.batch_id}:{frame.video_id}:{frame.source_frame_idx}".encode()
        ).hexdigest()
        return CanonicalImageProbe(
            frame.batch_id,
            frame.video_id,
            frame.source_frame_idx,
            frame.timestamp_ms,
            revision,
            self.fingerprint,
        )

    def resolve(self, frame: FrameRecord) -> ResolvedCanonicalImage:
        self.resolve_calls += 1
        return super().resolve(frame)


class _SignalingLeaseStore(QASessionStore):
    def __init__(self, path: Path, waiting: threading.Event) -> None:
        super().__init__(path)
        self.waiting = waiting

    def acquire_cache_lease(self, cache_key: str, owner_token: str, *, ttl_seconds: float) -> bool:
        acquired = super().acquire_cache_lease(cache_key, owner_token, ttl_seconds=ttl_seconds)
        if cache_key.startswith("window-work:") and not acquired:
            self.waiting.set()
        return acquired


class _ReasoningLeaseReleaseStore(QASessionStore):
    def __init__(self, path: Path, released: threading.Event) -> None:
        super().__init__(path)
        self.released = released

    def release_cache_lease(self, cache_key: str, owner_token: str) -> None:
        super().release_cache_lease(cache_key, owner_token)
        if cache_key.startswith("reasoning:"):
            self.released.set()


class _AcquireAfterReasoningReleaseStore(QASessionStore):
    def __init__(
        self,
        path: Path,
        cache_checked: threading.Event,
        prior_released: threading.Event,
    ) -> None:
        super().__init__(path)
        self.cache_checked = cache_checked
        self.prior_released = prior_released
        self._forced_empty = False

    def get_reasoning_many(self, cache_keys):
        if not self._forced_empty:
            self._forced_empty = True
            self.cache_checked.set()
            return {}
        return super().get_reasoning_many(cache_keys)

    def acquire_cache_lease(self, cache_key: str, owner_token: str, *, ttl_seconds: float) -> bool:
        if cache_key.startswith("reasoning:") and self.cache_checked.is_set():
            assert self.prior_released.wait(timeout=5)
        return super().acquire_cache_lease(
            cache_key, owner_token, ttl_seconds=ttl_seconds
        )


class _Reasoner:
    fingerprint = "concurrency-fake-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.guard = threading.Lock()

    def reason(self, _request, _images, spec):
        with self.guard:
            self.calls += 1
        return QAReasoningResult(
            raw_answer="three",
            normalized_answer=spec.normalize_answer("three"),
            answerable=True,
            evidence_frame_indices=(2,),
            provider="fake",
            model_name="fake-v1",
            prompt_version="fixture-v1",
            latency_ms=1.0,
        )

    def unload(self) -> None:
        return None


class _FailingReasoner(_Reasoner):
    fingerprint = "failing-concurrency-fake-v1"

    def reason(self, _request, _images, _spec):
        with self.guard:
            self.calls += 1
        raise RuntimeError("sensitive provider detail")


class _MetadataFailingReasoner(_Reasoner):
    fingerprint = "metadata-failing-fake-v1"

    def reason(self, _request, _images, _spec):
        with self.guard:
            self.calls += 1
        raise QAProviderRequestError(
            "safe provider request failure",
            provider="gemini",
            model="gemini-test",
            attempts=2,
            category="rate_limited",
            status_code=429,
        )


class _UnanswerableReasoner(_Reasoner):
    fingerprint = "unanswerable-fake-v1"

    def reason(self, _request, _images, _spec):
        with self.guard:
            self.calls += 1
        return QAReasoningResult(
            raw_answer="",
            normalized_answer="",
            answerable=False,
            evidence_frame_indices=(),
            provider="fake",
            model_name="fake-v1",
            prompt_version="fixture-v1",
            latency_ms=1.0,
        )


class _BlockingReasoner(_Reasoner):
    fingerprint = "blocking-fake-v1"

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.started = started
        self.release = release

    def reason(self, request, images, spec):
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().reason(request, images, spec)


class _OrderedProvider:
    def __init__(self, name: str, *, transient_failure: bool = False) -> None:
        self.provider_name = name
        self.fingerprint = f"{name}-fingerprint-v1"
        self.transient_failure = transient_failure
        self.calls = 0

    def reason(self, _request, _images, spec):
        self.calls += 1
        if self.transient_failure:
            raise QAProviderRequestError(
                "sanitized transient fixture",
                attempts=5,
                category="server_unavailable",
            )
        return QAReasoningResult(
            raw_answer="three",
            normalized_answer=spec.normalize_answer("three"),
            answerable=True,
            evidence_frame_indices=(2,),
            provider=self.provider_name,
            model_name=f"{self.provider_name}-model",
            prompt_version="fixture-v1",
            latency_ms=1.0,
        )

    def unload(self) -> None:
        return None


def test_concurrent_expansion_reasons_each_new_cluster_once(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa.sqlite3")
    sessions.initialize()
    reasoner = _Reasoner()
    pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    request = QAQueryRequest(
        query_id="concurrent-search",
        event_description="people holding fruit",
        question="How many?",
    )
    pipeline.search(request, session_id="concurrent-search")

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = tuple(
            workers.map(lambda _value: pipeline.expand("concurrent-search", 2), range(2))
        )

    assert reasoner.calls == 2
    assert results[0] == results[1]
    assert sessions.get_snapshot("concurrent-search")["target_count"] == 2


def test_cross_pipeline_concurrent_expansion_merges_with_sqlite_cas(tmp_path: Path) -> None:
    path = tmp_path / "qa-cross-process.sqlite3"
    first_store = QASessionStore(path)
    second_store = QASessionStore(path)
    first_store.initialize()
    second_store.initialize()
    reasoner = _Reasoner()
    first_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(2),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=first_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    second_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(2),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=second_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    request = QAQueryRequest(
        query_id="cross-process-search",
        event_description="people holding fruit",
        question="How many?",
    )
    first_pipeline.search(request, session_id="cross-process-search")

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = tuple(
            workers.map(
                lambda pipeline: pipeline.expand("cross-process-search", 2),
                (first_pipeline, second_pipeline),
            )
        )

    assert reasoner.calls == 2
    assert results[0] == results[1]
    snapshot = first_store.get_snapshot("cross-process-search")
    assert snapshot["reasoned_count"] == 2
    assert snapshot["revision"] >= 4


def test_cross_pipeline_candidate_cache_uses_sqlite_single_flight(tmp_path: Path) -> None:
    path = tmp_path / "qa-candidate-single-flight.sqlite3"
    first_store = QASessionStore(path)
    second_store = QASessionStore(path)
    first_store.initialize()
    second_store.initialize()
    started = threading.Event()
    release = threading.Event()
    first_retriever = _BlockingRetriever(started, release)
    second_retriever = _BlockingRetriever(started, release)
    reasoner = _Reasoner()
    first_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=first_retriever,
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=first_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    second_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=second_retriever,
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=second_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    requests = (
        QAQueryRequest("candidate-one", "people holding fruit", "How many?"),
        QAQueryRequest("candidate-two", "people holding fruit", "How many?"),
    )
    with ThreadPoolExecutor(max_workers=2) as workers:
        first_future = workers.submit(
            first_pipeline.search, requests[0], session_id="candidate-one"
        )
        assert started.wait(timeout=5)
        second_future = workers.submit(
            second_pipeline.search, requests[1], session_id="candidate-two"
        )
        release.set()
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert first[0]["answer"] == second[0]["answer"]
    assert first_retriever.calls + second_retriever.calls == 1
    assert reasoner.calls == 1


def test_reasoning_lease_rechecks_cache_after_prior_owner_releases(tmp_path: Path) -> None:
    path = tmp_path / "qa-reasoning-lease-release-race.sqlite3"
    cache_checked = threading.Event()
    prior_released = threading.Event()
    provider_started = threading.Event()
    provider_release = threading.Event()
    first_store = _ReasoningLeaseReleaseStore(path, prior_released)
    second_store = _AcquireAfterReasoningReleaseStore(
        path, cache_checked, prior_released
    )
    first_store.initialize()
    second_store.initialize()
    reasoner = _BlockingReasoner(provider_started, provider_release)
    first_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(1),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=first_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    second_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(1),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=second_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    requests = (
        QAQueryRequest("lease-owner", "people holding fruit", "How many?"),
        QAQueryRequest("lease-waiter", "people holding fruit", "How many?"),
    )

    with ThreadPoolExecutor(max_workers=2) as workers:
        first_future = workers.submit(
            first_pipeline.search, requests[0], session_id="lease-owner"
        )
        assert provider_started.wait(timeout=5)
        second_future = workers.submit(
            second_pipeline.search, requests[1], session_id="lease-waiter"
        )
        assert cache_checked.wait(timeout=5)
        provider_release.set()
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert first[0]["answer"] == second[0]["answer"]
    assert reasoner.calls == 1
    waiter_outcome = second_pipeline.get_search("lease-waiter")["window_outcomes"][0]
    assert waiter_outcome["cache_layers"]["reasoning"] == "hit"


def test_cross_pipeline_window_lease_skips_waiter_image_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qa-window-work-single-flight.sqlite3"
    first_store = QASessionStore(path)
    waiter_waiting = threading.Event()
    second_store = _SignalingLeaseStore(path, waiter_waiting)
    first_store.initialize()
    second_store.initialize()
    provider_started = threading.Event()
    provider_release = threading.Event()
    reasoner = _BlockingReasoner(provider_started, provider_release)
    first_resolver = _ProbeResolver()
    second_resolver = _ProbeResolver()
    first_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(1),
        reasoner=reasoner,
        image_resolver=first_resolver,
        session_store=first_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    second_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(1),
        reasoner=reasoner,
        image_resolver=second_resolver,
        session_store=second_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    requests = (
        QAQueryRequest("window-one", "people holding fruit", "How many?"),
        QAQueryRequest("window-two", "people holding fruit", "How many?"),
    )

    with ThreadPoolExecutor(max_workers=2) as workers:
        first_future = workers.submit(first_pipeline.search, requests[0], session_id="window-one")
        assert provider_started.wait(timeout=5)
        second_future = workers.submit(second_pipeline.search, requests[1], session_id="window-two")
        assert waiter_waiting.wait(timeout=5)
        provider_release.set()
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert first[0]["answer"] == second[0]["answer"]
    assert first_resolver.resolve_calls == 3
    assert second_resolver.resolve_calls == 0
    assert reasoner.calls == 1
    outcome = second_pipeline.get_search("window-two")["window_outcomes"][0]
    assert outcome["cache_layers"]["window_work"] == "wait_hit"


def test_candidate_cache_hit_restores_safe_retrieval_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "qa-candidate-diagnostics.sqlite3"
    first_store = QASessionStore(path)
    first_store.initialize()
    first_retriever = _Retriever(1)
    first_retriever.textual_pipeline = type(
        "TextualDiagnostics",
        (),
        {
            "last_channel_statuses": {"organizer_clip": {"state": "ready", "reason_code": None}},
            "last_channel_errors": {"semantic_text": "unsafe raw detail"},
        },
    )()
    reasoner = _Reasoner()
    first_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=first_retriever,
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=first_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    first_pipeline.search(
        QAQueryRequest("diagnostics-one", "people holding fruit", "How many?"),
        session_id="diagnostics-one",
    )

    second_store = QASessionStore(path)
    second_store.initialize()
    second_retriever = _Retriever(1)
    second_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=second_retriever,
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=second_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    second_pipeline.search(
        QAQueryRequest("diagnostics-two", "people holding fruit", "How many?"),
        session_id="diagnostics-two",
    )
    snapshot = second_pipeline.get_search("diagnostics-two")

    assert second_retriever.calls == 0
    assert snapshot["channel_statuses"] == {
        "organizer_clip": {"state": "ready", "reason_code": None}
    }
    assert snapshot["retrieval_errors"] == {"semantic_text": "channel_unavailable"}
    assert "unsafe raw detail" not in str(snapshot)


def test_reasoning_cache_replays_across_query_ids_with_hit_provenance(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa-cache.sqlite3")
    sessions.initialize()
    reasoner = _Reasoner()
    pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    first_request = QAQueryRequest(
        query_id="query-one",
        event_description="people holding fruit",
        question="How many?",
    )
    second_request = QAQueryRequest(
        query_id="query-two",
        event_description=first_request.event_description,
        question=first_request.question,
    )

    first = pipeline.search(first_request, session_id="session-one")
    second = pipeline.search(second_request, session_id="session-two")

    assert reasoner.calls == 1
    assert first[0]["evidence"]["cache_status"] == "miss"
    assert second[0]["evidence"]["cache_status"] == "hit"

    recorded = sessions.record_operator_edit(
        "session-two",
        second[0]["window_id"],
        {
            "original_raw_answer": "three",
            "edited_raw_answer": "3",
            "normalized_answer": "3",
        },
    )
    assert recorded["edit_id"] == 1

    restarted_store = QASessionStore(tmp_path / "qa-cache.sqlite3")
    restarted_store.initialize()
    restored = restarted_store.get_snapshot("session-two")
    assert restored is not None
    assert restored["state"]["predictions"][0]["source_frame_idx"] == 10
    assert restored["operator_edits"][0]["normalized_answer"] == "3"


def test_provider_chain_cache_uses_actual_provider_and_probes_in_order(
    tmp_path: Path,
) -> None:
    sessions = QASessionStore(tmp_path / "qa-provider-order.sqlite3")
    sessions.initialize()
    primary = _OrderedProvider("primary", transient_failure=True)
    fallback = _OrderedProvider("fallback")
    chain = ProviderChainReasoner((primary, fallback))
    retriever = _Retriever(1)
    pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=retriever,
        reasoner=chain,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    first_request = QAQueryRequest(
        query_id="provider-order-one",
        event_description="people holding fruit",
        question="How many?",
    )

    first = pipeline.search(first_request, session_id="provider-order-one")

    frame_fingerprints = []
    for frame in retriever.clusters[0].frames:
        data = f"jpeg-{frame.video_id}-{frame.source_frame_idx}".encode()
        frame_fingerprints.append(
            {
                "batch_id": frame.batch_id,
                "video_id": frame.video_id,
                "source_frame_idx": frame.source_frame_idx,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    request_fingerprint = pipeline._semantic_request_fingerprint(
        first_request.event_description, first_request.question
    )
    primary_key = reasoning_cache_key(
        provider_fingerprint=primary.fingerprint,
        request_fingerprint=request_fingerprint,
        answer_spec_id=first_request.answer_spec_id,
        frame_fingerprints=frame_fingerprints,
    )
    fallback_key = reasoning_cache_key(
        provider_fingerprint=fallback.fingerprint,
        request_fingerprint=request_fingerprint,
        answer_spec_id=first_request.answer_spec_id,
        frame_fingerprints=frame_fingerprints,
    )
    chain_key = reasoning_cache_key(
        provider_fingerprint=chain.fingerprint,
        request_fingerprint=request_fingerprint,
        answer_spec_id=first_request.answer_spec_id,
        frame_fingerprints=frame_fingerprints,
    )
    assert sessions.get_reasoning(primary_key) is None
    fallback_result = sessions.get_reasoning(fallback_key)
    assert fallback_result is not None
    assert sessions.get_reasoning(chain_key) is None

    second_request = QAQueryRequest(
        query_id="provider-order-two",
        event_description=first_request.event_description,
        question=first_request.question,
    )
    second = pipeline.search(second_request, session_id="provider-order-two")
    assert primary.calls == 1
    assert fallback.calls == 1
    assert second[0]["evidence"]["provider"] == "fallback"

    primary_result = dict(fallback_result)
    primary_result["provider"] = "primary"
    primary_result["model_name"] = "primary-model"
    sessions.put_reasoning(primary_key, primary_result)
    third_request = QAQueryRequest(
        query_id="provider-order-three",
        event_description=first_request.event_description,
        question=first_request.question,
    )
    third = pipeline.search(third_request, session_id="provider-order-three")

    assert first[0]["answer"] == second[0]["answer"] == third[0]["answer"]
    assert third[0]["evidence"]["provider"] == "primary"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_reasoning_lease_ttl_tracks_provider_retry_budget_and_caps() -> None:
    provider = _OrderedProvider("timed")
    provider._timeout_seconds = 2.0
    provider._max_attempts = 3
    provider._retry_max_seconds = 1.0
    assert VisualQAPipeline._reasoning_lease_ttl(provider) == 18.0

    primary = _OrderedProvider("slow-primary")
    primary._timeout_seconds = 30.0
    primary._max_attempts = 3
    fallback = _OrderedProvider("slow-fallback")
    fallback._timeout_seconds = 60.0
    assert (
        VisualQAPipeline._reasoning_lease_ttl(ProviderChainReasoner((primary, fallback))) == 160.0
    )

    fallback._timeout_seconds = 400.0
    assert (
        VisualQAPipeline._reasoning_lease_ttl(ProviderChainReasoner((primary, fallback))) == 500.0
    )

    fallback._timeout_seconds = 1_000.0
    assert (
        VisualQAPipeline._reasoning_lease_ttl(ProviderChainReasoner((primary, fallback))) == 600.0
    )


def test_cascade_adds_at_most_five_provider_calls_per_step(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa-limit.sqlite3")
    sessions.initialize()
    reasoner = _Reasoner()
    pipeline = VisualQAPipeline(
        top_k_retrieve=5,
        retriever=_Retriever(20),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    request = QAQueryRequest(
        query_id="bounded-cascade",
        event_description="people holding fruit",
        question="How many?",
    )

    pipeline.search(request, session_id="bounded-cascade")
    assert reasoner.calls == 5
    pipeline.expand("bounded-cascade", 10)
    assert reasoner.calls == 10
    pipeline.expand("bounded-cascade", 10)
    assert reasoner.calls == 10
    pipeline.expand("bounded-cascade", 15)
    assert reasoner.calls == 15
    predictions = pipeline.expand("bounded-cascade", 20)
    assert reasoner.calls == 20
    assert len(predictions) == 60


def test_failed_windows_are_not_retried_during_later_expansions(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa-failure-limit.sqlite3")
    sessions.initialize()
    reasoner = _FailingReasoner()
    pipeline = VisualQAPipeline(
        top_k_retrieve=5,
        retriever=_Retriever(20),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    request = QAQueryRequest(
        query_id="bounded-failures",
        event_description="people holding fruit",
        question="How many?",
    )

    with pytest.raises(VisualQATotalFailureError) as initial_failure:
        pipeline.search(request, session_id="bounded-failures")
    assert initial_failure.value.search_id == "bounded-failures"
    assert initial_failure.value.failure_codes == ("internal_reasoning_error",)
    assert reasoner.calls == 5
    for expected_calls, target in ((10, 10), (15, 15), (20, 20)):
        with pytest.raises(VisualQATotalFailureError):
            pipeline.expand("bounded-failures", target)
        assert reasoner.calls == expected_calls
    snapshot = sessions.get_snapshot("bounded-failures")
    assert set(snapshot["state"]["failures"].values()) == {"internal_reasoning_error"}
    assert snapshot["status"] == "failed"
    assert "sensitive provider detail" not in str(snapshot)


def test_legacy_terminal_failure_is_seeded_and_never_retried(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa-legacy-terminal.sqlite3")
    sessions.initialize()
    reasoner = _FailingReasoner()
    request = QAQueryRequest(
        query_id="legacy-terminal",
        event_description="people holding fruit",
        question="How many?",
    )
    target_key = "b1:L21_V001:10"
    sessions.create_session(
        "legacy-terminal",
        request.to_dict(),
        state={"results": {}, "failures": {target_key: "reasoning_failed"}},
    )
    sessions.expand_targets(
        "legacy-terminal",
        (
            {
                "batch_id": "b1",
                "video_id": "L21_V001",
                "source_frame_idx": 10,
                "timestamp_ms": 1_000,
                "window_id": target_key,
            },
        ),
    )
    pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(1),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )

    with pytest.raises(VisualQATotalFailureError) as captured:
        pipeline.expand("legacy-terminal", 1)

    assert captured.value.failure_codes == ("internal_reasoning_error",)
    assert reasoner.calls == 0
    outcome = pipeline.get_search("legacy-terminal")["window_outcomes"][0]
    assert outcome["failure_code"] == "internal_reasoning_error"
    assert outcome["failure_stage"] == "internal"


def test_provider_failure_outcome_keeps_only_safe_execution_metadata(
    tmp_path: Path,
) -> None:
    sessions = QASessionStore(tmp_path / "qa-provider-metadata.sqlite3")
    sessions.initialize()
    pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=_Retriever(1),
        reasoner=_MetadataFailingReasoner(),
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    request = QAQueryRequest(
        query_id="provider-metadata",
        event_description="people holding fruit",
        question="How many?",
    )

    with pytest.raises(VisualQATotalFailureError):
        pipeline.search(request, session_id="provider-metadata")

    outcome = pipeline.get_search("provider-metadata")["window_outcomes"][0]
    assert outcome["failure_code"] == "provider_request_failed"
    assert outcome["failure_stage"] == "provider"
    assert outcome["provider"] == "gemini"
    assert outcome["model_name"] == "gemini-test"
    assert outcome["attempts"] == 2
    assert outcome["provider_error_category"] == "rate_limited"
    assert outcome["provider_status_code"] == 429
    assert "safe provider request failure" not in str(outcome)


def test_unanswerable_windows_are_successful_visible_outcomes(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa-unanswerable.sqlite3")
    sessions.initialize()
    reasoner = _UnanswerableReasoner()
    pipeline = VisualQAPipeline(
        top_k_retrieve=2,
        retriever=_Retriever(2),
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    request = QAQueryRequest(
        query_id="unanswerable-search",
        event_description="an ambiguous event",
        question="How many?",
    )

    assert pipeline.search(request, session_id="unanswerable-search") == []
    snapshot = pipeline.get_search("unanswerable-search")

    assert reasoner.calls == 2
    assert snapshot["status"] == "completed_no_answer"
    assert snapshot["reasoned_count"] == 2
    assert snapshot["answerable_count"] == 0
    assert snapshot["unanswerable_count"] == 2
    assert snapshot["failed_count"] == 0
    assert {outcome["status"] for outcome in snapshot["window_outcomes"]} == {"unanswerable"}
    assert all(outcome["frames"] for outcome in snapshot["window_outcomes"])


def test_persisted_replay_skips_retrieval_media_decode_and_provider(tmp_path: Path) -> None:
    path = tmp_path / "qa-fast-replay.sqlite3"
    first_store = QASessionStore(path)
    first_store.initialize()
    first_retriever = _Retriever(1)
    first_resolver = _ProbeResolver()
    reasoner = _Reasoner()
    first_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=first_retriever,
        reasoner=reasoner,
        image_resolver=first_resolver,
        session_store=first_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    first_request = QAQueryRequest(
        query_id="fast-one",
        event_description="people holding fruit",
        question="How many?",
    )
    first = first_pipeline.search(first_request, session_id="fast-one")

    restarted_store = QASessionStore(path)
    restarted_store.initialize()
    replay_retriever = _Retriever(1)
    replay_resolver = _ProbeResolver()
    replay_pipeline = VisualQAPipeline(
        top_k_retrieve=1,
        retriever=replay_retriever,
        reasoner=reasoner,
        image_resolver=replay_resolver,
        session_store=restarted_store,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    replay_request = QAQueryRequest(
        query_id="fast-two",
        event_description=first_request.event_description,
        question=first_request.question,
    )
    replay = replay_pipeline.search(replay_request, session_id="fast-two")

    assert [value["source_frame_idx"] for value in replay] == [
        value["source_frame_idx"] for value in first
    ]
    assert [value["answer"] for value in replay] == [value["answer"] for value in first]
    assert first_retriever.calls == 1
    assert replay_retriever.calls == 0
    assert first_resolver.resolve_calls == 3
    assert replay_resolver.resolve_calls == 0
    assert reasoner.calls == 1
    replay_snapshot = replay_pipeline.get_search("fast-two")
    layers = replay_snapshot["window_outcomes"][0]["cache_layers"]
    assert layers == {
        "candidate": "hit",
        "image_fingerprint": "hit",
        "reasoning": "hit",
    }


def test_retrieval_exhaustion_prevents_repeat_work(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa-exhausted.sqlite3")
    sessions.initialize()
    retriever = _Retriever(2)
    reasoner = _Reasoner()
    pipeline = VisualQAPipeline(
        top_k_retrieve=5,
        retriever=retriever,
        reasoner=reasoner,
        image_resolver=_Resolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.historical_numeric(),
    )
    request = QAQueryRequest(
        query_id="exhausted",
        event_description="people holding fruit",
        question="How many?",
    )

    pipeline.search(request, session_id="exhausted")
    before = pipeline.get_search("exhausted")
    pipeline.expand("exhausted", 20)
    after = pipeline.get_search("exhausted")

    assert before["retrieved_clusters"] == 2
    assert before["retrieval_exhausted"] is True
    assert after["requested_target_clusters"] == 20
    assert after["retrieval_exhausted"] is True
    assert retriever.calls == 1
    assert reasoner.calls == 2
