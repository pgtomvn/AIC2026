"""Offline tests for Visual Q&A media, retrieval, session, and cascade core."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from src.contracts import (
    FrameRecord,
    QACompetitionSpec,
    QAQueryRequest,
    QAReasoningResult,
    RefinedFrameRecord,
)
from src.retrieval.qa_media import CanonicalImageError, CanonicalImageResolver
from src.retrieval.qa_retrieval import QACandidateCluster, VisualQARetriever
from src.retrieval.qa_session import QASessionStore, QASessionStoreError, reasoning_cache_key
from src.retrieval.visual_qa import VisualQAPipeline, VisualQATotalFailureError


def _frame(
    image_uri: str,
    *,
    batch_id: str = "b1",
    video_id: str = "L21_V001",
    source_frame_idx: int = 10,
    timestamp_ms: int = 1_000,
    shot_id: str = "s1",
) -> FrameRecord:
    return FrameRecord(
        batch_id=batch_id,
        video_id=video_id,
        keyframe_ordinal=source_frame_idx + 1,
        source_frame_idx=source_frame_idx,
        pts=source_frame_idx,
        time_base="1/10",
        timestamp_ms=timestamp_ms,
        fps_num=10,
        fps_den=1,
        shot_id=shot_id,
        sampling_reasons=("fixture",),
        image_uri=image_uri,
        video_uri="fixture.mp4",
        source_sha256="a" * 64,
        extractor_version="fixture-v1",
    )


def _jpeg_bytes(
    *,
    size: tuple[int, int] = (80, 40),
    orientation: int | None = None,
    software: str | None = None,
) -> bytes:
    output = io.BytesIO()
    image = Image.new("RGB", size, "red")
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    if software is not None:
        exif[305] = software
    image.save(output, "JPEG", exif=exif)
    return output.getvalue()


def test_canonical_image_resolver_handles_local_and_zip_exif_with_bounds(
    tmp_path: Path,
) -> None:
    raw = _jpeg_bytes(orientation=6)
    local = tmp_path / "frame.jpg"
    local.write_bytes(raw)
    archive_path = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("keyframes/L21_V001/001.jpg", raw)
    resolver = CanonicalImageResolver((tmp_path,), max_dimension=32)
    zip_uri = f"zip://{archive_path.resolve()}!/keyframes/L21_V001/001.jpg"

    local_result = resolver.resolve(_frame(str(local)))
    zip_result = resolver.resolve(_frame(zip_uri))

    assert resolver.source_uri_resolvable(str(local)) is True
    assert resolver.source_uri_resolvable(zip_uri) is True
    assert resolver.source_uris_resolvable((str(local), zip_uri)) is True
    cached_archive = next(iter(resolver._archives.values()))
    assert isinstance(cached_archive[2], frozenset)
    assert cached_archive[2] == frozenset()
    assert (local_result.width, local_result.height) == (16, 32)
    assert local_result.content_sha256 == zip_result.content_sha256
    assert local_result.data.startswith(b"\xff\xd8")
    assert zip_result.mime_type == "image/jpeg"

    metadata_variant = tmp_path / "metadata-variant.jpg"
    metadata_variant.write_bytes(_jpeg_bytes(orientation=6, software="different-container"))
    variant_result = resolver.resolve(_frame(str(metadata_variant)))
    assert variant_result.content_sha256 == local_result.content_sha256


def test_canonical_image_resolver_rejects_escape_oversize_and_invalid_content(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg_bytes())
    resolver = CanonicalImageResolver((allowed,), max_input_bytes=32)

    with pytest.raises(CanonicalImageError, match="outside approved roots"):
        resolver.resolve(_frame(str(outside)))

    oversized = allowed / "large.jpg"
    oversized.write_bytes(_jpeg_bytes())
    with pytest.raises(CanonicalImageError, match="exceeds configured bounds"):
        resolver.resolve(_frame(str(oversized)))

    invalid = allowed / "invalid.jpg"
    invalid.write_bytes(b"not an image")
    permissive = CanonicalImageResolver((allowed,))
    with pytest.raises(CanonicalImageError, match="invalid canonical image"):
        permissive.resolve(_frame(str(invalid)))


def test_canonical_image_resolver_normalizes_same_frame_once_concurrently(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "single-flight.jpg"
    image_path.write_bytes(_jpeg_bytes())

    class CountingResolver(CanonicalImageResolver):
        def __init__(self) -> None:
            super().__init__((tmp_path,))
            self.read_count = 0
            self.count_guard = threading.Lock()

        def _read_local(self, uri: str) -> bytes:
            with self.count_guard:
                self.read_count += 1
            time.sleep(0.05)
            return super()._read_local(uri)

    resolver = CountingResolver()
    frame = _frame(str(image_path))

    with ThreadPoolExecutor(max_workers=8) as workers:
        resolved = tuple(workers.map(lambda _: resolver.resolve(frame), range(8)))

    assert resolver.read_count == 1
    assert len({value.content_sha256 for value in resolved}) == 1


def test_canonical_image_resolver_rejects_unsafe_duplicate_and_corrupt_zip_members(
    tmp_path: Path,
) -> None:
    raw = _jpeg_bytes()
    archive_path = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("frames/001.jpg", raw)
            archive.writestr("frames/001.jpg", raw)
    resolver = CanonicalImageResolver((tmp_path,))
    duplicate_uri = f"zip://{archive_path.resolve()}!/frames/001.jpg"

    with pytest.raises(CanonicalImageError, match="duplicated"):
        resolver.resolve(_frame(duplicate_uri))
    assert resolver.source_uri_resolvable(duplicate_uri) is False
    assert (
        resolver.source_uri_resolvable(
            f"zip://{archive_path.resolve()}!/frames/missing.jpg"
        )
        is False
    )
    assert resolver.source_uris_resolvable((duplicate_uri,)) is False
    assert (
        resolver.source_uris_resolvable(
            (f"zip://{archive_path.resolve()}!/frames/missing.jpg",)
        )
        is False
    )
    cached_archive = next(iter(resolver._archives.values()))
    assert cached_archive[2] == frozenset({"frames/001.jpg"})
    with pytest.raises(CanonicalImageError, match="unsafe ZIP image member"):
        resolver.resolve(_frame(f"zip://{archive_path.resolve()}!/../001.jpg"))

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(CanonicalImageError, match="cannot read canonical ZIP image"):
        resolver.resolve(_frame(f"zip://{corrupt.resolve()}!/frames/001.jpg"))
    with pytest.raises(CanonicalImageError, match="unsupported canonical image URI"):
        resolver.resolve(_frame("https://example.invalid/frame.jpg"))


def test_canonical_image_resolver_enforces_decoded_pixel_limit(tmp_path: Path) -> None:
    image_path = tmp_path / "pixels.jpg"
    image_path.write_bytes(_jpeg_bytes(size=(10, 10)))
    resolver = CanonicalImageResolver((tmp_path,), max_pixels=99)

    with pytest.raises(CanonicalImageError, match="pixel bounds"):
        resolver.resolve(_frame(str(image_path)))


def test_canonical_image_resolver_accepts_persisted_refined_frame_and_checks_hash(
    tmp_path: Path,
) -> None:
    output = io.BytesIO()
    Image.new("RGB", (32, 16), "blue").save(output, "WEBP")
    raw = output.getvalue()
    image_path = tmp_path / "refined.webp"
    image_path.write_bytes(raw)
    record = RefinedFrameRecord(
        batch_id="b1",
        video_id="L21_V001",
        source_frame_idx=11,
        coarse_source_frame_idx=10,
        pts=11,
        time_base="1/10",
        timestamp_ms=1_100,
        fps_num=10,
        fps_den=1,
        shot_id="s1",
        image_uri=str(image_path),
        video_uri="fixture.mp4",
        source_sha256="a" * 64,
        image_sha256=hashlib.sha256(raw).hexdigest(),
        image_width=32,
        image_height=16,
        image_bytes=len(raw),
        refiner_fingerprint="refiner-v1",
    )

    resolved = CanonicalImageResolver((tmp_path,)).resolve(record)

    assert resolved.source_frame_idx == 11
    assert resolved.raw_sha256 == record.image_sha256
    image_path.write_bytes(raw + b"tampered")
    with pytest.raises(CanonicalImageError, match="checksum"):
        CanonicalImageResolver((tmp_path,)).resolve(record)


class _FakeTextualPipeline:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.calls: list[tuple[object, int | None, bool]] = []

    def search(self, query, *, limit=None, refine_raw=True):
        self.calls.append((query, limit, refine_raw))
        return self.hits


class _FakeRepository:
    def __init__(self, frames: list[FrameRecord]) -> None:
        self.frames = {
            (frame.batch_id, frame.video_id, frame.source_frame_idx): frame for frame in frames
        }

    def get_frame(self, batch_id, video_id, source_frame_idx):
        return self.frames.get((batch_id, video_id, source_frame_idx))

    def neighbors(self, batch_id, video_id, timestamp_ms, radius_ms=4_000, limit=None):
        values = [
            frame
            for frame in self.frames.values()
            if frame.batch_id == batch_id
            and frame.video_id == video_id
            and abs(frame.timestamp_ms - timestamp_ms) <= radius_ms
        ]
        values.sort(key=lambda frame: (abs(frame.timestamp_ms - timestamp_ms), frame.timestamp_ms))
        return values[:limit]


def test_qa_retrieval_uses_event_only_no_raw_refine_and_round_robins_videos(
    tmp_path: Path,
) -> None:
    seeds = [
        _frame(
            str(tmp_path / f"{video}-{index}.jpg"),
            video_id=video,
            source_frame_idx=index,
            timestamp_ms=time,
            shot_id=shot,
        )
        for video, index, time, shot in (
            ("L21_V001", 1, 1_000, "reference:1"),
            ("L21_V001", 2, 2_000, "reference:2"),
            ("L21_V001", 3, 8_000, "scene:2"),
            ("L21_V002", 4, 1_500, "scene:3"),
        )
    ]
    frames = seeds + [
        _frame(
            str(tmp_path / f"neighbor-{video}-{index}.jpg"),
            video_id=video,
            source_frame_idx=index,
            timestamp_ms=time,
            shot_id=shot,
        )
        for video, index, time, shot in (
            ("L21_V001", 11, 500, "reference:1"),
            ("L21_V001", 12, 1_500, "reference:1"),
            ("L21_V001", 13, 7_500, "scene:2"),
            ("L21_V001", 14, 8_500, "scene:2"),
            ("L21_V002", 15, 1_000, "scene:3"),
            ("L21_V002", 16, 2_000, "scene:3"),
        )
    ]
    hits = [
        {
            "batch_id": frame.batch_id,
            "video_id": frame.video_id,
            "source_frame_idx": frame.source_frame_idx,
            "timestamp_ms": frame.timestamp_ms,
            "shot_id": frame.shot_id,
            "score": 1 / rank,
        }
        for rank, frame in enumerate(seeds, 1)
    ]
    textual = _FakeTextualPipeline(hits)
    retriever = VisualQARetriever(textual, _FakeRepository(frames), max_clusters=20)

    clusters = retriever.retrieve("người đang cầm ô", batch_filters=("b1",), max_clusters=3)

    assert textual.calls == [
        ({"original_text": "người đang cầm ô", "batch_filters": ("b1",)}, 100, False)
    ]
    assert [(value.video_id, value.source_frame_idx) for value in clusters] == [
        ("L21_V001", 1),
        ("L21_V002", 4),
        ("L21_V001", 3),
    ]
    assert all(1 <= len(value.frames) <= 5 for value in clusters)


def test_qa_retrieval_uses_canonical_anchor_timing_for_collapsed_organizer_rows(
    tmp_path: Path,
) -> None:
    seed = _frame(
        str(tmp_path / "anchor.jpg"),
        source_frame_idx=10,
        timestamp_ms=1_000,
        shot_id="reference:10",
    )
    frames = [
        _frame(
            str(tmp_path / f"frame-{index}.jpg"),
            source_frame_idx=index,
            timestamp_ms=timestamp,
            shot_id=f"reference:{index}",
        )
        for index, timestamp in ((9, 960), (11, 1_040))
    ]
    hit = {
        "batch_id": "b1",
        "video_id": "L21_V001",
        "source_frame_idx": 10,
        # A neighboring organizer ordinal collapsed to source frame 10.
        "timestamp_ms": 1_040,
        "shot_id": "reference:11",
        "score": 0.5,
    }
    retriever = VisualQARetriever(
        _FakeTextualPipeline([hit]),
        _FakeRepository([seed, *frames]),
        max_clusters=1,
    )

    clusters = retriever.retrieve("cat", batch_filters=("b1",), max_clusters=1)

    assert len(clusters) == 1
    assert clusters[0].source_frame_idx == seed.source_frame_idx
    assert clusters[0].timestamp_ms == seed.timestamp_ms
    assert clusters[0].shot_id == seed.shot_id
    assert seed in clusters[0].frames


def test_qa_session_ttls_cache_and_target_expansion_are_explicit_and_idempotent(
    tmp_path: Path,
) -> None:
    now = [1_000.0]
    store = QASessionStore(tmp_path / "qa.sqlite3", clock=lambda: now[0])
    assert not store.path.exists()
    store.initialize()
    session = store.create_session("s1", {"question": "what?"})
    assert session.expires_at == 1_000 + 86_400
    target = {"batch_id": "b1", "video_id": "L21_V001", "source_frame_idx": 3}
    assert store.expand_targets("s1", (target, target)) == (target,)
    assert store.expand_targets("s1", (target,)) == ()
    assert store.get_targets("s1") == (target,)
    assert store.get_snapshot("s1")["target_count"] == 1

    key = reasoning_cache_key(
        provider_fingerprint="provider-v1",
        request_fingerprint="request-v1",
        image_sha256s=("a" * 64,),
    )
    store.put_reasoning(key, {"answer": "red", "confidence": 0.9})
    assert store.get_reasoning(key)["answer"] == "red"
    now[0] += 86_401
    assert store.get_session("s1") is None
    with pytest.raises(QASessionStoreError, match="missing or expired"):
        store.expand_targets("s1", (target,))
    assert store.get_reasoning(key) is not None
    now[0] += 30 * 24 * 60 * 60
    assert store.get_reasoning(key) is None
    assert store.purge_expired() == {"sessions": 1, "reasoning": 1}


def test_qa_session_schema_upgrade_is_additive(tmp_path: Path) -> None:
    path = tmp_path / "legacy-qa.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE qa_sessions (
                session_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        connection.commit()

    store = QASessionStore(path)
    store.initialize()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(qa_sessions)")}
        candidate_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(qa_candidate_sets)")
        }
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "revision" in columns
    assert "diagnostics_json" in candidate_columns
    assert {
        "qa_candidate_sets",
        "qa_session_windows",
        "qa_image_fingerprints",
        "qa_cache_leases",
    } <= tables


def test_legacy_snapshot_is_ordered_safe_terminal_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "legacy-restore.sqlite3"
    store = QASessionStore(path, clock=lambda: 1_000.0)
    store.initialize()
    first_key = "b1:Z_VIDEO:20"
    second_key = "b1:A_VIDEO:10"
    third_key = "b1:M_VIDEO:30"
    legacy_state = {
        "window_outcomes": {},
        "results": {
            first_key: {
                "answer": "2",
                "reasoning": {
                    "answer": "2",
                    "raw_answer": "two",
                    "answerable": True,
                    "evidence_frame_indices": [2],
                    "provider": "gemini",
                    "model_name": "gemini-test",
                    "attempts": 1,
                },
            }
        },
        "failures": {
            second_key: "reasoning_failed",
            third_key: "unknown-secret-sdk-failure",
        },
        "predictions": [
            {
                "rank": 1,
                "batch_id": "b1",
                "video_id": "Z_VIDEO",
                "source_frame_idx": 20,
                "window_id": first_key,
                "answer": "2",
                "evidence": {
                    "representative_source_frame_idx": 20,
                    "window_frames": [
                        {"position": 1, "source_frame_idx": 19, "timestamp_ms": 900},
                        {"position": 2, "source_frame_idx": 20, "timestamp_ms": 1_000},
                        {"position": 3, "source_frame_idx": 21, "timestamp_ms": 1_100},
                    ],
                },
            }
        ],
    }
    store.create_session("legacy", {"question": "How many?"}, state=legacy_state)
    store.expand_targets(
        "legacy",
        (
            {
                "batch_id": "b1",
                "video_id": "Z_VIDEO",
                "source_frame_idx": 20,
                "timestamp_ms": 1_000,
                "window_id": first_key,
            },
            {
                "batch_id": "b1",
                "video_id": "A_VIDEO",
                "source_frame_idx": 10,
                "timestamp_ms": 2_000,
                "window_id": second_key,
            },
            {
                "batch_id": "b1",
                "video_id": "M_VIDEO",
                "source_frame_idx": 30,
                "timestamp_ms": 3_000,
                "window_id": third_key,
            },
        ),
    )
    with sqlite3.connect(path) as connection:
        stored_before = connection.execute(
            "SELECT state_json FROM qa_sessions WHERE session_id = 'legacy'"
        ).fetchone()[0]

    snapshot = store.get_snapshot("legacy")

    assert [value["window_id"] for value in snapshot["window_outcomes"]] == [
        first_key,
        second_key,
        third_key,
    ]
    assert [frame["source_frame_idx"] for frame in snapshot["window_outcomes"][0]["frames"]] == [
        19,
        20,
        21,
    ]
    failure = snapshot["window_outcomes"][1]
    assert failure["status"] == "internal_failed"
    assert failure["failure_code"] == "internal_reasoning_error"
    assert failure["failure_stage"] == "internal"
    assert failure["frames"] == [
        {
            "position": 1,
            "batch_id": "b1",
            "video_id": "A_VIDEO",
            "source_frame_idx": 10,
            "timestamp_ms": 2_000,
        }
    ]
    unknown = snapshot["window_outcomes"][2]
    assert unknown["failure_code"] == "internal_reasoning_error"
    assert unknown["failure_stage"] == "internal"
    assert "unknown-secret-sdk-failure" not in str(snapshot)
    assert snapshot["state"]["state_version"] == 2
    with sqlite3.connect(path) as connection:
        stored_after = connection.execute(
            "SELECT state_json FROM qa_sessions WHERE session_id = 'legacy'"
        ).fetchone()[0]
    assert stored_after == stored_before


class _FakeRetriever:
    max_clusters = 20

    def __init__(self, clusters: tuple[QACandidateCluster, ...]) -> None:
        self.clusters = clusters
        self.events: list[str] = []

    def retrieve(self, event_description, *, batch_filters=(), max_clusters=None):
        self.events.append(event_description)
        return self.clusters[:max_clusters]


class _FakeResolver:
    def resolve(self, frame: FrameRecord):
        from src.retrieval.qa_media import ResolvedCanonicalImage

        return ResolvedCanonicalImage(
            frame.batch_id,
            frame.video_id,
            frame.source_frame_idx,
            frame.timestamp_ms,
            hashlib.sha256(b"jpeg").hexdigest(),
            8,
            8,
            "image/jpeg",
            b"jpeg",
        )


class _FakeReasoner:
    fingerprint = "fake-reasoner-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.image_indices: list[list[int]] = []

    def reason(self, request, images, spec):
        self.calls += 1
        self.image_indices.append([image.index for image in images])
        return QAReasoningResult(
            raw_answer="Màu đỏ",
            normalized_answer="màu đỏ",
            answerable=True,
            evidence_frame_indices=(1,),
            provider="fixture",
            model_name="fixture-v1",
            prompt_version="fixture-prompt-v1",
            latency_ms=3.0,
        )


class _OneAttemptReasoner(_FakeReasoner):
    max_attempts = 1


def test_visual_qa_cascade_resumes_session_and_reuses_reasoning_cache(tmp_path: Path) -> None:
    frames = (
        _frame(str(tmp_path / "unused-9.jpg"), source_frame_idx=9, timestamp_ms=500),
        _frame(str(tmp_path / "unused-10.jpg")),
        _frame(str(tmp_path / "unused-11.jpg"), source_frame_idx=11, timestamp_ms=1_500),
    )
    cluster = QACandidateCluster(
        "b1:L21_V001:10",
        1,
        0.5,
        "b1",
        "L21_V001",
        10,
        1_000,
        "s1",
        frames,
    )
    retriever = _FakeRetriever((cluster,))
    reasoner = _FakeReasoner()
    sessions = QASessionStore(tmp_path / "qa.sqlite3")
    sessions.initialize()
    pipeline = VisualQAPipeline(
        retriever=retriever,
        reasoner=reasoner,
        image_resolver=_FakeResolver(),
        session_store=sessions,
        competition_spec=QACompetitionSpec.conservative_general_text(),
    )
    request = {"question": "Áo màu gì?", "event_description": "người mặc áo", "session_id": "s1"}

    first = pipeline.answer(request)
    second = pipeline.answer(request)

    assert reasoner.calls == 1
    assert reasoner.image_indices == [[1, 2, 3]]
    assert first == second
    assert first[0]["rank"] == 1
    assert [value["source_frame_idx"] for value in first] == [10, 9, 11]
    assert first[0]["answer"] == "màu đỏ"
    assert 0.0 <= first[0]["confidence_score"] <= 1.0
    assert first[1]["confidence_score"] > first[0]["confidence_score"]
    assert first[0]["window_id"] == "b1:L21_V001:10"
    assert first[0]["evidence"]["window_frames"] == [
        {"position": 1, "source_frame_idx": 9, "timestamp_ms": 500},
        {"position": 2, "source_frame_idx": 10, "timestamp_ms": 1_000},
        {"position": 3, "source_frame_idx": 11, "timestamp_ms": 1_500},
    ]
    assert first[0]["evidence"]["evidence_source_frame_indices"] == [9]
    assert retriever.events == ["người mặc áo"]
    snapshot = pipeline.get_search("s1")
    assert snapshot["target_count"] == 1
    assert snapshot["reasoned_count"] == 1
    assert pipeline.expand("s1", 1) == first
    assert reasoner.calls == 1
    third = pipeline.answer(
        {
            "query_id": "different-query-id",
            "question": "Áo màu gì?",
            "event_description": "người mặc áo",
        },
        session_id="s2",
    )
    assert third[0]["answer"] == first[0]["answer"]
    assert reasoner.calls == 1


def test_live_one_window_check_is_bounded_and_does_not_create_a_session(
    tmp_path: Path,
) -> None:
    frames = (
        _frame(str(tmp_path / "unused-9.jpg"), source_frame_idx=9, timestamp_ms=500),
        _frame(str(tmp_path / "unused-10.jpg")),
        _frame(str(tmp_path / "unused-11.jpg"), source_frame_idx=11, timestamp_ms=1_500),
    )
    cluster = QACandidateCluster(
        "b1:L21_V001:10", 1, 0.5, "b1", "L21_V001", 10, 1_000, "s1", frames
    )
    reasoner = _OneAttemptReasoner()
    session_path = tmp_path / "must-not-be-created.sqlite3"
    pipeline = VisualQAPipeline(
        retriever=_FakeRetriever((cluster,)),
        reasoner=reasoner,
        image_resolver=_FakeResolver(),
        session_store=QASessionStore(session_path),
    )
    request = QAQueryRequest(
        query_id="one-window-live",
        event_description="a person wearing a shirt",
        question="What color is the shirt?",
        answer_spec_id=QACompetitionSpec.conservative_general_text().spec_id,
    )

    outcome = pipeline.check_one_window_live(request, reasoner=reasoner)

    assert outcome["status"] == "answered"
    assert outcome["attempts"] == 1
    assert len(outcome["frames"]) == 3
    assert reasoner.calls == 1
    assert not session_path.exists()


def test_visual_qa_inline_cap_persists_only_sanitized_failure(tmp_path: Path) -> None:
    frames = (
        _frame(str(tmp_path / "unused-9.jpg"), source_frame_idx=9, timestamp_ms=500),
        _frame(str(tmp_path / "unused-10.jpg")),
        _frame(str(tmp_path / "unused-11.jpg"), source_frame_idx=11, timestamp_ms=1_500),
    )
    cluster = QACandidateCluster(
        "b1:L21_V001:10", 1, 0.5, "b1", "L21_V001", 10, 1_000, "s1", frames
    )
    sessions = QASessionStore(tmp_path / "qa.sqlite3")
    sessions.initialize()
    pipeline = VisualQAPipeline(
        retriever=_FakeRetriever((cluster,)),
        reasoner=_FakeReasoner(),
        image_resolver=_FakeResolver(),
        session_store=sessions,
        max_inline_image_bytes=5,
    )
    request = {"question": "what?", "event_description": "event", "session_id": "bounded"}

    with pytest.raises(VisualQATotalFailureError) as caught:
        pipeline.search(request)

    assert caught.value.failure_codes == ("media_window_too_large",)
    assert "inline-image" not in str(caught.value)
    snapshot = pipeline.get_search("bounded")
    assert set(snapshot["state"]["failures"].values()) == {"media_window_too_large"}
    assert snapshot["window_outcomes"][0]["status"] == "media_failed"
    assert snapshot["window_outcomes"][0]["failure_stage"] == "media"


def test_visual_qa_rejects_unknown_answer_spec_before_provider_call(tmp_path: Path) -> None:
    sessions = QASessionStore(tmp_path / "qa.sqlite3")
    sessions.initialize()
    reasoner = _FakeReasoner()
    pipeline = VisualQAPipeline(
        retriever=_FakeRetriever(()),
        reasoner=reasoner,
        image_resolver=_FakeResolver(),
        session_store=sessions,
    )

    with pytest.raises(ValueError, match="unknown Visual Q&A answer_spec_id"):
        pipeline.search(
            {
                "question": "what?",
                "event_description": "event",
                "answer_spec_id": "unknown-spec",
            }
        )

    assert reasoner.calls == 0
