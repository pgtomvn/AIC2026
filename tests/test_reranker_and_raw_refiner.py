"""Offline unit tests for reranking and local raw-frame refinement."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image

from src.contracts import FrameRecord
from src.ingest.artifacts import ArtifactRepository
from src.retrieval.query_plan import build_query_plan
from src.retrieval.raw_refiner import (
    LocalFrameRefiner,
    PersistentRefinedFrameSink,
    _source_from_uri,
)
from src.retrieval.reranker import Reranker


class FakeScratch:
    """Scratch fake that never reads or copies the supplied media path."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sources = []

    @contextmanager
    def materialize(self, source) -> Iterator[Path]:
        self.sources.append(source)
        yield source.path


class FakeEncoder:
    def __init__(
        self,
        query_vector: Sequence[float],
        image_vectors: Sequence[Sequence[float]],
    ) -> None:
        self.query_vector = np.asarray(query_vector, dtype=np.float32)
        self.image_vectors = np.asarray(image_vectors, dtype=np.float32)
        self.text_calls: list[str] = []
        self.image_calls: list[list[Path]] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.text_calls.append(text)
        return self.query_vector.copy()

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        self.image_calls.append(list(paths))
        return self.image_vectors.copy()


class StubFrameRefiner(LocalFrameRefiner):
    def __init__(self, *args, decoded_frames: Sequence[dict[str, Any]], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoded_frames = [dict(frame) for frame in decoded_frames]

    def _decode(self, media_path: Path, center_ms: int, output_dir: Path, **_kwargs):
        return [dict(frame) for frame in self.decoded_frames]


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    def persist(self, candidate, sample, *, refiner_fingerprint):
        self.calls.append((dict(candidate), dict(sample), refiner_fingerprint))
        return SimpleNamespace(
            source_frame_idx=sample["source_frame_idx"],
            timestamp_ms=sample["timestamp_ms"],
            pts=sample.get("pts", sample["timestamp_ms"]),
            time_base=sample.get("time_base", "1/1000"),
            image_uri=str(sample["path"]),
        )


def _candidate(frame: int, *, evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "batch_id": "b2",
        "video_id": "L21_V001",
        "source_frame_idx": frame,
        "timestamp_ms": 2000,
        "pts": 2000,
        "time_base": "1/1000",
        "sampling_reasons": ["scene"],
        "shot_id": "shot-1",
        "fps_num": 25,
        "fps_den": 1,
        "source_sha256": "a" * 64,
        "score": 0.5,
        "video_uri": "unused-video.mp4",
        "evidence": [] if evidence is None else evidence,
    }


def _anchor(frame: int = 41) -> FrameRecord:
    return FrameRecord(
        batch_id="b2",
        video_id="L21_V001",
        keyframe_ordinal=1,
        source_frame_idx=frame,
        pts=2000,
        time_base="1/1000",
        timestamp_ms=2000,
        fps_num=25,
        fps_den=1,
        shot_id="shot-1",
        sampling_reasons=("scene",),
        image_uri="coarse.webp",
        video_uri="unused-video.mp4",
        source_sha256="a" * 64,
        extractor_version="extract:v1",
    )


def test_injected_reranker_scores_copies_and_uses_deterministic_ties() -> None:
    original = [
        {"video_id": "L21_V002", "source_frame_idx": 20, "score": 0.8},
        {"video_id": "L21_V001", "source_frame_idx": 10, "score": 0.7},
        {"video_id": "L21_V003", "source_frame_idx": 30, "score": 0.6},
    ]
    seen: list[tuple[str, list[dict[str, Any]]]] = []

    def score(query: str, candidates: list[dict[str, Any]]) -> list[float]:
        seen.append((query, candidates))
        return [0.2, 0.9, 0.9]

    reranked = Reranker(device="cpu", scorer=score).rerank("query", original, top_k=2)

    assert [(item["video_id"], item["source_frame_idx"]) for item in reranked] == [
        ("L21_V001", 10),
        ("L21_V003", 30),
    ]
    assert reranked[0]["retrieval_score"] == 0.7
    assert reranked[0]["score"] == 0.9
    assert seen[0][0] == "query"
    assert original[1] == {"video_id": "L21_V001", "source_frame_idx": 10, "score": 0.7}


def test_injected_reranker_rejects_wrong_score_count() -> None:
    reranker = Reranker(device="cpu", scorer=lambda _query, _items: [1.0])

    with pytest.raises(ValueError, match="wrong number"):
        reranker.rerank("query", [{"score": 1.0}, {"score": 0.5}])


def test_mmr_prefers_a_diverse_candidate_after_the_best_match() -> None:
    candidates = [
        {"video_id": "L21_V001", "source_frame_idx": 1, "score": 1.0},
        {"video_id": "L21_V001", "source_frame_idx": 2, "score": 0.9},
        {"video_id": "L21_V002", "source_frame_idx": 3, "score": 0.8},
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    reranked = Reranker(device="cpu").mmr_rerank(
        candidates,
        embeddings,
        lambda_param=0.5,
        top_k=2,
    )

    assert [item["source_frame_idx"] for item in reranked] == [1, 3]


@pytest.mark.parametrize("lambda_param", [-0.01, 1.01])
def test_mmr_rejects_out_of_range_lambda(lambda_param: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        Reranker(device="cpu").mmr_rerank([], np.empty((0, 2)), lambda_param=lambda_param)


def test_mmr_rejects_embedding_row_mismatch() -> None:
    with pytest.raises(ValueError, match="one row per candidate"):
        Reranker(device="cpu").mmr_rerank(
            [{"score": 1.0}],
            np.empty((2, 3), dtype=np.float32),
        )


def test_source_from_uri_parses_loose_and_zip_video_sources(tmp_path: Path) -> None:
    loose = tmp_path / "L21_V001.mp4"
    loose.write_bytes(b"video")
    archive = tmp_path / "videos.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as handle:
        handle.writestr("nested/L21_V002.mp4", b"zipped-video")

    loose_source = _source_from_uri("L21_V001", str(loose))
    zip_source = _source_from_uri(
        "L21_V002",
        f"zip://{archive}!/nested/L21_V002.mp4",
    )

    assert loose_source.video_id == "L21_V001"
    assert loose_source.path == loose
    assert loose_source.member is None
    assert zip_source.path == archive
    assert zip_source.member == "nested/L21_V002.mp4"
    assert zip_source.size_bytes == len(b"zipped-video")


def test_raw_decode_uses_pts_anchor_for_exact_vfr_source_ordinals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeFrame:
        def __init__(self, pts: int) -> None:
            self.pts = pts

        def to_image(self) -> Image.Image:
            return Image.new("RGB", (8, 8), color=(self.pts % 255, 0, 0))

    class FakeContainer:
        def __init__(self) -> None:
            stream = SimpleNamespace(
                time_base=Fraction(1, 1000),
                average_rate=Fraction(25, 1),
                guessed_rate=Fraction(25, 1),
            )
            self.streams = SimpleNamespace(video=[stream])
            self.seek_calls: list[tuple[int, Any]] = []
            self.closed = False

        def seek(self, offset: int, **kwargs: Any) -> None:
            self.seek_calls.append((offset, kwargs))

        def decode(self, _stream: Any):
            # Irregular PTS gaps demonstrate that ordinals come from decoded
            # order relative to the persisted anchor, not nominal FPS math.
            for pts in (1800, 1840, 1880, 1940, 2000, 2070, 2110, 2200):
                yield FakeFrame(pts)

        def close(self) -> None:
            self.closed = True

    container = FakeContainer()
    monkeypatch.setitem(
        __import__("sys").modules,
        "av",
        SimpleNamespace(open=lambda _path: container),
    )
    refiner = LocalFrameRefiner(
        FakeEncoder((1.0, 0.0), ((1.0, 0.0),)),
        FakeScratch(tmp_path / "scratch"),
        radius_seconds=0.2,
        sample_fps=5,
    )
    output = tmp_path / "frames"
    output.mkdir()

    frames = refiner._decode(
        tmp_path / "fixture.mp4",
        2000,
        output,
        anchor_source_frame_idx=100,
        anchor_pts=2000,
        anchor_time_base="1/1000",
        fps_num=25,
        fps_den=1,
    )

    assert [frame["source_frame_idx"] for frame in frames] == [96, 100, 103]
    assert [frame["timestamp_ms"] for frame in frames] == [1800, 2000, 2200]
    assert all(Path(frame["path"]).is_file() for frame in frames)
    assert container.closed

    with pytest.raises(ValueError, match="exactly align"):
        refiner._decode(
            tmp_path / "fixture.mp4",
            2000,
            output,
            anchor_source_frame_idx=100,
            anchor_pts=1999,
            anchor_time_base="1/1000",
        )


def test_raw_refiner_selects_best_exact_source_frame_without_mutating_input(
    tmp_path: Path,
) -> None:
    scratch = FakeScratch(tmp_path / "scratch")
    encoder = FakeEncoder((1.0, 0.0), ((0.0, 1.0), (1.0, 0.0)))
    sink = RecordingSink()
    decoded = [
        {
            "source_frame_idx": 40,
            "timestamp_ms": 1900,
            "pts": 1900,
            "time_base": "1/1000",
            "path": tmp_path / "40.jpg",
        },
        {
            "source_frame_idx": 42,
            "timestamp_ms": 2100,
            "pts": 2100,
            "time_base": "1/1000",
            "path": tmp_path / "42.jpg",
        },
    ]
    refiner = StubFrameRefiner(
        encoder,
        scratch,
        decoded_frames=decoded,
        radius_seconds=4,
        sample_fps=6,
        top_windows=1,
        frame_sink=sink,
    )
    original_evidence = [{"channel": "visual", "rank": 1}]
    candidate = _candidate(41, evidence=original_evidence)

    refined = refiner(build_query_plan("người áo đỏ"), [candidate])

    assert len(refined) == 1
    assert refined[0]["coarse_source_frame_idx"] == 41
    assert refined[0]["source_frame_idx"] == 42
    assert refined[0]["timestamp_ms"] == 2100
    assert refined[0]["refinement_score"] == pytest.approx(1.0)
    assert refined[0]["evidence"][-1]["channel"] == "raw_refinement"
    assert len(sink.calls) == 1
    assert sink.calls[0][1]["source_frame_idx"] == 42
    assert candidate["source_frame_idx"] == 41
    assert candidate["evidence"] == [{"channel": "visual", "rank": 1}]


def test_raw_refiner_without_a_durable_sink_returns_only_coarse_hits(
    tmp_path: Path,
) -> None:
    encoder = FakeEncoder((1.0, 0.0), ((1.0, 0.0),))
    refiner = StubFrameRefiner(
        encoder,
        FakeScratch(tmp_path / "scratch"),
        decoded_frames=(
            {
                "source_frame_idx": 42,
                "timestamp_ms": 2100,
                "pts": 2100,
                "time_base": "1/1000",
                "path": tmp_path / "42.jpg",
            },
        ),
    )
    candidate = _candidate(41)

    assert refiner(build_query_plan("query"), [candidate]) == [candidate]
    assert encoder.text_calls == []
    assert encoder.image_calls == []


def test_persistent_sink_keeps_bounded_webp_and_metadata_across_restart(
    tmp_path: Path,
) -> None:
    repository = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    repository.initialize()
    repository.upsert_frame(_anchor())
    sink = PersistentRefinedFrameSink(
        repository,
        tmp_path / "refined",
        max_dimension=64,
        max_total_bytes=1024 * 1024,
        reserve_bytes=0,
    )
    selected = tmp_path / "selected.jpg"
    Image.new("RGB", (320, 160), color=(200, 10, 10)).save(selected)
    sample = {
        "source_frame_idx": 42,
        "timestamp_ms": 2100,
        "pts": 2100,
        "time_base": "1/1000",
        "path": selected,
    }

    record = sink.persist(_candidate(41), sample, refiner_fingerprint="f" * 64)

    assert record.source_frame_idx == 42
    assert repository.batch_counts("b2").frame_count == 1
    assert repository.refined_frame_count("b2") == 1
    image_path = Path(record.image_uri)
    assert image_path.is_file()
    with Image.open(image_path) as image:
        assert image.format == "WEBP"
        assert max(image.size) <= 64
    assert "keyframe_ordinal" not in record.to_dict()

    restarted = ArtifactRepository(repository.path)
    restarted.initialize()
    restored = restarted.get_source_frame("b2", "L21_V001", 42)
    assert restored == record
    assert Path(restored.image_uri).read_bytes() == image_path.read_bytes()


def test_persistent_sink_limit_failure_does_not_commit_or_return_exact_hit(
    tmp_path: Path,
) -> None:
    repository = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    repository.initialize()
    repository.upsert_frame(_anchor())
    selected = tmp_path / "selected.jpg"
    Image.new("RGB", (64, 64), color=(20, 30, 40)).save(selected)
    sink = PersistentRefinedFrameSink(
        repository,
        tmp_path / "refined",
        max_total_bytes=1,
        reserve_bytes=0,
    )
    refiner = StubFrameRefiner(
        FakeEncoder((1.0, 0.0), ((1.0, 0.0),)),
        FakeScratch(tmp_path / "scratch"),
        decoded_frames=(
            {
                "source_frame_idx": 42,
                "timestamp_ms": 2100,
                "pts": 2100,
                "time_base": "1/1000",
                "path": selected,
            },
        ),
        top_windows=1,
        frame_sink=sink,
    )
    candidate = _candidate(41)

    assert refiner(build_query_plan("query"), [candidate]) == [candidate]
    assert repository.refined_frame_count("b2") == 0
    assert not (tmp_path / "refined").exists()


def test_raw_refiner_falls_back_when_a_zip_archive_is_unavailable(tmp_path: Path) -> None:
    scratch = FakeScratch(tmp_path / "scratch")
    encoder = FakeEncoder((1.0, 0.0), ((1.0, 0.0),))
    refiner = StubFrameRefiner(
        encoder,
        scratch,
        decoded_frames=(),
        top_windows=1,
    )
    candidate = _candidate(41)
    candidate["video_uri"] = f"zip://{tmp_path / 'missing.zip'}!/L21_V001.mp4"

    refined = refiner(build_query_plan("query"), [candidate])

    assert refined == [candidate]


def test_raw_refiner_preserves_candidates_beyond_refinement_budget(tmp_path: Path) -> None:
    scratch = FakeScratch(tmp_path / "scratch")
    encoder = FakeEncoder((1.0, 0.0), ((1.0, 0.0),))
    refiner = StubFrameRefiner(
        encoder,
        scratch,
        decoded_frames=(),
        top_windows=1,
    )
    candidates = [_candidate(41), _candidate(82)]
    for candidate in candidates:
        candidate.pop("video_uri")

    refined = refiner(build_query_plan("query"), candidates)

    assert refined == candidates


def test_raw_refiner_never_uses_estimated_pts_as_an_exact_anchor(tmp_path: Path) -> None:
    scratch = FakeScratch(tmp_path / "scratch")
    encoder = FakeEncoder((1.0, 0.0), ((1.0, 0.0),))
    refiner = StubFrameRefiner(
        encoder,
        scratch,
        decoded_frames=(
            {"source_frame_idx": 42, "timestamp_ms": 2100, "path": tmp_path / "42.jpg"},
        ),
        top_windows=1,
    )
    candidate = _candidate(41)
    candidate["sampling_reasons"] = ["scene", "estimated_pts"]

    assert refiner(build_query_plan("query"), [candidate]) == [candidate]
    assert encoder.image_calls == []


@pytest.mark.parametrize(
    "settings",
    [
        {"radius_seconds": 0},
        {"sample_fps": 0},
        {"top_windows": 0},
    ],
)
def test_raw_refiner_rejects_non_positive_settings(
    tmp_path: Path,
    settings: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LocalFrameRefiner(
            FakeEncoder((1.0, 0.0), ((1.0, 0.0),)),
            FakeScratch(tmp_path),
            **settings,
        )
