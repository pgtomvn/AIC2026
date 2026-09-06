"""Focused offline tests for adaptive, spatial, color, and QA isolation."""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image

from src.api import _public_hits, _safe_frame_neighbor, _safe_query_processing
from src.contracts import FrameRecord
from src.ingest.artifacts import ArtifactRepository
from src.query.query_decomposer import ColorConstraint, SpatialConstraint, decompose_query
from src.query.translator import QueryTextAdapter
from src.retrieval.color_filter import AdvancedKISReranker, HSVColorVerifier
from src.retrieval.fusion import ChannelRanking
from src.retrieval.qa_retrieval import VisualQARetriever
from src.retrieval.spatial_rerank import (
    ObjectDetection,
    ObjectEvidence,
    OrganizerObjectArchive,
    labels_compatible,
    occupied_cells,
    spatial_constraint_score,
)
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline


def _frame(
    source_frame_idx: int = 25,
    *,
    ordinal: int = 1,
    batch_id: str = "aic25-b1-reference",
) -> FrameRecord:
    return FrameRecord(
        batch_id=batch_id,
        video_id="L01_V001",
        keyframe_ordinal=ordinal,
        source_frame_idx=source_frame_idx,
        pts=source_frame_idx,
        time_base="1/25",
        timestamp_ms=source_frame_idx * 40,
        fps_num=25,
        fps_den=1,
        shot_id="shot-1",
        sampling_reasons=("reference",),
        image_uri="memory://frame",
        video_uri="memory://video",
        source_sha256="a" * 64,
        extractor_version="test:v1",
    )


def _jpeg(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(output, format="JPEG", quality=95)
    return output.getvalue()


class _Resolver:
    fingerprint = "resolver:test"

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls = 0

    def resolve(self, frame: FrameRecord) -> SimpleNamespace:
        del frame
        self.calls += 1
        return SimpleNamespace(data=self.data)


def test_query_intent_precedence_and_bound_constraints() -> None:
    spatial = decompose_query(
        "a red car on the left",
        (),
        english_text="a red car on the left",
    )
    assert spatial.intent == "spatial_color"
    assert [(value.zone, value.bound) for value in spatial.spatial_constraints] == [
        ("left", True)
    ]
    assert [(value.color, value.subject) for value in spatial.color_constraints] == [
        ("red", "car")
    ]

    lexical = decompose_query(
        'the banner says "FANA" and number 12',
        (),
        english_text='the banner says "FANA" and number 12',
    )
    assert lexical.intent == "lexical_dominant"

    unbound = decompose_query("on the right", (), english_text="on the right")
    assert unbound.intent == "visual_dense"
    assert len(unbound.spatial_constraints) == 1
    assert unbound.spatial_constraints[0].bound is False


@pytest.mark.parametrize(
    ("text", "zone"),
    (
        ("một người ở bên trái", "left"),
        ("a person on the right", "right"),
        ("a person in the center", "center"),
        ("a bird at the top", "top"),
        ("một chiếc xe ở phía dưới", "bottom"),
    ),
)
def test_bilingual_spatial_zones(text: str, zone: str) -> None:
    value = decompose_query(text, (), english_text=text)
    assert any(item.zone == zone and item.bound for item in value.spatial_constraints)


def test_occupancy_and_spatial_score_are_deterministic() -> None:
    detection = ObjectDetection("person", 0.9, (0.0, 0.0, 1.0, 3 / 7))
    cells = occupied_cells(detection.box)
    assert len(cells) == 21
    score, matching = spatial_constraint_score(
        SpatialConstraint("người đàn ông", "left", "bên trái"),
        (detection,),
    )
    assert score == pytest.approx(1.0)
    assert matching == (detection,)
    missing, matches = spatial_constraint_score(
        SpatialConstraint("dog", "right", "right"),
        (detection,),
    )
    assert missing is None
    assert matches == ()


def test_object_archive_is_lazy_bounded_and_thresholded(tmp_path) -> None:
    archive_path = tmp_path / "objects.zip"
    payload = {
        "detection_scores": ["0.50", "0.49"],
        "detection_class_entities": ["person", "dog"],
        "detection_boxes": [[0.0, 0.0, 1.0, 0.4], [0.0, 0.6, 1.0, 1.0]],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("objects/L01_V001/001.json", json.dumps(payload))
    reader = OrganizerObjectArchive(
        archive_path,
        allowed_root=tmp_path,
        max_entries=1,
        max_cache_bytes=4096,
    )
    assert reader.diagnostics["archive_state"] == "available"
    first = reader.get(_frame())
    assert first.state == "ready"
    assert [value.label for value in first.detections] == ["person"]
    second = reader.get(_frame())
    assert second.cache_hit is True
    assert reader.diagnostics["cache_entries"] == 1
    assert reader.get(_frame(50, ordinal=2)).state == "member_missing"
    reader.close()


def test_hsv_red_wraparound_and_descriptor_cache() -> None:
    resolver = _Resolver(_jpeg((255, 0, 0)))
    verifier = HSVColorVerifier(resolver, max_entries=2, max_cache_bytes=1024)
    ratio, cached = verifier.ratio(_frame(), "red", box=None)
    assert ratio > 0.95
    assert cached is False
    second, cached = verifier.ratio(_frame(), "red", box=None)
    assert second == pytest.approx(ratio)
    assert cached is True
    assert resolver.calls == 1


class _Repository:
    def __init__(self, frame: FrameRecord) -> None:
        self.frame = frame

    def get_frame(self, batch_id: str, video_id: str, source_frame_idx: int):
        if (batch_id, video_id, source_frame_idx) == (
            self.frame.batch_id,
            self.frame.video_id,
            self.frame.source_frame_idx,
        ):
            return self.frame
        return None


class _Objects:
    fingerprint = "objects:test"
    diagnostics = {"archive_state": "ready"}

    def get(self, frame: FrameRecord) -> ObjectEvidence:
        del frame
        return ObjectEvidence(
            "ready",
            (ObjectDetection("car", 0.9, (0.1, 0.1, 0.9, 0.9)),),
        )


def test_automatic_color_mismatch_penalty_is_applied_only_once() -> None:
    frame = _frame()
    verifier = HSVColorVerifier(_Resolver(_jpeg((255, 0, 0))))
    reranker = AdvancedKISReranker(
        _Repository(frame),
        _Objects(),  # type: ignore[arg-type]
        verifier,
        color_absent_multiplier=0.65,
    )
    processing = QueryTextAdapter().prepare("a blue and green car")
    assert processing.decomposition is not None
    assert {
        value.color for value in processing.decomposition.color_constraints
    } == {"blue", "green"}
    output = reranker(
        processing,
        [
            {
                "batch_id": frame.batch_id,
                "video_id": frame.video_id,
                "source_frame_idx": frame.source_frame_idx,
                "score": 1.0,
            }
        ],
    )
    assert output[0]["score"] == pytest.approx(0.10)
    assert output[0]["color_penalty_applied"] is True


def test_adaptive_rrf_preserves_group_ratios_without_mutation() -> None:
    dense = DenseChannel("visual", object(), object(), weight=2.0)
    lexical = LexicalChannel("metadata", object(), weight=1.0)
    pipeline = TextualKISPipeline(dense_channels=(dense,), lexical_channels=(lexical,))
    rankings = [
        ChannelRanking("visual", ({"video_id": "V1", "source_frame_idx": 1},), 2.0),
        ChannelRanking("metadata", ({"video_id": "V2", "source_frame_idx": 2},), 1.0),
    ]

    visual = QueryTextAdapter().prepare("a person walking")
    weighted, effective_k, weights = pipeline._apply_adaptive_weights(rankings, visual)
    assert effective_k == 60
    assert weights == pytest.approx({"visual": 0.70, "metadata": 0.30})
    assert [value.weight for value in rankings] == [2.0, 1.0]
    assert [value.weight for value in weighted] == pytest.approx([0.70, 0.30])

    lexical_processing = QueryTextAdapter().prepare("the sign says 12")
    _weighted, effective_k, weights = pipeline._apply_adaptive_weights(
        rankings,
        lexical_processing,
    )
    assert effective_k == 40
    assert weights == pytest.approx({"visual": 0.25, "metadata": 0.75})


class _QASearch:
    def __init__(self, frame: FrameRecord) -> None:
        self.frame = frame
        self.kwargs = None

    def search(self, request, **kwargs):
        del request
        self.kwargs = kwargs
        return [
            {
                "batch_id": self.frame.batch_id,
                "video_id": self.frame.video_id,
                "source_frame_idx": self.frame.source_frame_idx,
                "timestamp_ms": self.frame.timestamp_ms,
                "shot_id": self.frame.shot_id,
                "score": 1.0,
            }
        ]


def test_qa_stage_one_explicitly_disables_advanced_media_rerank() -> None:
    frame = _frame()
    pipeline = _QASearch(frame)
    retriever = VisualQARetriever(pipeline, _Repository(frame))
    candidates = retriever.retrieve_frames(
        "a red car on the left",
        batch_filters=(frame.batch_id,),
        top_k=1,
    )
    assert len(candidates) == 1
    assert pipeline.kwargs["refine_raw"] is False
    assert pipeline.kwargs["advanced_rerank"] is False



def test_canonical_adjacency_uses_ordinals_and_deduplicates_aliases(tmp_path) -> None:
    repository = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    repository.initialize()
    first = _frame(25, ordinal=1, batch_id="b2")
    middle = FrameRecord(
        **{
            **first.to_dict(),
            "keyframe_ordinal": 3,
            "source_frame_idx": 50,
            "pts": 50,
            "timestamp_ms": 2_000,
        }
    )
    last = FrameRecord(
        **{
            **first.to_dict(),
            "keyframe_ordinal": 4,
            "source_frame_idx": 75,
            "pts": 75,
            "timestamp_ms": 3_000,
        }
    )
    repository.upsert_frames((first, middle, last))
    repository.upsert_frame_mappings(
        "b2",
        "L01_V001",
        {1: 25, 2: 25, 3: 50, 4: 75},
    )

    from_first = repository.adjacent_frames("b2", "L01_V001", 25, radius=3)
    assert [(offset, frame.source_frame_idx) for offset, frame in from_first] == [
        (0, 25),
        (2, 50),
        (3, 75),
    ]
    from_middle = repository.adjacent_frames("b2", "L01_V001", 50, radius=2)
    assert [(offset, frame.source_frame_idx) for offset, frame in from_middle] == [
        (-1, 25),
        (0, 50),
        (1, 75),
    ]

def test_label_matching_uses_phrase_boundaries() -> None:
    assert labels_compatible("red car", "car")
    assert not labels_compatible("patterned carpet", "car")


def test_advanced_reranker_records_deterministic_rank_delta() -> None:
    frame = _frame()
    verifier = HSVColorVerifier(_Resolver(_jpeg((255, 0, 0))))
    reranker = AdvancedKISReranker(
        _Repository(frame),
        _Objects(),  # type: ignore[arg-type]
        verifier,
        color_absent_multiplier=0.65,
    )
    processing = QueryTextAdapter().prepare("a blue car")
    output = reranker(
        processing,
        [
            {
                "batch_id": frame.batch_id,
                "video_id": frame.video_id,
                "source_frame_idx": frame.source_frame_idx,
                "score": 1.0,
            },
            {
                "batch_id": frame.batch_id,
                "video_id": "L01_V002",
                "source_frame_idx": 50,
                "score": 0.9,
            },
        ],
    )
    assert [item["video_id"] for item in output] == ["L01_V002", "L01_V001"]
    assert [item["advanced_rank_delta"] for item in output] == [1, -1]


def test_public_advanced_diagnostics_are_sanitized() -> None:
    processing = _safe_query_processing(
        {
            "intent": "spatial_color",
            "advanced_rerank": {
                "state": "ready",
                "spatial_candidates": 100,
                "color_candidates": 12,
                "object_cache": {
                    "archive_state": "ready",
                    "archive_members": 177321,
                    "cache_hits": 2,
                    "archive_open_ms": 3.2,
                    "archive_path": "D:/secret/objects.zip",
                },
                "color_cache": {"cache_hits": 1, "cache_entries": 1},
            },
        }
    )
    assert processing["advanced_rerank"]["object_cache"] == {
        "archive_state": "ready",
        "archive_members": 177321,
        "cache_hits": 2,
        "archive_open_ms": 3.2,
    }
    assert "archive_path" not in json.dumps(processing)

    public = _public_hits(
        [
            {
                "batch_id": "b2",
                "video_id": "L01_V001",
                "source_frame_idx": 25,
                "timestamp_ms": 1000,
                "score": 0.5,
                "advanced_rank_delta": 3,
            }
        ]
    )
    assert public[0]["advanced_rank_delta"] == 3


def test_neighbor_projection_never_exposes_media_paths() -> None:
    value = _safe_frame_neighbor(_frame(), offset=-1)
    assert value["offset"] == -1
    assert value["thumbnail_url"].endswith("/L01_V001/25/image")
    rendered = json.dumps(value)
    assert "image_uri" not in rendered
    assert "video_uri" not in rendered
    assert "memory://" not in rendered
