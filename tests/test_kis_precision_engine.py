"""Offline quality gates for Visione-informed Advanced Textual KIS."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.cache_kis_channel_rankings import main as cache_rankings_main
from scripts.lock_kis_qrels import main as lock_qrels_main
from src.contracts import FrameRecord, QueryRequest
from src.evaluation.qrels import (
    IMPORTANT_INTENTS,
    QrelsManifest,
    QrelsPack,
    advanced_activation_gate,
    build_locked_qrels_split,
    load_qrels_pack,
    validate_production_qrels,
)
from src.evaluation.textual_kis import GroundTruthInterval, GroundTruthQuery
from src.query.query_decomposer import RelativeSpatialConstraint, decompose_query
from src.query.translator import QueryTextAdapter
from src.retrieval.calibrator import CalibrationWeights, blend_components, calibrate_scores
from src.retrieval.color_filter import AdvancedKISReranker
from src.retrieval.spatial_graph import class_aware_nms, match_spatial_graph, relation_satisfied
from src.retrieval.spatial_rerank import ObjectDetection, ObjectEvidence
from src.retrieval.textual_kis import DenseChannel, TextualKISPipeline


def _constraint(relation: str) -> RelativeSpatialConstraint:
    return RelativeSpatialConstraint("person", relation, "car", relation, "a", "b")


def test_negation_is_parsed_before_positive_color_and_removed_from_dense() -> None:
    value = decompose_query(
        "a yellow cyclist left of a blue car without a red bus",
        (),
        english_text="a yellow cyclist left of a blue car without a red bus",
    )
    assert [(item.subject, item.color) for item in value.negation_constraints] == [("bus", "red")]
    assert {(item.subject, item.color) for item in value.color_constraints} == {
        ("cyclist", "yellow"),
        ("car", "blue"),
    }
    assert "red bus" not in (value.visual_description_en or "").casefold()
    assert value.relative_constraints[0].relation == "left_of"


@pytest.mark.parametrize(
    ("relation", "subject_box", "object_box"),
    (
        ("left_of", (0.2, 0.1, 0.4, 0.3), (0.2, 0.7, 0.4, 0.9)),
        ("right_of", (0.2, 0.7, 0.4, 0.9), (0.2, 0.1, 0.4, 0.3)),
        ("above", (0.1, 0.2, 0.3, 0.4), (0.7, 0.2, 0.9, 0.4)),
        ("below", (0.7, 0.2, 0.9, 0.4), (0.1, 0.2, 0.3, 0.4)),
        ("inside", (0.2, 0.2, 0.4, 0.4), (0.1, 0.1, 0.5, 0.5)),
        ("contains", (0.1, 0.1, 0.5, 0.5), (0.2, 0.2, 0.4, 0.4)),
    ),
)
def test_supported_relation_geometry(relation: str, subject_box, object_box) -> None:
    subject = ObjectDetection("person", 0.9, subject_box)
    object_detection = ObjectDetection("car", 0.9, object_box)
    assert relation_satisfied(_constraint(relation), subject, object_detection)


def test_relation_margin_and_inside_coverage_are_strict() -> None:
    close_left = ObjectDetection("person", 0.9, (0.1, 0.40, 0.3, 0.50))
    close_right = ObjectDetection("car", 0.9, (0.1, 0.42, 0.3, 0.52))
    assert not relation_satisfied(_constraint("left_of"), close_left, close_right)
    mostly_outside = ObjectDetection("person", 0.9, (0.0, 0.0, 0.5, 0.5))
    container = ObjectDetection("car", 0.9, (0.1, 0.1, 0.5, 0.5))
    assert not relation_satisfied(_constraint("inside"), mostly_outside, container)


def test_nms_and_same_class_assignment_use_distinct_detections() -> None:
    duplicate = (
        ObjectDetection("person", 0.95, (0.1, 0.1, 0.3, 0.3)),
        ObjectDetection("person", 0.90, (0.11, 0.11, 0.31, 0.31)),
        ObjectDetection("person", 0.85, (0.1, 0.7, 0.3, 0.9)),
    )
    assert len(class_aware_nms(duplicate)) == 2
    graph = match_spatial_graph(
        (RelativeSpatialConstraint("person", "left_of", "person", "", "left", "right"),),
        duplicate,
    )
    assert graph.state == "verified"
    assert graph.score == 1.0
    assert len({item.detection.box for item in graph.assignments}) == 2


def test_near_and_behind_are_supported_relations() -> None:
    value = decompose_query(
        "a person near a car and a person behind a bus", (), english_text="same"
    )
    assert {item.relation for item in value.relative_constraints} == {"near", "behind"}
    assert all(item.bound for item in value.relative_constraints)
    assert not value.unsupported_relations

def test_finite_safe_calibration_is_neutral_for_invalid_and_constant_pools() -> None:
    assert calibrate_scores([2.0, 2.0, np.nan, np.inf]).tolist() == pytest.approx([0.5] * 4)
    output = blend_components(
        {"base": [1.0, 0.0], "global_image": [np.nan, 0.8], "crop": [0.2, 0.9]},
        method="rank",
        weights=CalibrationWeights(),
    )
    assert np.isfinite(output).all()
    assert ((0.0 <= output) & (output <= 1.0)).all()


def test_query_request_precision_profile_validation_and_roundtrip() -> None:
    request = QueryRequest("q", "a person left of a car", precision_profile="advanced")
    assert QueryRequest.from_dict(request.to_dict()).precision_profile == "advanced"
    with pytest.raises(ValueError, match="precision_profile"):
        QueryRequest("q", "text", precision_profile="turbo")
    with pytest.raises(TypeError, match="precision_profile"):
        QueryRequest("q", "text", precision_profile=1)  # type: ignore[arg-type]


class _Encoder:
    fingerprint = "encoder:test"

    def encode_text(self, text):
        del text
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _Store:
    def __init__(self):
        self.depths = []

    def search(self, vector, *, top_k, filters):
        del vector, filters
        self.depths.append(top_k)
        return [{"video_id": "V1", "source_frame_idx": 1, "timestamp_ms": 0, "score": 1.0}]


class _Tier2:
    fingerprint = "tier2:test"
    last_diagnostics = {"state": "ready"}

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, processing, candidates, *, precision_profile="standard"):
        del processing
        self.calls.append(("tier2", precision_profile))
        return candidates


def test_automatic_constraint_order_and_channel_depth() -> None:
    calls, store = [], _Store()

    def raw(plan, candidates, **kwargs):
        del plan, kwargs
        calls.append(("raw", None))
        return candidates

    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", _Encoder(), store, depth=20),),
        temporal_windows_ms=(),
        raw_refiner=raw,
        advanced_reranker=_Tier2(calls),
    )
    standard = QueryRequest("s", "person", precision_profile="standard")
    pipeline.search(standard, refine_raw=True)
    assert calls == [("raw", None)]
    assert store.depths[-1] == 20

    for profile in ("standard", "advanced", "experimental_cross_modal"):
        calls.clear()
        constrained = QueryRequest(
            profile,
            "a person left of a car",
            precision_profile=profile,
        )
        pipeline.search(constrained, refine_raw=True, advanced_rerank=False)
        assert calls == [("tier2", "automatic"), ("raw", None)]
        assert store.depths[-1] == 300
        assert pipeline.last_query_processing["precision_profile"] == "automatic"


def test_experimental_profile_does_not_trigger_tier3() -> None:
    store = _Store()
    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", _Encoder(), store),),
        temporal_windows_ms=(),
        advanced_reranker=_Tier2([]),
    )
    output = pipeline.search(
        QueryRequest("q", "a person left of a car", precision_profile="experimental_cross_modal")
    )
    assert output[0]["video_id"] == "V1"
    assert "cross_modal_rerank" not in pipeline.last_channel_statuses
    assert pipeline.last_query_processing["precision_profile"] == "automatic"


class _FrameRepository:
    frame = FrameRecord(
        batch_id="aic25-b1-reference",
        video_id="L01_V001",
        keyframe_ordinal=1,
        source_frame_idx=25,
        pts=25,
        time_base="1/25",
        timestamp_ms=1000,
        fps_num=25,
        fps_den=1,
        shot_id="shot",
        sampling_reasons=("reference",),
        image_uri="memory://image",
        video_uri="memory://video",
        source_sha256="a" * 64,
        extractor_version="test:v1",
    )

    def get_frame(self, batch_id, video_id, source_frame_idx):
        return (
            self.frame
            if (batch_id, video_id, source_frame_idx)
            == (self.frame.batch_id, self.frame.video_id, self.frame.source_frame_idx)
            else None
        )


class _ObjectArchive:
    fingerprint = "objects:test"
    diagnostics = {"archive_state": "ready"}

    def get(self, frame):
        del frame
        return ObjectEvidence("ready", (ObjectDetection("car", 0.9, (0.1, 0.1, 0.9, 0.9)),))


class _ColorVerifier:
    fingerprint = "color:test"
    diagnostics = {"cache_hits": 0, "cache_misses": 1}

    def ratio(self, frame, color, *, box=None):
        del frame, color, box
        return 0.0, False


def _candidate() -> dict:
    frame = _FrameRepository.frame
    return {
        "batch_id": frame.batch_id,
        "video_id": frame.video_id,
        "source_frame_idx": frame.source_frame_idx,
        "score": 1.0,
    }


def test_strict_color_mismatch_and_soft_negation_each_penalize_once() -> None:
    reranker = AdvancedKISReranker(
        _FrameRepository(),
        _ObjectArchive(),
        _ColorVerifier(),
        strict_mismatch_multiplier=0.10,
        negation_multiplier=0.10,
    )
    color = reranker(
        QueryTextAdapter().prepare("a blue car"),
        [_candidate()],
        precision_profile="advanced",
    )
    assert color[0]["score"] == pytest.approx(0.10)
    assert color[0]["color_penalty_applied"] is True
    negative = reranker(
        QueryTextAdapter().prepare("a street without a car"),
        [_candidate()],
        precision_profile="advanced",
    )
    assert negative[0]["score"] == pytest.approx(0.10)
    assert negative[0]["negation_penalty_applied"] is True


def test_qrels_pack_hash_and_activation_gate(tmp_path) -> None:
    text = "a person left of a car"
    record = {
        "query_id": "q1",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "intent": "relative_spatial",
        "video_group": "g1",
        "intervals": [{"video_id": "V1", "start_frame_idx": 1, "end_frame_idx": 2}],
    }
    payload = (json.dumps(record) + "\n").encode()
    (tmp_path / "qrels.jsonl").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "spec_id": "test:v1",
        "corpus_fingerprint": "corpus",
        "index_fingerprint": "index",
        "model_fingerprint": "model",
        "content_file": "qrels.jsonl",
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "split_metadata": {"method": "video_group", "seed": 7},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    pack = load_qrels_pack(path)
    assert pack.queries[0].intervals[0].video_id == "V1"
    with pytest.raises(ValueError, match="at least 120"):
        validate_production_qrels(pack)
    baseline = {
        "recall_at": {"1": 0.5, "5": 0.7},
        "mean_top_k_score": 0.5,
        "mrr": 0.6,
        "by_intent_class": {},
        "latency_ms": {"p95": 100.0},
    }
    candidate = {
        "recall_at": {"1": 0.5, "5": 0.7},
        "mean_top_k_score": 0.51,
        "mrr": 0.6,
        "by_intent_class": {},
        "latency_ms": {"p95": 120.0},
    }
    assert advanced_activation_gate(baseline, candidate)["passed"] is True
    candidate["latency_ms"] = {"p95": 120.1}
    assert advanced_activation_gate(baseline, candidate)["passed"] is False
    candidate.pop("latency_ms")
    assert advanced_activation_gate(baseline, candidate)["passed"] is False


def test_production_qrels_require_locked_grouped_120_query_contract() -> None:
    queries: list[GroundTruthQuery] = []
    groups: dict[str, str] = {}
    for index in range(120):
        query_id = f"q{index + 1:03d}"
        video_index = index % 60 + 1
        queries.append(
            GroundTruthQuery(
                query_id,
                (GroundTruthInterval(f"L21_V{video_index:03d}", 10, 20),),
                text=f"reviewed query {index + 1}",
                intent_class=IMPORTANT_INTENTS[index % len(IMPORTANT_INTENTS)],
            )
        )
        groups[query_id] = f"video-group-{video_index:03d}"
    split_metadata = {
        "method": "video_group",
        "ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "seed": 2026,
        "locked_test_sha256": "pending",
    }
    manifest = QrelsManifest(
        spec_id="test:v1",
        corpus_fingerprint="corpus",
        index_fingerprint="index",
        model_fingerprint="model",
        content_sha256="a" * 64,
        split_metadata=split_metadata,
    )
    draft = QrelsPack(manifest, tuple(queries), groups)
    locked = build_locked_qrels_split(draft, seed=2026)
    split_metadata["locked_test_sha256"] = locked.test_sha256
    pack = QrelsPack(
        QrelsManifest(
            spec_id=manifest.spec_id,
            corpus_fingerprint=manifest.corpus_fingerprint,
            index_fingerprint=manifest.index_fingerprint,
            model_fingerprint=manifest.model_fingerprint,
            content_sha256=manifest.content_sha256,
            split_metadata=split_metadata,
        ),
        tuple(queries),
        groups,
    )

    validate_production_qrels(pack)
    assert len(locked.train_query_ids) == 72
    assert len(locked.validation_query_ids) == 24
    assert len(locked.test_query_ids) == 24
    assert not set(locked.train_query_ids) & set(locked.test_query_ids)

    split_metadata["locked_test_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="locked test fingerprint"):
        validate_production_qrels(pack)


def test_lock_qrels_cli_atomically_records_split_fingerprint(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = []
    for index in range(3):
        text = f"reviewed query {index}"
        records.append(
            {
                "query_id": f"q{index}",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "intent": IMPORTANT_INTENTS[index],
                "video_group": f"g{index}",
                "intervals": [
                    {
                        "video_id": f"L21_V{index + 1:03d}",
                        "start_frame_idx": 1,
                        "end_frame_idx": 2,
                    }
                ],
            }
        )
    content = "".join(json.dumps(value) + "\n" for value in records).encode()
    (tmp_path / "qrels.jsonl").write_bytes(content)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "spec_id": "test:v1",
                "corpus_fingerprint": "corpus",
                "index_fingerprint": "index",
                "model_fingerprint": "model",
                "content_file": "qrels.jsonl",
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "split_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    assert lock_qrels_main([str(manifest_path), "--seed", "7", "--write"]) == 0
    report = json.loads(capsys.readouterr().out)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["written"] is True
    assert manifest["split_metadata"]["ratios"] == {
        "train": 0.6,
        "validation": 0.2,
        "test": 0.2,
    }
    assert manifest["split_metadata"]["locked_test_sha256"] == report[
        "locked_test_sha256"
    ]
    with pytest.raises(ValueError, match="at least 120"):
        cache_rankings_main(
            [
                "--qrels-manifest",
                str(manifest_path),
                "--batch-id",
                "aic25-b1-reference",
            ]
        )
