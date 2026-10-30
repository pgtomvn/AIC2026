"""CPU/offline graph tests: evidence, scope, bounds, identity, and corruption."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.build_topic_graph import (
    imported_asr_path,
    read_asr_jsonl,
    read_inputs,
    read_ordinal_mappings,
)
from scripts.build_topic_graph import main as build_main
from src.graph.object_graph import (
    FrameRef,
    ObjectGraphIndex,
    ObjectVocabulary,
    QueryGraph,
    QueryNode,
    build_object_index,
    parse_query_graph,
    predicate_holds,
    write_object_index,
)
from src.graph.topic_graph import (
    AsrSegment,
    Keyframe,
    Topic,
    TopicGraph,
    TopicGraphParams,
    build_topic_graph,
    segment_topics,
    write_topic_graph,
)


def topic(index: int, video: str = "L21_V001") -> Topic:
    return Topic(index, "aic25-b1-reference", video, index, index * 1000,
                 index * 1000 + 900, f"Transcript {index}", index * 25, index + 1)


def graph(params: TopicGraphParams | None = None) -> TopicGraph:
    return TopicGraph(
        "aic25-b1-reference", [topic(0), topic(1), topic(2, "L22_V001")],
        np.array([[1, 0], [0.8, 0.6], [0.8, 0.6]]),
        [(0, 1, "NEXT", 1.0), (0, 2, "GRAPH_RELATION", 1.0)], [],
        params or TopicGraphParams(candidate_k=1),
    )


def objects(max_candidates: int = 4000) -> ObjectGraphIndex:
    return ObjectGraphIndex(
        "aic25-b1-reference", ["Person", "Banana", "Table"],
        [FrameRef("L21_V001", 1, 25), FrameRef("L22_V001", 1, 50)],
        np.array([0, 0, 0, 1]), np.array([0, 1, 2, 1]),
        np.array([0.95, 0.92, 0.90, 0.80]),
        np.array([[0.1, 0.1, 0.8, 0.3], [0.3, 0.6, 0.4, 0.7],
                  [0.6, 0.5, 0.9, 0.95], [0.2, 0.3, 0.4, 0.5]]),
        max_candidates=max_candidates,
    )


def test_segmentation_preserves_overlap_and_checks_input() -> None:
    segments = [AsrSegment("V", 0, 10000, "long"), AsrSegment("V", 1000, 2000, "short"),
                AsrSegment("V", 9000, 11000, "overlap"), AsrSegment("V", 20000, 21000, "new")]
    assert list(map(len, segment_topics(segments))) == [3, 1]
    with pytest.raises(ValueError, match="one video"):
        segment_topics([segments[0], AsrSegment("other", 1, 2, "other")])
    with pytest.raises(ValueError, match="timestamps"):
        segment_topics([AsrSegment("V", -1, 20, "invalid")])


def test_build_preserves_transcript_and_actual_keyframe_identity(tmp_path: Path) -> None:
    artifact = build_topic_graph(
        "aic25-b1-reference", {"L21_V001": [AsrSegment("L21_V001", 0, 2000, "real ASR")]},
        lambda texts: np.array([[1.0, 0.0]]),
        keyframes_by_video={"L21_V001": [Keyframe(7, 123, 1000)]},
        summarize=lambda text: "condensed", encoder_fingerprint="test-encoder",
    )
    directory = tmp_path / "topic"
    write_topic_graph(artifact, directory)
    loaded = TopicGraph.load(directory)
    assert loaded.encoder_fingerprint == "test-encoder"
    match = loaded.search(np.array([1, 0]))["matches"][0]
    assert (match["text"], match["transcript"], match["source_frame_idx"], match["keyframe_ordinal"]) == (
        "condensed", "real ASR", 123, 7)
    with pytest.raises(FileExistsError):
        write_topic_graph(artifact, directory)
    with (directory / "entities.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="integrity"):
        TopicGraph.load(directory)


def test_topic_expansion_cannot_escape_selected_video() -> None:
    result = graph().search(np.array([1, 0]), np.array([0.8, 0.6]), video_ids=["L21_V001"])
    assert {item["video_id"] for item in result["matches"]} == {"L21_V001"}
    assert result["subgraph_size"] == 2
    assert result["matches"][0]["path"] == [
        {"topic_id": 0, "relation": None}, {"topic_id": 1, "relation": "NEXT"}]
    assert graph().search(np.array([1, 0]), video_ids=[])["fallback_reason"] == "no_topic_evidence"


def test_topic_similarity_and_refinement_fallback() -> None:
    assert graph().search(np.array([-1, 0]))["fallback_reason"] == "no_topic_evidence"
    result = graph().search(np.array([1, 0]), np.array([-1, 0]))
    assert result["matches"] == []
    assert result["fallback_reason"] == "no_refinement_evidence"


def test_graph_bonus_and_display_have_hard_bounds() -> None:
    params = TopicGraphParams(candidate_k=1, max_search_topics=2, max_display_topics=1,
                              next_bonus=1, semantic_bonus=1, tau=0.2)
    result = graph(params).search(np.array([1, 0]), final_k=100)
    assert result["subgraph_size"] == 2
    assert len(result["nodes"]) == 1
    assert max(item["graph_bonus"] for item in result["matches"]) <= 0.2


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 0], [1, 0, 0]])
def test_invalid_query_embedding_is_rejected(vector: list[float]) -> None:
    with pytest.raises(ValueError):
        graph().search(np.asarray(vector))


def test_invalid_topic_edges_and_params_fail_clearly() -> None:
    with pytest.raises(ValueError, match="NEXT"):
        TopicGraph("aic25-b1-reference", [topic(0), topic(1, "other")], np.eye(2),
                   [(0, 1, "NEXT", 1)], [])
    with pytest.raises(ValueError, match="positive"):
        TopicGraphParams(candidate_k=0)


def test_vietnamese_objects_and_geometric_evidence() -> None:
    result = objects().search_with_evidence("người bên trái chuối")
    assert result["fallback_reason"] is None
    assert result["matches"][0]["score"] == 1
    evidence = result["matches"][0]["evidence"]
    assert evidence["edges"][0]["predicate"] == "left_of"
    assert evidence["edges"][0]["satisfied"] is True
    assert evidence["nodes"][0]["box"] == [0.1, 0.1, 0.8, 0.3]


@pytest.mark.parametrize("text", ["person behind table", "person holding banana",
                                  "person wearing banana", "red banana", "without banana"])
def test_missing_visual_semantics_trigger_conventional_fallback(text: str) -> None:
    result = objects().search_with_evidence(text)
    assert result["fallback_reason"] == "unverified_object_constraints"
    assert result["query_graph"]["unverified"]
    for match in result["matches"]:
        for edge in match["evidence"]["edges"]:
            if not edge["verified"]:
                assert edge["satisfied"] is False


def test_generic_person_does_not_prove_man_or_woman() -> None:
    vocabulary = ObjectVocabulary(["Person"])
    assert not vocabulary.lookup("man")
    assert not vocabulary.lookup("woman")
    assert vocabulary.lookup("người") == frozenset()
    assert parse_query_graph("người", vocabulary).nodes[0].classes == frozenset({0})


def test_repeated_nodes_require_distinct_detections_even_without_edges() -> None:
    result = objects().search_with_evidence("banana and banana")
    assert result["fallback_reason"] == "incomplete_object_evidence"
    assert result["matches"][0]["score"] == 0.5
    nodes = result["matches"][0]["evidence"]["nodes"]
    assert sum(node["matched"] for node in nodes) == 1


def test_overlapping_duplicate_detections_do_not_inflate_count() -> None:
    index = ObjectGraphIndex("B", ["Banana"], [FrameRef("V", 1, 25)],
                             np.array([0, 0]), np.array([0, 0]), np.array([0.9, 0.8]),
                             np.array([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2]]))
    result = index.search_with_evidence("two bananas")
    assert result["fallback_reason"] == "incomplete_object_evidence"
    assert result["matches"][0]["evidence"]["nodes"][0]["count"] == 1


def test_video_filter_precedes_object_candidate_bound() -> None:
    result = objects(max_candidates=1).search_with_evidence("banana", video_ids=["L22_V001"])
    assert [hit["video_id"] for hit in result["matches"]] == ["L22_V001"]
    assert objects().search_with_evidence("banana", video_ids=[])["matches"] == []


def test_prior_candidates_remain_bounded_and_finite() -> None:
    index = objects(max_candidates=1)
    results = index.search(index.parse("banana"), prior={("L21_V001", 25): 0.1, ("L22_V001", 50): 0.9})
    assert len(results) == 1
    with pytest.raises(ValueError, match="finite"):
        index.search(index.parse("banana"), prior={("L21_V001", 25): float("nan")})


def test_object_archive_reader_preserves_real_boxes_and_rejects_corruption(tmp_path: Path) -> None:
    archive = tmp_path / "objects.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Objects/L21_V001/007.json", json.dumps({
            "detection_scores": [0.9], "detection_class_entities": ["Banana"],
            "detection_boxes": [[0.1, 0.2, 0.3, 0.4]],
        }))
    artifact = build_object_index("B", [archive], lambda video, ordinal: 123 if ordinal == 7 else None)
    assert artifact.frames == [FrameRef("L21_V001", 7, 123)]
    directory = tmp_path / "object"
    write_object_index(artifact, directory)
    index = ObjectGraphIndex.load(directory)
    assert index.frame_objects("L21_V001", 123)[0]["box"] == [0.1, 0.2, 0.3, 0.4]
    (directory / "classes.json").write_text('["Apple"]', encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        ObjectGraphIndex.load(directory)


def test_nan_object_boxes_fail_at_index_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        ObjectGraphIndex("B", ["Banana"], [FrameRef("V", 1, 25)], np.array([0]),
                         np.array([0]), np.array([0.9]), np.array([[0.1, 0.2, float("nan"), 0.4]]))


def test_builder_reads_canonical_asr_and_preserves_alias_ordinals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    database = tmp_path / "artifacts.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
        CREATE TABLE canonical_text_segments(batch_id,video_id,segment_id,kind,start_ms,end_ms,text_nfc,source_frame_idx,index_ready);
        CREATE TABLE canonical_frames(batch_id,video_id,keyframe_ordinal,source_frame_idx,timestamp_ms,index_ready);
        CREATE TABLE canonical_frame_mappings(batch_id,video_id,keyframe_ordinal,source_frame_idx);
        INSERT INTO canonical_text_segments VALUES('aic25-b1-reference','L21_V001','1','asr',0,1000,'real ASR',25,1);
        INSERT INTO canonical_text_segments VALUES('aic25-b1-reference','L21_V001','2','ocr',0,1000,'OCR excluded',25,1);
        INSERT INTO canonical_frames VALUES('aic25-b1-reference','L21_V001',1,25,500,1);
        INSERT INTO canonical_frame_mappings VALUES('aic25-b1-reference','L21_V001',2,25);
        """)
    before = database.read_bytes()
    segments, frames, mappings = read_inputs(database, "aic25-b1-reference", "L21_V001")
    assert [segment.text for segment in segments["L21_V001"]] == ["real ASR"]
    assert frames["L21_V001"] == [Keyframe(1, 25, 500)]
    assert mappings[("L21_V001", 2)] == 25
    assert build_main(["--database", str(database)]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "inspect"
    assert before == database.read_bytes()


def test_b2_builds_objects_by_ordinal_but_refuses_topics(tmp_path: Path) -> None:
    database = tmp_path / "artifacts.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
        CREATE TABLE canonical_ordinal_frames(batch_id,video_id,keyframe_ordinal,source_frame_idx,index_ready);
        INSERT INTO canonical_ordinal_frames VALUES('aic26-b2','M01_V001',1,0,1);
        INSERT INTO canonical_ordinal_frames VALUES('aic26-b2','M01_V001',2,90,1);
        INSERT INTO canonical_ordinal_frames VALUES('aic26-b2','M01_V001',3,180,0);
        INSERT INTO canonical_ordinal_frames VALUES('aic26-b2','M01_V002',1,0,1);
        """)
    assert read_ordinal_mappings(database, "aic26-b2", "M01_V001") == {("M01_V001", 1): 0, ("M01_V001", 2): 90}
    with pytest.raises(ValueError, match="B2 supports --kind object"):
        build_main(["--batch-id", "aic26-b2", "--kind", "topic", "--full", "--database", str(database)])


def test_cloud_asr_source_is_scoped_and_requires_batch_identity(tmp_path: Path) -> None:
    source = tmp_path / "asr.jsonl.gz"
    row = {"batch_id": "B", "video_id": "V1", "segment_id": "asr:B:V1:0", "kind": "asr",
           "start_ms": 20, "end_ms": 100, "source_frame_idx": 25, "text_nfc": "real ASR"}
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.write(json.dumps({**row, "video_id": "V2", "segment_id": "asr:B:V2:0"}) + "\n")
    result = read_asr_jsonl(source, "B", "V1")
    assert result == {"V1": [AsrSegment("V1", 20, 100, "real ASR", 25)]}
    with pytest.raises(ValueError, match="different batch"):
        read_asr_jsonl(source, "other")


def test_completed_import_receipt_selects_and_validates_asr_revision(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    transcripts = tmp_path / "b01_asr_transcripts.jsonl.gz"
    transcripts.touch()
    receipt = tmp_path / "asr.json"
    receipt.write_text(json.dumps({
        "contract": "cloud-import-channel-v2", "batch_id": "B", "channel": "asr",
        "snapshot_import_complete": True, "coverage_complete": True,
        "manifest_path": str(manifest), "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
    }), encoding="utf-8")
    assert imported_asr_path(receipt, "B") == transcripts
    manifest.write_text(" ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        imported_asr_path(receipt, "B")


def test_rounded_cosine_weights_just_above_one_are_accepted() -> None:
    from src.graph.topic_graph import Topic, TopicGraph

    topics = [Topic(i, "b1", f"v{i}", 0, 0, 900, "t", 0) for i in range(2)]
    graph = TopicGraph("b1", topics, np.eye(2, dtype=np.float32), [(0, 1, "GRAPH_RELATION", 1.0000001)], [])
    assert graph.neighbors(0)[0].weight == 1.0
    with pytest.raises(ValueError):
        TopicGraph("b1", topics, np.eye(2, dtype=np.float32), [(0, 1, "GRAPH_RELATION", 1.01)], [])


def test_fusion_rewards_agreement_and_keeps_graph_only_moments() -> None:
    from src.graph.fusion import FusionWeights, fuse

    def frame(video: str, source: int, ms: int) -> dict:
        return {"batch_id": "b1", "video_id": video, "source_frame_idx": source,
                "keyframe_ordinal": None, "timestamp_ms": ms}

    conventional = [frame("v1", 10, 1_000), frame("v2", 20, 50_000), frame("v3", 30, 9_000)]
    topics = [{**frame("v2", 25, 52_000), "graph_evidence": {"start_ms": 45_000, "end_ms": 60_000, "text": "said"}},
              {**frame("v9", 90, 5_000), "graph_evidence": {"start_ms": 0, "end_ms": 9_000, "text": "only speech"}}]
    hits = fuse(conventional, topics, weights=FusionWeights(), limit=10)
    assert hits[0]["video_id"] == "v2" and hits[0]["sources"] == ["conventional", "topic"]
    assert hits[0]["graph_evidence"]["text"] == "said"
    assert any(h["video_id"] == "v9" and h["sources"] == ["topic"] for h in hits)
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    # Without graph evidence the conventional order is unchanged.
    assert [h["video_id"] for h in fuse(conventional, [])] == ["v1", "v2", "v3"]


def test_fusion_object_term_scales_with_satisfied_fraction() -> None:
    from src.graph.fusion import fuse

    base = [{"batch_id": "b1", "video_id": f"v{i}", "source_frame_idx": i, "timestamp_ms": i} for i in range(3)]
    partial = {**base[2], "fraction": 0.5, "graph_evidence": {"nodes": []}}
    full = {**base[2], "fraction": 1.0, "graph_evidence": {"nodes": []}}
    score = {name: next(h["fused_score"] for h in fuse(base, [], hits) if h["video_id"] == "v2")
             for name, hits in (("none", []), ("partial", [partial]), ("full", [full]))}
    assert score["none"] < score["partial"] < score["full"]
    lifted = fuse(base, [], [full])
    assert lifted[0]["video_id"] == "v2" and "object_evidence" in lifted[0]
    assert fuse(base, [], [{**base[2], "fraction": 0.0}])[2]["sources"] == ["conventional"]


@pytest.mark.parametrize("value", [{"k": 0}, {"topic": -1}])
def test_fusion_weights_are_validated(value: dict) -> None:
    from src.graph.fusion import FusionWeights

    with pytest.raises(ValueError):
        FusionWeights.from_mapping(value)
