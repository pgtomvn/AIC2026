"""Offline federation, coordinate and rollback contracts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.contracts import (
    QACandidateRequest,
    TRAKEQueryRequest,
    TRAKEResponse,
    TRAKEResponseItem,
    TRAKEStage,
)
from src.retrieval.batch_scope import B01, B02, BATCHES, merge_rankings, search_batches
from src.retrieval.textual_kis import TextualKISPipeline
from src.retrieval.trake_router import TRAKEPipelineRouter
from src.submission.combined import serialize_combined_submission
from tests.test_b02_release_paths import frame
from tests.test_b02_trake import Repository, sequence


def test_equal_rank_merge_preserves_ordinals_and_not_raw_scores():
    first = dict(batch_id=B01, video_id="L21_V001", source_frame_idx=0, score=999)
    second = dict(frame(1, "N001-V001").to_dict(), score=-1)
    third = dict(frame(2, "N001-V001").to_dict(), score=-2)
    result = merge_rankings({B02: [second, third], B01: [first]}, limit=3)
    assert [r["batch_id"] for r in result] == [B01, B02, B02]
    assert [r.get("keyframe_ordinal") for r in result] == [None, 1, 2]
    assert result[0]["score"] == result[1]["score"] == 1/61
    assert result[0]["fused_score"] == result[0]["score"]
    assert first["score"] == 999
    with pytest.raises(ValueError):
        merge_rankings({B01: [second]}, limit=3)


def test_scoped_search_dispatches_once_per_batch_and_keeps_single_calls_unchanged():
    calls = []
    class Pipeline(TextualKISPipeline):
        def search(self, query, **kwargs):
            if len(query["batch_filters"]) > 1:
                return super().search(query, **kwargs)
            batch = query["batch_filters"][0]
            calls.append((batch, kwargs))
            self.last_channel_statuses = {"organizer_clip": {"state": "ready"}}
            self.last_channel_errors = {}
            self.last_query_processing = {"original_text": query["original_text"]}
            return [dict(frame(1, "M01_V001").to_dict(), batch_id=batch, rank=1, score=.8)]
    pipeline = Pipeline()
    query = {"original_text": "a person", "batch_filters": BATCHES}
    hits = pipeline.search(query, limit=2, refine_raw=False, advanced_rerank=False)
    assert [row["batch_id"] for row in hits] == list(BATCHES)
    assert [call[0] for call in calls] == list(BATCHES)
    assert all(not kw["refine_raw"] and not kw["advanced_rerank"] for _, kw in calls)
    assert query["batch_filters"] == BATCHES
    assert set(pipeline.last_channel_statuses) == {f"{b}:organizer_clip" for b in BATCHES}


def test_request_scope_and_rollback_default():
    assert search_batches({}, B01) == (B01,)
    assert search_batches({"retrieval": {"all_batches": {"enabled": True}}}, B01) == BATCHES
    assert QACandidateRequest("q", "event", batch_filters=BATCHES).batch_filters == BATCHES
    stages = tuple(TRAKEStage(stage_index=n, description=str(n)) for n in (1, 2))
    request = TRAKEQueryRequest(query_id="q", global_description="event", stages=stages, batch_filters=BATCHES)
    with pytest.raises(ValueError):
        request.resolve_batch()
    for bad in ((B01, B01), (B01, "unknown"), (*BATCHES, "third")):
        with pytest.raises(ValueError):
            QACandidateRequest("q", "event", batch_filters=bad)


def test_trake_federation_uses_independent_cores_and_whole_chains():
    repo, calls = Repository(), []
    class Core:
        def __init__(self, batch):
            self.batch = batch
        def search(self, request, **kwargs):
            assert request.batch_filters == (self.batch,)
            calls.append(self.batch)
            row = dict(sequence(), confidence_score=.8, batch_id=self.batch)
            if self.batch == B01:
                row["video_id"] = "L21_V001"
            return TRAKEResponse(query_id="q", results=(TRAKEResponseItem(**row),))
    cores = {batch: Core(batch) for batch in BATCHES}
    router = TRAKEPipelineRouter({}, {"readiness": {"default_batch_kind": "reference"}},
                                 batch_pipelines=cores, artifact_repository=repo)
    request = TRAKEQueryRequest(query_id="q", global_description="event", batch_filters=BATCHES,
        stages=tuple(TRAKEStage(stage_index=n, description=str(n)) for n in (1, 2)))
    result = router.search(request)
    assert calls == list(BATCHES)
    assert [item.batch_id for item in result.results] == list(BATCHES)
    assert [item.rank for item in result.results] == [1, 2]
    assert result.results[1].stage_matches[0].keyframe_ordinal == 1


def test_combined_csv_resolves_each_row_not_selected_batch():
    b01 = replace(frame(), batch_id=B01, video_id="L21_V001", source_frame_idx=1581)
    traffic = frame(1, "N001-V001")
    repo = SimpleNamespace(get_source_frame=lambda *args: b01,
                           get_ordinal_frame=lambda *args: traffic)
    rows = [dict(b01.to_dict(), rank=1), dict(traffic.to_dict(), rank=2)]
    assert serialize_combined_submission(repo, rows) == b"L21_V001,1581\nN001-V001,1250\n"
    for bad in ([dict(rows[0], source_frame_idx=72)],
                [dict(rows[0], timestamp_ms=0)], [dict(rows[1], rank=1, keyframe_ordinal=2)],
                [dict(rows[0], rank=True)]):
        with pytest.raises(ValueError):
            serialize_combined_submission(repo, bad)


def test_batch_warmup_does_not_publish_all_ready_after_only_one_child():
    from src.batch_warmup import BatchWarmup, scoped_snapshot

    states = {B01: True, B02: False}
    def child(batch):
        return SimpleNamespace(snapshot=lambda: {"state": "ready" if states[batch] else "idle",
            "fully_ready": states[batch], "workflows": {w: {"usable": states[batch]} for w in ("kis", "qa", "trake")}})
    service = BatchWarmup({b: child(b) for b in BATCHES}, primary=B01)
    assert not service.snapshot()["search_usable"]
    assert scoped_snapshot(service.snapshot(), (B01,))["workflows"]["kis"]["usable"]
    states[B02] = True
    assert service.snapshot()["search_usable"]


def test_reference_feedback_loads_only_its_origin_but_keeps_combined_scope():
    import numpy as np

    from src.contracts import FrameReference, QueryRequest
    from src.retrieval.textual_kis import DenseChannel

    calls, searches = [], []
    def feedback(channel, keys, **kwargs):
        calls.append((channel, keys, kwargs))
        return np.array([[1., 0.]])
    pipeline = TextualKISPipeline(dense_channels=(
        DenseChannel("organizer_clip", SimpleNamespace(), SimpleNamespace(vector_size=2)),),
        feedback_provider=feedback)
    def search(query, **kwargs):
        searches.append((query.batch_filters, kwargs["query_vectors"]))
        return []
    pipeline.search = search
    request = QueryRequest("q", "a person", batch_filters=BATCHES, image_only=True,
        reference_frame=FrameReference(B02, "M01_V001", 0, 1))
    pipeline.refine(request)
    assert calls == [("organizer_clip", (("M01_V001", 1),),
                      {"batch_id": B02, "coordinate": "keyframe_ordinal"})]
    assert searches[0][0] == BATCHES
    with pytest.raises(ValueError):
        pipeline.refine(request, negative_ids=(("L21_V001", 0),))
