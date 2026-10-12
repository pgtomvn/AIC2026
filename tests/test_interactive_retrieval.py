"""Offline regression for explicit production interaction tools."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.contracts import FrameReference, QueryRequest
from src.evaluation.interactive_pilot import anchor_metrics, local_release_gate
from src.retrieval.frame_evidence import frame_evidence
from src.retrieval.interactive import SearchConversation, edit_conditions, rocchio, unit_vector
from src.retrieval.temporal_sequence import score_sequence
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline


class Encoder:
    def encode_text(self, text):
        return np.array([1., 0.])


def pipeline(provider):
    result = TextualKISPipeline(dense_channels=tuple(
        DenseChannel(name, Encoder(), SimpleNamespace(vector_size=2))
        for name in ("visual", "organizer_clip", "semantic_text")
    ), feedback_provider=provider)
    result.search = Mock(return_value=[])
    return result


def test_feedback_updates_every_visual_once_and_deduplicates_reference():
    provider = Mock(side_effect=lambda name, keys, **kwargs: np.array([[0., 1.]] * len(keys)))
    p = pipeline(provider)
    ref = FrameReference("b1", "L21_V001", 0)
    request = QueryRequest("q", "a person", batch_filters=("b1",), reference_frame=ref, positive_frames=(ref,))
    p.refine(request)
    assert p.search.call_count == 1
    vectors = p.search.call_args.kwargs["query_vectors"]
    assert set(vectors) == {"visual", "organizer_clip"}
    assert all(np.allclose(v, [.8, .6]) for v in vectors.values())
    assert provider.call_count == 2
    assert all(call.kwargs == {"batch_id": "b1"} for call in provider.call_args_list)


def test_channel_failure_does_not_discard_other_space():
    def provider(name, keys, *, batch_id):
        return np.array([[np.nan, 1.]]) if name == "visual" else np.array([[0., 1.]])
    p = pipeline(provider)
    p.refine(QueryRequest("q", "a person", reference_frame=FrameReference("b", "v", 0), image_only=True))
    assert set(p.search.call_args.kwargs["query_vectors"]) == {"organizer_clip"}
    assert p.search.call_args.args[0].batch_filters == ("b",)


def test_same_video_different_batch_is_rejected_not_silently_merged():
    p = pipeline(Mock())
    with pytest.raises(ValueError, match="batch"):
        p.refine(QueryRequest("q", "a person", batch_filters=("b1",), reference_frame=FrameReference("b2", "same", 0)))
    assert not p.feedback_provider.called


@pytest.mark.parametrize("value", ([0., 0.], [np.nan, 0.], [np.inf, 1.], [1e-12, 0.], [[1., 0.]]))
def test_invalid_vectors_rejected(value):
    with pytest.raises(ValueError):
        unit_vector(value)


def test_invalid_space_and_zero_sum_rejected():
    with pytest.raises(ValueError):
        rocchio([1., 0.], reference=[1., 0., 0.])
    with pytest.raises(ValueError):
        rocchio([1., 0.], negative=[[1., 0.]], gamma=1.)


def test_history_bounded_isolated_and_restores_original():
    first, second = SearchConversation(), SearchConversation()
    original = {"results": [0]}
    first.push(original)
    original["results"].append(99)
    for i in range(1, 24):
        first.push({"results": [i]})
    assert len(first.turns) == 20
    assert first.undo() == {"results": [22]}
    assert first.reset() == {"results": [0]}
    assert second.turns == []


def test_explicit_condition_edit_does_not_guess_or_repeat_history():
    original = ("Người mặc áo đỏ", "đang chạy")
    assert edit_conditions(original, "add", "đang chạy") == original
    assert edit_conditions(original, "replace", "đang đi", 1) == (original[0], "đang đi")
    assert edit_conditions(original, "remove", index=1) == (original[0],)
    with pytest.raises(ValueError):
        edit_conditions(original, "replace", "màu xanh")


def cache(times=(0, 1000, 2000), vectors=None):
    trajectory = SimpleNamespace(vectors=np.array(vectors if vectors is not None else [[1., 0.], [0., 1.], [1., 0.]]),
                                 source_frame_indices=np.array([0, 10, 20]),
                                 timestamps_ms=np.array(times), shot_codes=np.array([-1, -1, -1]))
    batch = SimpleNamespace(fingerprint="test-v2", video=lambda video: trajectory)
    return SimpleNamespace(batch=lambda b: batch if b == "b1" else None)


def test_sequence_strict_unique_frames_and_target_selection():
    hits, diagnostics = score_sequence([{"batch_id": "b1", "video_id": "v"}], cache=cache(), batch_id="b1",
                                       step_vectors=[np.array([1., 0.]), np.array([0., 1.])], global_vector=np.array([1., 0.]), target=1)
    assert hits[0]["source_frame_idx"] == 10
    assert [s["source_frame_idx"] for s in hits[0]["temporal_steps"]] == [0, 10]
    assert diagnostics["scored_video_count"] == 1
    assert diagnostics["cache_fingerprint"] == "test-v2"


@pytest.mark.parametrize("value", (None, cache(times=(0, 0, 2000)), cache(vectors=[[1., 0., 0.]] * 3)))
def test_sequence_invalid_cache_or_timestamps_falls_back(value):
    hits, _ = score_sequence([{"batch_id": "b1", "video_id": "v"}], cache=value, batch_id="b1",
                             step_vectors=[np.array([1., 0.])] * 2, global_vector=np.array([1., 0.]), target=1)
    assert hits == []


def test_sequence_cannot_cross_batches_and_caps_videos():
    candidates = [{"batch_id": "b1", "video_id": str(v)} for v in range(30)]
    candidates.insert(0, {"batch_id": "b2", "video_id": "outside"})
    hits, diag = score_sequence(candidates, cache=cache(), batch_id="b1", step_vectors=[np.array([1., 0.])] * 2,
                                global_vector=np.array([1., 0.]), target=0)
    assert len(hits) == 12 and diag["candidate_video_count"] == 12
    assert "outside" not in {h["video_id"] for h in hits}


def test_evidence_window_total_budget_and_frame_zero():
    segment = SimpleNamespace(segment_id="s", text_nfc="lời thoại", source_frame_idx=0, start_ms=0, end_ms=1000)
    repo = SimpleNamespace(get_frame=lambda *a: SimpleNamespace(timestamp_ms=1000),
                           text_segments_in_range=Mock(return_value=(segment,) * 150))
    result = frame_evidence(repo, "b", "v", 0)
    assert len(result["segments"]) == 200
    assert result["start_ms"] == 0 and result["end_ms"] == 31000
    assert result["speech_is_not_visual_confirmation"]
    for call in repo.text_segments_in_range.call_args_list:
        assert call.kwargs["index_ready_only"] and call.kwargs["limit"] == 100
    with pytest.raises(ValueError):
        frame_evidence(repo, "b", "v", 0, limit=201)


def test_evidence_reads_imported_ocr_when_local_text_table_is_empty():
    repo = SimpleNamespace(
        get_source_frame=lambda *args: SimpleNamespace(timestamp_ms=30_000),
        text_segments_in_range=Mock(return_value=()),
    )
    store = SimpleNamespace(search=Mock(return_value=[
        {"id": "second", "payload": {"batch_id": "b", "video_id": "v", "kind": "ocr",
                                      "start_ms": 40_000, "end_ms": 40_000,
                                      "source_frame_idx": 0, "text_nfc": "second"}},
        {"id": "first", "payload": {"batch_id": "b", "video_id": "v", "kind": "ocr",
                                     "start_ms": 0, "end_ms": 0,
                                     "source_frame_idx": 0, "text_nfc": "first"}},
        {"id": "other-video", "payload": {"batch_id": "b", "video_id": "other",
                                           "kind": "ocr", "start_ms": 0, "end_ms": 0,
                                           "text_nfc": "wrong"}},
    ]))

    result = frame_evidence(repo, "b", "v", 0, text_stores={"ocr": store})

    assert [item["text"] for item in result["segments"]] == ["first", "second"]
    assert all(item["source_frame_idx"] == 0 for item in result["segments"])
    assert repo.text_segments_in_range.call_count == 1  # ASR only; OCR came from the index.
    assert 'batch_id = "b"' in store.search.call_args.kwargs["filters"]
    assert 'video_id = "v"' in store.search.call_args.kwargs["filters"]
    assert "start_ms <= 60000" in store.search.call_args.kwargs["filters"]
    assert store.search.call_args.kwargs["top_k"] == 100


def test_contract_roundtrip_new_fields_and_old_request():
    old = QueryRequest("q", "người nói chuyện")
    new = replace(old, reference_frame=FrameReference("b", "v", 0), temporal_steps=("đi", "chạy"), temporal_target=1)
    assert QueryRequest.from_dict(new.to_dict()) == new
    assert QueryRequest.from_dict(old.to_dict()) == old


def test_pilot_gate_is_separate_fails_closed_and_requires_feature_labels():
    baseline = dict(query_sha256="q", split_sha256="s", health_ok=True, error_count=0, hit_at_1=.2,
                    hit_at_3=.4, hit_at_5=.6, mrr=.3, mean_top_k=.4, warm_p95_ms=100, rss_bytes=100)
    trial = {**baseline, "mean_top_k": .42}
    kwargs = dict(feature="feedback", data_kind="human_multiturn", parameters_locked=True, outside_scope_invariant=True)
    assert local_release_gate(baseline, trial, **kwargs)["automatic_activation_allowed"]
    assert not local_release_gate(baseline, trial, **{**kwargs, "data_kind": "kis"})["automatic_activation_allowed"]
    for bad in ({"rss_bytes": 121}, {"error_count": 1}, {"warm_p95_ms": 0}, {"mrr": None}, {"split_sha256": "different"}):
        assert not local_release_gate(baseline, {**trial, **bad}, **kwargs)["automatic_activation_allowed"]


def test_anchor_and_video_hit_are_reported_separately():
    query = {"query_id": "q", "intervals": [{"video_id": "v", "start_frame_idx": 10, "end_frame_idx": 10}]}
    metrics = anchor_metrics([{"query_id": "q", "hits": [{"video_id": "v", "source_frame_idx": 9}]}], [query])
    assert metrics["hit_at_1"] == 0 and metrics["video_hit_at_1"] == 1
    with pytest.raises(ValueError):
        anchor_metrics([], [query])


def test_regular_query_invariant_and_explicit_temporal_fallback_never_warms_cache():
    store = SimpleNamespace(search=lambda *args, **kwargs: [{
        "payload": {"batch_id": "b", "video_id": "v", "source_frame_idx": 0, "timestamp_ms": 0}, "score": 1.,
    }])
    p = TextualKISPipeline(dense_channels=(DenseChannel("organizer_clip", Encoder(), store),), temporal_windows_ms=())
    p.temporal_trajectories = SimpleNamespace(batch=lambda batch: None, warm=Mock())
    q = QueryRequest("q", "a person", batch_filters=("b",))
    before = p.search(q, refine_raw=False)
    explicit = replace(q, use_temporal_sequence=True, temporal_steps=("walking", "standing"))
    assert p.search(explicit, refine_raw=False) == before
    assert p.last_query_processing["temporal_sequence"]["reason_code"] == "trajectory_cache_not_ready"
    assert not p.temporal_trajectories.warm.called
    assert p.search(q, refine_raw=False) == before


def test_image_only_skips_text_spaces_and_lexical_channels():
    stores = {name: SimpleNamespace(search=Mock(return_value=[]), vector_size=2)
              for name in ("visual", "organizer_clip", "semantic_text", "asr", "ocr")}
    p = TextualKISPipeline(dense_channels=tuple(DenseChannel(name, Encoder(), stores[name])
                                               for name in ("visual", "organizer_clip", "semantic_text")),
                           lexical_channels=tuple(LexicalChannel(name, stores[name]) for name in ("asr", "ocr")),
                           feedback_provider=lambda name, keys, **kwargs: np.array([[1., 0.]]))
    p.refine(QueryRequest("q", "a person", image_only=True, reference_frame=FrameReference("b", "v", 0)))
    for name in ("visual", "organizer_clip"):
        assert stores[name].search.called
    for name in ("asr", "ocr", "semantic_text"):
        assert not stores[name].search.called


def test_feedback_text_keeps_vietnamese_and_clears_stale_translation():
    p = pipeline(Mock(return_value=np.array([[1., 0.]])))
    request = QueryRequest("q", "người đi bộ", english_text_override="walking", feedback_text="người đứng chờ")
    p.refine(request)
    sent = p.search.call_args.args[0]
    assert sent.original_text == "người đứng chờ" and sent.english_text_override is None
