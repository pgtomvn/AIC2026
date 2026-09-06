"""Canonical metadata batching must be evidence- and thread-safe."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from src.retrieval.frame_resolution import BatchedFrameResolver
from src.retrieval.textual_kis import TextualKISPipeline


class Repository:
    def __init__(self):
        self.calls = []
        self.nearest_calls = []
        self.version = 1

    def get_frames(self, batch, keys):
        self.calls.append((batch, list(keys)))
        return {key: SimpleNamespace(to_dict=lambda key=key: {
            "batch_id": batch, "video_id": key[0], "source_frame_idx": key[1],
            "timestamp_ms": self.version * 100, "shot_id": "s", "keyframe_ordinal": 1,
        }) for key in keys if key[1] != 999}

    def nearest_frame(self, batch, video, timestamp):
        self.nearest_calls.append((batch, video, timestamp))
        return None


def hit(frame=0, batch="b", score=.7):
    return {"id": f"ocr:{frame}", "score": score, "evidence": [str(score)],
            "payload": {"batch_id": batch, "video_id": "v", "source_frame_idx": frame,
                        "kind": "ocr", "timestamp_ms": 0, "text": str(score)}}


def test_batch_dedup_preserves_per_hit_payload_and_frame_zero():
    repository = Repository()
    resolver = BatchedFrameResolver(repository)
    pipeline = TextualKISPipeline(frame_resolver=resolver)
    results = pipeline._resolve_results([hit(score=.7), hit(score=.8), hit(1)])
    assert repository.calls == [("b", [("v", 0), ("v", 1)])]
    assert [r["score"] for r in results] == [.7, .8, .7]
    assert results[1]["payload"]["text"] == "0.8"
    assert results[0]["payload"]["timestamp_ms"] == 100
    assert results[0]["payload"]["source_frame_idx"] == 0
    assert pipeline._resolve_results(results) == results
    assert len(repository.calls) == 1


def test_chunk_boundary_batch_isolation_and_request_reset():
    repository = Repository()
    resolver = BatchedFrameResolver(repository)
    resolver.prefetch([*[hit(i) for i in range(201)], hit(0, "other")])
    assert [len(keys) for _, keys in repository.calls] == [100, 100, 1, 1]
    resolver.begin_request()
    repository.version = 2
    assert resolver(hit())["payload"]["timestamp_ms"] == 200
    assert len(repository.calls) == 5


def test_missing_frame_memoizes_timestamp_fallback_but_not_across_requests():
    repository = Repository()
    resolver = BatchedFrameResolver(repository)
    sample = hit(999)
    sample["payload"].update(start_ms=10, end_ms=20)
    assert resolver(sample) is None
    assert resolver(sample) is None
    assert repository.nearest_calls == [("b", "v", 15)]
    resolver.begin_request()
    resolver(sample)
    assert len(repository.nearest_calls) == 2


def test_cache_is_thread_local():
    repository = Repository()
    resolver = BatchedFrameResolver(repository)
    resolver(hit())
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(resolver, hit()).result()["payload"]["video_id"] == "v"
    assert len(repository.calls) == 2


def test_asr_only_scope_uses_exact_legacy_path_for_every_other_mode():
    calls = []

    def legacy(result):
        calls.append(result)
        return result

    repository = Repository()
    resolver = BatchedFrameResolver(repository, scalar_fallback=legacy)
    for mode in ("automatic", "visual_only", "unknown"):
        resolver.set_request_mode(mode)
        resolver.begin_request()
        sample = hit()
        resolver.prefetch([sample])
        assert resolver(sample) is sample
    assert len(calls) == 3
    assert repository.calls == []
    resolver.set_request_mode("asr_only")
    resolver.begin_request()
    assert resolver(hit())["payload"]["timestamp_ms"] == 100
    assert len(calls) == 3
    assert len(repository.calls) == 1
    resolver.set_request_mode("automatic")
    assert resolver(hit())["payload"]["timestamp_ms"] == 0


def test_invalid_batch_or_bool_frame_matches_scalar_fallback_behavior():
    resolver = BatchedFrameResolver(Repository())
    sample = hit(True)
    assert resolver(sample) is None
    assert resolver.repository.nearest_calls == [("b", "v", 0)]
    sample["payload"]["batch_id"] = None
    assert resolver(sample) is None


def test_real_sqlite_batch_matches_scalar_and_nearest_tie(tmp_path):
    from dataclasses import replace

    from src.contracts import FrameRecord
    from src.ingest.artifacts import ArtifactRepository

    repository = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    repository.initialize()
    frame = FrameRecord(
        batch_id="b", video_id="v", keyframe_ordinal=1, source_frame_idx=0,
        pts=0, time_base="1/25", timestamp_ms=0, fps_num=25, fps_den=1,
        shot_id="s1", sampling_reasons=("reference",),
        image_uri="zip://frames.zip!/v/001.jpg", video_uri="zip://videos.zip!/v.mp4",
        source_sha256="0" * 64, extractor_version="reference-v1",
    )
    repository.upsert_frames([frame, replace(frame, keyframe_ordinal=2,
        source_frame_idx=25, pts=25, timestamp_ms=1000, shot_id="s2")])
    resolver = BatchedFrameResolver(repository)
    samples = [hit(0), hit(25), hit(999)]
    samples[-1]["payload"].update(start_ms=400, end_ms=600)
    resolver.prefetch(samples)
    for sample in samples:
        payload = sample["payload"]
        canonical = repository.get_frame("b", "v", payload["source_frame_idx"])
        if canonical is None:
            canonical = repository.nearest_frame("b", "v", 500)
        expected = {**sample, "payload": {**payload, **canonical.to_dict()}}
        assert resolver(sample) == expected
    assert resolver(samples[-1])["payload"]["source_frame_idx"] == 0
