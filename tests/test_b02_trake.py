"""CPU-only B02 TRAKE scope, exact image identity and export gates."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.contracts import (
    TRAKEQueryRequest,
    TRAKEResponse,
    TRAKEResponseItem,
    TRAKEStage,
    TRAKEStageMatch,
)
from src.ingest.artifacts import ArtifactRepository
from src.retrieval.trake_alignment import TRAKEPipeline
from src.retrieval.trake_router import TRAKENotReadyError, TRAKEPipelineRouter
from src.retrieval.trake_scope import trake_video_allowed
from src.retrieval.trake_trajectory_cache import TRAKETrajectoryCache, TRAKETrajectoryCacheError
from src.submission.b02 import serialize_b02_submission
from tests.test_b02_release_paths import frame
from tests.test_trake_trajectory_cache import _MutableStore, _point

BATCH = "aic26-b2"


def point(source, time, vector, video="M01_V001", eligible=True):
    row = _point(source, vector, video_id=video, batch_id=BATCH, timestamp_ms=time)
    row["payload"]["trake_eligible"] = eligible
    return row


def test_scope_filters_before_cache_and_preserves_earlier_image_vector():
    store = _MutableStore()
    store.points = [point(0, 40, [0., 1.]), point(0, 0, [1., 0.]), point(25, 1000, [0., 1.]),
                    point(0, 0, [1., 0.], "N001-V001"), point(0, 0, [1., 0.], "M08_V001"),
                    point(0, 0, [1., 0.], "S01-V001", False)]
    cache = TRAKETrajectoryCache(store)
    cache.warm(BATCH)
    assert store.filters == [{"batch_id": BATCH, "trake_eligible": True}]
    entry = cache.batch(BATCH)
    assert set(entry.video_slices) == {"M01_V001"}
    video = entry.video("M01_V001")
    assert video.timestamps_ms.tolist() == [0, 1000]
    np.testing.assert_array_equal(video.vectors[0], [1., 0.])
    assert cache._fingerprint(BATCH) != cache._default_fingerprint(BATCH)
    assert cache._fingerprint("b01") == cache._default_fingerprint("b01")


@pytest.mark.parametrize("points", [
    [point(0, 0, [1., 0.]), point(0, 0, [0., 1.])],
    [point(0, 1000, [1., 0.]), point(25, 0, [0., 1.])],
    [point(0, 0, [float('nan'), 0.])],
])
def test_cache_fails_closed_on_ambiguous_or_invalid_data(points):
    store = _MutableStore()
    store.points = points
    cache = TRAKETrajectoryCache(store)
    with pytest.raises(TRAKETrajectoryCacheError):
        cache.warm(BATCH)
    assert cache.batch(BATCH) is None


def test_candidate_scope_is_applied_to_grouped_and_lexical_candidates():
    calls = []
    def search(vector, **kwargs):
        calls.append(kwargs)
        return [{"group_id": v, "hits": [{"score": .8}]} for v in
                ("N001-V001", "M08_V001", "M01_V001", "S01-V001")]
    core = TRAKEPipeline(encoder=SimpleNamespace(), vector_store=SimpleNamespace(search_grouped=search, vector_size=2),
                         trajectory_cache=SimpleNamespace())
    scores, _ = core._candidate_scores([np.array([[1., 0.]]), np.array([[0., 1.]])], BATCH,
                                      lexical_video_order=("N001-V001", "M08_V001"))
    assert set(scores) == {"M01_V001", "S01-V001"}
    assert all(x["filters"] == {"batch_id": BATCH, "trake_eligible": True} for x in calls)


class Repository:
    def __init__(self):
        self.frames = [replace(frame(1, "M01_V001"), timestamp_ms=0, source_frame_idx=0),
                       replace(frame(2, "M01_V001"), timestamp_ms=40, source_frame_idx=0),
                       replace(frame(3, "M01_V001"), timestamp_ms=1000, source_frame_idx=25)]

    def get_ordinal_frame(self, batch, video, ordinal):
        return next((f for f in self.frames if (f.batch_id, f.video_id, f.keyframe_ordinal) == (batch, video, ordinal)), None)

    def get_trake_frame(self, batch, video, source, *, timestamp_ms):
        return next((f for f in self.frames if (f.batch_id, f.video_id, f.source_frame_idx, f.timestamp_ms)
                     == (batch, video, source, timestamp_ms)), None)


def sequence():
    return dict(rank=1, batch_id=BATCH, video_id="M01_V001", stage_matches=[
        dict(stage_index=1, source_frame_idx=0, timestamp_ms=0, keyframe_ordinal=1, similarity=.8),
        dict(stage_index=2, source_frame_idx=25, timestamp_ms=1000, keyframe_ordinal=3, similarity=.7)])


def test_export_canonical_ordinals_and_monotonic_source_chain():
    repo = Repository()
    row = sequence()
    assert serialize_b02_submission(repo, [row], task="trake") == b"M01_V001,0,25\n"
    for video in ("N001-V001", "M08_V001", "L21_V001"):
        with pytest.raises(ValueError):
            serialize_b02_submission(repo, [{**row, "video_id": video}], task="trake")
    for bad in (
        {**row, "stage_matches": list(reversed(row["stage_matches"]))},
        {**row, "stage_matches": [row["stage_matches"][0], dict(row["stage_matches"][1], keyframe_ordinal=2)]},
        {**row, "stage_matches": [dict(row["stage_matches"][0], timestamp_ms=40), row["stage_matches"][1]]},
    ):
        with pytest.raises(ValueError):
            serialize_b02_submission(repo, [bad], task="trake")
    with pytest.raises(ValueError):
        serialize_b02_submission(repo, [row, {**row, "rank": 2}], task="trake")


def test_router_enriches_source_pts_into_exact_ordinal_without_reranking():
    repo = Repository()
    request = TRAKEQueryRequest(query_id="b02", global_description="a person leaves",
        stages=(TRAKEStage(stage_index=1, description="standing"), TRAKEStage(stage_index=2, description="leaving")),
        batch_filters=(BATCH,))
    row = sequence()
    raw = TRAKEResponse(query_id="b02", results=(TRAKEResponseItem(rank=1, batch_id=BATCH, video_id=row["video_id"],
        stage_matches=tuple(TRAKEStageMatch(**{k:v for k,v in m.items() if k != "keyframe_ordinal"}) for m in row["stage_matches"]),
        confidence_score=.7),))
    core = SimpleNamespace(search=lambda *a, **kw: raw)
    config = {"readiness": {"batches": {BATCH: {"kind":"reference", "trake_enabled":True}}}}
    router = TRAKEPipelineRouter({"organizer_clip":core}, config, artifact_repository=repo)
    response = router.search(request)
    assert [m.keyframe_ordinal for m in response.results[0].stage_matches] == [1, 3]
    assert response.results[0].confidence_score == raw.results[0].confidence_score
    repo.frames = []
    with pytest.raises(TRAKENotReadyError, match="required TRAKE"):
        router.search(request)


def test_ui_override_uses_ordinal_and_keeps_source_zero():
    from app.streamlit_app import _trake_sequence_with_overrides

    context = SimpleNamespace(artifact_repository=Repository())
    original = sequence()
    changed = _trake_sequence_with_overrides(context, original, {1:2})
    assert changed["stage_matches"][0] == dict(stage_index=1, source_frame_idx=0, timestamp_ms=40, keyframe_ordinal=2)
    assert original["stage_matches"][0]["timestamp_ms"] == 0
    with pytest.raises(ValueError):
        _trake_sequence_with_overrides(context, original, {1:3})


def test_repository_resolves_unique_source_pts_only(tmp_path):
    # Exercise real canonical reader without touching a production ledger.
    import sqlite3

    repo = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    repo.initialize()
    from src.ingest.artifacts import _content_fingerprint
    frames = Repository().frames
    with sqlite3.connect(repo.path) as db:
        for f in frames:
            import json
            db.execute("INSERT INTO canonical_ordinal_frames VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (f.batch_id,f.video_id,f.keyframe_ordinal,f.source_frame_idx,f.timestamp_ms,
                        json.dumps(f.to_dict()),"test",_content_fingerprint(f),1,0,0))
    assert repo.get_trake_frame(BATCH, "M01_V001", 0, timestamp_ms=40).keyframe_ordinal == 2
    assert repo.get_trake_frame(BATCH, "M01_V001", 0, timestamp_ms=41) is None
    assert repo.get_trake_frame(BATCH, "N001-V001", 0, timestamp_ms=0) is None
    assert trake_video_allowed(BATCH, "S01-V001")


def test_s_series_image_padding_is_verified_without_mutating_canonical_record(tmp_path):
    import io
    import zipfile

    from PIL import Image

    from src.retrieval.qa_media import CanonicalImageError, CanonicalImageResolver

    path = tmp_path / 'Keyframes_S01.zip'
    data = io.BytesIO()
    Image.new('RGB', (8,8), 'red').save(data, format='JPEG')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('keyframes/S01-V001/0030.jpg', data.getvalue())
    canonical = replace(frame(30, 'S01-V001'), image_uri=f'zip://{path}!/keyframes/S01-V001/030.jpg')
    resolver = CanonicalImageResolver((tmp_path,))
    result = resolver.resolve(canonical)
    assert result.width == 8 and result.video_id == canonical.video_id
    assert canonical.image_uri.endswith('/030.jpg')
    assert resolver.probe(canonical).source_revision == result.source_revision
    with pytest.raises(CanonicalImageError):
        resolver.resolve(replace(canonical, keyframe_ordinal=31))
    with pytest.raises(CanonicalImageError):
        resolver.resolve(replace(canonical, video_id='S01-V002'))
    resolver.close()
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('keyframes/S01-V001/030.jpg', data.getvalue())
    with pytest.raises(CanonicalImageError, match='ambiguous'):
        resolver.resolve(canonical)
    resolver.close()


def test_b01_response_does_not_add_ambiguous_null_ordinal():
    match = TRAKEStageMatch(stage_index=1, source_frame_idx=0, timestamp_ms=0, similarity=.5)
    assert 'keyframe_ordinal' not in match.model_dump(mode='json')


def test_vector_scroll_requests_b02_eligibility_but_preserves_b01_projection():
    from tests.test_vector_store import FakeQdrantClient, make_store

    client = FakeQdrantClient()
    store = make_store(client)
    list(store.iter_point_batches_with_vectors({'batch_id':BATCH,'trake_eligible':True}))
    assert 'trake_eligible' in client.scrolls[0]['with_payload']
    assert 'keyframe_ordinal' in client.scrolls[0]['with_payload']
    client.scrolls.clear()
    list(store.iter_point_batches_with_vectors({'batch_id':'b01'}))
    assert client.scrolls[0]['with_payload'] == ['batch_id','video_id','source_frame_idx','timestamp_ms']
