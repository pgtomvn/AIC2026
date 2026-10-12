"""Ordinal HTTP/image boundaries remain offline and preserve equal-PTS images."""
from types import SimpleNamespace

import pytest

from app.components.ordinal_images import load_ordinal_image
from app.http_backend import BackendError, _Frame, _ImageResolver, _Repository


class Client:
    def __init__(self):
        self.paths = []
        self.rows = [dict(batch_id="aic26-b2", video_id="M08_V001", keyframe_ordinal=n,
                          source_frame_idx=100, timestamp_ms=4000) for n in (72, 73)]
        self.error = None

    def json(self, method, path, *, params):
        self.paths.append(path)
        if self.error:
            raise self.error
        rows = self.rows if params["radius"] else self.rows[:1]
        return {"neighbors": rows}

    def bytes(self, path):
        self.paths.append(path)
        return path.encode()


def test_equal_pts_neighbors_have_distinct_image_paths():
    client = Client()
    frames = _Repository(client).ordinal_neighbors("aic26-b2", "M08_V001", 72, radius=1)
    assert [x.keyframe_ordinal for x in frames] == [72, 73]
    assert [x.source_frame_idx for x in frames] == [100, 100]
    resolver = _ImageResolver(client)
    assert resolver.resolve(frames[0]).data != resolver.resolve(frames[1]).data
    assert all("/v1/keyframes/" in path for path in client.paths)


def test_source_frame_zero_and_missing_ordinal_never_fallback():
    client = Client()
    client.rows[0]["source_frame_idx"] = 0
    repository = _Repository(client)
    assert repository.get_ordinal_frame("aic26-b2", "M08_V001", 72).source_frame_idx == 0
    with pytest.raises(ValueError):
        _ImageResolver(client).resolve(_Frame("aic26-b2", "M08_V001", 100))
    context = SimpleNamespace(artifact_repository=repository, qa_image_resolver=_ImageResolver(client))
    item = dict(batch_id="aic26-b2", video_id="M08_V001", source_frame_idx=0, keyframe_ordinal=72)
    assert b"/72/image" in load_ordinal_image(context, item)
    item.pop("keyframe_ordinal")
    assert load_ordinal_image(context, item) is None


@pytest.mark.parametrize("change", [
    {"batch_id": "b01"}, {"video_id": "other"}, {"keyframe_ordinal": 90},
    {"keyframe_ordinal": True}, {"timestamp_ms": -1}, {"source_frame_idx": True},
])
def test_corrupt_neighbors_rejected_before_image_loading(change):
    client = Client()
    client.rows[0].update(change)
    with pytest.raises(ValueError):
        _Repository(client).ordinal_neighbors("aic26-b2", "M08_V001", 72, radius=1)


def test_missing_and_unavailable_are_distinct():
    client = Client()
    repository = _Repository(client)
    client.error = BackendError("not_found", status_code=404)
    assert repository.get_ordinal_frame("aic26-b2", "M08_V001", 72) is None
    client.error = BackendError("unavailable", status_code=503)
    with pytest.raises(BackendError):
        repository.get_ordinal_frame("aic26-b2", "M08_V001", 72)


def test_legacy_image_path_and_unmatched_frame_identity():
    client = Client()
    resolver = _ImageResolver(client)
    assert resolver.resolve(_Frame("b01", "L21_V001", 0)).data == b"/v1/frames/b01/L21_V001/0/image"
    context = SimpleNamespace(
        artifact_repository=SimpleNamespace(get_ordinal_frame=lambda *args: _Frame(
            "aic26-b2", "M08_V001", 100, keyframe_ordinal=73)),
        qa_image_resolver=resolver,
    )
    assert load_ordinal_image(context, dict(batch_id="aic26-b2", video_id="M08_V001",
                                           keyframe_ordinal=72)) is None
    assert len(client.paths) == 1
