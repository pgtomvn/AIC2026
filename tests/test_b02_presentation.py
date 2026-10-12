"""Ordinal collisions, VFR coordinates, and bounded read-only image navigation."""
from types import SimpleNamespace

import pytest

from src.api import _public_hits
from src.ingest.b02_presentation import ordinal_neighborhood, present_ordinal_frame


def frame(ordinal=1, video='M08_V001', source=0, timestamp=0):
    return SimpleNamespace(batch_id='aic26-b2', video_id=video, keyframe_ordinal=ordinal,
                           source_frame_idx=source, timestamp_ms=timestamp)


def test_collision_images_are_distinct():
    a, b = frame(72), frame(73)
    assert present_ordinal_frame(a)['thumbnail_url'] != present_ordinal_frame(b)['thumbnail_url']
    assert present_ordinal_frame(a)['trake_eligible'] is False
    hits = _public_hits([vars(a), vars(b)])
    assert len(hits) == 2
    assert hits[0]['thumbnail_uri'].endswith('/72/image')
    assert hits[1]['thumbnail_uri'].endswith('/73/image')
    assert hits[0]['frame_semantics'] == 'keyframe_ordinal'


def test_vfr_uses_pts_and_keeps_zero_frame_valid():
    public = present_ordinal_frame(frame(video='N001-V001', source=0, timestamp=2520))
    assert (public['submission_unit'], public['submission_value']) == ('timestamp_ms', 2520)
    assert not public['trake_eligible']
    public = present_ordinal_frame(frame(video='M01_V001'))
    assert (public['submission_unit'], public['submission_value']) == ('source_frame_idx', 0)
    assert public['trake_eligible']


def test_ordinal_neighborhood_bounded_and_keeps_equal_pts():
    calls=[]
    def get(batch, video, ordinal):
        calls.append((batch,video,ordinal))
        return frame(ordinal) if 1 <= ordinal <= 4 else None
    result=ordinal_neighborhood(SimpleNamespace(get_ordinal_frame=get),'aic26-b2','M08_V001',2)
    assert [item['keyframe_ordinal'] for item in result['neighbors']] == [1,2,3,4]
    assert len(calls) <= 11
    assert all(item['timestamp_ms']==0 for item in result['neighbors'])
    with pytest.raises(ValueError):
        ordinal_neighborhood(None,'aic26-b2','M08_V001',1,6)


def test_invalid_identity_fails_closed_and_legacy_is_unchanged():
    for value in (0, -1, True, None):
        with pytest.raises(ValueError):
            present_ordinal_frame(frame(value))
    invalid=vars(frame())
    invalid.pop('keyframe_ordinal')
    assert _public_hits([invalid]) == []
    legacy={**vars(frame()),'batch_id':'aic25-b1-reference'}
    assert _public_hits([legacy])[0]['thumbnail_uri'].endswith('/0/image')
