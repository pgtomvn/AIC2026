"""Manual local inspection: no external fallback, bounded wait, no ranking changes."""
import hashlib
import io
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from src.contracts import FrameReference
from src.retrieval.event_inspection import EventInspector


def service(budget=1):
    data = io.BytesIO()
    Image.new('RGB', (20, 20), 'red').save(data, format='PNG')
    blob = data.getvalue()
    resolver = SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(
        data=blob, mime_type='image/png', content_sha256=hashlib.sha256(blob).hexdigest())))
    repo = SimpleNamespace(get_frame=lambda b, v, f: SimpleNamespace(
        batch_id=b, video_id=v, source_frame_idx=f, timestamp_ms=f*10))
    provider = SimpleNamespace(validated_model_digest='digest1', fingerprint='provider-v1',
                               inspect_event=Mock(return_value={'state': 'ready', 'verdict': 'insufficient_evidence'}))
    return EventInspector(repo, resolver, provider, budget_seconds=budget), provider


def test_canonical_sort_cache_identity_and_model_invalidation():
    inspector, provider = service()
    refs = [FrameReference('b', 'v', 10), FrameReference('b', 'v', 0)]
    value = inspector.inspect(refs, 'Is the person running?')
    assert not value['ranking_applied']
    assert [f['source_frame_idx'] for f in value['frames']] == [0, 10]
    assert inspector.inspect(refs, 'Is the person running?')['cache_hit']
    assert provider.inspect_event.call_count == 1
    provider.validated_model_digest = 'digest2'
    assert not inspector.inspect(refs, 'Is the person running?')['cache_hit']
    assert not inspector.inspect([FrameReference('other-batch', 'v', 0)], 'Is the person running?')['cache_hit']


def test_one_video_unique_bounded_frames():
    inspector, _ = service()
    for refs in ([], [FrameReference('b', 'v', i) for i in range(4)],
                 [FrameReference('b', 'v', 0)]*2,
                 [FrameReference('b', 'v', 0), FrameReference('b', 'other', 1)]):
        with pytest.raises(ValueError):
            inspector.inspect(refs, 'a question')


def test_busy_has_no_inference_and_timeout_does_not_open_second_flight():
    inspector, provider = service(budget=.03)
    ref = [FrameReference('b', 'v', 0)]
    assert inspector.enter_retrieval()
    assert inspector.inspect(ref, 'Question')['reason_code'] == 'gpu_busy'
    provider.inspect_event.assert_not_called()
    inspector.leave_retrieval()
    release = threading.Event()
    stopped = threading.Event()

    def slow(*a, **kw):
        release.wait(timeout=1)
        stopped.set()
        return {'state': 'ready', 'verdict': 'insufficient_evidence'}
    provider.inspect_event.side_effect = slow
    try:
        assert inspector.inspect(ref, 'Question')['reason_code'] == 'local_inference_deadline'
        assert inspector.inspect(ref, 'Question')['reason_code'] == 'gpu_busy'
        assert not inspector.enter_retrieval()
        assert provider.inspect_event.call_count == 1
    finally:
        release.set()
        stopped.wait(timeout=1)


def test_invalid_frame_never_infers_and_failure_never_retries():
    inspector, provider = service()
    inspector.repository.get_frame = lambda *a: None
    assert inspector.inspect([FrameReference('b','v',0)], 'Question')['state'] == 'unavailable'
    provider.inspect_event.assert_not_called()
