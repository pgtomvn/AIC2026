"""Offline B02 ordinal identity, QA, feedback, evidence and export regression gates."""
from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from app.components.result_gallery import pin_widget_key, result_identity
from src.api import create_app
from src.contracts import (
    QA_GENERAL_TEXT_SPEC_ID,
    FrameRecord,
    FrameReference,
    QACompetitionSpec,
    QAFrameInjectionRequest,
    QASingleFrameAnswerRequest,
    QueryRequest,
)
from src.ingest.frame_identity import candidate_identity, resolve_image_frame
from src.retrieval.frame_evidence import frame_evidence
from src.retrieval.frame_resolution import BatchedFrameResolver
from src.retrieval.qa_retrieval import VisualQARetriever
from src.retrieval.qa_session import QASessionStore
from src.retrieval.textual_kis import DenseChannel, TextualKISPipeline
from src.retrieval.visual_qa import VisualQAPipeline
from src.submission.b02 import serialize_b02_submission


def frame(ordinal=72, video="M08_V001"):
    return FrameRecord(batch_id="aic26-b2", video_id=video, keyframe_ordinal=ordinal,
                       source_frame_idx=0, pts=125, time_base="1/100", timestamp_ms=1250,
                       fps_num=25, fps_den=1, shot_id="shot", sampling_reasons=("fixture",),
                       image_uri=f"fixture-{ordinal}.jpg", video_uri="video.zip",
                       source_sha256="a" * 64, extractor_version="fixture")


class Repository:
    def __init__(self):
        self.frames = {(f.video_id, f.keyframe_ordinal): f for f in
                       (frame(), frame(73), frame(1, "N001-V001"), frame(1, "S01-V001"))}

    def get_frame(self, *args):
        raise AssertionError("source lookup forbidden for B02")

    def get_ordinal_frame(self, batch, video, ordinal):
        return self.frames.get((video, ordinal)) if batch == "aic26-b2" else None

    def text_segments_in_range(self, *args, **kwargs):
        return []


class Textual:
    last_channel_statuses = {}
    last_channel_errors = {}
    last_query_processing = {}

    def search(self, *args, **kwargs):
        return [dict(frame(n).to_dict(), score=0.9, channel_ranks={"organizer_clip": n-71}) for n in (72, 73)]


def test_collision_widget_and_candidate_identity():
    values = [frame(n).to_dict() for n in (72, 73)]
    assert len({pin_widget_key(result_identity(x)) for x in values}) == 2
    candidates = VisualQARetriever(Textual(), Repository()).retrieve_frames("two images", batch_filters=("aic26-b2",))
    assert [x.candidate_id for x in candidates] == [candidate_identity(frame(n)) for n in (72, 73)]
    assert [x.source_frame_idx for x in candidates] == [0, 0]


def test_canonical_mismatch_and_missing_ordinal_rejected():
    with pytest.raises((ValueError, TypeError)):
        FrameReference("aic26-b2", "M08_V001", 0)
    with pytest.raises(ValueError):
        resolve_image_frame(Repository(), dict(frame().to_dict(), source_frame_idx=1))
    request = QueryRequest("q", "a person", batch_filters=("aic26-b2",),
                           reference_frame=FrameReference("aic26-b2", "M08_V001", 0, 72))
    assert QueryRequest.from_dict(request.to_dict()) == request


def test_pts_export_and_duplicate_submission_coordinates():
    repo = Repository()
    rows = [frame(1, "N001-V001").to_dict(), frame(1, "S01-V001").to_dict()]
    assert serialize_b02_submission(repo, rows) == b"N001-V001,1250\nS01-V001,0\n"
    qa = [dict(rows[0], answer=" YES ")]
    assert serialize_b02_submission(repo, qa, task="qa", answer_spec_id=QA_GENERAL_TEXT_SPEC_ID) == b"N001-V001,1250,yes\n"
    for bad in ([frame().to_dict(), frame(73).to_dict()], [dict(rows[0], timestamp_ms=999)],
                [dict(rows[0], batch_id="b01")], [dict(rows[0], rank=2)]):
        with pytest.raises(ValueError):
            serialize_b02_submission(repo, bad)
    with pytest.raises(ValueError):
        serialize_b02_submission(repo, rows, task="trake")


def test_ordinal_metadata_and_asr_window():
    repo = Repository()
    resolver = BatchedFrameResolver(repo)
    values = [resolver({"payload": frame(n).to_dict(), "score": .4}) for n in (72, 73)]
    assert [x["payload"]["keyframe_ordinal"] for x in values] == [72, 73]
    assert [x["score"] for x in values] == [.4, .4]
    result = frame_evidence(repo, "aic26-b2", "M08_V001", 0, keyframe_ordinal=73)
    assert result["anchor"]["keyframe_ordinal"] == 73
    assert (result["start_ms"], result["end_ms"]) == (0, 31250)


def test_feedback_passes_explicit_ordinal_not_source():
    calls = []
    def feedback(channel, keys, **kwargs):
        calls.append((channel, keys, kwargs))
        return np.array([[1., 0.] for _ in keys])
    pipeline = TextualKISPipeline(dense_channels=[DenseChannel(
        "organizer_clip", SimpleNamespace(), SimpleNamespace(vector_size=2), 1., 10)],
        feedback_provider=feedback)
    pipeline.search = lambda *a, **kw: []
    request = QueryRequest("q", "a person", batch_filters=("aic26-b2",), image_only=True,
                           reference_frame=FrameReference("aic26-b2", "M08_V001", 0, 73))
    pipeline.refine(request)
    assert calls == [("organizer_clip", (("M08_V001", 73),),
                      {"batch_id": "aic26-b2", "coordinate": "keyframe_ordinal"})]
    with pytest.raises(ValueError):
        pipeline.refine(request, positive_ids=(("M08_V001", 0),))


def test_session_csv_append_update_and_failed_replace_preserves_data():
    from app.components.b02_export import stage_b02_rows

    repo = Repository()
    def export(rows, **kwargs):
        return serialize_b02_submission(repo, rows, **kwargs)
    context = SimpleNamespace(artifact_repository=SimpleNamespace(b02_export=export))
    state = {}
    first = frame(1, "N001-V001").to_dict()
    second = frame(1, "S01-V001").to_dict()
    stage_b02_rows(context, state, "q.csv", [first], task="kis")
    before = state["b02_submission_files"]["q.csv"]
    with pytest.raises(ValueError):
        stage_b02_rows(context, state, "q.csv", [second], task="kis")
    assert state["b02_submission_files"]["q.csv"] == before
    stage_b02_rows(context, state, "q.csv", [second], task="kis", append=True)
    assert len(state["b02_submission_rows"]["q.csv"]) == 2
    with pytest.raises(ValueError):
        stage_b02_rows(context, state, "q.csv", [second], task="kis", update_rank=1)
    assert state["b02_submission_rows"]["q.csv"][0]["video_id"] == "N001-V001"


def test_qa_injection_manual_answer_retains_ordinal(tmp_path):
    repository = Repository()
    calls = []
    def resolve(f):
        calls.append(f.keyframe_ordinal)
        return SimpleNamespace(batch_id=f.batch_id, video_id=f.video_id,
                               source_frame_idx=f.source_frame_idx, timestamp_ms=f.timestamp_ms,
                               data=b"fixture", content_sha256="a"*64, mime_type="image/jpeg")
    sessions = QASessionStore(tmp_path / "qa.sqlite3")
    sessions.initialize()
    pipeline = VisualQAPipeline(retriever=VisualQARetriever(Textual(), repository),
                                image_resolver=SimpleNamespace(resolve=resolve), session_store=sessions,
                                competition_spec=QACompetitionSpec.conservative_general_text())
    request = QAFrameInjectionRequest("q", "aic26-b2", "M08_V001", 0, "kis_bridge", "person", 73)
    candidate = pipeline.start_injected_session(request, session_id="s")
    answer = pipeline.answer_candidate("s", QASingleFrameAnswerRequest(candidate.candidate_id, "what?",
        answer_spec_id=QA_GENERAL_TEXT_SPEC_ID, reasoning_policy="local_only", answer_engine="manual", manual_answer="yes"))
    assert answer["prediction"]["keyframe_ordinal"] == 73
    assert calls == [73]  # manual answer does not run any model or re-read image
    assert serialize_b02_submission(repository, [answer["prediction"]], task="qa",
                                    answer_spec_id=QA_GENERAL_TEXT_SPEC_ID) == b"M08_V001,0,yes\n"
    with pytest.raises(ValueError):
        pipeline.start_injected_session(replace(request, source_frame_idx=5))


def test_export_api_gates_and_payload_hash(tmp_path):
    context = SimpleNamespace(config={"competition": {"output_dir": str(tmp_path)},
        "readiness": {"batches": {"aic26-b2": {"retrieval_enabled": True}}}},
        pipeline=Textual(), ledger=SimpleNamespace(is_batch_active=lambda batch: True),
        artifact_repository=Repository())
    app = create_app(context)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/v1/b02/submissions/preview")
    payload = {"rows": [frame(1, "N001-V001").to_dict()]}
    result = endpoint(payload)
    data = base64.b64decode(result["csv_base64"])
    assert data == b"N001-V001,1250\n"
    assert hashlib.sha256(data).hexdigest() == result["sha256"]
    context.config["readiness"]["batches"]["aic26-b2"]["retrieval_enabled"] = False
    with pytest.raises(HTTPException) as error:
        endpoint(payload)
    assert error.value.status_code == 409
