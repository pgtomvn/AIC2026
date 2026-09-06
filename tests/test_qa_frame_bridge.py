"""Offline contract tests for QA frame injection and dynamic retrieval depth."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.contracts import (
    FrameRecord,
    QACandidateRequest,
    QACompetitionSpec,
    QAFrameInjectionRequest,
    QAReasoningResult,
    QASingleFrameAnswerRequest,
)
from src.retrieval.qa_media import CanonicalImageError
from src.retrieval.qa_retrieval import VisualQARetriever
from src.retrieval.qa_session import QASessionStore
from src.retrieval.visual_qa import (
    QACandidateMediaUnavailableError,
    QAInjectedFrameNotFoundError,
    VisualQAPipeline,
)


def _frame(index: int) -> FrameRecord:
    return FrameRecord(
        batch_id="b1",
        video_id="L01_V028",
        keyframe_ordinal=index + 1,
        source_frame_idx=index,
        pts=index,
        time_base="1/25",
        timestamp_ms=index * 40,
        fps_num=25,
        fps_den=1,
        shot_id=f"shot-{index // 10}",
        sampling_reasons=("fixture",),
        image_uri=f"fixture/{index}.jpg",
        video_uri="fixture.mp4",
        source_sha256="a" * 64,
        extractor_version="fixture-v1",
    )


class _Repository:
    def __init__(self, count: int = 60) -> None:
        self.frames = {index: _frame(index) for index in range(count)}

    def get_frame(self, batch_id: str, video_id: str, source_frame_idx: int):
        if (batch_id, video_id) != ("b1", "L01_V028"):
            return None
        return self.frames.get(source_frame_idx)


class _Textual:
    def __init__(self, count: int = 60) -> None:
        self.count = count
        self.calls = 0
        self.last_channel_statuses = {}
        self.last_channel_errors = {}
        self.last_query_processing = {}

    def search(self, request, *, limit: int, refine_raw: bool):
        self.calls += 1
        assert refine_raw is False
        return [
            {
                "batch_id": "b1",
                "video_id": "L01_V028",
                "source_frame_idx": index,
                "fused_score": 1.0 / (index + 1),
                "channel_ranks": {"organizer_clip": index + 1},
            }
            for index in range(min(limit, self.count))
        ]


class _Resolver:
    def __init__(self, *, broken: bool = False) -> None:
        self.calls = 0
        self.broken = broken
        output = io.BytesIO()
        Image.new("RGB", (8, 8), "cyan").save(output, "JPEG")
        self.data = output.getvalue()

    def resolve(self, frame: FrameRecord):
        self.calls += 1
        if self.broken:
            raise CanonicalImageError("fixture image is unavailable")
        return SimpleNamespace(
            batch_id=frame.batch_id,
            video_id=frame.video_id,
            source_frame_idx=frame.source_frame_idx,
            timestamp_ms=frame.timestamp_ms,
            data=self.data,
            mime_type="image/jpeg",
            content_sha256=hashlib.sha256(self.data).hexdigest(),
        )


class _Reasoner:
    fingerprint = "bridge-reasoner-v1"

    def __init__(self) -> None:
        self.calls = 0

    def reason(self, request, images, spec):
        self.calls += 1
        assert len(images) == 1
        return QAReasoningResult(
            raw_answer="yes",
            normalized_answer="yes",
            answerable=True,
            evidence_frame_indices=(1,),
            provider="fixture",
            model_name="fixture-vlm",
            prompt_version="fixture-v1",
            latency_ms=1.0,
        )


def _pipeline(tmp_path: Path, *, broken_media: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = _Repository()
    textual = _Textual()
    retriever = VisualQARetriever(textual, repository)
    resolver = _Resolver(broken=broken_media)
    reasoner = _Reasoner()
    store = QASessionStore(tmp_path / "qa.sqlite3")
    store.initialize()
    pipeline = VisualQAPipeline(
        retriever=retriever,
        image_resolver=resolver,
        reasoner=reasoner,
        session_store=store,
        competition_spec=QACompetitionSpec.conservative_general_text(),
    )
    return pipeline, store, textual, resolver, reasoner


@pytest.mark.parametrize("top_k", (5, 10, 15, 20, 30, 50))
def test_dynamic_top_k_contract_and_retrieval_count(
    tmp_path: Path, top_k: int
) -> None:
    pipeline, _store, textual, resolver, reasoner = _pipeline(tmp_path)
    request = QACandidateRequest("q", "a news presenter", ("b1",), top_k)

    candidates = pipeline.start_candidate_search(request, session_id=f"top-{top_k}")

    assert len(candidates) == top_k
    assert textual.calls == 1
    assert resolver.calls == 0
    assert reasoner.calls == 0


@pytest.mark.parametrize("top_k", (0, 51, True))
def test_dynamic_top_k_rejects_out_of_contract_values(top_k: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        QACandidateRequest("q", "event", ("b1",), top_k)  # type: ignore[arg-type]


def test_injection_freezes_one_candidate_and_answers_without_stage1(
    tmp_path: Path,
) -> None:
    pipeline, store, textual, resolver, reasoner = _pipeline(tmp_path)
    request = QAFrameInjectionRequest(
        query_id="manual-bridge",
        batch_id="b1",
        video_id="L01_V028",
        source_frame_idx=10,
        source="manual",
        event_description="operator-selected canonical frame",
    )

    candidate = pipeline.start_injected_session(request, session_id="injected")

    assert candidate.candidate_id == "b1:L01_V028:10"
    assert candidate.evidence == {"origin": "manual"}
    assert textual.calls == 0
    assert resolver.calls == 1
    assert reasoner.calls == 0
    snapshot = store.get_snapshot("injected")
    assert snapshot is not None
    assert snapshot["selected_candidate_id"] == candidate.candidate_id
    assert len(snapshot["session_candidates"]) == 1

    manual = pipeline.answer_candidate(
        "injected",
        QASingleFrameAnswerRequest(
            candidate_id=candidate.candidate_id,
            question="Does the man wear glasses?",
            reasoning_policy="local_only",
            answer_engine="manual",
            manual_answer="Yes",
        ),
    )
    assert manual["prediction"]["answer"] == "yes"
    assert resolver.calls == 1
    assert reasoner.calls == 0

    result = pipeline.answer_candidate(
        "injected",
        QASingleFrameAnswerRequest(
            candidate_id=candidate.candidate_id,
            question="Does the man wear glasses?",
            reasoning_policy="local_only",
        ),
    )
    assert result["prediction"]["answer"] == "yes"
    assert resolver.calls == 2
    assert reasoner.calls == 1


def test_injection_rejects_unknown_or_unreadable_canonical_frame(
    tmp_path: Path,
) -> None:
    pipeline, _store, textual, resolver, reasoner = _pipeline(tmp_path)
    missing = QAFrameInjectionRequest(
        "missing", "b1", "L01_V028", 999, "manual", "selected frame"
    )
    with pytest.raises(QAInjectedFrameNotFoundError):
        pipeline.start_injected_session(missing, session_id="missing")
    assert textual.calls == resolver.calls == reasoner.calls == 0

    broken, _store, textual, resolver, reasoner = _pipeline(
        tmp_path / "broken", broken_media=True
    )
    request = QAFrameInjectionRequest(
        "broken", "b1", "L01_V028", 10, "kis_bridge", "news presenter"
    )
    with pytest.raises(QACandidateMediaUnavailableError):
        broken.start_injected_session(request, session_id="broken")
    assert textual.calls == 0
    assert resolver.calls == 1
    assert reasoner.calls == 0


def test_injection_contract_rejects_ordinal_aliases() -> None:
    with pytest.raises(ValueError, match="source_frame_idx"):
        QAFrameInjectionRequest.from_dict(
            {
                "query_id": "ordinal",
                "batch_id": "b1",
                "video_id": "L01_V028",
                "frame_idx": 10,
                "source": "manual",
                "event_description": "selected frame",
            }
        )
