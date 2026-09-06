"""Offline regression tests for retrieval-only and single-frame Visual Q&A."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.contracts import (
    QA_GENERAL_TEXT_SPEC_ID,
    QA_HISTORICAL_NUMERIC_SPEC_ID,
    FrameRecord,
    QACandidateRequest,
    QACompetitionSpec,
    QAQueryRequest,
    QAReasoningResult,
    QASingleFrameAnswerRequest,
)
from src.retrieval.qa_media import CanonicalImageError
from src.retrieval.qa_retrieval import VisualQARetriever
from src.retrieval.qa_session import QASessionStore
from src.retrieval.visual_qa import (
    QACandidateMediaUnavailableError,
    QAManualAnswerValidationError,
    VisualQAPipeline,
)
from src.vlm.florence2 import Florence2TaskReasoner
from src.vlm.gpu_coordinator import GPUModelCoordinator
from src.vlm.qa_reasoner import QAProviderRequestError, QAProviderUnavailableError
from src.vlm.qa_router import QAReasonerRouter


def _frame(path: Path, frame_idx: int = 10) -> FrameRecord:
    return FrameRecord(
        batch_id="b1",
        video_id="V001",
        keyframe_ordinal=1,
        source_frame_idx=frame_idx,
        pts=frame_idx,
        time_base="1/10",
        timestamp_ms=frame_idx * 100,
        fps_num=10,
        fps_den=1,
        shot_id="shot-1",
        sampling_reasons=("fixture",),
        image_uri=str(path),
        video_uri="fixture.mp4",
        source_sha256="a" * 64,
        extractor_version="fixture-v1",
    )


class _Repository:
    def __init__(self, frame: FrameRecord) -> None:
        self.frame = frame

    def get_frame(self, batch_id: str, video_id: str, source_frame_idx: int):
        identity = (batch_id, video_id, source_frame_idx)
        expected = (self.frame.batch_id, self.frame.video_id, self.frame.source_frame_idx)
        return self.frame if identity == expected else None


class _Textual:
    def __init__(self, frame: FrameRecord) -> None:
        self.frame = frame
        self.calls = 0
        self.last_channel_statuses = {
            "organizer_clip": {"state": "ready", "applicability": "required"}
        }
        self.last_channel_errors = {}

    def search(self, request, *, limit: int, refine_raw: bool):
        self.calls += 1
        assert request["original_text"] == "a news presenter"
        assert limit >= 5
        assert refine_raw is False
        return [
            {
                "batch_id": self.frame.batch_id,
                "video_id": self.frame.video_id,
                "source_frame_idx": self.frame.source_frame_idx,
                "fused_score": 0.9,
                "channel_ranks": {"organizer_clip": 1},
                "evidence": {"image_uri": "must-not-persist"},
            }
        ]


class _Resolver:
    def __init__(self) -> None:
        self.calls = 0
        output = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(output, "JPEG")
        self.data = output.getvalue()

    def resolve(self, frame: FrameRecord):
        self.calls += 1
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
    fingerprint = "single-frame-reasoner-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.last_metrics = {"handoff_ms": 3.0}

    def reason(self, request, images, spec):
        self.calls += 1
        assert len(images) == 1
        assert images[0].index == 1
        return QAReasoningResult(
            raw_answer="yes",
            normalized_answer=spec.normalize_answer("yes", request.question),
            answerable=True,
            evidence_frame_indices=(1,),
            provider="fixture",
            model_name="fixture-vlm",
            prompt_version="fixture-v1",
            latency_ms=5.0,
        )


def _pipeline(tmp_path: Path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 16), "red").save(image_path, "JPEG")
    frame = _frame(image_path)
    textual = _Textual(frame)
    repository = _Repository(frame)
    retriever = VisualQARetriever(textual, repository)
    resolver = _Resolver()
    reasoner = _Reasoner()
    sessions = QASessionStore(tmp_path / "qa.sqlite3")
    sessions.initialize()
    pipeline = VisualQAPipeline(
        retriever=retriever,
        reasoner=reasoner,
        image_resolver=resolver,
        session_store=sessions,
        competition_spec=QACompetitionSpec.conservative_general_text(),
    )
    return pipeline, sessions, textual, resolver, reasoner


def test_retrieval_stage_never_resolves_media_or_calls_provider(tmp_path: Path) -> None:
    pipeline, _sessions, textual, resolver, reasoner = _pipeline(tmp_path)
    request = QACandidateRequest(
        query_id="retrieve-only",
        event_description="a news presenter",
        batch_filters=("b1",),
        top_k=5,
    )

    candidates = pipeline.start_candidate_search(request, session_id="retrieve-only")

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "b1:V001:10"
    assert candidates[0].evidence == {"retrieval_channels": ["organizer_clip"]}
    assert textual.calls == 1
    assert resolver.calls == 0
    assert reasoner.calls == 0


def test_answer_stage_accepts_only_session_candidate_and_exactly_one_image(
    tmp_path: Path,
) -> None:
    pipeline, sessions, _textual, resolver, reasoner = _pipeline(tmp_path)
    pipeline.start_candidate_search(
        QACandidateRequest("q", "a news presenter", ("b1",), 5),
        session_id="answer-session",
    )
    request = QASingleFrameAnswerRequest(
        candidate_id="b1:V001:10",
        question="Does the man wear glasses?",
        reasoning_policy="local_only",
    )
    session = sessions.get_session("answer-session")
    assert session is not None
    stale_state = dict(session.state)
    stale_state["failures"] = {"b1:V001:10": "provider_request_failed"}
    sessions.update_state(
        "answer-session", stale_state, expected_revision=session.revision
    )

    first = pipeline.answer_candidate("answer-session", request)
    second = pipeline.answer_candidate("answer-session", request)

    assert first["prediction"]["answer"] == "yes"
    assert second["reasoning"]["cache_status"] == "hit"
    assert reasoner.calls == 1
    # First call resolves once for the image SHA and reuses those same bytes.
    # The cache replay still resolves once to validate current content identity.
    assert resolver.calls == 2
    snapshot = pipeline.get_search("answer-session")
    assert snapshot["state_version"] == 3
    assert snapshot["selected_candidate_id"] == "b1:V001:10"
    assert snapshot["failures"] == {}

    outsider = QASingleFrameAnswerRequest(
        candidate_id="b1:V001:11",
        question="Does the man wear glasses?",
        reasoning_policy="local_only",
    )
    with pytest.raises(ValueError, match="not part"):
        pipeline.answer_candidate("answer-session", outsider)
    assert reasoner.calls == 1


def test_manual_answer_is_normalized_audited_and_bypasses_media_and_vlm(
    tmp_path: Path,
) -> None:
    pipeline, sessions, _textual, resolver, reasoner = _pipeline(tmp_path)
    pipeline.start_candidate_search(
        QACandidateRequest("q-manual", "a news presenter", ("b1",), 5),
        session_id="manual-session",
    )
    request = QASingleFrameAnswerRequest(
        candidate_id="b1:V001:10",
        question="Địa điểm này ở đâu?",
        answer_spec_id=QA_GENERAL_TEXT_SPEC_ID,
        reasoning_policy="local_only",
        answer_engine="manual",
        manual_answer='  “XÃ Cam Hiệp Bắc!”\n',
    )

    first = pipeline.answer_candidate("manual-session", request)
    first_session = sessions.get_session("manual-session")
    assert first_session is not None
    second = pipeline.answer_candidate("manual-session", request)
    second_session = sessions.get_session("manual-session")
    assert second_session is not None

    assert first["prediction"]["answer"] == "xã cam hiệp bắc"
    assert first["prediction"]["answer_source"] == "operator"
    assert first["reasoning"]["cache_status"] == "bypass"
    assert second == first
    assert second_session.revision == first_session.revision
    assert resolver.calls == 0
    assert reasoner.calls == 0
    snapshot = pipeline.get_search("manual-session")
    assert snapshot["selected_candidate_id"] == "b1:V001:10"
    assert snapshot["predictions"][0]["answer"] == "xã cam hiệp bắc"
    assert len(snapshot["operator_edits"]) == 1
    assert snapshot["operator_edits"][0]["provider"] == "operator"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("  005  ", "5"),
        ("x" * 100, "x" * 100),
    ],
)
def test_manual_answer_uses_selected_competition_normalizer(
    tmp_path: Path,
    answer: str,
    expected: str,
) -> None:
    pipeline, _sessions, _textual, resolver, reasoner = _pipeline(tmp_path)
    pipeline.start_candidate_search(
        QACandidateRequest("q-normalize", "a news presenter", ("b1",), 5),
        session_id="manual-normalize",
    )
    numeric = answer.strip().isdigit()
    result = pipeline.answer_candidate(
        "manual-normalize",
        QASingleFrameAnswerRequest(
            candidate_id="b1:V001:10",
            question="Có bao nhiêu người?" if numeric else "Địa điểm này ở đâu?",
            answer_spec_id=(
                QA_HISTORICAL_NUMERIC_SPEC_ID
                if numeric
                else QA_GENERAL_TEXT_SPEC_ID
            ),
            reasoning_policy="local_only",
            answer_engine="manual",
            manual_answer=answer,
        ),
    )

    assert result["prediction"]["answer"] == expected
    assert resolver.calls == 0
    assert reasoner.calls == 0


def test_manual_answer_rejects_more_than_100_normalized_code_points(
    tmp_path: Path,
) -> None:
    pipeline, sessions, _textual, resolver, reasoner = _pipeline(tmp_path)
    pipeline.start_candidate_search(
        QACandidateRequest("q-long", "a news presenter", ("b1",), 5),
        session_id="manual-too-long",
    )

    with pytest.raises(QAManualAnswerValidationError, match="100 Unicode"):
        pipeline.answer_candidate(
            "manual-too-long",
            QASingleFrameAnswerRequest(
                candidate_id="b1:V001:10",
                question="Địa điểm này ở đâu?",
                answer_spec_id=QA_GENERAL_TEXT_SPEC_ID,
                reasoning_policy="local_only",
                answer_engine="manual",
                manual_answer="x" * 101,
            ),
        )

    assert resolver.calls == 0
    assert reasoner.calls == 0
    snapshot = sessions.get_snapshot("manual-too-long")
    assert snapshot is not None
    assert snapshot["predictions"] == []
    assert snapshot["operator_edits"] == []


def test_missing_candidate_media_is_persisted_without_calling_provider(
    tmp_path: Path,
) -> None:
    pipeline, _sessions, _textual, _resolver, reasoner = _pipeline(tmp_path)
    pipeline.start_candidate_search(
        QACandidateRequest("q-media", "a news presenter", ("b1",), 5),
        session_id="missing-media",
    )

    class MissingResolver:
        def resolve(self, _frame):
            raise CanonicalImageError("C:/private/missing-frame.jpg")

    pipeline.image_resolver = MissingResolver()
    request = QASingleFrameAnswerRequest(
        candidate_id="b1:V001:10",
        question="Does the man wear glasses?",
        reasoning_policy="local_only",
    )

    with pytest.raises(QACandidateMediaUnavailableError) as raised:
        pipeline.answer_candidate("missing-media", request)

    assert raised.value.candidate_id == "b1:V001:10"
    assert reasoner.calls == 0
    snapshot = pipeline.get_search("missing-media")
    assert snapshot["failures"] == {"b1:V001:10": "media_unavailable"}
    assert "private" not in str(snapshot)


def test_local_fallback_is_session_only_and_does_not_mask_gemini_recovery(
    tmp_path: Path,
) -> None:
    pipeline, _sessions, _textual, _resolver, _reasoner = _pipeline(tmp_path)

    class Gemini:
        provider_name = "gemini"
        model_name = "gemini-test"
        fingerprint = "gemini-test-v1"

        def __init__(self) -> None:
            self.calls = 0

        def reason(self, request, images, spec):
            self.calls += 1
            if self.calls == 1:
                raise QAProviderRequestError(
                    "safe", provider="gemini", category="network_unreachable"
                )
            return QAReasoningResult(
                raw_answer="yes",
                normalized_answer=spec.normalize_answer("yes", request.question),
                answerable=True,
                evidence_frame_indices=(1,),
                provider="gemini",
                model_name=self.model_name,
                prompt_version="test-v1",
                latency_ms=4.0,
            )

        def unload(self):
            return None

    class Local(_Reasoner):
        provider_name = "ollama-qwen"
        model_name = "qwen-test"
        fingerprint = "qwen-test-v1"

        def reason(self, request, images, spec):
            result = super().reason(request, images, spec)
            return QAReasoningResult.from_dict(
                {**result.to_dict(), "provider": self.provider_name, "model_name": self.model_name}
            )

        def unload(self):
            return None

    gemini = Gemini()
    local = Local()
    pipeline.reasoner = QAReasonerRouter(gemini, local)
    pipeline.start_candidate_search(
        QACandidateRequest("q-cache", "a news presenter", ("b1",), 5),
        session_id="cache-session",
    )
    request = QASingleFrameAnswerRequest(
        candidate_id="b1:V001:10",
        question="Does the man wear glasses?",
        reasoning_policy="gemini_then_local",
        allow_external_frame_transfer=True,
    )

    first = pipeline.answer_candidate("cache-session", request)
    second = pipeline.answer_candidate("cache-session", request)
    third = pipeline.answer_candidate("cache-session", request)

    assert first["reasoning"]["provider"] == "ollama-qwen"
    assert first["reasoning"]["fallback_used"] is True
    assert second["reasoning"]["provider"] == "gemini"
    assert second["reasoning"]["cache_status"] == "miss"
    assert third["reasoning"]["cache_status"] == "hit"
    assert gemini.calls == 2
    assert local.calls == 1


def test_legacy_v2_session_is_read_only_for_single_frame_answer(tmp_path: Path) -> None:
    pipeline, sessions, _textual, _resolver, reasoner = _pipeline(tmp_path)
    sessions.create_session(
        "legacy",
        QAQueryRequest(
            query_id="legacy",
            event_description="a news presenter",
            question="question",
            reasoning_policy="local_only",
        ).to_dict(),
        state={"state_version": 2},
    )
    with pytest.raises(RuntimeError, match="legacy_qa_session_read_only"):
        pipeline.answer_candidate(
            "legacy",
            QASingleFrameAnswerRequest(
                "b1:V001:10", "question", reasoning_policy="local_only"
            ),
        )
    assert reasoner.calls == 0


def test_gpu_coordinator_keeps_cpu_encoder_and_clears_at_threshold(monkeypatch) -> None:
    calls: list[str] = []

    class Encoder:
        def __init__(self, device: str) -> None:
            self.device = device

        def unload(self) -> None:
            calls.append(str(self.device))

    cpu = SimpleNamespace(name="organizer_clip", encoder=Encoder("cpu"))
    cuda = SimpleNamespace(name="visual", encoder=Encoder("cuda:0"))
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (10, 100),
        synchronize=lambda: calls.append("sync"),
        empty_cache=lambda: calls.append("empty"),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(cuda=fake_cuda))
    coordinator = GPUModelCoordinator(
        dense_channels=(cpu, cuda),
        local_reasoner=SimpleNamespace(unload=lambda: calls.append("qwen")),
        clear_threshold=0.8,
    )

    metrics = coordinator.before_vlm_inference("native-batch")

    assert "cpu" not in calls
    assert calls[:3] == ["cuda:0", "sync", "empty"]
    assert metrics["cuda_cache_cleared"] is True
    assert metrics["cuda_encoders_unloaded"] == 1
    assert coordinator.release_vlm_for_retrieval("b1-reference") is False


def test_florence_is_offline_and_rejects_general_questions(tmp_path: Path) -> None:
    reasoner = Florence2TaskReasoner(
        tmp_path / "missing",
        model_name="microsoft/Florence-2-base",
        revision="pinned",
    )
    with pytest.raises(QAProviderUnavailableError) as missing:
        reasoner.probe()
    assert missing.value.category == "model_missing"
    with pytest.raises(Exception) as raised:
        reasoner.reason(
            QAQueryRequest("q", "event", "Is this red?", reasoning_policy="local_only"),
            (),
            QACompetitionSpec.conservative_general_text(),
        )
    assert getattr(raised.value, "category", None) == "invalid_request"

    with pytest.raises(ValueError, match="florence_task_unsupported"):
        QASingleFrameAnswerRequest(
            candidate_id="b1:V001:10",
            question="Does the man wear glasses?",
            reasoning_policy="local_only",
            answer_engine="florence2",
            florence_task="ocr",
        )
