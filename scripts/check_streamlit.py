"""Execute the real Streamlit app script using Streamlit's offline AppTest API."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from src.codabench_query_id import codabench_query_filename
from src.contracts import QACompetitionSpec
from src.query.entity_extractor import entity_identity


class _Pipeline:
    last_channel_errors: dict[str, str] = {}

    def __init__(self) -> None:
        self.requests = []

    def search(self, request, *, limit=None, refine_raw=True):
        self.requests.append(request)
        if request.original_text == "a person holding fruit":
            return []
        return [
            {
                "rank": rank,
                "batch_id": "aic25-b1-reference",
                "video_id": f"L21_V{rank:03d}",
                "source_frame_idx": rank * 10,
                "timestamp_ms": rank * 1_000,
                "shot_id": f"shot-{rank}",
                "fused_score": 1.0 / rank,
                "channel_ranks": {"organizer_clip": rank},
                "evidence": {},
            }
            for rank in range(1, min(int(limit or 7), 7) + 1)
        ]


class _TRAKEPipeline:
    def __init__(self) -> None:
        self.requests = []

    @staticmethod
    def channel_readiness(_batch_id):
        return True, {"organizer_clip": {"state": "ready"}}

    def search(self, request, *, default_batch_id=None):
        self.requests.append(request)
        batch_id = request.batch_filters[0] if request.batch_filters else default_batch_id
        matches = [
            {
                "stage_index": stage.stage_index,
                "source_frame_idx": stage.stage_index * 10,
                "timestamp_ms": stage.stage_index * 1_000,
                "similarity": 0.8 - stage.stage_index / 100,
            }
            for stage in request.stages
        ]
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "query_id": request.query_id,
                "results": [
                    {
                        "rank": 1,
                        "batch_id": batch_id,
                        "video_id": "L21_V001",
                        "stage_matches": matches,
                        "frame_ids": [item["source_frame_idx"] for item in matches],
                        "confidence_score": 0.9,
                    }
                ],
                "channel_statuses": {"organizer_clip": {"state": "ready"}},
            }
        )


class _QAPipeline:
    def __init__(self) -> None:
        self.last_snapshot = None
        self.retrieve_calls = 0
        self.answer_calls = 0
        self.manual_calls = 0
        self.inject_calls = 0
        self.last_request = None

    def retrieve(self, request, *, session_id=None):
        del session_id
        self.retrieve_calls += 1
        self.last_request = request
        candidates = [
            {
                "candidate_id": f"aic25-b1-reference:L21_V{rank:03d}:{rank * 10}",
                "retrieval_rank": rank,
                "retrieval_score": 1.0 / rank,
                "batch_id": "aic25-b1-reference",
                "video_id": f"L21_V{rank:03d}",
                "source_frame_idx": rank * 10,
                "timestamp_ms": rank * 1_000,
                "shot_id": f"shot-{rank}",
                "channel_ranks": {"organizer_clip": rank},
                "evidence": {},
            }
            for rank in range(1, int(request.top_k) + 1)
        ]
        self.last_snapshot = {
            "search_id": "qa-apptest-session",
            "state_version": 3,
            "status": "candidates_ready",
            "request": request.to_dict(),
            "candidates": candidates,
            "answers": [],
            "predictions": [],
            "stage_timings": {"candidate_retrieval_ms": 10.0},
        }
        return candidates

    def inject(self, request, *, session_id=None):
        del session_id
        self.inject_calls += 1
        self.last_request = request
        candidate_id = (
            f"{request.batch_id}:{request.video_id}:{request.source_frame_idx}"
        )
        candidate = {
            "candidate_id": candidate_id,
            "retrieval_rank": 1,
            "retrieval_score": 0.0,
            "batch_id": request.batch_id,
            "video_id": request.video_id,
            "source_frame_idx": request.source_frame_idx,
            "timestamp_ms": request.source_frame_idx * 100,
            "shot_id": "operator",
            "channel_ranks": {},
            "evidence": {"origin": request.source},
        }
        self.last_snapshot = {
            "search_id": "qa-injected-session",
            "state_version": 3,
            "status": "candidates_ready",
            "selected_candidate_id": candidate_id,
            "request": request.to_dict(),
            "candidates": [candidate],
            "answers": [],
            "predictions": [],
            "stage_timings": {"candidate_retrieval_ms": 0.0},
        }
        return [candidate]

    def answer_candidate(self, search_id, request):
        assert search_id == "qa-apptest-session"
        assert request.candidate_id == "aic25-b1-reference:L21_V001:10"
        if request.answer_engine == "manual":
            self.manual_calls += 1
            answer = QACompetitionSpec.conservative_general_text().normalize_answer(
                request.manual_answer,
                request.question,
            )
            provider = "operator"
        else:
            self.answer_calls += 1
            answer = "yes"
            provider = "ollama-qwen"
        prediction = {
            "rank": 1,
            "batch_id": "aic25-b1-reference",
            "video_id": "L21_V001",
            "source_frame_idx": 10,
            "timestamp_ms": 1_000,
            "answer": answer,
            "answer_source": provider,
            "confidence_score": 1.0,
            "window_id": request.candidate_id,
            "evidence": {"provider": provider, "latency_ms": 0.0},
        }
        self.last_snapshot = {
            **self.last_snapshot,
            "status": "answered",
            "selected_candidate_id": request.candidate_id,
            "answers": [{"candidate_id": request.candidate_id, "status": "answered"}],
            "predictions": [prediction],
        }
        return [prediction]

    def get_search(self, search_id):
        return self.last_snapshot if search_id == "qa-apptest-session" else None


class _Repository:
    def __init__(self) -> None:
        self.adjacent_calls = 0

    @staticmethod
    def existing_source_frames(_batch_id, identities, *, include_refined=True):
        assert include_refined is True
        return frozenset(identities)

    @staticmethod
    def frame_index_mappings_for_sources(_batch_id, identities):
        result = {}
        for ordinal, (video_id, source_frame_idx) in enumerate(identities, 1):
            result.setdefault(video_id, {})[ordinal] = source_frame_idx
        return result

    @staticmethod
    def get_source_frame(batch_id, video_id, source_frame_idx):
        return SimpleNamespace(
            batch_id=batch_id,
            video_id=video_id,
            source_frame_idx=source_frame_idx,
            timestamp_ms=source_frame_idx * 100,
            shot_id="fixture-shot",
        )

    get_frame = get_source_frame

    @staticmethod
    def neighbors(batch_id, video_id, timestamp_ms, radius_ms=4_000, limit=25):
        del radius_ms, limit
        return [
            SimpleNamespace(
                batch_id=batch_id,
                video_id=video_id,
                source_frame_idx=max(0, int(timestamp_ms) // 100),
                timestamp_ms=int(timestamp_ms),
                shot_id="fixture-shot",
            )
        ]

    def adjacent_frames(
        self,
        batch_id,
        video_id,
        source_frame_idx,
        *,
        radius=5,
    ):
        assert radius == 5
        self.adjacent_calls += 1
        return [
            (
                offset,
                self.get_source_frame(
                    batch_id,
                    video_id,
                    source_frame_idx + offset * 10,
                ),
            )
            for offset in (-1, 0, 1)
        ]


class _ImageResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, frame):
        self.calls += 1
        return SimpleNamespace(
            data=base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
            timestamp_ms=frame.timestamp_ms,
        )


class _ReadinessReport:
    def __init__(self, value):
        self.value = copy.deepcopy(value)

    def to_dict(self):
        return copy.deepcopy(self.value)


class _Readiness:
    def __init__(self, multimodal_state: str) -> None:
        healthy = multimodal_state == "healthy"
        unavailable = [] if healthy else ["asr"]
        self.report = _ReadinessReport(
            {
                "kis": {"state": "ready"},
                "qa": {"state": "ready"},
                "trake": {"state": "ready"},
                "channels": [],
                "multimodal": {
                    "status": "healthy" if healthy else "partial",
                    "status_label": (
                        "HEALTHY · Full Multi-Modal"
                        if healthy
                        else "DEGRADED · Multi-Modal Incomplete"
                    ),
                    "expected_videos": 873,
                    "full_multimodal": {
                        "healthy": healthy,
                        "unavailable_channels": unavailable,
                    },
                    "siglip2": {
                        "canonical_frames": 176707,
                        "expected_frames": 176707,
                    },
                    "ocr": {
                        "scanned_canonical_frames": 176707,
                        "expected_frames": 176707,
                    },
                    "asr": {
                        "source_videos": 873,
                        "completed_videos": 873 if healthy else 0,
                        "no_audio_videos": 0,
                        "missing_source_videos": 0,
                    },
                },
            }
        )

    def check(self, _batch_id, *, mode="app"):
        assert mode == "app"
        return self.report

    @staticmethod
    def workflow_readiness(_report, _workflow):
        return _ReadinessReport({"state": "ready"})


def _context(*, canonical: bool, multimodal_state: str | None = None):
    selected_batch_id = "aic25-b1-reference"
    pipeline = _Pipeline()

    def validate_batch_filters(values=None):
        result = tuple(values or (selected_batch_id,))
        if result != (selected_batch_id,):
            raise ValueError("test context accepts only the selected batch")
        return result

    bundle_requests = []
    staging_snapshot = {
        "schema_version": 1,
        "revision": 0,
        "updated_at_utc": None,
        "entries": {},
    }

    def load_submission_staging():
        return copy.deepcopy(staging_snapshot)

    def stage_submission_entry(payload):
        entry = payload["entry"]
        assert payload["expected_revision"] == staging_snapshot["revision"]
        filename = codabench_query_filename(entry["query_id"], entry["task"])
        data = base64.b64decode(entry["csv_base64"], validate=True)
        staging_snapshot["revision"] += 1
        staging_snapshot["updated_at_utc"] = "2026-08-20T09:30:00Z"
        staging_snapshot["entries"][filename] = {
            **entry,
            "filename": filename,
            "batch_id": payload["batch_id"],
            "status": "READY",
            "row_count": len(data.decode("utf-8").splitlines()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "updated_at_utc": "2026-08-20T09:30:00Z",
        }
        return load_submission_staging()

    def delete_submission_entry(filename, *, expected_revision):
        assert expected_revision == staging_snapshot["revision"]
        staging_snapshot["entries"].pop(filename, None)
        staging_snapshot["revision"] += 1
        return load_submission_staging()

    def clear_submission_staging(*, expected_revision):
        assert expected_revision == staging_snapshot["revision"]
        staging_snapshot["entries"] = {}
        staging_snapshot["revision"] += 1
        return load_submission_staging()

    def create_submission_bundle(payload):
        bundle_requests.append(payload)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("submission/", b"")
        data = buffer.getvalue()
        return {
            "run_id": "run_20260817T120000123456Z",
            "filename": "AIC2026_submission_20260817T120000123456Z.zip",
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
            "file_count": len(payload.get("entry_filenames", payload.get("entries", ()))),
            "status": "valid",
            "data": data,
        }

    return SimpleNamespace(
        pipeline=pipeline,
        trake_pipeline=_TRAKEPipeline(),
        artifact_repository=_Repository(),
        qa_pipeline=_QAPipeline(),
        qa_session_store=None,
        qa_image_resolver=_ImageResolver(),
        readiness_service=(
            _Readiness(multimodal_state) if multimodal_state is not None else None
        ),
        selected_batch_id=selected_batch_id,
        validate_batch_filters=validate_batch_filters,
        launch_identity={
            "canonical": canonical,
            "contract": "run_app-v2" if canonical else "direct-streamlit",
            "launch_id": "offline-apptest",
            "port": 8501,
            "batch_id": selected_batch_id,
            "profile": "laptop",
        },
        config={
            "profile": "laptop",
            "competition": {"output_dir": "outputs/submissions"},
            "retrieval": {"visual_qa": {}},
        },
        ledger=SimpleNamespace(is_batch_active=lambda _batch_id: True),
        create_submission_bundle=create_submission_bundle,
        load_submission_staging=load_submission_staging,
        stage_submission_entry=stage_submission_entry,
        delete_submission_entry=delete_submission_entry,
        clear_submission_staging=clear_submission_staging,
        staging_snapshot=staging_snapshot,
        bundle_requests=bundle_requests,
    )


def _messages(elements) -> list[str]:
    return [str(getattr(item, "value", "")) for item in elements]


def main() -> int:
    os.chdir(PROJECT_ROOT)
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "app").rglob("*.py")
    )
    if "use_container_width" in source:
        raise RuntimeError("Streamlit source still uses deprecated use_container_width")
    forbidden_qa_render_paths = (
        "st.dataframe(predictions",
        "st.json(st.session_state.qa_edits",
        '{"channel": name, **dict(status)}',
    )
    for render_path in forbidden_qa_render_paths:
        if render_path in source:
            raise RuntimeError(
                "Streamlit source renders a Visual Q&A persistence object wholesale: "
                + render_path
            )

    stale_app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30
    )
    stale_context = _context(canonical=True)
    stale_context.is_remote = True
    stale_context.capabilities = {}
    retained_candidates = [{"candidate_id": "retained-candidate"}]
    retained_staging = {"query-p1-15-qa.csv": {"status": "READY"}}
    stale_app.session_state["_aic_runtime_override"] = stale_context
    stale_app.session_state["qa_candidates"] = retained_candidates
    stale_app.session_state["vqa_final_answer_input"] = "operator draft"
    stale_app.session_state["codabench_staging"] = retained_staging
    stale_app.run()
    if stale_app.exception:
        messages = [getattr(item, "message", str(item)) for item in stale_app.exception]
        raise RuntimeError("stale-contract AppTest crashed: " + " | ".join(messages))
    if not any(
        "runtime_contract_mismatch" in value for value in _messages(stale_app.error)
    ):
        raise RuntimeError("stale runtime contract did not render the restart banner")
    if stale_context.pipeline.requests or stale_context.qa_pipeline.retrieve_calls:
        raise RuntimeError("stale runtime contract dispatched a retrieval request")
    if stale_app.session_state["qa_candidates"] != retained_candidates:
        raise RuntimeError("stale runtime contract cleared retained QA candidates")
    if stale_app.session_state["vqa_final_answer_input"] != "operator draft":
        raise RuntimeError("stale runtime contract cleared the operator draft")
    if stale_app.session_state["codabench_staging"] != retained_staging:
        raise RuntimeError("stale runtime contract cleared durable staging projection")

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    context = _context(canonical=True)
    app.session_state["_aic_runtime_override"] = context
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("Streamlit AppTest failed: " + " | ".join(messages))

    full_health_app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30
    )
    full_health_app.session_state["_aic_runtime_override"] = _context(
        canonical=True, multimodal_state="healthy"
    )
    full_health_app.run()
    if full_health_app.exception:
        messages = [getattr(item, "message", str(item)) for item in full_health_app.exception]
        raise RuntimeError("full-multimodal AppTest failed: " + " | ".join(messages))
    if not any(
        "HEALTHY · Full Multi-Modal" in value
        for value in _messages(full_health_app.success)
    ):
        raise RuntimeError("Full Multi-Modal health label is not visible in Streamlit")

    degraded_health_app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30
    )
    degraded_health_app.session_state["_aic_runtime_override"] = _context(
        canonical=True, multimodal_state="degraded"
    )
    degraded_health_app.run()
    if degraded_health_app.exception:
        messages = [getattr(item, "message", str(item)) for item in degraded_health_app.exception]
        raise RuntimeError("degraded-multimodal AppTest failed: " + " | ".join(messages))
    degraded_messages = _messages(degraded_health_app.warning)
    if not any(
        "DEGRADED · Multi-Modal Incomplete" in value and "asr" in value
        for value in degraded_messages
    ):
        raise RuntimeError("degraded Multi-Modal label does not identify the missing channel")

    routing_widget_keys = {
        getattr(widget, "key", None)
        for collection in (app.radio, app.selectbox)
        for widget in collection
    }
    if {"kis_mode", "kis_precision_profile"} & routing_widget_keys:
        raise RuntimeError("legacy KIS routing selectors are still visible")
    workflow = app.segmented_control(key="active_workflow")
    if tuple(workflow.options) != (
        "[ KIS ENGINE ]",
        "[ VISUAL QA ]",
        "[ TRAKE TIMELINE ]",
    ):
        raise RuntimeError("Streamlit workflow selector is incomplete")
    if workflow.value != "kis":
        raise RuntimeError("Streamlit did not preserve the default KIS workflow")
    no_match_message = "Không có kết quả đã lập chỉ mục phù hợp"
    if any(no_match_message in value for value in _messages(app.info)):
        raise RuntimeError("initial KIS view incorrectly reports an empty completed search")
    if not any("run_app-v2" in value for value in _messages(app.caption)):
        raise RuntimeError("canonical launch identity is not visible")
    collapsed = {block.label: block.proto.expanded for block in app.get("expander")}
    for label in ("Tùy chọn tìm kiếm", "Bài nộp · CSV / ZIP"):
        if collapsed.get(label) is not False:
            raise RuntimeError(f"secondary controls should start collapsed: {label}")
    if any("Nhập mô tả và chọn Tìm kiếm" in value for value in _messages(app.caption)):
        raise RuntimeError("redundant initial KIS tutorial is still visible")
    app.text_area(key="query_text").set_value(
        "CLB FANA tổ chức từ thiện tại tỉnh Khánh Hòa"
    )
    app.run()
    kis_entity_key = "entity_mode_" + hashlib.sha256(
        f"kis:{entity_identity('FANA')}".encode("utf-8")
    ).hexdigest()[:12]
    app.button(key=kis_entity_key).click()
    app.run()
    app.button(key="query_submit").click()
    app.run()
    if context.pipeline.requests[-1].entity_constraints[0].mode != "required":
        raise RuntimeError("KIS entity Required toggle was not sent to FastAPI")
    scrub_identity = ("aic25-b1-reference", "L21_V001", "10")
    if context.pipeline.requests[-1].mode != "automatic":
        raise RuntimeError("KIS request was not forced through automatic routing")
    scrub_action_digest = hashlib.sha256(
        "\0".join(scrub_identity).encode("utf-8")
    ).hexdigest()[:20]
    app.button(key=f"kis_scrub_{scrub_action_digest}").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("KIS temporal scrubber AppTest failed: " + " | ".join(messages))
    if context.artifact_repository.adjacent_calls != 1:
        raise RuntimeError("KIS scrubber did not lazily request adjacency exactly once")
    if len(app.session_state["kis_scrub_frames"]) != 3:
        raise RuntimeError("KIS scrubber did not retain canonical adjacency metadata")
    scrub_panel_digest = scrub_action_digest[:16]
    app.button(key=f"kis_scrub_quick_{scrub_panel_digest}_1").click()
    app.run()
    if app.session_state["kis_scrub_offset"] != 1:
        raise RuntimeError("KIS scrubber did not preserve the selected ordinal offset")
    app.button(key=f"kis_scrub_pin_{scrub_panel_digest}_1").click()
    app.run()
    neighbor = next(
        (
            item
            for item in app.session_state["kis_pinned"]
            if item.get("source_frame_idx") == 20
        ),
        None,
    )
    if neighbor is None or neighbor.get("provenance", {}).get("source") != (
        "temporal_neighbor"
    ):
        raise RuntimeError("KIS temporal neighbor pin lost its provenance")
    if "score" in neighbor or "channel_ranks" in neighbor:
        raise RuntimeError("KIS temporal neighbor copied anchor ranking evidence")

    app.session_state["qa_question"] = "Does the man wear glasses?"
    app.session_state["codabench_query_id_qa"] = "p1-15"
    bridge_key = "kis_to_qa_" + hashlib.sha256(
        "\0".join(("aic25-b1-reference", "L21_V007", "70")).encode("utf-8")
    ).hexdigest()[:20]
    app.button(key=bridge_key).click()
    app.run()
    if app.session_state["active_workflow"] != "qa":
        raise RuntimeError("KIS bridge did not switch to Visual Q&A")
    if context.qa_pipeline.inject_calls != 1:
        raise RuntimeError("KIS bridge did not use the authoritative injection endpoint")
    if app.session_state["qa_selected_candidate_id"] != (
        "aic25-b1-reference:L21_V007:70"
    ):
        raise RuntimeError("KIS candidate #7 was not selected after the bridge")
    if app.session_state["qa_question"] != "Does the man wear glasses?":
        raise RuntimeError("KIS bridge cleared the existing QA question")
    # Streamlit AppTest retains stale KIS widget nodes across an in-script tab
    # redirect. Continue the independent QA controls in a fresh browser session;
    # the bridge state itself was asserted above.
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30
    )
    app.session_state["_aic_runtime_override"] = context
    app.session_state["active_workflow"] = "qa"
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("QA workflow AppTest failed: " + " | ".join(messages))
    app.text_area(key="qa_event_description").set_value(
        "CLB FANA tổ chức từ thiện tại tỉnh Khánh Hòa"
    )
    app.run()
    qa_entity_key = "entity_mode_" + hashlib.sha256(
        f"qa:{entity_identity('FANA')}".encode("utf-8")
    ).hexdigest()[:12]
    app.button(key=qa_entity_key).click()
    app.run()
    app.text_input(key="qa_question").set_value("Does the man wear glasses?")
    app.select_slider(key="qa_top_k").set_value(50)
    image_calls_before = context.qa_image_resolver.calls
    app.button(key="qa_retrieve_candidates").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("QA candidate retrieval AppTest failed: " + " | ".join(messages))
    if context.qa_pipeline.retrieve_calls != 1 or context.qa_pipeline.answer_calls != 0:
        raise RuntimeError("QA Stage 1 did not remain retrieval-only")
    if context.qa_pipeline.last_request.entity_constraints[0].mode != "required":
        raise RuntimeError("QA entity Required toggle was not persisted in retrieval")
    if len(app.session_state["qa_candidates"]) != 50:
        raise RuntimeError("QA Top-50 candidates were not persisted across the rerun")
    if context.qa_pipeline.last_request.top_k != 50:
        raise RuntimeError("QA dynamic Top-K was not sent to FastAPI")
    if context.qa_image_resolver.calls - image_calls_before > 12:
        raise RuntimeError("QA page loaded more than 12 candidate thumbnails")
    if app.radio(key="qa_selected_candidate_id").value != (
        "aic25-b1-reference:L21_V001:10"
    ):
        raise RuntimeError("QA did not select the stable Top-1 candidate identity")
    if app.selectbox(key="qa_engine").value != "gemini":
        raise RuntimeError("Gemini Flash is not the default QA answer engine")
    app.text_input(key="vqa_final_answer_input").set_value("operator draft")
    app.select_slider(key="qa_candidate_page").set_value(2)
    page_calls_before = context.qa_image_resolver.calls
    app.run()
    if app.session_state["qa_selected_candidate_id"] != (
        "aic25-b1-reference:L21_V001:10"
    ):
        raise RuntimeError("QA pagination lost the selected candidate")
    if app.session_state["vqa_final_answer_input"] != "operator draft":
        raise RuntimeError("QA pagination lost the operator answer draft")
    if context.qa_image_resolver.calls - page_calls_before > 13:
        raise RuntimeError("QA page loaded more than 12 thumbnails plus selected preview")
    app.checkbox(key="qa_external_transfer_consent").set_value(True)
    app.run()
    app.button(key="qa_answer_selected_candidate").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("QA single-frame answering AppTest failed: " + " | ".join(messages))
    if context.qa_pipeline.answer_calls != 1:
        raise RuntimeError("QA Stage 2 did not invoke exactly one provider call")
    if app.session_state["qa_result"]["answer"] != "yes":
        raise RuntimeError("QA answer was not persisted across the answer rerun")
    app.run()
    if app.session_state["qa_result"]["answer"] != "yes":
        raise RuntimeError("QA result was lost on an unrelated rerun")
    app.text_input(key="codabench_query_id_qa").set_value("p1-15")
    app.text_input(key="vqa_final_answer_input").set_value("xã Cam Hiệp Bắc")
    app.button(key="qa_confirm_manual_and_stage").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("QA manual staging AppTest failed: " + " | ".join(messages))
    if context.qa_pipeline.answer_calls != 1 or context.qa_pipeline.manual_calls != 1:
        raise RuntimeError("manual QA staging called the VLM or skipped its audit request")
    if app.session_state["qa_result"]["answer"] != "xã cam hiệp bắc":
        raise RuntimeError("manual QA answer was not normalized and selected")
    staged = app.session_state["codabench_staging"].get("query-p1-15-qa.csv")
    if staged is None:
        raise RuntimeError("alphanumeric manual QA entry was not staged")
    if bytes(staged["data"]).decode("utf-8") != (
        "L21_V001,10,xã cam hiệp bắc\n"
    ):
        raise RuntimeError("manual QA staging did not serialize the selected candidate")
    if not any(
        "L21_V001,10,xã cam hiệp bắc" in value for value in _messages(app.code)
    ):
        raise RuntimeError("Visual Q&A did not display the serialized RFC 4180 preview")
    for alternate in ("Cam Bắc", "Cam Bắc Xã"):
        app.text_input(key="vqa_final_answer_input").set_value(alternate)
        app.button(key="qa_confirm_manual_and_stage").click()
        app.run()
        if app.exception:
            messages = [getattr(item, "message", str(item)) for item in app.exception]
            raise RuntimeError("QA multi-rank append failed: " + " | ".join(messages))
    staged = app.session_state["codabench_staging"]["query-p1-15-qa.csv"]
    expected_three = (
        "L21_V001,10,xã cam hiệp bắc\n"
        "L21_V001,10,cam bắc\n"
        "L21_V001,10,cam bắc xã\n"
    )
    if bytes(staged["data"]).decode("utf-8") != expected_three:
        raise RuntimeError(
            "QA alternate answers were not appended as ranks 1..3: "
            + repr(bytes(staged["data"]).decode("utf-8"))
        )
    delete_key = f"qa_staged_delete_{str(staged['sha256'])[:10]}_2"
    app.button(key=delete_key).click()
    app.run()
    staged = app.session_state["codabench_staging"]["query-p1-15-qa.csv"]
    expected_two = (
        "L21_V001,10,xã cam hiệp bắc\n"
        "L21_V001,10,cam bắc xã\n"
    )
    if bytes(staged["data"]).decode("utf-8") != expected_two:
        raise RuntimeError("deleting QA Rank #2 did not reindex the remaining rows")
    if context.qa_pipeline.answer_calls != 1 or context.qa_pipeline.manual_calls != 3:
        raise RuntimeError("multi-rank QA staging unexpectedly called the VLM")
    app.run()
    if "query-p1-15-qa.csv" not in app.session_state["codabench_staging"]:
        raise RuntimeError("manual QA staging was lost on an unrelated rerun")
    refreshed = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30
    )
    refreshed.session_state["_aic_runtime_override"] = context
    refreshed.run()
    if "query-p1-15-qa.csv" not in refreshed.session_state["codabench_staging"]:
        raise RuntimeError("disk-backed staging was not restored after a fresh UI session")
    app.segmented_control(key="active_workflow").set_value("trake")
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("TRAKE workflow AppTest failed: " + " | ".join(messages))
    if app.text_area(key="trake_global_description").label != "Mô tả toàn bộ sự kiện":
        raise RuntimeError("TRAKE global description is not rendered in Vietnamese")
    if app.text_input(key="trake_stage_1").label != "Giai đoạn 1":
        raise RuntimeError("TRAKE stage editor is not rendered in Vietnamese")
    if app.button(key="trake_search").disabled:
        raise RuntimeError("ready TRAKE runtime did not unlock its search button")
    if not app.button(key="trake_remove_stage").disabled:
        raise RuntimeError("TRAKE allowed fewer than two stages")
    app.button(key="trake_add_stage").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("TRAKE add-stage AppTest failed: " + " | ".join(messages))
    if app.text_input(key="trake_stage_3").label != "Giai đoạn 3":
        raise RuntimeError("TRAKE add-stage control did not create stage 3")
    if app.button(key="trake_remove_stage").disabled:
        raise RuntimeError("TRAKE remove-stage control did not unlock above two stages")
    app.button(key="trake_remove_stage").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("TRAKE remove-stage AppTest failed: " + " | ".join(messages))
    if int(app.session_state["trake_stage_count"]) != 2:
        raise RuntimeError("TRAKE remove-stage control did not restore two stages")
    if not app.button(key="trake_remove_stage").disabled:
        raise RuntimeError("TRAKE remove-stage control did not lock at two stages")
    if app.segmented_control(key="active_workflow").value != "trake":
        raise RuntimeError("TRAKE workflow selection was lost after stage reruns")
    app.text_area(key="trake_global_description").set_value("CLB FANA tổ chức sự kiện")
    app.run()
    trake_entity_key = "entity_mode_" + hashlib.sha256(
        f"global:{entity_identity('FANA')}".encode("utf-8")
    ).hexdigest()[:12]
    app.button(key=trake_entity_key).click()
    app.run()
    app.text_input(key="trake_stage_1").set_value("Người bước vào cửa hàng")
    app.text_input(key="trake_stage_2").set_value("Người thanh toán")
    app.button(key="trake_search").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("non-empty TRAKE AppTest failed: " + " | ".join(messages))
    if context.trake_pipeline.requests[-1].global_entity_constraints[0].mode != "required":
        raise RuntimeError("TRAKE prompt-scoped entity toggle was not sent to the backend")
    if app.selectbox(key="trake_selected_candidate").value != (
        "aic25-b1-reference:L21_V001"
    ):
        raise RuntimeError("TRAKE did not select the first stable candidate identity")
    app.text_input(key="codabench_query_id_trake").set_value("003")
    app.run()
    app.button(key="trake_pin_selected").click()
    app.run()
    if "query-003-trake.csv" not in app.session_state["codabench_staging"]:
        raise RuntimeError("TRAKE pin manager did not persist the selected sequence")
    if not app.download_button(key="trake_download"):
        raise RuntimeError("pinned TRAKE sequence did not expose a CSV download")
    if not app.button(key="codabench_build_bundle").disabled:
        raise RuntimeError("full Codabench bundle unlocked without KIS and QA entries")
    app.radio(key="codabench_bundle_mode").set_value("current")
    app.run()
    if app.button(key="codabench_build_bundle").disabled:
        raise RuntimeError("current-tab Codabench bundle did not unlock for a ready entry")
    app.button(key="codabench_build_bundle").click()
    app.run()
    if len(context.bundle_requests) != 1:
        raise RuntimeError("Codabench Hub did not invoke the thin-client backend exactly once")
    if not app.download_button(key="codabench_download_bundle"):
        raise RuntimeError("Codabench bundle download did not persist after the build rerun")
    app.run()
    if not app.download_button(key="codabench_download_bundle"):
        raise RuntimeError("Codabench bundle download was lost on an unrelated rerun")
    if not app.download_button(key="codabench_download_bundle"):
        raise RuntimeError("a prior immutable bundle download was lost after rerun")

    app.segmented_control(key="active_workflow").set_value("kis")
    app.run()
    app.text_area(key="query_text").set_value("a person holding fruit")
    app.button(key="query_submit").click()
    app.run()
    if not any(no_match_message in value for value in _messages(app.info)):
        raise RuntimeError("completed empty KIS search does not show the no-match state")
    if not context.pipeline.requests:
        raise RuntimeError("KIS search did not reach the injected pipeline")
    if context.pipeline.requests[-1].batch_filters != (context.selected_batch_id,):
        raise RuntimeError("KIS request did not use the validated selected batch")

    app.text_area(key="query_text").set_value(
        "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu với một "
        "chiếc đĩa trắng. Sau đó đầu bếp đặt bánh lên đĩa trắng và các lát chuối "
        "lên một chiếc bánh; chiếc bánh còn lại có dâu."
    )
    app.button(key="query_submit").click()
    app.run()
    if app.exception:
        raise RuntimeError("long-query UI request raised an exception")
    planned = context.pipeline.requests[-1]
    for field in ("named_entities", "objects", "actions", "attributes"):
        values = tuple(getattr(planned, field) or ())
        if len(values) != len({value.casefold() for value in values}):
            raise RuntimeError(f"long-query UI retained duplicate {field}")

    staged_before_reset = dict(app.session_state["codabench_staging"])
    app.button(key="aic_reset_session_state").click()
    app.run()
    if dict(app.session_state["codabench_staging"]) != staged_before_reset:
        raise RuntimeError("session recovery removed Codabench staging")
    if app.session_state["kis_results"] or app.session_state["kis_last_request"] is not None:
        raise RuntimeError("session recovery did not clear volatile KIS state")
    app.button(key="codabench_clear_all").click()
    app.run()
    if app.session_state["codabench_staging"] or context.staging_snapshot["entries"]:
        raise RuntimeError("clear-all did not publish and hydrate an empty staging revision")
    if app.session_state["codabench_bundle_cache"] is not None:
        raise RuntimeError("clear-all retained the prior browser bundle cache")

    manual_app = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30
    )
    manual_context = _context(canonical=True)
    manual_app.session_state["_aic_runtime_override"] = manual_context
    manual_app.session_state["active_workflow"] = "qa"
    manual_app.session_state["qa_event_description"] = "operator-selected canonical frame"
    manual_app.run()
    manual_app.text_input(key="vqa_manual_video_id").set_value("L01_V028")
    manual_app.number_input(key="vqa_manual_frame_idx").set_value(3450)
    manual_app.button(key="qa_inject_manual_frame").click()
    manual_app.run()
    if manual_app.exception:
        messages = [
            getattr(item, "message", str(item)) for item in manual_app.exception
        ]
        raise RuntimeError("manual QA injector AppTest failed: " + " | ".join(messages))
    if manual_context.qa_pipeline.inject_calls != 1:
        raise RuntimeError("manual frame injector did not call FastAPI")
    if manual_app.session_state["qa_selected_candidate_id"] != (
        "aic25-b1-reference:L01_V028:3450"
    ):
        raise RuntimeError("manual canonical frame was not previewed as selected")

    direct = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30
    )
    direct.session_state["_aic_runtime_override"] = _context(canonical=False)
    direct.run()
    if direct.exception:
        messages = [getattr(item, "message", str(item)) for item in direct.exception]
        raise RuntimeError("direct-launch AppTest failed: " + " | ".join(messages))
    if not any(
        "Phát hiện phiên Streamlit được chạy trực tiếp hoặc chưa xác minh" in value
        for value in _messages(direct.warning)
    ):
        raise RuntimeError("direct-launch warning is not visible")
    expected_launcher = (
        ".\\scripts\\run_app.ps1 -Profile laptop "
        "-BatchId aic25-b1-reference -Port 8501"
    )
    if not any(expected_launcher in value for value in _messages(direct.warning)):
        raise RuntimeError("direct-launch warning does not show the exact launcher command")
    print("Streamlit AppTest executed the application script successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
