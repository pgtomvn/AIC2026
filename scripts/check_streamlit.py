"""Execute the real Streamlit app script using Streamlit's offline AppTest API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


class _Pipeline:
    last_channel_errors: dict[str, str] = {}

    def __init__(self) -> None:
        self.requests = []

    def search(self, request, *, limit=None, refine_raw=True):
        self.requests.append(request)
        return []


class _TRAKEPipeline:
    @staticmethod
    def channel_readiness(_batch_id):
        return True, {"organizer_clip": {"state": "ready"}}

    @staticmethod
    def search(request, *, default_batch_id=None):
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


class _Repository:
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
    def get_source_frame(_batch_id, _video_id, source_frame_idx):
        return SimpleNamespace(timestamp_ms=source_frame_idx * 100)

    get_frame = get_source_frame


def _context(*, canonical: bool):
    selected_batch_id = "aic25-b1-reference"
    pipeline = _Pipeline()

    def validate_batch_filters(values=None):
        result = tuple(values or (selected_batch_id,))
        if result != (selected_batch_id,):
            raise ValueError("test context accepts only the selected batch")
        return result

    return SimpleNamespace(
        pipeline=pipeline,
        trake_pipeline=_TRAKEPipeline(),
        artifact_repository=_Repository(),
        qa_pipeline=SimpleNamespace(),
        qa_session_store=None,
        qa_image_resolver=None,
        readiness_service=None,
        selected_batch_id=selected_batch_id,
        validate_batch_filters=validate_batch_filters,
        launch_identity={
            "canonical": canonical,
            "contract": "run_app-v1" if canonical else "direct-streamlit",
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

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    context = _context(canonical=True)
    app.session_state["_aic_runtime_override"] = context
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("Streamlit AppTest failed: " + " | ".join(messages))
    workflow = app.segmented_control(key="active_workflow")
    if tuple(workflow.options) != ("KIS văn bản", "Hỏi đáp hình ảnh", "TRAKE"):
        raise RuntimeError("Streamlit workflow selector is incomplete")
    if workflow.value != "kis":
        raise RuntimeError("Streamlit did not preserve the default KIS workflow")
    no_match_message = "Không có kết quả đã lập chỉ mục phù hợp"
    if any(no_match_message in value for value in _messages(app.info)):
        raise RuntimeError("initial KIS view incorrectly reports an empty completed search")
    if not any("run_app-v1" in value for value in _messages(app.caption)):
        raise RuntimeError("canonical launch identity is not visible")
    workflow.set_value("qa")
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("QA workflow AppTest failed: " + " | ".join(messages))
    if app.text_input(key="qa_batch_filters").value != context.selected_batch_id:
        raise RuntimeError("QA batch filter was not prepopulated from selected_batch_id")
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
    app.text_area(key="trake_global_description").set_value("Một người mua hàng")
    app.text_input(key="trake_stage_1").set_value("Người bước vào cửa hàng")
    app.text_input(key="trake_stage_2").set_value("Người thanh toán")
    app.button(key="trake_search").click()
    app.run()
    if app.exception:
        messages = [getattr(item, "message", str(item)) for item in app.exception]
        raise RuntimeError("non-empty TRAKE AppTest failed: " + " | ".join(messages))
    if app.selectbox(key="trake_selected_candidate").value != (
        "aic25-b1-reference:L21_V001"
    ):
        raise RuntimeError("TRAKE did not select the first stable candidate identity")
    if not app.download_button(key="trake_download"):
        raise RuntimeError("validated TRAKE results did not expose a CSV download")

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
