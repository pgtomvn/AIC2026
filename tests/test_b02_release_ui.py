"""Offline production B02 controls carry ordinals through the UI boundary."""
from streamlit.testing.v1 import AppTest


def b02_app():
    from types import SimpleNamespace

    import streamlit as st

    from app.components.b02_export import render_b02_export
    from app.components.interactive_search import select_feedback
    from app.components.result_gallery import render_result_gallery
    from app.streamlit_app import _open_kis_scrubber

    row = dict(batch_id="aic26-b2", video_id="N001-V001", source_frame_idx=0,
               keyframe_ordinal=100, timestamp_ms=1250, submission_unit="timestamp_ms",
               submission_value=1250, rank=1, score=.9, channel_ranks={"organizer_clip": 1})
    def choose(item):
        select_feedback(item, "reference")
    def export(rows, **kwargs):
        st.session_state.sent_rows = rows
        assert rows[0]["keyframe_ordinal"] == 100
        return b"N001-V001,1250\n"
    render_result_gallery([row], on_reference=choose, on_open_scrubber=_open_kis_scrubber)
    render_b02_export(SimpleNamespace(artifact_repository=SimpleNamespace(b02_export=export)), [row], task="kis")


def test_ordinal_actions_and_export_confirmation():
    app = AppTest.from_function(b02_app).run()
    assert not app.exception
    assert app.code[0].value == "N001-V001,1250"
    for button in list(app.button):
        if button.key.startswith("b02_"):
            continue
        app.button(key=button.key).click().run()
        assert not app.exception
    assert app.session_state.kis_reference.keyframe_ordinal == 100
    assert app.session_state.kis_reference.source_frame_idx == 0
    assert app.session_state.kis_scrub_anchor["keyframe_ordinal"] == 100
    app.text_input(key="b02_export_query_kis").set_value("f-1")
    app.button(key="b02_export_prepare_kis").click().run()
    assert not app.exception
    original = dict(app.session_state.b02_submission_files)
    assert original == {"query-f-1-kis.csv": b"N001-V001,1250\n"}
    app.button(key="b02_export_prepare_kis").click().run()
    assert app.warning and app.session_state.b02_submission_files == original
    app.checkbox(key="b02_export_replace_kis").check()
    app.button(key="b02_export_prepare_kis").click().run()
    assert not app.exception and not app.warning
