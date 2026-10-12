"""Streamlit production controls exercised offline with deterministic search."""

from streamlit.testing.v1 import AppTest


def interaction_app():
    from types import SimpleNamespace

    import streamlit as st

    from app.components.interactive_search import remember_search, render_interactive_search
    from src.contracts import FrameReference, QueryRequest

    class Pipeline:
        def search(self, request, **kwargs):
            st.session_state.sent_request = request.to_dict()
            return [{"video_id": "v", "source_frame_idx": 10}]

        refine = search

    if "kis_last_request" not in st.session_state:
        request = QueryRequest("q", "người đang đi", batch_filters=("b",))
        st.session_state.kis_last_request = request
        st.session_state.kis_results = [{"video_id": "v", "source_frame_idx": 0}]
        remember_search(request, st.session_state.kis_results, initial=True)
        st.session_state.kis_reference = FrameReference("b", "v", 0)
    render_interactive_search(SimpleNamespace(config={}, pipeline=Pipeline()), ready=True, limit=20)


def test_real_controls_find_next_undo_reset_and_isolate_session():
    app = AppTest.from_function(interaction_app).run()
    assert not app.exception
    assert "Lượt 1/20" in app.get("expander")[0].label
    app.text_area(key="kis_desired_text").set_value("người đang chạy")
    app.button(key="kis_find_next").click().run()
    assert not app.exception
    assert app.session_state.sent_request["original_text"] == "người đang chạy"
    assert app.session_state.sent_request["reference_frame"]["source_frame_idx"] == 0
    assert len(app.session_state.kis_conversation.turns) == 2
    assert "Lượt 2/20" in app.get("expander")[0].label
    app.button(key="kis_turn_undo").click().run()
    assert not app.exception
    assert app.session_state.kis_last_request.original_text == "người đang đi"
    assert "Lượt 1/20" in app.get("expander")[0].label
    app.button(key="kis_turn_reset").click().run()
    assert not app.exception
    assert app.session_state.kis_results[0]["source_frame_idx"] == 0
    other = AppTest.from_function(interaction_app).run()
    assert len(other.session_state.kis_conversation.turns) == 1


def test_condition_replace_is_explicit_and_preview_updates():
    app = AppTest.from_function(interaction_app).run()
    app.text_input(key="kis_condition_text").set_value("mặc áo xanh")
    app.button(key="kis_edit_condition").click().run()
    assert not app.exception
    assert app.text_area(key="kis_desired_text").value == "người đang đi. mặc áo xanh"


def test_english_edit_is_preview_only_until_explicit_accept():
    app = AppTest.from_function(interaction_app).run()
    app.text_area(key="kis_desired_text").set_value("A man in a red shirt").run()
    app.text_input(key="kis_edit_instruction").set_value("replace red shirt with blue shirt")
    app.button(key="kis_preview_edit").click().run()
    assert app.text_area(key="kis_desired_text").value == "A man in a red shirt"
    assert len(app.session_state.kis_conversation.turns) == 1
    app.button(key="kis_accept_edit").click().run()
    assert not app.exception
    assert app.text_area(key="kis_desired_text").value == "A man in a blue shirt"
    assert len(app.session_state.kis_conversation.turns) == 1
    app.button(key="kis_find_next").click().run()
    assert not app.exception
    assert "Lượt 2/20" in app.get("expander")[0].label


def test_refinement_is_collapsed_until_feedback_selected():
    app = AppTest.from_function(interaction_app).run()
    assert app.get("expander")[0].proto.expanded
    reference = app.session_state.kis_reference
    app.session_state.kis_reference = None
    app.run()
    assert not app.exception
    assert not app.get("expander")[0].proto.expanded
    assert app.button(key="kis_find_next").label == "Tìm tiếp"
    app.session_state.kis_positive_frames = [reference]
    app.run()
    assert app.get("expander")[0].proto.expanded


def test_compact_gallery_keeps_canonical_actions_and_hides_diagnostics():
    def gallery_app():
        import streamlit as st

        from app.components.result_gallery import render_result_gallery

        def select(item):
            st.session_state.selected = item

        render_result_gallery([
            {"batch_id": "b", "video_id": "v", "source_frame_idx": 0,
             "timestamp_ms": 1234, "rank": 1, "fused_score": .02,
             "channel_ranks": {"siglip2": 1}, "evidence": {}},
        ], on_reference=select, on_positive=select, on_negative=select,
           on_open_scrubber=select, on_send_to_qa=select)

    app = AppTest.from_function(gallery_app).run()
    assert not app.exception
    assert app.code[0].value == "v,0"
    proof = app.get("expander")[0]
    assert proof.label == "Bằng chứng" and not proof.proto.expanded
    assert any("Điểm hợp nhất" in item.value for item in proof.get("caption"))
    assert len(app.button) == 5
    for button in list(app.button):
        app.button(key=button.key).click().run()
        assert not app.exception
        assert app.session_state.selected["batch_id"] == "b"
        assert app.session_state.selected["source_frame_idx"] == 0
