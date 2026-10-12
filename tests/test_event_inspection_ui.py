"""Manual evidence actions cannot run inference or confirm labels implicitly."""
from streamlit.testing.v1 import AppTest


def event_controls_app():
    from types import SimpleNamespace

    import streamlit as st

    from app.components.event_inspection import render_event_inspection
    from src.contracts import QueryRequest

    class Repository:
        def inspect_event(self, frames, question):
            st.session_state.inspection = (frames, question)
            return {'state': 'ready', 'verdict': 'insufficient_evidence',
                    'explanation': 'Sparse still images do not establish motion.'}

        def confirm_answer(self, request, frame, elapsed):
            st.session_state.confirmed = (request.interaction_session_id, frame, elapsed)

    st.session_state.kis_last_request = QueryRequest(
        'q', 'A person opens a door', interaction_session_id='test-session',
        interaction_turn=1, event_source='benchmark')
    st.session_state.kis_scrub_frames = [
        (1, SimpleNamespace(batch_id='b', video_id='v', source_frame_idx=10)),
        (2, SimpleNamespace(batch_id='b', video_id='other', source_frame_idx=20))]
    render_event_inspection(SimpleNamespace(
        artifact_repository=Repository(),
        config={'retrieval': {'interactive': {'event_inspection_enabled': True}}}),
        {'batch_id': 'b', 'video_id': 'v', 'source_frame_idx': 0}, key='test')


def test_evidence_is_on_demand_and_confirmation_is_separate():
    app = AppTest.from_function(event_controls_app).run()
    assert not app.exception
    assert 'inspection' not in app.session_state
    assert 'confirmed' not in app.session_state
    assert len(app.multiselect[0].options) == 1
    app.button(key='test_event_run').click().run()
    assert not app.exception
    frames, question = app.session_state.inspection
    assert frames == [{'batch_id': 'b', 'video_id': 'v', 'source_frame_idx': 0}]
    assert question == 'A person opens a door'
    assert 'confirmed' not in app.session_state
    assert any('Không đủ bằng chứng' in element.value for element in app.markdown)
    app.button(key='test_confirm_answer').click().run()
    assert not app.exception
    assert app.session_state.confirmed[0] == 'test-session'
