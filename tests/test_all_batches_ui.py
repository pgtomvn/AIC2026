"""Real Streamlit request scope and mixed-batch pin/export smoke."""
from streamlit.testing.v1 import AppTest


def scope_app():
    import streamlit as st

    from app.streamlit_app import _initialize_state, _render_qa, _validated_batch_filters
    from scripts.check_streamlit import _context
    from src.retrieval.batch_scope import BATCHES

    _initialize_state()
    context = _context(canonical=True)
    context.config['retrieval']['all_batches'] = {'enabled': True}
    context.validate_batch_filters = lambda values: tuple(values)
    st.selectbox('scope', ('all', *BATCHES), key='search_scope')
    st.session_state.scope_sent = _validated_batch_filters(context)
    _render_qa(context)
    if context.qa_pipeline.last_request is not None:
        st.session_state.qa_sent = context.qa_pipeline.last_request.batch_filters


def test_default_all_and_optional_scope_reach_qa():
    from src.retrieval.batch_scope import B02, BATCHES

    app = AppTest.from_function(scope_app, default_timeout=15).run()
    assert not app.exception
    assert app.session_state.scope_sent == BATCHES
    app.text_area(key='qa_event_description').set_value('A person walks')
    app.button(key='qa_retrieve_candidates').click().run()
    assert not app.exception and app.session_state.qa_sent == BATCHES
    app.selectbox(key='search_scope').select(B02).run()
    assert not app.exception and app.session_state.scope_sent == (B02,)
    app.button(key='qa_retrieve_candidates').click().run()
    assert not app.exception and app.session_state.qa_sent == (B02,)


def trake_app():
    from dataclasses import replace
    from types import SimpleNamespace

    import streamlit as st

    from app.streamlit_app import _initialize_state, _render_trake
    from src.retrieval.batch_scope import B01, B02
    from src.submission.combined import serialize_combined_submission
    from tests.test_b02_trake import Repository, sequence

    _initialize_state()
    repo = Repository()
    b01frames = [replace(f, batch_id=B01, video_id='L21_V001') for f in (repo.frames[0], repo.frames[2])]
    repo.get_source_frame = lambda b,v,s: next((f for f in b01frames if (f.batch_id,f.video_id,f.source_frame_idx)==(b,v,s)),None)
    repo.get_frame = repo.get_source_frame
    repo.ordinal_neighbors = lambda *args, **kwargs: ()
    repo.submission_export = lambda rows, **kw: serialize_combined_submission(repo, rows, **kw)
    first = dict(sequence(), batch_id=B01, video_id='L21_V001', confidence_score=.8)
    first['stage_matches'] = [dict(m, keyframe_ordinal=1 if i==0 else 3) for i,m in enumerate(first['stage_matches'])]
    second = dict(sequence(), rank=2, confidence_score=.7)
    def search(request, **kwargs):
        st.session_state.request_batches = request.batch_filters
        return {'query_id':request.query_id,'results':[first,second],'channel_statuses':{}}
    context = SimpleNamespace(selected_batch_id=B01,
        config={'retrieval':{'all_batches':{'enabled':True}},'competition':{'output_dir':'outputs/submissions'}},
        artifact_repository=repo, qa_image_resolver=None,
        validate_batch_filters=lambda batches:tuple(batches),
        trake_pipeline=SimpleNamespace(search=search,channel_readiness=lambda batch:(True,{})),
        ledger=SimpleNamespace(is_batch_active=lambda batch:batch in (B01,B02)))
    _render_trake(context)


def test_trake_all_batches_mixed_pins_and_csv():
    from src.retrieval.batch_scope import BATCHES

    app = AppTest.from_function(trake_app, default_timeout=15).run()
    assert not app.exception
    app.text_input(key='codabench_query_id_trake').set_value('f-3')
    app.text_area(key='trake_global_description').set_value('A person stands then leaves')
    app.text_input(key='trake_stage_1').set_value('A person stands')
    app.text_input(key='trake_stage_2').set_value('A person leaves')
    app.button(key='trake_search').click().run()
    assert not app.exception and app.session_state.request_batches == BATCHES
    app.button(key='trake_pin_selected').click().run()
    assert not app.exception and not app.warning
    selector=app.selectbox(key='trake_selected_candidate')
    selector.select_index(1).run()
    app.button(key='trake_pin_selected').click().run()
    assert not app.exception and not app.warning
    assert app.session_state.b02_submission_files['query-f-3-trake.csv'] == b'L21_V001,0,25\nM01_V001,0,25\n'
