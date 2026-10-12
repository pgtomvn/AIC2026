"""Real Streamlit TRAKE panel with offline B02 ordinal transport fixtures."""
from streamlit.testing.v1 import AppTest


def trake_app():
    import io
    from types import SimpleNamespace

    import streamlit as st
    from PIL import Image

    from app.streamlit_app import _initialize_state, _render_trake
    from src.submission.b02 import serialize_b02_submission
    from tests.test_b02_trake import BATCH, Repository, sequence

    _initialize_state()
    repo = Repository()
    repo.ordinal_neighbors = lambda batch, video, ordinal, **kw: tuple(repo.frames)
    repo.b02_export = lambda rows, **kw: serialize_b02_submission(repo, rows, **kw)
    image = io.BytesIO()
    Image.new('RGB',(12,12),'blue').save(image,format='JPEG')
    def search(request, **kwargs):
        return {'query_id':request.query_id,'results':[dict(sequence(),confidence_score=.8)],'channel_statuses':{}}
    context = SimpleNamespace(
        selected_batch_id=BATCH,config={'retrieval':{},'competition':{'output_dir':'outputs/submissions'}},
        artifact_repository=repo, qa_image_resolver=SimpleNamespace(resolve=lambda frame:SimpleNamespace(data=image.getvalue())),
        trake_pipeline=SimpleNamespace(search=search,channel_readiness=lambda batch:(True,{'organizer_clip':{'state':'ready'}})),
        validate_batch_filters=lambda batches:tuple(batches),ledger=SimpleNamespace(is_batch_active=lambda batch:True),
    )
    _render_trake(context)


def test_b02_trake_search_thumbnails_ordinal_swap_pin_and_csv():
    app = AppTest.from_function(trake_app).run()
    assert not app.exception
    app.text_input(key='codabench_query_id_trake').set_value('f-2')
    app.text_area(key='trake_global_description').set_value('A person stands then leaves')
    app.text_input(key='trake_stage_1').set_value('A person stands')
    app.text_input(key='trake_stage_2').set_value('A person leaves')
    app.button(key='trake_search').click().run()
    assert not app.exception and not app.error
    swaps = [b for b in app.button if b.key.startswith('trake_swap_')]
    assert len(swaps)==2 and len({b.key for b in swaps})==2
    assert [b.label for b in swaps]==['Chọn ảnh 1','Chọn ảnh 2']
    app.button(key=swaps[1].key).click().run()
    assert not app.exception
    app.button(key='trake_pin_selected').click().run()
    assert not app.exception and not app.warning
    assert app.session_state.b02_submission_files == {'query-f-2-trake.csv':b'M01_V001,0,25\n'}
    row = app.session_state.b02_submission_rows['query-f-2-trake.csv'][0]
    assert row['stage_matches'][0]['keyframe_ordinal']==2
    assert row['stage_matches'][0]['timestamp_ms']==40
    # Updating the same video must not append a second rank or enter L-only staging.
    app.button(key='trake_pin_selected').click().run()
    assert not app.exception and not app.warning
    assert len(app.session_state.b02_submission_rows['query-f-2-trake.csv'])==1
    assert not app.session_state['codabench_staging']
