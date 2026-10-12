"""Offline acceptance of explicit edits and fail-closed English auto routing."""
import hashlib
import json
from dataclasses import replace

import pytest

from src.contracts import QueryRequest
from src.retrieval.refinement_preview import preview_edit
from src.retrieval.temporal_router import (
    PARAMETERS,
    PARSER_VERSION,
    load_release_record,
    parse_english_steps,
    release_allowed,
)


def test_preview_unique_phrase_only_and_original_is_unchanged():
    query = 'A man in a red shirt. A blue bicycle.'
    result = preview_edit(query, 'replace red shirt with blue shirt')
    assert result['original_text'] == query
    assert result['draft_text'] == 'A man in a blue shirt. A blue bicycle.'
    assert result['applied']
    assert not preview_edit('red shirt and red shirt', 'replace red shirt with blue shirt')['applied']
    assert not preview_edit(query, 'replace yellow shirt with blue shirt')['applied']
    assert not preview_edit(query, 'make it different')['applied']
    assert not preview_edit(query, 'add A blue bicycle')['applied']
    assert preview_edit(query, 'remove A blue bicycle')['draft_text'] == 'A man in a red shirt'
    assert not preview_edit('a cat', 'remove a cat')['applied']
    assert not preview_edit('a man with glasses', 'replace a man with glasses with a woman')['applied']
    assert preview_edit('a man with glasses', 'replace "a man with glasses" with "a woman"')['draft_text'] == 'a woman'


@pytest.mark.parametrize('text,expected', [
    ('A man opens a door before a woman walks in', ['A man opens a door', 'a woman walks in']),
    ('A man leaves after a woman enters', ['a woman enters', 'A man leaves']),
    ('Before a woman enters, a man leaves', ['a man leaves', 'a woman enters']),
    ('After a woman enters, a man leaves', ['a woman enters', 'a man leaves']),
    ('First a man opens a door, then a woman walks in, finally they sit',
     ['a man opens a door', 'a woman walks in', 'they sit']),
    ('A person shows "before then after", then a man leaves',
     ['A person shows "before then after"', 'a man leaves']),
])
def test_explicit_order(text, expected):
    parsed = parse_english_steps(text)
    assert parsed['steps'] == expected
    assert parsed['target'] == len(expected)-1


@pytest.mark.parametrize('text', [
    'A man standing behind a bicycle', 'A person looks before a building',
    'A man opens a door if a woman enters', 'A man enters then leaves after a woman sits',
    '"first then finally"', 'A man opens a door then a man opens a door',
    'People enter one after another. Green leaves lie on the ground.',
    'A person arrives day after day. Someone opens the door.',
    ' then '.join(['A man opens a door'] * 7),
])
def test_ambiguous_or_spatial_is_not_temporal(text):
    assert not parse_english_steps(text)['steps']


def test_release_gate_not_a_boolean_or_yaml_toggle():
    args = dict(retrieval_fingerprint='r', cache_fingerprint='c', source_fingerprint='s')
    assert not release_allowed({'activation_gate': {'passed': True}}, **args)
    baseline = dict(query_sha256='q', split_sha256='s', health_ok=True, error_count=0,
                    hit_at_1=.2, hit_at_3=.4, hit_at_5=.6, mrr=.3, mean_top_k=.4,
                    warm_p95_ms=100, rss_bytes=100)
    record = dict(schema_version=1, feature='temporal_sequence', language='en',
                  parser_version=PARSER_VERSION, parameters=PARAMETERS, **args,
                  labels={'source': 'user_confirmed', 'ordered_event_count': 6, 'sha256': 'a'*64},
                  verified_labels_sha256='a'*64,
                  baseline=baseline, trial={**baseline, 'mean_top_k': .42},
                  parameters_locked=True, outside_scope_invariant=True)
    assert release_allowed(record, **args)
    for patch in ({'source_fingerprint': 'old'}, {'labels': {}}, {'outside_scope_invariant': False},
                  {'verified_labels_sha256': ''},
                  {'trial': {**record['trial'], 'error_count': 1}}):
        assert not release_allowed({**record, **patch}, **args)


def test_contract_roundtrip_and_invalid_provenance():
    old = QueryRequest('q', 'A man opens a door')
    new = replace(old, interaction_session_id='session-1', interaction_turn=2,
                  event_source='agent', temporal_mode='off')
    assert QueryRequest.from_dict(new.to_dict()) == new
    for patch in ({'event_source': 'robot'}, {'interaction_turn': True}, {'temporal_mode': 'force'},
                  {'interaction_session_id': '../bad'}):
        with pytest.raises((ValueError, TypeError)):
            replace(old, **patch)


def test_release_loader_checks_actual_local_human_evidence(tmp_path):
    evaluation = tmp_path / 'outputs' / 'evaluation'
    evaluation.mkdir(parents=True)
    label_path = evaluation / 'ordered-labels.json'
    rows = [{'query_id': f'q{i}', 'event_source': 'user', 'confirmed': True,
             'steps': ['A man opens a door', 'A man enters'], 'target': 1} for i in range(6)]
    blob = json.dumps(rows).encode()
    label_path.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    record_path = evaluation / 'release.json'
    record = {'labels': {'path': 'outputs/evaluation/ordered-labels.json',
                        'sha256': digest, 'ordered_event_count': 6},
              'verified_labels_sha256': 'forged'}
    record_path.write_text(json.dumps(record))
    loaded = load_release_record(record_path, tmp_path)
    assert loaded is not None and loaded['verified_labels_sha256'] == digest
    label_path.write_bytes(blob + b' ')
    assert load_release_record(record_path, tmp_path) is None
    rows[0]['event_source'] = 'benchmark'
    blob = json.dumps(rows).encode()
    label_path.write_bytes(blob)
    record['labels']['sha256'] = hashlib.sha256(blob).hexdigest()
    record_path.write_text(json.dumps(record))
    assert load_release_record(record_path, tmp_path) is None
    record['labels']['path'] = '../../outside.json'
    record_path.write_text(json.dumps(record))
    assert load_release_record(record_path, tmp_path) is None
