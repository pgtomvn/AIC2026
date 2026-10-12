"""Object resume rejects partial-store corruption before any mutation."""
from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5

import pytest

from src.ingest.reference_resume import AuditedObjectResumeStore


class Importer:
    docs = [{"id": "object:b:v:ordinal:1", "text_raw": "Car", "timestamp_ms": 0},
            {"id": "object:b:v:ordinal:2", "text_raw": "Person", "timestamp_ms": 5}]

    def import_objects(self, store, *, create_index=False):
        assert not create_index
        store.add_documents(deepcopy(self.docs))


class Store:
    index_name = "b02_objects"

    def __init__(self):
        self.docs = deepcopy(Importer.docs[:1])
        for doc in self.docs:
            doc['external_id'] = doc['id']
            doc['id'] = str(uuid5(NAMESPACE_URL, f"aic2026:meilisearch:{self.index_name}:{doc['id']}"))
        self.writes = []

    def iter_documents(self):
        yield from self.docs

    def add_documents(self, docs):
        self.writes.extend(docs)


def test_resume_skips_only_exact_audited_documents():
    store = Store()
    resume = AuditedObjectResumeStore.audit(Importer(), store)
    assert store.writes == []
    assert resume.expected_count == 2
    resume.add_documents(Importer.docs)
    assert store.writes == Importer.docs[1:]
    assert (resume.skipped, resume.written) == (1, 1)


@pytest.mark.parametrize('corruption', ['text', 'id', 'unknown', 'duplicate', 'extra'])
def test_resume_audit_rejects_corruption_without_writes(corruption):
    store = Store()
    if corruption == 'text':
        store.docs[0]['text_raw'] = 'Wrong'
    elif corruption == 'id':
        store.docs[0]['id'] = 'wrong'
    elif corruption == 'unknown':
        store.docs[0]['external_id'] = 'foreign-batch'
    elif corruption == 'extra':
        store.docs[0]['unexpected'] = True
    else:
        store.docs.append(dict(store.docs[0]))
    with pytest.raises(ValueError):
        AuditedObjectResumeStore.audit(Importer(), store)
    assert store.writes == []


def test_resume_rejects_changed_source_after_audit():
    store = Store()
    resume = AuditedObjectResumeStore.audit(Importer(), store)
    changed = deepcopy(Importer.docs)
    changed[0]['timestamp_ms'] = 10
    with pytest.raises(ValueError, match='source changed'):
        resume.add_documents(changed)
    assert store.writes == []
