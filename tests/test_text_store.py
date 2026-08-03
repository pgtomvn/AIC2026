"""Offline unit tests for the Meilisearch lexical-store adapter."""

from __future__ import annotations

import pytest

from src.indexing.text_store import TextStore


class IndexNotFoundError(Exception):
    code = "index_not_found"
    status_code = 404


class FakeIndex:
    def __init__(self, client, *, primary_key: str = "id") -> None:
        self.client = client
        self.primary_key = primary_key
        self.searchable_attributes = None
        self.filterable_attributes = None
        self.document_batches = []
        self.search_calls = []
        self.search_response = {"hits": []}

    def update_searchable_attributes(self, attributes):
        self.searchable_attributes = list(attributes)
        return self.client.new_task()

    def update_filterable_attributes(self, attributes):
        self.filterable_attributes = list(attributes)
        return self.client.new_task()

    def add_documents(self, documents):
        self.document_batches.append(list(documents))
        return self.client.new_task()

    def search(self, query, options):
        self.search_calls.append((query, options))
        return self.search_response


class FakeMeilisearchClient:
    def __init__(self, *, exists: bool = True) -> None:
        self.index = FakeIndex(self) if exists else None
        self.created = []
        self.waited = []
        self.next_task_uid = 1
        self.failed_tasks = set()

    def new_task(self):
        uid = self.next_task_uid
        self.next_task_uid += 1
        return {"taskUid": uid}

    def get_index(self, index_name):
        if self.index is None:
            raise IndexNotFoundError(index_name)
        return self.index

    def create_index(self, index_name, options):
        self.created.append((index_name, options))
        self.index = FakeIndex(self, primary_key=options["primaryKey"])
        return self.new_task()

    def wait_for_task(self, task_uid, *, timeout_in_ms):
        self.waited.append((task_uid, timeout_in_ms))
        if task_uid in self.failed_tasks:
            return {"uid": task_uid, "status": "failed", "error": {"message": "bad task"}}
        return {"uid": task_uid, "status": "succeeded"}


def make_store(client: FakeMeilisearchClient) -> TextStore:
    store = TextStore()
    store._client = client
    return store


def test_constructor_is_lazy() -> None:
    store = TextStore()

    assert store.url == "http://localhost:7700"
    assert store.index_name == "aic2026_text"
    assert store._client is None
    assert store._index is None


def test_create_index_is_idempotent_and_configures_lexical_fields() -> None:
    client = FakeMeilisearchClient(exists=True)
    store = make_store(client)

    store.create_index()

    assert client.created == []
    assert "text_nfc" in client.index.searchable_attributes
    assert "text_accent_folded" in client.index.searchable_attributes
    assert "video_id" in client.index.filterable_attributes
    assert len(client.waited) == 2


def test_create_missing_index_waits_before_configuration() -> None:
    client = FakeMeilisearchClient(exists=False)
    store = make_store(client)

    store.create_index(primary_key="segment_id")

    assert client.created == [("aic2026_text", {"primaryKey": "segment_id"})]
    assert store._primary_key == "segment_id"
    assert "segment_id" in client.index.filterable_attributes
    assert len(client.waited) == 3


def test_create_index_surfaces_failed_configuration_task() -> None:
    client = FakeMeilisearchClient(exists=True)
    client.failed_tasks.add(1)
    store = make_store(client)

    with pytest.raises(RuntimeError, match="failed"):
        store.create_index()


def test_add_documents_batches_and_waits_for_each_task() -> None:
    client = FakeMeilisearchClient(exists=True)
    store = make_store(client)
    documents = [{"id": f"segment_{index}", "text_nfc": "người đi xe đạp"} for index in range(5)]

    store.add_documents(documents, batch_size=2)

    assert [len(batch) for batch in client.index.document_batches] == [2, 2, 1]
    assert len(client.waited) == 3


def test_add_documents_normalizes_meili_invalid_ids_and_preserves_external_id() -> None:
    client = FakeMeilisearchClient(exists=True)
    store = make_store(client)

    store.add_documents([{"id": "ocr:batch:video:42", "text_nfc": "xin chao"}])

    document = client.index.document_batches[0][0]
    assert document["external_id"] == "ocr:batch:video:42"
    assert ":" not in document["id"]
    assert len(document["id"]) == 36


def test_add_documents_validates_primary_keys_and_duplicates() -> None:
    store = make_store(FakeMeilisearchClient())

    with pytest.raises(ValueError, match="primary key"):
        store.add_documents([{"text": "missing id"}])
    with pytest.raises(ValueError, match="unique"):
        store.add_documents([{"id": "same"}, {"id": "same"}])


def test_search_returns_canonical_ranked_results_and_filter() -> None:
    client = FakeMeilisearchClient(exists=True)
    client.index.search_response = {
        "hits": [
            {
                "id": "ocr_42",
                "video_id": "L21_V001",
                "text_nfc": "người đi xe đạp",
                "_rankingScore": 0.875,
            },
            {"id": 7, "video_id": "L21_V002", "_rankingScore": 0.5},
        ]
    }
    store = make_store(client)

    results = store.search(
        "người đi xe đạp",
        top_k=20,
        filters='batch_id = "b1"',
    )

    assert client.index.search_calls == [
        (
            "người đi xe đạp",
            {
                "limit": 20,
                "showRankingScore": True,
                "filter": 'batch_id = "b1"',
            },
        )
    ]
    assert results[0] == {
        "rank": 1,
        "id": "ocr_42",
        "score": 0.875,
        "payload": {
            "id": "ocr_42",
            "video_id": "L21_V001",
            "text_nfc": "người đi xe đạp",
        },
    }
    assert results[1]["rank"] == 2
    assert results[1]["id"] == "7"


def test_search_rejects_malformed_response() -> None:
    client = FakeMeilisearchClient(exists=True)
    client.index.search_response = {"hits": "not-a-list"}
    store = make_store(client)

    with pytest.raises(RuntimeError, match="malformed"):
        store.search("query")


def test_availability_is_scoped_and_requires_a_document() -> None:
    client = FakeMeilisearchClient(exists=True)
    store = make_store(client)

    assert store.is_available("batch_id = 'b1'") is False
    client.index.search_response = {"hits": [{"id": "one"}]}
    assert store.is_available("batch_id = 'b1'") is True
    assert client.index.search_calls[-1] == (
        "",
        {"limit": 1, "filter": "batch_id = 'b1'"},
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.add_documents([], batch_size=0),
        lambda store: store.search("query", top_k=0),
        lambda store: store.search("query", filters={"batch_id": "b1"}),
    ],
)
def test_argument_validation(operation) -> None:
    with pytest.raises((TypeError, ValueError)):
        operation(TextStore())
