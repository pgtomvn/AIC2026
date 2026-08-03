"""Offline unit tests for the Qdrant vector-store adapter."""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import qdrant_client
from qdrant_client import models

from src.indexing.vector_store import (
    BatchVectorStore,
    VectorPointBatch,
    VectorStore,
    batch_collection_name,
    collection_batch_token,
)


class FakeQdrantClient:
    def __init__(self, *, exists: bool = False) -> None:
        self.exists = exists
        self.created: list[dict] = []
        self.deleted: list[dict] = []
        self.payload_indexes: list[dict] = []
        self.upserts: list[dict] = []
        self.queries: list[dict] = []
        self.counts: list[dict] = []
        self.scrolls: list[dict] = []
        self.count_value = 0
        self.vector_size = 4
        self.distance = models.Distance.COSINE

    def collection_exists(self, **kwargs):
        return self.exists

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.exists = True

    def delete_collection(self, **kwargs):
        self.deleted.append(kwargs)
        self.exists = False

    def get_collection(self, **kwargs):
        vectors = SimpleNamespace(size=self.vector_size, distance=self.distance)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="ef70fd4f-55bb-5fca-a36b-31b81b86d6e4",
                    score=0.91,
                    payload={"external_id": "L21_V001:42", "video_id": "L21_V001"},
                )
            ]
        )

    def count(self, **kwargs):
        self.counts.append(kwargs)
        return SimpleNamespace(count=self.count_value)

    def scroll(self, **kwargs):
        self.scrolls.append(kwargs)
        offset = kwargs.get("offset")
        if offset is None:
            return (
                [SimpleNamespace(payload={"batch_id": "b2", "video_id": "L99_V001"})],
                "page-2",
            )
        return ([SimpleNamespace(payload={"batch_id": "b2", "video_id": "L99_V002"})], None)


class LegacyQdrantClient:
    def __init__(self) -> None:
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(id=7, score=0.5, payload={"video_id": "L21_V007"})]


class RoutedQdrantClient(FakeQdrantClient):
    def __init__(self) -> None:
        super().__init__()
        self.names = {
            "aic_native_aic25-b1_siglip2_b16_224_r75de2d55_768d_v1",
            "aic_native_aic26-b2_siglip2_b16_224_r75de2d55_768d_v1",
        }

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.names])

    def collection_exists(self, **kwargs):
        return kwargs["collection_name"] in self.names

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        batch_id = "aic25-b1" if "aic25-b1" in kwargs["collection_name"] else "aic26-b2"
        score = 0.7 if batch_id == "aic25-b1" else 0.9
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=batch_id,
                    score=score,
                    payload={"external_id": batch_id, "batch_id": batch_id},
                )
            ]
        )


def make_store(client, **kwargs) -> VectorStore:
    store = VectorStore(vector_size=4, **kwargs)
    store._client = client
    return store


class TestVectorStore:
    def test_store_init_is_lazy(self) -> None:
        store = VectorStore()

        assert store.url == "http://localhost:6333"
        assert store.collection_name == "aic2026_keyframes"
        assert store.vector_size == 1024
        assert store.distance == "Cosine"
        assert store.prefer_grpc is False
        assert store.grpc_port is None
        assert store._client is None

    def test_store_custom_params(self) -> None:
        store = VectorStore(
            url="http://custom:6333",
            collection_name="test_collection",
            vector_size=512,
            distance="Euclid",
        )

        assert store.url == "http://custom:6333"
        assert store.collection_name == "test_collection"
        assert store.vector_size == 512

    def test_grpc_transport_is_forwarded_to_lazy_client(self, monkeypatch) -> None:
        captured: list[dict] = []

        class RecordingClient:
            def __init__(self, **kwargs) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(qdrant_client, "QdrantClient", RecordingClient)
        store = VectorStore(
            url="http://127.0.0.1:6333",
            api_key="local-key",
            prefer_grpc=True,
            grpc_port=6334,
        )

        store._connect()

        assert captured == [
            {
                "url": "http://127.0.0.1:6333",
                "api_key": "local-key",
                "prefer_grpc": True,
                "grpc_port": 6334,
            }
        ]

    def test_default_transport_keeps_rest_preference(self, monkeypatch) -> None:
        captured: list[dict] = []

        class RecordingClient:
            def __init__(self, **kwargs) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(qdrant_client, "QdrantClient", RecordingClient)

        VectorStore()._connect()

        assert captured[0]["prefer_grpc"] is False
        assert "grpc_port" not in captured[0]

    def test_lazy_client_initialization_is_single_flight(self, monkeypatch) -> None:
        created: list[object] = []
        creation_lock = Lock()

        class SlowClient:
            def __init__(self, **_kwargs) -> None:
                time.sleep(0.01)
                with creation_lock:
                    created.append(self)

        monkeypatch.setattr(qdrant_client, "QdrantClient", SlowClient)
        store = VectorStore(vector_size=4)

        with ThreadPoolExecutor(max_workers=8) as executor:
            clients = tuple(executor.map(lambda _: store._get_client(), range(16)))

        assert len(created) == 1
        assert all(client is clients[0] for client in clients)

    @pytest.mark.parametrize(
        "url",
        (
            "http://localhost:6333",
            "http://127.0.0.2:6333",
            "http://[::1]:6333",
        ),
    )
    def test_loopback_api_key_suppresses_only_qdrants_exact_warning(
        self, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        class WarningClient:
            def __init__(self, **_kwargs) -> None:
                warnings.warn(
                    "Api key is used with an insecure connection.",
                    UserWarning,
                    stacklevel=2,
                )

        monkeypatch.setattr(qdrant_client, "QdrantClient", WarningClient)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            VectorStore(url=url, api_key="local-key")._connect()

        assert not [item for item in captured if "insecure connection" in str(item.message)]

    def test_remote_plaintext_api_key_warning_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class WarningClient:
            def __init__(self, **_kwargs) -> None:
                warnings.warn(
                    "Api key is used with an insecure connection.",
                    UserWarning,
                    stacklevel=2,
                )

        monkeypatch.setattr(qdrant_client, "QdrantClient", WarningClient)

        with pytest.warns(UserWarning, match="insecure connection"):
            VectorStore(url="http://qdrant.internal:6333", api_key="remote-key")._connect()

    def test_create_collection_uses_dynamic_config_and_payload_indexes(self) -> None:
        client = FakeQdrantClient()
        store = make_store(client, distance="Dot")

        store.create_collection()

        config = client.created[0]["vectors_config"]
        assert config.size == 4
        assert config.distance == models.Distance.DOT
        assert {call["field_name"] for call in client.payload_indexes} == {
            "batch_id",
            "video_id",
            "source_frame_idx",
            "shot_id",
        }

    def test_recreate_deletes_existing_collection(self) -> None:
        client = FakeQdrantClient(exists=True)
        store = make_store(client)

        store.create_collection(recreate=True)

        assert client.deleted == [{"collection_name": "aic2026_keyframes"}]
        assert len(client.created) == 1

    def test_existing_incompatible_collection_is_rejected(self) -> None:
        client = FakeQdrantClient(exists=True)
        client.vector_size = 512
        store = make_store(client)

        with pytest.raises(ValueError, match="recreate=True"):
            store.create_collection()

    def test_upsert_batches_and_maps_arbitrary_ids_to_stable_uuids(self) -> None:
        client = FakeQdrantClient()
        store = make_store(client)
        ids = ["L21_V001/frame 1", "L21_V001/frame 2", "unicode/khung-hình"]
        payloads = [{"video_id": "L21_V001"} for _ in ids]
        vectors = np.arange(12, dtype=np.float32).reshape(3, 4)

        store.upsert(ids, vectors, payloads, batch_size=2)

        assert [len(call["points"]) for call in client.upserts] == [2, 1]
        points = [point for call in client.upserts for point in call["points"]]
        assert [point.payload["external_id"] for point in points] == ids
        assert all(str(UUID(str(point.id))) == str(point.id) for point in points)
        assert points[0].id == store._point_id(ids[0])
        assert payloads[0] == {"video_id": "L21_V001"}

    @pytest.mark.parametrize(
        ("vectors", "message"),
        [
            (np.zeros((2, 3), dtype=np.float32), "dimension"),
            (np.array([[0.0, 1.0, np.nan, 3.0]], dtype=np.float32), "NaN"),
        ],
    )
    def test_upsert_validates_vectors(self, vectors: np.ndarray, message: str) -> None:
        store = make_store(FakeQdrantClient())
        ids = [str(index) for index in range(len(vectors))]
        payloads = [{} for _ in ids]

        with pytest.raises(ValueError, match=message):
            store.upsert(ids, vectors, payloads)

    def test_upsert_rejects_mismatched_lengths_and_duplicate_ids(self) -> None:
        store = make_store(FakeQdrantClient())
        vectors = np.zeros((2, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            store.upsert(["one"], vectors, [{}, {}])
        with pytest.raises(ValueError, match="unique"):
            store.upsert(["one", "one"], vectors, [{}, {}])

    def test_current_query_api_translates_flat_filters(self) -> None:
        client = FakeQdrantClient()
        store = make_store(client)

        results = store.search(
            np.ones(4, dtype=np.float32),
            top_k=12,
            filters={"batch_id": "b1", "source_frame_idx": {"gte": 10}},
        )

        call = client.queries[0]
        assert call["limit"] == 12
        assert isinstance(call["query_filter"], models.Filter)
        assert len(call["query_filter"].must) == 2
        assert results == [
            {
                "id": "L21_V001:42",
                "score": 0.91,
                "payload": {
                    "external_id": "L21_V001:42",
                    "video_id": "L21_V001",
                },
            }
        ]

    def test_legacy_search_api_is_supported(self) -> None:
        client = LegacyQdrantClient()
        store = make_store(client)

        results = store.search(np.ones(4, dtype=np.float32), top_k=3)

        assert client.calls[0]["query_vector"] == [1.0, 1.0, 1.0, 1.0]
        assert results[0]["id"] == "7"

    def test_count_returns_exact_count(self) -> None:
        client = FakeQdrantClient()
        client.count_value = 17
        store = make_store(client)

        assert store.count() == 17
        assert store.count(filters={"batch_id": "b2", "video_id": "L99_V001"}) == 17
        assert isinstance(client.counts[-1]["count_filter"], models.Filter)
        assert len(client.counts[-1]["count_filter"].must) == 2

    def test_fixed_collection_availability_requires_selected_points(self) -> None:
        client = FakeQdrantClient(exists=True)
        store = make_store(client)

        assert store.is_available({"batch_id": ["b1"]}) is False
        client.count_value = 3
        assert store.is_available({"batch_id": ["b1"]}) is True
        assert client.counts[-1]["count_filter"] is not None

    def test_iter_payloads_follows_scroll_pagination(self) -> None:
        client = FakeQdrantClient()
        store = make_store(client)

        payloads = list(store.iter_payloads({"batch_id": "b2"}, batch_size=1))

        assert [payload["video_id"] for payload in payloads] == ["L99_V001", "L99_V002"]
        assert client.scrolls[0]["offset"] is None
        assert client.scrolls[1]["offset"] == "page-2"
        assert isinstance(client.scrolls[0]["scroll_filter"], models.Filter)

    def test_grouped_search_uses_qdrant_group_api(self) -> None:
        class GroupedClient(FakeQdrantClient):
            def query_points_groups(self, **kwargs):
                self.queries.append(kwargs)
                hit = SimpleNamespace(
                    id="point-1",
                    score=0.93,
                    payload={"external_id": "frame-1", "video_id": "L21_V001"},
                )
                return SimpleNamespace(groups=[SimpleNamespace(id="L21_V001", hits=[hit])])

        client = GroupedClient()
        store = make_store(client)

        groups = store.search_grouped(
            np.ones(4, dtype=np.float32),
            group_by="video_id",
            top_k_groups=300,
            group_size=1,
            filters={"batch_id": "b1"},
        )

        assert groups == [
            {
                "group_id": "L21_V001",
                "score": 0.93,
                "hits": [
                    {
                        "id": "frame-1",
                        "score": 0.93,
                        "payload": {
                            "external_id": "frame-1",
                            "video_id": "L21_V001",
                        },
                        "_point_id": "point-1",
                    }
                ],
            }
        ]
        assert client.queries[0]["group_by"] == "video_id"
        assert client.queries[0]["limit"] == 300
        assert client.queries[0]["with_vectors"] is False

    def test_grouped_search_can_return_unnamed_or_named_vectors(self) -> None:
        class GroupedVectorClient(FakeQdrantClient):
            def query_points_groups(self, **kwargs):
                self.queries.append(kwargs)
                hits = [
                    SimpleNamespace(
                        id="point-1",
                        score=0.93,
                        payload={"external_id": "frame-1", "video_id": "L21_V001"},
                        vector=[1.0, 0.0, 0.0, 0.0],
                    ),
                    SimpleNamespace(
                        id="point-2",
                        score=0.91,
                        payload={"external_id": "frame-2", "video_id": "L21_V001"},
                        vector={"image": [0.0, 1.0, 0.0, 0.0]},
                    ),
                ]
                return SimpleNamespace(groups=[SimpleNamespace(id="L21_V001", hits=hits)])

        client = GroupedVectorClient()
        store = make_store(client)

        groups = store.search_grouped(
            np.ones(4, dtype=np.float32),
            group_size=2,
            with_vectors=True,
        )

        assert groups[0]["hits"][0]["vector"] == [1.0, 0.0, 0.0, 0.0]
        assert groups[0]["hits"][1]["vector"] == [0.0, 1.0, 0.0, 0.0]
        assert client.queries[0]["with_vectors"] is True

    def test_retrieve_points_deduplicates_batches_and_preserves_request_order(self) -> None:
        class RetrieveClient(FakeQdrantClient):
            def __init__(self) -> None:
                super().__init__()
                self.retrieves: list[dict] = []

            def retrieve(self, **kwargs):
                self.retrieves.append(kwargs)
                points = {
                    "point-1": SimpleNamespace(
                        id="point-1",
                        payload={"external_id": "frame-1", "video_id": "L21_V001"},
                        vector=[1.0, 0.0, 0.0, 0.0],
                    ),
                    "point-2": SimpleNamespace(
                        id="point-2",
                        payload={"external_id": "frame-2", "video_id": "L21_V001"},
                        vector=[0.0, 1.0, 0.0, 0.0],
                    ),
                }
                # Qdrant does not promise to echo caller order.
                return [points[item] for item in reversed(kwargs["ids"]) if item in points]

        client = RetrieveClient()
        store = make_store(client)

        points = store.retrieve_points_with_vectors(
            ["point-2", "point-1", "point-2", "missing"],
            batch_size=2,
        )

        assert [point["id"] for point in points] == ["frame-2", "frame-1"]
        assert [point["_point_id"] for point in points] == ["point-2", "point-1"]
        assert [call["ids"] for call in client.retrieves] == [
            ["point-2", "point-1"],
            ["missing"],
        ]
        assert all(call["with_vectors"] is True for call in client.retrieves)

    def test_retrieve_points_keeps_missing_vector_for_caller_validation(self) -> None:
        class MissingVectorClient(FakeQdrantClient):
            def retrieve(self, **_kwargs):
                return [
                    SimpleNamespace(
                        id="point-1",
                        payload={"external_id": "frame-1", "video_id": "L21_V001"},
                        vector=None,
                    )
                ]

        points = make_store(MissingVectorClient()).retrieve_points_with_vectors(["point-1"])

        assert points[0]["_point_id"] == "point-1"
        assert points[0]["vector"] is None

    def test_retrieve_points_initializes_grpc_client_lazily(self, monkeypatch) -> None:
        captured: list[dict] = []

        class RecordingClient:
            def __init__(self, **kwargs) -> None:
                captured.append(kwargs)

            def retrieve(self, **_kwargs):
                return []

        monkeypatch.setattr(qdrant_client, "QdrantClient", RecordingClient)
        store = VectorStore(prefer_grpc=True, grpc_port=6334)

        assert store._client is None
        assert store.retrieve_points_with_vectors([]) == []
        assert store._client is None
        assert store.retrieve_points_with_vectors(["point-1"]) == []
        assert captured[0]["prefer_grpc"] is True
        assert captured[0]["grpc_port"] == 6334

    def test_iter_points_with_vectors_follows_scroll_pagination(self) -> None:
        class VectorScrollClient(FakeQdrantClient):
            def scroll(self, **kwargs):
                self.scrolls.append(kwargs)
                index = 1 if kwargs.get("offset") is None else 2
                point = SimpleNamespace(
                    id=f"point-{index}",
                    payload={
                        "external_id": f"frame-{index}",
                        "video_id": "L21_V001",
                        "source_frame_idx": index,
                    },
                    vector=[float(index), 0.0, 0.0, 0.0],
                )
                return ([point], "page-2" if index == 1 else None)

        client = VectorScrollClient()
        store = make_store(client)

        points = list(store.iter_points_with_vectors({"video_id": "L21_V001"}))

        assert [point["id"] for point in points] == ["frame-1", "frame-2"]
        assert points[0]["vector"] == [1.0, 0.0, 0.0, 0.0]
        assert client.scrolls[0]["with_vectors"] is True
        assert client.scrolls[1]["offset"] == "page-2"

    def test_iter_point_batches_vectorizes_each_qdrant_page(self) -> None:
        class VectorPageClient(FakeQdrantClient):
            def scroll(self, **kwargs):
                self.scrolls.append(kwargs)
                index = 1 if kwargs.get("offset") is None else 2
                return (
                    [
                        SimpleNamespace(
                            id=f"point-{index}",
                            payload={
                                "external_id": f"frame-{index}",
                                "video_id": "L21_V001",
                            },
                            vector=[float(index), 0.0, 0.0, 0.0],
                        ),
                        # Invalid vector rows are excluded without poisoning
                        # the contiguous page matrix.
                        SimpleNamespace(
                            id=f"bad-{index}",
                            payload={"video_id": "L21_V001"},
                            vector=None,
                        ),
                        SimpleNamespace(
                            id=f"malformed-{index}",
                            payload={"video_id": "L21_V001"},
                            vector=["not-a-number", "x", "y", "z"],
                        ),
                    ],
                    "page-2" if index == 1 else None,
                )

        client = VectorPageClient()
        batches = list(make_store(client).iter_point_batches_with_vectors(batch_size=8))

        assert all(isinstance(batch, VectorPointBatch) for batch in batches)
        assert [batch.vectors.shape for batch in batches] == [(1, 4), (1, 4)]
        assert [batch.source_point_count for batch in batches] == [3, 3]
        assert batches[0].vectors.flags.c_contiguous
        assert batches[1].payloads[0]["external_id"] == "frame-2"
        assert client.scrolls[0]["limit"] == 8
        assert client.scrolls[1]["offset"] == "page-2"

    def test_iter_point_batches_yields_all_invalid_page_count(self) -> None:
        class InvalidPageClient(FakeQdrantClient):
            def scroll(self, **_kwargs):
                return (
                    [
                        SimpleNamespace(
                            id="bad",
                            payload={"video_id": "L21_V001"},
                            vector=None,
                        )
                    ],
                    None,
                )

        batches = list(
            make_store(InvalidPageClient()).iter_point_batches_with_vectors()
        )

        assert len(batches) == 1
        assert batches[0].source_point_count == 1
        assert batches[0].vectors.shape == (0, 4)
        assert batches[0].payloads == ()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vector_size": 0},
            {"distance": "Chebyshev"},
            {"collection_name": ""},
        ],
    )
    def test_constructor_validation(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            VectorStore(**kwargs)

    @pytest.mark.parametrize("grpc_port", (0, 65_536, True, 6334.5))
    def test_constructor_rejects_invalid_grpc_port(self, grpc_port: object) -> None:
        with pytest.raises(ValueError, match="grpc_port"):
            VectorStore(grpc_port=grpc_port)  # type: ignore[arg-type]

    def test_constructor_rejects_non_boolean_prefer_grpc(self) -> None:
        with pytest.raises(TypeError, match="prefer_grpc"):
            VectorStore(prefer_grpc=1)  # type: ignore[arg-type]


class TestBatchVectorStore:
    template = "aic_native_{batch}_siglip2_b16_224_r75de2d55_768d_v1"

    def test_batch_name_is_readable_and_sanitization_cannot_alias(self) -> None:
        assert batch_collection_name(self.template, "aic26-b2") == (
            "aic_native_aic26-b2_siglip2_b16_224_r75de2d55_768d_v1"
        )
        assert collection_batch_token("batch/2").startswith("batch_2_")
        assert collection_batch_token("batch/2") != collection_batch_token("batch:2")

    def test_template_must_have_exactly_one_batch_placeholder(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            BatchVectorStore(
                url="http://localhost:6333",
                collection_template="global_vectors",
                vector_size=4,
            )

    def test_for_batch_binds_mutating_store_to_one_physical_collection(self) -> None:
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
        )

        bound = routed.for_batch("aic26-b2")

        assert isinstance(bound, VectorStore)
        assert bound.collection_name == ("aic_native_aic26-b2_siglip2_b16_224_r75de2d55_768d_v1")

    def test_for_batch_propagates_grpc_transport(self) -> None:
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
            prefer_grpc=True,
            grpc_port=6334,
        )

        bound = routed.for_batch("aic26-b2")

        assert bound.prefer_grpc is True
        assert bound.grpc_port == 6334

    def test_batch_constructor_validates_grpc_transport(self) -> None:
        with pytest.raises(ValueError, match="grpc_port"):
            BatchVectorStore(
                url="http://localhost:6333",
                collection_template=self.template,
                vector_size=4,
                grpc_port=0,
            )

    def test_batch_lazy_client_initialization_is_single_flight(self, monkeypatch) -> None:
        created: list[object] = []
        creation_lock = Lock()

        class SlowClient:
            def __init__(self, **_kwargs) -> None:
                time.sleep(0.01)
                with creation_lock:
                    created.append(self)

        monkeypatch.setattr(qdrant_client, "QdrantClient", SlowClient)
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            clients = tuple(executor.map(lambda _: routed._get_client(), range(16)))

        assert len(created) == 1
        assert all(client is clients[0] for client in clients)

    def test_search_routes_filters_and_merges_requested_batches(self) -> None:
        client = RoutedQdrantClient()
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
        )
        routed._client = client

        results = routed.search(
            np.ones(4, dtype=np.float32),
            top_k=10,
            filters={"batch_id": ["aic25-b1", "aic26-b2"]},
        )

        assert [item["payload"]["batch_id"] for item in results] == [
            "aic26-b2",
            "aic25-b1",
        ]
        assert {call["collection_name"] for call in client.queries} == client.names

    def test_unfiltered_search_discovers_only_matching_batch_collections(self) -> None:
        client = RoutedQdrantClient()
        client.names.add("unrelated_collection")
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
        )
        routed._client = client

        routed.search(np.ones(4, dtype=np.float32), top_k=10)

        assert {call["collection_name"] for call in client.queries} == client.names - {
            "unrelated_collection"
        }

    def test_vector_stream_routes_to_selected_batch_collection(self) -> None:
        class RoutedVectorClient(RoutedQdrantClient):
            def scroll(self, **kwargs):
                self.scrolls.append(kwargs)
                return (
                    [
                        SimpleNamespace(
                            id="point",
                            payload={"external_id": "frame", "batch_id": "aic26-b2"},
                            vector=[1.0, 0.0, 0.0, 0.0],
                        )
                    ],
                    None,
                )

        client = RoutedVectorClient()
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
        )
        routed._client = client

        points = list(
            routed.iter_points_with_vectors(
                {"batch_id": "aic26-b2"},
                batch_size=4,
            )
        )

        assert [point["id"] for point in points] == ["frame"]
        assert client.scrolls[0]["collection_name"] == (
            "aic_native_aic26-b2_siglip2_b16_224_r75de2d55_768d_v1"
        )

    def test_grouped_vector_search_routes_vector_request(self) -> None:
        class RoutedGroupedClient(RoutedQdrantClient):
            def query_points_groups(self, **kwargs):
                self.queries.append(kwargs)
                return SimpleNamespace(
                    groups=[
                        SimpleNamespace(
                            id="L21_V001",
                            hits=[
                                SimpleNamespace(
                                    id="point",
                                    score=0.9,
                                    payload={
                                        "batch_id": "aic26-b2",
                                        "video_id": "L21_V001",
                                    },
                                    vector=[1.0, 0.0, 0.0, 0.0],
                                )
                            ],
                        )
                    ]
                )

        client = RoutedGroupedClient()
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
        )
        routed._client = client

        groups = routed.search_grouped(
            np.ones(4, dtype=np.float32),
            filters={"batch_id": "aic26-b2"},
            with_vectors=True,
        )

        assert groups[0]["hits"][0]["vector"] == [1.0, 0.0, 0.0, 0.0]
        assert client.queries[0]["with_vectors"] is True

    def test_point_retrieval_routes_to_only_the_selected_batch(self) -> None:
        class RoutedRetrieveClient(RoutedQdrantClient):
            def __init__(self) -> None:
                super().__init__()
                self.retrieves: list[dict] = []

            def retrieve(self, **kwargs):
                self.retrieves.append(kwargs)
                batch_id = (
                    "aic25-b1" if "aic25-b1" in kwargs["collection_name"] else "aic26-b2"
                )
                return [
                    SimpleNamespace(
                        id=kwargs["ids"][0],
                        payload={"external_id": "frame", "batch_id": batch_id},
                        vector=[1.0, 0.0, 0.0, 0.0],
                    )
                ]

        client = RoutedRetrieveClient()
        routed = BatchVectorStore(
            url="http://localhost:6333",
            collection_template=self.template,
            vector_size=4,
        )
        routed._client = client

        points = routed.retrieve_points_with_vectors(
            ["point"],
            filters={"batch_id": "aic26-b2"},
        )

        assert points[0]["payload"]["batch_id"] == "aic26-b2"
        assert len(client.retrieves) == 1
        assert "aic26-b2" in client.retrieves[0]["collection_name"]
