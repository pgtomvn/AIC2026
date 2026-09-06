"""Lazy Qdrant adapter for dense retrieval.

Importing or constructing :class:`VectorStore` does not contact Qdrant.  The
first operation that needs the service creates the client.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

import numpy as np

_DISTANCE_NAMES = {
    "cosine": "COSINE",
    "euclid": "EUCLID",
    "dot": "DOT",
    "manhattan": "MANHATTAN",
}

_PAYLOAD_INDEXES = {
    "batch_id": "KEYWORD",
    "video_id": "KEYWORD",
    "source_frame_idx": "INTEGER",
    "shot_id": "KEYWORD",
}

_BATCH_PLACEHOLDER = "{batch}"
_SAFE_COLLECTION_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
_INSECURE_KEY_WARNING = r"^Api key is used with an insecure connection\.$"
_TRAJECTORY_PAYLOAD_FIELDS = (
    "batch_id",
    "video_id",
    "source_frame_idx",
    "timestamp_ms",
)


@dataclass(frozen=True, slots=True)
class VectorPointBatch:
    """One contiguous Qdrant vector page with aligned payload metadata."""

    point_ids: tuple[Any, ...]
    payloads: tuple[Mapping[str, Any], ...]
    vectors: np.ndarray
    source_point_count: int


def _validate_grpc_transport(*, prefer_grpc: bool, grpc_port: int | None) -> None:
    if not isinstance(prefer_grpc, bool):
        raise TypeError("prefer_grpc must be a boolean")
    if grpc_port is not None and (
        isinstance(grpc_port, bool)
        or not isinstance(grpc_port, int)
        or not 1 <= grpc_port <= 65_535
    ):
        raise ValueError("grpc_port must be an integer between 1 and 65535")


def _is_literal_loopback_http(url: str) -> bool:
    """Return true only for an HTTP URL whose literal host is loopback."""

    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme.casefold() != "http" or host is None:
            return False
        if host.casefold() == "localhost":
            return True
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def collection_batch_token(batch_id: str) -> str:
    """Return a stable, Qdrant-safe token that preserves batch identity.

    Ordinary competition batch IDs (letters, digits, hyphens, and underscores)
    remain readable.  Unusual IDs receive a digest suffix so sanitization cannot
    silently alias two distinct batches.
    """

    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("batch_id must be a non-empty string")
    value = batch_id.strip()
    if _SAFE_COLLECTION_COMPONENT.fullmatch(value):
        return value
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "batch"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}_{digest}"


def batch_collection_name(collection_template: str, batch_id: str) -> str:
    """Resolve exactly one ``{batch}`` placeholder to a safe physical name."""

    if collection_template.count(_BATCH_PLACEHOLDER) != 1:
        raise ValueError("a batch collection template must contain exactly one {batch} placeholder")
    name = collection_template.replace(_BATCH_PLACEHOLDER, collection_batch_token(batch_id))
    if len(name) > 255 or not _SAFE_COLLECTION_COMPONENT.fullmatch(name):
        raise ValueError(f"invalid resolved Qdrant collection name: {name!r}")
    return name


class VectorStore:
    """Qdrant vector database client wrapper.

    Args:
        url: Qdrant server URL.
        collection_name: Name of the Qdrant collection.
        vector_size: Dimensionality of stored vectors.
        distance: Distance metric (``Cosine``, ``Euclid``, ``Dot``, or
            ``Manhattan``).
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "aic2026_keyframes",
        vector_size: int = 1024,
        distance: str = "Cosine",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        grpc_port: int | None = None,
    ) -> None:
        if not url:
            raise ValueError("url must not be empty")
        if not collection_name:
            raise ValueError("collection_name must not be empty")
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if not isinstance(distance, str):
            raise TypeError("distance must be a string")
        if distance.casefold() not in _DISTANCE_NAMES:
            choices = ", ".join(name.title() for name in _DISTANCE_NAMES)
            raise ValueError(f"unsupported distance {distance!r}; expected one of {choices}")
        _validate_grpc_transport(prefer_grpc=prefer_grpc, grpc_port=grpc_port)

        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self.grpc_port = grpc_port
        self._client: Any = None
        self._client_lock = RLock()

    def _client_kwargs(self) -> dict[str, Any]:
        """Return transport settings for lazy Qdrant client construction."""

        kwargs: dict[str, Any] = {
            "url": self.url,
            "api_key": self.api_key,
            "prefer_grpc": self.prefer_grpc,
        }
        if self.grpc_port is not None:
            kwargs["grpc_port"] = self.grpc_port
        return kwargs

    def _connect(self) -> None:
        """Create the Qdrant client without issuing a service request."""
        if self._client is not None:
            return
        with self._client_lock:
            if self._client is not None:
                return
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover - dependency is part of the lockfile
                raise RuntimeError(
                    "Qdrant support is unavailable; install the locked project dependencies"
                ) from exc
            client_kwargs = self._client_kwargs()
            if self.api_key and _is_literal_loopback_http(self.url):
                # Authentication is intentionally retained for the loopback-only
                # Compose service. Suppress only Qdrant's exact generic warning;
                # non-loopback plaintext endpoints must continue to warn.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=_INSECURE_KEY_WARNING,
                        category=UserWarning,
                    )
                    self._client = QdrantClient(**client_kwargs)
            else:
                self._client = QdrantClient(**client_kwargs)

    def _get_client(self) -> Any:
        if self._client is None:
            self._connect()
        return self._client

    def _distance_model(self) -> Any:
        from qdrant_client import models

        return getattr(models.Distance, _DISTANCE_NAMES[self.distance.casefold()])

    def _validate_existing_collection(self, client: Any) -> None:
        """Reject accidental reuse of an incompatible collection when inspectable."""
        get_collection = getattr(client, "get_collection", None)
        if not callable(get_collection):
            return

        info = get_collection(collection_name=self.collection_name)
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return
        if isinstance(vectors, Mapping):
            raise ValueError(
                f"collection {self.collection_name!r} uses named vectors; expected one "
                "unnamed vector. Pass recreate=True to replace it."
            )

        actual_size = getattr(vectors, "size", None)
        actual_distance = getattr(vectors, "distance", None)
        if hasattr(actual_distance, "value"):
            actual_distance = actual_distance.value

        expected_distance = self.distance.casefold()
        wrong_size = actual_size is not None and int(actual_size) != self.vector_size
        wrong_distance = (
            actual_distance is not None and str(actual_distance).casefold() != expected_distance
        )
        if wrong_size or wrong_distance:
            raise ValueError(
                f"collection {self.collection_name!r} has vector configuration "
                f"size={actual_size}, distance={actual_distance}; expected "
                f"size={self.vector_size}, distance={self.distance}. "
                "Pass recreate=True to replace it."
            )

    def _ensure_payload_indexes(self, client: Any) -> None:
        create_payload_index = getattr(client, "create_payload_index", None)
        if not callable(create_payload_index):
            return

        from qdrant_client import models

        for field_name, schema_name in _PAYLOAD_INDEXES.items():
            try:
                create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=getattr(models.PayloadSchemaType, schema_name),
                    wait=True,
                )
            except NotImplementedError:
                # Some embedded/test backends intentionally do not implement payload indexes.
                return

    def create_collection(self, recreate: bool = False) -> None:
        """Create the configured collection and its canonical payload indexes.

        Existing collections are validated against ``vector_size`` and ``distance``.
        Set ``recreate`` to explicitly replace an incompatible collection.
        """
        from qdrant_client import models

        client = self._get_client()
        exists = bool(client.collection_exists(collection_name=self.collection_name))
        if recreate and exists:
            client.delete_collection(collection_name=self.collection_name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self._distance_model(),
                ),
            )
        else:
            self._validate_existing_collection(client)

        self._ensure_payload_indexes(client)

    def is_available(self, filters: Optional[dict] = None) -> bool:
        """Return whether the fixed collection has points for ``filters``.

        Fixed organizer collections can contain more than one logical batch.
        Checking existence alone therefore risks loading a heavy encoder for a
        batch that has no compatible points.  A filtered exact count keeps the
        availability decision aligned with the query that will follow.
        """

        client = self._get_client()
        if not client.collection_exists(collection_name=self.collection_name):
            return False
        return self.count(filters) > 0

    def _point_id(self, external_id: str) -> str:
        """Map an arbitrary external string ID to a stable Qdrant UUID."""
        return str(uuid5(NAMESPACE_URL, f"aic2026:qdrant:{self.collection_name}:{external_id}"))

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        """Validate and batch-upsert vectors and payloads.

        Qdrant point IDs can only be unsigned integers or UUIDs.  Arbitrary
        application IDs are deterministically converted to UUID5 values and are
        retained in each payload as ``external_id``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        matrix = np.asarray(vectors)
        if matrix.ndim != 2:
            raise ValueError(f"vectors must have shape (N, D); received {matrix.shape}")
        if matrix.shape[1] != self.vector_size:
            raise ValueError(
                f"vector dimension {matrix.shape[1]} does not match collection "
                f"dimension {self.vector_size}"
            )
        if len(ids) != matrix.shape[0] or len(payloads) != matrix.shape[0]:
            raise ValueError("ids, vectors, and payloads must have the same length")
        if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
            matrix.dtype, np.complexfloating
        ):
            raise TypeError("vectors must contain numeric values")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("vectors must not contain NaN or infinite values")
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("every point ID must be a non-empty string")
        if len(set(ids)) != len(ids):
            raise ValueError("point IDs must be unique within an upsert call")
        if any(not isinstance(payload, dict) for payload in payloads):
            raise TypeError("every payload must be a dictionary")
        if not ids:
            return

        from qdrant_client import models

        client = self._get_client()
        float_vectors = matrix.astype(np.float32, copy=False)
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            points = []
            for index in range(start, stop):
                external_id = ids[index]
                payload = dict(payloads[index])
                existing_external_id = payload.get("external_id", external_id)
                if str(existing_external_id) != external_id:
                    raise ValueError(
                        f"payload external_id {existing_external_id!r} conflicts with "
                        f"point ID {external_id!r}"
                    )
                payload["external_id"] = external_id
                points.append(
                    models.PointStruct(
                        id=self._point_id(external_id),
                        vector=float_vectors[index].tolist(),
                        payload=payload,
                    )
                )
            client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    @staticmethod
    def _build_filter(filters: Optional[dict[str, Any]]) -> Any:
        if filters is None:
            return None

        from qdrant_client import models

        if isinstance(filters, models.Filter):
            return filters
        if not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or Qdrant Filter")
        if not filters:
            return None
        if {"must", "should", "must_not", "min_should"}.intersection(filters):
            return models.Filter.model_validate(filters)

        conditions = []
        range_fields = {"gt", "gte", "lt", "lte"}
        for field_name, value in filters.items():
            if isinstance(value, Mapping) and range_fields.intersection(value):
                conditions.append(
                    models.FieldCondition(key=field_name, range=models.Range(**dict(value)))
                )
            elif isinstance(value, (list, tuple, set, frozenset)):
                conditions.append(
                    models.FieldCondition(
                        key=field_name,
                        match=models.MatchAny(any=list(value)),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=field_name,
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[dict] = None,
        *,
        exact: bool = False,
    ) -> list[dict]:
        """Return nearest neighbors as ``id``, ``score``, and ``payload`` mappings.

        ``exact`` is reserved for parity/verification tools. Production retrieval keeps the
        configured ANN path unless a caller explicitly requests an exhaustive search.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(query_vector)
        if vector.ndim != 1 or vector.shape[0] != self.vector_size:
            raise ValueError(
                f"query_vector must have shape ({self.vector_size},); received {vector.shape}"
            )
        if not np.issubdtype(vector.dtype, np.number) or np.issubdtype(
            vector.dtype, np.complexfloating
        ):
            raise TypeError("query_vector must contain numeric values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("query_vector must not contain NaN or infinite values")

        client = self._get_client()
        query_filter = self._build_filter(filters)
        query = vector.astype(np.float32, copy=False).tolist()

        search_params = None
        if exact:
            from qdrant_client import models

            search_params = models.SearchParams(exact=True)

        query_points = getattr(client, "query_points", None)
        if callable(query_points):
            query_kwargs = dict(
                collection_name=self.collection_name,
                query=query,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            if search_params is not None:
                query_kwargs["search_params"] = search_params
            response = query_points(**query_kwargs)
            hits = getattr(response, "points", response)
        else:
            legacy_search = getattr(client, "search", None)
            if not callable(legacy_search):
                raise RuntimeError("the installed Qdrant client has no supported search method")
            search_kwargs = dict(
                collection_name=self.collection_name,
                query_vector=query,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            if search_params is not None:
                search_kwargs["search_params"] = search_params
            hits = legacy_search(**search_kwargs)

        results = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", None) or {})
            point_id = getattr(hit, "id", None)
            external_id = payload.get("external_id", point_id)
            results.append(
                {
                    "id": str(external_id),
                    "score": float(getattr(hit, "score", 0.0)),
                    "payload": payload,
                }
            )
        return results

    @staticmethod
    def _normalized_hit(
        hit: Any,
        *,
        with_vector: bool = False,
    ) -> dict[str, Any]:
        payload = dict(
            (hit.get("payload") if isinstance(hit, Mapping) else getattr(hit, "payload", None))
            or {}
        )
        point_id = hit.get("id") if isinstance(hit, Mapping) else getattr(hit, "id", None)
        score = hit.get("score", 0.0) if isinstance(hit, Mapping) else getattr(hit, "score", 0.0)
        normalized = {
            "id": str(payload.get("external_id", point_id)),
            "score": float(score),
            "payload": payload,
        }
        if point_id is not None:
            # ``id`` remains the canonical/external identity exposed by the
            # adapter.  TRAKE uses this private store identity to retrieve
            # vectors only after it has pruned grouped candidates.
            normalized["_point_id"] = point_id
        if with_vector:
            vector = hit.get("vector") if isinstance(hit, Mapping) else getattr(hit, "vector", None)
            if isinstance(vector, Mapping):
                vector = next(iter(vector.values())) if len(vector) == 1 else None
            normalized["vector"] = vector
        return normalized

    def retrieve_points_with_vectors(
        self,
        point_ids: list[Any] | tuple[Any, ...],
        *,
        filters: Optional[dict[str, Any]] = None,
        batch_size: int = 4_096,
    ) -> list[dict[str, Any]]:
        """Retrieve exact Qdrant points by private store identity.

        ``filters`` is accepted for parity with :class:`BatchVectorStore`; a
        fixed collection is already the isolation boundary.  Input IDs are
        deduplicated in first-seen order, and records are returned in that same
        deterministic order even if Qdrant reorders its response.
        """

        del filters
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        ordered_ids: list[Any] = []
        seen: set[str] = set()
        for point_id in point_ids:
            if point_id is None:
                continue
            identity = str(point_id)
            if identity in seen:
                continue
            seen.add(identity)
            ordered_ids.append(point_id)
        if not ordered_ids:
            return []

        client = self._get_client()
        retrieve = getattr(client, "retrieve", None)
        if not callable(retrieve):
            raise NotImplementedError("the Qdrant client does not support point retrieval")

        result_by_id: dict[str, dict[str, Any]] = {}
        for start in range(0, len(ordered_ids), batch_size):
            requested = ordered_ids[start : start + batch_size]
            response = retrieve(
                collection_name=self.collection_name,
                ids=requested,
                with_payload=True,
                with_vectors=True,
            )
            points = (
                response.get("points", ())
                if isinstance(response, Mapping)
                else getattr(response, "points", response)
            )
            for point in points:
                normalized = self._normalized_hit(point, with_vector=True)
                actual_id = normalized.get("_point_id")
                if actual_id is None:
                    continue
                identity = str(actual_id)
                result_by_id.setdefault(identity, normalized)

        return [
            result_by_id[str(point_id)]
            for point_id in ordered_ids
            if str(point_id) in result_by_id
        ]

    def search_grouped(
        self,
        query_vector: np.ndarray,
        *,
        group_by: str = "video_id",
        top_k_groups: int = 300,
        group_size: int = 1,
        filters: Optional[dict] = None,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Search nearest frames grouped by one canonical payload field.

        Current Qdrant clients use ``query_points_groups``.  A deterministic
        client-side fallback preserves offline fakes and older compatible
        clients without changing score semantics.
        """

        if not isinstance(group_by, str) or not group_by.strip():
            raise ValueError("group_by must be a non-empty string")
        if top_k_groups < 1 or group_size < 1:
            raise ValueError("top_k_groups and group_size must be positive")
        vector = np.asarray(query_vector)
        if vector.ndim != 1 or vector.shape[0] != self.vector_size:
            raise ValueError(
                f"query_vector must have shape ({self.vector_size},); received {vector.shape}"
            )
        if not np.issubdtype(vector.dtype, np.number) or np.issubdtype(
            vector.dtype, np.complexfloating
        ):
            raise TypeError("query_vector must contain numeric values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("query_vector must not contain NaN or infinite values")

        client = self._get_client()
        grouped_query = getattr(client, "query_points_groups", None)
        if callable(grouped_query):
            response = grouped_query(
                collection_name=self.collection_name,
                query=vector.astype(np.float32, copy=False).tolist(),
                group_by=group_by,
                limit=top_k_groups,
                group_size=group_size,
                query_filter=self._build_filter(filters),
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            groups = (
                response.get("groups", ())
                if isinstance(response, Mapping)
                else getattr(response, "groups", response)
            )
            results: list[dict[str, Any]] = []
            for group in groups:
                raw_hits = (
                    group.get("hits", ())
                    if isinstance(group, Mapping)
                    else getattr(group, "hits", ())
                )
                hits = [self._normalized_hit(hit, with_vector=with_vectors) for hit in raw_hits]
                if not hits:
                    continue
                raw_id = (
                    group.get("id") if isinstance(group, Mapping) else getattr(group, "id", None)
                )
                group_id = raw_id if raw_id is not None else hits[0]["payload"].get(group_by)
                if group_id is None:
                    continue
                hits.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
                results.append(
                    {
                        "group_id": str(group_id),
                        "score": float(hits[0]["score"]),
                        "hits": hits,
                    }
                )
            results.sort(key=lambda item: (-float(item["score"]), str(item["group_id"])))
            return results[:top_k_groups]

        hits = self.search(
            vector,
            top_k=top_k_groups * group_size,
            filters=filters,
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for hit in hits:
            group_id = hit["payload"].get(group_by)
            if group_id is None:
                continue
            grouped.setdefault(str(group_id), []).append(hit)
        results = []
        for group_id, group_hits in grouped.items():
            group_hits.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
            retained = group_hits[:group_size]
            results.append(
                {
                    "group_id": group_id,
                    "score": float(retained[0]["score"]),
                    "hits": retained,
                }
            )
        results.sort(key=lambda item: (-float(item["score"]), str(item["group_id"])))
        return results[:top_k_groups]

    def grouped_search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Compatibility alias for :meth:`search_grouped`."""

        return self.search_grouped(*args, **kwargs)

    def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        """Return an exact total, optionally scoped by canonical payload filters."""
        result = self._get_client().count(
            collection_name=self.collection_name,
            count_filter=self._build_filter(filters),
            exact=True,
        )
        if isinstance(result, Mapping):
            return int(result["count"])
        if hasattr(result, "count"):
            return int(result.count)
        return int(result)

    def iter_payloads(
        self,
        filters: Optional[dict[str, Any]] = None,
        *,
        batch_size: int = 256,
    ) -> Iterator[dict[str, Any]]:
        """Stream payloads exactly, following Qdrant scroll pagination."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        client = self._get_client()
        offset: Any = None
        while True:
            response = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_filter(filters),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(response, tuple):
                points, next_offset = response
            else:
                points = getattr(response, "points", response)
                next_offset = getattr(response, "next_page_offset", None)
            for point in points:
                yield dict(getattr(point, "payload", None) or {})
            if next_offset is None:
                break
            offset = next_offset

    def iter_points_with_vectors(
        self,
        filters: Optional[dict[str, Any]] = None,
        *,
        batch_size: int = 256,
    ) -> Iterator[dict[str, Any]]:
        """Stream exact point identity, payload, and vectors with pagination."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        client = self._get_client()
        offset: Any = None
        while True:
            response = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_filter(filters),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if isinstance(response, tuple):
                points, next_offset = response
            else:
                points = getattr(response, "points", response)
                next_offset = getattr(response, "next_page_offset", None)
            for point in points:
                payload = dict(
                    (
                        point.get("payload")
                        if isinstance(point, Mapping)
                        else getattr(point, "payload", None)
                    )
                    or {}
                )
                point_id = (
                    point.get("id") if isinstance(point, Mapping) else getattr(point, "id", None)
                )
                vector = (
                    point.get("vector")
                    if isinstance(point, Mapping)
                    else getattr(point, "vector", None)
                )
                if isinstance(vector, Mapping):
                    if len(vector) != 1:
                        raise ValueError("point has ambiguous named vectors")
                    vector = next(iter(vector.values()))
                yield {
                    "id": str(payload.get("external_id", point_id)),
                    "payload": payload,
                    "vector": vector,
                }
            if next_offset is None:
                break
            offset = next_offset

    def iter_point_batches_with_vectors(
        self,
        filters: Optional[dict[str, Any]] = None,
        *,
        batch_size: int = 16_384,
    ) -> Iterator[VectorPointBatch]:
        """Stream contiguous vector pages without per-point adapter objects."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        client = self._get_client()
        offset: Any = None
        while True:
            response = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_filter(filters),
                limit=batch_size,
                offset=offset,
                with_payload=list(_TRAJECTORY_PAYLOAD_FIELDS),
                with_vectors=True,
            )
            if isinstance(response, tuple):
                points, next_offset = response
            else:
                points = getattr(response, "points", response)
                next_offset = getattr(response, "next_page_offset", None)
            point_ids: list[Any] = []
            payloads: list[Mapping[str, Any]] = []
            raw_vectors: list[Any] = []
            source_point_count = 0
            for point in points:
                source_point_count += 1
                payload = (
                    point.get("payload", {})
                    if isinstance(point, Mapping)
                    else getattr(point, "payload", None) or {}
                )
                vector = (
                    point.get("vector")
                    if isinstance(point, Mapping)
                    else getattr(point, "vector", None)
                )
                if isinstance(vector, Mapping):
                    vector = next(iter(vector.values())) if len(vector) == 1 else None
                try:
                    vector_length = len(vector)  # type: ignore[arg-type]
                except TypeError:
                    continue
                if vector_length != self.vector_size:
                    continue
                point_id = (
                    point.get("id") if isinstance(point, Mapping) else getattr(point, "id", None)
                )
                point_ids.append(point_id)
                payloads.append(payload)
                raw_vectors.append(vector)
            matrix = np.empty((0, self.vector_size), dtype=np.float32)
            if raw_vectors:
                try:
                    matrix = np.asarray(raw_vectors, dtype=np.float32)
                except (TypeError, ValueError):
                    valid_ids: list[Any] = []
                    valid_payloads: list[Mapping[str, Any]] = []
                    valid_rows: list[np.ndarray] = []
                    for point_id, payload, raw_vector in zip(
                        point_ids, payloads, raw_vectors, strict=True
                    ):
                        try:
                            row = np.asarray(raw_vector, dtype=np.float32)
                        except (TypeError, ValueError):
                            continue
                        if row.shape != (self.vector_size,):
                            continue
                        valid_ids.append(point_id)
                        valid_payloads.append(payload)
                        valid_rows.append(row)
                    point_ids = valid_ids
                    payloads = valid_payloads
                    matrix = (
                        np.stack(valid_rows)
                        if valid_rows
                        else np.empty((0, self.vector_size), dtype=np.float32)
                    )
            if matrix.shape != (len(payloads), self.vector_size):
                point_ids = []
                payloads = []
                matrix = np.empty((0, self.vector_size), dtype=np.float32)
            # Empty/all-invalid pages still carry the physical source count so
            # cache fingerprints and status snapshots cannot silently undercount.
            yield VectorPointBatch(
                point_ids=tuple(point_ids),
                payloads=tuple(payloads),
                vectors=np.ascontiguousarray(matrix),
                source_point_count=source_point_count,
            )
            if next_offset is None:
                break
            offset = next_offset

    def vectors_for_frames(
        self,
        frame_keys: list[tuple[str, int]] | tuple[tuple[str, int], ...],
        *,
        batch_id: str | None = None,
    ) -> np.ndarray:
        """Fetch vectors by canonical payload identity for Rocchio feedback.

        A missing frame or an identity duplicated across batches is rejected;
        callers may pass ``batch_id`` to disambiguate a multi-batch collection.
        """

        client = self._get_client()
        rows: list[np.ndarray] = []
        for video_id, source_frame_idx in frame_keys:
            filters: dict[str, Any] = {
                "video_id": video_id,
                "source_frame_idx": int(source_frame_idx),
            }
            if batch_id is not None:
                filters["batch_id"] = batch_id
            response = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_filter(filters),
                limit=2,
                with_payload=True,
                with_vectors=True,
            )
            points = (
                response[0]
                if isinstance(response, tuple)
                else getattr(response, "points", response)
            )
            if len(points) != 1:
                reason = "missing" if not points else "ambiguous across batches"
                raise ValueError(f"feedback frame {video_id}:{source_frame_idx} is {reason}")
            vector = getattr(points[0], "vector", None)
            if isinstance(vector, Mapping):
                if len(vector) != 1:
                    raise ValueError("feedback point has ambiguous named vectors")
                vector = next(iter(vector.values()))
            row = np.asarray(vector, dtype=np.float32)
            if row.shape != (self.vector_size,):
                raise ValueError("feedback point has an incompatible vector dimension")
            rows.append(row)
        if not rows:
            return np.empty((0, self.vector_size), dtype=np.float32)
        return np.stack(rows)


class BatchVectorStore:
    """Route one logical channel across batch-qualified Qdrant collections.

    The wrapper intentionally exposes only retrieval and explicit ``for_batch``
    binding.  Mutating ingestion operations must bind a batch first, preventing
    writes to a global or accidentally selected collection.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_template: str,
        vector_size: int,
        distance: str = "Cosine",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        grpc_port: int | None = None,
    ) -> None:
        # Validate the template independently of any real batch.
        batch_collection_name(collection_template, "validation")
        _validate_grpc_transport(prefer_grpc=prefer_grpc, grpc_port=grpc_port)
        self.url = url
        self.collection_template = collection_template
        self.collection_name = collection_template
        self.vector_size = vector_size
        self.distance = distance
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self.grpc_port = grpc_port
        self._client: Any = None
        self._client_lock = RLock()

    def _store(self, collection_name: str) -> VectorStore:
        store = VectorStore(
            url=self.url,
            api_key=self.api_key,
            collection_name=collection_name,
            vector_size=self.vector_size,
            distance=self.distance,
            prefer_grpc=self.prefer_grpc,
            grpc_port=self.grpc_port,
        )
        if self._client is not None:
            store._client = self._client
        return store

    def for_batch(self, batch_id: str) -> VectorStore:
        """Return a normal store bound to one physical batch collection."""

        return self._store(batch_collection_name(self.collection_template, batch_id))

    def _get_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    probe = self._store(batch_collection_name(self.collection_template, "probe"))
                    self._client = probe._get_client()
        return self._client

    def _discover_collection_names(self) -> list[str]:
        response = self._get_client().get_collections()
        collections = (
            response.get("collections", ())
            if isinstance(response, Mapping)
            else getattr(response, "collections", ())
        )
        prefix, suffix = self.collection_template.split(_BATCH_PLACEHOLDER)
        names = []
        for item in collections:
            name = item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)
            if isinstance(name, str) and name.startswith(prefix) and name.endswith(suffix):
                names.append(name)
        return sorted(names)

    @staticmethod
    def _requested_batches(filters: Optional[dict]) -> tuple[str, ...] | None:
        if not filters or "batch_id" not in filters:
            return None
        value = filters["batch_id"]
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(str(item) for item in value)
        return (str(value),)

    def _selected_collection_names(self, filters: Optional[dict]) -> list[str]:
        batches = self._requested_batches(filters)
        if batches is None:
            return self._discover_collection_names()
        return [batch_collection_name(self.collection_template, batch) for batch in batches]

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Search selected batch collections and merge compatible cosine scores."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        client = self._get_client()
        hits: list[dict] = []
        for collection_name in self._selected_collection_names(filters):
            if not client.collection_exists(collection_name=collection_name):
                continue
            hits.extend(self._store(collection_name).search(query_vector, top_k, filters))
        hits.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("id", ""))))
        return hits[:top_k]

    def search_grouped(
        self,
        query_vector: np.ndarray,
        *,
        group_by: str = "video_id",
        top_k_groups: int = 300,
        group_size: int = 1,
        filters: Optional[dict] = None,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Route a grouped search across the selected physical collections."""

        if top_k_groups < 1:
            raise ValueError("top_k_groups must be positive")
        client = self._get_client()
        groups: list[dict[str, Any]] = []
        for collection_name in self._selected_collection_names(filters):
            if not client.collection_exists(collection_name=collection_name):
                continue
            groups.extend(
                self._store(collection_name).search_grouped(
                    query_vector,
                    group_by=group_by,
                    top_k_groups=top_k_groups,
                    group_size=group_size,
                    filters=filters,
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                )
            )
        groups.sort(key=lambda item: (-float(item["score"]), str(item["group_id"])))
        return groups[:top_k_groups]

    def grouped_search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Compatibility alias for :meth:`search_grouped`."""

        return self.search_grouped(*args, **kwargs)

    def retrieve_points_with_vectors(
        self,
        point_ids: list[Any] | tuple[Any, ...],
        *,
        filters: Optional[dict[str, Any]] = None,
        batch_size: int = 4_096,
    ) -> list[dict[str, Any]]:
        """Retrieve private point IDs from the selected batch collections."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not point_ids:
            return []
        client = self._get_client()
        points: list[dict[str, Any]] = []
        for collection_name in self._selected_collection_names(filters):
            if not client.collection_exists(collection_name=collection_name):
                continue
            points.extend(
                self._store(collection_name).retrieve_points_with_vectors(
                    point_ids,
                    filters=filters,
                    batch_size=batch_size,
                )
            )
        return points

    def is_available(self, filters: Optional[dict] = None) -> bool:
        """Return whether a selected compatible collection contains points."""

        client = self._get_client()
        for name in self._selected_collection_names(filters):
            if not client.collection_exists(collection_name=name):
                continue
            if self._store(name).count(filters) > 0:
                return True
        return False

    def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        """Return the exact point count across selected batch collections."""

        client = self._get_client()
        total = 0
        for name in self._selected_collection_names(filters):
            if client.collection_exists(collection_name=name):
                total += self._store(name).count(filters)
        return total

    def vectors_for_frames(
        self,
        frame_keys: list[tuple[str, int]] | tuple[tuple[str, int], ...],
        *,
        batch_id: str | None = None,
    ) -> np.ndarray:
        """Fetch feedback vectors, rejecting missing or cross-batch ambiguity."""

        names = (
            [batch_collection_name(self.collection_template, batch_id)]
            if batch_id is not None
            else self._discover_collection_names()
        )
        matches: list[np.ndarray] = []
        for name in names:
            if not self._get_client().collection_exists(collection_name=name):
                continue
            try:
                matches.append(self._store(name).vectors_for_frames(frame_keys, batch_id=batch_id))
            except ValueError as error:
                if " is missing" not in str(error):
                    raise
        if len(matches) != 1:
            reason = "missing" if not matches else "ambiguous across batches"
            raise ValueError(f"feedback frames are {reason}")
        return matches[0]

    def iter_points_with_vectors(
        self,
        filters: Optional[dict[str, Any]] = None,
        *,
        batch_size: int = 256,
    ) -> Iterator[dict[str, Any]]:
        """Stream vectors from each selected, existing batch collection."""

        client = self._get_client()
        for collection_name in self._selected_collection_names(filters):
            if not client.collection_exists(collection_name=collection_name):
                continue
            yield from self._store(collection_name).iter_points_with_vectors(
                filters,
                batch_size=batch_size,
            )

    def iter_point_batches_with_vectors(
        self,
        filters: Optional[dict[str, Any]] = None,
        *,
        batch_size: int = 16_384,
    ) -> Iterator[VectorPointBatch]:
        """Stream contiguous vector pages from selected batch collections."""

        client = self._get_client()
        for collection_name in self._selected_collection_names(filters):
            if not client.collection_exists(collection_name=collection_name):
                continue
            yield from self._store(collection_name).iter_point_batches_with_vectors(
                filters,
                batch_size=batch_size,
            )
