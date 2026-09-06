"""Lazy Meilisearch adapter for lexical OCR, ASR, and metadata retrieval.

Meilisearch has its own lexical ranking rules; it is not a BM25 service.
Importing or constructing :class:`TextStore` does not contact the service.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5

_SEARCHABLE_ATTRIBUTES = [
    "text",
    "text_raw",
    "text_nfc",
    "text_accent_folded",
    "ocr_text",
    "asr_text",
    "description",
    "objects",
]

_FILTERABLE_ATTRIBUTES = [
    "id",
    "batch_id",
    "video_id",
    "kind",
    "source_frame_idx",
    "shot_id",
    "start_ms",
    "end_ms",
]

_TASK_TIMEOUT_MS = 120_000
_TASK_POLL_INTERVAL_MS = 500
_MEILI_ID = re.compile(r"[A-Za-z0-9_-]{1,511}\Z")

VIETNAMESE_STOP_WORDS = (
    "một", "những", "các", "đang", "được", "ở", "tại", "trong", "trên",
    "với", "và", "của", "là", "có",
)

BILINGUAL_METADATA_SYNONYMS = {
    "cảnh sát": ("police", "police officer"),
    "police": ("cảnh sát", "police officer"),
    "xe ô tô": ("car", "automobile", "xe hơi"),
    "xe hơi": ("car", "automobile", "xe ô tô"),
    "car": ("xe ô tô", "xe hơi", "automobile"),
    "xe cấp cứu": ("ambulance",),
    "ambulance": ("xe cấp cứu",),
    "xe máy": ("motorcycle", "motorbike"),
    "motorcycle": ("xe máy", "motorbike"),
    "điện thoại": ("phone", "mobile phone"),
    "phone": ("điện thoại", "mobile phone"),
    "kính mắt": ("glasses", "eyeglasses"),
    "glasses": ("kính mắt", "eyeglasses"),
}


class TextStore:
    """Meilisearch lexical-search client wrapper.

    Args:
        url: Meilisearch server URL.
        api_key: API key, or ``None`` for an unsecured local service.
        index_name: Name of the Meilisearch index.
    """

    def __init__(
        self,
        url: str = "http://localhost:7700",
        api_key: Optional[str] = None,
        index_name: str = "aic2026_text",
    ) -> None:
        if not url:
            raise ValueError("url must not be empty")
        if not index_name:
            raise ValueError("index_name must not be empty")

        self.url = url
        self.api_key = api_key
        self.index_name = index_name
        self._client: Any = None
        self._index: Any = None
        self._primary_key = "id"

    def _connect(self) -> None:
        """Create the Meilisearch client without issuing a service request."""
        if self._client is not None:
            return
        try:
            import meilisearch
        except ImportError as exc:  # pragma: no cover - dependency is part of the lockfile
            raise RuntimeError(
                "Meilisearch support is unavailable; install the locked project dependencies"
            ) from exc
        self._client = meilisearch.Client(self.url, self.api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            self._connect()
        return self._client

    @staticmethod
    def _task_value(task: Any, *names: str) -> Any:
        if isinstance(task, Mapping):
            for name in names:
                if name in task:
                    return task[name]
        for name in names:
            if hasattr(task, name):
                return getattr(task, name)
        return None

    def _wait_for_task(self, task_info: Any, operation: str) -> Any:
        task_uid = self._task_value(task_info, "task_uid", "taskUid", "uid")
        if task_uid is None:
            raise RuntimeError(f"Meilisearch {operation} did not return a task UID")

        task = self._get_client().wait_for_task(
            int(task_uid),
            timeout_in_ms=_TASK_TIMEOUT_MS,
            interval_in_ms=_TASK_POLL_INTERVAL_MS,
        )
        status = self._task_value(task, "status")
        if status != "succeeded":
            error = self._task_value(task, "error")
            raise RuntimeError(
                f"Meilisearch {operation} task {task_uid} ended with status {status!r}: {error!r}"
            )
        return task

    @staticmethod
    def _is_index_not_found(error: Exception) -> bool:
        return (
            getattr(error, "code", None) == "index_not_found"
            or getattr(error, "status_code", None) == 404
        )

    def _fetch_index(self) -> Any:
        if self._index is not None:
            return self._index
        self._index = self._get_client().get_index(self.index_name)
        return self._index

    def create_index(self, primary_key: str = "id") -> None:
        """Idempotently create and configure the lexical-search index."""
        if not isinstance(primary_key, str) or not primary_key:
            raise ValueError("primary_key must be a non-empty string")

        client = self._get_client()
        try:
            index = client.get_index(self.index_name)
        except Exception as exc:
            if not self._is_index_not_found(exc):
                raise
            task_info = client.create_index(
                self.index_name,
                {"primaryKey": primary_key},
            )
            self._wait_for_task(task_info, f"create index {self.index_name!r}")
            index = client.get_index(self.index_name)

        actual_primary_key = getattr(index, "primary_key", None)
        if actual_primary_key is not None and actual_primary_key != primary_key:
            raise ValueError(
                f"index {self.index_name!r} uses primary key {actual_primary_key!r}; "
                f"expected {primary_key!r}"
            )

        self._index = index
        self._primary_key = primary_key
        filterable_attributes = list(dict.fromkeys([primary_key, *_FILTERABLE_ATTRIBUTES]))
        settings = index.get_settings()
        if isinstance(settings, Mapping):
            current_searchable = settings.get(
                "searchableAttributes",
                settings.get("searchable_attributes"),
            )
            current_filterable = settings.get(
                "filterableAttributes",
                settings.get("filterable_attributes"),
            )
        else:
            current_searchable = getattr(settings, "searchable_attributes", None)
            current_filterable = getattr(settings, "filterable_attributes", None)

        if current_searchable != _SEARCHABLE_ATTRIBUTES:
            searchable_task = index.update_searchable_attributes(_SEARCHABLE_ATTRIBUTES)
            self._wait_for_task(searchable_task, "update searchable attributes")
        if current_filterable != filterable_attributes:
            filterable_task = index.update_filterable_attributes(filterable_attributes)
            self._wait_for_task(filterable_task, "update filterable attributes")

    def configure_shadow_v2(self, profile: str) -> None:
        """Apply versioned multilingual settings to a non-live v2 index.

        The method refuses v1 names so an operator cannot accidentally trigger
        a live reindex while preparing the shadow deployment.
        """

        if not self.index_name.endswith("_v2"):
            raise ValueError("multilingual settings may be applied only to a shadow v2 index")
        if profile not in {"ocr", "asr", "metadata"}:
            raise ValueError("shadow profile must be ocr, asr, or metadata")
        index = self._fetch_index()
        localized = [
            {
                "attributePatterns": [
                    "text", "text_raw", "text_nfc", "text_accent_folded",
                    "ocr_text" if profile == "ocr" else "asr_text",
                ],
                "locales": ["vie"],
            }
        ]
        settings: dict[str, Any] = {"localizedAttributes": localized}
        if profile in {"ocr", "asr"}:
            settings["stopWords"] = list(VIETNAMESE_STOP_WORDS)
        else:
            settings["synonyms"] = {
                key: list(values) for key, values in BILINGUAL_METADATA_SYNONYMS.items()
            }
            settings["localizedAttributes"] = [
                {
                    "attributePatterns": ["text", "text_raw", "text_nfc"],
                    "locales": ["vie", "eng"],
                },
                {
                    "attributePatterns": ["text_accent_folded", "objects"],
                    "locales": ["eng"],
                },
            ]
        task_info = index.update_settings(settings)
        self._wait_for_task(task_info, f"configure multilingual shadow {profile!r}")

    def add_documents(self, documents: list[dict[str, Any]], batch_size: int = 1000) -> None:
        """Validate and add documents in acknowledged batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(not isinstance(document, dict) for document in documents):
            raise TypeError("every document must be a dictionary")

        document_ids = []
        for document in documents:
            if self._primary_key not in document:
                raise ValueError(f"every document must contain primary key {self._primary_key!r}")
            document_id = document[self._primary_key]
            if not isinstance(document_id, (str, int)) or isinstance(document_id, bool):
                raise TypeError("document IDs must be strings or integers")
            if isinstance(document_id, str) and not document_id:
                raise ValueError("document IDs must not be empty")
            document_ids.append(document_id)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document IDs must be unique within an add call")
        if not documents:
            return

        prepared_documents: list[dict[str, Any]] = []
        prepared_ids: list[str | int] = []
        for document, document_id in zip(documents, document_ids, strict=True):
            prepared = dict(document)
            if isinstance(document_id, str) and not _MEILI_ID.fullmatch(document_id):
                prepared["external_id"] = document_id
                prepared[self._primary_key] = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"aic2026:meilisearch:{self.index_name}:{document_id}",
                    )
                )
            prepared_documents.append(prepared)
            prepared_ids.append(prepared[self._primary_key])
        if len(set(prepared_ids)) != len(prepared_ids):
            raise ValueError("normalized document IDs must be unique within an add call")

        index = self._fetch_index()
        for start in range(0, len(prepared_documents), batch_size):
            stop = min(start + batch_size, len(prepared_documents))
            task_info = index.add_documents(prepared_documents[start:stop])
            self._wait_for_task(task_info, f"add documents {start}:{stop}")

    def search(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[str] = None,
        locales: tuple[str, ...] | list[str] | None = None,
    ) -> list[dict]:
        """Run lexical search and return canonical ranked result mappings.

        Each result contains a one-based ``rank``, string ``id``, Meilisearch
        ``score`` (when supplied by the server), and the original document in
        ``payload``.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if filters is not None and not isinstance(filters, str):
            raise TypeError("filters must be a Meilisearch filter expression string")
        if locales is not None:
            if isinstance(locales, (str, bytes)) or any(
                not isinstance(value, str) or not value for value in locales
            ):
                raise TypeError("locales must contain non-empty locale identifiers")

        options: dict[str, Any] = {
            "limit": top_k,
            "showRankingScore": True,
        }
        if filters:
            options["filter"] = filters
        if locales:
            options["locales"] = list(locales)
        response = self._fetch_index().search(query, options)
        if not isinstance(response, Mapping):
            raise RuntimeError("Meilisearch returned a malformed search response")
        hits = response.get("hits", [])
        if not isinstance(hits, list):
            raise RuntimeError("Meilisearch returned a malformed search response")

        return self._results_from_hits(hits)

    def _results_from_hits(self, hits: Sequence[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rank, raw_hit in enumerate(hits, start=1):
            if not isinstance(raw_hit, Mapping):
                raise RuntimeError("Meilisearch returned a non-object search hit")
            payload = dict(raw_hit)
            score = payload.pop("_rankingScore", None)
            document_id = payload.get(self._primary_key)
            if document_id is None:
                raise RuntimeError(f"Meilisearch hit is missing primary key {self._primary_key!r}")
            results.append(
                {
                    "rank": rank,
                    "id": str(payload.get("external_id", document_id)),
                    "score": float(score) if score is not None else None,
                    "payload": payload,
                }
            )
        return results

    def search_many(
        self,
        queries: Sequence[tuple[str, tuple[str, ...] | None]],
        *,
        top_k: int = 50,
        filters: Optional[str] = None,
    ) -> list[list[dict[str, Any]]]:
        """Execute bounded lexical variants in one Meilisearch HTTP request."""

        if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
            raise TypeError("queries must be a sequence")
        if not 1 <= len(queries) <= 128:
            raise ValueError("queries must contain between 1 and 128 variants")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if filters is not None and not isinstance(filters, str):
            raise TypeError("filters must be a Meilisearch filter expression string")
        payload: list[dict[str, Any]] = []
        for query, locales in queries:
            if not isinstance(query, str):
                raise TypeError("every query must be a string")
            item: dict[str, Any] = {
                "indexUid": self.index_name,
                "q": query,
                "limit": top_k,
                "showRankingScore": True,
            }
            if filters:
                item["filter"] = filters
            if locales:
                item["locales"] = list(locales)
            payload.append(item)
        client = self._get_client()
        multi_search = getattr(client, "multi_search", None)
        if not callable(multi_search):
            return [
                self.search(query, top_k=top_k, filters=filters, locales=locales)
                for query, locales in queries
            ]
        response = multi_search(payload)
        raw_results = response.get("results") if isinstance(response, Mapping) else getattr(response, "results", None)
        if not isinstance(raw_results, (list, tuple)) or len(raw_results) != len(queries):
            raise RuntimeError("Meilisearch returned a malformed multi-search response")
        values: list[list[dict[str, Any]]] = []
        for raw_result in raw_results:
            hits = raw_result.get("hits") if isinstance(raw_result, Mapping) else getattr(raw_result, "hits", None)
            if not isinstance(hits, (list, tuple)):
                raise RuntimeError("Meilisearch returned malformed multi-search hits")
            values.append(self._results_from_hits(hits))
        return values

    def is_available(self, filters: Optional[str] = None) -> bool:
        """Return whether the index has at least one selected document.

        The lightweight probe prevents empty OCR/ASR indexes from appearing as
        runtime failures and avoids issuing the full query when a selected
        batch has no lexical coverage.
        """

        if filters is not None and not isinstance(filters, str):
            raise TypeError("filters must be a Meilisearch filter expression string")
        options: dict[str, Any] = {"limit": 1}
        if filters:
            options["filter"] = filters
        response = self._fetch_index().search("", options)
        if not isinstance(response, Mapping):
            raise RuntimeError("Meilisearch returned a malformed availability response")
        hits = response.get("hits", ())
        if not isinstance(hits, list):
            raise RuntimeError("Meilisearch returned malformed availability hits")
        return bool(hits)

    def count_documents(self, filters: Optional[str] = None) -> int:
        """Return an exact service-reported count, optionally scoped by filter."""

        if filters is not None and not isinstance(filters, str):
            raise TypeError("filters must be a Meilisearch filter expression string")
        index = self._fetch_index()
        if not filters:
            stats = index.get_stats()
            if isinstance(stats, Mapping):
                value = stats.get("numberOfDocuments", stats.get("number_of_documents"))
            else:
                value = getattr(stats, "number_of_documents", None)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError("Meilisearch stats are missing a valid document count")
            return value
        # ``estimatedTotalHits`` is capped by Meilisearch's pagination
        # ``maxTotalHits`` setting (1000 by default), so it is not an exact
        # import/readiness contract.  ``batch_id`` is filterable and therefore
        # facetable; its distribution reports the exact filtered population.
        options: dict[str, Any] = {"limit": 0, "facets": ["batch_id"]}
        if filters:
            options["filter"] = filters
        response = index.search("", options)
        if not isinstance(response, Mapping):
            raise RuntimeError("Meilisearch returned a malformed count response")
        distribution = response.get(
            "facetDistribution",
            response.get("facet_distribution"),
        )
        if isinstance(distribution, Mapping):
            batch_counts = distribution.get("batch_id")
            if isinstance(batch_counts, Mapping):
                values = tuple(batch_counts.values())
                if all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in values
                ):
                    return sum(values)
        value = response.get("estimatedTotalHits", response.get("totalHits"))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("Meilisearch count response is missing a valid total")
        return value

    def iter_documents(
        self,
        *,
        filters: Optional[str] = None,
        batch_size: int = 1_000,
    ):
        """Yield stored documents in bounded pages for an explicit shadow copy."""

        if filters is not None and not isinstance(filters, str):
            raise TypeError("filters must be a Meilisearch filter expression string")
        if batch_size < 1 or batch_size > 10_000:
            raise ValueError("batch_size must be between 1 and 10000")
        index = self._fetch_index()
        offset = 0
        while True:
            options: dict[str, Any] = {"offset": offset, "limit": batch_size}
            if filters:
                options["filter"] = filters
            response = index.get_documents(options)
            values = (
                response.get("results", ())
                if isinstance(response, Mapping)
                else getattr(response, "results", ())
            )
            if not isinstance(values, (list, tuple)):
                raise RuntimeError("Meilisearch returned malformed documents")
            for value in values:
                if isinstance(value, Mapping):
                    yield dict(value)
                elif hasattr(value, "__dict__"):
                    yield dict(vars(value))
                else:
                    raise RuntimeError("Meilisearch returned a non-object document")
            if len(values) < batch_size:
                break
            offset += len(values)
