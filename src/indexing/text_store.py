"""
Text Store
==========
Meilisearch client wrapper for full-text BM25 search over
OCR-extracted text and Whisper ASR transcripts.

Usage:
    store = TextStore()
    store.create_index()
    store.add_documents(docs)
    results = store.search("người đi xe đạp", top_k=50)
"""

from __future__ import annotations

from typing import Any, Optional


class TextStore:
    """Meilisearch full-text search client wrapper.

    Args:
        url: Meilisearch server URL.
        api_key: Master API key.
        index_name: Name of the Meilisearch index.
    """

    def __init__(
        self,
        url: str = "http://localhost:7700",
        api_key: Optional[str] = None,
        index_name: str = "aic2026_text",
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.index_name = index_name
        self._client = None

    def _connect(self) -> None:
        """Establish connection to Meilisearch."""
        # TODO: import meilisearch
        #       self._client = meilisearch.Client(self.url, self.api_key)
        raise NotImplementedError("TextStore._connect")

    def create_index(self, primary_key: str = "id") -> None:
        """Create the search index.

        Args:
            primary_key: Field to use as the document primary key.
        """
        # TODO: Create index, configure searchable/filterable attributes
        raise NotImplementedError("TextStore.create_index")

    def add_documents(
        self, documents: list[dict[str, Any]], batch_size: int = 1000
    ) -> None:
        """Add documents to the index in batches.

        Args:
            documents: List of dicts with at minimum 'id' and text fields.
            batch_size: Documents per batch.
        """
        # TODO: Batch add documents
        raise NotImplementedError("TextStore.add_documents")

    def search(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[str] = None,
    ) -> list[dict]:
        """Full-text search with BM25 ranking.

        Args:
            query: Search query string.
            top_k: Maximum number of results.
            filters: Optional Meilisearch filter expression.

        Returns:
            List of matching documents with relevance scores.
        """
        # TODO: Execute search, return hits
        raise NotImplementedError("TextStore.search")
