"""
Vector Store
=============
Qdrant client wrapper for indexing and querying CLIP embeddings.
Handles collection management, batch upsert, and similarity search.

Usage:
    store = VectorStore()
    store.create_collection()
    store.upsert(ids, vectors, payloads)
    results = store.search(query_vector, top_k=100)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class VectorStore:
    """Qdrant vector database client wrapper.

    Args:
        url: Qdrant server URL.
        collection_name: Name of the Qdrant collection.
        vector_size: Dimensionality of stored vectors.
        distance: Distance metric ("Cosine", "Euclid", "Dot").
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "aic2026_keyframes",
        vector_size: int = 1024,
        distance: str = "Cosine",
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self._client = None

    def _connect(self) -> None:
        """Establish connection to Qdrant."""
        # TODO: from qdrant_client import QdrantClient
        #       self._client = QdrantClient(url=self.url)
        raise NotImplementedError("VectorStore._connect")

    def create_collection(self, recreate: bool = False) -> None:
        """Create (or recreate) the Qdrant collection.

        Args:
            recreate: If True, delete existing collection first.
        """
        # TODO: Create collection with vector config
        raise NotImplementedError("VectorStore.create_collection")

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        """Batch upsert vectors with metadata payloads.

        Args:
            ids: Unique point IDs.
            vectors: (N, D) embedding matrix.
            payloads: Metadata dicts (video_id, frame_id, etc.).
            batch_size: Number of points per upsert request.
        """
        # TODO: Batch upsert using qdrant_client.models.PointStruct
        raise NotImplementedError("VectorStore.upsert")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Search for nearest neighbors.

        Args:
            query_vector: (D,) query embedding.
            top_k: Number of results to return.
            filters: Optional Qdrant filter conditions.

        Returns:
            List of dicts: {id, score, payload}.
        """
        # TODO: Run qdrant search, return results
        raise NotImplementedError("VectorStore.search")

    def count(self) -> int:
        """Return the number of points in the collection."""
        # TODO: self._client.count(self.collection_name)
        raise NotImplementedError("VectorStore.count")
