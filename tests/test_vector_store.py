"""
Tests for VectorStore
=====================
"""

from __future__ import annotations

import pytest


class TestVectorStore:
    """Test suite for the Qdrant vector store wrapper."""

    def test_store_init(self):
        """Test that VectorStore can be instantiated with default params."""
        from src.indexing.vector_store import VectorStore

        store = VectorStore()
        assert store.url == "http://localhost:6333"
        assert store.collection_name == "aic2026_keyframes"
        assert store.vector_size == 1024
        assert store.distance == "Cosine"

    def test_store_custom_params(self):
        """Test VectorStore with custom parameters."""
        from src.indexing.vector_store import VectorStore

        store = VectorStore(
            url="http://custom:6333",
            collection_name="test_collection",
            vector_size=512,
            distance="Euclid",
        )
        assert store.url == "http://custom:6333"
        assert store.collection_name == "test_collection"
        assert store.vector_size == 512

    # TODO: Add integration tests with a running Qdrant instance
    #       Use @pytest.mark.integration to gate these tests
