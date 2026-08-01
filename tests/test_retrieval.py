"""
Tests for Retrieval Pipelines
=============================
"""

from __future__ import annotations

import pytest


class TestTextualKISPipeline:
    """Test suite for the Textual KIS retrieval pipeline."""

    def test_pipeline_init(self):
        """Test that the pipeline can be instantiated."""
        from src.retrieval.textual_kis import TextualKISPipeline

        pipeline = TextualKISPipeline(
            vector_weight=0.6,
            bm25_weight=0.4,
            top_k=50,
        )
        assert pipeline.vector_weight == 0.6
        assert pipeline.bm25_weight == 0.4
        assert pipeline.top_k == 50


class TestVisualQAPipeline:
    """Test suite for the Visual Q&A pipeline."""

    def test_pipeline_init(self):
        """Test that the pipeline can be instantiated."""
        from src.retrieval.visual_qa import VisualQAPipeline

        pipeline = VisualQAPipeline(top_k_retrieve=10, top_k_final=50)
        assert pipeline.top_k_retrieve == 10
        assert pipeline.top_k_final == 50


class TestTRAKEPipeline:
    """Test suite for the TRAKE temporal alignment pipeline."""

    def test_pipeline_init(self):
        """Test that the pipeline can be instantiated."""
        from src.retrieval.trake import TRAKEPipeline

        pipeline = TRAKEPipeline(top_k_videos=5)
        assert pipeline.top_k_videos == 5

    # TODO: Add tests for DTW alignment logic once implemented
