"""
Tests for CLIPEncoder
=====================
"""

from __future__ import annotations

import numpy as np
import pytest


class TestCLIPEncoder:
    """Test suite for the CLIP encoder module."""

    def test_encoder_init(self):
        """Test that CLIPEncoder can be instantiated with default params."""
        from src.embedding.clip_encoder import CLIPEncoder

        encoder = CLIPEncoder(device="cpu")
        assert encoder.model_name == "ViT-H-14"
        assert encoder.device == "cpu"
        assert encoder.batch_size == 64

    def test_compute_similarity(self):
        """Test cosine similarity computation with normalized vectors."""
        from src.embedding.clip_encoder import CLIPEncoder

        encoder = CLIPEncoder(device="cpu")

        # Create simple normalized vectors
        query = np.array([[1.0, 0.0, 0.0]])
        images = np.array([
            [1.0, 0.0, 0.0],   # identical → sim = 1.0
            [0.0, 1.0, 0.0],   # orthogonal → sim = 0.0
            [0.707, 0.707, 0.0],  # 45° → sim ≈ 0.707
        ])

        sims = encoder.compute_similarity(query, images)
        assert sims.shape == (3,)
        assert abs(sims[0] - 1.0) < 1e-6
        assert abs(sims[1] - 0.0) < 1e-6
        assert abs(sims[2] - 0.707) < 1e-3

    # TODO: Add integration tests for encode_images/encode_text
    #       once the model loading is implemented
