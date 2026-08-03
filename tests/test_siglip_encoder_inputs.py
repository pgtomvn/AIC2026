"""CPU-only input-boundary tests for the lazy SigLIP2 adapter."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from src.embedding.siglip_encoder import SigLIP2Encoder


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_siglip_accepts_bytes_and_owned_pil_images_without_loading_paths() -> None:
    encoder = SigLIP2Encoder(device="cpu", batch_size=2, max_batch_size=2)
    encoder._model = object()
    encoder._encode_image_batch = lambda images: np.asarray(  # type: ignore[method-assign]
        [[float(image.getpixel((0, 0))[0]), 1.0] for image in images],
        dtype=np.float32,
    )
    pil = Image.new("RGB", (4, 3), (10, 0, 0))

    vectors = encoder.encode_images([_png_bytes((20, 0, 0)), pil])

    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6)
    assert pil.getpixel((0, 0)) == (10, 0, 0)
