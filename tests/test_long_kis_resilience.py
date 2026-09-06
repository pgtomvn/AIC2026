"""Regression tests for long-query construction and recoverable failures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.components.query_input import _comma_values
from src.embedding.cuda_health import CUDAEncoderRestartRequired
from src.embedding.siglip_encoder import SigLIP2Encoder
from src.embedding.text_chunks import chunk_text
from src.query.translator import (
    OfflineVietnameseTranslator,
    normalize_query_text,
    split_translation_text,
)
from src.retrieval.query_plan import build_query_plan
from src.retrieval.textual_kis import DenseChannel, TextualKISPipeline

P1_12 = (
    "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng nằm trên một khay gỗ hình chữ nhật. "
    "Bên cạnh chiếc đĩa sứ là một chén đựng một vài trái dâu, nhưng có 2 trái bị rơi ra ngoài. Ngoài ra, bên cạnh đĩa sứ còn có một chén sứ nhỏ "
    "màu trắng đựng chuối đã được cắt sẵn và một cái thìa nhỏ màu nâu. Phân cảnh tiếp theo cho thấy đầu bếp đặt 2 chiếc bánh rán lên đĩa sứ "
    "và bắt đầu trang trí. Bước đầu tiên là việc rưới chocolate lên trên mặt bánh. Sau đó, đầu bếp đặt các lát chuối lên trên một chiếc bánh rán, "
    "chiếc còn lại được đặt các lát dâu tây lên."
)


class _WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(value.split("-")[-1]) for value in text.split()]

    def decode(self, values: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"token-{value}" for value in values)


def test_long_planner_and_editor_hints_are_stably_unique() -> None:
    plan = build_query_plan(P1_12)

    for values in (
        plan.quoted_text,
        plan.named_entities,
        plan.objects,
        plan.actions,
        plan.attributes,
    ):
        assert len(values) == len({value.casefold() for value in values})
    assert _comma_values("trắng, nhỏ, TRẮNG, nhỏ") == ("trắng", "nhỏ")


def test_oversized_dense_query_uses_four_full_windows_and_both_ends() -> None:
    chunks = chunk_text(
        _WordTokenizer(),
        " ".join(f"token-{index}" for index in range(520)),
        context_limit=77,
        content_limit=75,
        max_chunks=4,
        overlap=8,
    )

    assert len(chunks.texts) == 4
    assert all(len(value.split()) == 75 for value in chunks.texts)
    assert chunks.texts[0].startswith("token-0 ")
    assert chunks.texts[-1].endswith(" token-519")
    assert chunks.diagnostics.processed_tokens == 300
    assert chunks.diagnostics.dropped_tokens == 220


def test_query_sanitization_and_sentence_bounds_preserve_searchable_content() -> None:
    text = "\x00Cảnh quay!!!\u202e\nMột người đang chạy.... " + ("dứa " * 120)
    normalized = normalize_query_text(text)
    chunks = split_translation_text(normalized, maximum_chars=400)

    assert "\x00" not in normalized and "\u202e" not in normalized
    assert "!!!" not in normalized and "...." not in normalized
    assert chunks and all(0 < len(value) <= 400 for value in chunks)
    assert "dứa" in " ".join(chunks)


def test_translator_timeout_is_sanitized_and_releases_singleflight_lock(
    tmp_path: Path,
) -> None:
    translator = OfflineVietnameseTranslator(
        model_name="fixture",
        revision="rev",
        model_path=tmp_path / "model",
        tokenizer_path=tmp_path / "tokenizer",
        request_timeout_seconds=0.01,
        startup_timeout_seconds=0.01,
    )

    def timeout(_text: str) -> tuple[str, str]:
        raise TimeoutError

    translator._translate_supervised = timeout  # type: ignore[method-assign]
    result = translator.translate("một người đang chạy")

    assert result.english_text is None
    assert result.failure_code == "translator_timeout"
    assert translator._key_locks == {}


def test_fatal_cuda_error_opens_encoder_circuit_until_restart() -> None:
    encoder = SigLIP2Encoder(device="cuda")
    calls = 0

    def fail(_text):
        nonlocal calls
        calls += 1
        raise RuntimeError("CUDA error: device-side assert triggered")

    encoder._encode_text_unlocked = fail  # type: ignore[method-assign]
    for _ in range(2):
        try:
            encoder.encode_text("query")
        except CUDAEncoderRestartRequired as error:
            assert error.restart_required is True
        else:
            raise AssertionError("fatal CUDA failure was not isolated")
    assert calls == 1


class _Encoder:
    def encode_text(self, _text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _Store:
    def search(self, _vector, **_kwargs):
        return [
            {
                "batch_id": "b1",
                "video_id": "L21_V001",
                "source_frame_idx": 10,
                "timestamp_ms": 100,
                "shot_id": "s1",
                "score": 0.9,
            }
        ]


def test_raw_refiner_failure_keeps_coarse_results() -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", _Encoder(), _Store()),),
        raw_refiner=fail,
        temporal_windows_ms=(),
    )

    hits = pipeline.search("a person", refine_raw=True)

    assert hits[0]["source_frame_idx"] == 10
    assert pipeline.last_channel_statuses["raw_refinement"]["state"] == "service_unavailable"
