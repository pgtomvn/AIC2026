"""Token limits must respect the model position table, not only the tokenizer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embedding.text_encoder import VietnameseTextEncoder


@pytest.mark.parametrize("model_type,expected", [("roberta", 256), ("xlm-roberta", 256), ("bert", 258)])
def test_input_limit_respects_position_table(model_type: str, expected: int) -> None:
    encoder = VietnameseTextEncoder()
    encoder._model = SimpleNamespace(config=SimpleNamespace(
        max_position_embeddings=258, model_type=model_type, pad_token_id=1))
    encoder._tokenizer = SimpleNamespace(model_max_length=512)
    assert encoder._input_limit() == expected


def test_input_limit_keeps_a_smaller_tokenizer_limit() -> None:
    encoder = VietnameseTextEncoder()
    encoder._model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=514, model_type="roberta"))
    encoder._tokenizer = SimpleNamespace(model_max_length=128)
    assert encoder._input_limit() == 128
