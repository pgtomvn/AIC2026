"""Offline coverage for tokenizer-aware Textual KIS query chunking."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.embedding.text_chunks import chunk_text
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline


class _WordTokenizer:
    def __init__(self) -> None:
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        for word in text.split():
            if word not in self._word_to_id:
                token_id = len(self._word_to_id) + 1
                self._word_to_id[word] = token_id
                self._id_to_word[token_id] = word
            result.append(self._word_to_id[word])
        return result

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return " ".join(self._id_to_word[value] for value in token_ids)


class _DiagnosticEncoder:
    def __init__(self, tokenizer: _WordTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode_text_with_diagnostics(self, text: str):
        chunks = chunk_text(
            self.tokenizer,
            text,
            context_limit=64,
            content_limit=62,
            max_chunks=4,
            overlap=8,
        )
        vectors = np.asarray(
            [[float(index + 1), 1.0] for index in range(len(chunks.texts))],
            dtype=np.float32,
        )
        return vectors, chunks.diagnostics


class _DenseStore:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def search(self, vector, *, top_k: int, filters: Any):
        self.calls.append(np.asarray(vector))
        index = len(self.calls)
        return [
            {
                "batch_id": "b1",
                "video_id": f"L21_V{index:03d}",
                "source_frame_idx": index,
                "timestamp_ms": index * 1000,
                "shot_id": f"shot-{index}",
                "score": 1.0,
            }
        ]


class _LexicalStore:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int, filters: Any):
        self.queries.append(query)
        return []


def _words(count: int) -> str:
    return " ".join(f"token-{index}" for index in range(count))


def test_short_query_is_one_chunk_and_keeps_the_existing_dense_path() -> None:
    tokenizer = _WordTokenizer()
    store = _DenseStore()
    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", _DiagnosticEncoder(tokenizer), store),),
        temporal_windows_ms=(),
    )

    pipeline.search("một người áo đỏ", refine_raw=False)

    assert len(store.calls) == 1
    status = pipeline.last_channel_statuses["visual"]
    assert status["chunk_count"] == 1
    assert status["query_chunked"] is False
    assert status["dropped_tokens"] == 0


def test_145_word_query_is_chunked_without_silent_token_loss() -> None:
    chunks = chunk_text(
        _WordTokenizer(),
        _words(145),
        context_limit=64,
        content_limit=62,
        max_chunks=4,
        overlap=8,
    )

    assert len(chunks.texts) == 3
    assert chunks.diagnostics.token_count == 145
    assert chunks.diagnostics.processed_tokens == 145
    assert chunks.diagnostics.dropped_tokens == 0
    assert chunks.diagnostics.query_chunked is True
    assert chunks.diagnostics.query_truncated is False


def test_query_beyond_four_chunks_reports_exact_loss_and_keeps_both_ends() -> None:
    tokenizer = _WordTokenizer()
    chunks = chunk_text(
        tokenizer,
        _words(400),
        context_limit=64,
        content_limit=62,
        max_chunks=4,
        overlap=8,
    )

    assert len(chunks.texts) == 4
    assert chunks.texts[0].startswith("token-0 ")
    assert chunks.texts[-1].endswith(" token-399")
    assert chunks.diagnostics.dropped_tokens > 0
    assert (
        chunks.diagnostics.processed_tokens + chunks.diagnostics.dropped_tokens
        == chunks.diagnostics.token_count
    )
    assert chunks.diagnostics.query_truncated is True


def test_dense_chunks_fuse_inside_one_channel_and_lexical_gets_full_query() -> None:
    tokenizer = _WordTokenizer()
    dense = _DenseStore()
    lexical = _LexicalStore()
    query = _words(145)
    pipeline = TextualKISPipeline(
        dense_channels=(DenseChannel("visual", _DiagnosticEncoder(tokenizer), dense),),
        lexical_channels=(LexicalChannel("ocr", lexical),),
        temporal_windows_ms=(),
    )

    hits = pipeline.search(query, refine_raw=False)

    assert len(dense.calls) == 3
    assert lexical.queries == [query]
    assert len(hits) == 3
    assert pipeline.last_channel_statuses["visual"]["chunk_count"] == 3
