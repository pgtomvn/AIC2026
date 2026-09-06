"""Tokenizer-aware query chunking shared by compatible dense encoders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts import TextEncodingDiagnostics


@dataclass(frozen=True, slots=True)
class TextChunks:
    """Selected text chunks and their exact tokenizer accounting."""

    texts: tuple[str, ...]
    diagnostics: TextEncodingDiagnostics


def _tokenizer_backend(tokenizer: Any) -> Any:
    backend = getattr(tokenizer, "tokenizer", None)
    return backend if backend is not None else tokenizer


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    backend = _tokenizer_backend(tokenizer)
    if callable(backend):
        try:
            encoded = backend(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )
        except TypeError:
            encoded = None
        if isinstance(encoded, dict):
            values = encoded.get("input_ids")
            if isinstance(values, Sequence) and values and isinstance(values[0], Sequence):
                values = values[0]
            if isinstance(values, Sequence):
                return [int(value) for value in values]
    encode = getattr(backend, "encode", None)
    if callable(encode):
        try:
            values = encode(text, add_special_tokens=False)
        except TypeError:
            values = encode(text)
        return [int(value) for value in values]
    raise RuntimeError("encoder tokenizer does not expose lossless token IDs")


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    backend = _tokenizer_backend(tokenizer)
    decode = getattr(backend, "decode", None)
    if not callable(decode):
        raise RuntimeError("encoder tokenizer does not expose token decoding")
    try:
        value = decode(list(token_ids), skip_special_tokens=True)
    except TypeError:
        value = decode(list(token_ids))
    text = str(value).strip()
    if not text:
        raise RuntimeError("encoder tokenizer produced an empty query chunk")
    return text


def _all_spans(
    *,
    token_count: int,
    content_limit: int,
    overlap: int,
) -> list[tuple[int, int]]:
    stride = content_limit - overlap
    starts = list(range(0, token_count, stride))
    last_start = max(0, token_count - content_limit)
    if starts and starts[-1] + content_limit < token_count:
        starts.append(last_start)
    elif starts:
        starts[-1] = last_start
    return [(start, min(token_count, start + content_limit)) for start in sorted(set(starts))]


def _select_spans(spans: Sequence[tuple[int, int]], maximum: int) -> list[tuple[int, int]]:
    if len(spans) <= maximum:
        return list(spans)
    if maximum == 1:
        return [spans[0]]
    indices = {round(position * (len(spans) - 1) / (maximum - 1)) for position in range(maximum)}
    if len(indices) < maximum:
        indices.update(index for index in range(len(spans)) if len(indices) < maximum)
    return [spans[index] for index in sorted(indices)[:maximum]]


def chunk_text(
    tokenizer: Any,
    text: str,
    *,
    context_limit: int,
    content_limit: int,
    max_chunks: int = 4,
    overlap: int = 8,
) -> TextChunks:
    """Split a query by actual encoder tokens and retain even whole-query coverage."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if context_limit < 1 or content_limit < 1 or content_limit > context_limit:
        raise ValueError("invalid text context limits")
    if max_chunks < 1 or overlap < 0 or overlap >= content_limit:
        raise ValueError("invalid text chunking limits")
    token_ids = _token_ids(tokenizer, text)
    if not token_ids:
        raise ValueError("text tokenizer produced no content tokens")
    if len(token_ids) <= content_limit:
        return TextChunks(
            texts=(text,),
            diagnostics=TextEncodingDiagnostics(
                token_count=len(token_ids),
                context_limit=context_limit,
                content_limit=content_limit,
                chunk_count=1,
                processed_tokens=len(token_ids),
                dropped_tokens=0,
                query_chunked=False,
                query_truncated=False,
            ),
        )
    all_spans = _all_spans(
        token_count=len(token_ids),
        content_limit=content_limit,
        overlap=overlap,
    )
    capacity = max_chunks * content_limit - (max_chunks - 1) * overlap
    if len(token_ids) <= capacity:
        selected = all_spans
    elif max_chunks == 1:
        selected = [(0, content_limit)]
    else:
        last_start = len(token_ids) - content_limit
        starts = {round(index * last_start / (max_chunks - 1)) for index in range(max_chunks)}
        selected = [(start, min(len(token_ids), start + content_limit)) for start in sorted(starts)]
    covered = set()
    for start, end in selected:
        covered.update(range(start, end))
    dropped = len(token_ids) - len(covered)
    return TextChunks(
        texts=tuple(_decode(tokenizer, token_ids[start:end]) for start, end in selected),
        diagnostics=TextEncodingDiagnostics(
            token_count=len(token_ids),
            context_limit=context_limit,
            content_limit=content_limit,
            chunk_count=len(selected),
            processed_tokens=len(covered),
            dropped_tokens=dropped,
            query_chunked=len(all_spans) > 1,
            query_truncated=dropped > 0,
        ),
    )


def normalized_mean(vectors: np.ndarray) -> np.ndarray:
    """Return a normalized mean for one or more already-compatible vectors."""

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("query vectors must be a non-empty matrix")
    mean = values.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("query vectors have an invalid mean")
    return mean / norm


def encode_query_mean(encoder: Any, text: str) -> np.ndarray:
    """Encode a query and average chunks when diagnostics are supported."""

    diagnostic_encoder: Callable[[str], tuple[np.ndarray, Any]] | None = getattr(
        encoder,
        "encode_text_with_diagnostics",
        None,
    )
    vectors = (
        diagnostic_encoder(text)[0] if callable(diagnostic_encoder) else encoder.encode_text(text)
    )
    return normalized_mean(np.asarray(vectors, dtype=np.float32))


__all__ = ["TextChunks", "chunk_text", "encode_query_mean", "normalized_mean"]
