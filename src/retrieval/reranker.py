"""Lightweight deterministic reranking and diversity helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class Reranker:
    """Optional scoring callback plus a CPU Maximal Marginal Relevance path."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        scorer: Callable[[str, list[dict[str, Any]]], list[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.scorer = scorer
        self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Rerank with an injected scorer or preserve deterministic fused order."""

        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        output = [dict(candidate) for candidate in candidates]
        if self.scorer is not None and output:
            scores = self.scorer(query, output)
            if len(scores) != len(output):
                raise ValueError("reranker scorer returned the wrong number of scores")
            for candidate, rerank_score in zip(output, scores, strict=True):
                candidate["retrieval_score"] = candidate.get("score", 0.0)
                candidate["score"] = float(rerank_score)
        output.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", item.get("frame_id", 0))),
            )
        )
        return output[:top_k]

    def mmr_rerank(
        self,
        candidates: list[dict[str, Any]],
        embeddings: np.ndarray,
        lambda_param: float = 0.5,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Diversify candidates using cosine-based Maximal Marginal Relevance."""

        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be between zero and one")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(candidates):
            raise ValueError("embeddings must have one row per candidate")
        if not candidates or top_k <= 0:
            return []
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        relevance = np.asarray(
            [float(candidate.get("score", 0.0)) for candidate in candidates],
            dtype=np.float32,
        )
        relevance_range = float(np.ptp(relevance))
        if relevance_range > 0:
            relevance = (relevance - relevance.min()) / relevance_range

        remaining = set(range(len(candidates)))
        selected: list[int] = []
        while remaining and len(selected) < min(top_k, len(candidates)):
            best_index = -1
            best_score = -float("inf")
            for index in sorted(remaining):
                redundancy = (
                    max(float(matrix[index] @ matrix[chosen]) for chosen in selected)
                    if selected
                    else 0.0
                )
                score = lambda_param * float(relevance[index]) - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(best_index)
            remaining.remove(best_index)
        return [dict(candidates[index]) for index in selected]
