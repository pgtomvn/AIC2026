"""
Textual KIS Pipeline
====================
Find exact video moments matching a natural language text description.
Combines vector search (CLIP) with BM25 keyword search and reranking.

Output format: (video_id, frame_id)
"""

from __future__ import annotations

from typing import Optional


class TextualKISPipeline:
    """Textual Known Item Search pipeline.

    Workflow:
    1. Encode query text → CLIP embedding
    2. Vector search via Qdrant → top-K candidates
    3. BM25 search via Meilisearch → complementary candidates
    4. Score fusion (weighted sum)
    5. Rerank top candidates
    6. Return ranked list of (video_id, frame_id)

    Args:
        clip_encoder: CLIPEncoder instance.
        vector_store: VectorStore instance.
        text_store: TextStore instance.
        reranker: Optional Reranker instance.
        vector_weight: Fusion weight for vector similarity.
        bm25_weight: Fusion weight for BM25 relevance.
        top_k: Number of final results to return.
    """

    def __init__(
        self,
        clip_encoder=None,
        vector_store=None,
        text_store=None,
        reranker=None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 100,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.vector_store = vector_store
        self.text_store = text_store
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k

    def search(self, query: str) -> list[dict]:
        """Execute the full KIS retrieval pipeline.

        Args:
            query: Natural language text description.

        Returns:
            Ranked list of dicts: {video_id, frame_id, score}.
        """
        # TODO:
        #   1. query_emb = self.clip_encoder.encode_text(query)
        #   2. vector_results = self.vector_store.search(query_emb)
        #   3. text_results = self.text_store.search(query)
        #   4. fused = self._fuse_scores(vector_results, text_results)
        #   5. reranked = self.reranker.rerank(query, fused) if reranker
        #   6. return reranked[:self.top_k]
        raise NotImplementedError("TextualKISPipeline.search")

    def _fuse_scores(
        self, vector_results: list[dict], text_results: list[dict]
    ) -> list[dict]:
        """Reciprocal Rank Fusion or weighted score combination."""
        # TODO: Implement score fusion logic
        raise NotImplementedError("TextualKISPipeline._fuse_scores")
