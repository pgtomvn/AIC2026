"""
Visual Q&A Pipeline
===================
Locate a relevant video moment and extract a specific answer using
a Visual Language Model (VLM).

Output format: (video_id, frame_id, answer)
"""

from __future__ import annotations


class VisualQAPipeline:
    """Visual Question Answering pipeline.

    Workflow:
    1. Encode query → CLIP embedding for initial retrieval
    2. Retrieve top-K candidate frames
    3. Feed each candidate frame + question to VLM
    4. Extract and validate the answer
    5. Rerank by VLM confidence
    6. Return (video_id, frame_id, answer) tuples

    Args:
        clip_encoder: CLIPEncoder instance.
        vector_store: VectorStore instance.
        vlm_engine: VLMEngine instance.
        top_k_retrieve: Number of candidates to retrieve for VLM.
        top_k_final: Number of final answers to return.
    """

    def __init__(
        self,
        clip_encoder=None,
        vector_store=None,
        vlm_engine=None,
        top_k_retrieve: int = 20,
        top_k_final: int = 100,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.vector_store = vector_store
        self.vlm_engine = vlm_engine
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final

    def answer(self, question: str) -> list[dict]:
        """Execute the Visual Q&A pipeline.

        Args:
            question: The visual question to answer.

        Returns:
            Ranked list of dicts: {video_id, frame_id, answer, score}.
        """
        # TODO:
        #   1. Retrieve candidate frames via CLIP
        #   2. For each candidate, run VLM with (image, question)
        #   3. Collect answers with confidence scores
        #   4. Rerank and return top results
        raise NotImplementedError("VisualQAPipeline.answer")
