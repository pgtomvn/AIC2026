"""Deterministic federation of independently ranked canonical batches."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

B01 = "aic25-b1-reference"
B02 = "aic26-b2"
BATCHES = (B01, B02)


def search_batches(config: Mapping[str, Any], selected: str) -> tuple[str, ...]:
    """Opt-in default scope; the launch identity remains a single batch."""
    enabled = config.get("retrieval", {}).get("all_batches", {}).get("enabled", False)
    return BATCHES if enabled else (selected,)


def validate_scope(batches: Sequence[str]) -> tuple[str, ...]:
    """Only the two audited batches can be federated; never silently truncate."""
    if len(batches) != 2 or set(batches) != set(BATCHES):
        raise ValueError("combined search requires the audited B01 and B02 batches")
    return BATCHES


def merge_rankings(rankings: Mapping[str, Sequence[Mapping[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    """Equal-weight rank fusion, not cross-model raw-score comparison.

    Batch identities are disjoint. A single RRF contribution per item is
    equivalent to stable rank interleaving; ties follow canonical batch order.
    Original branch scores remain available and coordinates are never rewritten.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    merged: list[dict[str, Any]] = []
    for batch in BATCHES:
        seen = set()
        for rank, item in enumerate(rankings.get(batch, ()), 1):
            if item.get("batch_id") != batch:
                raise ValueError("retrieval returned an out-of-scope batch identity")
            coordinate = item.get("keyframe_ordinal") if batch == B02 else item.get("source_frame_idx")
            identity = (batch, item.get("video_id"), coordinate)
            if identity in seen:
                raise ValueError("retrieval returned duplicate canonical identities")
            seen.add(identity)
            merged.append(dict(item, batch_rank=rank, batch_score=item.get("score"),
                               score=1.0 / (60 + rank), fused_score=1.0 / (60 + rank)))
    merged.sort(key=lambda item: (item["batch_rank"], BATCHES.index(item["batch_id"])))
    return [dict(item, rank=rank) for rank, item in enumerate(merged[:limit], 1)]
