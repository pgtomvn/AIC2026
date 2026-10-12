"""Read-only TRAKE scope; traffic and ambiguous equal-PTS images stay excluded."""
from __future__ import annotations

import re
from typing import Any

B02_TRAKE_POLICY = "b02-ms-first-source-pts-v1"


def trake_video_allowed(batch_id: str, video_id: str) -> bool:
    if batch_id != "aic26-b2":
        return True
    return bool(re.fullmatch(r"(?:M\d{2}_V\d{3}|S\d{2}-V\d{3})", video_id)) and video_id != "M08_V001"


def trake_filters(batch_id: str, **extra: Any) -> dict[str, Any]:
    """Apply eligibility before candidate budgets and before scrolling vectors."""
    return {**extra, "batch_id": batch_id,
            **({"trake_eligible": True} if batch_id == "aic26-b2" else {})}
