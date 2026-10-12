"""Pure, bounded state and vector operations for explicit user-driven retrieval.

Rocchio coefficients are system defaults, not claimed OpenCubee parameters.
No inferred negatives, model downloads, or process-global conversation state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def unit_vector(value: Any, *, dimension: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("invalid_vector")
    if dimension is not None and vector.size != dimension:
        raise ValueError("vector_dimension_mismatch")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("zero_vector")
    return vector / norm


def rocchio(
    original: Any, *, positive: Any = None, negative: Any = None,
    reference: Any = None, alpha: float = 1.0, beta: float = .75,
    gamma: float = .25, reference_weight: float = .75,
) -> np.ndarray:
    """Normalize each observation within ONE vector space, then normalize q'."""
    weights = (alpha, beta, gamma, reference_weight)
    if any(not np.isfinite(v) or v < 0 for v in weights):
        raise ValueError("invalid_feedback_weight")
    query = unit_vector(original)
    updated = alpha * query
    for matrix, weight in ((positive, beta), (negative, -gamma)):
        if matrix is not None:
            values = np.asarray(matrix)
            if values.ndim != 2 or not len(values):
                raise ValueError("invalid_feedback_matrix")
            rows = np.stack([unit_vector(row, dimension=query.size) for row in values])
            updated += weight * rows.mean(axis=0)
    if reference is not None:
        updated += reference_weight * unit_vector(reference, dimension=query.size)
    return unit_vector(updated)


def edit_conditions(
    conditions: tuple[str, ...], operation: str, text: str = "", index: int | None = None,
) -> tuple[str, ...]:
    """Explicit indexed replacement/removal: never guess which condition changed."""
    values = list(conditions)
    text = text.strip()
    if operation not in {"add", "replace", "remove"}:
        raise ValueError("unknown_condition_operation")
    if operation != "add" and (
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(values)
    ):
        raise ValueError("select_condition_to_edit")
    if operation != "remove" and not text:
        raise ValueError("condition_text_required")
    if operation == "add" and text not in values:
        values.append(text)
    elif operation == "replace":
        values[index] = text
    elif operation == "remove":
        values.pop(index)
    if not values or len(values) > 20 or len(". ".join(values)) > 8192:
        raise ValueError("condition_budget_exceeded")
    return tuple(dict.fromkeys(values))


@dataclass
class SearchConversation:
    """One Streamlit session owns one instance. Snapshots are detached copies."""

    turns: list[dict[str, Any]] = field(default_factory=list)
    initial: dict[str, Any] | None = None

    def push(self, snapshot: dict[str, Any]) -> None:
        value = deepcopy(snapshot)
        if self.initial is None:
            self.initial = deepcopy(value)
        self.turns.append(value)
        self.turns[:] = self.turns[-20:]

    def undo(self) -> dict[str, Any] | None:
        if len(self.turns) > 1:
            self.turns.pop()
        return deepcopy(self.turns[-1]) if self.turns else None

    def reset(self) -> dict[str, Any] | None:
        self.turns = [deepcopy(self.initial)] if self.initial is not None else []
        return deepcopy(self.initial)
