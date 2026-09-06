"""Canonical Codabench query-ID parsing and deterministic ordering."""

from __future__ import annotations

import re
from typing import Literal

CodabenchTask = Literal["kis", "qa", "trake"]

_QUERY_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_FILENAME = re.compile(
    r"^query-([A-Za-z0-9_-]{1,32})-(kis|qa|trake)\.csv$"
)
_NATURAL_TOKEN = re.compile(r"([0-9]+)")


def normalize_codabench_query_id(value: str) -> str:
    """Trim and validate one operator-supplied Codabench query ID."""

    if not isinstance(value, str):
        raise TypeError("query_id must be a string")
    normalized = value.strip()
    if _QUERY_ID.fullmatch(normalized) is None:
        raise ValueError(
            "query_id must contain 1-32 ASCII letters, digits, underscores, or hyphens"
        )
    return normalized


def codabench_query_filename(query_id: str, task: CodabenchTask) -> str:
    """Return the canonical server-owned CSV filename for one query."""

    if task not in {"kis", "qa", "trake"}:
        raise ValueError("task must be kis, qa, or trake")
    return f"query-{normalize_codabench_query_id(query_id)}-{task}.csv"


def parse_codabench_query_filename(filename: str) -> tuple[str, CodabenchTask]:
    """Parse an exact safe Codabench filename without normalizing the filename."""

    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    match = _FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError("filename must match query-<id>-<task>.csv")
    return match.group(1), match.group(2)  # type: ignore[return-value]


def codabench_query_id_sort_key(query_id: str) -> tuple[object, ...]:
    """Return a natural, case-stable ordering key while preserving leading zeros."""

    normalized = normalize_codabench_query_id(query_id)
    folded = normalized.casefold()
    tokens: tuple[tuple[int, object], ...] = tuple(
        (0, int(token)) if token.isdigit() else (1, token)
        for token in _NATURAL_TOKEN.split(folded)
        if token
    )
    return tokens, folded, normalized


def codabench_query_id_collision_key(query_id: str) -> str:
    """Return the Windows-safe, case-insensitive collision identity."""

    return normalize_codabench_query_id(query_id).casefold()


__all__ = [
    "CodabenchTask",
    "codabench_query_filename",
    "codabench_query_id_collision_key",
    "codabench_query_id_sort_key",
    "normalize_codabench_query_id",
    "parse_codabench_query_filename",
]
