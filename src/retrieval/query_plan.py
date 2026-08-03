"""Deterministic, editable query planning for Textual KIS.

The planner deliberately keeps the user's original text intact.  Its clauses are
lightweight retrieval hints, not an LLM rewrite, so it remains usable during an
offline competition run.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

_QUOTED = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,})[\"“”'‘’]")
_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_BEFORE = re.compile(r"\b(?:before|prior to|trước khi|trước đó)\b", re.IGNORECASE)
_AFTER = re.compile(r"\b(?:after|following|sau khi|sau đó|tiếp theo)\b", re.IGNORECASE)
_THEN = re.compile(r"\b(?:then|next|rồi|sau đó|tiếp theo)\b", re.IGNORECASE)

# These lists intentionally cover common competition phrasing only.  They are
# hints for channel weighting/evidence display, never query replacements.
_ACTION_WORDS = frozenset(
    {
        "walking",
        "running",
        "driving",
        "riding",
        "holding",
        "talking",
        "opening",
        "closing",
        "entering",
        "leaving",
        "đi",
        "chạy",
        "lái",
        "cầm",
        "nói",
        "mở",
        "đóng",
        "vào",
        "ra",
    }
)
_ATTRIBUTE_WORDS = frozenset(
    {
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "small",
        "large",
        "đỏ",
        "xanh",
        "vàng",
        "đen",
        "trắng",
        "nhỏ",
        "lớn",
    }
)
_OBJECT_WORDS = frozenset(
    {
        "person",
        "man",
        "woman",
        "child",
        "car",
        "bus",
        "truck",
        "motorbike",
        "bicycle",
        "phone",
        "bag",
        "door",
        "bridge",
        "table",
        "chair",
        "người",
        "đàn",
        "ông",
        "bà",
        "trẻ",
        "xe",
        "ô",
        "tô",
        "buýt",
        "máy",
        "đạp",
        "điện",
        "thoại",
        "túi",
        "cửa",
        "cầu",
        "bàn",
        "ghế",
    }
)


def normalize_nfc(text: str) -> str:
    """Return trimmed NFC-normalized Unicode without altering accents."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def accent_fold(text: str) -> str:
    """Return a case-folded auxiliary form suitable for lexical fallback."""

    normalized = unicodedata.normalize("NFD", normalize_nfc(text))
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D").casefold()


@dataclass(frozen=True)
class QueryPlan:
    """Editable retrieval hints derived from an untouched original query."""

    original_text: str
    normalized_text: str
    accent_folded_text: str
    quoted_text: tuple[str, ...] = ()
    named_entities: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    before: str | None = None
    current: str | None = None
    after: str | None = None
    tokens: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for API/UI editing."""

        return asdict(self)


def _split_temporal_clauses(text: str) -> tuple[str | None, str | None, str | None]:
    markers: list[tuple[int, int, str]] = []
    for kind, pattern in (("before", _BEFORE), ("after", _AFTER), ("then", _THEN)):
        markers.extend((match.start(), match.end(), kind) for match in pattern.finditer(text))
    markers.sort()
    if not markers:
        return None, text or None, None

    first_start, first_end, first_kind = markers[0]
    left = text[:first_start].strip(" ,.;:-") or None
    right = text[first_end:].strip(" ,.;:-") or None
    if first_kind == "before":
        return left, right, None
    if first_kind == "after":
        # Both English and Vietnamese use "A after/sau khi B": B precedes A.
        return right, left, None
    return left, right, None


def build_query_plan(text: str) -> QueryPlan:
    """Build a deterministic plan while preserving the exact user query."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("query text must be a non-empty string")
    normalized = normalize_nfc(text)
    tokens = tuple(_TOKEN.findall(normalized))
    lowered = tuple(token.casefold() for token in tokens)
    quoted = tuple(normalize_nfc(value) for value in _QUOTED.findall(normalized))
    actions = tuple(
        token for token, folded in zip(tokens, lowered, strict=True) if folded in _ACTION_WORDS
    )
    attributes = tuple(
        token for token, folded in zip(tokens, lowered, strict=True) if folded in _ATTRIBUTE_WORDS
    )
    objects = tuple(
        token for token, folded in zip(tokens, lowered, strict=True) if folded in _OBJECT_WORDS
    )
    # Capitalized multi-character tokens are useful entity hints.  Avoid
    # interpreting the sentence-initial word as an entity on its own.
    entities = tuple(
        token
        for index, token in enumerate(tokens)
        if index > 0 and len(token) > 1 and token[0].isupper()
    )
    before, current, after = _split_temporal_clauses(normalized)
    return QueryPlan(
        original_text=text,
        normalized_text=normalized,
        accent_folded_text=accent_fold(normalized),
        quoted_text=quoted,
        named_entities=entities,
        objects=objects,
        actions=actions,
        attributes=attributes,
        before=before,
        current=current,
        after=after,
        tokens=tokens,
    )
