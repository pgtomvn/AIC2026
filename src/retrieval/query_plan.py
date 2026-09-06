"""Deterministic, editable query planning for Textual KIS.

The planner deliberately keeps the user's original text intact.  Its clauses are
lightweight retrieval hints, not an LLM rewrite, so it remains usable during an
offline competition run.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from src.query.entity_extractor import EntityExtractor
from src.query.query_decomposer import (
    ProspectiveRelation,
    SequenceConstraint,
    prospective_relations,
    sequence_constraints,
)

_ENTITY_EXTRACTOR = EntityExtractor()

_QUOTED = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,})[\"“”'‘’]")
_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_BEFORE = re.compile(r"\b(?:before|prior to|trước khi|trước đó)\b", re.IGNORECASE)
_AFTER = re.compile(r"\b(?:after|following|sau khi|sau đó|tiếp theo)\b", re.IGNORECASE)
_THEN = re.compile(r"\b(?:then|next|rồi|sau đó|tiếp theo)\b", re.IGNORECASE)
_ASR_CUE = re.compile(
    r"(?<!\w)(?:"
    r"nói|đang nói|phát biểu|phỏng vấn|hỏi|trả lời|lời thoại|thuyết minh|"
    r"đọc|hát|ca từ|người dẫn|nghe thấy|âm thanh|"
    r"xin chào|chào mừng|tạm biệt|cảm ơn|"
    r"say|says|said|speaking|speech|interview|ask|asks|answer|answers|"
    r"dialogue|narration|narrator|lyrics|announce|announces|audio|"
    r"hello|welcome|goodbye|thank you"
    r")(?!\w)",
    re.IGNORECASE,
)

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


def stable_unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return NFC/case-insensitive unique values while preserving display order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_nfc(value)
        if not normalized:
            continue
        identity = normalized.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        result.append(normalized)
    return tuple(result)


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
    sequence_constraints: tuple[SequenceConstraint, ...] = ()
    prospective_relations: tuple[ProspectiveRelation, ...] = ()
    tokens: tuple[str, ...] = field(default_factory=tuple)
    asr_intent: bool = False
    asr_query: str = ""
    asr_terms: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan for API/UI editing."""

        return asdict(self)


def _split_temporal_clauses(
    text: str,
    *,
    ignored_spans: tuple[tuple[int, int], ...] = (),
) -> tuple[str | None, str | None, str | None]:
    markers: list[tuple[int, int, str]] = []
    for kind, pattern in (("before", _BEFORE), ("after", _AFTER), ("then", _THEN)):
        markers.extend(
            (match.start(), match.end(), kind)
            for match in pattern.finditer(text)
            if not any(
                match.start() < ignored_end and ignored_start < match.end()
                for ignored_start, ignored_end in ignored_spans
            )
        )
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
    quoted = stable_unique(tuple(normalize_nfc(value) for value in _QUOTED.findall(normalized)))
    actions = stable_unique(
        tuple(
            token for token, folded in zip(tokens, lowered, strict=True) if folded in _ACTION_WORDS
        )
    )
    attributes = stable_unique(
        tuple(
            token
            for token, folded in zip(tokens, lowered, strict=True)
            if folded in _ATTRIBUTE_WORDS
        )
    )
    objects = stable_unique(
        tuple(
            token for token, folded in zip(tokens, lowered, strict=True) if folded in _OBJECT_WORDS
        )
    )
    entities = tuple(entity.text for entity in _ENTITY_EXTRACTOR.extract(normalized))
    numeric_terms = tuple(
        token for token in tokens if any(character.isdigit() for character in token)
    )
    asr_intent = bool(quoted or _ASR_CUE.search(normalized) or entities or numeric_terms)
    asr_terms = stable_unique(
        [
            *quoted,
            *entities,
            *numeric_terms,
        ]
    )
    asr_query = " ".join(quoted).strip() if quoted else normalized
    prospective = prospective_relations(normalized)
    sequences = tuple(
        SequenceConstraint(
            value.kind,
            value.source_text,
            value.member_label,
            prospective[0].action if prospective else None,
            value.confidence,
        )
        for value in sequence_constraints(normalized)
    )
    ignored_spans = tuple(
        (match.start(), match.end())
        for pattern in (
            re.compile(r"\bone\s+(?:after|behind)\s+another\b", re.IGNORECASE),
            re.compile(
                r"\b(?P<label>[^\W_]+)\s+after\s+(?P=label)\b",
                re.IGNORECASE | re.UNICODE,
            ),
            re.compile(
                r"(?<!\w)(?:lần\s+lượt|nối\s+đuôi\s+nhau)(?!\w)",
                re.IGNORECASE,
            ),
        )
        for match in pattern.finditer(normalized)
    )
    before, current, after = _split_temporal_clauses(
        normalized,
        ignored_spans=ignored_spans,
    )
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
        sequence_constraints=sequences,
        prospective_relations=prospective,
        tokens=tokens,
        asr_intent=asr_intent,
        asr_query=asr_query,
        asr_terms=asr_terms,
    )
