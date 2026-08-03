"""Deterministic answer normalization for Visual Q&A evaluation and export.

The general-text policy is deliberately conservative: it keeps Vietnamese
accents and every semantic word.  Numeric conversion is a separate policy for
historical tasks whose answers are quantities.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

NUMERIC = "numeric"
GENERAL_TEXT = "general_text"
SUPPORTED_MODES = frozenset({NUMERIC, GENERAL_TEXT})

_INTEGER = re.compile(r"^[+]?[0-9]+$")
_QUANTITY_QUESTION_MARKERS = (
    "bao nhiêu",
    "mấy",
    "số lượng",
    "số ",
    "thứ mấy",
    "how many",
    "how much",
    "what number",
    "which number",
    "number of",
)
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "không": 0,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
    "mười một": 11,
    "mười hai": 12,
    "mười ba": 13,
    "mười bốn": 14,
    "mười tư": 14,
    "mười lăm": 15,
    "mười năm": 15,
    "mười sáu": 16,
    "mười bảy": 17,
    "mười tám": 18,
    "mười chín": 19,
    "hai mươi": 20,
}


class AnswerNormalizationError(ValueError):
    """Raised when an answer cannot satisfy the selected output policy."""


def _require_answer(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("answer must be a string")
    normalized = unicodedata.normalize("NFC", value)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise AnswerNormalizationError("answer must not be empty")
    return normalized


def _is_edge_punctuation(value: str) -> bool:
    return unicodedata.category(value).startswith("P")


def normalize_general_text(value: str) -> str:
    """Case-fold and trim outer punctuation while preserving answer meaning."""

    normalized = _require_answer(value).casefold()
    start = 0
    end = len(normalized)
    while start < end and _is_edge_punctuation(normalized[start]):
        start += 1
    while end > start and _is_edge_punctuation(normalized[end - 1]):
        end -= 1
    normalized = " ".join(normalized[start:end].split())
    if not normalized:
        raise AnswerNormalizationError("answer contains no text after normalization")
    return normalized


def normalize_numeric(value: str, question: str | None = None) -> str:
    """Normalize one complete Arabic, English, or Vietnamese quantity answer."""

    normalized_question = None
    if question is not None:
        normalized_question = normalize_general_text(question)
    normalized = normalize_general_text(value)
    if _INTEGER.fullmatch(normalized):
        return str(int(normalized, 10))
    number = _NUMBER_WORDS.get(normalized)
    if number is None:
        raise AnswerNormalizationError(
            "numeric answer must be an integer or a supported whole number word"
        )
    if normalized_question is not None and not any(
        marker in normalized_question for marker in _QUANTITY_QUESTION_MARKERS
    ):
        raise AnswerNormalizationError(
            "number-word answer requires a quantity or number question"
        )
    return str(number)


def normalize_answer(
    value: str,
    *,
    mode: str = NUMERIC,
    question: str | None = None,
) -> str:
    """Normalize an answer using a versioned competition policy name."""

    if mode == NUMERIC:
        return normalize_numeric(value, question=question)
    if mode == GENERAL_TEXT:
        if question is not None:
            _require_answer(question)
        return normalize_general_text(value)
    raise ValueError(f"unsupported answer-normalization mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class AnswerNormalizer:
    """Small injectable normalization engine for reasoners and exporters."""

    mode: str = NUMERIC

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported answer-normalization mode: {self.mode!r}")

    def normalize(self, value: str, question: str | None = None) -> str:
        """Return the canonical answer for this engine's configured mode."""

        return normalize_answer(value, mode=self.mode, question=question)

    def __call__(self, value: str, question: str | None = None) -> str:
        return self.normalize(value, question=question)


__all__ = [
    "AnswerNormalizationError",
    "AnswerNormalizer",
    "GENERAL_TEXT",
    "NUMERIC",
    "SUPPORTED_MODES",
    "normalize_answer",
    "normalize_general_text",
    "normalize_numeric",
]
