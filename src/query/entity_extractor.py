"""Deterministic Vietnamese named-entity extraction for retrieval queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

EntityCategory = Literal["location", "person", "organization"]
EntityMode = Literal["optional", "required"]

ENTITY_EXTRACTOR_VERSION = "vi-entity-rules-v1"
MAX_ENTITIES_PER_PROMPT = 16
MAX_ENTITIES_PER_TRAKE_QUERY = 64


def normalize_entity_text(value: str) -> str:
    """Return the canonical display form used by public entity contracts."""

    if not isinstance(value, str):
        raise TypeError("entity text must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", value).split()).strip(" ,.;:!?\"'()[]{}")
    if not normalized:
        raise ValueError("entity text must not be empty")
    if len(normalized) > 128:
        raise ValueError("entity text cannot exceed 128 Unicode code points")
    return normalized


def fold_entity_text(value: str) -> str:
    """Return an accent-insensitive Vietnamese identity, including ``đ``."""

    normalized = unicodedata.normalize("NFD", normalize_entity_text(value))
    folded = "".join(character for character in normalized if unicodedata.category(character) != "Mn")
    return folded.replace("đ", "d").replace("Đ", "D").casefold()


def entity_identity(value: str) -> str:
    return fold_entity_text(value)


@dataclass(frozen=True, slots=True)
class NamedEntity:
    """One server-authoritative entity extracted from a query."""

    text: str
    category: EntityCategory
    aliases: tuple[str, ...] = ()
    start: int = -1
    end: int = -1
    source: str = "heuristic"

    def __post_init__(self) -> None:
        text = normalize_entity_text(self.text)
        if self.category not in {"location", "person", "organization"}:
            raise ValueError("unsupported entity category")
        aliases: list[str] = []
        seen = {entity_identity(text)}
        for value in self.aliases:
            alias = normalize_entity_text(value)
            identity = entity_identity(alias)
            if identity not in seen:
                aliases.append(alias)
                seen.add(identity)
        if self.start < -1 or self.end < -1 or (self.start >= 0 and self.end <= self.start):
            raise ValueError("entity span is invalid")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "aliases", tuple(aliases))

    @property
    def identity(self) -> str:
        return entity_identity(self.text)

    def lexical_terms(self) -> tuple[str, ...]:
        values = [self.text, *self.aliases]
        folded = fold_entity_text(self.text)
        if folded not in {value.casefold() for value in values}:
            values.append(folded)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class EntityConstraint:
    """Operator-selected entity strictness without client-supplied category data."""

    text: str
    mode: EntityMode = "optional"

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", normalize_entity_text(self.text))
        if self.mode not in {"optional", "required"}:
            raise ValueError("entity mode must be optional or required")

    @property
    def identity(self) -> str:
        return entity_identity(self.text)

    @classmethod
    def from_value(cls, value: EntityConstraint | Mapping[str, Any]) -> EntityConstraint:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("entity constraints must be objects")
        unknown = set(value) - {"text", "mode"}
        if unknown:
            raise ValueError("entity constraints contain unsupported fields")
        return cls(text=value.get("text"), mode=value.get("mode", "optional"))


def normalize_entity_constraints(
    values: Sequence[EntityConstraint | Mapping[str, Any]] | None,
    *,
    maximum: int = MAX_ENTITIES_PER_PROMPT,
) -> tuple[EntityConstraint, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("entity_constraints must be a sequence")
    normalized: list[EntityConstraint] = []
    seen: set[str] = set()
    for raw in values:
        value = EntityConstraint.from_value(raw)
        identity = value.identity
        if identity in seen:
            raise ValueError("entity_constraints must be unique ignoring case and accents")
        seen.add(identity)
        normalized.append(value)
    if len(normalized) > maximum:
        raise ValueError(f"entity_constraints cannot contain more than {maximum} entries")
    return tuple(normalized)


# The versioned legacy 63-province/city vocabulary is retained because the
# competition corpus predates the latest administrative consolidation.
_LEGACY_63 = (
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu",
    "Bắc Ninh", "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước",
    "Bình Thuận", "Cà Mau", "Cao Bằng", "Cần Thơ", "Đà Nẵng", "Đắk Lắk",
    "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang",
    "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương", "Hải Phòng", "Hậu Giang",
    "Hòa Bình", "Hồ Chí Minh", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
    "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An",
    "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên",
    "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị",
    "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa",
    "Thừa Thiên Huế", "Tiền Giang", "Trà Vinh", "Tuyên Quang", "Vĩnh Long",
    "Vĩnh Phúc", "Yên Bái",
)

_LOCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "Hồ Chí Minh": ("TP.HCM", "TP HCM", "TPHCM", "Sài Gòn", "Ho Chi Minh City", "HCMC"),
    "Hà Nội": ("Thủ đô Hà Nội", "Ha Noi", "Hanoi"),
    "Thừa Thiên Huế": ("Huế", "Thành phố Huế", "Hue"),
    "Bà Rịa - Vũng Tàu": ("Bà Rịa Vũng Tàu", "Vũng Tàu"),
    "Đắk Lắk": ("Đắc Lắc", "Dak Lak"),
    "Đắk Nông": ("Đắc Nông", "Dak Nong"),
}

_ADMIN_PREFIX = re.compile(
    r"(?u)\b(?i:tỉnh|thành\s+phố|tp\.?|quận|huyện|thị\s+xã|thị\s+trấn|xã|phường)\s+"
    r"(?P<name>[A-ZÀ-ỸĐ][\wÀ-ỹĐđ.-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ.-]*){0,4})"
)
_ACRONYM = re.compile(r"(?<![\wÀ-ỹ])(?:[A-ZĐ]{2,12}[0-9]{0,4})(?![\wÀ-ỹ])")
_PERSON_CONTEXT = re.compile(
    r"(?u)\b(?i:anh\s+hùng|nhà\s+thơ|chủ\s+tịch|ông|bà|tướng|nhạc\s+sĩ|ca\s+sĩ)\s+"
    r"(?P<name>[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ'-]*){1,4})"
)
_NAMED_ORG = re.compile(
    r"(?u)\b(?i:câu\s+lạc\s+bộ|clb|đài\s+truyền\s+hình|công\s+ty|tập\s+đoàn|"
    r"trường|ủy\s+ban)\s+(?i:tên\s+là\s+)?(?P<name>[A-ZĐ]{2,12}[0-9]{0,4})\b"
)
_COMMON_SURNAMES = frozenset(
    {"nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ", "đặng", "bùi", "đỗ", "hồ", "ngô", "dương", "lý"}
)
_CAPITALIZED_SEQUENCE = re.compile(
    r"(?<![\wÀ-ỹ])(?P<name>[A-ZÀ-ỸĐ][a-zà-ỹđ'-]+(?:\s+[A-ZÀ-ỸĐ][a-zà-ỹđ'-]+){1,4})(?![\wÀ-ỹ])"
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    text: str
    category: EntityCategory
    start: int
    end: int
    source: str
    aliases: tuple[str, ...] = ()
    priority: int = 10


class EntityExtractor:
    """Versioned regex/dictionary extractor with deterministic overlap handling."""

    version = ENTITY_EXTRACTOR_VERSION

    def __init__(self) -> None:
        lookup: dict[str, tuple[str, tuple[str, ...]]] = {}
        for canonical in _LEGACY_63:
            aliases = _LOCATION_ALIASES.get(canonical, ())
            for label in (canonical, *aliases, fold_entity_text(canonical)):
                lookup[fold_entity_text(label)] = (canonical, aliases)
        self._locations = lookup
        labels = sorted(
            {
                label
                for canonical in _LEGACY_63
                for label in (
                    canonical,
                    fold_entity_text(canonical),
                    *_LOCATION_ALIASES.get(canonical, ()),
                )
            },
            key=len,
            reverse=True,
        )
        self._location_pattern = re.compile(
            r"(?<![\wÀ-ỹ])(?:" + "|".join(re.escape(value) for value in labels) + r")(?![\wÀ-ỹ])",
            re.IGNORECASE,
        )

    @property
    def legacy_location_count(self) -> int:
        return len(_LEGACY_63)

    def _location(self, value: str) -> tuple[str, tuple[str, ...]] | None:
        return self._locations.get(fold_entity_text(value))

    def extract(self, text: str, *, maximum: int = MAX_ENTITIES_PER_PROMPT) -> tuple[NamedEntity, ...]:
        normalized = unicodedata.normalize("NFC", text)
        if not normalized.strip():
            return ()
        candidates: list[_Candidate] = []
        for match in self._location_pattern.finditer(normalized):
            location = self._location(match.group(0))
            if location is not None:
                canonical, aliases = location
                candidates.append(_Candidate(canonical, "location", match.start(), match.end(), "dictionary", aliases, 0))
        for match in _ADMIN_PREFIX.finditer(normalized):
            name = match.group("name").rstrip(".,;:!?")
            location = self._location(name)
            canonical, aliases = location if location is not None else (name, ())
            candidates.append(_Candidate(canonical, "location", match.start("name"), match.start("name") + len(name), "administrative", aliases, 1))
        for match in _NAMED_ORG.finditer(normalized):
            candidates.append(_Candidate(match.group("name"), "organization", match.start("name"), match.end("name"), "organization_context", (), 1))
        for match in _ACRONYM.finditer(normalized):
            value = match.group(0)
            if value not in {"CLB", "TP", "THPT", "THCS"}:
                candidates.append(_Candidate(value, "organization", match.start(), match.end(), "acronym", (), 2))
        for match in _PERSON_CONTEXT.finditer(normalized):
            name = match.group("name").rstrip(".,;:!?")
            candidates.append(_Candidate(name, "person", match.start("name"), match.start("name") + len(name), "person_context", (), 1))
        for match in _CAPITALIZED_SEQUENCE.finditer(normalized):
            name = match.group("name")
            first = fold_entity_text(name.split()[0])
            if first in _COMMON_SURNAMES:
                candidates.append(_Candidate(name, "person", match.start("name"), match.end("name"), "capitalization", (), 3))

        selected: list[_Candidate] = []
        for candidate in sorted(candidates, key=lambda item: (item.priority, -(item.end - item.start), item.start, item.text.casefold())):
            if any(candidate.start < existing.end and existing.start < candidate.end for existing in selected):
                continue
            if any(entity_identity(candidate.text) == entity_identity(existing.text) for existing in selected):
                continue
            selected.append(candidate)
        selected.sort(key=lambda item: (item.start, item.priority, item.text.casefold()))
        return tuple(
            NamedEntity(item.text, item.category, item.aliases, item.start, item.end, item.source)
            for item in selected[:maximum]
        )


def merge_entity_constraints(
    extracted: Sequence[NamedEntity],
    explicit: Sequence[EntityConstraint] = (),
    legacy_entities: Iterable[str] = (),
) -> tuple[tuple[NamedEntity, EntityMode], ...]:
    """Merge server extraction with backward-compatible operator hints."""

    modes = {value.identity: value.mode for value in explicit}
    result: list[tuple[NamedEntity, EntityMode]] = []
    seen: set[str] = set()
    for entity in extracted:
        result.append((entity, modes.get(entity.identity, "optional")))
        seen.add(entity.identity)
    for value in legacy_entities:
        text = normalize_entity_text(value)
        identity = entity_identity(text)
        if identity not in seen:
            result.append((NamedEntity(text, "organization", source="operator_hint"), modes.get(identity, "optional")))
            seen.add(identity)
    for constraint in explicit:
        if constraint.identity not in seen:
            result.append((NamedEntity(constraint.text, "organization", source="operator_constraint"), constraint.mode))
            seen.add(constraint.identity)
    return tuple(result[:MAX_ENTITIES_PER_PROMPT])
