"""Auditable, bounded noun/attribute proposals for the EviBind research path.

This is a small bilingual grammar, not an open-domain semantic parser. Unparsed
text remains visible. Clothing parts never inherit the enclosing person's ROI.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from src.query.event_scope import EventScopePlan, scope_explicit_events

VERSION = "evibind-nominal-grammar-v1"
# A narrow class is never widened to Person, Clothing or Vehicle for coverage.
NOUNS = {
    "xe ô tô": ("car", ("Car",)), "ô tô": ("car", ("Car",)),
    "car": ("car", ("Car",)), "xe máy": ("motorcycle", ("Motorcycle",)),
    "motorcycle": ("motorcycle", ("Motorcycle",)),
    "áo thun": ("shirt", ("Shirt", "T-shirt")),
    "áo sơ mi": ("shirt", ("Shirt",)), "sơ mi": ("shirt", ("Shirt",)),
    "áo dài tay": ("shirt", ("Shirt",)), "áo": ("shirt", ("Shirt",)),
    "shirt": ("shirt", ("Shirt",)), "t-shirt": ("shirt", ("T-shirt",)),
    "nón": ("hat", ("Hat", "Sun hat")), "mũ": ("hat", ("Hat", "Sun hat")),
    "hat": ("hat", ("Hat", "Sun hat")), "vest": ("suit", ("Suit",)),
    "suit": ("suit", ("Suit",)), "trang phục": ("clothing", ("Clothing",)),
    "clothing": ("clothing", ("Clothing",)), "khăn": ("scarf", ("Scarf",)),
    "khăn trùm đầu": ("scarf", ("Scarf",)), "scarf": ("scarf", ("Scarf",)),
    "cà vạt": ("tie", ("Tie",)), "tie": ("tie", ("Tie",)),
    "đĩa": ("plate", ("Plate",)), "plate": ("plate", ("Plate",)),
    "chén": ("bowl", ("Bowl",)), "bowl": ("bowl", ("Bowl",)),
    "đũa": ("chopsticks", ("Chopsticks",)),
    "chopsticks": ("chopsticks", ("Chopsticks",)),
    "ghe": ("boat", ("Boat",)), "thuyền": ("boat", ("Boat",)),
    "boat": ("boat", ("Boat",)), "hoa": ("flower", ("Flower",)),
    "flower": ("flower", ("Flower",)), "băng-rôn": ("banner", ("Banner",)),
    "banner": ("banner", ("Banner",)),
}
COLORS = {
    "xanh dương": "blue", "xanh da trời": "blue", "xanh lá cây": "green",
    "xanh lá": "green", "đỏ": "red", "đen": "black", "trắng": "white",
    "vàng": "yellow", "hồng": "pink", "tím": "purple", "red": "red",
    "black": "black", "white": "white", "yellow": "yellow", "blue": "blue",
    "green": "green", "pink": "pink", "purple": "purple",
}


def _options(values: object) -> str:
    return "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))


_NOUN = rf"(?<!\w)(?:{_options(NOUNS)})(?!\w)"
_COLOR = rf"(?<!\w)(?:{_options(COLORS)})(?!\w)"
_NC = re.compile(rf"(?P<noun>{_NOUN})\s+(?:(?:có\s+)?màu\s+)?(?P<color>{_COLOR})", re.I)
_CN = re.compile(rf"(?P<color>{_COLOR})\s+(?P<noun>{_NOUN})", re.I)
_NOUN_RX = re.compile(_NOUN, re.I)
_COLOR_RX = re.compile(_COLOR, re.I)
_POSITION = re.compile(
    r"\s+(?:ở\s+)?(?:bên|phía)\s+(?P<vi>trái|phải)\s+(?:khung\s+hình|màn\s+hình)"
    r"|\s+on\s+the\s+(?P<en>left|right)\s+of\s+the\s+(?:frame|screen)", re.I
)
_SUFFIX_MODIFIER = re.compile(
    r"\s*(?:[-/]\s*|(?:và|and|or|hoặc)\s+)?(?:"
    + _COLOR + r"|xanh|nhạt|đậm|ngà|sọc|striped|ish)(?!\w)", re.I
)
_NEGATION = re.compile(r"\b(?:không(?!\s+(?:gian|khí)\b)|chẳng|chưa|not|no|without)\b", re.I)


@dataclass(frozen=True)
class ObjectMention:
    variable: str
    event_id: str
    source_span: tuple[int, int]
    source_text: str
    semantic_class: str
    detector_labels: tuple[str, ...]


@dataclass(frozen=True)
class BoundAttribute:
    predicate_id: str
    variable: str
    event_id: str
    kind: str
    value: str
    source_span: tuple[int, int]
    source_text: str
    blocked_reason: str = ""


@dataclass(frozen=True)
class BindingPlan:
    version: str
    query_revision: str
    events: EventScopePlan
    objects: tuple[ObjectMention, ...]
    attributes: tuple[BoundAttribute, ...]
    unparsed_spans: tuple[tuple[int, int], ...]
    unbound_color_spans: tuple[tuple[int, int], ...]
    grammar_complete: bool
    semantic_validation: str = "not_human_validated"


def parse_binding_query(text: str) -> BindingPlan:
    """Propose adjacent nominal attributes with exact spans and explicit limits.

    Only a single positive noun/color phrase with optional screen position is
    declared grammar-complete. Other clauses, quantities, actions, ownership and
    temporal grounding remain unresolved, even when some attributes are proposed.
    """
    events = scope_explicit_events(text)
    revision = hashlib.sha256((VERSION + "\0" + text).encode("utf-8")).hexdigest()
    objects: dict[tuple[int, int], ObjectMention] = {}
    attributes: list[BoundAttribute] = []
    matched_spans: list[tuple[int, int]] = []
    color_spans: set[tuple[int, int]] = set()
    for pattern in (_NC, _CN):
        for match in pattern.finditer(text):
            ns, cs = match.span("noun"), match.span("color")
            # Do not reinterpret a second noun after a Vietnamese noun-color
            # phrase as the referent of that same color.
            if cs in color_spans:
                continue
            event = next((e for e in events.events
                          if e.start <= match.start() and match.end() <= e.end), None)
            if event is None:
                continue
            semantic_class, labels = NOUNS[match.group("noun").lower()]
            obj = objects.setdefault(ns, ObjectMention(
                f"x{ns[0]}_{ns[1]}", event.event_id, ns, text[slice(*ns)], semantic_class, labels
            ))
            clause = text[event.start:event.end]
            blocked = ("negation_in_text_event" if _NEGATION.search(clause)
                       else "qualified_or_multiple_color" if _SUFFIX_MODIFIER.match(text, cs[1])
                       else "")
            pid = hashlib.sha256(f"{revision}:{ns}:{cs}:color".encode()).hexdigest()[:24]
            attributes.append(BoundAttribute(
                pid, obj.variable, event.event_id, "color", COLORS[match.group("color").lower()],
                match.span(), match.group(), blocked
            ))
            color_spans.add(cs)
            stop = match.end()
            position = _POSITION.match(text, stop)
            if position:
                raw_side = position.group("vi") or position.group("en")
                side = "left" if raw_side.lower() in {"trái", "left"} else "right"
                pid = hashlib.sha256(f"{revision}:{ns}:{position.span()}:position".encode()).hexdigest()[:24]
                attributes.append(BoundAttribute(
                    pid, obj.variable, event.event_id, "screen_position", side,
                    (match.start(), position.end()), text[match.start():position.end()], blocked
                ))
                stop = position.end()
            matched_spans.append((match.start(), stop))
    # Keep lexical noun proposals even if no supported adjacent color was found.
    # They cannot create positive evidence by themselves.
    for match in _NOUN_RX.finditer(text):
        if match.span() not in objects:
            event = next(e for e in events.events if e.start <= match.start() < e.end)
            cls, labels = NOUNS[match.group().lower()]
            objects[match.span()] = ObjectMention(
                f"x{match.start()}_{match.end()}", event.event_id, match.span(), match.group(), cls, labels
            )
    covered = [False] * len(text)
    for start, end in matched_spans:
        covered[start:end] = [True] * (end-start)
    residual = tuple(m.span() for m in re.finditer(r"\S+", text)
                     if any(not covered[i] and text[i].isalnum() for i in range(*m.span())))
    # Articles are allowed only before a single otherwise complete simple phrase.
    prefix = text[:matched_spans[0][0]].strip().lower() if len(matched_spans) == 1 else "?"
    suffix = text[matched_spans[0][1]:].strip(" .!?\r\n\t") if len(matched_spans) == 1 else "?"
    complete = (len(matched_spans) == 1 and len(events.events) == 1
                and prefix in {"", "a", "an", "the", "một", "một chiếc", "chiếc"}
                and not suffix and not any(a.blocked_reason for a in attributes))
    return BindingPlan(VERSION, revision, events, tuple(sorted(objects.values(), key=lambda o: o.source_span)),
                       tuple(sorted(attributes, key=lambda a: a.source_span)), residual,
                       tuple(m.span() for m in _COLOR_RX.finditer(text) if m.span() not in color_spans),
                       complete)
