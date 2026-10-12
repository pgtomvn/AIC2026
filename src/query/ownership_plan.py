"""Research-only role proposals for explicit Vietnamese person/garment phrases.

The grammar proposes text constraints, never image evidence. It preserves counts,
source offsets and unhandled text; no output is a complete-query approval.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from src.query.binding_plan import NOUNS, BindingPlan, parse_binding_query

VERSION = "evibind-ownership-grammar-v1"
_COUNTS = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5}
_GARMENTS = {n: v for n, v in NOUNS.items()
             if v[0] in {"shirt", "hat", "suit", "clothing", "scarf", "tie"}}
_NOUNS = "|".join(re.escape(n) for n in sorted(_GARMENTS, key=len, reverse=True))
_PERSON = re.compile(
    r"(?<!\w)(?P<count>\d+|một|hai|ba|bốn|năm)\s+"
    r"(?P<person>người(?:\s+(?:phụ nữ|đàn ông|nam giới))?)\s+"
    r"(?P<verb>mặc|đội)\s+(?P<garment>" + _NOUNS + r")(?!\w)", re.I
)
_BETWEEN = re.compile(r"\s+(?:(?P<pose>ngồi|đứng)\s+)?giữa\s+", re.I)
_QUANTITY_MODIFIER = re.compile(
    r"(?:hơn|dưới|ít nhất|nhiều nhất|không quá|khoảng|gần|từ|đến)\s*$", re.I
)


@dataclass(frozen=True)
class RoleGroup:
    group_id: str
    event_id: str
    source_span: tuple[int, int]
    source_text: str
    count_text: str
    mentioned_count: int
    person_variables: tuple[str, ...]
    garment_variables: tuple[str, ...]
    garment_source_variable: str
    garment_class: str
    blocked_reason: str = ""


@dataclass(frozen=True)
class RoleConstraint:
    predicate_id: str
    event_id: str
    kind: str
    variables: tuple[str, ...]
    value: str
    source_span: tuple[int, int]
    source_text: str


@dataclass(frozen=True)
class OwnershipPlan:
    version: str
    query_revision: str
    nominal: BindingPlan
    groups: tuple[RoleGroup, ...]
    constraints: tuple[RoleConstraint, ...]
    distinct_pairs: tuple[tuple[str, str], ...]
    unhandled_spans: tuple[tuple[int, int], ...]
    grammar_complete: bool = False
    semantic_validation: str = "not_human_validated"


def parse_ownership_query(text: str) -> OwnershipPlan:
    """Propose wears/color and a bounded between relation without coreference.

    Count expansion creates requested roles, not detected people or an exact
    scene count. Only an explicit singleton-between-two phrase gets three
    distinct person roles. Approximate/large counts and negated phrases stay
    blocked. Residual actions, exclusivity, event grounding and identity are
    intentionally unresolved, so full-query promotion is never enabled here.
    """
    nominal = parse_binding_query(text)
    revision = hashlib.sha256((VERSION + "\0" + text).encode()).hexdigest()
    groups: list[RoleGroup] = []
    constraints: list[RoleConstraint] = []
    covered: list[tuple[int, int]] = []
    distinct: set[tuple[str, str]] = set()

    def add(kind: str, variables: tuple[str, ...], value: str,
            span: tuple[int, int], event_id: str) -> None:
        pid = hashlib.sha256(f"{revision}:{kind}:{variables}:{span}".encode()).hexdigest()[:24]
        constraints.append(RoleConstraint(pid, event_id, kind, variables, value,
                                          span, text[slice(*span)]))

    for m in _PERSON.finditer(text):
        obj = next((o for o in nominal.objects if o.source_span == m.span("garment")), None)
        if obj is None or not any(e.event_id == obj.event_id and e.start <= m.start()
                                  for e in nominal.events.events):
            continue
        attrs = [a for a in nominal.attributes if a.variable == obj.variable and a.kind == "color"]
        count_text = m.group("count").lower()
        count = _COUNTS.get(count_text, 0) if not count_text.isdigit() else (
            int(count_text) if len(count_text) < 4 else 999
        )
        span = (m.start(), max([m.end(), *(a.source_span[1] for a in attrs)]))
        before = text[max(0, m.start()-25):m.start()]
        event = next(e for e in nominal.events.events if e.event_id == obj.event_id)
        # Conservative negatives are not silently removed from a person clause.
        clause = text[event.start:span[1]]
        blocked = ("modified_quantity" if _QUANTITY_MODIFIER.search(before)
                   else "unsupported_role_count" if not 1 <= count <= 5
                   else "unsupported_wearing_target"
                   if ((m.group("verb").lower() == "đội") != (obj.semantic_class == "hat"))
                   else "negated_or_unresolved_attribute" if any(a.blocked_reason for a in attrs)
                   or re.search(r"\b(?:không(?!\s+(?:gian|khí)\b)|chẳng|chưa)\b", clause, re.I)
                   else "")
        gid = f"g{m.start()}_{m.end()}"
        persons = tuple(f"{gid}_p{i}" for i in range(count)) if not blocked else ()
        garments = tuple(f"{gid}_c{i}" for i in range(count)) if not blocked else ()
        group = RoleGroup(gid, obj.event_id, span, text[slice(*span)], count_text, count,
                          persons, garments, obj.variable, obj.semantic_class, blocked)
        groups.append(group)
        if blocked:
            continue
        covered.append(span)
        for person, garment in zip(persons, garments):
            add("wears", (person, garment), obj.semantic_class, m.span(), obj.event_id)
            for a in attrs:
                add("color", (garment,), a.value, a.source_span, obj.event_id)
        # Explicit plural people require distinct people. The clothing roles are
        # not globally made distinct: identity must be grounded by observations.
        for i, person in enumerate(persons):
            distinct.update(tuple(sorted((person, other))) for other in persons[i+1:])

    for left, right in zip(groups, groups[1:]):
        if (left.blocked_reason or right.blocked_reason or left.event_id != right.event_id
                or len(left.person_variables) != 1 or len(right.person_variables) != 2):
            continue
        gap = _BETWEEN.fullmatch(text[left.source_span[1]:right.source_span[0]])
        if not gap:
            continue
        variables = left.person_variables + right.person_variables
        span = (left.source_span[0], right.source_span[1])
        add("between", variables, "perspective_unspecified", span, left.event_id)
        if gap.group("pose"):
            add("pose", left.person_variables, {"ngồi": "seated", "đứng": "standing"}[
                gap.group("pose").lower()], span, left.event_id)
        # A text relation does not license a 2D center proxy or a depth claim.
        covered.append((left.source_span[1], right.source_span[0]))
        for i, person in enumerate(variables):
            distinct.update(tuple(sorted((person, other))) for other in variables[i+1:])

    mask = [False] * len(text)
    for start, end in covered:
        mask[start:end] = [True] * (end-start)
    residual = tuple(m.span() for m in re.finditer(r"\S+", text)
                     if any(text[i].isalnum() and not mask[i] for i in range(*m.span())))
    return OwnershipPlan(VERSION, revision, nominal, tuple(groups), tuple(constraints),
                         tuple(sorted(distinct)), residual)
