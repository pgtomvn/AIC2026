"""Research-only source-span scoping of explicit temporal cues; no video grounding.

This deterministic audit does not repair entity parsing or infer implicit event
boundaries. Text partitions are hypotheses, not observed video events.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

VERSION = "explicit-event-scope-v1"
# Deliberately require clause boundaries. 'sau' / 'before' inside a clause can
# denote a spatial relation and must not silently become a temporal boundary.
_CUE = re.compile(
    r"(?P<boundary>^|[.;!?\n\r])\s*(?P<cue>"
    r"sau\s+đó|tiếp\s+theo|trước\s+đó|"
    r"(?:(?:đoạn\s+)?(?:clip|video|phim)|mẩu\s+tin)\s+"
    r"(?:cần\s+tìm\s+)?(?:bắt\s+đầu|kết\s+thúc)|"
    r"afterwards|subsequently|previously|then|next|"
    r"(?:the\s+)?(?:clip|video)\s+(?:begins|starts|ends))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EventSpan:
    event_id: str
    start: int
    end: int
    cue: str
    relation_to_previous: str


@dataclass(frozen=True)
class PredicateLocation:
    source_phrase: str
    occurrences: tuple[tuple[int, int], ...]
    event_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class EventScopePlan:
    source_text: str
    query_revision: str
    events: tuple[EventSpan, ...]

    def locate(self, phrase: str) -> PredicateLocation:
        """Locate exact source text, retaining repetitions instead of guessing."""
        if not isinstance(phrase, str) or not phrase.strip():
            return PredicateLocation("", (), (), "missing_source_phrase")
        # Original codepoint offsets are preserved. No accent/whitespace
        # normalization that could fabricate a precise span is performed.
        phrase = phrase.strip()
        occurrences = tuple((m.start(), m.start() + len(phrase))
                            for m in re.finditer(r"(?=" + re.escape(phrase) + r")", self.source_text))
        if not occurrences:
            return PredicateLocation(phrase, (), (), "source_phrase_not_exact")
        matches = [tuple(e.event_id for e in self.events if e.start <= a and b <= e.end)
                   for a, b in occurrences]
        event_ids = tuple(sorted({eid for group in matches for eid in group}))
        reason = ("crosses_event_boundary" if any(not m for m in matches)
                  else "ambiguous_across_events" if len(event_ids) != 1
                  else "explicit_partition_only")
        return PredicateLocation(phrase, occurrences, event_ids, reason)


def scope_explicit_events(text: str) -> EventScopePlan:
    """Partition on explicit clause-initial cues; preserve every source character."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("query text must be nonempty")
    markers = [(m.start("cue"), m.group("cue")) for m in _CUE.finditer(text)]
    cuts = {0: ""}
    cuts.update(markers)
    positions = sorted(cuts)
    events = []
    for i, start in enumerate(positions):
        cue = cuts[start]
        low = cue.lower()
        relation = ("before_previous" if low in {"trước đó", "previously"}
                    else "after_previous" if low in {"sau đó", "tiếp theo", "afterwards",
                                                       "subsequently", "then", "next"}
                    else "clip_start" if re.search(r"bắt\s+đầu|begins|starts", low)
                    else "clip_end" if re.search(r"kết\s+thúc|ends", low)
                    else "unspecified")
        events.append(EventSpan(f"e{i}", start, positions[i+1] if i+1 < len(positions)
                                else len(text), cue, relation))
    revision = hashlib.sha256((VERSION + "\0" + text).encode("utf-8")).hexdigest()
    return EventScopePlan(text, revision, tuple(events))
