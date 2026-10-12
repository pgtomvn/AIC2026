"""Conservative English temporal parsing and evidence-bound local release gate."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.evaluation.interactive_pilot import local_release_gate

PARSER_VERSION = "english-ordered-events-v2"
PARAMETERS = {"weight": .6, "candidate_videos": 12, "max_steps": 6}
_VERB = re.compile(r"\b(?:is|are|was|were|opens?|closes?|walks?|runs?|enters?|leaves?|"
                   r"sits?|stands?|picks?|puts?|turns?|shows?|appears?|cuts?|pours?|holds?|"
                   r"rides?|drives?|speaks?|talks?|starts?|ends?|looks?|moves?|takes?|gives?|"
                   r"opening|closing|walking|running|entering|leaving|sitting|standing|"
                   r"picking|putting|turning|showing|appearing|cutting|pouring|holding|"
                   r"riding|driving|speaking|talking|starting|ending|looking|moving|taking|giving|"
                   r"opened|closed|walked|entered|left|sat|stood|picked|turned|showed|appeared|"
                   r"poured|held|rode|drove|spoke|talked|started|ended|looked|moved|took|gave)\b", re.I)


def load_release_record(path: Path, root: Path) -> dict[str, Any] | None:
    """Load bounded local evidence; reject missing/modified or unconfirmed labels."""
    try:
        if not path.is_file() or path.stat().st_size > 1024*1024:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        record.pop("verified_labels_sha256", None)
        labels = record["labels"]
        evidence = (root / labels["path"]).resolve()
        evidence.relative_to((root / "outputs" / "evaluation").resolve())
        if evidence.stat().st_size > 1024*1024:
            return None
        blob = evidence.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        values = json.loads(blob)
        if digest != labels["sha256"] or not isinstance(values, list) or len(values) < 6:
            return None
        ids = set()
        for row in values:
            if (not isinstance(row, Mapping) or row.get("event_source") != "user"
                    or row.get("confirmed") is not True or not isinstance(row.get("query_id"), str)
                    or not isinstance(row.get("steps"), list) or not 2 <= len(row["steps"]) <= 6
                    or any(not isinstance(s, str) or not s.strip() for s in row["steps"])
                    or type(row.get("target")) is not int or not 0 <= row["target"] < len(row["steps"])):
                return None
            ids.add(row["query_id"])
        if len(ids) != len(values) or labels.get("ordered_event_count") != len(values):
            return None
        record["verified_labels_sha256"] = digest
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_english_steps(text: str) -> dict[str, Any]:
    """Recognize explicit orders only. Unsupported grammar returns no guessed steps."""
    result: dict[str, Any] = {"parser_version": PARSER_VERSION, "steps": [],
                             "target": None, "reason_code": "ambiguous_order"}
    if not isinstance(text, str) or not 0 < len(text) <= 8192:
        return result
    masked = re.sub(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]*\x27(?!\w)',
                    lambda m: ' ' * len(m[0]), text)
    if re.search(r"\b(?:not\s+(?:before|after)|until|meanwhile|simultaneously|if|unless)\b", masked, re.I):
        return result
    # Repetition idioms do not identify two separable event clauses. Do not
    # manufacture a reversed sequence from nouns following "after".
    if re.search(r"\b(?:one\s+after\s+(?:another|the\s+other)|(?:day|year|time)\s+after\s+(?:day|year|time))\b", masked, re.I):
        return result
    connectors = list(re.finditer(r"\b(?:and\s+then|then|next|finally|afterwards|subsequently)\b", masked, re.I))
    binary = list(re.finditer(r"\b(?:before|after|prior\s+to)\b", masked, re.I))
    steps: list[str] = []
    if connectors and not binary:
        start = 0
        for marker in connectors:
            steps.append(text[start:marker.start()])
            start = marker.end()
        steps.append(text[start:])
        steps[0] = re.sub(r"^\s*(?:first|initially)\b[,\s]*", "", steps[0], flags=re.I)
    elif len(binary) == 1 and not connectors:
        marker = binary[0]
        left, right = text[:marker.start()].strip(), text[marker.end():].strip()
        if not left:
            comma = masked.find(',', marker.end())
            if comma < 0:
                return result
            earlier_clause, main_clause = text[marker.end():comma], text[comma+1:]
            steps = [main_clause, earlier_clause] if marker[0].lower() in {"before", "prior to"} else [earlier_clause, main_clause]
        else:
            steps = [left, right] if marker[0].lower() in {"before", "prior to"} else [right, left]
    steps = [step.strip(' ,.;:') for step in steps]
    if not 2 <= len(steps) <= 6 or any(not _VERB.search(step) for step in steps):
        return result
    if len(set(s.casefold() for s in steps)) != len(steps):
        return result
    return {**result, "steps": steps, "target": len(steps)-1, "reason_code": "explicit_english_order"}


def release_allowed(record: Any, *, retrieval_fingerprint: str, cache_fingerprint: str,
                    source_fingerprint: str) -> bool:
    """Recompute the gate; a copied boolean or configuration toggle is not evidence."""
    if not isinstance(record, Mapping):
        return False
    expected = {"schema_version": 1, "feature": "temporal_sequence", "language": "en",
                "parser_version": PARSER_VERSION, "parameters": PARAMETERS,
                "retrieval_fingerprint": retrieval_fingerprint,
                "cache_fingerprint": cache_fingerprint, "source_fingerprint": source_fingerprint}
    if any(not value or record.get(key) != value for key, value in expected.items()):
        return False
    labels = record.get("labels", {})
    if not isinstance(labels, Mapping) or labels.get("source") != "user_confirmed":
        return False
    count = labels.get("ordered_event_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 6:
        return False
    if not re.fullmatch(r"[a-f0-9]{64}", str(labels.get("sha256", ""))):
        return False
    if record.get("verified_labels_sha256") != labels["sha256"]:
        return False
    baseline, trial = record.get("baseline"), record.get("trial")
    if not isinstance(baseline, Mapping) or not isinstance(trial, Mapping):
        return False
    return local_release_gate(baseline, trial, feature="temporal_sequence", data_kind="ordered_events",
        parameters_locked=record.get("parameters_locked") is True,
        outside_scope_invariant=record.get("outside_scope_invariant") is True)["automatic_activation_allowed"]
