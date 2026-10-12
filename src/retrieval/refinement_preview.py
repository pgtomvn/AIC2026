"""Deterministic English edit commands. Preview only; never searches or guesses."""
from __future__ import annotations

import re
from typing import Any

VERSION = "explicit-edit-v1"


def preview_edit(query: str, instruction: str) -> dict[str, Any]:
    """Return an unchanged draft on ambiguous/missing edits, including duplicates."""
    if not all(isinstance(v, str) and 0 < len(v.strip()) <= 8192 for v in (query, instruction)):
        raise ValueError("bounded_query_and_instruction_required")
    result: dict[str, Any] = {"version": VERSION, "original_text": query,
                             "draft_text": query, "applied": False, "changes": [],
                             "reason_code": "unsupported_edit"}
    command = instruction.strip()
    operation = "replace"
    if re.match(r"replace\s+", command, re.I):
        masked = re.sub(r'"[^"\n]*"', lambda m: ' '*len(m[0]), command)
        separators = list(re.finditer(r"\bwith\b", masked, re.I))
        if len(separators) != 1 or command.count('"') % 2:
            return {**result, "reason_code": "ambiguous_edit"}
        separator = separators[0]
        old = re.sub(r"^replace\s+", "", command[:separator.start()], flags=re.I).strip().strip('"')
        new = command[separator.end():].strip().strip('"')
        if not old or not new:
            return result
    else:
        match = re.fullmatch(r"(add|remove)\s+(.+)", command, re.I)
        if not match:
            return result
        operation, text = match.groups()
        operation, text = operation.lower(), text.strip().strip('"')
        old, new = (text, "") if operation == "remove" else ("", text)
    if not (old or new):
        return result
    if operation == "add":
        if re.search(r"(?<!\w)" + re.escape(new) + r"(?!\w)", query, re.I):
            return {**result, "reason_code": "already_present"}
        draft = query.rstrip().rstrip('.') + ". " + new
    else:
        occurrences = list(re.finditer(r"(?<!\w)" + re.escape(old) + r"(?!\w)", query, re.I))
        if len(occurrences) != 1:
            return {**result, "reason_code": "ambiguous_edit" if occurrences else "phrase_not_found"}
        occurrence = occurrences[0]
        draft = query[:occurrence.start()] + new + query[occurrence.end():]
        if operation == "remove":
            draft = re.sub(r"\s+", " ", draft).strip(' .;,')
    if not draft.strip() or len(draft) > 8192:
        return {**result, "reason_code": "invalid_draft"}
    return {**result, "draft_text": draft, "applied": draft != query,
            "reason_code": "preview_ready", "changes": [{"operation": operation, "old": old, "new": new}]}
