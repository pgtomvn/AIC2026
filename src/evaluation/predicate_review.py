"""Validate manual predicate reviews without rewriting annotator decisions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


def validate_predicate_review(
    review: Mapping[str, Any], manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an exact packet match; return descriptive counts, never accuracy.

    Image bytes and the source probe must be hash-checked by the importing CLI.
    Notes are opaque data, not instructions. Semantic disagreements require a
    separate adjudication record, not silent changes to this raw export.
    """
    schema = "fourbigpigs-predicate-review-v1"  # Existing packet interchange format.
    if review.get("schema") != schema or manifest.get("schema") != schema:
        raise ValueError("review schema mismatch")
    if (type(review.get("reviewer_count")) is not int or review["reviewer_count"] != 1
            or manifest.get("reviewer_count") != 1):
        raise ValueError("expected one reviewer")
    if review.get("export_status") != "complete":
        raise ValueError("review is not marked complete")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("reviewer identifier missing")
    digest = manifest.get("source_probe_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or review.get("source_probe_sha256") != digest:
        raise ValueError("source probe revision mismatch")
    expected = manifest.get("cases")
    actual = review.get("cases")
    if not isinstance(expected, list) or not isinstance(actual, list) or not expected:
        raise ValueError("cases must be nonempty lists")
    by_id = {c["case_id"]: c for c in expected}
    actual_ids = [c["case_id"] for c in actual]
    if len(by_id) != len(expected) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("duplicate case identity")
    if set(by_id) != set(actual_ids):
        raise ValueError("missing or unexpected cases")
    counts: dict[str, Counter[str]] = {
        key: Counter() for key in ("verdict", "parse_match", "event_scope")
    }
    rows = []
    for case in actual:
        reference = by_id[case["case_id"]]
        for field in ("query_id", "scope", "image_sha256", "annotation_scope"):
            if case.get(field) != reference.get(field):
                raise ValueError(f"{case['case_id']}: {field} mismatch")
        if case.get("image_checked") is not True:
            raise ValueError("image viewing not attested")
        predicates = case.get("predicates")
        requested = [p["predicate_id"] for p in reference["predicates"]]
        if len(requested) != len(set(requested)):
            raise ValueError("duplicate predicate in manifest")
        if not isinstance(predicates, Mapping) or set(predicates) != set(requested):
            raise ValueError("predicate identities mismatch")
        for pid, record in predicates.items():
            for key, allowed in (
                ("verdict", {"support", "contradiction", "unknown"}),
                ("parse_match", {"yes", "no", "uncertain"}),
                ("event_scope", {"this_event", "other_event", "uncertain"}),
            ):
                if record.get(key) not in allowed:
                    raise ValueError(f"invalid {key}")
                counts[key][record[key]] += 1
            if not isinstance(record.get("reason"), str):
                raise ValueError("reason must be text")
            if record["verdict"] == "contradiction" and not record["reason"].strip():
                raise ValueError("contradiction needs a reason")
            rows.append({"case_id": case["case_id"], "predicate_id": pid,
                         **{k: record[k] for k in (*counts, "reason")}})
    return {"case_count": len(actual), "predicate_frame_count": len(rows),
            "unique_predicate_count": len({r['predicate_id'] for r in rows}),
            "counts": {k: dict(v) for k, v in counts.items()}, "rows": rows,
            "quality_metric_computed": False, "reviewer_count": 1}
