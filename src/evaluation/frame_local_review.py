"""Validate one blinded frame-local annotation without inferring witnesses."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from typing import Any

SATISFACTIONS = ("yes", "no", "uncertain")
VISIBILITIES = ("clear", "partly_occluded", "poor_image_quality", "uncertain")
UNASSESSED_FIELDS = (
    "full_query_relevance", "temporal_segment_relevance", "model_proposal_validity",
    "model_proposal_ownership", "available_evidence_joint_witness",
)
_REVIEW_FIELDS = {
    "schema", "packet_id", "source_input_manifest_sha256", "local_clause",
    "judgment_unit", "status", "reviewer", "reviewer_count", "exported_at",
    "unassessed", "reviews",
}
_PACKET_FIELDS = {
    "schema", "packet_id", "created_at", "source_input_manifest_sha256",
    "local_clause", "judgment_unit", "reviewer_count_planned",
    "reviewer_model_outputs_hidden", "blinding_limit", "sample_scope",
    "unassessed", "candidates",
}
_PRIVATE_FIELDS = {
    "schema", "packet_id", "source_input_manifest_sha256", "shuffle_seed", "mapping",
}
_SCOPE_FIELDS = {
    "batch_id", "video_id", "source_frame_idx", "timestamp_ms", "view_revision",
}


def _mapping(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name}: missing or unexpected fields")
    return value


def _text(value: Any, name: str, *, limit: int = 3000, blank: bool = False) -> str:
    if (not isinstance(value, str) or len(value) > limit
            or (not blank and not value.strip())):
        raise ValueError(f"{name}: expected bounded text")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name}: expected lowercase SHA-256")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    token = _text(value, name, limit=80)
    try:
        result = datetime.fromisoformat(token)
    except ValueError as error:
        raise ValueError(f"{name}: expected ISO-8601 timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name}: timezone required")
    return result


def _eight(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError(f"{name}: expected exactly eight rows")
    return value


def _unassessed(value: Any, name: str) -> None:
    _mapping(value, set(UNASSESSED_FIELDS), name)
    if any(item is not None for item in value.values()):
        raise ValueError(f"{name}: all out-of-scope judgments must remain null")


def _scope(value: Any, digest: str, name: str) -> Mapping[str, Any]:
    scope = _mapping(value, _SCOPE_FIELDS, name)
    for key in ("batch_id", "video_id"):
        _text(scope[key], f"{name}.{key}", limit=200)
    for key in ("source_frame_idx", "timestamp_ms"):
        if type(scope[key]) is not int or scope[key] < 0:
            raise ValueError(f"{name}.{key}: expected nonnegative integer")
    if scope["view_revision"] != f"original-jpeg-sha256:{digest}":
        raise ValueError(f"{name}: original-image revision mismatch")
    return scope


def validate_frame_local_review(
    review: Mapping[str, Any], packet: Mapping[str, Any],
    private_map: Mapping[str, Any], acquisition_input: Mapping[str, Any], *,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    """Return detached single-reviewer labels joined to verified frame identities.

    The importing caller must reject duplicate JSON object keys, hash the original
    acquisition-manifest bytes into ``input_manifest_sha256``, and verify image
    bytes separately. A Mapping cannot establish its original byte hash. This
    function performs no file, model, network or clock access. Notes remain opaque
    data; no annotation establishes proposal validity, an available-evidence
    witness, full-query truth, physical domain closure, or temporal relevance.
    """
    _mapping(review, _REVIEW_FIELDS, "review")
    _mapping(packet, _PACKET_FIELDS, "packet")
    _mapping(private_map, _PRIVATE_FIELDS, "private_map")
    if not isinstance(acquisition_input, Mapping):
        raise ValueError("acquisition_input must be a mapping")
    for document in (review, packet, private_map, acquisition_input):
        if "qa_only" in document or "demo" in document:
            raise ValueError("QA/demo metadata is not a human review")
    for document, expected in (
        (review, "evibind-frame-local-review-v1"),
        (packet, "evibind-frame-local-review-packet-v1"),
        (private_map, "evibind-frame-local-private-map-v1"),
        (acquisition_input, "evibind-frozen-acquisition-input-v1"),
    ):
        if document.get("schema") != expected:
            raise ValueError("schema mismatch")
    if review["status"] != "complete":
        raise ValueError("review is not complete")
    for value in (review["reviewer_count"], packet["reviewer_count_planned"]):
        if type(value) is not int or value != 1:
            raise ValueError("expected a single reviewer")
    if packet["reviewer_model_outputs_hidden"] is not True:
        raise ValueError("packet does not attest hidden model outputs")
    _text(review["reviewer"], "reviewer", limit=120, blank=True)
    for key in ("sample_scope", "blinding_limit"):
        _text(packet[key], f"packet.{key}")
    packet_id = _text(packet["packet_id"], "packet_id", limit=100)
    if not re.fullmatch(r"evibind-frame-local-[0-9a-f]{12}", packet_id):
        raise ValueError("packet_id format mismatch")
    digest = _digest(input_manifest_sha256, "input_manifest_sha256")
    for document in (review, packet, private_map):
        if document["packet_id"] != packet_id:
            raise ValueError("packet identity mismatch")
        if document["source_input_manifest_sha256"] != digest:
            raise ValueError("input manifest SHA-256 mismatch")
    _digest(private_map["shuffle_seed"], "shuffle_seed")
    clause = _text(packet["local_clause"], "local_clause")
    if review["local_clause"] != clause or acquisition_input.get("local_clause") != clause:
        raise ValueError("local clause mismatch")
    for document in (packet, review):
        if document["judgment_unit"] != "single_original_frame":
            raise ValueError("judgment unit must be single_original_frame")
        _unassessed(document["unassessed"], "unassessed")
    if acquisition_input.get("full_query_representation_complete") is not False:
        raise ValueError("acquisition must retain incomplete full-query representation")
    exported = _timestamp(review["exported_at"], "exported_at")
    created = _timestamp(packet["created_at"], "packet.created_at")
    acquired = _timestamp(acquisition_input.get("created_at"), "acquisition.created_at")
    if not acquired <= created <= exported:
        raise ValueError("acquisition, packet and export timestamps are out of order")

    candidates = _eight(packet["candidates"], "packet.candidates")
    mapping = _eight(private_map["mapping"], "private_map.mapping")
    judgments = _eight(review["reviews"], "review.reviews")
    images = _eight(acquisition_input.get("images"), "acquisition.images")
    by_rank: dict[int, Mapping[str, Any]] = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError("acquisition image must be a mapping")
        rank = image.get("candidate_rank")
        if type(rank) is not int or not 1 <= rank <= 8 or rank in by_rank:
            raise ValueError("acquisition candidate ranks must be unique 1..8")
        by_rank[rank] = image
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_ranks: set[int] = set()
    for position, (candidate, source, judgment) in enumerate(
        zip(candidates, mapping, judgments, strict=True), start=1,
    ):
        _mapping(candidate, {"candidate_id", "image", "image_sha256", "width", "height"},
                 "candidate")
        _mapping(source, {"candidate_id", "source_image", "candidate_rank",
                          "original_input_scope", "image_sha256"}, "private source")
        _mapping(judgment, {"candidate_id", "image_sha256", "satisfaction",
                            "visibility", "notes"}, "judgment")
        cid = candidate["candidate_id"]
        if not isinstance(cid, str) or not re.fullmatch(r"F-[0-9A-F]{6}", cid):
            raise ValueError("candidate_id format mismatch")
        image_digest = _digest(candidate["image_sha256"], "candidate.image_sha256")
        if cid in seen_ids or image_digest in seen_hashes:
            raise ValueError("duplicate candidate identity or image hash")
        seen_ids.add(cid)
        seen_hashes.add(image_digest)
        for other in (source, judgment):
            if other["candidate_id"] != cid or other["image_sha256"] != image_digest:
                raise ValueError("candidate identity, hash or packet order mismatch")
        if candidate["image"] != f"images/{cid}.jpg":
            raise ValueError("candidate image path mismatch")
        rank = source["candidate_rank"]
        if type(rank) is not int or rank not in by_rank or rank in seen_ranks:
            raise ValueError("private candidate ranks must be unique 1..8")
        seen_ranks.add(rank)
        image = by_rank[rank]
        for dimension in ("width", "height"):
            if type(candidate[dimension]) is not int or candidate[dimension] <= 0:
                raise ValueError("candidate dimensions must be positive integers")
        if (image.get("raw_sha256") != image_digest
                or image.get("size") != [candidate["width"], candidate["height"]]
                or source["source_image"] != image.get("image")
                or source["source_image"] != f"images/rank-{rank:02d}-original.jpg"):
            raise ValueError("private source does not match acquisition image")
        scope = _scope(source["original_input_scope"], image_digest, "private scope")
        acquired_scope = _scope(image.get("original_input_scope"), image_digest,
                                "acquisition scope")
        if dict(scope) != dict(acquired_scope):
            raise ValueError("private source scope mismatch")
        for key, allowed in (("satisfaction", SATISFACTIONS), ("visibility", VISIBILITIES)):
            if key == "visibility" and judgment[key] is None:
                continue  # Optional in the original packet; never fill it by inference.
            if not isinstance(judgment[key], str) or judgment[key] not in allowed:
                raise ValueError(f"invalid or unanswered {key}")
        _text(judgment["notes"], "notes", blank=True)
        rows.append({
            "packet_position": position, "candidate_id": cid, "candidate_rank": rank,
            "original_input_scope": dict(scope), "image_sha256": image_digest,
            **{key: judgment[key] for key in ("satisfaction", "visibility", "notes")},
        })
    counts = {}
    for key, allowed in (("satisfaction", SATISFACTIONS), ("visibility", VISIBILITIES)):
        observed = Counter(row[key] for row in rows)
        counts[key] = {value: observed[value] for value in allowed}
        if key == "visibility":
            counts[key]["unrecorded"] = observed[None]
    return {
        "schema": "evibind-frame-local-review-validated-v1", "packet_id": packet_id,
        "source_input_manifest_sha256": digest, "local_clause": clause,
        "judgment_unit": "single_original_frame", "reviewer": review["reviewer"],
        "reviewer_count": 1, "exported_at": review["exported_at"], "frame_count": 8,
        "unique_video_count": len({row["original_input_scope"]["video_id"] for row in rows}),
        "counts": counts, "rows": rows,
        "unassessed": {key: None for key in UNASSESSED_FIELDS},
        "quality_metric_computed": False, "physical_domain_completeness_inferred": False,
        "temporal_tracking_inferred": False,
    }
