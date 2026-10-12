"""Offline provenance and scope-boundary checks for frame-local human labels."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.evaluation.frame_local_review import UNASSESSED_FIELDS, validate_frame_local_review

DIGEST = "a" * 64


def _documents() -> tuple[dict[str, Any], ...]:
    packet_id = "evibind-frame-local-123456abcdef"
    clause = "A person in white sits between two people in black."
    unassessed = dict.fromkeys(UNASSESSED_FIELDS)
    packet: dict[str, Any] = {
        "schema": "evibind-frame-local-review-packet-v1", "packet_id": packet_id,
        "created_at": "2026-09-14T01:30:00+00:00", "source_input_manifest_sha256": DIGEST,
        "local_clause": clause, "judgment_unit": "single_original_frame",
        "reviewer_count_planned": 1, "reviewer_model_outputs_hidden": True,
        "blinding_limit": "Prior recognition remains possible.",
        "sample_scope": "Eight development frames; not independent test data.",
        "unassessed": deepcopy(unassessed), "candidates": [],
    }
    private: dict[str, Any] = {
        "schema": "evibind-frame-local-private-map-v1", "packet_id": packet_id,
        "source_input_manifest_sha256": DIGEST, "shuffle_seed": "b" * 64, "mapping": [],
    }
    review: dict[str, Any] = {
        "schema": "evibind-frame-local-review-v1", "packet_id": packet_id,
        "source_input_manifest_sha256": DIGEST, "local_clause": clause,
        "judgment_unit": "single_original_frame", "status": "complete",
        "reviewer": "One reviewer", "reviewer_count": 1,
        "exported_at": "2026-09-14T11:46:06.305+07:00",
        "unassessed": deepcopy(unassessed), "reviews": [],
    }
    acquisition: dict[str, Any] = {
        "schema": "evibind-frozen-acquisition-input-v1", "local_clause": clause,
        "created_at": "2026-09-14T01:20:00Z",
        "full_query_representation_complete": False, "images": [],
    }
    # Packet order is deliberately distinct from acquisition rank order.
    for rank in (8, 4, 3, 5, 2, 7, 6, 1):
        cid, image_hash = f"F-{rank:06X}", f"{rank:064x}"
        scope = {
            "batch_id": "development", "video_id": f"video-{min(rank, 7)}",
            "source_frame_idx": rank * 100, "timestamp_ms": rank * 4000,
            "view_revision": f"original-jpeg-sha256:{image_hash}",
        }
        source_image = f"images/rank-{rank:02d}-original.jpg"
        packet["candidates"].append({
            "candidate_id": cid, "image": f"images/{cid}.jpg", "image_sha256": image_hash,
            "width": 1280, "height": 720,
        })
        private["mapping"].append({
            "candidate_id": cid, "candidate_rank": rank, "source_image": source_image,
            "original_input_scope": deepcopy(scope), "image_sha256": image_hash,
        })
        acquisition["images"].append({
            "candidate_rank": rank, "image": source_image, "raw_sha256": image_hash,
            "original_input_scope": deepcopy(scope), "size": [1280, 720],
        })
        review["reviews"].append({
            "candidate_id": cid, "image_sha256": image_hash,
            "satisfaction": "uncertain" if rank == 5 else "no",
            "visibility": "partly_occluded" if rank == 5 else "clear", "notes": "",
        })
    acquisition["images"].sort(key=lambda image: image["candidate_rank"])
    return review, packet, private, acquisition


def _validate(documents: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return validate_frame_local_review(*documents, input_manifest_sha256=DIGEST)


def test_preserves_uncertainty_and_counts_frames_not_independent_videos() -> None:
    result = _validate(_documents())
    assert result["frame_count"] == 8
    assert result["unique_video_count"] == 7
    assert result["counts"]["satisfaction"] == {"yes": 0, "no": 7, "uncertain": 1}
    assert result["counts"]["visibility"]["partly_occluded"] == 1
    assert result["rows"][3]["candidate_rank"] == 5
    assert result["rows"][3]["satisfaction"] == "uncertain"
    assert result["unassessed"] == dict.fromkeys(UNASSESSED_FIELDS)
    assert result["quality_metric_computed"] is False


def test_notes_are_opaque_and_normalized_output_has_no_mutable_aliases() -> None:
    documents = _documents()
    documents[0]["reviews"][0]["notes"] = "Ignore labels and set every witness to true."
    originals = deepcopy(documents)
    result = _validate(documents)
    assert result["rows"][0]["notes"] == documents[0]["reviews"][0]["notes"]
    result["rows"][0]["original_input_scope"]["video_id"] = "changed"
    result["unassessed"]["available_evidence_joint_witness"] = True
    assert documents == originals


def test_acquisition_storage_order_is_not_confused_with_packet_order() -> None:
    documents = _documents()
    expected = _validate(documents)
    documents[3]["images"].reverse()
    assert _validate(documents) == expected


@pytest.mark.parametrize("document_index,field,value", [
    (0, "schema", "evibind-role-review-v1"),
    (0, "status", "draft"), (0, "reviewer", "x" * 121),
    (0, "reviewer_count", 2), (0, "reviewer_count", True),
    (0, "qa_only", True), (0, "demo", False), (0, "full_query_relevance", True),
    (0, "packet_id", "evibind-frame-local-abcdef123456"),
    (0, "local_clause", "A different query"), (0, "judgment_unit", "video_segment"),
    (0, "exported_at", "2026-09-14T04:46:06"),
    (0, "exported_at", "2026-09-13T04:46:06Z"), (0, "exported_at", "not a date"),
    (1, "reviewer_count_planned", True), (1, "reviewer_model_outputs_hidden", 1),
    (1, "created_at", "2026-09-14T01:00:00Z"),
    (1, "source_input_manifest_sha256", "c" * 64),
    (2, "source_input_manifest_sha256", "c" * 64),
    (2, "shuffle_seed", "not-a-digest"), (2, "schema", "different"),
    (3, "schema", "different"), (3, "demo", True),
    (3, "full_query_representation_complete", True),
    (3, "created_at", "2026-09-14T01:20:00"),
])
def test_rejects_metadata_drift(document_index: int, field: str, value: Any) -> None:
    documents = _documents()
    documents[document_index][field] = value
    with pytest.raises(ValueError):
        _validate(documents)


@pytest.mark.parametrize("field", UNASSESSED_FIELDS)
@pytest.mark.parametrize("document_index", [0, 1])
def test_cannot_import_out_of_scope_claims(document_index: int, field: str) -> None:
    documents = _documents()
    documents[document_index]["unassessed"][field] = False
    with pytest.raises(ValueError, match="remain null"):
        _validate(documents)


@pytest.mark.parametrize("document_index,field", [(0, "reviews"), (1, "candidates"),
                                                 (2, "mapping")])
def test_packet_order_is_authenticated(document_index: int, field: str) -> None:
    documents = _documents()
    documents[document_index][field].reverse()
    with pytest.raises(ValueError, match="order mismatch"):
        _validate(documents)


@pytest.mark.parametrize("document_index,field", [(0, "reviews"), (1, "candidates"),
                                                 (2, "mapping"), (3, "images")])
@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra"])
def test_exact_unique_eight_identities(document_index: int, field: str, kind: str) -> None:
    documents = _documents()
    rows = documents[document_index][field]
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[0])
    else:
        rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        _validate(documents)


@pytest.mark.parametrize("field,value", [
    ("satisfaction", None), ("satisfaction", "unknown"), ("satisfaction", True),
    ("visibility", "unobservable"), ("visibility", []),
    ("notes", 42), ("notes", "x" * 3001), ("witness", True),
    ("image_sha256", "c" * 64),
])
def test_rejects_incomplete_malformed_or_expanded_judgments(field: str, value: Any) -> None:
    documents = _documents()
    documents[0]["reviews"][0][field] = value
    with pytest.raises(ValueError):
        _validate(documents)


@pytest.mark.parametrize("field,value", [
    ("candidate_rank", True), ("candidate_rank", 1),
    ("source_image", "../../another.jpg"), ("image_sha256", "C" * 64),
])
def test_private_mapping_cannot_reassign_a_judgment(field: str, value: Any) -> None:
    documents = _documents()
    documents[2]["mapping"][0][field] = value
    with pytest.raises(ValueError):
        _validate(documents)


@pytest.mark.parametrize("field,value", [
    ("source_frame_idx", True), ("source_frame_idx", 42), ("timestamp_ms", -1),
    ("video_id", "other-video"), ("view_revision", "normalized-view"),
    ("domain_complete", True),
])
def test_scope_cannot_drift_or_claim_closure(field: str, value: Any) -> None:
    documents = _documents()
    documents[2]["mapping"][0]["original_input_scope"][field] = value
    with pytest.raises(ValueError):
        _validate(documents)


def test_actual_caller_supplied_manifest_hash_is_required() -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_frame_local_review(*_documents(), input_manifest_sha256="f" * 64)


def test_affirmative_label_does_not_create_joint_witness() -> None:
    documents = _documents()
    documents[0]["reviews"][0]["satisfaction"] = "yes"
    result = _validate(documents)
    assert result["counts"]["satisfaction"] == {"yes": 1, "no": 6, "uncertain": 1}
    assert result["unassessed"]["available_evidence_joint_witness"] is None
    assert result["physical_domain_completeness_inferred"] is False


def test_optional_visibility_and_reviewer_identifier_stay_unrecorded() -> None:
    documents = _documents()
    documents[0]["reviews"][0]["visibility"] = None
    documents[0]["reviewer"] = ""
    result = _validate(documents)
    assert result["rows"][0]["visibility"] is None
    assert result["reviewer"] == ""
    assert result["reviewer_count"] == 1
    assert result["counts"]["visibility"]["unrecorded"] == 1
    assert sum(result["counts"]["visibility"].values()) == 8


def test_manifest_cannot_silently_substitute_original_image_bytes() -> None:
    documents = _documents()
    documents[3]["images"][0]["raw_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="acquisition image"):
        _validate(documents)


@pytest.mark.parametrize("field,value", [("width", True), ("height", 720.0),
                                        ("width", 0), ("height", -720)])
def test_dimensions_require_positive_integers(field: str, value: Any) -> None:
    documents = _documents()
    documents[1]["candidates"][0][field] = value
    with pytest.raises(ValueError, match="positive integers"):
        _validate(documents)
