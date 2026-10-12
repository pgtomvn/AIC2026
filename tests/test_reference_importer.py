"""CPU-only tests for streaming organizer reference-pack ingestion."""

from __future__ import annotations

import io
import json
import unicodedata
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.ingest.artifacts import ArtifactRepository
from src.ingest.ledger import JobLedger
from src.ingest.reference_importer import (
    CanonicalReferenceStats,
    OrganizerReferenceImporter,
    ReferenceStats,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    return path


def _npy(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _frame_map(rows: list[tuple[int, str, str, int]]) -> bytes:
    lines = ["n,pts_time,fps,frame_idx"]
    lines.extend(f"{ordinal},{pts},{fps},{source}" for ordinal, pts, fps, source in rows)
    return ("\n".join(lines) + "\n").encode()


@dataclass(frozen=True)
class ReferencePack:
    importer: OrganizerReferenceImporter
    vectors: dict[str, np.ndarray]
    images: dict[bytes, np.ndarray]


@pytest.fixture
def reference_pack(tmp_path: Path) -> ReferencePack:
    vectors = {
        "L99_V001": np.asarray([[1, 0, 0], [0, 2, 0]], dtype=np.float16),
        "L99_V002": np.asarray([[0, 0, 3]], dtype=np.float16),
    }
    map_archive = _write_zip(
        tmp_path / "maps.zip",
        {
            "maps/L99_V001.csv": _frame_map(
                [(1, "0.000", "30000/1001", 0), (2, "1.234", "30000/1001", 37)]
            ),
            "maps/L99_V002.csv": _frame_map([(1, "2.500", "25", 62)]),
        },
    )
    feature_archive = _write_zip(
        tmp_path / "features.zip",
        {f"features/{video_id}.npy": _npy(array) for video_id, array in vectors.items()},
    )
    image_vectors = {
        b"jpeg-v1-ordinal-1": vectors["L99_V001"][0],
        b"jpeg-v1-ordinal-2": vectors["L99_V001"][1],
        b"jpeg-v2-ordinal-1": vectors["L99_V002"][0],
    }
    keyframe_archive = _write_zip(
        tmp_path / "keyframes.zip",
        {
            "keyframes/L99_V001/001.jpg": b"jpeg-v1-ordinal-1",
            "keyframes/L99_V001/002.jpg": b"jpeg-v1-ordinal-2",
            "keyframes/L99_V002/001.jpg": b"jpeg-v2-ordinal-1",
        },
    )
    object_archive = _write_zip(
        tmp_path / "objects.zip",
        {
            "objects/L99_V001/001.json": json.dumps(
                {
                    "detection_scores": [0.90, 0.50, 0.49],
                    "detection_class_names": ["Người", "Xe đạp", "Bỏ qua"],
                    "detection_class_entities": ["Person", "Xe đạp", "Ignored"],
                },
                ensure_ascii=False,
            ).encode(),
            "objects/L99_V001/002.json": json.dumps(
                {
                    "detection_scores": [0.49],
                    "detection_class_names": ["Không lập chỉ mục"],
                },
                ensure_ascii=False,
            ).encode(),
            "objects/L99_V002/001.json": json.dumps(
                {
                    "detection_scores": [0.75],
                    "detection_class_names": ["Cà phê"],
                    "detection_class_entities": ["Coffee"],
                },
                ensure_ascii=False,
            ).encode(),
        },
    )
    decomposed_title = unicodedata.normalize("NFD", "Cà phê")
    metadata_archive = _write_zip(
        tmp_path / "metadata.zip",
        {
            "metadata/L99_V001.json": json.dumps(
                {
                    "title": decomposed_title,
                    "description": "  Quán   đông  ",
                    "keywords": ["đỏ", "buổi sáng"],
                },
                ensure_ascii=False,
            ).encode(),
            "metadata/L99_V002.json": json.dumps(
                {
                    "title": "Cycling",
                    "description": "A rider passes",
                    "keywords": "bicycle",
                }
            ).encode(),
        },
    )
    importer = OrganizerReferenceImporter(
        map_archive=map_archive,
        feature_archive=feature_archive,
        keyframe_archives=[keyframe_archive],
        object_archive=object_archive,
        metadata_archive=metadata_archive,
        batch_id="fixture-b1",
        object_confidence=0.50,
    )
    return ReferencePack(importer=importer, vectors=vectors, images=image_vectors)


class FakeVectorStore:
    def __init__(self, *, count_override: int | None = None) -> None:
        self.create_calls: list[bool] = []
        self.upserts: list[tuple[list[str], np.ndarray, list[dict[str, Any]]]] = []
        self.count_override = count_override

    def create_collection(self, *, recreate: bool = False) -> None:
        self.create_calls.append(recreate)

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        self.upserts.append(
            (list(ids), np.asarray(vectors).copy(), [dict(item) for item in payloads])
        )

    def count(self, filters: dict[str, Any] | None = None) -> int:
        if self.count_override is not None:
            return self.count_override
        payloads = [payload for _, _, batch in self.upserts for payload in batch]
        if filters:
            payloads = [
                payload
                for payload in payloads
                if all(
                    payload.get(key) in value if isinstance(value, list) else payload.get(key) == value
                    for key, value in filters.items()
                )
            ]
        return len(payloads)

    def iter_payloads(self, filters: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        for _, _, payloads in self.upserts:
            for payload in payloads:
                if not filters or all(payload.get(key) == value for key, value in filters.items()):
                    yield dict(payload)


class FakeTextStore:
    def __init__(self) -> None:
        self.create_calls = 0
        self.batches: list[list[dict[str, Any]]] = []

    @property
    def documents(self) -> list[dict[str, Any]]:
        return [document for batch in self.batches for document in batch]

    def create_index(self) -> None:
        self.create_calls += 1

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        self.batches.append([dict(document) for document in documents])


def test_validate_confirms_all_reference_content_is_aligned(
    reference_pack: ReferencePack,
) -> None:
    stats = reference_pack.importer.validate(expected_videos=2, expected_frames=3)

    assert stats == ReferenceStats(
        videos=2,
        frames=3,
        feature_rows=3,
        object_documents=3,
        metadata_documents=2,
    )


def test_import_vectors_streams_per_video_with_canonical_source_mapping(
    reference_pack: ReferencePack,
) -> None:
    store = FakeVectorStore()

    stats = reference_pack.importer.import_vectors(
        store, recreate=True, expected_videos=2, expected_frames=3
    )

    assert stats == ReferenceStats(2, 3, 3, 0, 0)
    assert store.create_calls == [True]
    assert len(store.upserts) == 2

    first_ids, first_vectors, first_payloads = store.upserts[0]
    assert first_ids == [
        "fixture-b1:L99_V001:ordinal:1",
        "fixture-b1:L99_V001:ordinal:2",
    ]
    np.testing.assert_array_equal(first_vectors, reference_pack.vectors["L99_V001"])
    assert first_payloads[1] == {
        "video_id": "L99_V001",
        "keyframe_ordinal": 2,
        "source_frame_idx": 37,
        "timestamp_ms": 1234,
        "fps_num": 30000,
        "fps_den": 1001,
        "shot_id": "reference:2",
        "image_uri": first_payloads[1]["image_uri"],
        "batch_id": "fixture-b1",
        "feature_source": "organizer-clip-32",
    }
    assert first_payloads[1]["image_uri"].startswith("zip://")
    assert first_payloads[1]["image_uri"].endswith("!/keyframes/L99_V001/002.jpg")

    second_ids, second_vectors, second_payloads = store.upserts[1]
    assert second_ids == ["fixture-b1:L99_V002:ordinal:1"]
    np.testing.assert_array_equal(second_vectors, reference_pack.vectors["L99_V002"])
    assert second_payloads[0]["keyframe_ordinal"] == 1
    assert second_payloads[0]["source_frame_idx"] == 62
    assert second_payloads[0]["timestamp_ms"] == 2500


def test_import_vectors_rejects_vector_store_count_mismatch(
    reference_pack: ReferencePack,
) -> None:
    with pytest.raises(ValueError, match="vector index count 2 does not match 3"):
        reference_pack.importer.import_vectors(FakeVectorStore(count_override=2))


def test_bounded_import_only_writes_selected_video(reference_pack: ReferencePack) -> None:
    store = FakeVectorStore()
    seen = []
    stats = reference_pack.importer.import_vectors(
        store, video_ids=["L99_V002"], expected_videos=1, expected_frames=1,
        before_video=seen.append,
    )
    assert stats.frames == 1
    assert seen == ["L99_V002"]
    assert store.upserts[0][0] == ["fixture-b1:L99_V002:ordinal:1"]
    other = FakeVectorStore()
    with pytest.raises(ValueError, match="subset"):
        reference_pack.importer.import_vectors(other, video_ids=["not-in-map"])
    assert other.create_calls == []


def test_verify_existing_vectors_checks_every_payload_identity(
    reference_pack: ReferencePack,
) -> None:
    store = FakeVectorStore()
    reference_pack.importer.import_vectors(store)

    stats = reference_pack.importer.verify_vector_store(store, expected_videos=2, expected_frames=3)

    assert stats == ReferenceStats(2, 3, 3, 0, 0)
    store.upserts[0][2][0]["source_frame_idx"] = 999
    with pytest.raises(ValueError, match="identities do not match"):
        reference_pack.importer.verify_vector_store(store)


def test_import_canonical_frames_wires_neighbors_mappings_and_batch_activation(
    reference_pack: ReferencePack, tmp_path: Path
) -> None:
    repository = ArtifactRepository(tmp_path / "state" / "artifacts.sqlite3")
    ledger = JobLedger(tmp_path / "state" / "jobs.sqlite3")
    repository.initialize()
    ledger.initialize()

    stats = reference_pack.importer.import_canonical_frames(
        repository, ledger=ledger, vectors_ready=True, competition="aic25"
    )

    assert stats == CanonicalReferenceStats(videos=2, mapping_rows=3, canonical_frames=3)
    assert repository.frame_index_mappings("fixture-b1") == {
        "L99_V001": {1: 0, 2: 37},
        "L99_V002": {1: 62},
    }
    assert [
        frame.source_frame_idx for frame in repository.neighbors("fixture-b1", "L99_V001", 1_000)
    ] == [0, 37]
    assert ledger.is_batch_active("fixture-b1")
    assert ledger.retrieval_generation("fixture-b1") == 1


def test_import_vectors_preserves_duplicate_source_mapping_rows(tmp_path: Path) -> None:
    maps = _write_zip(
        tmp_path / "maps.zip",
        {"maps/L99_V001.csv": _frame_map([(1, "19.20", "25", 480), (2, "19.24", "25", 480)])},
    )
    features = _write_zip(
        tmp_path / "features.zip",
        {"features/L99_V001.npy": _npy(np.ones((2, 3), dtype=np.float16))},
    )
    store = FakeVectorStore()

    OrganizerReferenceImporter(map_archive=maps, feature_archive=features).import_vectors(store)

    ids, _, payloads = store.upserts[0]
    assert ids == [
        "aic25-b1-reference:L99_V001:ordinal:1",
        "aic25-b1-reference:L99_V001:ordinal:2",
    ]
    assert [payload["source_frame_idx"] for payload in payloads] == [480, 480]

    repository = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    repository.initialize()
    importer = OrganizerReferenceImporter(map_archive=maps, feature_archive=features)
    canonical = importer.import_canonical_frames(repository)
    assert canonical.canonical_frames == 1
    assert repository.frame_index_mappings("aic25-b1-reference") == {"L99_V001": {1: 480, 2: 480}}


def test_b02_ordinal_import_keeps_equal_pts_and_source_rows(tmp_path: Path) -> None:
    rows = {
        "M08_V001": [(1, "190.04", "25", 4751), (2, "190.04", "25", 4751)],
        "N001-V001": [(1, "2.000", "25", 50), (2, "2.520", "60", 50)],
        "S01-V001": [(1, "1.2345", "30000/1001", 31)],
    }
    maps = _write_zip(
        tmp_path / "maps.zip",
        {f"maps/{video_id}.csv": _frame_map(values) for video_id, values in rows.items()},
    )
    features = _write_zip(
        tmp_path / "features.zip",
        {
            f"features/{video_id}.npy": _npy(np.ones((len(values), 3), dtype=np.float16))
            for video_id, values in rows.items()
        },
    )
    keyframes = _write_zip(
        tmp_path / "keyframes.zip",
        {
            f"keyframes/{video_id}/{ordinal:03d}.jpg": f"{video_id}:{ordinal}".encode()
            for video_id, values in rows.items()
            for ordinal, *_ in values
        },
    )
    object_archives = [
        _write_zip(
            tmp_path / f"objects-{video_id}.zip",
            {
                f"{video_id}/{ordinal:03d}.json": json.dumps(
                    {"detection_scores": [0.9], "detection_class_names": ["car"]}
                ).encode()
                for ordinal, *_ in values
            },
        )
        for video_id, values in rows.items()
    ]
    media = _write_zip(
        tmp_path / "media.zip",
        {
            "media-info/M08_V001.json": b"{}",
            "media-info/S01_V001.json": b"{}",
        },
    )
    importer = OrganizerReferenceImporter(
        map_archive=maps,
        feature_archive=features,
        keyframe_archives=(keyframes,),
        object_archives=object_archives,
        metadata_archive=media,
        metadata_policy="partial",
        identity_mode="ordinal",
        batch_id="aic26-b2",
    )
    assert importer.validate(expected_videos=3, expected_frames=5) == ReferenceStats(
        3, 5, 5, 5, 2
    )
    repository = ArtifactRepository(tmp_path / "artifacts.sqlite3")
    repository.initialize()
    assert importer.import_canonical_frames(repository) == CanonicalReferenceStats(3, 5, 5)
    assert repository.get_ordinal_frame("aic26-b2", "M08_V001", 1).timestamp_ms == 190_040
    assert repository.get_ordinal_frame("aic26-b2", "M08_V001", 2).timestamp_ms == 190_040
    assert repository.get_ordinal_frame("aic26-b2", "M08_V001", 1).image_uri != (
        repository.get_ordinal_frame("aic26-b2", "M08_V001", 2).image_uri
    )
    assert [
        frame.timestamp_ms
        for frame in repository.iter_ordinal_frames(batch_id="aic26-b2", video_id="N001-V001")
    ] == [2_000, 2_520]
    assert repository.get_frame("aic26-b2", "M08_V001", 4751) is None
    assert repository.batch_counts("aic26-b2").frame_count == 5
    exact = repository.get_ordinal_frame("aic26-b2", "S01-V001", 1)
    assert (exact.pts, exact.time_base, exact.timestamp_ms) == (2469, "1/2000", 1234)

    store = FakeVectorStore()
    importer.import_vectors(store, expected_videos=3, expected_frames=5)
    n_payloads = next(payloads for ids, _, payloads in store.upserts if "N001-V001" in ids[0])
    assert [(item["keyframe_ordinal"], item["source_frame_idx"], item["timestamp_ms"])
            for item in n_payloads] == [(1, 50, 2_000), (2, 50, 2_520)]
    assert all(item["trake_eligible"] is False for item in n_payloads)
    m_payloads = next(payloads for ids, _, payloads in store.upserts if "M08_V001" in ids[0])
    assert all(item["trake_eligible"] is False for item in m_payloads)
    s_payloads = next(payloads for ids, _, payloads in store.upserts if "S01-V001" in ids[0])
    assert s_payloads[0]["trake_eligible"] is True
    assert importer.verify_vector_store(store).frames == 5

    object_store = FakeTextStore()
    assert importer.import_objects(object_store) == 5
    m_objects = [
        item for item in object_store.documents if item["video_id"] == "M08_V001"
    ]
    assert [item["keyframe_ordinal"] for item in m_objects] == [1, 2]
    assert [item["source_frame_idx"] for item in m_objects] == [4751, 4751]
    assert importer.import_metadata(FakeTextStore()) == 0


def test_validate_rejects_map_and_feature_video_mismatch(tmp_path: Path) -> None:
    maps = _write_zip(
        tmp_path / "maps.zip",
        {"maps/L99_V001.csv": _frame_map([(1, "0", "25", 0)])},
    )
    features = _write_zip(
        tmp_path / "features.zip",
        {"features/L99_V002.npy": _npy(np.ones((1, 2), dtype=np.float16))},
    )

    importer = OrganizerReferenceImporter(map_archive=maps, feature_archive=features)

    with pytest.raises(
        ValueError,
        match=r"missing features=\['L99_V001'\], missing maps=\['L99_V002'\]",
    ):
        importer.validate()


def test_validate_rejects_feature_row_and_ordinal_mismatches(tmp_path: Path) -> None:
    maps = _write_zip(
        tmp_path / "maps.zip",
        {"maps/L99_V001.csv": _frame_map([(1, "0", "25", 0), (2, "1", "25", 25)])},
    )
    features = _write_zip(
        tmp_path / "features.zip",
        {"features/L99_V001.npy": _npy(np.ones((1, 2), dtype=np.float16))},
    )
    importer = OrganizerReferenceImporter(map_archive=maps, feature_archive=features)

    with pytest.raises(ValueError, match="feature/map row mismatch for L99_V001"):
        importer.validate()

    maps = _write_zip(
        tmp_path / "maps-noncontiguous.zip",
        {"maps/L99_V001.csv": _frame_map([(1, "0", "25", 0), (3, "1", "25", 25)])},
    )
    features = _write_zip(
        tmp_path / "features-two-rows.zip",
        {"features/L99_V001.npy": _npy(np.ones((2, 2), dtype=np.float16))},
    )
    importer = OrganizerReferenceImporter(map_archive=maps, feature_archive=features)

    with pytest.raises(ValueError, match="non-contiguous keyframe ordinals"):
        importer.validate()


@pytest.mark.parametrize(
    ("companion", "message"),
    [
        ("keyframes", "keyframe ordinals do not match map for L99_V001"),
        ("objects", "object ordinals do not match map for L99_V001"),
        ("metadata", "metadata videos do not exactly match frame-map videos"),
    ],
)
def test_validate_rejects_companion_pack_mismatches(
    tmp_path: Path, companion: str, message: str
) -> None:
    maps = _write_zip(
        tmp_path / "maps.zip",
        {"maps/L99_V001.csv": _frame_map([(1, "0", "25", 0)])},
    )
    features = _write_zip(
        tmp_path / "features.zip",
        {"features/L99_V001.npy": _npy(np.ones((1, 2), dtype=np.float16))},
    )
    keyframes = None
    objects = None
    metadata = None
    if companion == "keyframes":
        keyframes = _write_zip(
            tmp_path / "keyframes.zip",
            {"keyframes/L99_V001/002.jpg": b"wrong ordinal"},
        )
    elif companion == "objects":
        objects = _write_zip(
            tmp_path / "objects.zip",
            {"objects/L99_V001/002.json": b"{}"},
        )
    else:
        metadata = _write_zip(
            tmp_path / "metadata.zip",
            {"metadata/L99_V999.json": b"{}"},
        )
    importer = OrganizerReferenceImporter(
        map_archive=maps,
        feature_archive=features,
        keyframe_archives=[keyframes] if keyframes is not None else (),
        object_archive=objects,
        metadata_archive=metadata,
    )

    with pytest.raises(ValueError, match=message):
        importer.validate()


def test_import_objects_applies_inclusive_confidence_and_builds_lexical_docs(
    reference_pack: ReferencePack,
) -> None:
    store = FakeTextStore()

    count = reference_pack.importer.import_objects(store)

    assert count == 2
    assert store.create_calls == 1
    assert [document["id"] for document in store.documents] == [
        "object:fixture-b1:L99_V001:ordinal:1",
        "object:fixture-b1:L99_V002:ordinal:1",
    ]
    first = store.documents[0]
    assert first["keyframe_ordinal"] == 1
    assert first["source_frame_idx"] == 0
    assert first["kind"] == "objects"
    assert first["text_raw"] == "Người Person Xe đạp"
    assert first["text_nfc"] == "Người Person Xe đạp"
    assert first["text_accent_folded"] == "nguoi person xe dap"
    assert "Không lập chỉ mục" not in " ".join(document["text_raw"] for document in store.documents)


def test_import_metadata_uses_first_source_frame_and_normalized_search_text(
    reference_pack: ReferencePack,
) -> None:
    store = FakeTextStore()

    count = reference_pack.importer.import_metadata(store)

    assert count == 2
    assert store.create_calls == 1
    documents = {document["video_id"]: document for document in store.documents}
    first = documents["L99_V001"]
    assert first["id"] == "metadata:fixture-b1:L99_V001"
    assert first["source_frame_idx"] == 0
    assert first["keyframe_ordinal"] == 1
    assert first["kind"] == "metadata"
    assert first["text_raw"] == "Cà phê Quán đông đỏ buổi sáng"
    assert first["text_nfc"] == first["text_raw"]
    assert first["text_accent_folded"] == "ca phe quan dong do buoi sang"
    assert unicodedata.normalize("NFC", first["title"]) == "Cà phê"

    second = documents["L99_V002"]
    assert second["source_frame_idx"] == 62
    assert second["text_raw"] == "Cycling A rider passes bicycle"


class FixtureEncoder:
    def __init__(self, vectors_by_image: dict[bytes, np.ndarray], *, fail: bool = False) -> None:
        self.vectors_by_image = vectors_by_image
        self.fail = fail
        self.paths_existed_during_call = False

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        self.paths_existed_during_call = all(
            path.is_file() and path.suffix == ".jpg" for path in paths
        )
        vectors = np.asarray([self.vectors_by_image[path.read_bytes()] for path in paths])
        return -vectors if self.fail else vectors * 7


def test_verify_encoder_passes_normalized_equivalent_vectors(
    reference_pack: ReferencePack,
) -> None:
    encoder = FixtureEncoder(reference_pack.images)

    median = reference_pack.importer.verify_encoder(encoder, sample_size=3)

    assert median == pytest.approx(1.0)
    assert encoder.paths_existed_during_call is True


def test_verify_encoder_rejects_low_cosine_similarity(reference_pack: ReferencePack) -> None:
    encoder = FixtureEncoder(reference_pack.images, fail=True)

    with pytest.raises(
        ValueError,
        match=r"median cosine -1\.000000 is below 0\.999000",
    ):
        reference_pack.importer.verify_encoder(encoder, sample_size=3)
