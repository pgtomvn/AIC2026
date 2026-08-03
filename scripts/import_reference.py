"""Validate or index the organizer AIC25-B1 reference packs without extraction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.clip_encoder import CLIPEncoder
from src.indexing.text_store import TextStore
from src.indexing.vector_store import VectorStore
from src.ingest.artifacts import ArtifactRepository
from src.ingest.ledger import JobLedger
from src.ingest.reference_importer import OrganizerReferenceImporter
from src.organizer_verification import (
    organizer_verification_passed,
    persist_organizer_verification,
)
from src.utils.io_helpers import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=PROJECT_ROOT / "map-keyframes-aic25-b1.zip")
    parser.add_argument(
        "--features", type=Path, default=PROJECT_ROOT / "clip-features-32-aic25-b1.zip"
    )
    parser.add_argument("--objects", type=Path, default=PROJECT_ROOT / "objects-aic25-b1.zip")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "media-info-aic25-b1.zip")
    parser.add_argument("--keyframes", type=Path, nargs="*")
    parser.add_argument("--expected-videos", type=int, default=873)
    parser.add_argument("--expected-frames", type=int, default=177_321)
    parser.add_argument("--index-vectors", action="store_true")
    parser.add_argument("--index-objects", action="store_true")
    parser.add_argument("--index-metadata", action="store_true")
    parser.add_argument("--verify-encoder", action="store_true")
    parser.add_argument("--import-canonical", action="store_true")
    parser.add_argument(
        "--activate-existing-vectors",
        action="store_true",
        help="verify the existing organizer collection, then activate canonical reference records",
    )
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)
    keyframes = args.keyframes
    if keyframes is None:
        keyframes = sorted(PROJECT_ROOT.glob("Keyframes_*.zip"))
    importer = OrganizerReferenceImporter(
        map_archive=args.map,
        feature_archive=args.features,
        keyframe_archives=keyframes,
        object_archive=args.objects,
        metadata_archive=args.metadata,
        batch_id=args.batch_id,
    )
    stats = importer.validate(
        expected_videos=args.expected_videos,
        expected_frames=args.expected_frames,
    )
    config = load_config()
    qdrant = config["services"]["qdrant"]
    meili = config["services"]["meilisearch"]
    if args.verify_encoder:
        model = config["models"]["organizer_clip"]
        encoder = CLIPEncoder(
            model["model_name"],
            model["pretrained"],
            device=model.get("verification_device", "cuda"),
            batch_size=16,
            cache_dir=PROJECT_ROOT / model.get("cache_dir", "models/open_clip"),
            local_files_only=bool(model.get("local_files_only", True)),
        )
        median = importer.verify_encoder(
            encoder,
            sample_size=int(model["verification_sample_size"]),
            minimum_median_cosine=float(model["verification_median_cosine"]),
        )
        verification_record = persist_organizer_verification(
            model,
            encoder_fingerprint=encoder.fingerprint,
            median_cosine=median,
            sample_size=int(model["verification_sample_size"]),
            project_root=PROJECT_ROOT,
        )
        print(
            json.dumps(
                {
                    "verification_median_cosine": median,
                    "verification_record": str(verification_record),
                }
            )
        )
    vector_store = None
    if args.index_vectors or args.activate_existing_vectors:
        model = config["models"]["organizer_clip"]
        vector_store = VectorStore(
            url=qdrant["url"],
            api_key=qdrant.get("api_key"),
            collection_name=qdrant["collections"]["organizer_frames"],
            vector_size=int(model["embedding_dim"]),
            distance=qdrant["distance"],
        )
        if args.index_vectors:
            stats = importer.import_vectors(
                vector_store,
                recreate=args.recreate,
                expected_videos=args.expected_videos,
                expected_frames=args.expected_frames,
            )
        else:
            importer.verify_vector_store(
                vector_store,
                expected_videos=args.expected_videos,
                expected_frames=args.expected_frames,
            )
    if args.index_objects:
        object_store = TextStore(meili["url"], meili.get("api_key"), meili["indexes"]["metadata"])
        stats = replace(stats, object_documents=importer.import_objects(object_store))
    if args.index_metadata:
        metadata_store = TextStore(meili["url"], meili.get("api_key"), meili["indexes"]["metadata"])
        stats = replace(
            stats,
            metadata_documents=importer.import_metadata(
                metadata_store,
                create_index=not args.index_objects,
            ),
        )
    report = asdict(stats)
    if args.import_canonical or args.index_vectors or args.activate_existing_vectors:
        model = config["models"]["organizer_clip"]
        verification_encoder = CLIPEncoder(
            model["model_name"],
            model["pretrained"],
            device=model.get("query_device", "cpu"),
            cache_dir=PROJECT_ROOT / model.get("cache_dir", "models/open_clip"),
            local_files_only=bool(model.get("local_files_only", True)),
        )
        organizer_verified = organizer_verification_passed(
            model,
            expected_fingerprint=verification_encoder.fingerprint,
            project_root=PROJECT_ROOT,
        )
        report["organizer_encoder_verified"] = organizer_verified
        data = config["data"]
        artifact_path = Path(data["artifact_db"])
        ledger_path = Path(data["catalog_db"])
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        if not ledger_path.is_absolute():
            ledger_path = PROJECT_ROOT / ledger_path
        repository = ArtifactRepository(artifact_path)
        ledger = JobLedger(ledger_path)
        repository.initialize()
        ledger.initialize()
        canonical = importer.import_canonical_frames(
            repository,
            ledger=ledger,
            vectors_ready=(args.index_vectors or args.activate_existing_vectors)
            and organizer_verified,
        )
        report["canonical"] = asdict(canonical)
        report["batch_active"] = ledger.is_batch_active(args.batch_id)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
