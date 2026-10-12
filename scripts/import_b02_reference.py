"""Validate B02 reference ZIPs; explicitly opt in to isolated B02 imports.

The default command is read-only. Source videos and audio are outside this
reference-import path. Media-info is checked for M/S coverage, but contains
technical fields rather than searchable title/description text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.clip_encoder import CLIPEncoder
from src.indexing.text_store import TextStore
from src.indexing.vector_store import VectorStore, batch_collection_name
from src.ingest.artifacts import ArtifactRepository
from src.ingest.b02_reference import (
    encoder_receipt_matches,
    file_sha256,
    load_b02_reference_plan,
)
from src.ingest.ledger import JobLedger
from src.ingest.reference_importer import OrganizerReferenceImporter
from src.ingest.reference_resume import AuditedObjectResumeStore
from src.utils.io_helpers import load_config

OBJECT_INDEX = "aic26_b2_objects_v1"
VERIFICATION_RECORD = PROJECT_ROOT / "outputs/checkpoints/b02-p1/b02-clip-encoder-verification.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/checkpoints/b02-p0/b02-p0-manifest-v3.json"
DEFAULT_EVIDENCE = PROJECT_ROOT / "outputs/checkpoints/b02-p0/m08-v001-pts-collision-evidence.json"


def _configured_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _b01_counts(config: dict[str, Any]) -> dict[str, int]:
    """Take an exact, read-only baseline of the existing B01 channels."""
    qdrant = config["services"]["qdrant"]
    meili = config["services"]["meilisearch"]
    batch_id = "aic25-b1-reference"
    counts = {}
    for name, collection, dimension in (
        ("organizer", qdrant["collections"]["organizer_frames"], 512),
        ("siglip2", batch_collection_name(qdrant["collections"]["native_frames"], batch_id), 768),
    ):
        counts[name] = VectorStore(
            url=qdrant["url"], api_key=qdrant.get("api_key"),
            collection_name=collection, vector_size=dimension,
        ).count({"batch_id": batch_id})
    for name in ("ocr", "asr", "metadata"):
        counts[name] = TextStore(
            meili["url"], meili.get("api_key"), meili["indexes"][name]
        ).count_documents(f'batch_id = "{batch_id}"')
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--collision-evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--apply", action="store_true", help="permit the selected write operation")
    parser.add_argument("--index-vectors", action="store_true")
    parser.add_argument("--index-objects", action="store_true")
    parser.add_argument("--import-canonical", action="store_true")
    parser.add_argument("--video", action="append", help="bounded pilot video; repeat as needed")
    parser.add_argument("--resume", action="store_true", help="idempotently upsert existing B02 points")
    parser.add_argument("--report", type=Path, help="new durable run report (never overwrite a prior run)")
    parser.add_argument("--qdrant-host-disk", type=Path, help="host disk containing Docker data")
    args = parser.parse_args(argv)
    selected_writes = args.index_vectors or args.index_objects or args.import_canonical
    if selected_writes and not args.apply:
        parser.error("write operations require --apply")
    if args.apply and not selected_writes:
        parser.error("--apply needs an explicit write operation")
    if args.apply and (args.report is None or args.qdrant_host_disk is None):
        parser.error("--apply requires --report and --qdrant-host-disk for checkpoint/capacity checks")
    if args.video and args.index_objects:
        parser.error("object import is a full-source operation; omit --video")
    if args.report is not None and args.report.exists():
        parser.error("report already exists; use a new path for each run")

    plan = load_b02_reference_plan(args.root, args.manifest, args.collision_evidence)
    importer = OrganizerReferenceImporter(
        map_archive=plan.map_archive,
        feature_archive=plan.feature_archive,
        keyframe_archives=plan.keyframe_archives,
        object_archives=plan.object_archives,
        metadata_archive=plan.media_archive,
        metadata_policy="partial",
        batch_id=plan.batch_id,
        identity_mode="ordinal",
        reference_manifest_sha256=plan.manifest_sha256,
    )
    stats = importer.validate(expected_videos=plan.videos, expected_frames=plan.frames)
    config = load_config()
    model = config["models"]["organizer_clip"]
    qdrant = config["services"]["qdrant"]
    meili = config["services"]["meilisearch"]
    vector_collection = batch_collection_name(
        qdrant["collections"]["organizer_compatible_frames"], plan.batch_id
    )
    if vector_collection == qdrant["collections"]["organizer_frames"]:
        raise ValueError("B02 collection must differ from the B01 organizer collection")
    if OBJECT_INDEX == meili["indexes"]["metadata"]:
        raise ValueError("B02 object index must differ from the B01 metadata index")
    encoder = CLIPEncoder(
        model["model_name"],
        model["pretrained"],
        device="cpu",
        cache_dir=PROJECT_ROOT / model.get("cache_dir", "models/open_clip"),
        local_files_only=True,
    )
    feature_sha256 = file_sha256(plan.feature_archive)
    verified = encoder_receipt_matches(
        VERIFICATION_RECORD,
        plan=plan,
        feature_sha256=feature_sha256,
        model_name=model["model_name"],
        pretrained=model["pretrained"],
        encoder_fingerprint=encoder.fingerprint,
        minimum_median_cosine=float(model["verification_median_cosine"]),
        minimum_sample_size=6,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_videos = sorted(set(args.video or manifest["videos"]))
    if not selected_videos or set(selected_videos) - set(manifest["videos"]):
        parser.error("pilot video is not present in the P0 manifest")
    selected_frames = sum(manifest["videos"][video]["keyframe_count"] for video in selected_videos)
    report: dict[str, Any] = {
        "batch_id": plan.batch_id,
        "mode": "apply" if args.apply else "validate_only",
        "manifest_sha256": plan.manifest_sha256,
        "quarantined_M08_pts_pairs": len(plan.quarantined_pts_pairs),
        "reference": asdict(stats),
        "encoder_verified": verified,
        "vector_collection": vector_collection,
        "object_index": OBJECT_INDEX,
        "selected_videos": selected_videos,
        "selected_frames": selected_frames,
        "started_at_unix": time.time(),
        "batch_active": False,
    }
    if not selected_writes:
        print(json.dumps(report, indent=2))
        return 0

    if not verified:
        raise RuntimeError("B02 encoder verification receipt is absent or does not match source")

    reserve = float(config["data"].get("free_space_reserve_gib", 30))
    def guard(video_id: str = "") -> None:
        for path, minimum in ((args.root, reserve), (args.qdrant_host_disk, 5.0)):
            if shutil.disk_usage(path).free < minimum * 1024 ** 3:
                raise RuntimeError(f"free-space reserve reached on {path}; resumable import stopped")
        if video_id:
            print(f"B02 importing {video_id}", flush=True)

    guard()
    report["b01_before"] = _b01_counts(config)
    report["state"] = "running"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    def checkpoint() -> None:
        temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.report)

    if args.index_vectors:
        store = VectorStore(
            url=qdrant["url"],
            api_key=qdrant.get("api_key"),
            collection_name=vector_collection,
            vector_size=plan.feature_dimension,
            distance=qdrant["distance"],
        )
        client = store._get_client()
        if client.collection_exists(collection_name=vector_collection) and store.count():
            if not args.resume:
                raise RuntimeError("B02 collection is nonempty; explicit --resume is required")
            for payload in store.iter_payloads():
                if (payload.get("batch_id") != plan.batch_id or
                        payload.get("reference_manifest_sha256") != plan.manifest_sha256):
                    raise RuntimeError("existing B02 point has unverified source provenance")
        report["vector_import"] = asdict(
            importer.import_vectors(
                store, recreate=False, expected_videos=len(selected_videos),
                expected_frames=selected_frames, video_ids=args.video, before_video=guard,
            )
        )
        report["vector_verified"] = asdict(importer.verify_vector_store(
            store, expected_videos=len(selected_videos), expected_frames=selected_frames,
            video_ids=args.video,
        ))
        checkpoint()
    if args.index_objects:
        store = TextStore(meili["url"], meili.get("api_key"), OBJECT_INDEX)
        target = store
        if args.resume:
            print("Auditing existing objects against frozen source before mutation", flush=True)
            target = AuditedObjectResumeStore.audit(importer, store)
            report["object_resume_audit"] = {
                "verified_existing": len(target.verified), "expected_documents": target.expected_count,
            }
            checkpoint()
            print(json.dumps(report["object_resume_audit"]), flush=True)

        def object_guard() -> None:
            guard()
            if isinstance(target, AuditedObjectResumeStore):
                report["object_progress"] = {"written": target.written, "skipped": target.skipped}
                checkpoint()

        report["object_documents"] = importer.import_objects(target, before_batch=object_guard)
        if isinstance(target, AuditedObjectResumeStore):
            if report["object_documents"] != target.expected_count:
                raise RuntimeError("object source count changed after audit")
            report["object_progress"] = {"written": target.written, "skipped": target.skipped}
        report["object_external_count"] = store.count_documents(f'batch_id = "{plan.batch_id}"')
        if report["object_external_count"] != report["object_documents"]:
            raise RuntimeError("B02 object external count differs from imported documents")
        checkpoint()
    if args.import_canonical:
        data = config["data"]
        repository = ArtifactRepository(_configured_path(data["artifact_db"], PROJECT_ROOT))
        ledger = JobLedger(_configured_path(data["catalog_db"], PROJECT_ROOT))
        repository.initialize()
        ledger.initialize()
        report["canonical"] = asdict(
            importer.import_canonical_frames(
                repository,
                ledger=ledger,
                vectors_ready=False,
                competition="aic26",
                video_ids=args.video,
                before_video=guard,
            )
        )
        report["batch_active"] = ledger.is_batch_active(plan.batch_id)
        checkpoint()
    report["b01_after"] = _b01_counts(config)
    if report["b01_after"] != report["b01_before"]:
        raise RuntimeError("B01 external counts changed during B02 import; investigate before activation")
    report["state"] = "complete"
    report["completed_at_unix"] = time.time()
    report["free_bytes_after"] = {
        str(path): shutil.disk_usage(path).free for path in (args.root, args.qdrant_host_disk)
    }
    checkpoint()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
