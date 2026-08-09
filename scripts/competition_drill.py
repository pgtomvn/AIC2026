"""Run an offline readiness drill without mutating indexes or source archives."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assets import verify_asset_manifest
from src.contracts import CompetitionSpec, QueryRequest
from src.embedding.clip_encoder import CLIPEncoder
from src.indexing.vector_store import batch_collection_name
from src.ingest import (
    PROTECTED_ARCHIVE_NAMES,
    ProtectedArchiveError,
    discover_archives,
    relocate_archive,
    update_checksum_manifest,
)
from src.ingest.factory import build_native_ingestor
from src.ingest.ledger import JobLedger
from src.ingest.native_pipeline import discover_video_sources
from src.organizer_verification import organizer_verification_passed
from src.runtime import build_runtime
from src.submission import FrameIndexEvidence, export_submission
from src.tooling.ffmpeg import verify_ffmpeg_manifest
from src.utils.io_helpers import load_config
from src.utils.offline import enforce_huggingface_offline


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _wait_for_services(config: dict[str, object], *, timeout_seconds: float = 60.0) -> None:
    """Wait for both loopback search services after an intentional restart."""

    services = config["services"]  # type: ignore[index]
    qdrant = services["qdrant"]  # type: ignore[index]
    meili = services["meilisearch"]  # type: ignore[index]
    endpoints = (
        (
            str(qdrant["url"]).rstrip("/") + "/healthz",
            {"api-key": qdrant["api_key"]} if qdrant.get("api_key") else {},
        ),
        (
            str(meili["url"]).rstrip("/") + "/health",
            {"Authorization": f"Bearer {meili['api_key']}"} if meili.get("api_key") else {},
        ),
    )
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            for url, headers in endpoints:
                response = requests.get(url, headers=headers, timeout=3)
                response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(f"services did not recover within {timeout_seconds:g}s: {last_error}")


def _physical_service_restart(config: dict[str, object]) -> None:
    """Restart persisted local services for the explicitly selected operational drill."""

    completed = subprocess.run(
        ["docker", "compose", "restart", "qdrant", "meilisearch"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"docker compose restart failed: {detail}")
    _wait_for_services(config)


def _catalog_and_ingest_arrivals(
    config: dict[str, object],
    incoming: tuple[Path, ...],
    *,
    batch_id: str,
    competition: str,
    cpu_workers: int,
    queue_size: int,
) -> dict[str, object]:
    """Verify, relocate, and core-ingest stable video arrivals without full extraction."""

    ingestion = config["ingestion"]  # type: ignore[index]
    data = config["data"]  # type: ignore[index]
    inspections = discover_archives(
        incoming,
        min_age_seconds=float(ingestion["stable_archive_seconds"]),
        verify_crc=True,
        compute_sha256=True,
    )
    if not inspections:
        raise RuntimeError("no stable ZIP arrivals were discovered")
    protected = {name.casefold() for name in PROTECTED_ARCHIVE_NAMES}
    if any(item.path.name.casefold() in protected for item in inspections):
        raise ProtectedArchiveError("the operational drill cannot relocate a protected archive")

    ingestor = build_native_ingestor(
        config,  # type: ignore[arg-type]
        batch_id=batch_id,
        competition=competition,
        enrichment=False,
    )
    ingestor.initialize(recover_interrupted=True)
    relocated: list[Path] = []
    manifest = PROJECT_ROOT / "checksums" / "archives.sha256"
    for inspection in inspections:
        if inspection.crc_ok is not True or inspection.sha256 is None:
            raise RuntimeError(f"arrival verification was incomplete: {inspection.path}")
        ingestor.ledger.record_archive(inspection)
        destination = relocate_archive(
            inspection.path,
            _path(str(data["archives"])),
            competition=competition,
            batch_id=batch_id,
            kinds=inspection.kinds,
            expected_sha256=inspection.sha256,
        )
        update_checksum_manifest(
            manifest,
            destination,
            inspection.sha256,
            previous_path=inspection.path,
            root=PROJECT_ROOT,
        )
        ingestor.ledger.record_archive(inspection, relocated_to=destination)
        relocated.append(destination)

    sources = discover_video_sources(relocated)
    if not sources:
        raise RuntimeError("verified arrivals contained no supported raw videos")
    outcomes = ingestor.ingest_many(
        sources,
        cpu_workers=cpu_workers,
        queue_size=queue_size,
        enrich=False,
    )
    failures = {key: value for key, value in outcomes.items() if value.startswith("failed")}
    if failures:
        raise RuntimeError(f"core ingestion failed: {failures}")
    return {
        "archives": [str(path) for path in relocated],
        "videos": len(sources),
        "outcomes": outcomes,
        "status": ingestor.ledger.status_summary(batch_id).as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id")
    parser.add_argument("--query")
    parser.add_argument("--skip-services", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="exercise feedback, export, client restart, and interrupted-stage recovery",
    )
    parser.add_argument(
        "--operational",
        action="store_true",
        help=(
            "mutating end-to-end drill: relocate --incoming ZIPs, core-ingest them, and "
            "physically restart local search services"
        ),
    )
    parser.add_argument("--incoming", action="append", type=Path, default=[])
    parser.add_argument("--competition", default="aic2026")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.full and (not args.batch_id or not args.query):
        parser.error("--full requires both --batch-id and --query")
    if args.operational and (not args.full or not args.incoming):
        parser.error("--operational requires --full and at least one --incoming path")
    if args.operational and (args.skip_services or args.skip_assets):
        parser.error("--operational cannot skip service or asset verification")
    enforce_huggingface_offline()
    config = load_config()
    report: dict[str, object] = {"offline": True, "checks": {}, "errors": []}
    errors: list[str] = report["errors"]  # type: ignore[assignment]

    if not args.skip_assets:
        models = config["models"]
        required_assets = {
            _path(models["visual"]["cache_dir"]),
            _path(models["text"]["cache_dir"]),
            _path(models["whisper"]["local_path"]),
            _path(models["ocr"]["cache_dir"]),
        }
        for model_name in ("organizer_clip", "accelerated_visual"):
            model = models.get(model_name, {})
            enabled_profiles = tuple(
                str(value) for value in model.get("enabled_profiles", ())
            )
            if model.get("enabled", False) and (
                not enabled_profiles or str(config.get("profile")) in enabled_profiles
            ):
                required_assets.add(_path(model.get("cache_dir", "models/open_clip")))
        checked, asset_errors = verify_asset_manifest(
            PROJECT_ROOT / "checksums/runtime-assets.sha256",
            root=PROJECT_ROOT,
            required_paths=required_assets,
        )
        report["checks"]["runtime_assets"] = checked  # type: ignore[index]
        errors.extend(asset_errors)
        ffmpeg_manifest = config.get("tools", {}).get("ffmpeg", {}).get("manifest")
        if not isinstance(ffmpeg_manifest, str) or not ffmpeg_manifest:
            errors.append("tools.ffmpeg.manifest is not configured")
        else:
            try:
                installation = verify_ffmpeg_manifest(_path(ffmpeg_manifest))
                report["checks"]["ffmpeg"] = installation.version  # type: ignore[index]
            except (FileNotFoundError, OSError, TypeError, ValueError) as error:
                errors.append(f"FFmpeg tool verification failed: {error}")
        organizer = models.get("organizer_clip", {})
        if organizer.get("enabled", False):
            organizer_encoder = CLIPEncoder(
                organizer["model_name"],
                organizer["pretrained"],
                device=organizer.get("query_device", "cpu"),
                batch_size=1,
                cache_dir=_path(organizer.get("cache_dir", "models/open_clip")),
                local_files_only=bool(organizer.get("local_files_only", True)),
            )
            verified = organizer_verification_passed(
                organizer,
                expected_fingerprint=organizer_encoder.fingerprint,
                project_root=PROJECT_ROOT,
            )
            report["checks"]["organizer_encoder_verification"] = verified  # type: ignore[index]
            if not verified:
                errors.append(
                    "organizer encoder verification record is missing or incompatible; "
                    "run scripts/import_reference.py --verify-encoder"
                )
    data = config["data"]
    free = shutil.disk_usage(PROJECT_ROOT).free
    reserve = int(float(data["free_space_reserve_gib"]) * 1024**3)
    report["checks"]["free_bytes"] = free  # type: ignore[index]
    if free < reserve:
        errors.append(f"free disk {free} is below configured reserve {reserve}")

    if not args.skip_services:
        services = config["services"]
        qdrant_headers = {}
        if services["qdrant"].get("api_key"):
            qdrant_headers["api-key"] = services["qdrant"]["api_key"]
        endpoints = (
            ("qdrant", services["qdrant"]["url"].rstrip("/") + "/healthz", qdrant_headers),
            ("meilisearch", services["meilisearch"]["url"].rstrip("/") + "/health", {}),
        )
        for name, url, headers in endpoints:
            try:
                response = requests.get(url, headers=headers, timeout=3)
                response.raise_for_status()
                report["checks"][name] = "healthy"  # type: ignore[index]
            except requests.RequestException as error:
                errors.append(f"{name} is unavailable: {error}")
        qdrant = services["qdrant"]
        qdrant_headers = (
            {"api-key": qdrant["api_key"]} if qdrant.get("api_key") else {}
        )
        required_collections = {
            "native_frames": (
                qdrant["collections"]["native_frames"],
                int(config["models"]["visual"]["embedding_dim"]),
            ),
            "text_segments": (
                qdrant["collections"]["text_segments"],
                int(config["models"]["text"]["embedding_dim"]),
            ),
        }
        if config["models"].get("organizer_clip", {}).get("enabled", False):
            required_collections["organizer_frames"] = (
                qdrant["collections"]["organizer_frames"],
                int(config["models"]["organizer_clip"]["embedding_dim"]),
            )
        accelerated = config["models"].get("accelerated_visual", {})
        accelerated_profiles = tuple(
            str(value) for value in accelerated.get("enabled_profiles", ())
        )
        if accelerated.get("enabled", False) and (
            not accelerated_profiles
            or str(config.get("profile")) in accelerated_profiles
        ):
            required_collections["accelerated_frames"] = (
                qdrant["collections"]["accelerated_frames"],
                int(config["models"]["accelerated_visual"]["embedding_dim"]),
            )
        for logical_name, (collection_name, expected_dimension) in required_collections.items():
            batch_scoped = "{batch}" in collection_name
            if batch_scoped:
                if not args.batch_id:
                    report["checks"][f"qdrant:{logical_name}"] = (  # type: ignore[index]
                        "skipped: --batch-id is required for a batch collection"
                    )
                    continue
                collection_name = batch_collection_name(collection_name, args.batch_id)
            try:
                response = requests.get(
                    qdrant["url"].rstrip("/") + f"/collections/{collection_name}",
                    headers=qdrant_headers,
                    timeout=10,
                )
                if batch_scoped and response.status_code == 404:
                    report["checks"][f"qdrant:{logical_name}"] = (  # type: ignore[index]
                        f"not initialized for batch {args.batch_id}"
                    )
                    continue
                response.raise_for_status()
                result = response.json()["result"]
                vector_config = result["config"]["params"]["vectors"]
                actual_dimension = int(vector_config["size"])
                if actual_dimension != expected_dimension:
                    raise ValueError(
                        f"dimension {actual_dimension} does not match {expected_dimension}"
                    )
                if result["status"] != "green" or result["optimizer_status"] != "ok":
                    raise ValueError(
                        f"status={result['status']}, optimizer={result['optimizer_status']}"
                    )
                report["checks"][f"qdrant:{logical_name}"] = {  # type: ignore[index]
                    "points": int(result["points_count"]),
                    "dimension": actual_dimension,
                    "status": result["status"],
                }
            except (KeyError, TypeError, ValueError, requests.RequestException) as error:
                errors.append(f"Qdrant collection {logical_name} is not ready: {error}")

        meili = services["meilisearch"]
        meili_headers = (
            {"Authorization": f"Bearer {meili['api_key']}"}
            if meili.get("api_key")
            else {}
        )
        for logical_name, index_name in meili["indexes"].items():
            try:
                response = requests.get(
                    meili["url"].rstrip("/") + f"/indexes/{index_name}/stats",
                    headers=meili_headers,
                    timeout=10,
                )
                response.raise_for_status()
                report["checks"][f"meilisearch:{logical_name}"] = {  # type: ignore[index]
                    "documents": int(response.json()["numberOfDocuments"])
                }
            except (KeyError, TypeError, ValueError, requests.RequestException) as error:
                errors.append(f"Meilisearch index {logical_name} is not ready: {error}")

    context = None
    if args.operational:
        try:
            report["checks"]["arrival_and_core_ingest"] = _catalog_and_ingest_arrivals(  # type: ignore[index]
                config,
                tuple(args.incoming),
                batch_id=args.batch_id,
                competition=args.competition,
                cpu_workers=args.cpu_workers,
                queue_size=args.queue_size,
            )
        except Exception as error:
            errors.append(f"arrival/core-ingestion drill failed: {error}")
    if args.batch_id or args.query or args.full:
        context = build_runtime(config)
        context.ledger.initialize()
        if context.artifact_repository is not None:
            context.artifact_repository.initialize()
    if args.batch_id:
        try:
            assert context is not None
            report["batch"] = context.ledger.status_summary(args.batch_id).as_dict()
        except Exception as error:
            errors.append(f"batch status failed: {error}")
    hits: list[dict[str, object]] = []
    request = None
    if args.query:
        assert context is not None
        request = QueryRequest(
            query_id="offline-drill",
            original_text=args.query,
            batch_filters=(args.batch_id,) if args.batch_id else (),
            limit=100,
        )
        hits = context.pipeline.search(request, refine_raw=False)
        report["query_hits"] = len(hits)
        if not hits:
            errors.append(
                f"query returned no hits; channels={context.pipeline.last_channel_errors}"
            )
    if args.full and request is not None and hits:
        assert context is not None
        feedback_hits: list[dict[str, object]] | None = None
        feedback_error = "no candidate had an unambiguous primary visual vector"
        for hit in hits[:20]:
            try:
                feedback_hits = context.pipeline.refine(
                    request,
                    positive_ids=((str(hit["video_id"]), int(hit["source_frame_idx"])),),
                    limit=100,
                )
                break
            except (RuntimeError, ValueError) as error:
                feedback_error = str(error)
        if not feedback_hits:
            errors.append(f"feedback refinement failed: {feedback_error}")
        else:
            report["feedback_hits"] = len(feedback_hits)

        if context.artifact_repository is None:
            errors.append("canonical artifact repository is unavailable for export")
        else:
            spec = CompetitionSpec.from_config(config.get("competition", {}))
            identities = [
                (spec.normalize_video_id(hit["video_id"]), int(hit["source_frame_idx"]))
                for hit in hits[:100]
            ]
            existing = context.artifact_repository.existing_source_frames(
                args.batch_id,
                identities,
            )
            mappings = context.artifact_repository.frame_index_mappings_for_sources(
                args.batch_id,
                identities,
            )
            try:
                with tempfile.TemporaryDirectory(prefix="aic-export-drill-") as directory:
                    destination = Path(directory) / "submission.csv"
                    export_submission(
                        hits,
                        destination,
                        spec=spec,
                        batch_id=args.batch_id,
                        is_batch_active=context.ledger.is_batch_active,
                        is_canonical_source_frame=lambda batch, video, frame: (
                            batch == args.batch_id and (video, frame) in existing
                        ),
                        frame_index_evidence=FrameIndexEvidence(mappings),
                    )
                    report["checks"]["export_rows"] = len(  # type: ignore[index]
                        destination.read_text(encoding="utf-8").splitlines()
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                errors.append(f"submission export failed: {error}")

        try:
            if args.operational:
                _physical_service_restart(config)
                report["checks"]["physical_service_restart"] = "passed"  # type: ignore[index]
            restarted = build_runtime(config)
            restarted.ledger.initialize()
            if restarted.artifact_repository is not None:
                restarted.artifact_repository.initialize()
            restart_hits = restarted.pipeline.search(request, limit=5, refine_raw=False)
            if not restart_hits:
                raise RuntimeError("restarted retrieval client returned no hits")
            report["checks"]["client_restart_query_hits"] = len(restart_hits)  # type: ignore[index]
        except Exception as error:
            errors.append(f"client restart/reconnect failed: {error}")

        cache_root = _path(".cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="aic-recovery-drill-",
                dir=cache_root,
                ignore_cleanup_errors=True,
            ) as directory:
                recovery = JobLedger(Path(directory) / "jobs.sqlite3")
                recovery.initialize()
                recovery.register_batch("drill", "aic2026", now=1.0)
                recovery.register_video("drill", "L99_V999", "fixture://video", now=1.0)
                recovery.start_stage("drill", "L99_V999", "visual", "fixture:v1", now=2.0)
                recovered = recovery.recover_interrupted("drill", now=3.0)
                if recovered != 1:
                    raise RuntimeError(f"expected one recovered stage, found {recovered}")
                report["checks"]["interrupted_stage_recovery"] = "passed"  # type: ignore[index]
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            errors.append(f"interrupted-stage recovery failed: {error}")
    report["ok"] = not errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
