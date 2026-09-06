#!/usr/bin/env python3
"""Provision reference Batch 1 multimodal indexes in isolated stage processes."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import PROJECT_ROOT
from src.assets import verify_asset_manifest
from src.embedding.siglip_encoder import SigLIP2Encoder
from src.extraction.asr_engine import ASREngine
from src.extraction.audio import AudioExtractor
from src.extraction.rapid_ocr_engine import RapidOCREngine
from src.ingest.batch_lock import BatchIngestLock
from src.ingest.checkpoint_mgr import CheckpointMirror
from src.ingest.native_pipeline import ScratchManager, discover_video_sources
from src.ingest.reference_provisioner import (
    REFERENCE_BATCH_ID,
    ReferenceProvisioner,
    StageReport,
    normalize_reference_batch,
)
from src.ingest.resource_watchdog import ResourceBudgetGuard
from src.retrieval.qa_media import CanonicalImageResolver
from src.runtime import build_runtime
from src.tooling.ffmpeg import verify_ffmpeg_manifest
from src.utils.io_helpers import load_config


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _require_service(url: str, code: str) -> None:
    try:
        response = requests.get(url, timeout=(1.0, 2.0))
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(code) from error


_PUBLIC_ERROR_CODES = frozenset(
    {
        "artifact_repository_unavailable",
        "asr_asset_manifest_failed",
        "meilisearch_unavailable",
        "ocr_asset_manifest_failed",
        "provisioning_channels_invalid",
        "provisioning_gpu_budget_exceeded",
        "provisioning_requires_api_and_streamlit_stopped",
        "provisioning_rss_budget_exceeded",
        "qdrant_unavailable",
        "reference_batch_unsupported",
        "reprocess_stage_not_selected",
        "scoped_reprocess_requires_stage_and_video",
    }
)


def _public_error_code(error: BaseException) -> str:
    code = str(error)
    if code in _PUBLIC_ERROR_CODES:
        return code
    for prefix in ("provisioning_stage_failed:", "provisioning_stage_report_missing:"):
        if code.startswith(prefix) and code.rsplit(":", 1)[-1] in {"ocr", "siglip2", "asr"}:
            return code
    return "reference_provisioning_failed"


def _unload_ollama(config: dict[str, Any]) -> None:
    service = config.get("services", {}).get("ollama_vlm", {})
    url = str(service.get("url", "http://127.0.0.1:11434")).rstrip("/")
    model = str(service.get("model", ""))
    if not model:
        return
    try:
        requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=(1.0, 5.0),
        )
    except requests.RequestException:
        return


def _video_archive_paths(config: dict[str, Any]) -> tuple[Path, ...]:
    data = config["data"]
    roots = (
        PROJECT_ROOT,
        _path(data["archives"]) / "aic25" / "b1" / "videos",
        _path(data["raw_videos"]),
    )
    values: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root == PROJECT_ROOT:
            candidates = root.glob("Videos_*.zip")
        else:
            candidates = root.rglob("*")
        for path in candidates:
            if path.is_file() and (
                path.suffix.casefold() == ".zip"
                or path.suffix.casefold() in {".mp4", ".mkv", ".mov", ".avi", ".webm"}
            ):
                values.add(path.resolve())
    return tuple(sorted(values))


def _verified_ffmpeg_for_stage(config: dict[str, Any], stage: str) -> Any | None:
    if stage != "asr":
        return None
    try:
        return verify_ffmpeg_manifest(_path(config["tools"]["ffmpeg"]["manifest"]))
    except OSError:
        # PyAV/faster-whisper remains local and has a distinct stage fingerprint.
        return None


def build_provisioner(
    config: dict[str, Any],
    *,
    stage: str,
    checkpoint_dir: Path,
) -> ReferenceProvisioner:
    """Compose lazy stage dependencies and verify managed video archives."""

    if stage not in {"ocr", "siglip2", "asr"}:
        raise ValueError("provisioning_stage_invalid")
    context = build_runtime(config)
    if context.artifact_repository is None:
        raise RuntimeError("artifact_repository_unavailable")
    context.ledger.initialize()
    context.artifact_repository.initialize()

    visual_channel = next(
        channel for channel in context.pipeline.dense_channels if channel.name == "visual"
    )
    lexical = {channel.name: channel for channel in context.pipeline.lexical_channels}
    reference = config["models"]["reference_provisioning"]
    visual = config["models"]["visual"]
    ocr = reference["rapidocr"]
    whisper = reference["whisper"]
    limits = reference["resource_limits"]
    data = config["data"]
    ffmpeg = _verified_ffmpeg_for_stage(config, stage)

    archive_paths = _video_archive_paths(config)
    sources = discover_video_sources(archive_paths)
    source_map = {source.video_id: source for source in sources}
    for path in sorted({source.path.resolve() for source in sources if source.member is not None}):
        context.ledger.assert_archive_ready(path)

    encoder = SigLIP2Encoder(
        model_name=visual["model_name"],
        revision=visual.get("revision"),
        device=visual["device"],
        batch_size=int(reference["siglip_initial_batch_size"]),
        max_batch_size=int(reference["siglip_max_batch_size"]),
        cache_dir=_path(visual["cache_dir"]),
        local_files_only=bool(visual.get("local_files_only", True)),
    )
    ocr_engine = RapidOCREngine(
        model_dir=_path(ocr["model_dir"]),
        confidence_threshold=float(ocr["confidence_threshold"]),
        min_text_characters=int(ocr["min_text_characters"]),
        local_files_only=bool(ocr.get("local_files_only", True)),
        package_version=str(ocr["package_version"]),
    )
    asr_engine = ASREngine(
        str(whisper["hub_model_id"]),
        whisper.get("language"),
        str(whisper["device"]),
        compute_type=str(whisper["compute_type"]),
        beam_size=int(whisper["beam_size"]),
        retry_beam_size=int(whisper["retry_beam_size"]),
        vad_filter=bool(whisper["vad_filter"]),
        model_dir=_path("models/whisper"),
        model_path=_path(whisper["local_path"]),
        model_revision=str(whisper["revision"]),
        local_files_only=bool(whisper.get("local_files_only", True)),
    )
    scratch = ScratchManager(
        _path(data["scratch"]),
        limit_bytes=int(float(data["scratch_limit_gib"]) * 1024**3),
        reserve_bytes=int(float(data["free_space_reserve_gib"]) * 1024**3),
        archive_materializers=1,
    )
    resolver = CanonicalImageResolver(
        (PROJECT_ROOT,),
        max_input_bytes=32 * 1024 * 1024,
        max_dimension=4096,
        normalized_cache_entries=4,
        normalized_cache_bytes=64 * 1024 * 1024,
    )
    readiness = config["readiness"]["batches"][REFERENCE_BATCH_ID]
    return ReferenceProvisioner(
        ledger=context.ledger,
        artifacts=context.artifact_repository,
        image_resolver=resolver,
        resource_guard=ResourceBudgetGuard(
            rss_limit_bytes=int(float(limits["process_rss_gib"]) * 1024**3),
            gpu_limit_bytes=int(float(limits["gpu_process_gib"]) * 1024**3),
        ),
        checkpoint_mirror=CheckpointMirror(checkpoint_dir),
        visual_encoder=encoder,
        visual_store=visual_channel.store.for_batch(REFERENCE_BATCH_ID),
        ocr_engine=ocr_engine,
        ocr_store=lexical["ocr"].store,
        asr_engine=asr_engine,
        asr_store=lexical["asr"].store,
        scratch=scratch,
        audio_extractor=(
            None
            if ffmpeg is None
            else AudioExtractor(
                ffmpeg.ffmpeg,
                tool_fingerprint=ffmpeg.archive_sha256,
            )
        ),
        ffprobe_path=None if ffmpeg is None else ffmpeg.ffprobe,
        video_sources=source_map,
        expected_videos=int(readiness["expected_videos"]),
        expected_frames=int(readiness["expected_canonical_frames"]),
        ocr_upsert_batch_size=int(reference["ocr_upsert_batch_size"]),
        vector_upsert_batch_size=int(reference["vector_upsert_batch_size"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="B01")
    parser.add_argument(
        "--channels",
        default="ocr,siglip2,asr",
        help="Comma-separated subset of ocr,siglip2,asr",
    )
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/provisioning"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reprocess-stage", choices=("ocr", "siglip2", "asr"))
    parser.add_argument("--reprocess-video", action="append", default=[])
    parser.add_argument("--stage", choices=("ocr", "siglip2", "asr"), help=argparse.SUPPRESS)
    return parser


def _channels(value: str) -> tuple[str, ...]:
    parts = tuple(item.strip().casefold() for item in value.split(",") if item.strip())
    if not parts or len(set(parts)) != len(parts):
        raise ValueError("provisioning_channels_invalid")
    allowed = ("ocr", "siglip2", "asr")
    if any(item not in allowed for item in parts):
        raise ValueError("provisioning_channels_invalid")
    return tuple(item for item in allowed if item in parts)


def _verify_stage_assets(config: dict[str, Any], stage: str) -> None:
    reference = config["models"]["reference_provisioning"]
    if stage == "ocr":
        model = reference["rapidocr"]
    elif stage == "asr":
        model = reference["whisper"]
    else:
        return
    manifest = _path(model["asset_manifest"])
    required = _path(model["model_dir"] if stage == "ocr" else model["local_path"])
    _checked, errors = verify_asset_manifest(
        manifest,
        root=PROJECT_ROOT,
        required_paths=(required,),
    )
    if errors:
        raise RuntimeError(f"{stage}_asset_manifest_failed")


def _dry_run_report(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> StageReport:
    specification = config["readiness"]["batches"][REFERENCE_BATCH_ID]
    expected_videos = int(specification["expected_videos"])
    selected_videos = min(args.limit_videos or expected_videos, expected_videos)
    source_videos = len(
        {source.video_id for source in discover_video_sources(_video_archive_paths(config))}
    )
    source_missing = (
        max(0, selected_videos - min(selected_videos, source_videos)) if args.stage == "asr" else 0
    )
    return StageReport(
        batch_id=REFERENCE_BATCH_ID,
        stage=args.stage,
        state="dry_run",
        selected_videos=selected_videos,
        completed_videos=0,
        failed_videos=0,
        source_missing_videos=source_missing,
        canonical_frames=0,
        records_written=0,
        elapsed_ms=0.0,
        peak_rss_bytes=0,
        peak_gpu_bytes=0,
        frames_per_hour=0.0,
        videos_per_hour=0.0,
    )


def _run_child(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.dry_run:
        report = _dry_run_report(args, config)
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    _verify_stage_assets(config, args.stage)
    provisioner = build_provisioner(
        config,
        checkpoint_dir=_path(args.checkpoint_dir),
        stage=args.stage,
    )
    report = provisioner.run_stage(
        args.stage,
        limit_videos=args.limit_videos,
        dry_run=args.dry_run,
        reprocess_videos=(args.reprocess_video if args.reprocess_stage == args.stage else ()),
    )
    payload = report.to_dict()
    _atomic_json(_path(args.output_dir) / f"b01-{args.stage}-report.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if report.failed_videos == 0 else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        batch_id = normalize_reference_batch(args.batch)
        if batch_id != REFERENCE_BATCH_ID:
            raise ValueError("reference_batch_unsupported")
        channels = _channels(args.channels)
        if bool(args.reprocess_stage) != bool(args.reprocess_video):
            raise ValueError("scoped_reprocess_requires_stage_and_video")
        if args.reprocess_stage and args.reprocess_stage not in channels:
            raise ValueError("reprocess_stage_not_selected")
        config = load_config()
        if args.stage is not None:
            return _run_child(args, config)
        if args.dry_run:
            reports: list[dict[str, Any]] = []
            for stage in channels:
                stage_args = argparse.Namespace(**vars(args))
                stage_args.stage = stage
                reports.append(_dry_run_report(stage_args, config).to_dict())
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "batch_id": REFERENCE_BATCH_ID,
                        "state": "dry_run",
                        "channels": reports,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if _port_open(8080) or _port_open(8504):
            raise RuntimeError("provisioning_requires_api_and_streamlit_stopped")
        _require_service(
            str(config["services"]["qdrant"]["url"]).rstrip("/") + "/healthz",
            "qdrant_unavailable",
        )
        _require_service(
            str(config["services"]["meilisearch"]["url"]).rstrip("/") + "/health",
            "meilisearch_unavailable",
        )
        if not args.dry_run:
            _unload_ollama(config)

        output_dir = _path(args.output_dir)
        checkpoint_dir = _path(args.checkpoint_dir)
        child_reports: list[dict[str, Any]] = []
        lock = BatchIngestLock(
            _path("data/state/ingest-locks"),
            REFERENCE_BATCH_ID,
        )
        with lock:
            for stage in channels:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--batch",
                    REFERENCE_BATCH_ID,
                    "--channels",
                    stage,
                    "--stage",
                    stage,
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--output-dir",
                    str(output_dir),
                ]
                if args.limit_videos is not None:
                    command.extend(("--limit-videos", str(args.limit_videos)))
                if args.dry_run:
                    command.append("--dry-run")
                if args.reprocess_stage == stage:
                    command.extend(("--reprocess-stage", stage))
                    for video_id in args.reprocess_video:
                        command.extend(("--reprocess-video", video_id))
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"provisioning_stage_failed:{stage}")
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                if not lines:
                    raise RuntimeError(f"provisioning_stage_report_missing:{stage}")
                child_reports.append(json.loads(lines[-1]))

        state = (
            "complete"
            if all(report.get("state") == "complete" for report in child_reports)
            else "partial"
        )
        report = {
            "schema_version": 1,
            "batch_id": REFERENCE_BATCH_ID,
            "state": state,
            "channels": child_reports,
        }
        _atomic_json(output_dir / "b01-multimodal-report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"status": "failed", "code": _public_error_code(error)},
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
