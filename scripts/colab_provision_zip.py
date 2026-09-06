#!/usr/bin/env python3
"""Produce resumable B01 multimodal artifacts directly from ZIP archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding.siglip_encoder import SigLIP2Encoder
from src.extraction.asr_engine import ASREngine
from src.extraction.rapid_ocr_engine import RapidOCREngine
from src.ingest.cloud_artifacts import (
    CloudArtifactProducer,
    asr_profile_settings,
    cloud_model_contracts,
    load_cloud_partition,
)
from src.utils.io_helpers import load_config


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stages(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip().casefold() for item in value.split(",") if item.strip())
    allowed = ("siglip2", "ocr", "asr")
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(item not in allowed for item in requested)
    ):
        raise argparse.ArgumentTypeError("stages must be a unique subset of siglip2,ocr,asr")
    return tuple(stage for stage in allowed if stage in requested)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keyframes-zip",
        type=Path,
        action="append",
        default=[],
        help="Keyframe ZIP path; repeat for every B01 ZIP pack",
    )
    parser.add_argument(
        "--videos-zip",
        type=Path,
        action="append",
        default=[],
        help="Video ZIP path; repeat for every B01 ZIP pack used by ASR",
    )
    parser.add_argument(
        "--maps-zip",
        type=Path,
        help="Organizer CSV-map ZIP; defaults to the first keyframe ZIP",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--stages", type=_stages, default=_stages("siglip2,ocr,asr"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--asr-profile",
        choices=("production", "reference"),
        default="reference",
        help="Pinned ASR model/inference contract; production requires CUDA",
    )
    parser.add_argument("--siglip-batch-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        help="Ephemeral local ASR scratch; defaults outside the output/Drive tree",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="Atomic progress heartbeat; defaults to output/cloud_work/progress.json",
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--finalize-channels",
        type=_stages,
        default=_stages("siglip2,ocr,asr"),
        help="Completed channels to include in the progressive manifest",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-videos", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--partition-manifest", type=Path)
    parser.add_argument("--partition-id")
    parser.add_argument("--worker-slot", type=int)
    parser.add_argument("--worker-count", type=int)
    return parser


def _producer(args: argparse.Namespace, config: dict[str, Any]) -> CloudArtifactProducer:
    keyframes = tuple(_path(value) for value in args.keyframes_zip)
    if not keyframes:
        raise ValueError("at least one --keyframes-zip is required")
    maps = _path(args.maps_zip) if args.maps_zip is not None else keyframes[0]
    partition: dict[str, Any] | None = None
    partition_values = (
        args.partition_manifest,
        args.partition_id,
        args.worker_slot,
        args.worker_count,
    )
    if any(value is not None for value in partition_values):
        if any(value is None for value in partition_values):
            raise ValueError(
                "partition manifest, id, worker slot, and worker count are all required"
            )
        if len(args.stages) != 1 or args.finalize or args.finalize_only:
            raise ValueError("a partition worker must run exactly one extraction stage")
        partition = load_cloud_partition(
            _path(args.partition_manifest),
            batch_id=args.batch_id,
            stage=args.stages[0],
            partition_id=args.partition_id,
            worker_slot=args.worker_slot,
            worker_count=args.worker_count,
        )
        archive_name = str(partition["archive"])
        if args.stages[0] == "ocr":
            keyframes = tuple(value for value in keyframes if value.name == archive_name)
            if len(keyframes) != 1:
                raise ValueError("partition keyframe archive is missing or duplicated")
        else:
            videos = tuple(
                _path(value) for value in args.videos_zip if _path(value).name == archive_name
            )
            if len(videos) != 1:
                raise ValueError("partition video archive is missing or duplicated")
            args.videos_zip = list(videos)
    specification = config["readiness"]["batches"].get(args.batch_id, {})
    expected_videos = args.expected_videos
    expected_frames = args.expected_frames
    if expected_videos is None:
        expected_videos = int(specification.get("expected_videos", 0)) or None
    if expected_frames is None:
        expected_frames = int(specification.get("expected_canonical_frames", 0)) or None
    progress_path = _path(args.progress_file) if args.progress_file is not None else None
    if progress_path is None and partition is not None:
        progress_path = (
            _path(args.output_dir)
            / "cloud_work"
            / "progress"
            / args.stages[0]
            / f"{args.partition_id}.json"
        )
    return CloudArtifactProducer(
        map_archive=maps,
        keyframe_archives=keyframes,
        video_archives=tuple(_path(value) for value in args.videos_zip),
        output_dir=_path(args.output_dir),
        batch_id=args.batch_id,
        resume=args.resume,
        expected_videos=expected_videos,
        expected_frames=expected_frames,
        scratch_dir=_path(args.scratch_dir) if args.scratch_dir is not None else None,
        progress_path=progress_path,
        progress_callback=lambda event: print(
            json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True
        ),
        model_contracts=cloud_model_contracts(config, asr_profile=args.asr_profile),
        selected_video_ids=(partition["videos"] if partition is not None else None),
        partition_id=(args.partition_id if partition is not None else None),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config()
    producer = _producer(args, config)
    if args.finalize_only:
        manifest = producer.finalize(
            channels=args.finalize_channels,
            allow_partial=args.allow_partial,
        )
        print(
            json.dumps(
                {
                    "state": "finalized",
                    "videos": manifest["videos"],
                    "vectors": manifest["vector"]["count"],
                    "text": {name: value["count"] for name, value in manifest["text"].items()},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    device = _device(args.device)
    reference = config["models"]["reference_provisioning"]
    visual = config["models"]["visual"]
    reports: dict[str, Any] = {}
    for stage in args.stages:
        engine: Any = None
        try:
            if stage == "siglip2":
                engine = SigLIP2Encoder(
                    model_name=str(visual["model_name"]),
                    revision=str(visual["revision"]),
                    device=device,
                    batch_size=args.siglip_batch_size,
                    max_batch_size=args.siglip_batch_size,
                    cache_dir=_path(visual["cache_dir"]),
                    local_files_only=not args.allow_model_download,
                )
                reports[stage] = producer.run_siglip2(
                    engine,
                    batch_size=args.siglip_batch_size,
                )
            elif stage == "ocr":
                values = reference["rapidocr"]
                engine = RapidOCREngine(
                    model_dir=_path(values["model_dir"]),
                    confidence_threshold=max(0.5, float(values["confidence_threshold"])),
                    min_text_characters=int(values["min_text_characters"]),
                    local_files_only=not args.allow_model_download,
                    package_version=str(values["package_version"]),
                    source_revision=str(values["source_revision"]),
                )
                reports[stage] = producer.run_ocr(engine)
            else:
                values = asr_profile_settings(config, args.asr_profile)
                if args.asr_profile == "production":
                    if device != "cuda":
                        raise RuntimeError("production ASR requires --device cuda")
                    try:
                        import torch
                    except ImportError as error:
                        raise RuntimeError("production ASR requires a CUDA-enabled torch build") from error
                    if not torch.cuda.is_available():
                        raise RuntimeError("production ASR requires an available CUDA device")
                model_path = _path(values["local_path"])
                engine = ASREngine(
                    str(values["hub_model_id"]),
                    values.get("language"),
                    device,
                    compute_type=str(values["compute_type"]),
                    beam_size=int(values["beam_size"]),
                    retry_beam_size=int(values["retry_beam_size"]),
                    vad_filter=bool(values.get("vad_filter", True)),
                    model_dir=model_path.parent,
                    model_path=model_path if model_path.is_dir() else None,
                    model_revision=str(values["revision"]),
                    local_files_only=not args.allow_model_download,
                    profile_name=args.asr_profile,
                    vad_parameters=dict(values.get("vad_parameters", {})),
                    condition_on_previous_text=bool(
                        values.get("condition_on_previous_text", True)
                    ),
                )
                reports[stage] = producer.run_asr(engine, ffmpeg_path=args.ffmpeg_path)
        finally:
            unload = getattr(engine, "unload", None)
            if callable(unload):
                unload()
    if args.finalize:
        manifest = producer.finalize(
            channels=args.finalize_channels,
            allow_partial=args.allow_partial,
        )
        reports["finalize"] = {
            "videos": manifest["videos"],
            "vectors": manifest["vector"]["count"],
        }
    print(json.dumps(reports, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
