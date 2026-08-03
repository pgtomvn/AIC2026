"""Calibrate adaptive extraction on an explicitly selected raw-video subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.calibration import persist_calibration_report, run_extraction_calibration
from src.extraction.keyframe_extractor import KeyframeExtractor
from src.ingest.artifacts import ArtifactRepository
from src.ingest.native_pipeline import ScratchManager, VideoSource, discover_video_sources
from src.utils.io_helpers import load_config, resolve_path

_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate scene threshold/max-gap without extracting the full corpus"
    )
    parser.add_argument("--reference-batch", required=True)
    parser.add_argument("--video-id", action="append", required=True, help="Repeat explicitly")
    parser.add_argument("--raw-root", type=Path, help="Defaults to data.raw_videos")
    parser.add_argument("--artifact-db", type=Path, help="Defaults to data.artifact_db")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/calibration/extraction-selection.json"),
    )
    return parser


def _find_videos(raw_root: Path, video_ids: list[str]) -> dict[str, VideoSource]:
    wanted = set(video_ids)
    matches: dict[str, list[VideoSource]] = {video_id: [] for video_id in wanted}
    candidates = [raw_root] if raw_root.is_file() else sorted(raw_root.rglob("*"))
    for path in candidates:
        if not path.is_file() or path.name.casefold().endswith(".crdownload"):
            continue
        if path.suffix.casefold() in _VIDEO_EXTENSIONS and path.stem in wanted:
            matches[path.stem].append(VideoSource(path.stem, path))
        elif path.suffix.casefold() == ".zip":
            for source in discover_video_sources((path,)):
                if source.video_id in wanted:
                    matches[source.video_id].append(source)
    missing = sorted(key for key, values in matches.items() if not values)
    duplicates = {key: values for key, values in matches.items() if len(values) != 1}
    if missing:
        raise FileNotFoundError(f"selected raw videos not found under {raw_root}: {missing}")
    if duplicates:
        detail = {key: [source.uri for source in values] for key, values in duplicates.items()}
        raise ValueError(f"selected video IDs are ambiguous: {detail}")
    return {key: values[0] for key, values in matches.items()}


def _source_fingerprint(source: VideoSource) -> dict[str, object]:
    digest = hashlib.sha256()
    if source.member is None:
        with source.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        size = source.path.stat().st_size
    else:
        size = int(source.size_bytes or 0)
        with zipfile.ZipFile(source.path) as archive, archive.open(source.member) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return {"source_uri": source.uri, "source_sha256": digest.hexdigest(), "size_bytes": size}


def main(argv: list[str] | None = None) -> int:
    """Run the configured grid, persist the decision, and print the full report."""

    args = _parser().parse_args(argv)
    config = load_config()
    ingestion = config["ingestion"]
    data = config["data"]
    raw_root = (args.raw_root or resolve_path(data["raw_videos"])).resolve()
    artifact_db = (args.artifact_db or resolve_path(data["artifact_db"])).resolve()
    sources = _find_videos(raw_root, args.video_id)
    videos = {video_id: source.path for video_id, source in sources.items()}

    repository = ArtifactRepository(artifact_db)
    repository.initialize()
    references = {
        video_id: tuple(
            frame.timestamp_ms
            for frame in repository.iter_frames(
                batch_id=args.reference_batch, video_id=video_id
            )
        )
        for video_id in sources
    }
    scratch_manager = ScratchManager(
        resolve_path(data["scratch"]),
        limit_bytes=int(float(data["scratch_limit_gib"]) * 1024**3),
        reserve_bytes=int(float(data["free_space_reserve_gib"]) * 1024**3),
    )

    def configured_runner(
        video_path: Path,
        video_id: str,
        threshold: float,
        gap: float,
        scratch: Path,
    ) -> tuple[int, ...]:
        del video_path
        with scratch_manager.materialize(sources[video_id]) as materialized:
            extractor = KeyframeExtractor(
                output_dir=scratch,
                batch_id="calibration",
                min_scene_change=threshold,
                max_gap_seconds=gap,
                min_separation_seconds=float(ingestion["min_separation_seconds"]),
                resize_max_dim=int(data["thumbnail_max_dim"]),
            )
            return tuple(
                int(record["timestamp_ms"])
                for record in extractor.extract(materialized, video_id=video_id)
            )

    report = run_extraction_calibration(
        videos,
        references,
        scene_thresholds=ingestion["scene_threshold_grid"],
        max_gaps_seconds=ingestion["max_gap_grid_seconds"],
        minimum_coverage=float(ingestion["calibration_min_coverage"]),
        tolerance_ms=int(ingestion["calibration_tolerance_ms"]),
        fallback=(
            float(ingestion["scene_threshold"]),
            float(ingestion["max_gap_seconds"]),
        ),
        runner=configured_runner,
        input_fingerprints={
            video_id: _source_fingerprint(source) for video_id, source in sources.items()
        },
        extractor_settings={
            "min_separation_seconds": float(ingestion["min_separation_seconds"]),
            "thumbnail_max_dim": int(data["thumbnail_max_dim"]),
            "extractor_version": "adaptive-v2",
        },
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    persist_calibration_report(report, output)
    payload = report.to_dict()
    payload["output"] = str(output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
