"""Inspect or explicitly build local ASR topic and detection object graph artifacts.

The default is read-only inspection. Use --sample VIDEO for one video or --full for
the entire selected batch. Sample artifacts live outside the active graph directory.
Existing graph directories are never overwritten; use --output to build a new version.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.object_graph import build_object_index, write_object_index
from src.graph.topic_graph import (
    AsrSegment,
    Keyframe,
    TopicGraphParams,
    build_topic_graph,
    write_topic_graph,
)
from src.utils.io_helpers import load_config, validate_batch_identifier

DEFAULT_OBJECT_ARCHIVES = {
    "aic25-b1-reference": ("objects-aic25-b1.zip",),
    "aic26-b2": ("Objects_M01_M10.zip", "Objects_N001_N100.zip", "Objects_S01.zip"),
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def inspect_inputs(database: Path, batch_id: str) -> dict[str, int]:
    """Count available canonical evidence without loading transcripts or images."""

    if not database.is_file():
        raise FileNotFoundError(f"canonical artifact database is unavailable: {database}")
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        asr_videos, asr_segments = connection.execute(
            "SELECT COUNT(DISTINCT video_id), COUNT(*) FROM canonical_text_segments "
            "WHERE batch_id = ? AND index_ready = 1 AND kind = 'asr'", (batch_id,),
        ).fetchone()
        frame_videos, canonical_frames = connection.execute(
            "SELECT COUNT(DISTINCT video_id), COUNT(*) FROM canonical_frames "
            "WHERE batch_id = ? AND index_ready = 1", (batch_id,),
        ).fetchone()
    return {"asr_videos": int(asr_videos), "asr_segments": int(asr_segments),
            "frame_videos": int(frame_videos), "canonical_frames": int(canonical_frames)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def imported_asr_path(receipt_path: Path, batch_id: str) -> Path | None:
    """Resolve the selected completed import instead of guessing among bundle revisions."""

    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("contract") != "cloud-import-channel-v2" or receipt.get("batch_id") != batch_id
            or receipt.get("channel") != "asr" or not receipt.get("snapshot_import_complete")
            or not receipt.get("coverage_complete")):
        raise ValueError("ASR import receipt is not complete for the selected batch")
    manifest = Path(receipt["manifest_path"])
    if _sha256(manifest) != receipt.get("manifest_sha256"):
        raise ValueError("ASR import manifest integrity mismatch")
    source = manifest.parent / "b01_asr_transcripts.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(f"imported ASR transcripts are unavailable: {source}")
    return source


def read_asr_jsonl(path: Path, batch_id: str, video_id: str | None = None) -> dict[str, list[AsrSegment]]:
    """Stream canonical cloud ASR rows, preserving actual time and frame references."""

    segments: dict[str, list[AsrSegment]] = defaultdict(list)
    seen: set[str] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("batch_id") != batch_id or row.get("kind") != "asr":
                raise ValueError("ASR source contains a different batch or text kind")
            video = validate_batch_identifier(row["video_id"])
            if video_id is not None and video != video_id:
                continue
            identity = str(row.get("segment_id", row.get("id", "")))
            if not identity or identity in seen:
                raise ValueError("ASR source contains a missing or duplicate segment id")
            seen.add(identity)
            start, end = row["start_ms"], row["end_ms"]
            source = row.get("source_frame_idx")
            if (not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start
                    or (source is not None and (not isinstance(source, int) or source < 0))):
                raise ValueError("ASR source contains invalid time/frame identity")
            text = row.get("text_nfc", row.get("text_raw", row.get("text", "")))
            if not isinstance(text, str):
                raise ValueError("ASR source text must be a string")
            if text.strip():
                segments[video].append(AsrSegment(video, start, end, text, source))
    return dict(segments)


def read_inputs(
    database: Path, batch_id: str, video_id: str | None = None,
) -> tuple[dict[str, list[AsrSegment]], dict[str, list[Keyframe]], dict[tuple[str, int], int]]:
    """Read current canonical ASR/frame mappings without migrating or writing the DB."""

    if not database.is_file():
        raise FileNotFoundError(f"canonical artifact database is unavailable: {database}")
    params: list[Any] = [batch_id]
    where = "batch_id = ? AND index_ready = 1"
    if video_id is not None:
        where += " AND video_id = ?"
        params.append(video_id)
    segments: dict[str, list[AsrSegment]] = defaultdict(list)
    frames: dict[str, list[Keyframe]] = defaultdict(list)
    mappings: dict[tuple[str, int], int] = {}
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        for video, start, end, text, source in connection.execute(
            f"SELECT video_id, start_ms, end_ms, text_nfc, source_frame_idx "
            f"FROM canonical_text_segments WHERE {where} AND kind = 'asr' "
            "ORDER BY video_id, start_ms, end_ms, segment_id", params,
        ):
            segments[str(video)].append(AsrSegment(str(video), int(start), int(end), str(text), source))
        for video, ordinal, source, timestamp in connection.execute(
            f"SELECT video_id, keyframe_ordinal, source_frame_idx, timestamp_ms "
            f"FROM canonical_frames WHERE {where} ORDER BY video_id, timestamp_ms, keyframe_ordinal",
            params,
        ):
            frames[str(video)].append(Keyframe(int(ordinal), int(source), int(timestamp)))
            mappings[(str(video), int(ordinal))] = int(source)
        # B1 has additional organizer ordinals sharing a canonical source frame.
        mapping_where = "mapping.batch_id = ? AND frame.index_ready = 1"
        if video_id is not None:
            mapping_where += " AND mapping.video_id = ?"
        for video, ordinal, source in connection.execute(
            "SELECT mapping.video_id, mapping.keyframe_ordinal, mapping.source_frame_idx "
            "FROM canonical_frame_mappings AS mapping JOIN canonical_frames AS frame "
            "ON frame.batch_id = mapping.batch_id AND frame.video_id = mapping.video_id "
            "AND frame.source_frame_idx = mapping.source_frame_idx "
            f"WHERE {mapping_where}", params,
        ):
            mappings[(str(video), int(ordinal))] = int(source)
    return dict(segments), dict(frames), mappings


def read_ordinal_mappings(
    database: Path, batch_id: str, video_id: str | None = None,
) -> dict[tuple[str, int], int]:
    """Read ordinal-keyed frames (B2) without migrating or writing the DB."""

    if not database.is_file():
        raise FileNotFoundError(f"canonical artifact database is unavailable: {database}")
    params: list[Any] = [batch_id]
    where = "batch_id = ? AND index_ready = 1"
    if video_id is not None:
        where += " AND video_id = ?"
        params.append(video_id)
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            f"SELECT video_id, keyframe_ordinal, source_frame_idx FROM canonical_ordinal_frames WHERE {where}",
            params,
        ).fetchall()
    return {(str(video), int(ordinal)): int(source) for video, ordinal, source in rows}


def _progress(stage: str, done: int, total: int) -> None:
    """Report long builds on stderr; stdout stays a single JSON report."""

    if stage != "segment" or done == total or done % 100 == 0:
        print(f"{stage}: {done}/{total}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Inspect by default; load the local text model only for an explicit topic build."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--kind", choices=("topic", "object", "both"), default="topic")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--sample", metavar="VIDEO")
    scope.add_argument("--full", action="store_true")
    scope.add_argument("--inspect", action="store_true")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path, help="New batch artifact directory, containing topic/object")
    parser.add_argument("--objects", type=Path, action="append", help="Organizer Objects ZIP; repeatable")
    parser.add_argument("--asr-transcripts", type=Path,
                        help="Canonical ASR JSONL(.gz); otherwise use DB or completed ASR import receipt")
    args = parser.parse_args(argv)
    batch_id = validate_batch_identifier(args.batch_id)
    if args.sample:
        validate_batch_identifier(args.sample)
    config = load_config()
    graph_config = config.get("graph", {})
    database = _project_path(args.database or config["data"]["artifact_db"])
    report: dict[str, Any] = {
        "batch_id": batch_id, "sample": args.sample,
        "source": "canonical_artifact_database", "mode": "inspect",
    }
    if not args.sample and not args.full:
        report.update(inspect_inputs(database, batch_id))
        print(json.dumps(report, ensure_ascii=False))
        return 0
    # B2 is keyed by organizer ordinal: its detections map directly, while topic graphs
    # need ASR that is not imported and a timing check for its mixed ordinal/VFR identities.
    if batch_id == "aic26-b2" and args.kind == "object":
        segments, frames = {}, {}
        mappings = read_ordinal_mappings(database, batch_id, args.sample)
    elif batch_id == "aic25-b1-reference":
        segments, frames, mappings = read_inputs(database, batch_id, args.sample)
    else:
        raise ValueError("topic graphs support aic25-b1-reference only; B2 supports --kind object")
    report.update(asr_videos=len(segments), asr_segments=sum(len(items) for items in segments.values()),
                  frame_videos=len(frames), keyframe_mappings=len(mappings))
    root = _project_path(graph_config.get("root", "data/graphs"))
    output = _project_path(args.output) if args.output else (
        root / "samples" / args.sample / batch_id if args.sample else root / batch_id
    )
    kinds = ("topic", "object") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        if (output / kind).exists():
            raise FileExistsError(f"graph exists; choose a new --output directory: {output / kind}")
    if "topic" in kinds and not segments:
        source = _project_path(args.asr_transcripts) if args.asr_transcripts else imported_asr_path(
            ROOT / "data/checkpoints/cloud-import" / batch_id / "asr.json", batch_id,
        )
        if source is not None:
            segments = read_asr_jsonl(source, batch_id, args.sample)
            report.update(source="canonical_asr_jsonl", asr_source=str(source),
                          asr_source_sha256=_sha256(source), asr_videos=len(segments),
                          asr_segments=sum(len(items) for items in segments.values()))
    elif "topic" in kinds and args.asr_transcripts:
        source = _project_path(args.asr_transcripts)
        segments = read_asr_jsonl(source, batch_id, args.sample)
        report.update(source="canonical_asr_jsonl", asr_source=str(source),
                      asr_source_sha256=_sha256(source), asr_videos=len(segments),
                      asr_segments=sum(len(items) for items in segments.values()))
    if "topic" in kinds and not segments:
        raise ValueError("no ASR evidence is available; supply --asr-transcripts from a completed import")
    archives = [_project_path(value) for value in (args.objects or DEFAULT_OBJECT_ARCHIVES[batch_id])]
    if "object" in kinds:
        if not mappings:
            raise ValueError("no canonical frame mappings are available for this scope")
        for archive in archives:
            if not archive.is_file():
                raise FileNotFoundError(f"object archive is unavailable: {archive}")
    report.update(mode="build", output=str(output), artifacts={})
    if "topic" in kinds:
        from src.embedding.text_encoder import VietnameseTextEncoder

        model = dict(config["models"]["text"])
        encoder = VietnameseTextEncoder(**{name: model[name] for name in (
            "model_name", "revision", "device", "batch_size", "cache_dir",
            "text_max_chunks", "text_chunk_overlap",
        ) if name in model}, local_files_only=True)
        try:
            artifact = build_topic_graph(
                batch_id, segments, encoder.encode_text, keyframes_by_video=frames,
                params=TopicGraphParams.from_mapping(graph_config.get("topic")),
                encoder_fingerprint=encoder.fingerprint, progress=_progress,
            )
            if not artifact.topics:
                raise ValueError("ASR segments have no usable frame identity")
            artifact.extra.update(source=report["source"], sample=args.sample,
                                  asr_source_sha256=report.get("asr_source_sha256"))
            report["artifacts"]["topic"] = write_topic_graph(artifact, output / "topic")
        finally:
            encoder.unload()
    if "object" in kinds:
        artifact = build_object_index(batch_id, archives, lambda video, ordinal: mappings.get((video, ordinal)),
                                      progress=_progress)
        if not artifact.frames:
            raise ValueError("no object detections matched the canonical frame mappings")
        report["artifacts"]["object"] = write_object_index(artifact, output / "object")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
