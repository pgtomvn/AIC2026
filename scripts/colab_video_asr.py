"""Resumable, video-only ASR staging before organizer keyframe maps exist.

This does not finalize or import a multimodal bundle. Source frame IDs remain
null until a separate, validated canonical mapping step becomes possible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import sys
import tempfile
import threading
import time
import wave
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import TextSegment
from src.ingest.cloud_artifacts import (
    ASR_AUDIO_EXTRACTION_CONTRACT,
    CloudArtifactProducer,
    _atomic_json,
    _atomic_jsonl_gzip,
    _stable_json_hash,
    _text_document,
    asr_profile_settings,
    sha256_file,
)
from src.utils.io_helpers import load_config

CONTRACT = "video-only-asr-staging-v1"
EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def inventory(raw: Path, batch_id: str, workers: int, ignored: tuple[str, ...] = (),
              *, archive_names: tuple[str, ...] | None = None) -> dict:
    """Read ZIP directories only; no corpus hash, extraction or fake frame map."""
    if not SAFE_ID.fullmatch(batch_id) or not 1 <= workers <= 8:
        raise ValueError("Invalid batch ID or worker count")
    entries: list[dict[str, Any]] = []
    archives = []
    seen: set[str] = set()
    for path in sorted(raw.glob("*.zip")):
        if archive_names is not None and path.name not in archive_names:
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing symlink archive: {path.name}")
        size = path.stat().st_size
        if path.name in ignored:
            if size != 0:
                raise ValueError(f"Ignored empty archive is now non-empty: {path.name}")
            continue
        archive_videos = 0
        with zipfile.ZipFile(path) as handle:
            for member in sorted(handle.infolist(), key=lambda value: value.filename):
                name = PurePosixPath(member.filename.replace("\\", "/"))
                if member.is_dir() or name.suffix.lower() not in EXTENSIONS:
                    continue
                if name.is_absolute() or ".." in name.parts or member.flag_bits & 1:
                    raise ValueError(f"Unsafe or encrypted video member: {member.filename}")
                video_id = name.stem
                if not SAFE_ID.fullmatch(video_id) or member.file_size <= 0:
                    raise ValueError(f"Invalid video member: {member.filename}")
                if video_id.casefold() in seen:
                    raise ValueError(f"Duplicate video ID: {video_id}")
                seen.add(video_id.casefold())
                entries.append({
                    "video_id": video_id, "archive": path.name, "member": member.filename,
                    "size_bytes": member.file_size, "compressed_bytes": member.compress_size,
                    "crc32": member.CRC,
                })
                archive_videos += 1
        if not archive_videos:
            raise ValueError(f"ZIP contains no videos: {path.name}")
        archives.append({"name": path.name, "size_bytes": size, "videos": archive_videos})
    if not entries:
        raise ValueError("No video archives found")
    loads = [0] * workers
    for entry in sorted(entries, key=lambda row: (-row["size_bytes"], row["video_id"])):
        slot = min(range(workers), key=lambda index: (loads[index], index))
        entry["worker_id"] = slot
        loads[slot] += entry["size_bytes"]
    plan = {
        "contract": CONTRACT, "batch_id": batch_id, "worker_count": workers,
        "expected_videos": len(entries), "archives": archives,
        "videos": sorted(entries, key=lambda row: (row["archive"], row["video_id"])),
        "worker_load_bytes": loads, "mapping_status": "awaiting_canonical_keyframes",
        "ignored_empty_archives": sorted(ignored),
    }
    plan["fingerprint"] = _stable_json_hash(plan)
    return plan


def inventory_b02_ms(raw: Path, manifest: dict, workers: int) -> dict:
    """Explicit B02 M/S scope; N traffic is excluded, never called no-audio inference."""
    from src.ingest.b02_reference import _checked_sha256_record

    _checked_sha256_record(manifest, "manifest_sha256")
    if manifest.get("batch_id") != "aic26-b2":
        raise ValueError("Expected B02 canonical manifest")
    expected = {v for v in manifest["videos"] if v.startswith(("M", "S"))}
    excluded = sorted(v for v in manifest["videos"] if v.startswith("N"))
    if len(expected) != 316 or len(excluded) != 298:
        raise ValueError("Expected 316 M/S and 298 excluded N videos")
    archives = tuple(p.name for p in raw.glob("*.zip")
                     if re.fullmatch(r"Videos?_(?:M\d{2}|S\d{2})\.zip", p.name))
    plan = inventory(raw, "aic26-b2", workers, archive_names=archives)
    actual = {v["video_id"] for v in plan["videos"]}
    if actual != expected:
        raise ValueError(f"M/S source coverage mismatch; missing={sorted(expected - actual)}; "
                         f"unexpected={sorted(actual - expected)}")
    plan.pop("fingerprint")
    plan.update(scope="b02-ms-only-v1", excluded_video_ids=excluded,
                exclusion_reason="user_declared_N_traffic_without_audio",
                canonical_manifest_sha256=manifest["manifest_sha256"])
    plan["fingerprint"] = _stable_json_hash(plan)
    return plan


def freeze_plan(path: Path, plan: dict) -> None:
    """Never silently replace a plan when inputs or assignments change."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Frozen plan mismatch; do not mix source versions")
    else:
        _atomic_json(path, plan)


def checkpoint_valid(receipt: Path, shard: Path, identity: dict) -> bool:
    """Fail closed on an existing inconsistent checkpoint; never overwrite it."""
    if not receipt.exists():
        if shard.exists():
            raise ValueError(f"Orphan shard needs audit: {shard.name}")
        return False
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if (saved.get("identity") != identity or not shard.is_file()
            or saved.get("size_bytes") != shard.stat().st_size
            or saved.get("sha256") != sha256_file(shard)):
        raise ValueError(f"Checkpoint mismatch needs audit: {receipt.name}")
    return True


def transcript_documents(batch_id: str, video_id: str, model: str, values: list) -> list:
    """Keep timestamp evidence without inventing a canonical frame ordinal."""
    result = []
    for index, value in enumerate(values):
        if not str(value.get("text_raw", "")).strip():
            continue
        logprob = float(value.get("avg_logprob", 0))
        if not math.isfinite(logprob):
            raise ValueError("ASR confidence must be finite")
        segment = TextSegment(
            segment_id=f"asr:{batch_id}:{video_id}:{index}", batch_id=batch_id,
            video_id=video_id, kind="asr", start_ms=int(value["start_ms"]),
            end_ms=int(value["end_ms"]), source_frame_idx=None,
            text_raw=str(value["text_raw"]), text_nfc=str(value["text_nfc"]),
            text_accent_folded=str(value["text_accent_folded"]),
            confidence=math.exp(min(0, logprob)), model_version=model,
        )
        result.append(_text_document(segment))
    return result


def execution_settings(config: dict, device: str) -> dict:
    """CPU is explicit opt-in; preserve the model/revision and decoding settings."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be explicitly cpu or cuda")
    values = asr_profile_settings(config, "production")
    values["device"] = device
    values["compute_type"] = "int8" if device == "cpu" else "int8_float16"
    return values


class WorkerLease:
    """Exclusive visible Drive lock; stale locks require explicit audit/recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> WorkerLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as error:
            raise RuntimeError(f"Worker already leased; audit before recovery: {self.path}") from error
        _atomic_json(self.path / "owner.json", {
            "host": socket.gethostname(), "pid": os.getpid(), "started_unix": time.time(),
        })
        return self

    def __exit__(self, *_: object) -> None:
        # Only these two files belong to this invocation. Never delete a tree.
        (self.path / "owner.json").unlink()
        self.path.rmdir()


def run_worker(
    raw: Path, output: Path, scratch: Path, plan: dict, worker_id: int, engine: Any,
    source_sha256: str, *, max_videos: int | None = None,
) -> dict:
    """Extract one seekable video at a time and commit each receipt last."""
    if worker_id not in range(plan["worker_count"]):
        raise ValueError("Worker ID outside frozen plan")
    if scratch.resolve() == output.resolve() or output.resolve() in scratch.resolve().parents:
        raise ValueError("Scratch must be separate from durable output")
    assigned = [row for row in plan["videos"] if row["worker_id"] == worker_id]
    model = CloudArtifactProducer._model_identity(engine, "asr")
    progress = {"contract": CONTRACT, "batch_id": plan["batch_id"],
                "plan_fingerprint": plan["fingerprint"], "worker_id": worker_id,
                "assigned_videos": len(assigned), "completed_videos": 0, "documents": 0,
                "resumed_videos": 0, "state": "starting", "current_video": None,
                "import_ready": False, "model": model, "source_sha256": source_sha256}
    progress_path = output / "progress" / f"worker-{worker_id}.json"
    mutex, stop = threading.Lock(), threading.Event()
    heartbeat_errors: list[Exception] = []

    def report(**updates: Any) -> None:
        with mutex:
            progress.update(updates)
            progress["heartbeat_unix"] = time.time()
            _atomic_json(progress_path, progress)

    def heartbeat() -> None:
        while not stop.wait(30):
            try:
                report()
            except Exception as error:
                heartbeat_errors.append(error)
                return

    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    with WorkerLease(output / "leases" / f"worker-{worker_id}.lock"):
        report()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for row in assigned:
                if heartbeat_errors:
                    raise RuntimeError("Drive heartbeat failed") from heartbeat_errors[0]
                video_id = row["video_id"]
                identity = {"contract": CONTRACT, "batch_id": plan["batch_id"],
                            "plan_fingerprint": plan["fingerprint"], "source": row,
                            "model": model, "source_sha256": source_sha256,
                            "extraction_contract": ASR_AUDIO_EXTRACTION_CONTRACT}
                receipt = output / "receipts" / f"{video_id}.json"
                shard = output / "shards" / f"{video_id}.jsonl.gz"
                if checkpoint_valid(receipt, shard, identity):
                    saved = json.loads(receipt.read_text(encoding="utf-8"))
                    count = saved["transcript_segment_count"]
                    report(completed_videos=progress["completed_videos"] + 1,
                           documents=progress["documents"] + count,
                           resumed_videos=progress["resumed_videos"] + 1)
                    continue
                report(state="extracting_audio", current_video=video_id)
                video_started = time.monotonic()
                duration_seconds = 0.0
                inference_seconds = 0.0
                with zipfile.ZipFile(raw / row["archive"]) as archive:
                    info = archive.getinfo(row["member"])
                    if info.CRC != row["crc32"] or info.file_size != row["size_bytes"]:
                        raise ValueError("Video source changed after inventory")
                    with tempfile.TemporaryDirectory(prefix=f"{video_id}-", dir=scratch) as tmp:
                        audio = Path(tmp) / "audio.wav"
                        has_audio = CloudArtifactProducer._stream_audio_member(
                            archive, row["member"], audio, ffmpeg_path="ffmpeg",
                        )
                        if has_audio:
                            with wave.open(str(audio), "rb") as wav:
                                duration_seconds = wav.getnframes() / wav.getframerate()
                        report(state="transcribing")
                        inference_started = time.monotonic()
                        transcript = engine.transcribe(audio) if has_audio else {"segments": []}
                        inference_seconds = time.monotonic() - inference_started
                docs = transcript_documents(
                    plan["batch_id"], video_id, model["name"], transcript["segments"],
                )
                count = _atomic_jsonl_gzip(shard, docs)
                reason = None if count else "no_speech_detected" if has_audio else "no_audio_stream"
                _atomic_json(receipt, {
                    "schema_version": 1, "contract": CONTRACT, "identity": identity,
                    "batch_id": plan["batch_id"], "video_id": video_id,
                    "audio_stream_present": has_audio, "transcript_segment_count": count,
                    "zero_transcript_reason": reason, "sha256": sha256_file(shard),
                    "size_bytes": shard.stat().st_size, "artifact": f"shards/{shard.name}",
                    "mapping_status": "awaiting_canonical_keyframes", "import_ready": False,
                    "execution_device": getattr(engine, "device", "unknown"),
                    "compute_type": getattr(engine, "compute_type", "unknown"),
                    "audio_duration_seconds": duration_seconds,
                    "inference_seconds": inference_seconds,
                    "elapsed_seconds": time.monotonic() - video_started,
                    "realtime_factor": (inference_seconds / duration_seconds
                                        if duration_seconds else None),
                })
                report(completed_videos=progress["completed_videos"] + 1,
                       documents=progress["documents"] + count, state="checkpoint_saved",
                       last_audio_seconds=duration_seconds,
                       last_inference_seconds=inference_seconds,
                       last_realtime_factor=(inference_seconds / duration_seconds
                                             if duration_seconds else None))
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                if max_videos is not None and (
                    progress["completed_videos"] - progress["resumed_videos"] >= max_videos
                ):
                    break
            report(state="worker_complete" if progress["completed_videos"] == len(assigned)
                   else "sample_complete", current_video=None)
        except BaseException as error:
            report(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            stop.set()
            thread.join(timeout=5)
    return progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--ignore-empty", action="append", default=[])
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--resume", action="store_true", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/content/b02-asr-scratch"))
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--b02-ms-manifest", type=Path,
                        help="B02 only: validate all 316 M/S videos and exclude N traffic")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        parser.error("Source SHA-256 is required")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("max-videos must be positive")
    if args.b02_ms_manifest:
        if args.batch_id != "aic26-b2" or args.ignore_empty:
            parser.error("M/S scope requires aic26-b2 and forbids ignored source archives")
        manifest = json.loads(args.b02_ms_manifest.read_text(encoding="utf-8"))
        plan = inventory_b02_ms(args.raw_dir, manifest, args.worker_count)
    else:
        plan = inventory(args.raw_dir, args.batch_id, args.worker_count, tuple(args.ignore_empty))
    freeze_plan(args.output_dir / "plan.json", plan)
    print(json.dumps({key: value for key, value in plan.items() if key != "videos"}), flush=True)
    if args.inventory_only:
        return 0
    from src.extraction.asr_engine import ASREngine

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU requires explicit --device cpu")
    values = execution_settings(load_config(), args.device)
    engine = ASREngine(
        str(values["hub_model_id"]), values.get("language"), args.device,
        compute_type=values["compute_type"], beam_size=values["beam_size"],
        retry_beam_size=values["retry_beam_size"], vad_filter=values["vad_filter"],
        model_path=ROOT / values["local_path"], model_revision=values["revision"],
        local_files_only=True, profile_name="production",
        vad_parameters=values["vad_parameters"],
        condition_on_previous_text=values["condition_on_previous_text"],
    )
    try:
        report = run_worker(args.raw_dir, args.output_dir, args.scratch_dir, plan,
                            args.worker_id, engine, args.source_sha256,
                            max_videos=args.max_videos)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        engine.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
