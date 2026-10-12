"""B02 ordinal-keyed OCR staging, streamed from immutable keyframe ZIP members."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.colab_video_asr import WorkerLease, freeze_plan
from src.extraction.vietocr_engine import VietOCREngine, digest_file
from src.ingest.b02_keyframe_mapping import parse_b02_keyframe_row
from src.ingest.b02_reference import _archive_fingerprint, _checked_sha256_record
from src.ingest.cloud_artifacts import _atomic_json, _atomic_jsonl_gzip, _stable_json_hash

CONTRACT = "b02-ordinal-ocr-v1"


def inventory(raw: Path, manifest: dict, workers: int = 2) -> dict:
    """Check ZIP directories against P0, then bind every image to its canonical map row."""
    _checked_sha256_record(manifest, "manifest_sha256")
    if workers not in (1, 2) or manifest.get("batch_id") != "aic26-b2":
        raise ValueError("Invalid B02 worker/source contract")
    names = sorted({v["keyframe_archive"] for v in manifest["videos"].values()})
    map_name = "aic26-b2-map-keyframes.zip"
    for name in [*names, map_name]:
        if PurePosixPath(name).name != name:
            raise ValueError("Unsafe archive name")
        size, fingerprint = _archive_fingerprint(raw / name)
        expected = manifest["inputs"][name]
        if (size != expected["archive_size_bytes"]
                or fingerprint != expected["central_directory_sha256"]):
            raise ValueError(f"Source archive differs from P0: {name}")
    images: dict[str, dict[int, dict]] = {}
    for name in names:
        with zipfile.ZipFile(raw / name) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename)
                if member.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                video, ordinal_text = member.parent.name, member.stem
                if not re.fullmatch(r"[MNS][A-Za-z0-9_-]+", video) or not ordinal_text.isdigit():
                    raise ValueError(f"Unrecognized frame identity: {info.filename}")
                ordinal = int(ordinal_text)
                if info.flag_bits & 1 or not 0 < info.file_size <= 50_000_000:
                    raise ValueError("Encrypted/empty/oversized keyframe")
                frames = images.setdefault(video, {})
                if ordinal < 1 or ordinal in frames:
                    raise ValueError("Duplicate ordinal identity")
                frames[ordinal] = {"member": info.filename, "archive": name,
                                   "crc32": info.CRC, "size_bytes": info.file_size}
    videos = []
    with zipfile.ZipFile(raw / map_name) as maps:
        for video, expected in sorted(manifest["videos"].items()):
            frames = images.pop(video, {})
            count = expected["keyframe_count"]
            if set(frames) != set(range(1, count + 1)):
                raise ValueError(f"Incomplete ordinal coverage: {video}")
            with maps.open(expected["map_member"]) as stream:
                rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
            mapped = [parse_b02_keyframe_row("aic26-b2", video, row) for row in rows]
            if (len(mapped) != count
                    or {m.identity.keyframe_ordinal for m in mapped} != set(frames)):
                raise ValueError(f"Map coverage mismatch: {video}")
            result = []
            for frame in sorted(mapped, key=lambda m: m.identity.keyframe_ordinal):
                ordinal = frame.identity.keyframe_ordinal
                entry = frames[ordinal]
                if entry["archive"] != expected["keyframe_archive"]:
                    raise ValueError("Keyframe archive mismatch")
                result.append(dict(entry, keyframe_ordinal=ordinal,
                                   timestamp_ms=frame.timestamp_ms,
                                   source_frame_idx=frame.organizer_frame_idx))
            videos.append({"video_id": video, "frames": result})
    if images:
        raise ValueError("Unexpected video images")
    if len(videos) != 614 or sum(len(v["frames"]) for v in videos) != 220270:
        raise ValueError("B02 contract must be 614 videos / 220270 keyframes")
    loads = [0] * workers
    for video in sorted(videos, key=lambda v: (-len(v["frames"]), v["video_id"])):
        worker = min(range(workers), key=lambda w: (loads[w], w))
        video["worker_id"] = worker
        loads[worker] += len(video["frames"])
    plan = {"contract": CONTRACT, "batch_id": "aic26-b2", "worker_count": workers,
            "p0_manifest_sha256": manifest["manifest_sha256"], "videos": videos}
    return dict(plan, fingerprint=_stable_json_hash(plan))


def pilot_selection(plan: dict, per_group: int = 100) -> dict[str, list[int]]:
    """Deterministic coverage across M/N/S, spread across videos and time."""
    selected: dict[str, list[int]] = {}
    for group in "MNS":
        pool = [(v["video_id"], f["keyframe_ordinal"]) for v in plan["videos"]
                if v["video_id"].startswith(group) for f in v["frames"]]
        if not pool:
            raise ValueError(f"Missing pilot group {group}")
        for i in range(min(per_group, len(pool))):
            video, ordinal = pool[(i * len(pool)) // min(per_group, len(pool))]
            selected.setdefault(video, []).append(ordinal)
    return selected


def validate_checkpoint(receipt: Path, shard: Path, identity: dict) -> bool:
    if not receipt.exists():
        if shard.exists():
            raise ValueError("Orphan OCR shard requires audit")
        return False
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if (saved.get("identity") != identity or not shard.is_file()
            or saved.get("sha256") != digest_file(shard)
            or saved.get("size_bytes") != shard.stat().st_size
            or saved.get("scanned_frames") != len(identity["ordinals"])):
        raise ValueError("OCR checkpoint mismatch; refusing overwrite")
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        count = sum(1 for _ in stream)
    if saved.get("documents") != count:
        raise ValueError("OCR checkpoint document count mismatch")
    return True


def run_worker(raw: Path, output: Path, plan: dict, worker_id: int, engine: Any,
               source_sha: str, *, full: bool = False) -> dict:
    """Commit shard first and receipt last; any failed frame aborts the video."""
    if worker_id not in range(plan["worker_count"]):
        raise ValueError("Invalid worker ID")
    model = engine.identity
    mode = "full" if full else "pilot"
    binding = {"plan_fingerprint": plan["fingerprint"], "source_sha256": source_sha,
               "model": model}
    if full:
        gate_path = output / "pilot-approval.json"
        if not gate_path.is_file():
            raise RuntimeError("Pilot review required before full OCR")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("binding") != binding or gate.get("approved") is not True:
            raise ValueError("Pilot approval does not match source/model/plan")
    selected = pilot_selection(plan)
    progress = {"mode": mode, "worker_id": worker_id, "completed_videos": 0,
                "scanned_frames": 0, "documents": 0, "state": "starting",
                "import_ready": False, **binding}
    destination = output / mode
    stop, mutex = threading.Event(), threading.Lock()
    failures: list[Exception] = []

    def report(**values: Any) -> None:
        with mutex:
            progress.update(values, heartbeat_unix=time.time())
            _atomic_json(destination / "progress" / f"worker-{worker_id}.json", progress)

    def heartbeat() -> None:
        while not stop.wait(30):
            try:
                report()
            except Exception as error:
                failures.append(error)
                return

    with WorkerLease(output / "locks" / f"worker-{worker_id}.lock"):
        freeze_plan(output / "plan.json", plan)
        freeze_plan(output / "run-binding.json", binding)
        report()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for video in plan["videos"]:
                name = video["video_id"]
                if video["worker_id"] != worker_id or (not full and name not in selected):
                    continue
                frames = [f for f in video["frames"]
                          if full or f["keyframe_ordinal"] in selected[name]]
                identity = {**binding, "batch_id": "aic26-b2", "video_id": name,
                            "mode": mode, "ordinals": [f["keyframe_ordinal"] for f in frames]}
                receipt = destination / "receipts" / f"{name}.json"
                shard = destination / "shards" / f"{name}.jsonl.gz"
                report(current_video=name, state="checking_checkpoint")
                if not validate_checkpoint(receipt, shard, identity):
                    docs = []
                    started = time.monotonic()
                    with zipfile.ZipFile(raw / frames[0]["archive"]) as archive:
                        for frame in frames:
                            if failures:
                                raise RuntimeError("Drive heartbeat write failed") from failures[0]
                            info = archive.getinfo(frame["member"])
                            if (info.CRC != frame["crc32"] or info.file_size != frame["size_bytes"]):
                                raise ValueError("Frame source changed")
                            with mutex:
                                progress.update(state="recognizing",
                                                current_ordinal=frame["keyframe_ordinal"])
                            # read() also verifies the ZIP member CRC; no full corpus extraction.
                            with Image.open(io.BytesIO(archive.read(info))) as image:
                                image.load()
                                rows = engine.extract_text(image)
                                width, height = image.size
                            for index, row in enumerate(rows):
                                ordinal = frame["keyframe_ordinal"]
                                docs.append({**row, "segment_id": f"ocr:aic26-b2:{name}:{ordinal}:{index}",
                                             "batch_id": "aic26-b2", "video_id": name,
                                             "keyframe_ordinal": ordinal,
                                             "source_frame_idx": frame["source_frame_idx"],
                                             "start_ms": frame["timestamp_ms"],
                                             "end_ms": frame["timestamp_ms"], "kind": "ocr",
                                             "image_width": width, "image_height": height,
                                             "model_version": _stable_json_hash(model)})
                    count = _atomic_jsonl_gzip(shard, docs)
                    _atomic_json(receipt, {"schema_version": 1, "contract": CONTRACT,
                                          "identity": identity, "documents": count,
                                          "scanned_frames": len(frames),
                                          "zero_text": count == 0, "import_ready": False,
                                          "coverage_complete": full,
                                          "elapsed_seconds": time.monotonic() - started,
                                          "size_bytes": shard.stat().st_size,
                                          "sha256": digest_file(shard)})
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                report(completed_videos=progress["completed_videos"] + 1,
                       scanned_frames=progress["scanned_frames"] + len(frames),
                       documents=progress["documents"] + saved["documents"], state="checkpoint_saved")
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            report(state="worker_complete", current_video=None, current_ordinal=None)
        except BaseException as error:
            report(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            stop.set()
            thread.join(timeout=5)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "output", "manifest", "assets"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--resume", action="store_true", required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        parser.error("Invalid source SHA-256")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = inventory(args.raw, manifest)
    print(json.dumps({"videos": len(plan["videos"]), "frames": 220270,
                      "fingerprint": plan["fingerprint"]}), flush=True)
    if args.inventory_only:
        return
    run_worker(args.raw, args.output, plan, args.worker_id, VietOCREngine(args.assets),
               args.source_sha256, full=args.full)


if __name__ == "__main__":
    main()
