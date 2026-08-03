"""Build per-video SigLIP2 matrices from canonical frame records."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.siglip_encoder import SigLIP2Encoder
from src.ingest.artifacts import ArtifactRepository
from src.utils.io_helpers import load_config


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.save(output, array, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_output_is_compatible(
    matrix_path: Path,
    mapping_path: Path,
    *,
    frames: list,
    model_fingerprint: str,
    vector_dimension: int,
) -> bool:
    if not matrix_path.is_file() or not mapping_path.is_file():
        return False
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    expected = [
        (frame.batch_id, frame.video_id, frame.source_frame_idx) for frame in frames
    ]
    actual = [
        (item.get("batch_id"), item.get("video_id"), item.get("source_frame_idx"))
        for item in mapping.get("frames", ())
    ]
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    return (
        mapping.get("model_fingerprint") == model_fingerprint
        and int(mapping.get("vector_dimension", -1)) == vector_dimension
        and actual == expected
        and matrix.shape == (len(frames), vector_dimension)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild compatible per-video outputs instead of resuming them",
    )
    args = parser.parse_args(argv)
    config = load_config()
    data = config["data"]
    visual = config["models"]["visual"]
    repository = ArtifactRepository(_project_path(data["artifact_db"]))
    repository.initialize()
    frames_by_video: dict[str, list] = defaultdict(list)
    for frame in repository.iter_frames(
        batch_id=args.batch_id,
        video_id=args.video_id,
        index_ready_only=True,
    ):
        frames_by_video[frame.video_id].append(frame)
    if not frames_by_video:
        raise ValueError("no canonical index-ready frames matched the request")
    encoder = SigLIP2Encoder(
        visual["model_name"],
        revision=visual.get("revision"),
        device=visual["device"],
        batch_size=int(visual["batch_size"]),
        max_batch_size=int(visual["max_batch_size"]),
        cache_dir=_project_path(visual["cache_dir"]),
        local_files_only=bool(visual.get("local_files_only", True)),
    )
    root = (
        args.output_dir or _project_path(data["embeddings"]) / args.batch_id / encoder.fingerprint
    )
    from src.retrieval.qa_media import CanonicalImageResolver

    visual_qa = config.get("retrieval", {}).get("visual_qa", {})
    resolver = CanonicalImageResolver(
        (PROJECT_ROOT,),
        max_input_bytes=int(visual_qa.get("image_max_source_bytes", 16 * 1024 * 1024)),
        max_dimension=int(visual_qa.get("image_max_dimension", 768)),
    )
    for video_id, frames in sorted(frames_by_video.items()):
        matrix_path = root / f"{video_id}.npy"
        mapping_path = root / f"{video_id}.json"
        if not args.force and _existing_output_is_compatible(
            matrix_path,
            mapping_path,
            frames=frames,
            model_fingerprint=encoder.fingerprint,
            vector_dimension=int(visual["embedding_dim"]),
        ):
            print(f"{video_id}: {len(frames)} vectors (resumed)")
            continue
        inputs = [
            resolver.resolve(frame).data
            if str(frame.image_uri).startswith("zip://")
            else frame.image_uri
            for frame in frames
        ]
        vectors = encoder.encode_images(inputs)
        if vectors.shape != (len(frames), int(visual["embedding_dim"])):
            raise ValueError(f"unexpected embedding shape for {video_id}: {vectors.shape}")
        _atomic_npy(vectors.astype(np.float16), matrix_path)
        mapping = {
            "batch_id": args.batch_id,
            "video_id": video_id,
            "model_fingerprint": encoder.fingerprint,
            "vector_dimension": int(vectors.shape[1]),
            "frames": [frame.to_dict() for frame in frames],
        }
        _atomic_json(mapping, mapping_path)
        print(f"{video_id}: {len(frames)} vectors")
    close = getattr(resolver, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
