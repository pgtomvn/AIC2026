"""Explicitly acquire pinned model snapshots, then freeze their checksums."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assets import freeze_assets
from src.utils.io_helpers import load_config


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("visual", "organizer", "accelerated", "text", "asr", "ocr", "qwen"),
        action="append",
        help="Prepare selected assets; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "checksums/runtime-assets.sha256",
    )
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("run `uv sync --group competition` before preparing models") from error
    config = load_config()
    models = config["models"]
    if args.only:
        selected = set(args.only)
    else:
        selected = {"visual", "text", "asr", "ocr"}
        if models.get("organizer_clip", {}).get("enabled", False):
            selected.add("organizer")
        accelerated = models.get("accelerated_visual", {})
        accelerated_profiles = tuple(
            str(value) for value in accelerated.get("enabled_profiles", ())
        )
        if accelerated.get("enabled", False) and (
            not accelerated_profiles or config.get("profile") in accelerated_profiles
        ):
            selected.add("accelerated")
        qwen = models.get("qwen_vl", {})
        if qwen.get("enabled", False) and config.get("profile") in qwen.get(
            "enabled_profiles", ()
        ):
            selected.add("qwen")
    frozen_paths: set[Path] = set()
    if "visual" in selected:
        visual = models["visual"]
        cache = _path(visual["cache_dir"])
        snapshot_download(
            repo_id=visual["model_name"],
            revision=visual["revision"],
            cache_dir=cache,
            max_workers=1,
            allow_patterns=(
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ),
        )
        frozen_paths.add(cache)
    if "organizer" in selected:
        organizer = models["organizer_clip"]
        cache = _path(organizer.get("cache_dir", "models/open_clip"))
        from src.embedding.clip_encoder import CLIPEncoder

        encoder = CLIPEncoder(
            organizer["model_name"],
            organizer["pretrained"],
            device="cpu",
            batch_size=1,
            cache_dir=cache,
            local_files_only=False,
        )
        encoder.encode_text("offline asset verification")
        frozen_paths.add(cache)
    if "accelerated" in selected:
        accelerated = models["accelerated_visual"]
        cache = _path(accelerated.get("cache_dir", "models/open_clip"))
        from src.embedding.clip_encoder import CLIPEncoder

        encoder = CLIPEncoder(
            accelerated["model_name"],
            accelerated["pretrained"],
            device="cpu",
            batch_size=1,
            cache_dir=cache,
            local_files_only=False,
        )
        encoder.encode_text("offline asset verification")
        frozen_paths.add(cache)
    if "text" in selected:
        text = models["text"]
        cache = _path(text["cache_dir"])
        snapshot_download(
            repo_id=text["model_name"],
            revision=text["revision"],
            cache_dir=cache,
            max_workers=1,
            allow_patterns=(
                "added_tokens.json",
                "bpe.codes",
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.txt",
            ),
        )
        frozen_paths.add(cache)
    if "asr" in selected:
        whisper = models["whisper"]
        local_path = _path(whisper["local_path"])
        snapshot_download(
            repo_id=whisper["hub_model_id"],
            revision=whisper["revision"],
            local_dir=local_path,
            max_workers=1,
            allow_patterns=(
                "config.json",
                "model.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocabulary.json",
            ),
        )
        frozen_paths.add(local_path)
    if "ocr" in selected:
        ocr = models["ocr"]
        cache = _path(ocr["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache)
        try:
            # Establish PyTorch's DLLs before Paddle on Windows; PaddleX also
            # imports torch through ModelScope during initialization.
            import torch  # noqa: F401
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "PaddleOCR is missing from the competition dependency group"
            ) from error
        PaddleOCR(
            ocr_version="PP-OCRv5",
            text_detection_model_name=ocr["detection_model"],
            text_recognition_model_name=ocr["recognition_model"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
        frozen_paths.add(cache)
    if "qwen" in selected:
        qwen = models["qwen_vl"]
        cache = _path(qwen["cache_dir"])
        snapshot_download(
            repo_id=qwen["model_name"],
            revision=qwen["revision"],
            cache_dir=cache,
            max_workers=1,
            allow_patterns=(
                "added_tokens.json",
                "chat_template.json",
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "merges.txt",
                "model-*.safetensors",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "processor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "video_preprocessor_config.json",
                "vocab.json",
            ),
        )
        frozen_paths.add(cache)
    count = freeze_assets(frozen_paths, root=PROJECT_ROOT, destination=args.manifest)
    print(f"Prepared {', '.join(sorted(selected))}; froze {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
