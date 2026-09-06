"""Explicitly acquire pinned model snapshots, then freeze their checksums."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assets import freeze_assets, sha256_file, verify_asset_manifest
from src.utils.io_helpers import load_config


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=(
            "visual",
            "organizer",
            "accelerated",
            "text",
            "asr",
            "ocr",
            "reference-asr",
            "reference-ocr",
            "qwen",
            "florence",
            "translation",
        ),
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
        selected = {"visual", "text", "asr", "ocr", "reference-asr", "reference-ocr"}
        if models.get("query_translation", {}).get("enabled", False):
            selected.add("translation")
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
        profile_manifest = _path(whisper["asset_manifest"])
        checked, errors = verify_asset_manifest(profile_manifest, root=PROJECT_ROOT)
        if errors or checked < 1:
            detail = "; ".join(errors) if errors else "manifest is empty"
            raise RuntimeError(f"production Whisper asset verification failed: {detail}")
        frozen_paths.update((local_path, profile_manifest))
    reference = models.get("reference_provisioning", {})
    if "reference-asr" in selected:
        whisper = reference["whisper"]
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
                "vocabulary.txt",
            ),
        )
        manifest = _path(whisper["asset_manifest"])
        freeze_assets((local_path,), root=PROJECT_ROOT, destination=manifest)
        frozen_paths.update((local_path, manifest))
    if "reference-ocr" in selected:
        rapidocr = reference["rapidocr"]
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "competition dependencies are required to prepare RapidOCR"
            ) from error
        model_dir = _path(rapidocr["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        remote_files = {
            "v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx":
                "PP-OCRv6_det_small.onnx",
            "v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx":
                "PP-OCRv6_rec_small.onnx",
            "v3.9.2/onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx":
                "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
            "v3.9.2/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt":
                "ppocrv6_dict.txt",
        }
        expected_hashes = {
            "PP-OCRv6_det_small.onnx":
                "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
            "PP-OCRv6_rec_small.onnx":
                "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
            "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx":
                "54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7",
        }
        for remote_name, local_name in remote_files.items():
            downloaded = Path(
                hf_hub_download(
                    repo_id=rapidocr["source_repo"],
                    revision=rapidocr["source_revision"],
                    filename=remote_name,
                )
            )
            target = model_dir / local_name
            target.write_bytes(downloaded.read_bytes())
            expected = expected_hashes.get(local_name)
            if expected is not None and sha256_file(target) != expected:
                raise RuntimeError("RapidOCR downloaded asset checksum mismatch")
        manifest = _path(rapidocr["asset_manifest"])
        freeze_assets((model_dir,), root=PROJECT_ROOT, destination=manifest)
        frozen_paths.update((model_dir, manifest))
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
    if "florence" in selected:
        florence = models["florence2"]
        local_path = _path(florence["model_path"])
        snapshot_download(
            repo_id=florence["model_name"],
            revision=florence["revision"],
            local_dir=local_path,
            max_workers=1,
            allow_patterns=(
                "config.json",
                "configuration_florence2.py",
                "model.safetensors",
                "modeling_florence2.py",
                "preprocessor_config.json",
                "processing_florence2.py",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        )
        frozen_paths.add(local_path)
    if "translation" in selected:
        translation = models["query_translation"]
        source_path = _path(translation["source_path"])
        converted_path = _path(translation["converted_path"])
        snapshot_download(
            repo_id=translation["model_name"],
            revision=translation["revision"],
            local_dir=source_path,
            max_workers=1,
            allow_patterns=(
                "config.json",
                "generation_config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "source.spm",
                "target.spm",
                "tokenizer_config.json",
                "vocab.json",
            ),
        )
        try:
            from ctranslate2.converters import TransformersConverter
        except ImportError as error:
            raise RuntimeError(
                "CTranslate2 is missing from the competition dependency group"
            ) from error
        class _PinnedTransformersConverter(TransformersConverter):
            """Bridge CTranslate2 4.8's new dtype keyword to Transformers 4.51."""

            def load_model(self, model_class, model_name_or_path, **kwargs):
                dtype = kwargs.pop("dtype", None)
                if dtype is not None:
                    kwargs["torch_dtype"] = dtype
                return model_class.from_pretrained(model_name_or_path, **kwargs)

        converter = _PinnedTransformersConverter(
            str(source_path),
            copy_files=("source.spm", "target.spm", "tokenizer_config.json", "vocab.json"),
        )
        converter.convert(
            str(converted_path),
            quantization=str(translation.get("compute_type", "int8")),
            force=True,
        )
        translation_manifest = _path(
            translation.get(
                "asset_manifest",
                "checksums/query-translation.sha256",
            )
        )
        freeze_assets(
            (source_path, converted_path),
            root=PROJECT_ROOT,
            destination=translation_manifest,
        )
        frozen_paths.update((source_path, converted_path))
    # A targeted ``--only`` run must extend the immutable manifest rather
    # than dropping checksums for already prepared runtime assets.
    if args.manifest.is_file():
        for raw_line in args.manifest.read_text(encoding="utf-8-sig").splitlines():
            parts = raw_line.strip().split("  ", 1)
            if len(parts) != 2:
                continue
            relative = Path(parts[1])
            if relative.is_absolute() or ".." in relative.parts:
                continue
            existing = (PROJECT_ROOT / relative).resolve()
            try:
                existing.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                continue
            if existing.is_file():
                frozen_paths.add(existing)
    count = freeze_assets(frozen_paths, root=PROJECT_ROOT, destination=args.manifest)
    print(f"Prepared {', '.join(sorted(selected))}; froze {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
