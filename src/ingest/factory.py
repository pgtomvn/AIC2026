"""Configuration-driven construction of the native ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src import PROJECT_ROOT
from src.embedding.text_encoder import VietnameseTextEncoder
from src.extraction.asr_engine import ASREngine
from src.extraction.audio import AudioExtractor
from src.extraction.calibration import load_calibration_selection
from src.extraction.keyframe_extractor import KeyframeExtractor
from src.extraction.ocr_engine import OCREngine
from src.ingest.native_pipeline import NativeIngestor, ScratchManager
from src.runtime import build_runtime
from src.tooling.ffmpeg import verify_ffmpeg_manifest


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_native_ingestor(
    config: dict[str, Any],
    *,
    batch_id: str,
    competition: str = "aic2026",
    enrichment: bool = True,
) -> NativeIngestor:
    """Compose ingestion stages without connecting to services or loading models."""

    context = build_runtime(config)
    if context.artifact_repository is None:
        raise RuntimeError("canonical artifact repository is unavailable")
    data = config["data"]
    ingestion = config["ingestion"]
    models = config["models"]
    ffmpeg_config = config.get("tools", {}).get("ffmpeg", {})
    tool_manifest = ffmpeg_config.get("manifest")
    if not tool_manifest:
        raise RuntimeError("tools.ffmpeg.manifest must identify the pinned local tools")
    manifest_path = _path(str(tool_manifest))
    ffmpeg_installation = verify_ffmpeg_manifest(manifest_path)
    ffprobe_path = ffmpeg_installation.ffprobe
    channels = {channel.name: channel for channel in context.pipeline.lexical_channels}
    scene_threshold = float(ingestion["scene_threshold"])
    max_gap_seconds = float(ingestion["max_gap_seconds"])
    calibration_record = ingestion.get("calibration_record")
    if calibration_record:
        calibration_path = _path(str(calibration_record))
        if calibration_path.is_file():
            selection = load_calibration_selection(calibration_path)
            scene_threshold = selection.scene_threshold
            max_gap_seconds = selection.max_gap_seconds
    extractor = KeyframeExtractor(
        min_scene_change=scene_threshold,
        resize_max_dim=int(data["thumbnail_max_dim"]),
        output_dir=_path(data["raw_keyframes"]),
        max_gap_seconds=max_gap_seconds,
        min_separation_seconds=float(ingestion["min_separation_seconds"]),
        batch_id=batch_id,
    )
    scratch = ScratchManager(
        _path(data["scratch"]),
        limit_bytes=int(float(data["scratch_limit_gib"]) * 1024**3),
        reserve_bytes=int(float(data["free_space_reserve_gib"]) * 1024**3),
    )
    ocr_engine = asr_engine = audio_extractor = text_encoder = text_store = None
    if enrichment:
        ocr = models["ocr"]
        ocr_engine = OCREngine(
            list(ocr["languages"]),
            bool(ocr["use_gpu"]),
            model_name=ocr["recognition_model"],
            detection_model_name=ocr["detection_model"],
            cache_dir=_path(ocr["cache_dir"]),
            local_files_only=bool(ocr.get("local_files_only", True)),
        )
        whisper = models["whisper"]
        asr_engine = ASREngine(
            whisper.get("hub_model_id", whisper["model_size"]),
            whisper.get("language"),
            whisper["device"],
            compute_type=whisper["compute_type"],
            beam_size=int(whisper["beam_size"]),
            retry_beam_size=int(whisper["retry_beam_size"]),
            vad_filter=bool(whisper["vad_filter"]),
            model_dir=_path(whisper["cache_dir"]),
            model_path=_path(whisper["local_path"]),
            model_revision=whisper.get("revision"),
            local_files_only=bool(whisper.get("local_files_only", True)),
        )
        if bool(whisper.get("extract_audio_16khz", True)):
            audio_extractor = AudioExtractor(
                ffmpeg_installation.ffmpeg,
                tool_fingerprint=ffmpeg_installation.archive_sha256,
            )
        text = models["text"]
        text_encoder = VietnameseTextEncoder(
            text["model_name"],
            revision=text.get("revision"),
            device=text["device"],
            batch_size=int(text["batch_size"]),
            cache_dir=_path(text["cache_dir"]),
            local_files_only=bool(text.get("local_files_only", True)),
        )
        semantic_channel = next(
            channel for channel in context.pipeline.dense_channels if channel.name == "semantic_text"
        )
        text_store = semantic_channel.store.for_batch(batch_id)
    primary = context.pipeline.dense_channels[0]
    return NativeIngestor(
        batch_id=batch_id,
        competition=competition,
        ledger=context.ledger,
        artifacts=context.artifact_repository,
        extractor=extractor,
        visual_encoder=primary.encoder,
        visual_store=primary.store.for_batch(batch_id),
        scratch=scratch,
        state_root=_path("data/state/manifests"),
        ffprobe_path=ffprobe_path,
        ocr_engine=ocr_engine,
        asr_engine=asr_engine,
        audio_extractor=audio_extractor,
        ocr_store=channels.get("ocr").store if enrichment and "ocr" in channels else None,
        asr_store=channels.get("asr").store if enrichment and "asr" in channels else None,
        text_encoder=text_encoder,
        text_vector_store=text_store,
        additional_visual_channels=(
            (channel.name, channel.encoder, channel.store.for_batch(batch_id))
            for channel in context.pipeline.dense_channels[1:]
            if channel.name == "accelerated_visual"
        ),
    )
