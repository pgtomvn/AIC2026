"""Timestamped Faster-Whisper transcription for Textual KIS."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from src.retrieval.query_plan import accent_fold, normalize_nfc
from src.utils.offline import enforce_huggingface_offline

_MEDIA_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".wav", ".mp3", ".m4a"})


class ASREngine:
    """Lazy large-v3-turbo wrapper using a memory-safe laptop configuration."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        language: str | None = "vi",
        device: str = "cuda",
        *,
        compute_type: str | None = None,
        beam_size: int = 1,
        retry_beam_size: int = 5,
        vad_filter: bool = True,
        model_dir: str | Path = "models/whisper",
        model_path: str | Path | None = None,
        model_revision: str | None = None,
        local_files_only: bool = True,
        low_confidence_logprob: float = -1.0,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type or ("int8_float16" if device == "cuda" else "int8")
        self.beam_size = beam_size
        self.retry_beam_size = retry_beam_size
        self.vad_filter = vad_filter
        self.model_dir = Path(model_dir)
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_revision = model_revision
        self.local_files_only = local_files_only
        self.low_confidence_logprob = low_confidence_logprob
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.local_files_only:
            enforce_huggingface_offline()
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "Faster-Whisper is not installed in the competition ML profile"
            ) from error
        model_reference = self.model_size
        if self.model_path is not None and self.model_path.is_dir():
            model_reference = str(self.model_path)
        elif self.model_path is not None and self.local_files_only:
            raise RuntimeError(
                f"offline Whisper model is missing: {self.model_path}; run prepare_models.py"
            )
        self._model = WhisperModel(
            model_reference,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.model_dir),
            local_files_only=self.local_files_only,
        )

    @staticmethod
    def _segment_record(segment: Any) -> dict[str, Any]:
        text = str(segment.text).strip()
        words = []
        for word in getattr(segment, "words", None) or []:
            words.append(
                {
                    "start_ms": round(float(word.start) * 1000),
                    "end_ms": round(float(word.end) * 1000),
                    "text": str(word.word),
                    "probability": float(word.probability),
                }
            )
        return {
            "start_ms": round(float(segment.start) * 1000),
            "end_ms": round(float(segment.end) * 1000),
            "text_raw": text,
            "text_nfc": normalize_nfc(text),
            "text_accent_folded": accent_fold(text),
            "avg_logprob": float(getattr(segment, "avg_logprob", 0.0)),
            "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0)),
            "words": words,
        }

    def transcribe(self, media_path: str | Path) -> dict[str, Any]:
        """Transcribe a media file into independently searchable segments."""

        path = Path(media_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._load_model()
        segments_iter, info = self._model.transcribe(
            str(path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=True,
        )
        segments = [self._segment_record(segment) for segment in segments_iter]
        for segment in segments:
            segment["low_confidence"] = segment["avg_logprob"] < self.low_confidence_logprob
        if self.retry_beam_size > self.beam_size:
            for index, segment in enumerate(segments):
                if not segment["low_confidence"]:
                    continue
                retry_iter, _ = self._model.transcribe(
                    str(path),
                    language=self.language,
                    beam_size=self.retry_beam_size,
                    vad_filter=False,
                    word_timestamps=True,
                    clip_timestamps=[
                        segment["start_ms"] / 1000,
                        segment["end_ms"] / 1000,
                    ],
                )
                retried = [self._segment_record(value) for value in retry_iter]
                if not retried:
                    continue
                replacement = max(retried, key=lambda value: value["avg_logprob"])
                replacement["low_confidence"] = (
                    replacement["avg_logprob"] < self.low_confidence_logprob
                )
                if replacement["avg_logprob"] > segment["avg_logprob"]:
                    segments[index] = replacement
        full_text = normalize_nfc(" ".join(segment["text_raw"] for segment in segments))
        return {
            "text_raw": full_text,
            "text_nfc": full_text,
            "text_accent_folded": accent_fold(full_text),
            "segments": segments,
            "language": str(getattr(info, "language", self.language or "unknown")),
            "language_probability": float(getattr(info, "language_probability", 0.0)),
            "duration_sec": float(getattr(info, "duration", 0.0)),
            "model_version": self.model_size,
            "compute_type": self.compute_type,
        }

    def transcribe_batch(
        self, media_dir: str | Path, output_path: str | Path | None = None
    ) -> dict[str, dict[str, Any]]:
        """Transcribe all supported loose media below a directory."""

        root = Path(media_dir)
        results = {
            str(path.relative_to(root)): self.transcribe(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.casefold() in _MEDIA_EXTENSIONS
        }
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results

    def unload(self) -> None:
        """Release the CTranslate2 model so another GPU stage can run safely."""

        self._model = None
        gc.collect()
