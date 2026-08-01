"""
ASR Engine
==========
Speech-to-text transcription using Faster-Whisper for audio tracks
extracted from competition videos.

Usage:
    engine = ASREngine(model_size="large-v3-turbo")
    transcript = engine.transcribe("path/to/video.mp4")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ASREngine:
    """Faster-Whisper wrapper for automatic speech recognition.

    Args:
        model_size: Whisper model variant (e.g. "large-v3-turbo").
        language: Target language code (e.g. "vi"). None = auto-detect.
        device: "cuda" or "cpu".
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        language: Optional[str] = "vi",
        device: str = "cuda",
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self._model = None  # Lazy-loaded WhisperModel

    def _load_model(self) -> None:
        """Lazy-load the Faster-Whisper model."""
        # TODO: from faster_whisper import WhisperModel
        #       self._model = WhisperModel(self.model_size, device=self.device)
        raise NotImplementedError("ASREngine._load_model")

    def transcribe(self, media_path: str | Path) -> dict:
        """Transcribe audio from a video/audio file.

        Args:
            media_path: Path to the video or audio file.

        Returns:
            Dict with keys: text (full transcript), segments (list of
            {start, end, text}), language, duration_sec.
        """
        # TODO: Extract audio if video, run whisper, collect segments
        raise NotImplementedError("ASREngine.transcribe")

    def transcribe_batch(
        self, media_dir: str | Path, output_path: Optional[str | Path] = None
    ) -> dict[str, dict]:
        """Transcribe all media files in a directory.

        Args:
            media_dir: Directory containing video/audio files.
            output_path: Optional path to save results as JSON.

        Returns:
            Dict mapping filename -> transcript dict.
        """
        # TODO: Iterate files, call transcribe, optionally save
        raise NotImplementedError("ASREngine.transcribe_batch")
