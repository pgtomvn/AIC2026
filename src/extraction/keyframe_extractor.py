"""
Keyframe Extractor
==================
Extract representative keyframes from video files using scene-change
detection and uniform temporal sampling.

Usage:
    extractor = KeyframeExtractor(config)
    frames = extractor.extract("path/to/video.mp4")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class KeyframeExtractor:
    """Extract keyframes from video files.

    Supports two strategies:
    - **Uniform sampling**: Extract N frames per second.
    - **Scene-change detection**: Detect abrupt transitions using
      histogram difference thresholds.

    Args:
        fps_sample: Number of frames to sample per second of video.
        min_scene_change: Histogram-diff threshold for scene-change detection.
        resize_max_dim: Resize longest edge to this value. None = no resize.
        output_dir: Directory to save extracted keyframe images.
    """

    def __init__(
        self,
        fps_sample: int = 2,
        min_scene_change: float = 30.0,
        resize_max_dim: Optional[int] = 1024,
        output_dir: str | Path = "data/raw/keyframes",
    ) -> None:
        self.fps_sample = fps_sample
        self.min_scene_change = min_scene_change
        self.resize_max_dim = resize_max_dim
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: str | Path) -> list[dict]:
        """Extract keyframes from a single video.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of dicts with keys: frame_id, timestamp_sec, image_path.
        """
        # TODO: Implement keyframe extraction logic
        #   1. Open video with cv2.VideoCapture
        #   2. Compute frame sampling interval from fps_sample
        #   3. Optionally apply scene-change detection
        #   4. Resize frames if resize_max_dim is set
        #   5. Save frames to output_dir and return metadata
        raise NotImplementedError("KeyframeExtractor.extract")

    def extract_batch(self, video_dir: str | Path) -> list[dict]:
        """Extract keyframes from all videos in a directory.

        Args:
            video_dir: Directory containing video files.

        Returns:
            Aggregated list of frame metadata dicts.
        """
        # TODO: Iterate over video files and call self.extract()
        raise NotImplementedError("KeyframeExtractor.extract_batch")
