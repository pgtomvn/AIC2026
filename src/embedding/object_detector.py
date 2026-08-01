"""
Object Detector
===============
Detect objects in keyframe images using YOLOv8 / YOLOWorld for
label-based filtering and bounding-box metadata.

Usage:
    detector = ObjectDetector()
    detections = detector.detect("frame001.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ObjectDetector:
    """YOLOv8/YOLOWorld object detection wrapper.

    Args:
        model_name: YOLO model checkpoint (e.g. "yolov8x.pt").
        confidence_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for NMS.
        device: "cuda" or "cpu".
    """

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the YOLO model."""
        # TODO: from ultralytics import YOLO
        #       self._model = YOLO(self.model_name)
        raise NotImplementedError("ObjectDetector._load_model")

    def detect(self, image_path: str | Path) -> list[dict]:
        """Run object detection on a single image.

        Args:
            image_path: Path to the keyframe image.

        Returns:
            List of dicts: {label, confidence, bbox: [x1, y1, x2, y2]}.
        """
        # TODO: Run inference, filter by thresholds, return structured results
        raise NotImplementedError("ObjectDetector.detect")

    def detect_batch(
        self, image_paths: list[str | Path]
    ) -> dict[str, list[dict]]:
        """Run detection on multiple images.

        Args:
            image_paths: List of image file paths.

        Returns:
            Dict mapping filename -> list of detection dicts.
        """
        # TODO: Batch inference with YOLO
        raise NotImplementedError("ObjectDetector.detect_batch")
