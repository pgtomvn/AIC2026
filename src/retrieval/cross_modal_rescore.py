"""Opt-in, batch-compatible cross-modal verification for Advanced KIS."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from src.query.translator import QueryProcessing
from src.retrieval.calibrator import CalibrationWeights, blend_components
from src.retrieval.fusion import canonical_key
from src.retrieval.qa_media import ImageResolver
from src.retrieval.spatial_rerank import FrameRepository


class CrossModalKISReranker:
    """Re-encode Top-K full frames and assigned crops in one vector space."""

    def __init__(
        self,
        repository: FrameRepository,
        resolver: ImageResolver,
        channels: Mapping[str, Any],
        *,
        candidate_depth: int = 20,
        max_object_crops: int = 2,
        calibration: str = "rank",
        weights: CalibrationWeights = CalibrationWeights(),
    ) -> None:
        if not 1 <= candidate_depth <= 20:
            raise ValueError("candidate_depth must be between one and twenty")
        if not 0 <= max_object_crops <= 2:
            raise ValueError("max_object_crops must be between zero and two")
        if calibration not in {"rank", "minmax", "zscore"}:
            raise ValueError("unsupported cross-modal calibration")
        self.repository, self.resolver = repository, resolver
        self.channels = dict(channels)
        self.candidate_depth, self.max_object_crops = int(candidate_depth), int(max_object_crops)
        self.calibration, self.weights = calibration, weights
        self._lock, self._state = threading.RLock(), threading.local()

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": "kis-cross-modal-v1",
            "candidate_depth": self.candidate_depth,
            "max_object_crops": self.max_object_crops,
            "calibration": self.calibration,
            "weights": self.weights.to_dict(),
            "channels": {
                name: getattr(getattr(channel, "encoder", None), "fingerprint", None)
                for name, channel in sorted(self.channels.items())
            },
            "resolver": getattr(self.resolver, "fingerprint", None),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def last_diagnostics(self) -> dict[str, Any]:
        return dict(getattr(self._state, "diagnostics", {}))

    @staticmethod
    def _source(item: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = item.get("payload")
        return payload if isinstance(payload, Mapping) else item

    @classmethod
    def _channel_name(cls, candidates: Sequence[Mapping[str, Any]]) -> str | None:
        names = set()
        for item in candidates:
            batch_id = str(cls._source(item).get("batch_id", ""))
            if not batch_id:
                return None
            names.add("organizer_clip" if "reference" in batch_id.casefold() else "visual")
        return next(iter(names)) if len(names) == 1 else None

    def _frame(self, item: Mapping[str, Any]):
        source = self._source(item)
        batch_id = source.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            return None
        try:
            video_id, source_frame_idx = canonical_key(item)
            return self.repository.get_frame(batch_id, video_id, source_frame_idx)
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _crop(image: Image.Image, box: Sequence[float]) -> Image.Image | None:
        if len(box) != 4:
            return None
        ymin, xmin, ymax, xmax = (float(value) for value in box)
        if not (0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0):
            return None
        width, height = image.size
        left, top = (
            max(0, min(width - 1, int(round(xmin * width)))),
            max(0, min(height - 1, int(round(ymin * height)))),
        )
        right, bottom = (
            max(left + 1, min(width, int(round(xmax * width)))),
            max(top + 1, min(height, int(round(ymax * height)))),
        )
        return image.crop((left, top, right, bottom))

    def __call__(
        self,
        processing: QueryProcessing,
        candidates: list[dict[str, Any]],
        *,
        query_vectors: Mapping[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        del processing
        started = time.monotonic()
        head = [dict(item) for item in candidates[: self.candidate_depth]]
        tail = [dict(item) for item in candidates[self.candidate_depth :]]
        channel_name = self._channel_name(head)
        channel, query_vector = (
            self.channels.get(channel_name or ""),
            query_vectors.get(channel_name or ""),
        )
        if not head or channel is None or query_vector is None:
            self._state.diagnostics = {
                "state": "degraded",
                "reason_code": "compatible_vector_space_unavailable",
                "candidate_count": len(head),
            }
            return [*head, *tail]
        encode_images = getattr(getattr(channel, "encoder", None), "encode_images", None)
        if not callable(encode_images):
            self._state.diagnostics = {
                "state": "degraded",
                "reason_code": "compatible_encoder_unavailable",
                "candidate_count": len(head),
            }
            return [*head, *tail]
        with self._lock:
            images: list[Image.Image] = []
            owners: list[tuple[int, str]] = []
            crop_count = 0
            try:
                for index, item in enumerate(head):
                    frame = self._frame(item)
                    if frame is None:
                        continue
                    try:
                        resolved = self.resolver.resolve(frame)
                        with Image.open(io.BytesIO(resolved.data)) as source:
                            full = source.convert("RGB").copy()
                    except Exception:
                        continue
                    images.append(full)
                    owners.append((index, "global"))
                    assignments = item.get("_object_assignments")
                    if isinstance(assignments, Mapping):
                        for _node, box in sorted(assignments.items())[: self.max_object_crops]:
                            if isinstance(box, (list, tuple)):
                                crop = self._crop(full, box)
                                if crop is not None:
                                    images.append(crop)
                                    owners.append((index, "crop"))
                                    crop_count += 1
                if not images:
                    self._state.diagnostics = {
                        "state": "degraded",
                        "reason_code": "cross_modal_media_unavailable",
                        "candidate_count": len(head),
                    }
                    return [*head, *tail]
                vectors = np.asarray(encode_images(images), dtype=np.float32)
            finally:
                for image in images:
                    image.close()
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vectors.ndim != 2 or vectors.shape[0] != len(owners) or vectors.shape[1] != query.size:
            raise ValueError("cross-modal vector space mismatch")
        global_scores, crop_scores = np.full(len(head), np.nan), np.full(len(head), np.nan)
        for vector, (owner, kind) in zip(vectors, owners, strict=True):
            score = float(np.dot(vector, query))
            if kind == "global":
                global_scores[owner] = score
            elif not np.isfinite(crop_scores[owner]) or score > crop_scores[owner]:
                crop_scores[owner] = score
        base_scores = np.asarray([float(item.get("score", 0.0)) for item in head])
        blended = blend_components(
            {"base": base_scores, "global_image": global_scores, "crop": crop_scores},
            method=self.calibration,
            weights=self.weights,
        )
        for item, score in zip(head, blended, strict=True):
            item["score"], item["cross_modal_score"], item["cross_modal_rerank_state"] = (
                float(score),
                float(score),
                "verified",
            )
        head.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("video_id", "")),
                int(item.get("source_frame_idx", 0)),
            )
        )
        self._state.diagnostics = {
            "state": "ready",
            "channel": channel_name,
            "calibration": self.calibration,
            "candidate_count": len(head),
            "crop_count": crop_count,
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
        }
        return [*head, *tail]


__all__ = ["CrossModalKISReranker"]
