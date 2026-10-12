"""Manual, bounded local image inspection. It never supplies ranking features."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError
from typing import Any

from src.contracts import FrameReference
from src.vlm.qa_reasoner import QAImage

PROMPT_VERSION = "canonical-image-event-v1"


class EventInspector:
    """One inference at a time; a timed-out worker keeps its lease until it exits."""

    def __init__(self, repository: Any, resolver: Any, provider: Any, *, budget_seconds: float = 60):
        self.repository, self.resolver, self.provider = repository, resolver, provider
        self.budget_seconds = min(60., max(.01, float(budget_seconds)))
        self._guard = threading.Lock()
        self._busy = False
        self._readers = 0
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def enter_retrieval(self) -> bool:
        with self._guard:
            if self._busy:
                return False
            self._readers += 1
            return True

    def leave_retrieval(self) -> None:
        with self._guard:
            self._readers -= 1

    def inspect(self, references: list[FrameReference], question: str) -> dict[str, Any]:
        if not 1 <= len(references) <= 3 or len(set(references)) != len(references):
            raise ValueError("one_to_three_unique_canonical_frames_required")
        if len({(r.batch_id, r.video_id) for r in references}) != 1:
            raise ValueError("one_video_required")
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
            raise ValueError("bounded_event_question_required")
        fallback = {"state": "unavailable", "verdict": "insufficient_evidence", "ranking_applied": False}
        if not all((self.repository, self.resolver, self.provider)):
            return {**fallback, "reason_code": "local_provider_unavailable"}
        with self._guard:
            if self._busy or self._readers:
                return {**fallback, "reason_code": "gpu_busy"}
            self._busy = True
        deadline = time.monotonic() + self.budget_seconds
        future: Future[dict[str, Any]] = Future()

        def work() -> None:
            try:
                value = self._inspect(references, question.strip(), deadline)
            except Exception:
                value = {**fallback, "reason_code": "local_inference_failed"}
            finally:
                with self._guard:
                    self._busy = False
            future.set_result(value)

        threading.Thread(target=work, name="manual-event-inspection", daemon=True).start()
        try:
            return future.result(timeout=max(.001, deadline-time.monotonic()))
        except TimeoutError:
            return {**fallback, "reason_code": "local_inference_deadline"}

    def _inspect(self, refs: list[FrameReference], question: str, deadline: float) -> dict[str, Any]:
        digest = self.provider.validated_model_digest
        if not digest:
            digest = self.provider.probe(deadline=deadline).get("digest")
        if not digest:
            return {"state": "unavailable", "reason_code": "model_digest_not_verified",
                    "verdict": "insufficient_evidence", "ranking_applied": False}
        frames = [self.repository.get_frame(r.batch_id, r.video_id, r.source_frame_idx) for r in refs]
        if any(f is None for f in frames):
            raise ValueError("canonical_frame_missing")
        if any((f.batch_id, f.video_id, f.source_frame_idx) !=
               (r.batch_id, r.video_id, r.source_frame_idx) for f, r in zip(frames, refs, strict=True)):
            raise ValueError("canonical_frame_mismatch")
        frames.sort(key=lambda f: (f.timestamp_ms, f.source_frame_idx))
        images, identities = [], []
        for index, frame in enumerate(frames, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError
            image = self.resolver.resolve(frame)
            images.append(QAImage(data=image.data, mime_type=image.mime_type, index=index,
                                  source_frame_idx=frame.source_frame_idx, timestamp_ms=frame.timestamp_ms,
                                  sha256=image.content_sha256))
            identities.append({"batch_id": frame.batch_id, "video_id": frame.video_id,
                               "source_frame_idx": frame.source_frame_idx, "timestamp_ms": frame.timestamp_ms,
                               "sha256": image.content_sha256})
        key = hashlib.sha256(json.dumps([identities, digest, self.provider.fingerprint, PROMPT_VERSION,
                                        question], sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return {**self._cache[key], "cache_hit": True}
        result = self.provider.inspect_event(question, images, deadline=deadline)
        if result.get("state") != "ready":
            return {**result, "ranking_applied": False}
        result = {**result, "frames": identities, "model_digest": digest, "prompt_version": PROMPT_VERSION,
                  "cache_key": key, "cache_hit": False, "ranking_applied": False,
                  "scope": "visible_image_evidence_only_not_motion_confirmation"}
        if time.monotonic() < deadline:
            self._cache[key] = result
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return result
