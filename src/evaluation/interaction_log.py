"""Append-only, fingerprinted retrieval interaction telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.retrieval.candidate_rescue import validate_rollout_mode
from src.retrieval.fusion import canonical_key


class InteractionLogger:
    """Write bounded DRES-like interaction events without constructor I/O."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "off",
        max_results: int = 100,
    ) -> None:
        self.path = Path(path)
        self.mode = validate_rollout_mode(mode, name="interaction_log")
        if max_results < 1 or max_results > 100:
            raise ValueError("interaction_log max_results must be between 1 and 100")
        self.max_results = max_results
        self._lock = threading.Lock()

    @staticmethod
    def query_fingerprint(
        *, query_text: str, retrieval_fingerprint: str, batch_ids: Sequence[str]
    ) -> str:
        value = {
            "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
            "retrieval_fingerprint": retrieval_fingerprint,
            "batch_ids": sorted(set(batch_ids)),
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _append(self, event: Mapping[str, Any]) -> None:
        if self.mode == "off":
            return
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(line.encode("utf-8")) > 256 * 1024:
            raise ValueError("interaction event exceeds the bounded log contract")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def record_search(
        self,
        *,
        query_id: str,
        query_text: str,
        retrieval_fingerprint: str,
        batch_ids: Sequence[str],
        results: Sequence[Mapping[str, Any]],
        latency_ms: float,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        identities: list[dict[str, Any]] = []
        for rank, result in enumerate(results[: self.max_results], start=1):
            try:
                video_id, frame_idx = canonical_key(result)
            except ValueError:
                continue
            identities.append(
                {"rank": rank, "video_id": video_id, "source_frame_idx": frame_idx}
            )
        self._append(
            {
                "schema_version": 1,
                "event": "search_results_returned",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "query_id": query_id,
                "query": query_text,
                "query_fingerprint": self.query_fingerprint(
                    query_text=query_text,
                    retrieval_fingerprint=retrieval_fingerprint,
                    batch_ids=batch_ids,
                ),
                "retrieval_fingerprint": retrieval_fingerprint,
                "batch_ids": list(batch_ids),
                "latency_ms": round(float(latency_ms), 3),
                "results": identities,
                "diagnostics": dict(diagnostics or {}),
            }
        )

    def record_feedback(
        self,
        *,
        query_id: str,
        positive_ids: Sequence[str],
        negative_ids: Sequence[str],
        submitted_frame_id: str | None = None,
        time_to_correct_ms: float | None = None,
    ) -> None:
        self._append(
            {
                "schema_version": 1,
                "event": "relevance_feedback",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "query_id": query_id,
                "positive_ids": list(positive_ids),
                "negative_ids": list(negative_ids),
                "submitted_frame_id": submitted_frame_id,
                "time_to_correct_ms": (
                    None if time_to_correct_ms is None else round(float(time_to_correct_ms), 3)
                ),
            }
        )

    def record_turn(self, request: Any, *, results: Sequence[Mapping[str, Any]],
                    latency_ms: float, retrieval_fingerprint: str, kind: str) -> None:
        """Versioned event for real UI or declared benchmark turns, never qrels."""
        identities = [{"batch_id": row.get("batch_id"), "video_id": row.get("video_id"),
                       "source_frame_idx": row.get("source_frame_idx"), "rank": rank}
                      for rank, row in enumerate(results[:self.max_results], 1)]
        self._append({
            "schema_version": 2, "event": kind, "recorded_at": datetime.now(timezone.utc).isoformat(),
            "session_id": request.interaction_session_id, "turn": request.interaction_turn,
            "query_id": request.query_id, "event_source": request.event_source,
            "query": request.feedback_text or request.original_text,
            "reference_frame": asdict(request.reference_frame) if request.reference_frame else None,
            "positive_frames": [asdict(v) for v in request.positive_frames],
            "negative_frames": [asdict(v) for v in request.negative_frames],
            "legacy_positive_ids": list(request.positive_feedback_ids),
            "legacy_negative_ids": list(request.negative_feedback_ids),
            "results": identities, "latency_ms": round(latency_ms, 3),
            "retrieval_fingerprint": retrieval_fingerprint, "human_label": False,
        })

    def record_confirmation(self, *, session_id: str, turn: int, query_id: str,
                            frame: Mapping[str, Any], elapsed_ms: float,
                            event_source: str) -> None:
        """Explicit operator attestation only; pin/export never calls this method."""
        self._append({"schema_version": 2, "event": "answer_confirmed",
                      "recorded_at": datetime.now(timezone.utc).isoformat(),
                      "session_id": session_id, "turn": turn, "query_id": query_id,
                      "frame": dict(frame), "time_to_correct_ms": elapsed_ms,
                      "event_source": event_source, "human_label": event_source == "user"})


__all__ = ["InteractionLogger"]
