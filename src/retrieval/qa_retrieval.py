"""Candidate retrieval and temporal window construction for Visual Q&A."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from src.contracts import FrameRecord, QACandidateWindow
from src.retrieval.fusion import canonical_key


class FrameRepository(Protocol):
    """Minimal canonical repository surface needed by QA retrieval."""

    def get_frame(
        self, batch_id: str, video_id: str, source_frame_idx: int
    ) -> FrameRecord | None: ...

    def neighbors(
        self,
        batch_id: str,
        video_id: str,
        timestamp_ms: int,
        radius_ms: int = 4_000,
        limit: int | None = None,
    ) -> list[FrameRecord]: ...


class TextualRetriever(Protocol):
    """TextualKISPipeline-compatible retrieval surface."""

    def search(
        self,
        query: str | Mapping[str, Any] | Any,
        *,
        limit: int | None = None,
        refine_raw: bool = True,
    ) -> list[dict[str, Any]]: ...


QACandidateCluster = QACandidateWindow


def _value(item: Mapping[str, Any], name: str) -> Any:
    payload = item.get("payload")
    if isinstance(payload, Mapping) and name in payload:
        return payload[name]
    return item.get(name)


def _round_robin_by_video(
    values: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[tuple[int, Mapping[str, Any]]]:
    groups: dict[str, deque[tuple[int, Mapping[str, Any]]]] = {}
    order: list[str] = []
    for value in values:
        video_id, _ = canonical_key(value[1])
        if video_id not in groups:
            groups[video_id] = deque()
            order.append(video_id)
        groups[video_id].append(value)
    output: list[tuple[int, Mapping[str, Any]]] = []
    while any(groups.values()):
        for video_id in order:
            if groups[video_id]:
                output.append(groups[video_id].popleft())
    return output


class VisualQARetriever:
    """Reuse Textual KIS retrieval, then build diverse pseudo-shot windows."""

    def __init__(
        self,
        textual_pipeline: TextualRetriever,
        frame_repository: FrameRepository,
        *,
        retrieval_depth: int = 100,
        max_clusters: int = 20,
        radius_ms: int = 4_000,
        frames_per_cluster: int = 5,
        pseudo_shot_separation_ms: int = 3_000,
    ) -> None:
        if not 1 <= retrieval_depth <= 100:
            raise ValueError("retrieval_depth must be between 1 and 100")
        if not 1 <= max_clusters <= 20:
            raise ValueError("max_clusters must be between 1 and 20")
        if not 3 <= frames_per_cluster <= 5:
            raise ValueError("frames_per_cluster must be between 3 and 5")
        if radius_ms < 0 or pseudo_shot_separation_ms < 0:
            raise ValueError("temporal bounds cannot be negative")
        self.textual_pipeline = textual_pipeline
        self.frame_repository = frame_repository
        self.retrieval_depth = retrieval_depth
        self.max_clusters = max_clusters
        self.radius_ms = radius_ms
        self.frames_per_cluster = frames_per_cluster
        self.pseudo_shot_separation_ms = pseudo_shot_separation_ms

    @staticmethod
    def _batch_id(item: Mapping[str, Any], batch_filters: tuple[str, ...]) -> str:
        batch_id = _value(item, "batch_id")
        if isinstance(batch_id, str) and batch_id:
            return batch_id
        if len(batch_filters) == 1:
            return batch_filters[0]
        raise ValueError("QA retrieval hit lacks batch_id and cannot be resolved canonically")

    def retrieve(
        self,
        event_description: str,
        *,
        batch_filters: Sequence[str] = (),
        max_clusters: int | None = None,
    ) -> tuple[QACandidateCluster, ...]:
        """Retrieve only with event description and never trigger raw refinement."""

        if not isinstance(event_description, str) or not event_description.strip():
            raise ValueError("event_description must be a non-empty string")
        filters = tuple(str(value) for value in batch_filters)
        requested = self.max_clusters if max_clusters is None else int(max_clusters)
        if not 1 <= requested <= self.max_clusters:
            raise ValueError(f"max_clusters must be between 1 and {self.max_clusters}")
        hits = self.textual_pipeline.search(
            {"original_text": event_description, "batch_filters": filters},
            limit=self.retrieval_depth,
            refine_raw=False,
        )
        unique: list[tuple[int, Mapping[str, Any]]] = []
        seen_frames: set[tuple[str, int]] = set()
        retained_pseudo_times: dict[str, list[int]] = defaultdict(list)
        retained_real_shots: dict[str, set[str]] = defaultdict(set)
        for rank, hit in enumerate(hits, start=1):
            key = canonical_key(hit)
            if key in seen_frames:
                continue
            timestamp = _value(hit, "timestamp_ms")
            if timestamp is None:
                continue
            timestamp_ms = int(timestamp)
            shot_id = str(_value(hit, "shot_id") or "")
            is_pseudo_shot = not shot_id or shot_id.startswith("reference:")
            if is_pseudo_shot:
                if any(
                    abs(timestamp_ms - prior_time) < self.pseudo_shot_separation_ms
                    for prior_time in retained_pseudo_times[key[0]]
                ):
                    continue
                retained_pseudo_times[key[0]].append(timestamp_ms)
            else:
                if shot_id in retained_real_shots[key[0]]:
                    continue
                retained_real_shots[key[0]].add(shot_id)
            seen_frames.add(key)
            unique.append((rank, hit))
        ordered = _round_robin_by_video(unique)

        clusters: list[QACandidateCluster] = []
        for rank, hit in ordered:
            video_id, source_frame_idx = canonical_key(hit)
            batch_id = self._batch_id(hit, filters)
            seed = self.frame_repository.get_frame(batch_id, video_id, source_frame_idx)
            if seed is None:
                continue
            # Organizer rows can collapse to the same canonical source-frame
            # identity while retaining a timestamp one decoded frame away.
            # Once the hit is resolved, the canonical frame is authoritative
            # for window construction and for the QACandidateWindow contract.
            timestamp_ms = seed.timestamp_ms
            neighbors = self.frame_repository.neighbors(
                batch_id,
                video_id,
                timestamp_ms,
                radius_ms=self.radius_ms,
                limit=None,
            )
            by_frame = {frame.source_frame_idx: frame for frame in neighbors}
            by_frame[seed.source_frame_idx] = seed
            nearest = sorted(
                by_frame.values(),
                key=lambda frame: (
                    abs(frame.timestamp_ms - timestamp_ms),
                    frame.timestamp_ms,
                    frame.source_frame_idx,
                ),
            )
            seed_is_real_shot = bool(seed.shot_id) and not seed.shot_id.startswith("reference:")
            if seed_is_real_shot:
                same_shot = [frame for frame in nearest if frame.shot_id == seed.shot_id]
                if len(same_shot) >= 3:
                    selected_frames = same_shot[: self.frames_per_cluster]
                else:
                    selected_frames = list(same_shot)
                    selected_keys = {frame.source_frame_idx for frame in selected_frames}
                    selected_frames.extend(
                        frame
                        for frame in nearest
                        if frame.source_frame_idx not in selected_keys
                    )
                    selected_frames = selected_frames[:3]
            else:
                selected_frames = nearest[: self.frames_per_cluster]
            frames = tuple(
                sorted(
                    selected_frames,
                    key=lambda frame: (frame.timestamp_ms, frame.source_frame_idx),
                )
            )
            if len(frames) < 3:
                continue
            shot_id = str(seed.shot_id or _value(hit, "shot_id"))
            clusters.append(
                QACandidateCluster(
                    cluster_id=f"{batch_id}:{video_id}:{source_frame_idx}",
                    retrieval_rank=rank,
                    retrieval_score=float(_value(hit, "score") or 0.0),
                    batch_id=batch_id,
                    video_id=video_id,
                    source_frame_idx=source_frame_idx,
                    timestamp_ms=timestamp_ms,
                    shot_id=shot_id,
                    frames=frames,
                )
            )
            if len(clusters) >= requested:
                break
        return tuple(clusters)


__all__ = ["QACandidateCluster", "QACandidateWindow", "VisualQARetriever"]
