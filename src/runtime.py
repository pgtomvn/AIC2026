"""Composition root for the Textual KIS, Visual Q&A, and TRAKE runtime.

Building objects is side-effect free: model loading and service connections are
deferred until a search/index call, and SQLite databases are initialized only
by explicit CLI/application startup.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT
from src.embedding.clip_encoder import CLIPEncoder
from src.embedding.siglip_encoder import SigLIP2Encoder
from src.embedding.text_encoder import VietnameseTextEncoder
from src.evaluation.tuning import load_tuning_override
from src.indexing.text_store import TextStore
from src.indexing.vector_store import BatchVectorStore, VectorStore
from src.ingest.ledger import JobLedger, LedgerError
from src.ingest.native_pipeline import ScratchManager
from src.organizer_verification import organizer_verification_passed
from src.retrieval.raw_refiner import LocalFrameRefiner, PersistentRefinedFrameSink
from src.retrieval.textual_kis import DenseChannel, LexicalChannel, TextualKISPipeline
from src.utils.io_helpers import (
    configured_batch_id,
    load_config,
    load_launch_identity,
    validate_batch_identifier,
)
from src.warmup import WarmupService


def validate_runtime_batch(
    batch_id: str,
    config: Mapping[str, Any],
    ledger: JobLedger | None = None,
) -> str:
    """Accept a configured reference batch or a registered native batch."""

    normalized = validate_batch_identifier(batch_id)
    readiness = config.get("readiness", {})
    configured = readiness.get("batches", {}) if isinstance(readiness, Mapping) else {}
    if isinstance(configured, Mapping) and normalized in configured:
        return normalized
    if ledger is not None:
        try:
            if ledger.is_batch_registered(normalized):
                return normalized
        except LedgerError as error:
            raise ValueError("registered batches could not be validated") from error
    raise ValueError(f"batch {normalized!r} is not configured or registered")


@dataclass
class RuntimeContext:
    """Application services shared by FastAPI, Streamlit, and CLI entry points."""

    config: dict[str, Any] = field(repr=False)
    pipeline: TextualKISPipeline
    ledger: JobLedger
    artifact_repository: Any = None
    qa_pipeline: Any = None
    qa_session_store: Any = None
    qa_image_resolver: Any = None
    readiness_service: Any = None
    selected_batch_id: str = ""
    launch_identity: dict[str, Any] = field(default_factory=dict)
    warmup: Any = None
    trake_pipeline: Any = None

    def validate_batch_filters(self, batch_filters: Iterable[str] | None = None) -> tuple[str, ...]:
        """Validate explicit filters, defaulting empty input to the selected batch."""

        if isinstance(batch_filters, str):
            raise TypeError("batch_filters must be an iterable of batch identifiers")
        values = tuple(batch_filters or (self.selected_batch_id,))
        if not values:
            values = (self.selected_batch_id,)
        validated: list[str] = []
        seen: set[str] = set()
        for value in values:
            batch_id = validate_runtime_batch(value, self.config, self.ledger)
            if batch_id not in seen:
                seen.add(batch_id)
                validated.append(batch_id)
        return tuple(validated)


_QA_GPU_LEASE = threading.RLock()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_pipeline(config: dict[str, Any] | None = None) -> TextualKISPipeline:
    """Compose the configured retrieval channels without loading models."""

    settings = config or load_config()
    models = settings["models"]
    services = settings["services"]
    retrieval = settings["retrieval"]
    visual = models["visual"]
    qdrant = services["qdrant"]
    meili = services["meilisearch"]
    qdrant_transport = {
        "prefer_grpc": bool(qdrant.get("prefer_grpc", False)),
        "grpc_port": (int(qdrant["grpc_port"]) if qdrant.get("grpc_port") is not None else None),
    }
    depths = {str(key): int(value) for key, value in retrieval["candidate_depths"].items()}
    weights = {str(key): float(value) for key, value in retrieval["channel_weights"].items()}
    rrf_k = int(retrieval["rrf_k"])
    text_query = retrieval.get("text_query", {})
    text_max_chunks = int(text_query.get("max_chunks", 4))
    text_chunk_overlap = int(text_query.get("overlap_tokens", 8))
    tuning_record = retrieval.get("tuning_record")
    if tuning_record:
        tuning_path = _project_path(str(tuning_record))
        if tuning_path.is_file():
            override = load_tuning_override(tuning_path, expected_channels=tuple(weights))
            if override is not None:
                rrf_k = override.rrf_k
                depths = override.depths
                weights = override.weights

    encoder = SigLIP2Encoder(
        model_name=visual["model_name"],
        revision=visual.get("revision"),
        device=visual["device"],
        batch_size=int(visual["batch_size"]),
        max_batch_size=int(visual["max_batch_size"]),
        cache_dir=_project_path(visual["cache_dir"]),
        local_files_only=bool(visual.get("local_files_only", True)),
        text_max_chunks=text_max_chunks,
        text_chunk_overlap=text_chunk_overlap,
    )
    dense = [
        DenseChannel(
            "visual",
            encoder,
            BatchVectorStore(
                url=qdrant["url"],
                api_key=qdrant.get("api_key"),
                collection_template=qdrant["collections"]["native_frames"],
                vector_size=int(visual["embedding_dim"]),
                distance=qdrant["distance"],
                **qdrant_transport,
            ),
            float(weights["visual"]),
            int(depths["visual"]),
        )
    ]
    semantic_text = models.get("text", {})
    semantic_collection = qdrant.get("collections", {}).get("text_segments")
    if semantic_text and semantic_collection and "semantic_text" in weights:
        semantic_encoder = VietnameseTextEncoder(
            semantic_text["model_name"],
            revision=semantic_text.get("revision"),
            device=semantic_text.get("device", "cpu"),
            batch_size=int(semantic_text.get("batch_size", 16)),
            cache_dir=_project_path(semantic_text["cache_dir"]),
            local_files_only=bool(semantic_text.get("local_files_only", True)),
            text_max_chunks=text_max_chunks,
            text_chunk_overlap=text_chunk_overlap,
        )
        dense.append(
            DenseChannel(
                "semantic_text",
                semantic_encoder,
                BatchVectorStore(
                    url=qdrant["url"],
                    api_key=qdrant.get("api_key"),
                    collection_template=semantic_collection,
                    vector_size=int(semantic_text["embedding_dim"]),
                    distance=qdrant["distance"],
                    **qdrant_transport,
                ),
                float(weights["semantic_text"]),
                int(depths["semantic_text"]),
            )
        )
    organizer = models.get("organizer_clip", {})
    if organizer.get("enabled", False):
        organizer_encoder = CLIPEncoder(
            organizer["model_name"],
            organizer["pretrained"],
            device=organizer.get("query_device", "cpu"),
            batch_size=16,
            cache_dir=_project_path(organizer.get("cache_dir", "models/open_clip")),
            local_files_only=bool(organizer.get("local_files_only", True)),
            text_max_chunks=text_max_chunks,
            text_chunk_overlap=text_chunk_overlap,
        )
        if organizer_verification_passed(
            organizer,
            expected_fingerprint=organizer_encoder.fingerprint,
            project_root=PROJECT_ROOT,
        ):
            dense.append(
                DenseChannel(
                    "organizer_clip",
                    organizer_encoder,
                    VectorStore(
                        url=qdrant["url"],
                        api_key=qdrant.get("api_key"),
                        collection_name=qdrant["collections"]["organizer_frames"],
                        vector_size=int(organizer["embedding_dim"]),
                        distance=qdrant["distance"],
                        **qdrant_transport,
                    ),
                    float(weights["organizer_clip"]),
                    int(depths["organizer_clip"]),
                )
            )
    accelerated = models.get("accelerated_visual", {})
    accelerated_profiles = tuple(str(value) for value in accelerated.get("enabled_profiles", ()))
    accelerated_enabled = bool(accelerated.get("enabled", False)) and (
        not accelerated_profiles or str(settings.get("profile")) in accelerated_profiles
    )
    if accelerated_enabled:
        accelerated_encoder = CLIPEncoder(
            accelerated["model_name"],
            accelerated["pretrained"],
            device=accelerated.get("device", "cuda"),
            batch_size=int(accelerated.get("batch_size", 8)),
            cache_dir=_project_path(accelerated.get("cache_dir", "models/open_clip")),
            local_files_only=bool(accelerated.get("local_files_only", True)),
            text_max_chunks=text_max_chunks,
            text_chunk_overlap=text_chunk_overlap,
        )
        dense.append(
            DenseChannel(
                "accelerated_visual",
                accelerated_encoder,
                BatchVectorStore(
                    url=qdrant["url"],
                    api_key=qdrant.get("api_key"),
                    collection_template=qdrant["collections"]["accelerated_frames"],
                    vector_size=int(accelerated["embedding_dim"]),
                    distance=qdrant["distance"],
                    **qdrant_transport,
                ),
                float(weights["accelerated_visual"]),
                int(depths["accelerated_visual"]),
            )
        )
    lexical = [
        LexicalChannel(
            name,
            TextStore(
                url=meili["url"],
                api_key=meili.get("api_key"),
                index_name=meili["indexes"][name],
            ),
            float(weights[name]),
            int(depths[name]),
        )
        for name in ("ocr", "asr", "metadata")
    ]
    diversity = retrieval.get("diversity", {})
    return TextualKISPipeline(
        dense_channels=dense,
        lexical_channels=lexical,
        rrf_k=rrf_k,
        top_k=int(retrieval["top_k"]),
        temporal_windows_ms=tuple(int(value) for value in retrieval["temporal_windows_ms"]),
        min_separation_ms=int(diversity.get("min_separation_ms", 3_000)),
        early_cutoff=int(diversity.get("early_cutoff", 20)),
        early_per_video=int(diversity.get("early_per_video", 2)),
        late_per_video=int(diversity.get("late_per_video", 5)),
    )


def build_runtime(config: dict[str, Any] | None = None) -> RuntimeContext:
    """Build a runtime context; callers explicitly initialize persisted state."""

    settings = config or load_config()
    from src.readiness import build_readiness_service

    ledger = JobLedger(_project_path(settings["data"]["catalog_db"]))
    selected_batch_id = configured_batch_id(settings)
    launch_identity = load_launch_identity(settings)
    readiness_service = build_readiness_service(
        settings,
        selected_batch_id=selected_batch_id,
        launch_identity=launch_identity,
    )
    repository = None
    try:
        from src.ingest.artifacts import ArtifactRepository

        repository = ArtifactRepository(_project_path(settings["data"]["artifact_db"]))
    except ImportError:
        # The optional repository module is independently testable; retaining a
        # null value keeps construction robust during partial installations.
        pass
    pipeline = build_pipeline(settings)
    if repository is not None:

        def resolve_segment(result: dict[str, Any]) -> dict[str, Any] | None:
            payload = result.get("payload", result)
            batch_id = payload.get("batch_id")
            video_id = payload.get("video_id")
            if not isinstance(batch_id, str) or not isinstance(video_id, str):
                return None
            start = int(payload.get("start_ms", payload.get("timestamp_ms", 0)))
            end = int(payload.get("end_ms", start))
            frame = repository.nearest_frame(batch_id, video_id, (start + end) // 2)
            if frame is None:
                return None
            mapped_payload = dict(payload)
            mapped_payload.update(frame.to_dict())
            mapped = dict(result)
            mapped["payload"] = mapped_payload
            return mapped

        pipeline.frame_resolver = resolve_segment

    dense_by_name = {channel.name: channel for channel in pipeline.dense_channels}

    from src.retrieval.trake_alignment import TRAKEPipeline
    from src.retrieval.trake_router import TRAKEPipelineRouter
    from src.retrieval.trake_snapshot import (
        TRAKE_SNAPSHOT_SCHEMA_VERSION,
        TRAKETrajectorySnapshotStore,
    )
    from src.retrieval.trake_trajectory_cache import (
        TRAKE_TRAJECTORY_CACHE_VERSION,
        TRAKETrajectoryCache,
    )

    trake_settings = settings.get("retrieval", {}).get("trake", {})
    trake_cores: dict[str, TRAKEPipeline] = {}
    for channel_name in ("visual", "organizer_clip"):
        channel = dense_by_name.get(channel_name)
        if channel is None:
            continue

        def trajectory_fingerprint(
            batch_id: str,
            *,
            selected_channel: str = channel_name,
            selected_store: Any = channel.store,
            selected_encoder_fingerprint: str = str(getattr(channel.encoder, "fingerprint", "")),
        ) -> str:
            try:
                generation: int | None = ledger.retrieval_generation(batch_id)
            except LedgerError:
                generation = None
            bound_store = selected_store
            bind = getattr(selected_store, "for_batch", None)
            if callable(bind):
                bound_store = bind(batch_id)
            count_points = getattr(bound_store, "count", None)
            point_count = (
                int(count_points({"batch_id": batch_id})) if callable(count_points) else None
            )
            payload = {
                "channel": selected_channel,
                "generation": generation,
                "collection": str(getattr(bound_store, "collection_name", "")),
                "vector_size": int(getattr(bound_store, "vector_size", 0)),
                "distance": str(getattr(bound_store, "distance", "")),
                "point_count": point_count,
                "encoder_fingerprint": selected_encoder_fingerprint,
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        scroll_batch_size = int(trake_settings.get("vector_scroll_batch_size", 4_096))
        snapshot_store = None
        if bool(trake_settings.get("persist_trajectory_cache", True)):
            configured_snapshot_schema = int(
                trake_settings.get(
                    "trajectory_snapshot_schema_version",
                    TRAKE_SNAPSHOT_SCHEMA_VERSION,
                )
            )
            if configured_snapshot_schema != TRAKE_SNAPSHOT_SCHEMA_VERSION:
                raise ValueError("configured TRAKE snapshot schema is unsupported")
            snapshot_store = TRAKETrajectorySnapshotStore(
                _project_path(
                    str(
                        settings.get("data", {}).get(
                            "trake_trajectory_cache",
                            "data/cache/trake-trajectories",
                        )
                    )
                ),
                channel=channel_name,
                vector_size=int(channel.store.vector_size),
                cache_version=TRAKE_TRAJECTORY_CACHE_VERSION,
                lock_timeout_seconds=float(
                    trake_settings.get("trajectory_snapshot_lock_timeout_seconds", 600)
                ),
            )
        trajectory_cache = TRAKETrajectoryCache(
            channel.store,
            vector_size=int(channel.store.vector_size),
            # Full-batch warm-up is sequential and benefits from fewer gRPC
            # round trips; Stage-B fallback scrolling retains its own setting.
            batch_size=max(16_384, scroll_batch_size),
            fingerprint_provider=trajectory_fingerprint,
            snapshot_store=snapshot_store,
        )
        trake_cores[channel_name] = TRAKEPipeline(
            encoder=channel.encoder,
            vector_store=channel.store,
            top_k_videos=int(trake_settings.get("max_candidate_videos", 200)),
            channel_name=channel_name,
            candidate_depth=int(trake_settings.get("candidate_depth", 300)),
            seed_frames_per_prompt=int(trake_settings.get("seed_frames_per_prompt", 4)),
            grouped_search_workers=int(trake_settings.get("grouped_search_workers", 8)),
            vector_scroll_batch_size=int(trake_settings.get("vector_scroll_batch_size", 4_096)),
            trajectory_cache=trajectory_cache,
        )
    trake_pipeline = TRAKEPipelineRouter(
        trake_cores,
        settings,
        readiness_service=readiness_service,
        readiness_cache_seconds=float(trake_settings.get("readiness_cache_seconds", 300)),
    )

    def feedback(channel_name: str, keys: Any) -> Any:
        channel = dense_by_name.get(channel_name)
        if channel is None or not hasattr(channel.store, "vectors_for_frames"):
            raise ValueError(f"channel {channel_name!r} cannot provide feedback vectors")
        return channel.store.vectors_for_frames(keys)

    pipeline.feedback_provider = feedback
    refinement = settings["retrieval"].get("raw_refinement", {})
    if refinement.get("enabled", False) and pipeline.dense_channels and repository is not None:
        data = settings["data"]
        scratch = ScratchManager(
            _project_path(data["scratch"]),
            limit_bytes=int(float(data["scratch_limit_gib"]) * 1024**3),
            reserve_bytes=int(float(data["free_space_reserve_gib"]) * 1024**3),
        )
        frame_sink = PersistentRefinedFrameSink(
            repository,
            _project_path(data.get("refined_frames", "data/cache/refined-frames")),
            max_dimension=int(data.get("thumbnail_max_dim", 512)),
            webp_quality=int(refinement.get("webp_quality", 85)),
            max_image_bytes=int(refinement.get("persistence_max_image_bytes", 2 * 1024 * 1024)),
            max_entries=int(refinement.get("persistence_max_entries", 20_000)),
            max_total_bytes=int(refinement.get("persistence_max_bytes", 2 * 1024**3)),
            reserve_bytes=int(float(data["free_space_reserve_gib"]) * 1024**3),
        )
        pipeline.raw_refiner = LocalFrameRefiner(
            pipeline.dense_channels[0].encoder,
            scratch,
            radius_seconds=float(refinement["radius_seconds"]),
            sample_fps=float(refinement["fps"]),
            top_windows=int(refinement["top_windows"]),
            frame_sink=frame_sink,
        )
    qa_pipeline = qa_session_store = qa_image_resolver = None
    warmup_service: WarmupService | None = None
    if repository is not None:
        from src.contracts import QACompetitionSpec
        from src.retrieval.qa_media import CanonicalImageResolver
        from src.retrieval.qa_retrieval import VisualQARetriever
        from src.retrieval.qa_session import QASessionStore
        from src.retrieval.visual_qa import VisualQAPipeline
        from src.vlm.qa_reasoner import (
            GeminiInteractionsReasoner,
            ProviderChainReasoner,
            Qwen3VLReasoner,
        )

        qa = settings.get("retrieval", {}).get("visual_qa", {})
        qa_session_store = QASessionStore(
            _project_path(settings["data"]["qa_state_db"]),
            session_ttl_seconds=int(qa.get("session_ttl_seconds", 86_400)),
            reasoning_ttl_seconds=int(qa.get("reasoning_cache_ttl_seconds", 2_592_000)),
        )
        qa_image_resolver = CanonicalImageResolver(
            (PROJECT_ROOT,),
            max_input_bytes=int(qa.get("image_max_source_bytes", 16 * 1024 * 1024)),
            max_dimension=int(qa.get("image_max_dimension", 768)),
        )
        qa_retriever = VisualQARetriever(
            pipeline,
            repository,
            retrieval_depth=int(qa.get("candidate_pool", 100)),
            max_clusters=int(qa.get("max_clusters", 20)),
            radius_ms=int(qa.get("window_radius_ms", 4_000)),
            frames_per_cluster=int(qa.get("frames_per_window", 5)),
            pseudo_shot_separation_ms=int(qa.get("min_cluster_separation_ms", 3_000)),
        )

        gemini = settings.get("services", {}).get("gemini", {})
        providers: list[Any] = [
            GeminiInteractionsReasoner(
                api_key=gemini.get("api_key"),
                model_name=str(gemini.get("model", "gemini-3.6-flash")),
                timeout_seconds=float(gemini.get("timeout_seconds", 30)),
                max_attempts=int(gemini.get("max_retries", 2)) + 1,
                requests_per_minute=int(gemini.get("requests_per_minute", 13)),
                max_concurrency=int(gemini.get("max_concurrency", 1)),
                retry_base_seconds=float(gemini.get("retry_base_seconds", 5)),
                retry_max_seconds=float(gemini.get("retry_max_seconds", 60)),
                rate_limit_cooldown_seconds=float(gemini.get("rate_limit_cooldown_seconds", 60)),
                cooldown_max_seconds=float(gemini.get("cooldown_max_seconds", 300)),
                min_request_interval_seconds=float(gemini.get("min_request_interval_seconds", 4.5)),
                max_total_image_bytes=int(qa.get("image_max_request_bytes", 20 * 1024 * 1024)),
            )
        ]
        qwen = settings.get("models", {}).get("qwen_vl", {})
        enabled_profiles = tuple(str(value) for value in qwen.get("enabled_profiles", ()))
        if bool(qwen.get("enabled", False)) and str(settings.get("profile")) in enabled_profiles:

            def unload_dense_encoders() -> None:
                for channel in pipeline.dense_channels:
                    unload = getattr(channel.encoder, "unload", None)
                    if callable(unload):
                        unload()
                if warmup_service is not None:
                    warmup_service.mark_evicted()

            providers.append(
                Qwen3VLReasoner(
                    model_name=str(qwen["model_name"]),
                    revision=qwen.get("revision"),
                    cache_dir=str(_project_path(str(qwen["cache_dir"]))),
                    device=str(qwen.get("device", "cuda")),
                    dtype=str(qwen.get("dtype", "float16")),
                    max_new_tokens=int(qwen.get("max_new_tokens", 32)),
                    max_pixels=int(qwen.get("max_pixels", 589_824)),
                    local_files_only=bool(qwen.get("local_files_only", True)),
                    max_total_image_bytes=int(qa.get("image_max_request_bytes", 20 * 1024 * 1024)),
                    gpu_lease=lambda: _QA_GPU_LEASE,
                    before_load=unload_dense_encoders,
                )
            )

        specs = (
            QACompetitionSpec.historical_numeric(),
            QACompetitionSpec.conservative_general_text(),
        )

        def qa_retrieval_fingerprint(batch_filters: tuple[str, ...]) -> str:
            batches = tuple(sorted(set(batch_filters)))
            generations: dict[str, int | None] = {}
            for batch_id in batches:
                try:
                    generations[batch_id] = ledger.retrieval_generation(batch_id)
                except LedgerError:
                    # Readiness still fingerprints physical resource counts and
                    # states when a batch has not been registered locally yet.
                    generations[batch_id] = None
            readiness_fingerprint = (
                readiness_service.retrieval_fingerprint(batches)
                if batches
                else pipeline.fingerprint
            )
            payload = {
                "version": "qa-runtime-retrieval-v1",
                "pipeline": pipeline.fingerprint,
                "readiness": readiness_fingerprint,
                "generations": generations,
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        qa_pipeline = VisualQAPipeline(
            retriever=qa_retriever,
            reasoner=ProviderChainReasoner(providers),
            image_resolver=qa_image_resolver,
            session_store=qa_session_store,
            competition_spec=specs[0],
            competition_specs={spec.spec_id: spec for spec in specs},
            top_k_retrieve=int(qa.get("initial_clusters", 5)),
            top_k_final=100,
            max_inline_image_bytes=int(qa.get("image_max_request_bytes", 20 * 1024 * 1024)),
            reasoning_workers=min(2, max(1, int(gemini.get("max_concurrency", 1)))),
            retrieval_fingerprint=qa_retrieval_fingerprint,
        )
    warmup_config = settings.get("warmup", {})
    if bool(warmup_config.get("enabled", True)):
        warmup_service = WarmupService(
            pipeline=pipeline,
            readiness_service=readiness_service,
            batch_id=selected_batch_id,
            profile=str(settings.get("profile", "laptop")),
            artifact_repository=repository,
            warm_query=str(warmup_config.get("warm_query", "một người đang đi bộ trong thành phố")),
            include_raw_refiner=bool(warmup_config.get("include_raw_refiner", True)),
            trake_pipeline=trake_pipeline,
            startup_budget_seconds=float(warmup_config.get("startup_budget_seconds", 90)),
        )
    return RuntimeContext(
        settings,
        pipeline,
        ledger,
        repository,
        qa_pipeline,
        qa_session_store,
        qa_image_resolver,
        readiness_service,
        selected_batch_id,
        launch_identity,
        warmup_service,
        trake_pipeline,
    )
