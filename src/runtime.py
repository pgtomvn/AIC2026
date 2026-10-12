"""Composition root for the Textual KIS, Visual Q&A, and TRAKE runtime.

Building objects is side-effect free: model loading and service connections are
deferred until a search/index call, and SQLite databases are initialized only
by explicit CLI/application startup.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT
from src.embedding.clip_encoder import CLIPEncoder
from src.embedding.siglip_encoder import SigLIP2Encoder
from src.embedding.text_encoder import VietnameseTextEncoder
from src.evaluation.tuning import (
    load_intent_policy_override,
    load_semantic_view_policy,
    load_tuning_override,
)
from src.indexing.text_store import TextStore
from src.indexing.vector_store import BatchVectorStore
from src.ingest.ledger import JobLedger, LedgerError
from src.ingest.native_pipeline import ScratchManager
from src.organizer_verification import organizer_batch_verification_passed
from src.query.translator import OfflineVietnameseTranslator, QueryTextAdapter, TranslationCache
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
    qa_reasoner: Any = None
    qa_session_store: Any = None
    qa_image_resolver: Any = None
    florence_reasoner: Any = None
    gpu_coordinator: Any = None
    readiness_service: Any = None
    selected_batch_id: str = ""
    launch_identity: dict[str, Any] = field(default_factory=dict)
    warmup: Any = None
    trake_pipeline: Any = None
    interaction_logger: Any = None

    @property
    def default_search_batches(self) -> tuple[str, ...]:
        from src.retrieval.batch_scope import search_batches

        return search_batches(self.config, self.selected_batch_id)

    def validate_batch_filters(self, batch_filters: Iterable[str] | None = None) -> tuple[str, ...]:
        """Validate explicit filters, defaulting empty input to the selected batch."""

        if isinstance(batch_filters, str):
            raise TypeError("batch_filters must be an iterable of batch identifiers")
        values = tuple(batch_filters or self.default_search_batches)
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


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_query_adapter(config: Mapping[str, Any]) -> QueryTextAdapter:
    """Compose the optional offline translator without loading it or creating files."""

    model = config.get("models", {}).get("query_translation", {})
    if not isinstance(model, Mapping) or not bool(model.get("enabled", False)):
        return QueryTextAdapter()
    data = config.get("data", {})
    cache = TranslationCache(
        _project_path(str(data.get("translation_cache_db", "data/state/query_translation.sqlite3"))),
        max_entries=int(model.get("disk_cache_entries", 10_000)),
    )
    translator = OfflineVietnameseTranslator(
        model_name=str(model["model_name"]),
        revision=str(model["revision"]),
        model_path=_project_path(str(model["converted_path"])),
        tokenizer_path=_project_path(str(model.get("tokenizer_path", model["source_path"]))),
        source_model_path=_project_path(str(model["source_path"])),
        cache=cache,
        memory_entries=int(model.get("memory_cache_entries", 2_048)),
        compute_type=str(model.get("compute_type", "int8")),
        intra_threads=int(model.get("intra_threads", 4)),
        max_decoding_length=int(model.get("max_decoding_length", 128)),
        request_timeout_seconds=float(model.get("request_timeout_seconds", 3.0)),
        startup_timeout_seconds=float(model.get("startup_timeout_seconds", 30.0)),
        sentence_max_chars=int(model.get("sentence_max_chars", 400)),
        source_token_limit=int(model.get("source_token_limit", 448)),
        allow_transformers_fallback=bool(model.get("allow_transformers_fallback", True)),
        asset_manifest=_project_path(str(model["asset_manifest"]))
        if model.get("asset_manifest")
        else None,
        asset_root=PROJECT_ROOT,
    )
    return QueryTextAdapter(translator)


def build_pipeline(config: dict[str, Any] | None = None) -> TextualKISPipeline:
    """Compose the configured retrieval channels without loading models."""

    settings = config or load_config()
    models = settings["models"]
    services = settings["services"]
    retrieval = settings["retrieval"]
    from src.evaluation.rollout import validate_rollout_activation

    for feature in (
        "candidate_rescue",
        "video_local",
        "ocr_quality",
        "text_reranker",
    ):
        selected = retrieval.get(feature, {})
        validate_rollout_activation(
            feature,
            selected if isinstance(selected, Mapping) else {},
            project_root=PROJECT_ROOT,
        )
    selected_batch_id = configured_batch_id(settings)
    readiness_batches = settings.get("readiness", {}).get("batches", {})
    selected_readiness = (
        readiness_batches.get(selected_batch_id, {})
        if isinstance(readiness_batches, Mapping)
        else {}
    )
    expected_visual_count = int(
        selected_readiness.get("expected_canonical_frames", 0)
        if isinstance(selected_readiness, Mapping)
        else 0
    )
    query_adapter = build_query_adapter(settings)
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
    intent_policies = dict(retrieval.get("intent_policies", {}))
    if tuning_record:
        tuning_path = _project_path(str(tuning_record))
        if tuning_path.is_file():
            override = load_tuning_override(tuning_path, expected_channels=tuple(weights))
            if override is not None:
                rrf_k = override.rrf_k
                depths = override.depths
                weights = override.weights
    intent_tuning_record = retrieval.get("intent_tuning_record")
    if intent_tuning_record:
        intent_tuning_path = _project_path(str(intent_tuning_record))
        if intent_tuning_path.is_file():
            intent_policies = load_intent_policy_override(intent_tuning_path)
    semantic_views = dict(retrieval.get("semantic_views", {}))
    asr_routing = dict(retrieval.get("asr_routing", {}))
    semantic_record = semantic_views.get("tuning_record")
    if semantic_record:
        semantic_record_path = _project_path(str(semantic_record))
        if semantic_record_path.is_file():
            semantic_policy = load_semantic_view_policy(semantic_record_path)
            semantic_views = {
                **semantic_views,
                "fusion": semantic_policy.fusion,
                "weights": semantic_policy.weights,
                "consensus_alpha": semantic_policy.consensus_alpha,
            }

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
            expected_visual_count > 0,
            expected_visual_count or None,
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
        verified_organizer_batches = {
            str(batch_id)
            for batch_id, specification in readiness_batches.items()
            if isinstance(specification, Mapping)
            and specification.get("kind") == "reference"
            and specification.get("retrieval_enabled", True)
            and organizer_batch_verification_passed(
                organizer,
                batch_id=str(batch_id),
                expected_fingerprint=organizer_encoder.fingerprint,
                project_root=PROJECT_ROOT,
            )
        }
        if verified_organizer_batches:
            dense.append(
                DenseChannel(
                    "organizer_clip",
                    organizer_encoder,
                    BatchVectorStore(
                        url=qdrant["url"],
                        api_key=qdrant.get("api_key"),
                        collection_template=qdrant["collections"]["organizer_compatible_frames"],
                        collection_overrides={
                            "aic25-b1-reference": qdrant["collections"]["organizer_frames"]
                        },
                        readable_batches=verified_organizer_batches,
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
    b02 = readiness_batches.get("aic26-b2", {})
    if b02.get("retrieval_enabled", False):
        lexical.append(LexicalChannel(
            "objects", TextStore(url=meili["url"], api_key=meili.get("api_key"),
                                 index_name="aic26_b2_objects_v1"),
            float(b02.get("object_weight", 0.3)), int(b02.get("object_depth", 100)),
        ))
    diversity = retrieval.get("diversity", {})
    entity_routing = retrieval.get("entity_routing", {})
    from src.retrieval.ocr_quality import OCRQualityConfig, OCRQualityStore

    ocr_quality = OCRQualityConfig.from_mapping(retrieval.get("ocr_quality"))
    ocr_quality_store = OCRQualityStore(_project_path(ocr_quality.sidecar_path))
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
        query_adapter=query_adapter,
        entity_lexical_multiplier=float(entity_routing.get("lexical_multiplier", 3.0)),
        entity_bonus_alpha=float(entity_routing.get("bonus_alpha", 0.25)),
        intent_policies=intent_policies,
        semantic_view_fusion=str(semantic_views.get("fusion", "global_only")),
        semantic_view_weights=semantic_views.get("weights", {}),
        semantic_consensus_alpha=float(semantic_views.get("consensus_alpha", 0.0)),
        asr_enabled=bool(asr_routing.get("enabled", True)),
        asr_max_weight=float(asr_routing.get("max_weight", weights.get("asr", 0.7))),
        asr_temporal_overlap_bonus=float(
            asr_routing.get("temporal_overlap_bonus", 0.3)
        ),
        asr_temporal_padding_ms=int(asr_routing.get("temporal_padding_ms", 1_000)),
        candidate_rescue=retrieval.get("candidate_rescue"),
        video_local=retrieval.get("video_local"),
        ocr_quality=ocr_quality,
        ocr_quality_store=ocr_quality_store,
        interactive=retrieval.get("interactive"),
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
            if batch_id == "aic26-b2":
                from src.ingest.frame_identity import resolve_image_frame

                try:
                    frame = resolve_image_frame(repository, payload)
                except ValueError:
                    return None
                if frame is None:
                    return None
                return {**result, "payload": {**payload, **frame.to_dict()}}
            frame = None
            raw_frame_idx = payload.get("source_frame_idx")
            if not isinstance(raw_frame_idx, bool):
                try:
                    source_frame_idx = int(raw_frame_idx)
                except (TypeError, ValueError):
                    source_frame_idx = -1
                if source_frame_idx >= 0:
                    frame = repository.get_frame(batch_id, video_id, source_frame_idx)
            if frame is None:
                try:
                    start = int(payload.get("start_ms", payload.get("timestamp_ms", 0)))
                    end = int(payload.get("end_ms", start))
                except (TypeError, ValueError):
                    return None
                frame = repository.nearest_frame(batch_id, video_id, (start + end) // 2)
            if frame is None:
                return None
            mapped_payload = dict(payload)
            mapped_payload.update(frame.to_dict())
            mapped = dict(result)
            mapped["payload"] = mapped_payload
            return mapped

        pipeline.frame_resolver = resolve_segment
        metadata_resolution = settings.get("retrieval", {}).get("metadata_resolution", "scalar")
        if metadata_resolution not in {"scalar", "batched", "asr_only_batched"}:
            raise ValueError("retrieval.metadata_resolution must be scalar, batched or asr_only_batched")
        if metadata_resolution in {"batched", "asr_only_batched"}:
            from src.retrieval.frame_resolution import BatchedFrameResolver

            pipeline.frame_resolver = BatchedFrameResolver(
                repository,
                scalar_fallback=resolve_segment if metadata_resolution == "asr_only_batched" else None,
            )

    advanced_kis = settings.get("retrieval", {}).get("advanced_kis", {})
    advanced_color_resolver = None
    if (
        repository is not None
        and isinstance(advanced_kis, Mapping)
        and bool(advanced_kis.get("enabled", True))
    ):
        from src.retrieval.color_filter import AdvancedKISReranker, HSVColorVerifier
        from src.retrieval.qa_media import CanonicalImageResolver
        from src.retrieval.spatial_graph import SpatialOntologyConfig
        from src.retrieval.spatial_rerank import OrganizerObjectArchive

        ontology_value = advanced_kis.get("spatial_ontology", {})
        ontology = ontology_value if isinstance(ontology_value, Mapping) else {}
        spatial_ontology = SpatialOntologyConfig(
            directional_margin=float(ontology.get("directional_margin", 0.03)),
            directional_scale=float(ontology.get("directional_scale", 0.25)),
            near_threshold=float(ontology.get("near_threshold", 0.35)),
            far_min=float(ontology.get("far_min", 0.30)),
            far_max=float(ontology.get("far_max", 0.70)),
            inside_satisfied_coverage=float(
                ontology.get("inside_satisfied_coverage", 0.90)
            ),
            between_projection_min=float(ontology.get("between_projection_min", 0.10)),
            between_projection_max=float(ontology.get("between_projection_max", 0.90)),
            between_perpendicular_threshold=float(
                ontology.get("between_perpendicular_threshold", 0.20)
            ),
            depth_area_ratio=float(ontology.get("depth_area_ratio", 1.20)),
            depth_sigmoid_scale=float(ontology.get("depth_sigmoid_scale", 10.0)),
            depth_vertical_tolerance=float(
                ontology.get("depth_vertical_tolerance", 0.05)
            ),
            perspective_weight=float(ontology.get("perspective_weight", 0.50)),
            nms_iou_threshold=float(ontology.get("nms_iou_threshold", 0.50)),
            max_detections_per_node=int(ontology.get("max_detections_per_node", 6)),
            beam_width=int(ontology.get("beam_width", 512)),
        )
        object_archive = OrganizerObjectArchive(
            _project_path(str(advanced_kis.get("object_archive", "objects-aic25-b1.zip"))),
            allowed_root=PROJECT_ROOT,
            batch_id=str(advanced_kis.get("object_batch_id", "aic25-b1-reference")),
            confidence_threshold=float(
                advanced_kis.get("object_confidence_threshold", 0.50)
            ),
            max_member_bytes=int(
                advanced_kis.get("object_max_member_bytes", 1024 * 1024)
            ),
            max_entries=int(advanced_kis.get("object_cache_entries", 2_048)),
            max_cache_bytes=int(
                advanced_kis.get("object_cache_bytes", 16 * 1024 * 1024)
            ),
        )
        color_resolver = CanonicalImageResolver(
            (PROJECT_ROOT,),
            max_input_bytes=int(
                advanced_kis.get("color_max_source_bytes", 16 * 1024 * 1024)
            ),
            max_dimension=int(advanced_kis.get("color_image_dimension", 224)),
            normalized_cache_entries=int(
                advanced_kis.get("color_cache_entries", 512)
            ),
            normalized_cache_bytes=int(
                advanced_kis.get("color_cache_bytes", 16 * 1024 * 1024)
            ),
        )
        advanced_color_resolver = color_resolver
        pipeline.advanced_reranker = AdvancedKISReranker(
            repository,
            object_archive,
            HSVColorVerifier(
                color_resolver,
                max_entries=int(advanced_kis.get("color_cache_entries", 512)),
                max_cache_bytes=int(
                    advanced_kis.get("color_cache_bytes", 16 * 1024 * 1024)
                ),
            ),
            spatial_depth=int(advanced_kis.get("spatial_candidate_depth", 100)),
            color_depth=int(advanced_kis.get("color_candidate_depth", 12)),
            spatial_beta=float(advanced_kis.get("spatial_beta", 0.35)),
            color_absent_multiplier=float(
                advanced_kis.get("color_absent_multiplier", 0.65)
            ),
            color_workers=int(advanced_kis.get("color_workers", 2)),
            verification_budget_ms=float(advanced_kis.get("verification_budget_ms", 20)),
            relation_beta=float(advanced_kis.get("relation_beta", 0.40)),
            strict_color_threshold=float(
                advanced_kis.get("strict_color_roi_threshold", 0.12)
            ),
            strict_color_beta=float(advanced_kis.get("strict_color_beta", 0.25)),
            strict_mismatch_multiplier=float(
                advanced_kis.get("strict_color_mismatch_multiplier", 0.10)
            ),
            negation_confidence_threshold=float(
                advanced_kis.get("negation_confidence_threshold", 0.60)
            ),
            negation_multiplier=float(
                advanced_kis.get("negation_multiplier", 0.10)
            ),
            spatial_ontology=spatial_ontology,
        )


    dense_by_name = {channel.name: channel for channel in pipeline.dense_channels}
    cross_modal = advanced_kis.get("cross_modal", {}) if isinstance(advanced_kis, Mapping) else {}
    allowed_profiles = (
        tuple(str(value) for value in cross_modal.get("allowed_profiles", ()))
        if isinstance(cross_modal, Mapping)
        else ()
    )
    if (
        repository is not None
        and advanced_color_resolver is not None
        and isinstance(cross_modal, Mapping)
        and bool(cross_modal.get("enabled", False))
        and str(settings.get("profile", "laptop")) in allowed_profiles
    ):
        from src.retrieval.calibrator import CalibrationWeights
        from src.retrieval.cross_modal_rescore import CrossModalKISReranker

        seed_weights = cross_modal.get("seed_weights", {})
        seed_weights = seed_weights if isinstance(seed_weights, Mapping) else {}
        pipeline.cross_modal_rescorer = CrossModalKISReranker(
            repository,
            advanced_color_resolver,
            dense_by_name,
            candidate_depth=int(cross_modal.get("candidate_depth", 20)),
            max_object_crops=int(cross_modal.get("max_object_crops", 2)),
            calibration=str(cross_modal.get("calibration", "rank")),
            weights=CalibrationWeights(
                base=float(seed_weights.get("base", 0.60)),
                global_image=float(seed_weights.get("global", 0.25)),
                crop=float(seed_weights.get("crop", 0.15)),
            ),
        )

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
            # Keep scroll responses bounded. Large 16K-vector gRPC responses
            # exceed the local Qdrant deadline on the canonical B01 collection.
            batch_size=max(512, scroll_batch_size),
            fingerprint_provider=trajectory_fingerprint,
            snapshot_store=snapshot_store,
        )
        if channel_name == pipeline.video_local.source_channel:
            pipeline.video_local_trajectories = trajectory_cache
        if channel_name == "organizer_clip":
            pipeline.temporal_trajectories = trajectory_cache
        trake_cores[channel_name] = TRAKEPipeline(
            encoder=channel.encoder,
            vector_store=channel.store,
            top_k_videos=int(trake_settings.get("max_candidate_videos", 200)),
            channel_name=channel_name,
            candidate_depth=int(trake_settings.get("candidate_depth", 300)),
            seed_frames_per_prompt=int(trake_settings.get("seed_frames_per_prompt", 4)),
            grouped_search_workers=int(trake_settings.get("grouped_search_workers", 8)),
            vector_scroll_batch_size=int(trake_settings.get("vector_scroll_batch_size", 4_096)),
            global_weight=float(trake_settings.get("global_weight", 0.25)),
            max_stage_gap_ms=(
                None
                if trake_settings.get("max_stage_gap_ms") is None
                else int(trake_settings["max_stage_gap_ms"])
            ),
            temporal_gap_penalty_per_second=float(
                trake_settings.get("temporal_gap_penalty_per_second", 0.0)
            ),
            shot_boundary_penalty=float(
                trake_settings.get("shot_boundary_penalty", 0.0)
            ),
            trajectory_cache=trajectory_cache,
            query_adapter=pipeline.query_adapter,
            lexical_channels=pipeline.lexical_channels,
        )
    from src.retrieval.batch_scope import search_batches

    scope = search_batches(settings, selected_batch_id)
    batch_cores = {}
    if len(scope) > 1:
        # Both audited references share an encoder, but must retain independent
        # resident trajectories. Never swap/reload the full cache in a request.
        from copy import copy

        organizer = trake_cores.get("organizer_clip")
        if organizer is not None:
            batch_cores[scope[0]] = organizer
            alternate = copy(organizer)
            original_cache = organizer.trajectory_cache
            alternate.trajectory_cache = TRAKETrajectoryCache(
                original_cache.vector_store, vector_size=original_cache.vector_size,
                batch_size=original_cache.batch_size,
                fingerprint_provider=original_cache.fingerprint_provider,
                snapshot_store=original_cache.snapshot_store,
            )
            batch_cores[scope[1]] = alternate
    trake_pipeline = TRAKEPipelineRouter(
        trake_cores,
        settings,
        readiness_service=readiness_service,
        artifact_repository=repository,
        batch_pipelines=batch_cores,
        readiness_cache_seconds=float(trake_settings.get("readiness_cache_seconds", 300)),
    )

    def feedback(channel_name: str, keys: Any, *, batch_id: str | None = None,
                 coordinate: str = "source_frame_idx") -> Any:
        channel = dense_by_name.get(channel_name)
        if channel is None or not hasattr(channel.store, "vectors_for_frames"):
            raise ValueError(f"channel {channel_name!r} cannot provide feedback vectors")
        if coordinate == "keyframe_ordinal":
            if batch_id != "aic26-b2" or channel_name != "organizer_clip":
                raise ValueError("unsupported ordinal vector space")
            return channel.store.vectors_for_frames(keys, batch_id=batch_id, coordinate=coordinate)
        return channel.store.vectors_for_frames(keys, batch_id=batch_id)

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
    qa_pipeline = qa_session_store = qa_image_resolver = qa_reasoner = None
    florence_reasoner = gpu_coordinator = None
    warmup_service: WarmupService | None = None
    if repository is not None:
        asr_routing = dict(settings.get("retrieval", {}).get("asr_routing", {}))
        from src.contracts import QACompetitionSpec
        from src.retrieval.qa_media import CanonicalImageResolver
        from src.retrieval.qa_retrieval import VisualQARetriever
        from src.retrieval.qa_session import QASessionStore
        from src.retrieval.visual_qa import VisualQAPipeline
        from src.vlm.gemini import GeminiFlashReasoner
        from src.vlm.gpu_coordinator import GPUModelCoordinator
        from src.vlm.local_qwen import OllamaQwenReasoner
        from src.vlm.qa_router import QAReasonerRouter

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
            frames_per_cluster=int(qa.get("frames_per_window", 3)),
            pseudo_shot_separation_ms=int(qa.get("min_cluster_separation_ms", 3_000)),
        )

        ollama = settings.get("services", {}).get("ollama_vlm", {})
        gemini = settings.get("services", {}).get("gemini_vlm", {})

        def coordinated_handoff(batch_id: str | None) -> Mapping[str, Any] | None:
            if gpu_coordinator is None:
                return None
            return gpu_coordinator.before_vlm_inference(batch_id)

        local_reasoner = OllamaQwenReasoner(
            base_url=str(ollama.get("url", "http://127.0.0.1:11434")),
            model_name=str(ollama.get("model", "qwen2.5vl:3b-q4_K_M")),
            expected_digest_prefix=ollama.get("expected_digest_prefix"),
            connect_timeout_seconds=float(ollama.get("connect_timeout_seconds", 2)),
            timeout_seconds=float(ollama.get("timeout_seconds", 60)),
            keep_alive=ollama.get("keep_alive", "15m"),
            num_ctx=int(ollama.get("num_ctx", 4096)),
            execution_mode=str(ollama.get("execution_mode", "auto")),
            max_new_tokens=int(ollama.get("max_new_tokens", 64)),
            max_images=int(ollama.get("max_images", 1)),
            max_total_image_bytes=int(qa.get("image_max_request_bytes", 20 * 1024 * 1024)),
            before_inference=coordinated_handoff,
        )
        gemini_reasoner = GeminiFlashReasoner(
            api_key=gemini.get("api_key"),
            model_name=str(gemini.get("model", "gemini-3.6-flash")),
            timeout_seconds=float(gemini.get("timeout_seconds", 5)),
            max_total_image_bytes=int(qa.get("image_max_request_bytes", 20 * 1024 * 1024)),
        )
        qa_reasoner = QAReasonerRouter(gemini_reasoner, local_reasoner)
        gpu_coordinator = GPUModelCoordinator(
            dense_channels=pipeline.dense_channels,
            local_reasoner=local_reasoner,
            clear_threshold=float(qa.get("cuda_cache_clear_threshold", 0.80)),
        )
        florence = settings.get("models", {}).get("florence2", {})
        if bool(florence.get("enabled", False)):
            from src.vlm.isolated_florence import IsolatedFlorence2TaskReasoner

            florence_reasoner = IsolatedFlorence2TaskReasoner(
                _project_path(str(florence["model_path"])),
                model_name=str(florence.get("model_name", "microsoft/Florence-2-base")),
                revision=str(florence["revision"]),
                expected_weights_sha256=florence.get("weights_sha256"),
                max_new_tokens=int(florence.get("max_new_tokens", 256)),
                timeout_seconds=float(florence.get("timeout_seconds", 90)),
            )

        specs = (
            QACompetitionSpec.historical_numeric(),
            QACompetitionSpec.conservative_general_text(),
        )
        qa_fingerprint_cache: dict[
            tuple[tuple[str, ...], tuple[tuple[str, int | None], ...]],
            tuple[float, str],
        ] = {}

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
            cache_key = (batches, tuple(sorted(generations.items())))
            now = time.monotonic()
            cached = qa_fingerprint_cache.get(cache_key)
            if cached is not None and now - cached[0] <= 30.0:
                return cached[1]
            warmed_report = None
            if warmup_service is not None:
                report_getter = getattr(warmup_service, "readiness_report", None)
                if callable(report_getter):
                    warmed_report = report_getter()
            if (
                warmed_report is not None
                and len(batches) == 1
                and getattr(warmed_report, "batch_id", None) in batches
            ):
                channel_snapshot = [
                    {
                        "name": channel.name,
                        "state": channel.state.value,
                        "count": channel.count,
                        "resource": channel.resource_name,
                        "dimension": channel.expected_dimension,
                    }
                    for channel in warmed_report.channels
                    if channel.name
                    in {
                        "visual",
                        "organizer_clip",
                        "semantic_text",
                        "ocr",
                        "asr",
                        "metadata",
                    }
                ]
                readiness_fingerprint = hashlib.sha256(
                    json.dumps(
                        channel_snapshot,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            else:
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
            fingerprint = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            qa_fingerprint_cache.clear()
            qa_fingerprint_cache[cache_key] = (now, fingerprint)
            return fingerprint

        qa_pipeline = VisualQAPipeline(
            retriever=qa_retriever,
            reasoner=qa_reasoner,
            florence_reasoner=florence_reasoner,
            image_resolver=qa_image_resolver,
            session_store=qa_session_store,
            competition_spec=specs[1],
            competition_specs={spec.spec_id: spec for spec in specs},
            top_k_retrieve=int(qa.get("initial_clusters", 1)),
            top_k_final=100,
            max_inline_image_bytes=int(qa.get("image_max_request_bytes", 20 * 1024 * 1024)),
            reasoning_workers=1,
            retrieval_fingerprint=qa_retrieval_fingerprint,
            asr_context_radius_ms=int(
                asr_routing.get("qa_context_radius_ms", 30_000)
            ),
            asr_context_max_characters=int(
                asr_routing.get("qa_context_max_characters", 4_000)
            ),
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
        if len(scope) > 1:
            from src.batch_warmup import BatchWarmup

            services = {}
            for batch in scope:
                services[batch] = warmup_service if batch == selected_batch_id else WarmupService(
                    pipeline=pipeline, readiness_service=readiness_service, batch_id=batch,
                    profile=str(settings.get("profile", "laptop")), artifact_repository=repository,
                    warm_query=warmup_service.warm_query,
                    include_raw_refiner=warmup_service.include_raw_refiner,
                    trake_pipeline=trake_pipeline,
                    startup_budget_seconds=warmup_service.startup_budget_seconds,
                )
            warmup_service = BatchWarmup(services, primary=selected_batch_id)
    if gpu_coordinator is not None:
        gpu_coordinator.warmup = warmup_service
    from src.evaluation.interaction_log import InteractionLogger
    from src.retrieval.temporal_router import load_release_record
    from src.runtime_contract import source_contract_fingerprint

    pipeline.temporal_source_fingerprint = source_contract_fingerprint()
    release_path = _project_path(str(settings.get("retrieval", {}).get("interactive", {}).get(
        "temporal_release_record", "outputs/evaluation/temporal-english-local-release.json")))
    pipeline.temporal_release_record = load_release_record(release_path, PROJECT_ROOT)

    interaction_settings = settings.get("retrieval", {}).get("interaction_log", {})
    interaction_logger = InteractionLogger(
        _project_path(
            str(interaction_settings.get("path", "outputs/evaluation/kis-interactions.jsonl"))
        ),
        mode=str(interaction_settings.get("mode", "off")),
        max_results=int(interaction_settings.get("max_results", 100)),
    )
    return RuntimeContext(
        config=settings,
        pipeline=pipeline,
        ledger=ledger,
        artifact_repository=repository,
        qa_pipeline=qa_pipeline,
        qa_reasoner=qa_reasoner,
        qa_session_store=qa_session_store,
        qa_image_resolver=qa_image_resolver,
        florence_reasoner=florence_reasoner,
        gpu_coordinator=gpu_coordinator,
        readiness_service=readiness_service,
        selected_batch_id=selected_batch_id,
        launch_identity=launch_identity,
        warmup=warmup_service,
        trake_pipeline=trake_pipeline,
        interaction_logger=interaction_logger,
    )
