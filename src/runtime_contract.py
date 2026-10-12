"""Side-effect-free runtime contract manifest shared by the local stack."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from src import PROJECT_ROOT

RUNTIME_CONTRACT_VERSION: Final[int] = 8
RUNTIME_CONTRACT_FEATURES: Final[tuple[str, ...]] = (
    "entity_constraints.kis",
    "entity_constraints.qa",
    "entity_constraints.trake",
    "qa.frame_injection",
    "qa.top_k_50",
    "kis.adaptive_rrf",
    "kis.spatial_color_rerank",
    "kis.temporal_adjacency",
    "kis.relative_relation_graph",
    "kis.soft_negation",
    "kis.strict_color_binding",
    "kis.experimental_cross_modal_calibration",
    "dataset.multimodal_coverage",
    "kis.full_spatial_ontology",
    "kis.interactive_feedback",
    "kis.session_conversation",
    "kis.manual_temporal_sequence",
    "kis.frame_evidence",
    "kis.edit_preview",
    "kis.gated_english_temporal",
    "kis.manual_event_inspection",
)
RUNTIME_CONTRACT_SURFACE: Final[tuple[str, ...]] = (
    "app/http_backend.py",
    "app/streamlit_app.py",
    "app/components/interactive_search.py",
    "app/components/event_inspection.py",
    "app/components/result_gallery.py",
    "src/api.py",
    "src/contracts.py",
    "src/query/query_decomposer.py",
    "src/query/spatial_types.py",
    "src/retrieval/calibrator.py",
    "src/ingest/reference_provisioner.py",
    "src/readiness.py",
    "src/retrieval/color_filter.py",
    "src/retrieval/raw_refiner.py",
    "src/retrieval/spatial_rerank.py",
    "src/retrieval/cross_modal_rescore.py",
    "src/retrieval/spatial_graph.py",
    "src/retrieval/textual_kis.py",
    "src/retrieval/interactive.py",
    "src/retrieval/temporal_sequence.py",
    "src/retrieval/temporal_router.py",
    "src/retrieval/refinement_preview.py",
    "src/retrieval/event_inspection.py",
    "src/vlm/local_qwen.py",
    "src/retrieval/frame_evidence.py",
    "src/ingest/frame_identity.py",
    "src/ingest/b02_presentation.py",
    "src/submission/b02.py",
    "src/retrieval/qa_retrieval.py",
    "app/components/b02_export.py",
    "app/components/ordinal_images.py",
    "src/runtime_contract.py",
    "src/retrieval/batch_scope.py",
    "src/retrieval/trake_router.py",
    "src/runtime.py",
    "src/batch_warmup.py",
    "src/submission/combined.py",
)


def source_contract_fingerprint(root: str | Path = PROJECT_ROOT) -> str:
    """Hash the runtime HTTP/UI contract surface in deterministic path order."""

    base = Path(root).resolve()
    digest = hashlib.sha256()
    for relative in RUNTIME_CONTRACT_SURFACE:
        path = base / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
        else:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "RUNTIME_CONTRACT_FEATURES",
    "RUNTIME_CONTRACT_SURFACE",
    "RUNTIME_CONTRACT_VERSION",
    "source_contract_fingerprint",
]
