"""Side-effect-free runtime contract manifest shared by the local stack."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from src import PROJECT_ROOT

RUNTIME_CONTRACT_VERSION: Final[int] = 6
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
)
RUNTIME_CONTRACT_SURFACE: Final[tuple[str, ...]] = (
    "app/http_backend.py",
    "app/streamlit_app.py",
    "src/api.py",
    "src/contracts.py",
    "src/query/query_decomposer.py",
    "src/query/spatial_types.py",
    "src/retrieval/calibrator.py",
    "src/ingest/reference_provisioner.py",
    "src/readiness.py",
    "src/retrieval/color_filter.py",
    "src/retrieval/cross_modal_rescore.py",
    "src/retrieval/spatial_graph.py",
    "src/retrieval/textual_kis.py",
    "src/runtime_contract.py",
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
