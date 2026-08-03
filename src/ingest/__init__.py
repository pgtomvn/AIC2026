"""Archive cataloging and resumable ingestion state.

The package is deliberately side-effect free on import.  Callers must explicitly
initialize a :class:`JobLedger` or request archive relocation.
"""

from src.ingest.archive_catalog import (
    PROTECTED_ARCHIVE_NAMES,
    ArchiveInspection,
    ArchiveKind,
    ArchiveObservation,
    ArchiveValidationError,
    ProtectedArchiveError,
    classify_archive,
    discover_archives,
    inspect_archive,
    is_stable_archive,
    observe_archive,
    relocate_archive,
    sha256_file,
    update_checksum_manifest,
    validate_zip_crc,
)
from src.ingest.artifacts import (
    ArtifactConflictError,
    ArtifactRepository,
    ArtifactRepositoryError,
    BatchArtifactCounts,
    RepositoryCorruptionError,
    RepositoryNotInitialized,
    UpsertResult,
    VideoReplacement,
)
from src.ingest.ledger import (
    ArtifactSpec,
    BatchStatus,
    JobLedger,
    LedgerError,
    StageFingerprintMismatch,
    StageState,
    StageTransition,
)

__all__ = [
    "ArchiveInspection",
    "ArchiveKind",
    "ArchiveObservation",
    "ArchiveValidationError",
    "ArtifactConflictError",
    "ArtifactRepository",
    "ArtifactRepositoryError",
    "ArtifactSpec",
    "BatchStatus",
    "BatchArtifactCounts",
    "JobLedger",
    "LedgerError",
    "ProtectedArchiveError",
    "PROTECTED_ARCHIVE_NAMES",
    "RepositoryCorruptionError",
    "RepositoryNotInitialized",
    "StageFingerprintMismatch",
    "StageState",
    "StageTransition",
    "UpsertResult",
    "VideoReplacement",
    "classify_archive",
    "discover_archives",
    "inspect_archive",
    "is_stable_archive",
    "observe_archive",
    "relocate_archive",
    "sha256_file",
    "update_checksum_manifest",
    "validate_zip_crc",
]
