"""Audit deterministic object documents before resuming an interrupted import."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5


def _digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class _SourceDigests:
    def __init__(self) -> None:
        self.expected: dict[str, str] = {}

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for document in documents:
            identity = document["id"]
            if identity in self.expected:
                raise ValueError(f"duplicate object source identity: {identity}")
            self.expected[identity] = _digest(document)


class AuditedObjectResumeStore:
    """Skip only documents audited field-for-field against the frozen source.

    Audit is read-only, covers the entire dedicated index (not a batch filter),
    and must finish before any create/settings/upsert operation. Callers must
    ensure no concurrent writer exists. Unknown or changed documents fail closed.
    """

    def __init__(self, store: Any, verified: dict[str, str], expected_count: int) -> None:
        self.store = store
        self.verified = verified
        self.expected_count = expected_count
        self.written = 0
        self.skipped = 0

    @classmethod
    def audit(cls, importer: Any, store: Any) -> AuditedObjectResumeStore:
        source = _SourceDigests()
        importer.import_objects(source, create_index=False)
        verified: dict[str, str] = {}
        for stored in store.iter_documents():
            document = dict(stored)
            identity = document.pop("external_id", None)
            if not isinstance(identity, str) or identity not in source.expected:
                raise ValueError("object index contains unknown source identity")
            expected_id = str(uuid5(NAMESPACE_URL, f"aic2026:meilisearch:{store.index_name}:{identity}"))
            if document.get("id") != expected_id or identity in verified:
                raise ValueError("object index contains invalid deterministic ID or duplicate")
            document["id"] = identity
            digest = _digest(document)
            if digest != source.expected[identity]:
                raise ValueError(f"stored object differs from source: {identity}")
            verified[identity] = digest
        return cls(store, verified, len(source.expected))

    def create_index(self) -> None:
        self.store.create_index()

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        pending = []
        for document in documents:
            digest = self.verified.get(document["id"])
            if digest is not None:
                if digest != _digest(document):
                    raise ValueError("source changed after object resume audit")
                self.skipped += 1
            else:
                pending.append(document)
        if pending:
            self.store.add_documents(pending)
            self.written += len(pending)
