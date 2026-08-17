"""Audit canonical source-frame identities and organizer ordinal media.

The organizer reference pack contains more map rows than canonical source-frame
identities because adjacent JPEG ordinals can point at the same submitted frame.
This command reports that distinction without extracting or modifying media.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.qa_media import CanonicalImageResolver
from src.utils.io_helpers import atomic_write_bytes, validate_batch_identifier


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument(
        "--artifact-db",
        type=Path,
        default=PROJECT_ROOT / "data" / "state" / "artifacts.sqlite3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "missing_canonical_frames.json",
    )
    parser.add_argument(
        "--aliases-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "canonical_ordinal_aliases.json",
    )
    return parser


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _target_output_path(batch_id: str, video_id: str, source_frame_idx: int) -> str:
    return (
        Path("data")
        / "cache"
        / "recovered-keyframes"
        / batch_id
        / video_id
        / f"{source_frame_idx}.webp"
    ).as_posix()


def _missing_uris(
    uris: Iterable[str], resolver: CanonicalImageResolver
) -> frozenset[str]:
    ordered = tuple(sorted(set(uris)))
    missing: set[str] = set()

    def visit(values: tuple[str, ...]) -> None:
        if not values or resolver.source_uris_resolvable(values):
            return
        if len(values) == 1:
            missing.add(values[0])
            return
        middle = len(values) // 2
        visit(values[:middle])
        visit(values[middle:])

    visit(ordered)
    return frozenset(missing)


def _ordinal_uri(canonical_uri: str, video_id: str, ordinal: int) -> str | None:
    if not canonical_uri.startswith("zip://") or "!/" not in canonical_uri:
        return None
    archive, _member = canonical_uri.rsplit("!/", 1)
    return f"{archive}!/keyframes/{video_id}/{ordinal:03d}.jpg"


def audit_canonical_frames(
    artifact_db: str | Path,
    batch_id: str,
    *,
    allowed_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a safe summary, true missing records, and duplicate ordinal aliases."""

    selected_batch = validate_batch_identifier(batch_id)
    database = Path(artifact_db).resolve()
    if not database.is_file():
        raise FileNotFoundError("artifact database is unavailable")
    uri = database.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as connection:
        connection.row_factory = sqlite3.Row
        canonical_rows = tuple(
            connection.execute(
                """
                SELECT video_id, source_frame_idx, keyframe_ordinal,
                       timestamp_ms, image_uri
                FROM canonical_frames
                WHERE batch_id = ?
                ORDER BY video_id, source_frame_idx
                """,
                (selected_batch,),
            )
        )
        mapping_rows = tuple(
            connection.execute(
                """
                SELECT video_id, source_frame_idx, keyframe_ordinal
                FROM canonical_frame_mappings
                WHERE batch_id = ?
                ORDER BY video_id, source_frame_idx, keyframe_ordinal
                """,
                (selected_batch,),
            )
        )

    canonical: dict[tuple[str, int], sqlite3.Row] = {
        (str(row["video_id"]), int(row["source_frame_idx"])): row
        for row in canonical_rows
    }
    mappings: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in mapping_rows:
        mappings[(str(row["video_id"]), int(row["source_frame_idx"]))].append(
            int(row["keyframe_ordinal"])
        )

    resolver = CanonicalImageResolver((Path(allowed_root).resolve(),))
    missing_canonical_uris = _missing_uris(
        (str(row["image_uri"]) for row in canonical_rows), resolver
    )
    ordinal_uri_to_identity: dict[str, tuple[str, int, int]] = {}
    for identity, ordinals in mappings.items():
        row = canonical.get(identity)
        if row is None:
            continue
        for ordinal in ordinals:
            ordinal_uri = _ordinal_uri(str(row["image_uri"]), identity[0], ordinal)
            if ordinal_uri is not None:
                ordinal_uri_to_identity[ordinal_uri] = (*identity, ordinal)
    missing_ordinal_uris = _missing_uris(ordinal_uri_to_identity, resolver)

    missing: list[dict[str, Any]] = []
    for identity in sorted(set(mappings) - set(canonical)):
        video_id, source_frame_idx = identity
        missing.append(
            {
                "video_id": video_id,
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": None,
                "target_output_path": _target_output_path(
                    selected_batch, video_id, source_frame_idx
                ),
                "reason_code": "canonical_record_missing",
            }
        )
    for identity, row in sorted(canonical.items()):
        if str(row["image_uri"]) not in missing_canonical_uris:
            continue
        video_id, source_frame_idx = identity
        missing.append(
            {
                "video_id": video_id,
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": int(row["timestamp_ms"]),
                "target_output_path": _target_output_path(
                    selected_batch, video_id, source_frame_idx
                ),
                "reason_code": "canonical_media_unavailable",
            }
        )
    for missing_uri in sorted(missing_ordinal_uris):
        if missing_uri in missing_canonical_uris:
            continue
        video_id, source_frame_idx, ordinal = ordinal_uri_to_identity[missing_uri]
        row = canonical[(video_id, source_frame_idx)]
        missing.append(
            {
                "video_id": video_id,
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": int(row["timestamp_ms"]),
                "target_output_path": _target_output_path(
                    selected_batch, video_id, source_frame_idx
                ),
                "keyframe_ordinal": ordinal,
                "reason_code": "organizer_ordinal_media_unavailable",
            }
        )

    aliases = [
        {
            "video_id": identity[0],
            "source_frame_idx": identity[1],
            "canonical_keyframe_ordinal": int(canonical[identity]["keyframe_ordinal"]),
            "keyframe_ordinals": sorted(ordinals),
        }
        for identity, ordinals in sorted(mappings.items())
        if len(ordinals) > 1 and identity in canonical
    ]
    summary = {
        "schema_version": 1,
        "batch_id": selected_batch,
        "mapping_rows": len(mapping_rows),
        "canonical_identities": len(canonical_rows),
        "unique_mapping_identities": len(mappings),
        "duplicate_alias_rows": len(mapping_rows) - len(mappings),
        "duplicate_alias_groups": len(aliases),
        "missing_identity_count": len(set(mappings) - set(canonical)),
        "missing_canonical_media_count": len(missing_canonical_uris),
        "missing_ordinal_media_count": len(missing_ordinal_uris),
        "missing_record_count": len(missing),
    }
    return summary, missing, aliases


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, missing, aliases = audit_canonical_frames(
        args.artifact_db,
        args.batch_id,
        allowed_root=PROJECT_ROOT,
    )
    atomic_write_bytes(args.output, _json_bytes(missing))
    atomic_write_bytes(args.aliases_output, _json_bytes(aliases))
    sys.stdout.buffer.write(_json_bytes(summary))
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
