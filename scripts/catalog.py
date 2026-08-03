"""Inspect stable incoming archives and optionally verify and relocate them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import (
    PROTECTED_ARCHIVE_NAMES,
    JobLedger,
    ProtectedArchiveError,
    discover_archives,
    relocate_archive,
    update_checksum_manifest,
)

_RELOCATION_MIN_AGE_SECONDS = 120.0


def build_parser() -> argparse.ArgumentParser:
    """Build the archive-catalog command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Files or directories to inspect")
    parser.add_argument("--min-age", type=float, default=120.0, help="Stable age in seconds")
    parser.add_argument("--verify-crc", action="store_true", help="Read every ZIP member")
    parser.add_argument("--sha256", action="store_true", help="Compute streaming SHA-256")
    parser.add_argument("--no-recursive", action="store_true", help="Do not descend into roots")
    parser.add_argument("--ledger", type=Path, help="Explicit SQLite ledger output")
    parser.add_argument(
        "--relocate",
        action="store_true",
        help="After forced CRC/SHA verification, atomically move stable archives",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "archives",
        help="Canonical archive root used with --relocate",
    )
    parser.add_argument("--competition", help="Safe competition folder name for --relocate")
    parser.add_argument("--batch-id", help="Safe batch folder name for --relocate")
    parser.add_argument(
        "--checksum-manifest",
        type=Path,
        default=PROJECT_ROOT / "checksums" / "archives.sha256",
        help="Archive checksum manifest updated after relocation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Catalog requested paths and perform relocation only when explicitly requested."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.relocate:
        if not args.competition or not args.batch_id:
            parser.error("--relocate requires --competition and --batch-id")
        if args.min_age < _RELOCATION_MIN_AGE_SECONDS:
            parser.error(
                f"--relocate requires --min-age >= {_RELOCATION_MIN_AGE_SECONDS:g} seconds"
            )
        # Relocation is never allowed from a metadata-only inspection.  Every
        # candidate is fully read and hashed before the first move occurs.
        args.verify_crc = True
        args.sha256 = True
    inspections = discover_archives(
        args.roots,
        min_age_seconds=args.min_age,
        recursive=not args.no_recursive,
        verify_crc=args.verify_crc,
        compute_sha256=args.sha256,
    )
    if args.relocate:
        protected = {name.casefold() for name in PROTECTED_ARCHIVE_NAMES}
        blocked = [item.path.name for item in inspections if item.path.name.casefold() in protected]
        if blocked:
            raise ProtectedArchiveError(
                "Protected archives cannot be relocated: " + ", ".join(sorted(blocked))
            )
    ledger_path = args.ledger
    if args.relocate and ledger_path is None:
        ledger_path = PROJECT_ROOT / "data" / "state" / "jobs.sqlite3"
    ledger = None
    if ledger_path is not None:
        ledger = JobLedger(ledger_path)
        ledger.initialize()
        for inspection in inspections:
            ledger.record_archive(inspection)

    payload: list[dict[str, object]] = []
    for item in inspections:
        row: dict[str, object] = {
            "path": str(item.path),
            "kinds": [kind.value for kind in item.kinds],
            "size_bytes": item.size_bytes,
            "member_count": item.member_count,
            "file_count": item.file_count,
            "uncompressed_bytes": item.uncompressed_bytes,
            "crc_ok": item.crc_ok,
            "sha256": item.sha256,
        }
        if args.relocate:
            if item.crc_ok is not True or item.sha256 is None:
                raise RuntimeError("relocation requires successful CRC and SHA-256 verification")
            previous_path = item.path
            destination = relocate_archive(
                previous_path,
                args.archive_root,
                competition=args.competition,
                batch_id=args.batch_id,
                kinds=item.kinds,
                expected_sha256=item.sha256,
            )
            # Post-move updates are atomic individually.  If either update
            # fails, the verified archive remains safely at its destination;
            # it is never deleted or moved back automatically.
            update_checksum_manifest(
                args.checksum_manifest,
                destination,
                item.sha256,
                previous_path=previous_path,
                root=PROJECT_ROOT,
            )
            if ledger is None:  # Defensive: relocation always creates one above.
                raise RuntimeError("relocation ledger was not initialized")
            ledger.record_archive(item, relocated_to=destination)
            row["relocated_to"] = str(destination)
        payload.append(row)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
