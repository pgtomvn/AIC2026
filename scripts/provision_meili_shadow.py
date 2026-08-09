"""Build multilingual Meilisearch v2 shadow indexes without promoting them."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.text_store import TextStore
from src.utils.io_helpers import load_config


def _batch_filter(batch_id: str | None) -> str | None:
    if batch_id is None:
        return None
    if not batch_id or any(value in batch_id for value in ('"', "\\", "\n", "\r")):
        raise ValueError("batch_id is unsafe")
    return f'batch_id = "{batch_id}"'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ocr", "asr", "metadata"), action="append")
    parser.add_argument("--batch-id")
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    service = config["services"]["meilisearch"]
    profiles = tuple(args.profile or ("ocr", "asr", "metadata"))
    filters = _batch_filter(args.batch_id)
    api_key = service.get("api_key") or os.environ.get(
        str(service.get("api_key_env", "MEILISEARCH_MASTER_KEY"))
    )
    report: dict[str, object] = {
        "mode": "dry_run" if args.dry_run else "shadow_copy",
        "batch_id": args.batch_id,
        "profiles": {},
        "promotion_performed": False,
    }
    for profile in profiles:
        source = TextStore(service["url"], api_key, service["indexes"][profile])
        shadow_name = service["shadow_indexes"][profile]
        if not str(shadow_name).endswith("_v2"):
            raise RuntimeError("configured shadow index must end in _v2")
        source_count = source.count_documents(filters)
        item: dict[str, object] = {
            "source_count": source_count,
            "shadow_index": shadow_name,
        }
        if not args.dry_run:
            shadow = TextStore(service["url"], api_key, shadow_name)
            shadow.create_index()
            shadow.configure_shadow_v2(profile)
            page: list[dict[str, object]] = []
            for document in source.iter_documents(
                filters=filters,
                batch_size=args.batch_size,
            ):
                page.append(document)
                if len(page) >= args.batch_size:
                    shadow.add_documents(page, batch_size=args.batch_size)
                    page = []
            if page:
                shadow.add_documents(page, batch_size=args.batch_size)
            shadow_count = shadow.count_documents(filters)
            item["shadow_count"] = shadow_count
            item["count_match"] = shadow_count == source_count
            if shadow_count != source_count:
                report["profiles"] = {**report["profiles"], profile: item}
                print(json.dumps(report, indent=2, sort_keys=True))
                return 2
        report["profiles"] = {**report["profiles"], profile: item}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
