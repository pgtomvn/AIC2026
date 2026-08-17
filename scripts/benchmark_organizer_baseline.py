"""Audit organizer archives and compare exact NumPy search with Qdrant.

This is a parity and latency harness, not an accuracy benchmark. The official Colab contains no
queries or ground truth, so this script never emits precision/recall quality claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.clip_encoder import CLIPEncoder
from src.indexing.vector_store import VectorStore
from src.organizer_baseline import (
    compare_rankings,
    exact_archive_search,
    inspect_reference_archives,
    latency_summary,
    load_query_file,
)
from src.utils.io_helpers import atomic_write_bytes, load_config

SYNTHETIC_QUERIES = PROJECT_ROOT / "tests" / "fixtures" / "organizer_colab" / "synthetic_smoke.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=PROJECT_ROOT / "map-keyframes-aic25-b1.zip")
    parser.add_argument(
        "--features", type=Path, default=PROJECT_ROOT / "clip-features-32-aic25-b1.zip"
    )
    parser.add_argument(
        "--objects", type=Path, default=PROJECT_ROOT / "objects-aic25-b1.zip"
    )
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query-file", type=Path)
    query_group.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--compare-qdrant", action="store_true")
    parser.add_argument("--include-ann", action="store_true")
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser


def _vector_store(config: dict[str, Any]) -> VectorStore:
    service = config["services"]["qdrant"]
    model = config["models"]["organizer_clip"]
    return VectorStore(
        url=service["url"],
        collection_name=service["collections"]["organizer_frames"],
        vector_size=int(model["embedding_dim"]),
        distance=service.get("distance", "Cosine"),
        api_key=service.get("api_key"),
        prefer_grpc=bool(service.get("prefer_grpc", False)),
        grpc_port=service.get("grpc_port"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if not args.inspect_only and not (args.query_file or args.synthetic_smoke):
        raise SystemExit("provide --query-file or --synthetic-smoke, or use --inspect-only")
    if args.include_ann and not args.compare_qdrant:
        raise SystemExit("--include-ann requires --compare-qdrant")

    started = time.perf_counter()
    stats = inspect_reference_archives(
        args.map,
        args.features,
        object_archive=args.objects,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_kind": "organizer_baseline_parity_not_accuracy",
        "artifact_stats": stats.to_dict(),
        "inspection_ms": round((time.perf_counter() - started) * 1_000, 3),
        "queries": [],
    }

    if not args.inspect_only:
        source = SYNTHETIC_QUERIES if args.synthetic_smoke else args.query_file
        queries = load_query_file(source)
        config = load_config()
        model = config["models"]["organizer_clip"]
        encoder = CLIPEncoder(
            model["model_name"],
            model["pretrained"],
            device=model.get("query_device", "cpu"),
            batch_size=1,
            cache_dir=PROJECT_ROOT / model.get("cache_dir", "models/open_clip"),
            local_files_only=bool(model.get("local_files_only", True)),
        )
        store = _vector_store(config) if args.compare_qdrant else None
        exact_latencies: list[float] = []
        qdrant_latencies: list[float] = []
        for query in queries:
            encode_started = time.perf_counter()
            vector = encoder.encode_text(query["text"])[0]
            encode_ms = (time.perf_counter() - encode_started) * 1_000

            exact_started = time.perf_counter()
            baseline = exact_archive_search(
                vector,
                args.map,
                args.features,
                top_k=args.top_k,
            )
            exact_ms = (time.perf_counter() - exact_started) * 1_000
            exact_latencies.append(exact_ms)
            item: dict[str, Any] = {
                "query_id": query["query_id"],
                "encoder_ms": round(encode_ms, 3),
                "exact_numpy_ms": round(exact_ms, 3),
                "exact_top_identity": list(baseline[0].identity) if baseline else None,
            }
            if store is not None:
                qdrant_started = time.perf_counter()
                exact_hits = store.search(
                    vector,
                    top_k=args.top_k,
                    filters={"batch_id": args.batch_id},
                    exact=True,
                )
                qdrant_ms = (time.perf_counter() - qdrant_started) * 1_000
                qdrant_latencies.append(qdrant_ms)
                item["exact_qdrant_ms"] = round(qdrant_ms, 3)
                item["exact_qdrant_parity"] = compare_rankings(baseline, exact_hits)
                if args.include_ann:
                    ann_started = time.perf_counter()
                    ann_hits = store.search(
                        vector,
                        top_k=args.top_k,
                        filters={"batch_id": args.batch_id},
                    )
                    item["ann_qdrant_ms"] = round(
                        (time.perf_counter() - ann_started) * 1_000,
                        3,
                    )
                    item["ann_overlap_not_precision"] = compare_rankings(baseline, ann_hits)
            report["queries"].append(item)
        report["exact_numpy_latency"] = latency_summary(exact_latencies)
        if qdrant_latencies:
            report["exact_qdrant_latency"] = latency_summary(qdrant_latencies)
        report["query_source"] = (
            "synthetic_smoke_not_official" if args.synthetic_smoke else "operator_supplied"
        )

    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.output is not None:
        atomic_write_bytes(args.output, encoded)
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
