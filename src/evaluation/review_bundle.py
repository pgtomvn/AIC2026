"""Deterministic candidate bundles for human-reviewed Textual KIS qrels."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.qrels import IMPORTANT_INTENTS, QRELS_TAGS
from src.evaluation.tuning import CachedRankings
from src.retrieval.fusion import ChannelRanking, canonical_key, weighted_rrf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(result: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = result.get("payload")
    return payload if isinstance(payload, Mapping) else result


def build_qrels_review_bundle(
    queries: Sequence[Mapping[str, Any]],
    cached: CachedRankings,
    output_dir: str | Path,
    *,
    candidate_depth: int = 20,
    thumbnail_base_url: str = "http://127.0.0.1:8080",
    minimum_queries: int = 120,
) -> dict[str, Any]:
    """Write a review CSV/JSONL/contact sheet without fabricating relevance labels."""

    if candidate_depth < 1 or candidate_depth > 100:
        raise ValueError("candidate_depth must be between 1 and 100")
    if len(queries) < minimum_queries:
        raise ValueError(f"review bundle requires at least {minimum_queries} queries")
    query_ids: set[str] = set()
    groups: set[str] = set()
    intent_counts: Counter[str] = Counter()
    normalized_queries: list[dict[str, Any]] = []
    for raw in queries:
        query_id = raw.get("query_id")
        text = raw.get("text")
        intent = raw.get("intent")
        video_group = raw.get("video_group")
        tags = raw.get("tags", ())
        if (
            not isinstance(query_id, str)
            or not query_id
            or query_id in query_ids
            or not isinstance(text, str)
            or not text.strip()
            or intent not in IMPORTANT_INTENTS
            or not isinstance(video_group, str)
            or not video_group
            or not isinstance(tags, (list, tuple))
            or any(tag not in QRELS_TAGS for tag in tags)
        ):
            raise ValueError("review query contract is invalid")
        if query_id not in cached:
            raise ValueError(f"cached rankings are missing query {query_id}")
        query_ids.add(query_id)
        groups.add(video_group)
        intent_counts[str(intent)] += 1
        normalized_queries.append(
            {
                "query_id": query_id,
                "text": text.strip(),
                "intent": str(intent),
                "video_group": video_group,
                "tags": list(dict.fromkeys(str(tag) for tag in tags)),
                "seed_intervals": list(raw.get("seed_intervals", ())),
            }
        )
    if minimum_queries >= 120:
        if len(groups) < 60:
            raise ValueError("review bundle requires at least 60 video groups")
        missing = [intent for intent in IMPORTANT_INTENTS if intent_counts[intent] < 20]
        if missing:
            raise ValueError("review bundle requires 20 queries for every core intent")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "review_candidates.jsonl"
    csv_path = root / "review_candidates.csv"
    html_path = root / "contact_sheet.html"
    rows: list[dict[str, Any]] = []
    base = thumbnail_base_url.rstrip("/")
    for query in normalized_queries:
        channels = cached[query["query_id"]]
        rankings = [
            ChannelRanking(name, value.results[:candidate_depth], 1.0)
            for name, value in sorted(channels.items())
            if value.results
        ]
        fused = weighted_rrf(rankings, k=60)[:candidate_depth]
        candidates: dict[tuple[str, int], dict[str, Any]] = {}
        for ranking in rankings:
            for rank, result in enumerate(ranking.results, start=1):
                key = canonical_key(result)
                source = _source(result)
                candidate = candidates.setdefault(
                    key,
                    {
                        "result": result,
                        "sources": {},
                        "fused_rank": None,
                        "batch_id": str(source.get("batch_id") or "aic25-b1-reference"),
                    },
                )
                candidate["sources"][ranking.name] = rank
        for rank, result in enumerate(fused, start=1):
            key = canonical_key(result)
            source = _source(result)
            candidate = candidates.setdefault(
                key,
                {
                    "result": result,
                    "sources": {},
                    "fused_rank": None,
                    "batch_id": str(source.get("batch_id") or "aic25-b1-reference"),
                },
            )
            candidate["fused_rank"] = rank
        ordered = sorted(
            candidates.items(),
            key=lambda item: (
                item[1]["fused_rank"] is None,
                item[1]["fused_rank"] or 10_000,
                min(item[1]["sources"].values(), default=10_000),
                item[0],
            ),
        )
        for candidate_index, ((video_id, frame_idx), candidate) in enumerate(
            ordered,
            start=1,
        ):
            batch_id = candidate["batch_id"]
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query_text": query["text"],
                    "intent": query["intent"],
                    "video_group": query["video_group"],
                    "tags": query["tags"],
                    "candidate_index": candidate_index,
                    "batch_id": batch_id,
                    "video_id": video_id,
                    "source_frame_idx": frame_idx,
                    "channel_ranks": dict(sorted(candidate["sources"].items())),
                    "fused_rank": candidate["fused_rank"],
                    "thumbnail_url": (
                        f"{base}/v1/frames/{batch_id}/{video_id}/{frame_idx}/image"
                    ),
                    "decision": "",
                    "interval_start_frame_idx": "",
                    "interval_end_frame_idx": "",
                    "review_notes": "",
                    "seed_intervals_unreviewed": query["seed_intervals"],
                }
            )

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    columns = (
        "query_id", "query_text", "intent", "video_group", "tags",
        "candidate_index", "batch_id", "video_id", "source_frame_idx",
        "channel_ranks", "fused_rank", "thumbnail_url", "decision",
        "interval_start_frame_idx", "interval_end_frame_idx", "review_notes",
        "seed_intervals_unreviewed",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "tags": json.dumps(row["tags"], ensure_ascii=False),
                    "channel_ranks": json.dumps(row["channel_ranks"], sort_keys=True),
                    "seed_intervals_unreviewed": json.dumps(
                        row["seed_intervals_unreviewed"], ensure_ascii=False
                    ),
                }
            )
    sections: list[str] = []
    for query in normalized_queries:
        cards = []
        for row in (value for value in rows if value["query_id"] == query["query_id"]):
            cards.append(
                "<figure><img loading='lazy' width='224' src='{}' alt='{}'>"
                "<figcaption>#{}, {}:{}</figcaption></figure>".format(
                    html.escape(row["thumbnail_url"], quote=True),
                    html.escape(f"{row['video_id']}:{row['source_frame_idx']}", quote=True),
                    row["candidate_index"],
                    html.escape(row["video_id"]),
                    row["source_frame_idx"],
                )
            )
        sections.append(
            f"<section><h2>{html.escape(query['query_id'])}</h2>"
            f"<p>{html.escape(query['text'])}</p><div class='grid'>{''.join(cards)}</div></section>"
        )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>KIS qrels review</title>"
        "<style>body{font-family:sans-serif}.grid{display:grid;grid-template-columns:"
        "repeat(auto-fill,minmax(240px,1fr));gap:8px}figure{margin:0;padding:8px;"
        "border:1px solid #ccc}img{max-width:100%;height:auto}</style>"
        + "".join(sections),
        encoding="utf-8",
    )
    artifacts = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (jsonl_path, csv_path, html_path)
    }
    manifest = {
        "schema_version": 1,
        "query_count": len(normalized_queries),
        "candidate_count": len(rows),
        "video_group_count": len(groups),
        "intent_counts": dict(sorted(intent_counts.items())),
        "candidate_depth_per_channel": candidate_depth,
        "decision_values": ["relevant", "non_relevant", "unsure"],
        "labels_are_locked": False,
        "seed_intervals_are_ground_truth": False,
        "artifacts": artifacts,
    }
    manifest_path = root / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["build_qrels_review_bundle"]
