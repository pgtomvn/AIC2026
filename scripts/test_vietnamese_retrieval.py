"""Compare legacy raw-Vietnamese, English, and channel-aware routed retrieval.

The default suite is intentionally unlabeled. It reports rank agreement and
overlap only; it never labels those diagnostics as retrieval precision.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests

PAIRS: tuple[tuple[str, str], ...] = (
    ("Người đàn ông dẫn chó đi dạo", "a man walking a dog"),
    ("Cảnh quay toàn cảnh thành phố ban đêm", "a panoramic view of a city at night"),
    ("Xe cấp cứu đang chạy trên đường", "an ambulance driving on the road"),
    ("Một người đàn ông đang đọc bản tin truyền hình", "a man presenting television news"),
    ("Cảnh sát đứng cạnh một chiếc xe", "police officers standing beside a car"),
    ("Một người đi xe máy qua cây cầu", "a person riding a motorcycle across a bridge"),
    ("Người phụ nữ mặc áo đỏ cầm điện thoại", "a woman in a red shirt holding a phone"),
    ("Một đứa trẻ chơi với quả bóng màu xanh", "a child playing with a blue ball"),
    ("Người đàn ông đeo kính đang nói chuyện", "a man wearing glasses and speaking"),
    ("Người bước vào cửa hàng rồi thanh toán", "a person entering a store and then paying"),
)


def _request(
    session: requests.Session,
    base_url: str,
    *,
    batch_id: str,
    text: str,
    language: str,
    query_id: str,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = session.post(
        f"{base_url.rstrip('/')}/v1/kis/search",
        json={
            "query_id": query_id,
            "original_text": text,
            "query_language": language,
            "batch_filters": [batch_id],
            "limit": 100,
        },
        timeout=(3, 30),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("benchmark received a malformed API response")
    return value, latency_ms


def _identities(response: dict[str, Any], limit: int) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    for item in response.get("hits", ()):
        if not isinstance(item, dict):
            continue
        video_id = item.get("video_id")
        frame = item.get("source_frame_idx")
        if isinstance(video_id, str) and isinstance(frame, int) and not isinstance(frame, bool):
            values.append((video_id, frame))
        if len(values) >= limit:
            break
    return tuple(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args(argv)
    session = requests.Session()
    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {
        "legacy_raw_vi": [],
        "english": [],
        "channel_aware_vi": [],
        "channel_aware_vi_cache": [],
    }
    predictions: dict[str, dict[str, tuple[tuple[str, int], ...]]] = {
        "legacy_raw_vi": {},
        "english": {},
        "channel_aware_vi": {},
    }
    labeled_latencies: dict[str, dict[str, float]] = {
        "legacy_raw_vi": {},
        "english": {},
        "channel_aware_vi": {},
    }
    for pair_index, (vietnamese, english) in enumerate(PAIRS, 1):
        query_id = f"vi-{pair_index:02d}"
        # query_language=en intentionally reproduces the pre-remediation raw
        # OpenCLIP path for diagnostic comparison only.
        raw, raw_ms = _request(
            session,
            args.api_url,
            batch_id=args.batch_id,
            text=vietnamese,
            language="en",
            query_id=query_id,
        )
        reference, english_ms = _request(
            session,
            args.api_url,
            batch_id=args.batch_id,
            text=english,
            language="en",
            query_id=query_id,
        )
        routed, routed_ms = _request(
            session,
            args.api_url,
            batch_id=args.batch_id,
            text=vietnamese,
            language="vi",
            query_id=query_id,
        )
        cached, cached_ms = _request(
            session,
            args.api_url,
            batch_id=args.batch_id,
            text=vietnamese,
            language="vi",
            query_id=query_id,
        )
        latencies["legacy_raw_vi"].append(raw_ms)
        latencies["english"].append(english_ms)
        latencies["channel_aware_vi"].append(routed_ms)
        latencies["channel_aware_vi_cache"].append(cached_ms)
        english_top5 = _identities(reference, 5)
        raw_top5 = _identities(raw, 5)
        routed_top5 = _identities(routed, 5)
        english_top1 = english_top5[0] if english_top5 else None
        routed_all = _identities(routed, 20)
        predictions["legacy_raw_vi"][query_id] = _identities(raw, 100)
        predictions["english"][query_id] = _identities(reference, 100)
        predictions["channel_aware_vi"][query_id] = _identities(routed, 100)
        labeled_latencies["legacy_raw_vi"][query_id] = raw_ms
        labeled_latencies["english"][query_id] = english_ms
        labeled_latencies["channel_aware_vi"][query_id] = routed_ms
        rows.append(
            {
                "vi": vietnamese,
                "en": english,
                "translation": routed.get("query_processing", {}),
                "raw_vi_top5_overlap": round(
                    len(set(raw_top5) & set(english_top5)) / max(1, len(set(english_top5))),
                    3,
                ),
                "routed_vi_top5_overlap": round(
                    len(set(routed_top5) & set(english_top5))
                    / max(1, len(set(english_top5))),
                    3,
                ),
                "routed_top1_agrees_with_english": bool(
                    english_top1 is not None and routed_top5 and routed_top5[0] == english_top1
                ),
                "english_top1_rank_in_routed": (
                    routed_all.index(english_top1) + 1
                    if english_top1 is not None and english_top1 in routed_all
                    else None
                ),
                "hit_counts": {
                    "legacy_raw_vi": len(_identities(raw, 100)),
                    "english": len(_identities(reference, 100)),
                    "channel_aware_vi": len(_identities(routed, 100)),
                    "channel_aware_vi_cached": len(_identities(cached, 100)),
                },
                "channel_statuses": routed.get("channel_statuses", {}),
            }
        )

    summary = {
        "mode": "unlabeled_rank_agreement",
        "batch_id": args.batch_id,
        "query_count": len(rows),
        "top1_agreement": round(
            sum(row["routed_top1_agrees_with_english"] for row in rows) / len(rows), 3
        ),
        "top5_overlap": {
            "legacy_raw_vi_mean": round(
                statistics.fmean(row["raw_vi_top5_overlap"] for row in rows), 3
            ),
            "channel_aware_vi_mean": round(
                statistics.fmean(row["routed_vi_top5_overlap"] for row in rows), 3
            ),
        },
        "latency_ms": {
            name: {
                "p50": _percentile(values, 0.5),
                "p95": _percentile(values, 0.95),
            }
            for name, values in latencies.items()
        },
        "queries": rows,
    }
    if args.ground_truth is not None:
        from src.evaluation.textual_kis import evaluate_queries, load_ground_truth

        ground_truth = load_ground_truth(args.ground_truth)
        expected = {f"vi-{index:02d}" for index in range(1, len(PAIRS) + 1)}
        actual = {query.query_id for query in ground_truth}
        if actual != expected:
            raise SystemExit(
                "ground-truth query IDs must be exactly vi-01..vi-10 for this benchmark"
            )
        summary["labeled_evaluation"] = {
            name: evaluate_queries(
                values,
                ground_truth,
                latencies_ms=labeled_latencies[name],
            ).to_dict()
            for name, values in predictions.items()
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    improved = (
        summary["top5_overlap"]["channel_aware_vi_mean"]
        > summary["top5_overlap"]["legacy_raw_vi_mean"]
    )
    return 0 if improved else 2


if __name__ == "__main__":
    raise SystemExit(main())
