"""Capture the canonical API; compare immutable reports without loading models.

Capture baseline/candidate in separate Launcher runs. Only their source version
may differ; this tool never changes ranking settings or initializes a database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests

MODES = ("visual_only", "asr_only", "automatic")
ROOT = Path(__file__).resolve().parent.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def channel_failures(payload: Mapping[str, Any]) -> dict[str, str]:
    """Include channel failures hidden by successful fallback hits."""
    failures = dict(payload.get("channel_errors") or {})
    for name, status in (payload.get("channel_statuses") or {}).items():
        if status.get("state") in {"service_unavailable", "blocked_dependency", "failed", "error", "degraded", "deadline_exceeded"} or status.get("reason_code") in {"tier2_deadline_exceeded", "advanced_evidence_unavailable", "cross_modal_evidence_unavailable"}:
            failures[name] = status.get("reason_code", "channel_failed")
    for name in ("advanced_rerank", "video_local", "candidate_rescue", "ocr_quality"):
        diagnostic = (payload.get("query_processing") or {}).get(name, {})
        if diagnostic.get("state") in {"degraded", "deadline_exceeded", "failed", "error"}:
            failures[name] = diagnostic.get("reason_code", diagnostic["state"])
    return failures


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def anchor_rank(hits: list[dict], intervals: list[dict]) -> int | None:
    """Match only annotated positive anchors, never expand relevance implicitly."""
    for rank, hit in enumerate(hits, 1):
        if any(hit["video_id"] == item["video_id"] and
               item["start_frame_idx"] <= hit["source_frame_idx"] <= item["end_frame_idx"]
               for item in intervals):
            return rank
    return None


def summary(rows: list[dict]) -> dict:
    result = {}
    for mode in sorted({r["mode"] for r in rows}):
        selected = [r for r in rows if r["mode"] == mode]
        latencies = [r["latency_ms"] for r in selected]
        judged = [r for r in selected if r["has_label"]]
        result[mode] = {
            "request_count": len(selected),
            "error_count": sum(bool(r["error"] or r["channel_failures"]) for r in selected),
            "p50_ms": percentile(latencies, .5), "p95_ms": percentile(latencies, .95),
            "asr_participation_requests": sum(r["asr_hits"] > 0 for r in selected),
            "mean_distinct_videos_top50": statistics.fmean(r["distinct_videos"] for r in selected),
            "mean_top_video_share_top50": statistics.fmean(r["top_share"] for r in selected),
        }
        if judged:
            result[mode]["anchor_mrr"] = statistics.fmean(1 / r["anchor_rank"] if r["anchor_rank"] else 0 for r in judged)
            for k in (1, 3, 5, 100):
                result[mode][f"anchor_recall_at_{k}"] = statistics.fmean(bool(r["anchor_rank"] and r["anchor_rank"] <= k) for r in judged)
    return result


def compare_reports(baseline: Mapping, candidate: Mapping) -> dict:
    """Fail closed, including missing metrics and resource gate failures."""
    compatible = all(baseline.get(k) == candidate.get(k) for k in (
        "batch_id", "query_sha256", "split_sha256", "partition", "modes", "repeats", "refine_raw"))
    left = {(r["query_id"], r["mode"], r["trial"]): r for r in baseline["queries"]}
    right = {(r["query_id"], r["mode"], r["trial"]): r for r in candidate["queries"]}
    complete = True
    for report, indexed in ((baseline, left), (candidate, right)):
        query_ids = {r["query_id"] for r in report["queries"]}
        expected = {(qid, mode, trial) for qid in query_ids
                    for mode in report["modes"] for trial in range(report["repeats"])}
        complete = complete and bool(expected) and len(indexed) == len(report["queries"]) and indexed.keys() == expected
    invariant = bool(left) and left.keys() == right.keys() and all(left[k]["hits"] == right[k]["hits"] for k in left)
    errors = all(not r["error"] and not r["channel_failures"] for r in [*left.values(), *right.values()])
    latency = {}
    for mode in baseline["modes"]:
        a = baseline["summary"].get(mode, {}).get("p95_ms")
        b = candidate["summary"].get(mode, {}).get("p95_ms")
        latency[mode] = isinstance(a, (int, float)) and a > 0 and isinstance(b, (int, float)) and b <= a * 1.2
    a, b = baseline.get("api_peak_rss_bytes"), candidate.get("api_peak_rss_bytes")
    rss = isinstance(a, int) and a > 0 and isinstance(b, int) and b <= a * 1.2
    return {"compatible": compatible, "complete_repeated_matrix": complete,
            "rank_score_evidence_metadata_invariant": invariant,
            "zero_errors": errors, "latency_within_1_2x": latency, "rss_within_1_2x": rss,
            "passed": bool(compatible and complete and invariant and errors and latency and all(latency.values()) and rss),
            "activation_allowed": False}


def api_rss(pid: int) -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except (psutil.Error, OSError):
        return None


def owns_api_port(pid: int, url: str) -> bool:
    """Do not measure RSS of a wrapper or unrelated process."""
    try:
        import psutil
    except ImportError:
        return False
    port = urlparse(url).port or 80
    try:
        return any(c.status == psutil.CONN_LISTEN and c.laddr.port == port
                   for c in psutil.Process(pid).net_connections(kind="tcp"))
    except (psutil.Error, OSError):
        return False


def capture_one(session: Any, url: str, query: dict, mode: str,
                batch: str, trial: int, refine_raw: bool) -> dict:
    started = time.monotonic()
    payload: dict = {}
    error = None
    try:
        response = session.post(url + "/v1/kis/search", json={
            "query_id": f"operational-{query['query_id']}-{mode}",
            "original_text": query["text"], "mode": mode, "batch_filters": [batch],
            "limit": 100, "refine_raw": refine_raw,
        }, timeout=(5, 120))
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        error = type(exc).__name__
    elapsed = (time.monotonic() - started) * 1000
    hits = payload.get("hits", [])
    videos = [h["video_id"] for h in hits[:50]]
    return {"query_id": query["query_id"], "mode": mode, "trial": trial,
            "latency_ms": elapsed, "hits": hits, "error": error,
            "channel_failures": channel_failures(payload),
            "channel_statuses": payload.get("channel_statuses", {}),
            "query_processing": payload.get("query_processing", {}),
            "has_label": bool(query.get("intervals")),
            "anchor_rank": anchor_rank(hits, query.get("intervals", [])),
            "asr_hits": sum(any("asr" in k for k in h.get("channel_ranks", {})) for h in hits),
            "distinct_videos": len(set(videos)),
            "top_share": max((videos.count(v) for v in set(videos)), default=0) / max(1, len(videos))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-queries", type=Path, required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--partition", choices=("development", "train", "validation", "test"), default="development")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--refine-raw", action="store_true")
    args = parser.parse_args(argv)
    if urlparse(args.api_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("only loopback APIs are allowed")
    if args.output.exists() or args.repeats < 2 or args.api_pid <= 0:
        parser.error("require a new output path, repeats >= 2 and live API PID")
    modes = list(dict.fromkeys(args.modes.split(",")))
    if not modes or set(modes) - set(MODES):
        parser.error("invalid request modes")
    queries = [json.loads(line) for line in args.proxy_queries.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [q.get("query_id") for q in queries]
    if not queries or len(ids) != len(set(ids)) or any(not q.get("text") for q in queries):
        parser.error("nonempty queries with unique IDs and original text are required")
    if args.split_file:
        split = json.loads(args.split_file.read_text(encoding="utf-8"))["query_ids"]
        selected = split["train"] + split["validation"] if args.partition == "development" else split[args.partition]
        if not set(selected) <= set(ids):
            parser.error("unknown query in split")
        queries = [q for q in queries if q["query_id"] in selected]
    if not queries:
        parser.error("empty selected partition")
    if not owns_api_port(args.api_pid, args.api_url):
        parser.error("api-pid must own the API listening port; RSS cannot use the launcher PID")
    source_hashes = {p: sha(ROOT / p) for p in (
        "src/runtime.py", "src/retrieval/textual_kis.py", "src/retrieval/frame_resolution.py",
        "config/settings.yaml", "scripts/benchmark_kis_operational.py")}
    rows, rss = [], []
    with requests.Session() as session:
        health_response = session.get(args.api_url + "/v1/health/status", timeout=60)
        health_response.raise_for_status()
        health = health_response.json()
        readiness_response = session.get(args.api_url + "/v1/readiness", timeout=60)
        readiness_response.raise_for_status()
        loaded_runtime_identity = readiness_response.json().get("launch_identity", {})
        for trial in range(-1, args.repeats):
            for mode in modes:
                for query in (queries if trial % 2 == 0 else list(reversed(queries))):
                    row = capture_one(session, args.api_url, query, mode, args.batch_id, trial, args.refine_raw)
                    if trial >= 0:
                        rows.append(row)
                        rss.append(api_rss(args.api_pid))
                    elif row["error"] or row["channel_failures"]:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        with args.output.open("x", encoding="utf-8") as handle:
                            json.dump({"evaluation_kind": "failed_warmup", "passed": False,
                                       "activation_eligible": False, "failure": row}, handle, ensure_ascii=False, indent=2)
                        raise RuntimeError(f"Warmup failed: {query['query_id']} {mode}: {row['channel_failures'] or row['error']}")
                    print(json.dumps({"trial": trial, "mode": mode, "query": query["query_id"], "ms": round(row["latency_ms"]), "error": row["error"], "failures": row["channel_failures"]}), flush=True)
                    time.sleep(.5)
    report = {"schema_version": 2, "evaluation_kind": "canonical_api_pilot",
              "batch_id": args.batch_id, "query_sha256": sha(args.proxy_queries),
              "split_sha256": sha(args.split_file) if args.split_file else None,
              "partition": args.partition, "modes": modes, "repeats": args.repeats,
              "refine_raw": args.refine_raw, "health_before": health,
              "loaded_runtime_identity": loaded_runtime_identity,
              "source_hashes": source_hashes,
              "api_peak_rss_bytes": max(rss) if rss and all(isinstance(m, int) for m in rss) else None,
              "queries": rows, "summary": summary(rows), "activation_eligible": False}
    if args.baseline:
        report["gate"] = compare_reports(json.loads(args.baseline.read_text(encoding="utf-8")), report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    valid = all(not r["error"] and not r["channel_failures"] for r in rows)
    valid = valid and report["api_peak_rss_bytes"] is not None and report.get("gate", {"passed": True})["passed"]
    print(json.dumps({"output": str(args.output), "passed": bool(valid), "summary": report["summary"]}, ensure_ascii=False))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
