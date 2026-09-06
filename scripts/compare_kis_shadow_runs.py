"""Compare rank-invariant shadow runs against an unjudged proxy baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.utils.io_helpers import atomic_write_bytes


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not str(value.get("evaluation_kind", "")).startswith("proxy_unjudged"):
        raise ValueError(f"{path} is not an unjudged proxy run")
    if value.get("activation_eligible") is not False:
        raise ValueError(f"{path} must be explicitly activation-ineligible")
    return value


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _feature_summary(run: dict[str, Any]) -> dict[str, Any]:
    feature = str(run["feature"])
    diagnostics = [dict(row.get("feature_diagnostics") or {}) for row in run["queries"]]
    latencies = [float(row.get("latency_ms", 0.0)) for row in run["queries"]][1:]
    output: dict[str, Any] = {
        "feature": feature,
        "query_count": len(run["queries"]),
        "warm_latency_p50_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "warm_latency_p95_ms": round(_percentile(latencies, 0.95), 3),
    }
    if feature == "candidate_rescue":
        triggered = [row for row in diagnostics if row.get("triggered")]
        output.update(
            {
                "triggered_queries": len(triggered),
                "mean_stage_latency_ms": round(
                    statistics.fmean(float(row.get("latency_ms", 0.0)) for row in triggered),
                    3,
                )
                if triggered
                else 0.0,
            }
        )
    elif feature == "video_local":
        scored = [row for row in diagnostics if int(row.get("scored_video_count", 0)) > 0]
        output.update(
            {
                "scored_queries": len(scored),
                "mean_scored_videos": round(
                    statistics.fmean(int(row["scored_video_count"]) for row in scored), 3
                )
                if scored
                else 0.0,
                "mean_stage_latency_ms": round(
                    statistics.fmean(float(row.get("latency_ms", 0.0)) for row in scored), 3
                )
                if scored
                else 0.0,
            }
        )
    elif feature == "ocr_quality":
        eligible = [row for row in diagnostics if int(row.get("candidate_count", 0)) > 0]
        feature_count = sum(int(row.get("feature_count", 0)) for row in eligible)
        candidate_count = sum(
            int(row.get("evaluated_candidate_count", row.get("candidate_count", 0)))
            for row in eligible
        )
        output.update(
            {
                "eligible_queries": len(eligible),
                "conditional_feature_coverage": round(
                    feature_count / candidate_count if candidate_count else 0.0, 6
                ),
                "quoted_text_exemptions": sum(
                    int(row.get("quoted_text_exemptions", 0)) for row in eligible
                ),
            }
        )
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# KIS proxy shadow execution report",
        "",
        "This report contains operational measurements only. The proxy queries are unjudged, "
        "so no retrieval-quality or production-activation claim is made.",
        "",
        f"- Queries: {report['query_count']}",
        f"- Exact rank identity across shadow runs: {report['all_rank_invariant']}",
        f"- Production activation eligible: {report['activation_eligible']}",
        "",
        "| Feature | Rank invariant | Warm p50 ms | Warm p95 ms | p95 / baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        lines.append(
            "| {feature} | {rank_invariant} | {warm_latency_p50_ms:.3f} | "
            "{warm_latency_p95_ms:.3f} | {warm_p95_ratio:.3f} |".format(**run)
        )
    lines.extend(
        [
            "",
            "All configured rollout features remain shadow-only. Activation still requires a "
            "separately locked judged evaluation set.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    baseline_rows = {str(row["query_id"]): row for row in baseline["queries"]}
    source_sha = baseline["query_source_sha256"]
    baseline_p95 = float(_feature_summary(baseline)["warm_latency_p95_ms"])
    comparisons: list[dict[str, Any]] = []
    for path in args.run:
        run = _load(path)
        if run["query_source_sha256"] != source_sha:
            raise ValueError(f"query source mismatch: {path}")
        run_rows = {str(row["query_id"]): row for row in run["queries"]}
        if set(run_rows) != set(baseline_rows):
            raise ValueError(f"query set mismatch: {path}")
        mismatches = [
            query_id
            for query_id, row in baseline_rows.items()
            if row["result_ids"] != run_rows[query_id]["result_ids"]
        ]
        summary = _feature_summary(run)
        summary.update(
            {
                "source": path.as_posix(),
                "rank_invariant": not mismatches,
                "rank_mismatch_query_ids": mismatches,
                "warm_p95_ratio": round(
                    float(summary["warm_latency_p95_ms"]) / baseline_p95
                    if baseline_p95
                    else 0.0,
                    6,
                ),
            }
        )
        comparisons.append(summary)

    report = {
        "schema_version": 1,
        "evaluation_kind": "proxy_unjudged",
        "activation_eligible": False,
        "query_source_sha256": source_sha,
        "query_count": len(baseline_rows),
        "baseline": {
            **_feature_summary(baseline),
            "source": args.baseline.as_posix(),
        },
        "runs": comparisons,
        "all_rank_invariant": all(run["rank_invariant"] for run in comparisons),
        "interpretation": "operational_only_no_quality_claim",
    }
    atomic_write_bytes(
        args.output_json,
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
