"""Operational acceptance must reject false-green fallback and resource results."""

import json
from copy import deepcopy

import pytest

from scripts.benchmark_kis_operational import anchor_rank, channel_failures, compare_reports


def report():
    result = {"batch_id": "b", "query_sha256": "q", "split_sha256": "s",
            "partition": "development", "modes": ["asr_only"], "repeats": 2,
            "refine_raw": False, "api_peak_rss_bytes": 100,
            "summary": {"asr_only": {"p95_ms": 100}},
            "queries": [{"query_id": "q", "mode": "asr_only", "trial": 0,
                         "hits": [{"video_id": "v", "source_frame_idx": 0, "fused_score": .5}],
                         "error": None, "channel_failures": {}}]}
    result["queries"].append({**deepcopy(result["queries"][0]), "trial": 1})
    return result


def test_gate_rejects_duplicate_or_incomplete_trial_matrix():
    baseline = report()
    candidate = deepcopy(baseline)
    candidate["queries"].append(deepcopy(candidate["queries"][0]))
    assert not compare_reports(baseline, candidate)["passed"]
    candidate = deepcopy(baseline)
    candidate["queries"].pop()
    assert not compare_reports(candidate, candidate)["passed"]


def test_gate_accepts_identical_and_rejects_latency_rss_and_unknown_metrics():
    baseline = report()
    assert compare_reports(baseline, deepcopy(baseline))["passed"]
    for key in ("latency", "rss", "missing"):
        candidate = deepcopy(baseline)
        if key == "latency":
            candidate["summary"]["asr_only"]["p95_ms"] = 121
        else:
            candidate["api_peak_rss_bytes"] = 121 if key == "rss" else None
        assert not compare_reports(baseline, candidate)["passed"]


def test_branch_errors_are_failures_but_intent_skip_is_not():
    assert not channel_failures({"channel_statuses": {"asr": {"state": "skipped"}}})
    failures = channel_failures({"hits": [{}], "channel_statuses": {
        "asr": {"state": "service_unavailable", "reason_code": "channel_query_failed"}}})
    assert failures == {"asr": "channel_query_failed"}
    candidate = report()
    candidate["queries"][0]["channel_failures"] = failures
    assert not compare_reports(report(), candidate)["passed"]


def test_api_sanitized_deadline_without_state_still_fails():
    assert channel_failures({"channel_statuses": {"advanced_rerank": {
        "reason_code": "tier2_deadline_exceeded"}}})
    assert channel_failures({"query_processing": {"advanced_rerank": {
        "state": "deadline_exceeded"}}})


def test_invariance_includes_scores_and_evidence_not_just_ids():
    baseline = report()
    candidate = deepcopy(baseline)
    candidate["queries"][0]["hits"][0]["fused_score"] = .6
    assert not compare_reports(baseline, candidate)["passed"]
    assert not compare_reports({**baseline, "queries": []}, {**baseline, "queries": []})["passed"]


def test_anchor_is_exact_positive_frame_and_accepts_zero():
    labels = [{"video_id": "v", "start_frame_idx": 0, "end_frame_idx": 0}]
    assert anchor_rank([{"video_id": "v", "source_frame_idx": 1}], labels) is None
    assert anchor_rank([{"video_id": "v", "source_frame_idx": 0}], labels) == 1


@pytest.mark.parametrize("resource_failure", ["latency", "rss"])
def test_cli_exits_nonzero_on_resource_gate_failure(tmp_path, monkeypatch, resource_failure):
    from scripts import benchmark_kis_operational as benchmark

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return self

        def raise_for_status(self):
            pass

        def json(self):
            return {"launch_identity": {"canonical": True}}

    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(json.dumps({"query_id": "q", "text": "hello"}), encoding="utf-8")
    baseline = report()
    baseline.update(query_sha256=benchmark.sha(query_path), split_sha256=None)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    output = tmp_path / "candidate.json"

    def capture(*args):
        return {**deepcopy(baseline["queries"][0]), "trial": args[5],
                "latency_ms": 121 if resource_failure == "latency" else 100,
                "has_label": False, "asr_hits": 1, "distinct_videos": 1, "top_share": 1}

    monkeypatch.setattr(benchmark.requests, "Session", Session)
    monkeypatch.setattr(benchmark, "owns_api_port", lambda *args: True)
    monkeypatch.setattr(benchmark, "capture_one", capture)
    monkeypatch.setattr(benchmark, "api_rss", lambda pid: 121 if resource_failure == "rss" else 100)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    assert benchmark.main([
        "--proxy-queries", str(query_path), "--batch-id", "b", "--api-pid", "1",
        "--output", str(output), "--baseline", str(baseline_path), "--modes", "asr_only",
    ]) == 2
    assert not json.loads(output.read_text(encoding="utf-8"))["gate"]["passed"]
