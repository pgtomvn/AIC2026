"""HTTP-only benchmark for adaptive, spatial, and color Textual KIS.

The script talks to one already-running FastAPI process. Rank deltas are
operational diagnostics, not retrieval accuracy measurements.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.runtime_contract import RUNTIME_CONTRACT_VERSION

DEFAULT_API_URL = "http://127.0.0.1:8080"
DEFAULT_BATCH_ID = "aic25-b1-reference"
RSS_LIMIT_BYTES = int(2.2 * 1024**3)
P50_LIMIT_MS = 950.0
P95_LIMIT_MS = 1_600.0


@dataclass(frozen=True, slots=True)
class Fixture:
    fixture_id: str
    text: str
    stratum: str = "unspecified"


FIXTURES = (
    Fixture(
        "finegrained-culinary-seed",
        (
            "A chef plating vegetarian fresh spring rolls (gỏi cuốn chay) wrapped "
            "in yellow and purple rice paper filled with tofu and rolled greens, "
            "delicately garnished on the plate with fresh green leaves and "
            "purple-and-yellow edible pansy flowers."
        ),
        "fine_grained",
    ),
    Fixture(
        "multi-vehicle-compositional-seed",
        (
            "Footage capturing vehicles wading through flood water, where a yellow "
            "car, a red car, and a black car prepare one after another to cross a "
            "bridge."
        ),
        "multi_instance_same_class",
    ),
    Fixture(
        "existing-cooking",
        (
            "Snap peas being added to stir-frying squid in a wok, with sliced "
            "onions and red chilies, ending by tossing the wok over a stove flame."
        ),
    ),
    Fixture(
        "existing-night-fishing",
        (
            "A person standing in the water shining a handheld flashlight at "
            "night, then pulling a fishing net at sunrise."
        ),
    ),
    Fixture(
        "synthetic-spatial",
        "A person in a green shirt on the left and a red car on the right.",
    ),
    Fixture(
        "synthetic-color-roi",
        "A cyclist wearing a yellow shirt in the center.",
    ),
    Fixture(
        "synthetic-lexical",
        'A sign showing "London Zoo" and the number 4.',
    ),
)


def _hard_gate_fixtures() -> tuple[Fixture, ...]:
    """Build 100 unique, stratified synthetic performance queries (not qrels)."""

    objects = ("person", "car", "bicycle", "dog", "bus")
    colors = ("red", "blue", "yellow", "green", "white")
    values: list[Fixture] = []
    actions = ("walking", "running", "resting", "carrying an object")
    for index in range(20):
        values.append(Fixture(
            f"visual-{index:03d}",
            f"A {objects[index % len(objects)]} {actions[index // len(objects)]} outdoors.",
            "visual",
        ))
    for index in range(20):
        values.append(Fixture(
            f"lexical-{index:03d}",
            f'A sign or subtitle displays "AIC CODE {index:03d}" and number {index}.' ,
            "lexical",
        ))
    zones = ("left", "right", "center", "top", "bottom")
    for index in range(15):
        values.append(Fixture(
            f"absolute-{index:03d}",
            f"A {objects[index % 5]} on the {zones[index // 5]} of the frame.",
            "absolute_spatial",
        ))
    relations = ("left of", "right of", "above", "below", "inside")
    for index in range(15):
        values.append(Fixture(
            f"relative-{index:03d}",
            f"A person {relations[index % 5]} a {objects[(index // 5 + 1) % 5]}.",
            "relative_spatial",
        ))
    for index in range(15):
        values.append(Fixture(
            f"color-{index:03d}",
            f"A {colors[index % 5]} {objects[index // 5]} in the center.",
            "color",
        ))
    for index in range(15):
        values.append(Fixture(
            f"negation-{index:03d}",
            f"A street scene without a {colors[index % 5]} {objects[index // 5]}.",
            "negation",
        ))
    return tuple(values)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


class ResourceSampler:
    """Sample the server RSS only through its sanitized HTTP status API."""

    def __init__(
        self,
        session: requests.Session,
        api_url: str,
        batch_id: str,
    ) -> None:
        self.session = session
        self.url = f"{api_url}/v1/ingest/status"
        self.batch_id = batch_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                response = self.session.get(
                    self.url,
                    params={"batch_id": self.batch_id},
                    timeout=(1, 2),
                )
                if response.status_code == 200:
                    payload = response.json()
                    resources = payload.get("resources", {})
                    rss = resources.get("process_rss_bytes")
                    if isinstance(rss, int) and not isinstance(rss, bool) and rss >= 0:
                        self.samples.append(rss)
            except (requests.RequestException, TypeError, ValueError):
                pass
            self._stop.wait(0.05)

    def __enter__(self) -> ResourceSampler:
        self._thread = threading.Thread(
            target=self._run,
            name="kis-http-resource-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def _request(
    session: requests.Session,
    *,
    api_url: str,
    batch_id: str,
    fixture: Fixture,
    iteration: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = session.post(
            f"{api_url}/v1/kis/search",
            json={
                "query_id": f"benchmark-advanced-{fixture.fixture_id}-{iteration}",
                "original_text": fixture.text,
                "query_language": "auto",
                "batch_filters": [batch_id],
                "limit": 100,
                "refine_raw": False,
                "precision_profile": "advanced",
            },
            timeout=(2, timeout_seconds),
        )
    except requests.Timeout:
        return {
            "fixture_id": fixture.fixture_id,
            "iteration": iteration,
            "ok": False,
            "error_code": "timeout",
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
        }
    except requests.RequestException:
        return {
            "fixture_id": fixture.fixture_id,
            "iteration": iteration,
            "ok": False,
            "error_code": "request_failed",
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
        }
    latency_ms = (time.perf_counter() - started) * 1_000
    if response.status_code != 200:
        code = "http_error"
        try:
            detail = response.json().get("detail", {})
            if isinstance(detail, dict) and isinstance(detail.get("code"), str):
                code = detail["code"]
        except (TypeError, ValueError):
            pass
        return {
            "fixture_id": fixture.fixture_id,
            "iteration": iteration,
            "ok": False,
            "status_code": response.status_code,
            "error_code": code,
            "latency_ms": round(latency_ms, 3),
        }
    payload = response.json()
    hits = payload.get("hits", [])
    processing = payload.get("query_processing", {})
    advanced = processing.get("advanced_rerank", {})
    rank_deltas = [
        int(hit["advanced_rank_delta"])
        for hit in hits
        if isinstance(hit, dict)
        and isinstance(hit.get("advanced_rank_delta"), int)
        and not isinstance(hit.get("advanced_rank_delta"), bool)
    ]
    seed_rank = next(
        (
            rank
            for rank, hit in enumerate(hits, start=1)
            if isinstance(hit, dict)
            and hit.get("video_id") == "L26_V056"
            and isinstance(hit.get("source_frame_idx"), int)
            and 6362 <= int(hit["source_frame_idx"]) <= 6528
        ),
        None,
    )
    multi_vehicle_rank = next(
        (
            rank
            for rank, hit in enumerate(hits, start=1)
            if isinstance(hit, dict)
            and hit.get("video_id") == "L21_V003"
            and hit.get("source_frame_idx") == 26220
        ),
        None,
    )
    return {
        "fixture_id": fixture.fixture_id,
        "stratum": fixture.stratum,
        "iteration": iteration,
        "ok": bool(hits),
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 3),
        "hit_count": len(hits),
        "intent": processing.get("intent"),
        "effective_rrf_k": processing.get("effective_rrf_k"),
        "advanced_rerank": {
            "state": advanced.get("state"),
            "spatial_latency_ms": advanced.get("spatial_latency_ms"),
            "relation_latency_ms": advanced.get("relation_latency_ms"),
            "negation_latency_ms": advanced.get("negation_latency_ms"),
            "color_latency_ms": advanced.get("color_latency_ms"),
            "total_latency_ms": advanced.get("total_latency_ms"),
            "object_cache": advanced.get("object_cache", {}),
            "color_cache": advanced.get("color_cache", {}),
        },
        "rank_delta": {
            "changed": sum(value != 0 for value in rank_deltas),
            "max_promotion": max(rank_deltas, default=0),
            "max_demotion": min(rank_deltas, default=0),
        },
        "finegrained_seed_rank": seed_rank,
        "multi_vehicle_seed_rank": multi_vehicle_rank,
        "top_ten": [
            [hit.get("video_id"), hit.get("source_frame_idx")]
            for hit in hits[:10]
            if isinstance(hit, dict)
        ],
        "tier2_state": advanced.get("state"),
        "rank_trace": processing.get("rank_trace", {}),
        "stage_rank_delta": processing.get("stage_rank_delta", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--query-file",
        help="JSON list of at least 100 unique {fixture_id,text} records for hard gating",
    )
    parser.add_argument("--hard-gate", action="store_true")
    parser.add_argument(
        "--finegrained-seed-gate",
        action="store_true",
        help="require L26_V056 frames 6362-6528 in Top-3 for the reviewed seed query",
    )
    parser.add_argument(
        "--multi-vehicle-seed-gate",
        action="store_true",
        help=(
            "require exact L21_V003:26220 in Top-3 with stable Top-10/Tier-2 "
            "state across three or more runs"
        ),
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 2:
        parser.error("--iterations must be at least 2 for cold/warm reporting")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    fixtures = FIXTURES
    if args.query_file:
        values = json.loads(Path(args.query_file).read_text(encoding="utf-8"))
        if not isinstance(values, list):
            parser.error("--query-file must contain a JSON list")
        fixtures = tuple(
            Fixture(
                str(value["fixture_id"]),
                str(value["text"]),
                str(value.get("stratum", "operator")),
            )
            for value in values
            if isinstance(value, dict) and value.get("fixture_id") and value.get("text")
        )
    elif args.hard_gate:
        fixtures = _hard_gate_fixtures()
    unique_ids = {value.fixture_id for value in fixtures}
    unique_texts = {value.text.casefold() for value in fixtures}
    if len(unique_ids) != len(fixtures):
        parser.error("benchmark fixture IDs must be unique")
    if len(unique_texts) != len(fixtures):
        parser.error("benchmark query texts must be unique")
    if args.hard_gate and len(fixtures) < 100:
        parser.error("--hard-gate requires at least 100 unique candidate sets")
    strata = {value.stratum for value in fixtures}
    if args.hard_gate and len(strata) < 6:
        parser.error("--hard-gate fixtures must cover at least six strata")

    api_url = str(args.api_url).rstrip("/")
    session = requests.Session()
    try:
        capability_response = session.get(
            f"{api_url}/v1/runtime/capabilities",
            timeout=(1, 3),
        )
    except requests.RequestException:
        print(json.dumps({"ok": False, "code": "api_unavailable"}))
        return 2
    if capability_response.status_code != 200:
        print(json.dumps({"ok": False, "code": "api_unhealthy"}))
        return 2
    capabilities = capability_response.json()
    contract = capabilities.get("runtime_contract", {})
    features = set(contract.get("features", ()))
    required = {
        "kis.adaptive_rrf",
        "kis.spatial_color_rerank",
        "kis.temporal_adjacency",
        "kis.relative_relation_graph",
        "kis.soft_negation",
        "kis.strict_color_binding",
    }
    if contract.get("version") != RUNTIME_CONTRACT_VERSION or not required.issubset(features):
        print(json.dumps({"ok": False, "code": "runtime_contract_mismatch"}))
        return 2

    runs: list[dict[str, Any]] = []
    with ResourceSampler(session, api_url, args.batch_id) as sampler:
        for iteration in range(1, args.iterations + 1):
            for fixture in fixtures:
                runs.append(
                    _request(
                        session,
                        api_url=api_url,
                        batch_id=args.batch_id,
                        fixture=fixture,
                        iteration=iteration,
                        timeout_seconds=args.timeout_seconds,
                    )
                )

    cold = [float(runs[0]["latency_ms"])] if runs and runs[0].get("ok") else []
    cold_evidence = [
        float(item["latency_ms"])
        for item in runs
        if item.get("ok") and int(item["iteration"]) == 1
    ]
    hot_cache = [
        float(item["latency_ms"])
        for item in runs
        if item.get("ok") and int(item["iteration"]) > 1
    ]
    peak_rss = max(sampler.samples, default=None)
    failures = [item for item in runs if not item.get("ok")]
    rss_gate = peak_rss is not None and peak_rss < RSS_LIMIT_BYTES
    cold_p50 = _percentile(cold_evidence, 0.50)
    cold_p95 = _percentile(cold_evidence, 0.95)
    latency_gate = (
        cold_p50 is not None and cold_p95 is not None
        and cold_p50 <= P50_LIMIT_MS and cold_p95 <= P95_LIMIT_MS
    )
    hard_gate_passed = not failures and rss_gate and latency_gate
    seed_runs = [
        item for item in runs if item.get("fixture_id") == "finegrained-culinary-seed"
    ]
    finegrained_seed_gate_passed = bool(seed_runs) and all(
        isinstance(item.get("finegrained_seed_rank"), int)
        and 1 <= int(item["finegrained_seed_rank"]) <= 3
        for item in seed_runs
    )
    multi_vehicle_runs = [
        item
        for item in runs
        if item.get("fixture_id") == "multi-vehicle-compositional-seed"
    ]
    multi_vehicle_seed_gate_passed = (
        len(multi_vehicle_runs) >= 3
        and all(
            isinstance(item.get("multi_vehicle_seed_rank"), int)
            and 1 <= int(item["multi_vehicle_seed_rank"]) <= 3
            for item in multi_vehicle_runs
        )
        and len(
            {
                json.dumps(item.get("top_ten", []), separators=(",", ":"))
                for item in multi_vehicle_runs
            }
        )
        == 1
        and len({item.get("tier2_state") for item in multi_vehicle_runs}) == 1
    )
    tier_latency: dict[str, dict[str, float | None]] = {}
    tier2_deadline_count = sum(
        isinstance(item.get("advanced_rerank"), dict)
        and item["advanced_rerank"].get("state") == "deadline_exceeded"
        for item in runs
    )
    for field in (
        "spatial_latency_ms",
        "relation_latency_ms",
        "negation_latency_ms",
        "color_latency_ms",
        "total_latency_ms",
    ):
        values = [
            float(value)
            for item in runs
            if isinstance(item.get("advanced_rerank"), dict)
            and item["advanced_rerank"].get("state") == "ready"
            and isinstance((value := item["advanced_rerank"].get(field)), (int, float))
            and not isinstance(value, bool)
        ]
        tier_latency[field] = {
            "p50": _rounded(_percentile(values, 0.50)),
            "p95": _rounded(_percentile(values, 0.95)),
        }
    last_advanced = next(
        (
            item["advanced_rerank"]
            for item in reversed(runs)
            if isinstance(item.get("advanced_rerank"), dict)
            and item["advanced_rerank"].get("state") == "ready"
        ),
        {},
    )
    report = {
        "ok": (
            (hard_gate_passed if args.hard_gate else not failures)
            and (
                finegrained_seed_gate_passed
                if args.finegrained_seed_gate
                else True
            )
            and (
                multi_vehicle_seed_gate_passed
                if args.multi_vehicle_seed_gate
                else True
            )
        ),
        "hard_gate_requested": bool(args.hard_gate),
        "hard_gate_passed": hard_gate_passed,
        "finegrained_seed_gate_requested": bool(args.finegrained_seed_gate),
        "finegrained_seed_gate_passed": finegrained_seed_gate_passed,
        "multi_vehicle_seed_gate_requested": bool(args.multi_vehicle_seed_gate),
        "multi_vehicle_seed_gate_passed": multi_vehicle_seed_gate_passed,
        "batch_id": args.batch_id,
        "http_only": True,
        "rank_delta_is_accuracy": False,
        "unlabeled_metrics_are_precision": False,
        "fixture_count": len(fixtures),
        "unique_candidate_sets": len(unique_texts),
        "strata": sorted(strata),
        "synthetic_performance_queries_are_accuracy": False,
        "request_count": len(runs),
        "successful_requests": len(runs) - len(failures),
        "failed_requests": len(failures),
        "latency_ms": {
            "cold_first_request": _rounded(cold[0] if cold else None),
            "warm_model_cold_evidence_p50": _rounded(cold_p50),
            "warm_model_cold_evidence_p95": _rounded(cold_p95),
            "hot_cache_p50": _rounded(_percentile(hot_cache, 0.50)),
            "hot_cache_p95": _rounded(_percentile(hot_cache, 0.95)),
            "hot_cache_max": _rounded(max(hot_cache) if hot_cache else None),
            "hard_limits": {"p50": P50_LIMIT_MS, "p95": P95_LIMIT_MS},
            "latency_gate_passed": latency_gate,
        },
        "tier_latency_ms": tier_latency,
        "tier2_transaction": {
            "ready_sample_count": sum(
                isinstance(item.get("advanced_rerank"), dict)
                and item["advanced_rerank"].get("state") == "ready"
                for item in runs
            ),
            "deadline_fallback_count": tier2_deadline_count,
            "deadline_fallback_preserves_stage1": True,
        },
        "final_cache_state": {
            "object_cache": last_advanced.get("object_cache", {}),
            "color_cache": last_advanced.get("color_cache", {}),
        },
        "resources": {
            "scope": "fastapi_process_only",
            "sample_interval_ms": 50,
            "rss_sample_count": len(sampler.samples),
            "peak_fastapi_rss_bytes": peak_rss,
            "hard_limit_bytes": RSS_LIMIT_BYTES,
            "rss_gate_passed": rss_gate,
        },
    }
    if not args.summary_only:
        report["runs"] = runs
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
