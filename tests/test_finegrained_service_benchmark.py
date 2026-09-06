"""Opt-in service benchmark for the reviewed fine-grained culinary seed."""

from __future__ import annotations

import os
import time

import pytest
import requests

from src.runtime_contract import RUNTIME_CONTRACT_VERSION

MULTI_VEHICLE_QUERY = (
    "Footage capturing vehicles wading through flood water, where a yellow car, "
    "a red car, and a black car prepare one after another to cross a bridge."
)


@pytest.mark.integration
def test_reviewed_culinary_seed_is_top_three_in_live_service() -> None:
    if os.environ.get("AIC_RUN_SERVICE_BENCHMARK") != "1":
        pytest.skip("set AIC_RUN_SERVICE_BENCHMARK=1 for the live KIS benchmark")
    api_url = os.environ.get("AIC_API_URL", "http://127.0.0.1:8080").rstrip("/")
    capability = requests.get(f"{api_url}/v1/runtime/capabilities", timeout=(1, 5))
    capability.raise_for_status()
    assert capability.json()["runtime_contract"]["version"] == RUNTIME_CONTRACT_VERSION

    query = (
        "A chef plating vegetarian fresh spring rolls (gỏi cuốn chay) wrapped in "
        "yellow and purple rice paper filled with tofu and rolled greens, delicately "
        "garnished on the plate with fresh green leaves and purple-and-yellow edible "
        "pansy flowers."
    )
    latencies: list[float] = []
    payload = None
    for iteration in range(3):
        started = time.perf_counter()
        response = requests.post(
            f"{api_url}/v1/kis/search",
            json={
                "query_id": f"finegrained-service-{iteration}",
                "original_text": query,
                "query_language": "auto",
                "batch_filters": ["aic25-b1-reference"],
                "limit": 10,
                "refine_raw": False,
                "precision_profile": "standard",
            },
            timeout=(2, 30),
        )
        latencies.append((time.perf_counter() - started) * 1_000)
        response.raise_for_status()
        payload = response.json()
    assert payload is not None
    target_ranks = [
        rank
        for rank, hit in enumerate(payload["hits"], start=1)
        if hit["video_id"] == "L26_V056"
        and 6362 <= int(hit["source_frame_idx"]) <= 6528
    ]
    assert target_ranks and min(target_ranks) <= 3
    processing = payload["query_processing"]
    assert processing["rank_trace"]["tier1_fusion"]
    assert processing["stage_rank_delta"]
    assert max(latencies[1:]) <= latencies[0] * 1.20 or max(latencies[1:]) <= 1_600


@pytest.mark.integration
def test_reviewed_multi_vehicle_seed_is_top_three_in_live_service() -> None:
    if os.environ.get("AIC_RUN_SERVICE_BENCHMARK") != "1":
        pytest.skip("set AIC_RUN_SERVICE_BENCHMARK=1 for the live KIS benchmark")
    api_url = os.environ.get("AIC_API_URL", "http://127.0.0.1:8080").rstrip("/")
    payload = None
    latencies: list[float] = []
    ranks: list[int] = []
    top_ten: list[tuple[tuple[str, int], ...]] = []
    tier2_states: list[tuple[str | None, str | None]] = []
    for iteration in range(3):
        started = time.perf_counter()
        response = requests.post(
            f"{api_url}/v1/kis/search",
            json={
                "query_id": f"multi-vehicle-service-{iteration}",
                "original_text": MULTI_VEHICLE_QUERY,
                "query_language": "en",
                "batch_filters": ["aic25-b1-reference"],
                "limit": 10,
                "refine_raw": False,
                "mode": "automatic",
                "precision_profile": "standard",
            },
            timeout=(2, 30),
        )
        latencies.append((time.perf_counter() - started) * 1_000)
        response.raise_for_status()
        payload = response.json()
        target = [
            rank
            for rank, hit in enumerate(payload["hits"], start=1)
            if hit["video_id"] == "L21_V003"
            and int(hit["source_frame_idx"]) == 26220
        ]
        assert target
        ranks.append(min(target))
        top_ten.append(
            tuple(
                (str(hit["video_id"]), int(hit["source_frame_idx"]))
                for hit in payload["hits"][:10]
            )
        )
        tier2 = payload.get("channel_statuses", {}).get("advanced_rerank", {})
        tier2_states.append((tier2.get("state"), tier2.get("reason_code")))

    assert payload is not None
    assert max(ranks) <= 3
    assert top_ten[0] == top_ten[1] == top_ten[2]
    assert tier2_states[0] == tier2_states[1] == tier2_states[2]
    processing = payload["query_processing"]
    assert processing["semantic_view_rankings"]
    assert processing["sequence_constraints"][0]["source_text"] == (
        "one after another"
    )
    assert processing["prospective_relations"][0]["verification_policy"] == (
        "positive_only"
    )
    assert max(latencies[1:]) <= latencies[0] * 1.20 or max(latencies[1:]) <= 1_600
