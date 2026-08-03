"""Offline tests for batch-aware TRAKE vector-space routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.contracts import TRAKEQueryRequest, TRAKEResponse, TRAKEStage
from src.retrieval.trake_router import TRAKENotReadyError, TRAKEPipelineRouter


def _request(batch_id: str) -> TRAKEQueryRequest:
    return TRAKEQueryRequest(
        query_id="trake-router",
        global_description="một người mua hàng rồi rời đi",
        stages=(
            TRAKEStage(stage_index=1, description="người chọn món hàng"),
            TRAKEStage(stage_index=2, description="người rời cửa hàng"),
        ),
        batch_filters=(batch_id,),
    )


class _Core:
    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.calls: list[tuple[str, dict]] = []
        self.warm_calls: list[str] = []

    def search(self, request, *, default_batch_id, channel_statuses):
        self.calls.append((default_batch_id, dict(channel_statuses)))
        return TRAKEResponse(
            query_id=request.query_id,
            channel_statuses=dict(channel_statuses),
        )

    def warm(self, batch_id: str):
        self.warm_calls.append(batch_id)
        return {"state": "ready"}


class _Readiness:
    def __init__(self, states: dict[str, str]) -> None:
        self.states = states
        self.calls = 0

    def check(self, _batch_id: str, *, mode: str):
        assert mode == "app"
        self.calls += 1
        return SimpleNamespace(
            channels=tuple(
                SimpleNamespace(
                    name=name,
                    to_dict=lambda name=name, state=state: {
                        "name": name,
                        "state": state,
                        "applicability": "required",
                        "resource_name": "must-not-leak",
                    },
                )
                for name, state in self.states.items()
            )
        )


def _config() -> dict:
    return {
        "readiness": {
            "default_batch_kind": "native",
            "batches": {"reference-b1": {"kind": "reference"}},
        }
    }


def test_router_never_crosses_reference_and_native_vector_spaces() -> None:
    organizer = _Core("organizer_clip")
    visual = _Core("visual")
    router = TRAKEPipelineRouter(
        {"organizer_clip": organizer, "visual": visual},
        _config(),
        readiness_service=_Readiness(
            {"organizer_clip": "ready", "visual": "ready"}
        ),
    )

    router.search(_request("reference-b1"))
    router.search(_request("native-b2"))

    assert [call[0] for call in organizer.calls] == ["reference-b1"]
    assert [call[0] for call in visual.calls] == ["native-b2"]
    assert "resource_name" not in organizer.calls[0][1]["organizer_clip"]


def test_router_warms_only_the_compatible_provisioned_channel() -> None:
    organizer = _Core("organizer_clip")
    visual = _Core("visual")
    router = TRAKEPipelineRouter(
        {"organizer_clip": organizer, "visual": visual},
        _config(),
        readiness_service=_Readiness(
            {"organizer_clip": "ready", "visual": "ready"}
        ),
    )

    assert router.warm("reference-b1") == {"state": "ready"}

    assert organizer.warm_calls == ["reference-b1"]
    assert visual.warm_calls == []


def test_router_caches_sanitized_ready_gate_and_supports_invalidation() -> None:
    clock = [10.0]
    readiness = _Readiness({"organizer_clip": "ready"})
    router = TRAKEPipelineRouter(
        {"organizer_clip": _Core("organizer_clip")},
        _config(),
        readiness_service=readiness,
        readiness_cache_seconds=60,
        monotonic=lambda: clock[0],
    )

    router.search(_request("reference-b1"))
    router.search(_request("reference-b1"))
    assert readiness.calls == 1

    router.invalidate_readiness("reference-b1")
    router.search(_request("reference-b1"))
    assert readiness.calls == 2

    clock[0] += 61
    router.search(_request("reference-b1"))
    assert readiness.calls == 3


def test_router_warm_refreshes_gate_after_long_cache_build() -> None:
    clock = [0.0]
    readiness = _Readiness({"organizer_clip": "ready"})
    core = _Core("organizer_clip")

    def slow_warm(batch_id: str):
        core.warm_calls.append(batch_id)
        clock[0] += 85
        return {"state": "ready"}

    core.warm = slow_warm  # type: ignore[method-assign]
    router = TRAKEPipelineRouter(
        {"organizer_clip": core},
        _config(),
        readiness_service=readiness,
        readiness_cache_seconds=60,
        monotonic=lambda: clock[0],
    )

    router.warm("reference-b1")
    router.search(_request("reference-b1"))

    assert readiness.calls == 1


def test_router_does_not_cache_failed_readiness_probe() -> None:
    class RecoveringReadiness(_Readiness):
        def check(self, batch_id: str, *, mode: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary service failure")
            self.calls -= 1
            return super().check(batch_id, mode=mode)

    readiness = RecoveringReadiness({"organizer_clip": "ready"})
    router = TRAKEPipelineRouter(
        {"organizer_clip": _Core("organizer_clip")},
        _config(),
        readiness_service=readiness,
    )

    first_usable, first_statuses = router.channel_readiness("reference-b1")
    second_usable, _second_statuses = router.channel_readiness("reference-b1")

    assert first_usable is False
    assert first_statuses["organizer_clip"]["state"] == "service_unavailable"
    assert second_usable is True
    assert readiness.calls == 2


def test_router_blocks_missing_required_encoder_even_if_index_reports_ready() -> None:
    router = TRAKEPipelineRouter(
        {"visual": _Core("visual")},
        _config(),
        readiness_service=_Readiness({"organizer_clip": "ready"}),
    )

    with pytest.raises(TRAKENotReadyError) as raised:
        router.search(_request("reference-b1"))

    assert raised.value.channel_name == "organizer_clip"
    assert raised.value.reason_code == "compatible_encoder_unavailable"
    assert raised.value.channel_statuses["organizer_clip"]["state"] == "blocked_asset"


def test_router_blocks_unprovisioned_selected_space_without_using_other_space() -> None:
    organizer = _Core("organizer_clip")
    visual = _Core("visual")
    router = TRAKEPipelineRouter(
        {"organizer_clip": organizer, "visual": visual},
        _config(),
        readiness_service=_Readiness(
            {"organizer_clip": "ready", "visual": "not_provisioned"}
        ),
    )

    with pytest.raises(TRAKENotReadyError) as raised:
        router.search(_request("native-b2"))

    assert raised.value.channel_name == "visual"
    assert not organizer.calls
    assert not visual.calls
