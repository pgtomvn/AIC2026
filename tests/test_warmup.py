"""CPU-only tests for the explicit runtime warm-up state machine."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from PIL import Image

from src.warmup import WarmupService, safe_resource_snapshot


class _Store:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.calls = 0

    def is_available(self, _filters=None) -> bool:
        self.calls += 1
        return self.available


class _Encoder:
    def __init__(
        self,
        *,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.calls = 0
        self.started = started
        self.release = release

    def encode_text(self, _text: str):
        self.calls += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        return [[1.0, 0.0]]


class _Pipeline:
    def __init__(self, dense=(), lexical=()) -> None:
        self.dense_channels = list(dense)
        self.lexical_channels = list(lexical)
        self.raw_refiner = None

    @staticmethod
    def _batch_filters(_request):
        return {"batch_id": ["batch-1"]}, "batch_id = 'batch-1'"


class _Readiness:
    def __init__(self, states: dict[str, str], required: tuple[str, ...]) -> None:
        self.states = states
        self.required = required

    def check(self, _batch_id: str, mode: str = "app"):
        assert mode == "app"
        return SimpleNamespace(
            kis=SimpleNamespace(required_channels=self.required),
            channels=tuple(
                SimpleNamespace(name=name, state=state)
                for name, state in self.states.items()
            ),
        )


def _service(pipeline: _Pipeline, readiness: _Readiness) -> WarmupService:
    return WarmupService(
        pipeline=pipeline,
        readiness_service=readiness,
        batch_id="batch-1",
        profile="laptop",
        resource_snapshot=lambda: {"process_rss_bytes": 1},
    )


def test_b02_warmup_never_builds_legacy_object_or_trajectory_cache():
    def forbidden(*args):
        raise AssertionError("unreleased B02 cache path")
    pipeline = _Pipeline()
    pipeline.advanced_reranker = SimpleNamespace(object_archive=SimpleNamespace(warmup=forbidden))
    pipeline.raw_refiner = SimpleNamespace(encoder=SimpleNamespace(encode_text=forbidden))
    class DisabledReadiness(_Readiness):
        def check(self, *args, **kwargs):
            report = super().check(*args, **kwargs)
            report.trake = SimpleNamespace(state="blocked", required_channels=("organizer_clip",),
                unavailable_channels=("batch_temporal_contract_unavailable",))
            return report
    service = WarmupService(pipeline=pipeline, readiness_service=DisabledReadiness({}, ()),
        batch_id="aic26-b2", profile="laptop",
        artifact_repository=SimpleNamespace(has_refinable_frames=forbidden),
        trake_pipeline=SimpleNamespace(channel_for_batch=lambda batch: "organizer_clip", warm=forbidden),
        resource_snapshot=lambda: {})
    service.start()
    snapshot = service.wait(5)
    assert all(row["name"] not in {"object_evidence", "raw_refiner", "trake_organizer_clip_trajectory"}
               for row in snapshot["channels"])
    assert snapshot["workflows"]["trake"]["reason_code"] == "batch_temporal_contract_unavailable"


def test_resource_snapshot_reports_process_memory_without_optional_psutil() -> None:
    resources = safe_resource_snapshot()

    assert isinstance(resources["process_rss_bytes"], int)
    assert resources["process_rss_bytes"] > 0


def test_warmup_is_single_flight_and_becomes_ready() -> None:
    started = threading.Event()
    release = threading.Event()
    encoder = _Encoder(started=started, release=release)
    channel = SimpleNamespace(name="visual", encoder=encoder, store=_Store())
    service = _service(
        _Pipeline(dense=(channel,)),
        _Readiness({"visual": "ready"}, ("visual",)),
    )

    assert service.start() is True
    assert started.wait(timeout=5)
    assert service.start() is False
    assert service.snapshot()["state"] == "warming"
    release.set()
    snapshot = service.wait(5)

    assert snapshot["state"] == "ready"
    assert snapshot["schema_version"] == 2
    assert snapshot["search_usable"] is True
    assert snapshot["workflows"]["kis"]["usable"] is True
    assert encoder.calls == 1
    assert service.readiness_report() is not None
    assert service.readiness_report().kis.required_channels == ("visual",)
    assert service.start() is False


def test_unprovisioned_dense_encoder_is_skipped_but_lexical_path_is_usable() -> None:
    encoder = _Encoder()
    visual = SimpleNamespace(name="visual", encoder=encoder, store=_Store(False))
    metadata_store = _Store(True)
    metadata = SimpleNamespace(name="metadata", store=metadata_store)
    service = _service(
        _Pipeline(dense=(visual,), lexical=(metadata,)),
        _Readiness(
            {"visual": "not_provisioned", "metadata": "ready"},
            ("visual",),
        ),
    )

    service.start()
    snapshot = service.wait(5)

    assert snapshot["state"] == "degraded"
    assert snapshot["search_usable"] is True
    assert encoder.calls == 0
    assert metadata_store.calls == 1
    assert {item["name"]: item["state"] for item in snapshot["channels"]} == {
        "visual": "not_provisioned",
        "metadata": "ready",
    }


def test_warmup_failure_is_sanitized_and_ready_state_can_be_evicted() -> None:
    class _BrokenEncoder(_Encoder):
        def encode_text(self, _text: str):
            raise RuntimeError("secret provider path")

    channel = SimpleNamespace(name="visual", encoder=_BrokenEncoder(), store=_Store())
    failed = _service(
        _Pipeline(dense=(channel,)),
        _Readiness({"visual": "ready"}, ("visual",)),
    )
    failed.start()
    snapshot = failed.wait(5)
    assert snapshot["state"] == "failed"
    assert snapshot["channels"][0]["reason_code"] == "warmup_runtime_failure"
    assert "secret" not in str(snapshot)

    encoder = _Encoder()
    ready_channel = SimpleNamespace(name="visual", encoder=encoder, store=_Store())
    ready = _service(
        _Pipeline(dense=(ready_channel,)),
        _Readiness({"visual": "ready"}, ("visual",)),
    )
    ready.start()
    assert ready.wait(5)["state"] == "ready"
    ready.mark_evicted()
    assert ready.snapshot()["state"] == "evicted"
    assert ready.snapshot()["search_usable"] is False
    assert ready.snapshot()["workflows"]["kis"]["state"] == "evicted"


def test_eligible_raw_refiner_uses_real_rgb_processor_input() -> None:
    calls: list[object] = []

    class _RefinerEncoder:
        @staticmethod
        def encode_text(text: str) -> list[list[float]]:
            calls.append(("text", text))
            return [[1.0, 0.0]]

        @staticmethod
        def encode_images(images: list[Image.Image]) -> list[list[float]]:
            assert len(images) == 1
            image = images[0]
            calls.append(("image", image.mode, image.size))
            return [[1.0, 0.0]]

    pipeline = _Pipeline()
    pipeline.raw_refiner = SimpleNamespace(encoder=_RefinerEncoder())
    repository = SimpleNamespace(has_refinable_frames=lambda _batch_id: True)
    service = WarmupService(
        pipeline=pipeline,
        readiness_service=_Readiness({}, ()),
        batch_id="batch-1",
        profile="laptop",
        artifact_repository=repository,
        resource_snapshot=lambda: {"process_rss_bytes": 1},
    )

    service.start()
    snapshot = service.wait(5)

    assert snapshot["state"] == "ready"
    assert snapshot["channels"] == [
        {
            "name": "raw_refiner",
            "state": "ready",
            "reason_code": "warmup_completed",
            "duration_ms": snapshot["channels"][0]["duration_ms"],
        }
    ]
    assert calls == [
        ("text", service.warm_query),
        ("image", "RGB", (224, 224)),
    ]


def test_warmup_loads_only_selected_provisioned_trake_trajectory() -> None:
    class _TrakeRouter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        @staticmethod
        def channel_for_batch(_batch_id: str) -> str:
            return "organizer_clip"

        def warm(self, batch_id: str) -> dict[str, str]:
            self.calls.append(batch_id)
            return {"state": "ready"}

    metadata = SimpleNamespace(name="metadata", store=_Store())
    router = _TrakeRouter()
    service = WarmupService(
        pipeline=_Pipeline(lexical=(metadata,)),
        readiness_service=_Readiness(
            {
                "metadata": "ready",
                "organizer_clip": "ready",
                "visual": "ready",
            },
            ("metadata",),
        ),
        batch_id="batch-1",
        profile="laptop",
        trake_pipeline=router,
        resource_snapshot=lambda: {"process_rss_bytes": 1},
    )

    service.start()
    snapshot = service.wait(5)

    assert snapshot["state"] == "ready"
    assert router.calls == ["batch-1"]
    assert [item["name"] for item in snapshot["channels"]] == [
        "trake_organizer_clip_trajectory",
        "metadata",
    ]


def test_warmup_skips_unprovisioned_trake_trajectory() -> None:
    class _TrakeRouter:
        calls = 0

        @staticmethod
        def channel_for_batch(_batch_id: str) -> str:
            return "visual"

        @classmethod
        def warm(cls, _batch_id: str) -> None:
            cls.calls += 1

    metadata = SimpleNamespace(name="metadata", store=_Store())
    router = _TrakeRouter()
    service = WarmupService(
        pipeline=_Pipeline(lexical=(metadata,)),
        readiness_service=_Readiness(
            {"metadata": "ready", "visual": "not_provisioned"},
            ("metadata",),
        ),
        batch_id="batch-1",
        profile="laptop",
        trake_pipeline=router,
        resource_snapshot=lambda: {"process_rss_bytes": 1},
    )

    service.start()
    snapshot = service.wait(5)

    assert snapshot["state"] == "degraded"
    assert router.calls == 0
    assert all(not item["name"].startswith("trake_") for item in snapshot["channels"])


def test_kis_unlocks_before_trajectory_warmup_finishes() -> None:
    trake_started = threading.Event()
    release_trake = threading.Event()

    class _TrakeRouter:
        @staticmethod
        def channel_for_batch(_batch_id: str) -> str:
            return "organizer_clip"

        @staticmethod
        def warm(_batch_id: str) -> dict[str, object]:
            trake_started.set()
            assert release_trake.wait(timeout=5)
            return {
                "state": "ready",
                "reason_code": "snapshot_hit",
                "source": "disk",
                "frame_count": 2,
            }

    metadata = SimpleNamespace(name="metadata", store=_Store())
    service = WarmupService(
        pipeline=_Pipeline(lexical=(metadata,)),
        readiness_service=_Readiness(
            {"metadata": "ready", "organizer_clip": "ready"},
            ("metadata",),
        ),
        batch_id="batch-1",
        profile="laptop",
        trake_pipeline=_TrakeRouter(),
        resource_snapshot=lambda: {"process_rss_bytes": 1},
    )

    service.start()
    assert trake_started.wait(timeout=5)
    partial = service.wait(0.1, workflow="kis")

    assert partial["search_usable"] is True
    assert partial["fully_ready"] is False
    assert partial["state"] == "degraded"
    assert partial["workflows"]["kis"]["state"] == "ready"
    assert partial["workflows"]["trake"]["state"] == "warming"

    release_trake.set()
    complete = service.wait(5)
    assert complete["workflows"]["trake"]["usable"] is True
    assert complete["channels"][0]["source"] == "disk"
