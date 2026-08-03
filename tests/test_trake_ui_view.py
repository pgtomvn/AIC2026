"""Offline view-model tests for the Vietnamese Streamlit TRAKE tab."""

from __future__ import annotations

from types import SimpleNamespace

from app.streamlit_app import (
    _invalidate_trake_results,
    _trake_csv_bytes,
    _trake_readiness,
    _trake_response_dict,
    _trake_result_rows,
    _vi_status,
)


def _result() -> dict:
    return {
        "rank": 1,
        "batch_id": "aic25-b1-reference",
        "video_id": "L21_V001",
        "stage_matches": [
            {
                "stage_index": 1,
                "source_frame_idx": 10,
                "timestamp_ms": 1_000,
                "similarity": 0.8,
            },
            {
                "stage_index": 2,
                "source_frame_idx": 20,
                "timestamp_ms": 2_000,
                "similarity": 0.7,
            },
        ],
        "confidence_score": 0.9,
        "image_uri": "D:/private/frame.webp",
        "internal_vector": [1.0, 2.0],
    }


def test_trake_table_uses_vietnamese_safe_columns_only() -> None:
    rows = _trake_result_rows([_result()])

    assert rows == [
        {
            "Hạng": 1,
            "Video": "L21_V001",
            "Khung nguồn theo giai đoạn": "10 → 20",
            "Độ tin cậy": 0.9,
        }
    ]
    assert "private" not in str(rows)
    assert "internal_vector" not in str(rows)
    assert _vi_status("compatible_encoder_unavailable") == "thiếu bộ mã hóa tương thích"


def test_trake_readiness_projects_only_safe_fields() -> None:
    pipeline = SimpleNamespace(
        channel_readiness=lambda _batch: (
            True,
            {
                "organizer_clip": {
                    "state": "ready",
                    "count": 100,
                    "endpoint": "http://secret.internal",
                    "raw_exception": "D:/private/error.log",
                }
            },
        )
    )

    usable, statuses = _trake_readiness(
        SimpleNamespace(trake_pipeline=pipeline), "aic25-b1-reference"
    )

    assert usable
    assert statuses == {"organizer_clip": {"state": "ready", "count": 100}}


def test_trake_readiness_does_not_probe_services_while_warmup_is_gating() -> None:
    def unexpected_probe(_batch):
        raise AssertionError("synchronous readiness probe must remain deferred")

    context = SimpleNamespace(
        warmup=SimpleNamespace(
            snapshot=lambda: {"state": "warming", "search_usable": False}
        ),
        trake_pipeline=SimpleNamespace(channel_readiness=unexpected_probe),
    )

    usable, statuses = _trake_readiness(context, "aic25-b1-reference")

    assert usable is False
    assert statuses == {
        "runtime": {"state": "warming", "reason_code": "runtime_warming"}
    }


def test_trake_response_conversion_uses_public_pydantic_aliases() -> None:
    value = SimpleNamespace(
        model_dump=lambda **kwargs: {
            "results": [{"frame_ids": [10, 20]}],
            "channel_statuses": {},
            "by_alias": kwargs["by_alias"],
        }
    )

    response = _trake_response_dict(value)

    assert response["by_alias"] is True
    assert response["results"][0]["frame_ids"] == [10, 20]


class _Repository:
    @staticmethod
    def frame_index_mappings(_batch_id):
        return {"L21_V001": {1: 10, 2: 20}}

    @staticmethod
    def get_frame(batch_id, video_id, source_frame_idx):
        if (
            batch_id == "aic25-b1-reference"
            and video_id == "L21_V001"
            and source_frame_idx in {10, 20}
        ):
            return SimpleNamespace()
        return None


def test_trake_download_is_validated_and_has_no_header() -> None:
    context = SimpleNamespace(
        artifact_repository=_Repository(),
        ledger=SimpleNamespace(is_batch_active=lambda _batch: True),
    )

    filename, data = _trake_csv_bytes(
        context,
        query_id="query vi/1",
        batch_id="aic25-b1-reference",
        stage_count=2,
        results=[_result()],
    )

    assert filename == "query-vi-1-trake-submission.csv"
    assert data.decode("utf-8") == "L21_V001,10,20\n"


def test_trake_query_change_invalidates_old_results_and_export_identity() -> None:
    state = {
        "trake_response": {"results": [_result()]},
        "trake_results": [_result()],
        "trake_query_id": "old-query",
        "trake_result_stage_count": 2,
        "trake_selected_candidate": "old-candidate",
    }

    _invalidate_trake_results(state)

    assert state == {
        "trake_response": None,
        "trake_results": [],
        "trake_query_id": None,
        "trake_result_stage_count": None,
        "trake_export_cache": None,
    }
