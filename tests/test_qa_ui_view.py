"""Offline view-model tests for the Streamlit Visual Q&A tab."""

from __future__ import annotations

import logging

from app.streamlit_app import (
    _cached_or_live_readiness,
    _qa_channel_status_rows,
    _qa_exportable_pins,
    _qa_failure_metadata,
    _qa_operator_edit_rows,
    _qa_outcomes,
    _qa_prediction_rows,
    _qa_summary,
    _qa_trace_rows,
    _qa_window_frames,
    _readiness_rows,
    _validated_batch_filters,
    _vi_status,
    _WindowsProactorConnectionResetFilter,
)


def test_completed_warmup_report_prevents_synchronous_readiness_rescan() -> None:
    audited_report = object()

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("completed warm-up must not trigger another exact audit")

    context = type(
        "Context",
        (),
        {
            "warmup": type(
                "Warmup", (), {"readiness_report": staticmethod(lambda: audited_report)}
            )(),
            "readiness_service": type(
                "Readiness", (), {"check": staticmethod(unexpected_probe)}
            )(),
        },
    )()

    assert _cached_or_live_readiness(context, "aic25-b1-reference") is audited_report


def _log_record(message: str, exception: BaseException) -> logging.LogRecord:
    return logging.LogRecord(
        name="asyncio",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=(type(exception), exception, None),
    )


def test_windows_asyncio_filter_suppresses_only_exact_proactor_reset() -> None:
    log_filter = _WindowsProactorConnectionResetFilter()
    reset = ConnectionResetError(10054, "fixture reset")

    assert not log_filter.filter(
        _log_record(
            "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
            reset,
        )
    )
    assert log_filter.filter(_log_record("another asyncio callback", reset))
    assert log_filter.filter(
        _log_record(
            "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
            RuntimeError("different fixture"),
        )
    )


def test_vietnamese_status_labels_cover_operator_facing_provider_failures() -> None:
    assert _vi_status("ready") == "sẵn sàng"
    assert _vi_status("rate_limited") == "đã vượt hạn mức nhà cung cấp"
    assert _vi_status("future_status") == "future status"


def _unanswerable_outcome() -> dict:
    return {
        "window_id": "b1:L21_V001:10",
        "batch_id": "b1",
        "video_id": "L21_V001",
        "source_frame_idx": 10,
        "timestamp_ms": 1_000,
        "frames": [
            {
                "position": position,
                "batch_id": "b1",
                "video_id": "L21_V001",
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": source_frame_idx * 100,
            }
            for position, source_frame_idx in enumerate((9, 10, 11), 1)
        ],
        "status": "unanswerable",
        "answerable": False,
        "evidence_frame_indices": [],
        "provider": "gemini",
        "model_name": "gemini-3.6-flash",
        "prompt_version": "qa-v1",
        "provider_latency_ms": 25.0,
        "served_latency_ms": 2.0,
        "attempts": 1,
        "cache_status": "hit",
        "cache_layers": {"reasoning": "hit"},
    }


def test_unanswerable_outcome_remains_visible_without_predictions() -> None:
    outcome = _unanswerable_outcome()
    snapshot = {
        "status": "completed_no_answer",
        "window_outcomes": [outcome],
        "state": {"predictions": []},
    }

    outcomes = _qa_outcomes(snapshot)
    summary = _qa_summary(snapshot)

    assert outcomes == [outcome]
    assert summary == {
        "answerable": 0,
        "unanswerable": 1,
        "failed": 0,
        "pending": 0,
        "reasoned": 1,
        "status": "completed_no_answer",
    }
    assert _qa_window_frames(outcomes[0], []) == outcome["frames"]


def test_execution_log_contains_metadata_but_no_answer_or_reasoning_text() -> None:
    outcome = {
        **_unanswerable_outcome(),
        "raw_answer": "provider-visible answer",
        "normalized_answer": "normalized answer",
        "hidden_reasoning": "must never render",
    }

    rows = _qa_trace_rows([outcome])

    assert rows == [
        {
            "window_id": "b1:L21_V001:10",
            "outcome": "unanswerable",
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "cache": "hit",
            "provider_latency_ms": 25.0,
            "served_latency_ms": 2.0,
            "attempts": 1,
            "cache_layers": "reasoning=hit",
        }
    ]
    assert "provider-visible answer" not in str(rows)
    assert "normalized answer" not in str(rows)
    assert "must never render" not in str(rows)
    assert "raw_answer" not in rows[0]
    assert "normalized_answer" not in rows[0]
    assert "reasoning" not in rows[0]


def test_execution_log_allows_only_sanitized_provider_failure_metadata() -> None:
    failed = {
        **_unanswerable_outcome(),
        "status": "provider_failed",
        "failure_code": "provider_request_failed",
        "failure_stage": "provider",
        "provider_error_category": "network_unreachable",
        "provider_status_code": 503,
    }
    unsafe = {
        **failed,
        "window_id": "b1:L21_V001:11",
        "provider_error_category": "C:/private/provider-body",
        "provider_status_code": 999,
    }

    rows = _qa_trace_rows([failed, unsafe])

    assert rows[0]["provider_error_category"] == "network_unreachable"
    assert rows[0]["provider_status_code"] == 503
    assert "provider_error_category" not in rows[1]
    assert "provider_status_code" not in rows[1]
    assert "private/provider-body" not in str(rows)


def test_prediction_table_projects_only_safe_operator_columns() -> None:
    prediction = {
        "rank": 1,
        "batch_id": "b1",
        "video_id": "L21_V001",
        "source_frame_idx": 10,
        "timestamp_ms": 1_000,
        "answer": "3",
        "answer_source": "model",
        "confidence_score": 0.8,
        "window_id": "w1",
        "raw_answer": "three",
        "model_name": "gemini-3.6-flash",
        "thumbnail_uri": "zip://protected.zip!/secret/member.jpg",
        "image_uri": "D:/private/frame.jpg",
        "video_uri": "D:/private/video.mp4",
        "evidence": {"provider_headers": {"authorization": "secret"}},
        "future_internal_field": "must not render",
    }

    rows = _qa_prediction_rows([prediction])

    assert rows == [
        {
            "rank": 1,
            "batch_id": "b1",
            "video_id": "L21_V001",
            "source_frame_idx": 10,
            "timestamp_ms": 1_000,
            "answer": "3",
            "answer_source": "model",
            "confidence_score": 0.8,
            "window_id": "w1",
        }
    ]
    rendered = str(rows)
    assert "zip://" not in rendered
    assert "D:/private" not in rendered
    assert "authorization" not in rendered
    assert "future_internal_field" not in rendered


def test_operator_edit_and_channel_tables_drop_extra_persisted_fields() -> None:
    edit_rows = _qa_operator_edit_rows(
        [
            {
                "edit_id": 4,
                "session_id": "internal-session-id",
                "window_id": "w1",
                "previous_normalized_answers": ["three"],
                "original_raw_answer": "three",
                "provider": "gemini",
                "model_name": "gemini-3.6-flash",
                "edited_raw_answer": "3",
                "normalized_answer": "3",
                "edited_at": 10.0,
                "created_at": 11.0,
                "evidence": {"image_uri": "zip://protected.zip!/member.jpg"},
                "provider_headers": {"authorization": "secret"},
            }
        ]
    )
    channel_rows = _qa_channel_status_rows(
        {
            "organizer_clip": {
                "applicability": "required",
                "state": "ready",
                "reason_code": "available",
                "coverage": 1.0,
                "count": 177_321,
                "collection_name": "organizer-clip",
                "index_name": None,
                "expected_dimension": 512,
                "remediation_id": None,
                "raw_error": "D:/private/service.log",
                "provider_headers": {"authorization": "secret"},
            }
        }
    )

    assert set(edit_rows[0]) == {
        "edit_id",
        "window_id",
        "previous_normalized_answers",
        "original_raw_answer",
        "provider",
        "model_name",
        "edited_raw_answer",
        "normalized_answer",
        "edited_at",
        "created_at",
    }
    assert "session_id" not in edit_rows[0]
    assert "zip://" not in str(edit_rows)
    assert "authorization" not in str(edit_rows)
    assert set(channel_rows[0]) == {
        "channel",
        "applicability",
        "state",
        "reason_code",
        "coverage",
        "count",
        "collection_name",
        "index_name",
        "expected_dimension",
        "remediation_id",
    }
    assert "D:/private" not in str(channel_rows)
    assert "authorization" not in str(channel_rows)


def test_legacy_answerable_prediction_synthesizes_one_ui_outcome() -> None:
    snapshot = {
        "state": {
            "predictions": [
                {
                    "window_id": "w1",
                    "batch_id": "b1",
                    "video_id": "L21_V001",
                    "source_frame_idx": 10,
                    "timestamp_ms": 1_000,
                    "answer": "3",
                    "raw_answer": "three",
                    "model_name": "gemini-3.6-flash",
                    "evidence": {
                        "provider": "gemini",
                        "cache_status": "miss",
                        "latency_ms": 30.0,
                        "attempts": 1,
                        "evidence_frame_indices": [1],
                    },
                }
            ],
            "results": {"w1": {"answerable": True}},
            "failures": {},
        }
    }

    outcomes = _qa_outcomes(snapshot)

    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "answered"
    assert outcomes[0]["normalized_answer"] == "3"
    assert _qa_summary(snapshot)["status"] == "completed"


def test_readiness_matrix_keeps_required_and_optional_states_distinct() -> None:
    rows = _readiness_rows(
        {
            "channels": [
                {
                    "name": "organizer_clip",
                    "applicability": "required",
                    "state": "blocked_dependency",
                    "reason_code": "dependency_missing",
                    "raw_error": "must not render",
                },
                {
                    "name": "semantic_text",
                    "applicability": "optional",
                    "state": "not_provisioned",
                    "reason_code": "collection_missing",
                },
            ]
        }
    )

    assert rows[0]["applicability"] == "required"
    assert rows[0]["state"] == "blocked_dependency"
    assert rows[1]["applicability"] == "optional"
    assert rows[1]["state"] == "not_provisioned"
    assert "raw_error" not in rows[0]


def test_legacy_unanswerable_and_failed_outcomes_keep_safe_anchor_frames() -> None:
    targets = [
        {
            "batch_id": "b1",
            "video_id": "L21_V001",
            "source_frame_idx": 10,
            "timestamp_ms": 1_000,
            "shot_id": "reference:10",
        },
        {
            "batch_id": "b1",
            "video_id": "L21_V002",
            "source_frame_idx": 20,
            "timestamp_ms": 2_000,
            "shot_id": "reference:20",
        },
    ]
    snapshot = {
        "targets": targets,
        "window_outcomes": [
            {
                **targets[0],
                "window_id": "b1:L21_V001:10",
                "status": "unanswerable",
                "answerable": False,
            },
            {
                **targets[1],
                "window_id": "b1:L21_V002:20",
                "status": "provider_failed",
                "failure_code": "reasoning_failed",
                "failure_stage": "provider",
            },
        ],
        "state": {"predictions": []},
    }

    outcomes = _qa_outcomes(snapshot)

    assert [len(item["frames"]) for item in outcomes] == [1, 1]
    assert outcomes[0]["frames"][0]["source_frame_idx"] == 10
    assert outcomes[1]["status"] == "internal_failed"
    assert outcomes[1]["failure_code"] == "internal_reasoning_error"
    assert outcomes[1]["failure_stage"] == "internal"
    assert "reasoning_failed" not in str(outcomes)
    assert _qa_window_frames(outcomes[1], []) == outcomes[1]["frames"]


def test_legacy_answered_outcome_recovers_full_prediction_frame_strip() -> None:
    snapshot = {
        "window_outcomes": [
            {
                "window_id": "b1:L21_V001:10",
                "batch_id": "b1",
                "video_id": "L21_V001",
                "source_frame_idx": 10,
                "timestamp_ms": 1_000,
                "status": "answered",
                "answerable": True,
            }
        ],
        "state": {
            "predictions": [
                {
                    "window_id": "b1:L21_V001:10",
                    "batch_id": "b1",
                    "video_id": "L21_V001",
                    "source_frame_idx": 10,
                    "evidence": {
                        "window_frames": [
                            {
                                "position": position,
                                "source_frame_idx": frame,
                                "timestamp_ms": frame * 100,
                            }
                            for position, frame in enumerate((9, 10, 11), 1)
                        ]
                    },
                }
            ]
        },
    }

    outcome = _qa_outcomes(snapshot)[0]

    assert [item["source_frame_idx"] for item in outcome["frames"]] == [9, 10, 11]
    assert {item["batch_id"] for item in outcome["frames"]} == {"b1"}
    assert {item["video_id"] for item in outcome["frames"]} == {"L21_V001"}


def test_unknown_failure_and_missing_provider_metadata_are_never_rendered_raw() -> None:
    status, code, stage = _qa_failure_metadata({"failure_code": "old-secret-detail"})
    rows = _qa_trace_rows(
        [
            {
                "window_id": "w1",
                "status": status,
                "failure_code": code,
                "failure_stage": stage,
            }
        ]
    )

    assert rows[0]["provider"] == "not recorded"
    assert rows[0]["model"] == "not recorded"
    assert rows[0]["failure_code"] == "internal_reasoning_error"
    assert "old-secret-detail" not in str(rows)


def test_batch_filter_helper_prefers_runtime_validator_and_has_safe_fallback() -> None:
    class Context:
        selected_batch_id = "b1"

        def __init__(self) -> None:
            self.calls = []

        def validate_batch_filters(self, values=None):
            self.calls.append(values)
            return tuple(values or (self.selected_batch_id,))

    context = Context()
    assert _validated_batch_filters(context) == ("b1",)
    assert _validated_batch_filters(context, ("b2",)) == ("b2",)
    assert context.calls == [None, ("b2",)]

    fallback = type("Fallback", (), {"selected_batch_id": "b1", "config": {}})()
    assert _validated_batch_filters(fallback) == ("b1",)


def test_only_answered_windows_can_reach_qa_export() -> None:
    pins = [
        {"window_id": "answered", "source_frame_idx": 1},
        {"window_id": "unanswerable", "source_frame_idx": 2},
        {"window_id": "failed", "source_frame_idx": 3},
    ]

    assert _qa_exportable_pins(pins, {"answered"}) == [pins[0]]
    assert _qa_exportable_pins(pins, set()) == []
