"""Interactive Textual KIS, Visual Q&A, and TRAKE competition console."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.components.cyberpunk_theme import (  # noqa: E402
    hud_container_key,
    inject_cyberpunk_theme,
    render_hud_header,
    render_hud_readout,
    render_hud_status_panel,
    render_hud_timeline_strip,
)
from app.components.export_panel import render_export_panel  # noqa: E402
from app.components.query_input import (  # noqa: E402
    render_query_input,
    render_text_encoding_diagnostics,
)
from app.components.result_gallery import (  # noqa: E402
    ResultIdentity,
    render_result_gallery,
    result_identity,
)
from app.http_backend import BackendError, build_http_runtime  # noqa: E402
from src import PROJECT_ROOT  # noqa: E402
from src.contracts import (  # noqa: E402
    CompetitionSpec,
    QACandidateRequest,
    QACompetitionSpec,
    QAQueryRequest,
    QASingleFrameAnswerRequest,
    QueryRequest,
    TRAKEQueryRequest,
    TRAKEStage,
)
from src.qa_submission import serialize_qa_submission  # noqa: E402
from src.runtime import RuntimeContext  # noqa: E402
from src.submission import FrameIndexEvidence  # noqa: E402
from src.trake_submission import (  # noqa: E402
    TRAKEFrameIndexEvidence,
    TRAKESubmissionValidationError,
    serialize_trake_submission,
    trake_submission_filename,
)
from src.utils.io_helpers import atomic_write_bytes  # noqa: E402
from src.utils.logger import (  # noqa: E402
    configure_app_error_logging,
    record_application_error,
)

_QA_FAILURE_CODES = frozenset(
    {
        "media_unavailable",
        "media_window_too_large",
        "provider_unavailable",
        "provider_request_failed",
        "provider_response_invalid",
        "internal_reasoning_error",
    }
)
_QA_PROVIDER_ERROR_CATEGORIES = frozenset(
    {
        "auth_or_permission",
        "client_initialization",
        "client_unavailable",
        "invalid_request",
        "model_unavailable",
        "model_missing",
        "network_unreachable",
        "not_configured",
        "ollama_context_exceeded",
        "ollama_timeout",
        "ollama_response_invalid",
        "provider_chain_exhausted",
        "provider_request_failed",
        "rate_limited",
        "request_rejected",
        "response_invalid",
        "resource_exhausted",
        "server_unavailable",
    }
)

_VI_STATUS_LABELS = {
    "ready": "sẵn sàng",
    "degraded": "hạn chế",
    "blocked": "bị chặn",
    "configured_unverified": "đã cấu hình, chưa xác minh",
    "not_provisioned": "chưa được cung cấp",
    "disabled_by_profile": "không bật trong cấu hình máy này",
    "blocked_dependency": "thiếu phụ thuộc",
    "blocked_asset": "thiếu tài nguyên mô hình",
    "incompatible_index": "chỉ mục không tương thích",
    "service_unavailable": "dịch vụ không khả dụng",
    "no_compatible_points": "không có vector tương thích",
    "no_compatible_documents": "không có văn bản tương thích",
    "query_completed": "truy vấn hoàn tất",
    "channel_query_failed": "truy vấn kênh thất bại",
    "answered": "đã trả lời",
    "unanswerable": "không đủ bằng chứng",
    "pending": "đang chờ",
    "failed": "thất bại",
    "partial": "một phần",
    "completed": "hoàn tất",
    "completed_no_answer": "hoàn tất, không có câu trả lời",
    "incomplete": "chưa hoàn tất",
    "media_failed": "lỗi dữ liệu hình ảnh",
    "provider_failed": "lỗi nhà cung cấp",
    "ollama_context_exceeded": "ngữ cảnh Ollama quá dài",
    "ollama_timeout": "Ollama quá thời gian chờ",
    "ollama_response_invalid": "Ollama trả về kết quả không đúng định dạng",
    "resource_exhausted": "không đủ tài nguyên cho Ollama",
    "internal_failed": "lỗi nội bộ",
    "auth_or_permission": "khóa hoặc quyền truy cập không hợp lệ",
    "network_unreachable": "không kết nối được mạng",
    "rate_limited": "đã vượt hạn mức nhà cung cấp",
    "model_unavailable": "mô hình không khả dụng",
    "server_unavailable": "máy chủ nhà cung cấp không khả dụng",
    "request_rejected": "yêu cầu bị nhà cung cấp từ chối",
    "response_invalid": "phản hồi của nhà cung cấp không hợp lệ",
    "trake_pipeline_unavailable": "bộ máy TRAKE chưa được cấu hình",
    "readiness_probe_failed": "không kiểm tra được trạng thái sẵn sàng",
    "compatible_encoder_unavailable": "thiếu bộ mã hóa tương thích",
    "collection_missing": "chưa có chỉ mục vector",
    "competition_dependency_missing": "thiếu phụ thuộc runtime thi đấu",
}


def _vi_status(value: Any) -> str:
    """Return a Vietnamese operator label while preserving internal status codes."""

    normalized = str(value or "").strip()
    return _VI_STATUS_LABELS.get(normalized, normalized.replace("_", " "))


def _render_query_processing(value: Any) -> None:
    """Show sanitized bilingual routing diagnostics without touching query state."""

    if not isinstance(value, dict) or not value:
        return
    language = str(value.get("detected_language") or "unknown")
    english = value.get("english_text")
    if value.get("translation_applied") and isinstance(english, str) and english:
        source = str(value.get("translation_source") or "offline")
        latency = value.get("translation_latency_ms")
        suffix = (
            f" · {float(latency):.1f} ms"
            if isinstance(latency, (int, float)) and not isinstance(latency, bool)
            else ""
        )
        st.info(f'🌐 Auto-translated ({source}{suffix}): “{english}”')
    elif language == "vi" and value.get("failure_code"):
        st.warning(
            "Bộ dịch Việt–Anh chưa sẵn sàng; các kênh OpenCLIP tiếng Anh được bỏ qua "
            "an toàn, còn kênh tiếng Việt vẫn tiếp tục hoạt động."
        )


def _active_workflow() -> str:
    value = str(st.session_state.get("active_workflow", "kis")).casefold()
    return value if value in {"kis", "qa", "trake"} else "app"


def _report_ui_error(
    error: BaseException,
    *,
    workflow: str,
    stage: str,
    batch_id: object = None,
    query_or_session_id: object = None,
    started_at: float | None = None,
) -> str:
    """Record server-side details and render only a safe Vietnamese event code."""

    latency_ms = None
    if started_at is not None:
        latency_ms = max(0.0, (time.monotonic() - started_at) * 1_000)
    event_id = record_application_error(
        error,
        workflow=workflow,
        stage=stage,
        batch_id=batch_id,
        query_or_session_id=query_or_session_id,
        latency_ms=latency_ms,
    )
    st.error(
        "Đã xảy ra lỗi khi xử lý yêu cầu. Vui lòng thử lại hoặc tra cứu nhật ký "
        f"bằng mã tương quan `{event_id}`."
    )
    return event_id


class _WindowsProactorConnectionResetFilter(logging.Filter):
    """Drop only the benign WebSocket-reset traceback emitted by asyncio on Windows."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "asyncio" or not record.exc_info:
            return True
        exception = record.exc_info[1]
        error_code = getattr(exception, "winerror", None) or getattr(exception, "errno", None)
        return not (
            isinstance(exception, ConnectionResetError)
            and error_code == 10054
            and record.getMessage().startswith(
                "Exception in callback _ProactorBasePipeTransport._call_connection_lost"
            )
        )


def _install_windows_asyncio_log_filter() -> None:
    """Install the exact Windows reset filter once per Streamlit process."""

    if sys.platform != "win32":
        return
    logger = logging.getLogger("asyncio")
    if any(isinstance(value, _WindowsProactorConnectionResetFilter) for value in logger.filters):
        return
    logger.addFilter(_WindowsProactorConnectionResetFilter())


@st.cache_resource
def _runtime() -> Any:
    return build_http_runtime(os.environ.get("AIC_API_URL", "http://127.0.0.1:8080"))


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _source_frame_image(
    context: RuntimeContext,
    batch_id: str,
    video_id: str,
    source_frame_idx: int,
) -> bytes | None:
    repository = getattr(context, "artifact_repository", None)
    resolver = getattr(context, "qa_image_resolver", None)
    if repository is None or resolver is None:
        return None
    getter = getattr(repository, "get_source_frame", None)
    if not callable(getter):
        getter = getattr(repository, "get_frame", None)
    if not callable(getter):
        return None
    frame = getter(batch_id, video_id, source_frame_idx)
    if frame is None:
        return None
    image = resolver.resolve(frame)
    data = getattr(image, "data", None)
    return data if isinstance(data, bytes) else None


def _kis_image_loader(context: RuntimeContext, item: dict[str, Any]) -> bytes | None:
    batch_id = item.get("batch_id")
    video_id = item.get("video_id")
    source_frame_idx = item.get("source_frame_idx")
    if (
        not isinstance(batch_id, str)
        or not isinstance(video_id, str)
        or isinstance(source_frame_idx, bool)
        or not isinstance(source_frame_idx, int)
    ):
        return None
    return _source_frame_image(context, batch_id, video_id, source_frame_idx)


def _initialize_state() -> None:
    defaults = {
        "active_workflow": "kis",
        "kis_results": [],
        "kis_history": [],
        "kis_pinned": [],
        "kis_last_request": None,
        "kis_pin_widget_keys": [],
        "kis_export_cache": None,
        "kis_query_language": "auto",
        "kis_translation_edit": False,
        "kis_english_override": "",
        "qa_predictions": [],
        "qa_pinned": [],
        "qa_edits": [],
        "qa_search_id": None,
        "qa_request": None,
        "qa_snapshot": None,
        "qa_candidate_windows": [],
        "qa_candidates": [],
        "qa_answers": [],
        "qa_result": None,
        "qa_last_failure": None,
        "qa_export_cache": None,
        "qa_selected_window_id": None,
        "qa_selected_candidate_id": None,
        "qa_batch_filters_seeded": False,
        "qa_engine": "gemini",
        "qa_answer_spec": QACompetitionSpec.conservative_general_text().spec_id,
        "qa_answer_engine": "vlm",
        "qa_reasoning_policy": "gemini_then_local",
        "qa_use_florence": False,
        "qa_florence_task": "ocr",
        "qa_florence_target_label": "person",
        "qa_allow_cloud_fallback": False,
        "qa_external_transfer_consent": False,
        "qa_diagnostic_result": None,
        "qa_query_language": "auto",
        "qa_translation_edit": False,
        "qa_event_description_en": "",
        "trake_results": [],
        "trake_response": None,
        "trake_query_id": None,
        "trake_stage_count": 2,
        "trake_result_stage_count": None,
        "trake_global_description": "",
        "trake_stage_1": "",
        "trake_stage_2": "",
        "trake_export_cache": None,
        "trake_query_language": "auto",
        "trake_translation_edit": False,
        "trake_global_description_en": "",
    }
    for name, value in defaults.items():
        st.session_state.setdefault(name, value)


def _pinned_identity_set(items: list[dict[str, Any]]) -> set[ResultIdentity]:
    result: set[ResultIdentity] = set()
    for item in items:
        try:
            result.add(result_identity(item))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _change_kis_pin(
    identity: ResultIdentity,
    item: dict[str, Any],
    selected: bool,
    widget_key: str,
) -> None:
    """Synchronize one KIS pin and its widget state atomically."""

    pinned = list(st.session_state.get("kis_pinned", ()))
    by_identity: dict[ResultIdentity, dict[str, Any]] = {}
    order: list[ResultIdentity] = []
    for existing in pinned:
        try:
            key = result_identity(existing)
        except (KeyError, TypeError, ValueError):
            continue
        if key not in by_identity:
            order.append(key)
        by_identity[key] = dict(existing)
    if selected:
        if identity not in by_identity:
            order.append(identity)
        by_identity[identity] = dict(item)
    else:
        by_identity.pop(identity, None)
        order = [key for key in order if key != identity]
    st.session_state.kis_pinned = [by_identity[key] for key in order if key in by_identity]
    keys = set(st.session_state.get("kis_pin_widget_keys", ()))
    keys.add(widget_key)
    st.session_state.kis_pin_widget_keys = sorted(keys)
    st.session_state.kis_export_cache = None


def _clear_kis_pins() -> None:
    """Clear the pin source of truth and every associated checkbox key."""

    for key in tuple(st.session_state.get("kis_pin_widget_keys", ())):
        st.session_state.pop(str(key), None)
    st.session_state.kis_pin_widget_keys = []
    st.session_state.kis_pinned = []
    st.session_state.kis_export_cache = None


def _kis_export_snapshot(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy visible pins and assign export-only contiguous ranks."""

    return [dict(item, rank=rank) for rank, item in enumerate(items, start=1)]


def _result_fingerprint(*, workflow: str, batch_id: str, rows: list[dict[str, Any]]) -> str:
    safe_rows = [
        {
            key: row.get(key)
            for key in (
                "rank",
                "batch_id",
                "video_id",
                "source_frame_idx",
                "answer",
                "stage_matches",
            )
            if key in row
        }
        for row in rows
    ]
    encoded = json.dumps(
        {"workflow": workflow, "batch_id": batch_id, "rows": safe_rows},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _submission_identities(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    identities: list[tuple[str, int]] = []
    for row in rows:
        video_id = row.get("video_id", row.get("video_name"))
        if not isinstance(video_id, str) or not video_id:
            continue
        values: list[object] = []
        if "source_frame_idx" in row or "frame_idx" in row:
            values.append(row.get("source_frame_idx", row.get("frame_idx")))
        if row.get("coarse_source_frame_idx") is not None:
            values.append(row.get("coarse_source_frame_idx"))
        matches = row.get("stage_matches")
        if isinstance(matches, (list, tuple)):
            values.extend(
                match.get("source_frame_idx") for match in matches if isinstance(match, dict)
            )
        frame_ids = row.get("frame_ids", row.get("source_frame_indices"))
        if isinstance(frame_ids, (list, tuple)):
            values.extend(frame_ids)
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                identities.append((video_id, value))
    return list(dict.fromkeys(identities))


def _submission_validation_context(
    context: RuntimeContext,
    batch_id: str,
    rows: list[dict[str, Any]],
) -> tuple[
    Any,
    FrameIndexEvidence,
    Any,
]:
    """Prepare bounded membership, ordinal evidence, and timestamp callbacks."""

    repository = getattr(context, "artifact_repository", None)
    if repository is None:
        raise ValueError("canonical source-frame repository is unavailable")
    identities = _submission_identities(rows)
    bulk_lookup = getattr(repository, "existing_source_frames", None)
    if callable(bulk_lookup):
        existing = frozenset(bulk_lookup(batch_id, identities, include_refined=True))
    else:
        getter = getattr(repository, "get_source_frame", None)
        if not callable(getter):
            getter = getattr(repository, "get_frame", None)
        existing = frozenset(
            identity
            for identity in identities
            if callable(getter) and getter(batch_id, identity[0], identity[1]) is not None
        )

    def is_canonical_source_frame(
        candidate_batch: str,
        video_id: str,
        source_frame_idx: int,
    ) -> bool:
        return candidate_batch == batch_id and (video_id, source_frame_idx) in existing

    evidence_lookup = getattr(repository, "frame_index_mappings_for_sources", None)
    if callable(evidence_lookup):
        mappings = evidence_lookup(batch_id, identities)
    else:
        mappings = repository.frame_index_mappings(batch_id)

    def canonical_timestamp_ms(
        candidate_batch: str,
        video_id: str,
        source_frame_idx: int,
    ) -> int | None:
        if candidate_batch != batch_id:
            return None
        getter = getattr(repository, "get_source_frame", None)
        if not callable(getter):
            getter = getattr(repository, "get_frame", None)
        frame = getter(batch_id, video_id, source_frame_idx) if callable(getter) else None
        timestamp = getattr(frame, "timestamp_ms", None)
        return timestamp if isinstance(timestamp, int) and not isinstance(timestamp, bool) else None

    return is_canonical_source_frame, FrameIndexEvidence(mappings), canonical_timestamp_ms


def _selected_batch_id(context: RuntimeContext) -> str | None:
    value = getattr(context, "selected_batch_id", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    config = getattr(context, "config", {})
    batches = config.get("readiness", {}).get("batches", {}) if isinstance(config, dict) else {}
    if isinstance(batches, dict) and len(batches) == 1:
        return str(next(iter(batches)))
    return None


def _validated_batch_filters(
    context: RuntimeContext,
    values: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    validator = getattr(context, "validate_batch_filters", None)
    if callable(validator):
        validated = validator(values or None)
        return tuple(str(item) for item in validated)
    if values:
        return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    selected = _selected_batch_id(context)
    return (selected,) if selected else ()


def _render_launch_identity(context: RuntimeContext) -> None:
    identity = getattr(context, "launch_identity", None)
    identity = dict(identity) if isinstance(identity, dict) else {}
    config = getattr(context, "config", {})
    config = config if isinstance(config, dict) else {}
    selected = str(identity.get("batch_id") or _selected_batch_id(context) or "chưa chọn")
    profile = str(identity.get("profile") or config.get("profile", "không rõ"))
    contract = str(identity.get("contract") or "direct-streamlit")
    launch_id = str(identity.get("launch_id") or "chưa ghi nhận")
    port = identity.get("port")
    lock_hash = identity.get("lock_sha256")
    lock_fingerprint = (
        str(lock_hash)[:12]
        if isinstance(lock_hash, str) and len(lock_hash) == 64
        else "chưa ghi nhận"
    )
    details = (
        f"Thông tin phiên chạy · hợp đồng `{contract}` · cấu hình `{profile}` · "
        f"batch `{selected}` · cổng `{port if port is not None else 'chưa ghi nhận'}` · "
        f"ID `{launch_id}` · khóa `{lock_fingerprint}`"
    )
    if identity.get("canonical") is True:
        st.caption(details)
        return
    recommended_port = port if isinstance(port, int) and 1 <= port <= 65_535 else 8501
    launcher = (
        f".\\scripts\\run_app.ps1 -Profile {profile} -BatchId {selected} -Port {recommended_port}"
    )
    st.warning(
        details + "\n\nPhát hiện phiên Streamlit được chạy trực tiếp hoặc chưa xác minh. "
        f"Hãy khởi động lại bằng `{launcher}` để kiểm tra runtime, batch, cổng và trạng thái sẵn sàng."
    )


def _readiness_batch_id(context: RuntimeContext) -> str | None:
    request = st.session_state.get("qa_request")
    filters = getattr(request, "batch_filters", ()) if request is not None else ()
    if len(filters) == 1:
        return str(filters[0])
    return _selected_batch_id(context)


def _readiness_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Project readiness channels onto safe operator-facing columns."""

    allowed = (
        "name",
        "applicability",
        "state",
        "reason_code",
        "count",
        "coverage",
        "resource_name",
        "expected_dimension",
        "remediation_id",
    )
    channels = report.get("channels", ())
    if not isinstance(channels, (list, tuple)):
        return []
    return [
        {name: item.get(name) for name in allowed} for item in channels if isinstance(item, dict)
    ]


def _cached_or_live_readiness(context: RuntimeContext, batch_id: str) -> Any:
    """Reuse warm-up's exact audit; probe only when no audited report exists."""

    warmup = getattr(context, "warmup", None)
    warmed_report = getattr(warmup, "readiness_report", None)
    report = warmed_report() if callable(warmed_report) else None
    if report is not None:
        return report
    service = getattr(context, "readiness_service", None)
    return service.check(batch_id, mode="app") if service is not None else None


def _render_readiness(context: RuntimeContext) -> None:
    # The exact canonical-image audit may inspect every archive member.  It is
    # deliberately performed by WarmupService off the Streamlit script thread;
    # probing here while that audit is running would freeze the first page and
    # duplicate the same work.  Search controls remain gated by this snapshot.
    warmup_snapshot = _warmup_snapshot(context)
    active_workflow = _active_workflow()
    workflow_snapshot = _warmup_workflow_snapshot(
        warmup_snapshot,
        active_workflow,
    )
    if warmup_snapshot is not None and not bool(workflow_snapshot.get("usable", False)):
        state = str(workflow_snapshot.get("state", warmup_snapshot.get("state", "idle")))
        if state in {"idle", "warming", "evicted"}:
            st.info(
                "Đang kiểm tra đầy đủ chỉ mục và ảnh canonical trong nền; "
                "tìm kiếm sẽ tự mở khóa sau khi hoàn tất."
            )
        else:
            st.error(
                "Kiểm tra readiness nền chưa tạo được đường truy xuất khả dụng. "
                "Xem chi tiết warm-up trong thanh bên."
            )
        return

    service = getattr(context, "readiness_service", None)
    batch_id = _readiness_batch_id(context)
    if service is None or batch_id is None:
        return
    try:
        readiness_report = _cached_or_live_readiness(context, batch_id)
        if readiness_report is None:
            return
        report = readiness_report.to_dict()
        workflow_probe = getattr(service, "workflow_readiness", None)
        if callable(workflow_probe):
            report["trake"] = workflow_probe(readiness_report, "trake").to_dict()
    except Exception:
        st.warning("Không thể đánh giá an toàn trạng thái sẵn sàng của hệ thống.")
        return
    kis = report.get("kis", {})
    qa = report.get("qa", {})
    trake = report.get("trake", {})
    kis_state = str(kis.get("state", "blocked")) if isinstance(kis, dict) else "blocked"
    qa_state = str(qa.get("state", "blocked")) if isinstance(qa, dict) else "blocked"
    trake_state = str(trake.get("state", "blocked")) if isinstance(trake, dict) else "blocked"
    message = (
        f"Trạng thái hệ thống · {batch_id} · KIS {_vi_status(kis_state)} · "
        f"Hỏi đáp hình ảnh {_vi_status(qa_state)} · TRAKE {_vi_status(trake_state)}"
    )
    if "blocked" in {kis_state, qa_state, trake_state}:
        st.error(message)
    elif "degraded" in {kis_state, qa_state, trake_state}:
        st.warning(message)
    else:
        st.success(message)

    rows = _readiness_rows(report)
    required_failures = [
        row
        for row in rows
        if str(row.get("applicability")) == "required"
        and str(row.get("state")) not in {"ready", "configured_unverified"}
    ]
    optional_missing = [
        row
        for row in rows
        if str(row.get("applicability")) == "optional"
        and str(row.get("state")) == "not_provisioned"
    ]
    if required_failures:
        st.error(
            "Các kênh bắt buộc đang lỗi: "
            + ", ".join(str(row.get("name")) for row in required_failures)
        )
    if optional_missing:
        st.caption(
            "Các kênh tùy chọn chưa được cung cấp: "
            + ", ".join(str(row.get("name")) for row in optional_missing)
        )
    with st.expander("Ma trận trạng thái hệ thống"):
        st.dataframe(rows, width="stretch", hide_index=True)


_WARMUP_VI = {
    "idle": "Chưa khởi động",
    "warming": "Đang làm nóng",
    "ready": "Sẵn sàng",
    "degraded": "Sẵn sàng hạn chế",
    "failed": "Không sẵn sàng",
    "evicted": "Cần làm nóng lại",
}


def _warmup_snapshot(context: RuntimeContext) -> dict[str, Any] | None:
    warmup = getattr(context, "warmup", None)
    snapshot = getattr(warmup, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        value = snapshot()
    except Exception:
        return {"state": "failed", "search_usable": False, "channels": []}
    return dict(value) if isinstance(value, dict) else None


def _warmup_search_usable(context: RuntimeContext) -> bool:
    return _warmup_workflow_usable(context, "kis")


def _warmup_workflow_snapshot(
    snapshot: dict[str, Any] | None,
    workflow: str,
) -> dict[str, Any]:
    if snapshot is None:
        return {"state": "ready", "usable": True}
    workflows = snapshot.get("workflows")
    if isinstance(workflows, dict) and isinstance(workflows.get(workflow), dict):
        return dict(workflows[workflow])
    return {
        "state": str(snapshot.get("state", "idle")),
        "usable": bool(snapshot.get("search_usable", False)),
    }


def _warmup_workflow_usable(context: RuntimeContext, workflow: str) -> bool:
    return bool(_warmup_workflow_snapshot(_warmup_snapshot(context), workflow)["usable"])


def _hud_state(value: object) -> str:
    """Map sanitized readiness states to the compact operator HUD vocabulary."""

    state = str(value or "").strip().lower()
    if state in {"ready", "online", "configured", "configured_unverified"}:
        return "READY"
    if state in {"idle", "warming", "evicted", "loading", "not_provisioned", "skipped"}:
        return "STANDBY"
    if state in {"degraded", "partial", "incompatible_index"}:
        return "DEGRADED"
    if state in {
        "failed",
        "offline",
        "blocked",
        "blocked_asset",
        "service_unavailable",
        "unavailable",
    }:
        return "OFFLINE"
    return "UNKNOWN"


def _aggregate_hud_channels(
    channels: list[dict[str, Any]],
    names: set[str],
) -> str:
    selected = [
        _hud_state(item.get("state"))
        for item in channels
        if str(item.get("name", "")) in names
    ]
    if not selected:
        return "UNKNOWN"
    if "READY" in selected:
        return "READY"
    if "DEGRADED" in selected:
        return "DEGRADED"
    if "STANDBY" in selected:
        return "STANDBY"
    if all(value == "OFFLINE" for value in selected):
        return "OFFLINE"
    return "UNKNOWN"


def _hud_service_statuses(
    context: RuntimeContext,
    snapshot: dict[str, Any],
) -> tuple[tuple[str, str, str | None], ...]:
    """Project existing snapshots into service badges without issuing new probes."""

    raw_channels = snapshot.get("channels", ())
    channels = [dict(item) for item in raw_channels if isinstance(item, dict)]
    capabilities = getattr(context, "capabilities", {})
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    qa_capabilities = capabilities.get("qa", {})
    qa_capabilities = qa_capabilities if isinstance(qa_capabilities, dict) else {}
    local = qa_capabilities.get("local", {})
    local = local if isinstance(local, dict) else {}
    cloud = qa_capabilities.get("cloud", {})
    cloud = cloud if isinstance(cloud, dict) else {}
    ollama_status = local.get("status", {})
    ollama_status = ollama_status if isinstance(ollama_status, dict) else {}
    gemini_status = cloud.get("status", {})
    gemini_status = gemini_status if isinstance(gemini_status, dict) else {}
    translation = capabilities.get("query_translation", {})
    translation = translation if isinstance(translation, dict) else {}
    model = str(local.get("model") or "Qwen2.5-VL")
    fastapi_state = "ONLINE" if bool(getattr(context, "is_remote", False)) else "STANDBY"
    fastapi_detail = (
        "runtime API đang phục vụ giao diện"
        if fastapi_state == "ONLINE"
        else "runtime kiểm thử được inject"
    )
    return (
        ("FastAPI", fastapi_state, fastapi_detail),
        (
            "Qdrant",
            _aggregate_hud_channels(
                channels,
                {"visual", "organizer_clip", "accelerated_visual", "semantic_text"},
            ),
            "vector retrieval",
        ),
        (
            "Meilisearch",
            _aggregate_hud_channels(channels, {"ocr", "asr", "metadata"}),
            "lexical retrieval",
        ),
        ("Ollama", _hud_state(ollama_status.get("state")), model),
        (
            "Gemini Flash",
            _hud_state(gemini_status.get("state")),
            str(cloud.get("model") or "gemini-3.6-flash"),
        ),
        (
            "VI→EN NMT",
            _hud_state(translation.get("state")),
            str(translation.get("model") or "offline translator"),
        ),
    )


@st.fragment(run_every="1s")
def _render_warmup_sidebar(context: RuntimeContext) -> None:
    snapshot = _warmup_snapshot(context)
    if snapshot is None:
        state = "ready"
        snapshot = {"state": state, "search_usable": True, "channels": []}
    else:
        state = str(snapshot.get("state", "idle"))
    usability = tuple(
        (
            workflow,
            bool(_warmup_workflow_snapshot(snapshot, workflow).get("usable", False)),
            str(_warmup_workflow_snapshot(snapshot, workflow).get("state", "idle")),
        )
        for workflow in ("kis", "qa", "trake")
    )
    previous_usability = st.session_state.get("_aic_warmup_workflow_usability")
    st.session_state["_aic_warmup_workflow_usability"] = usability
    if previous_usability is not None and previous_usability != usability:
        st.rerun(scope="app")
    label = _WARMUP_VI.get(state, "Không xác định")
    with st.sidebar:
        st.markdown(
            render_hud_status_panel(
                "SYSTEM LINK STATUS",
                _hud_service_statuses(context, snapshot),
            ),
            unsafe_allow_html=True,
        )
        st.subheader("Bộ máy truy xuất")
        if state == "ready":
            st.success(label)
        elif state == "degraded":
            st.warning(label)
        elif state in {"idle", "warming", "evicted"}:
            st.info(label)
        else:
            st.error(label)
        duration = snapshot.get("duration_ms")
        if isinstance(duration, (int, float)):
            st.caption(f"Thời gian làm nóng: {float(duration) / 1000:.1f} giây")
        channels = snapshot.get("channels", ())
        if isinstance(channels, list) and channels:
            with st.expander("Chi tiết kênh warm-up"):
                st.dataframe(channels, width="stretch", hide_index=True)


def _render_kis(context: RuntimeContext) -> None:
    st.subheader("Tìm kiếm sự kiện đã biết bằng văn bản")
    settings_col, results_col, refinement_col = st.columns((1, 1, 1))
    with settings_col:
        mode = st.radio(
            "Chế độ KIS",
            ("interactive", "automatic"),
            format_func=lambda value: "Tương tác" if value == "interactive" else "Tự động",
            horizontal=True,
            key="kis_mode",
        )
    with results_col:
        top_k = st.slider("Số kết quả KIS", 1, 100, 20, key="kis_top_k")
    with refinement_col:
        refine_raw = st.checkbox("Tinh chỉnh cục bộ từ video gốc", value=True, key="kis_refine_raw")

    runtime_ready = _warmup_search_usable(context)
    values = render_query_input(disabled=not runtime_ready)
    language_col, override_col = st.columns((1, 2))
    with language_col:
        query_language = st.selectbox(
            "Ngôn ngữ truy vấn",
            ("auto", "vi", "en"),
            format_func=lambda value: {
                "auto": "Tự động",
                "vi": "Tiếng Việt",
                "en": "English",
            }[value],
            key="kis_query_language",
        )
    with override_col:
        edit_translation = st.checkbox(
            "Chỉnh bản dịch tiếng Anh",
            key="kis_translation_edit",
        )
        english_override = (
            st.text_input(
                "Bản dịch tiếng Anh cho dense OpenCLIP",
                key="kis_english_override",
                placeholder="Để trống để dùng bộ dịch offline",
            )
            if edit_translation
            else ""
        )
    if not runtime_ready:
        st.info("Bộ máy truy xuất đang được làm nóng; bạn vẫn có thể soạn truy vấn.")
    if values["submitted"]:
        if not values["text"].strip():
            st.warning("Hãy nhập mô tả trước khi tìm kiếm.")
        else:
            search_started = time.monotonic()
            request: QueryRequest | None = None
            try:
                batch_filters = _validated_batch_filters(context)
                request = QueryRequest(
                    query_id=str(uuid4()),
                    original_text=values["text"],
                    mode=mode,
                    batch_filters=batch_filters,
                    limit=top_k,
                    before_clause=values["before_clause"],
                    current_clause=values["current_clause"],
                    after_clause=values["after_clause"],
                    quoted_text=values["quoted_text"],
                    named_entities=values["named_entities"],
                    objects=values["objects"],
                    actions=values["actions"],
                    attributes=values["attributes"],
                    query_language=query_language,
                    english_text_override=english_override or None,
                )
                with st.spinner("Đang tìm trên các kênh vector và từ khóa..."):
                    st.session_state.kis_results = context.pipeline.search(
                        request, limit=top_k, refine_raw=refine_raw
                    )
                    st.session_state.kis_last_request = request
                    st.session_state.kis_history.append(values["text"])
            except Exception as error:
                _report_ui_error(
                    error,
                    workflow="kis",
                    stage="search",
                    batch_id=(
                        batch_filters[0] if "batch_filters" in locals() and batch_filters else None
                    ),
                    query_or_session_id=(request.query_id if request is not None else None),
                    started_at=search_started,
                )

    channel_statuses = getattr(context.pipeline, "last_channel_statuses", {})
    if isinstance(channel_statuses, dict) and channel_statuses:
        rows = [
            {
                "Kênh": str(name),
                "Trạng thái": _vi_status(status.get("state")),
                "Lý do": _vi_status(status.get("reason_code")),
            }
            for name, status in sorted(channel_statuses.items())
            if isinstance(status, dict)
        ]
        with st.expander("Trạng thái các kênh KIS"):
            st.dataframe(rows, width="stretch", hide_index=True)
        render_text_encoding_diagnostics(channel_statuses)
    _render_query_processing(
        getattr(context.pipeline, "last_query_processing", {})
    )

    if st.session_state.kis_last_request is None:
        st.caption("Nhập mô tả và chọn Tìm kiếm để truy xuất các khung hình đã lập chỉ mục.")
    else:
        render_result_gallery(
            st.session_state.kis_results,
            image_loader=lambda item: _kis_image_loader(context, item),
            pinned_identities=_pinned_identity_set(st.session_state.kis_pinned),
            on_pin_change=_change_kis_pin,
        )

    if st.session_state.kis_results and st.session_state.kis_last_request is not None:
        if context.artifact_repository is not None:
            with st.expander("Xem các khung hình lân cận"):
                neighbor_choices = {
                    f"#{item['rank']} {item['video_id']}:{item['source_frame_idx']}": item
                    for item in st.session_state.kis_results[:20]
                    if item.get("batch_id")
                }
                if neighbor_choices:
                    label = st.selectbox("Ứng viên", list(neighbor_choices), key="kis_neighbor")
                    candidate = neighbor_choices[label]
                    neighbors = context.artifact_repository.neighbors(
                        candidate["batch_id"],
                        candidate["video_id"],
                        int(candidate.get("timestamp_ms", 0)),
                        radius_ms=4_000,
                        limit=25,
                    )
                    st.dataframe(
                        [
                            {
                                "source_frame_idx": frame.source_frame_idx,
                                "timestamp_ms": frame.timestamp_ms,
                                "shot_id": frame.shot_id,
                            }
                            for frame in neighbors
                        ],
                        width="stretch",
                        hide_index=True,
                    )

        with st.expander("Phản hồi mức độ liên quan"):
            choices = {
                f"#{item['rank']} {item['video_id']}:{item['source_frame_idx']}": (
                    item["video_id"],
                    item["source_frame_idx"],
                )
                for item in st.session_state.kis_results[:20]
            }
            positive = st.multiselect("Liên quan", list(choices), key="kis_positive_hits")
            negative = st.multiselect("Không liên quan", list(choices), key="kis_negative_hits")
            if set(positive) & set(negative):
                st.warning("Một khung hình không thể vừa liên quan vừa không liên quan.")
            elif st.button(
                "Tinh chỉnh bằng phản hồi Rocchio",
                key="kis_feedback",
                disabled=not runtime_ready,
            ):
                try:
                    st.session_state.kis_results = context.pipeline.refine(
                        st.session_state.kis_last_request,
                        positive_ids=tuple(choices[label] for label in positive),
                        negative_ids=tuple(choices[label] for label in negative),
                        limit=top_k,
                    )
                    st.rerun()
                except Exception as error:
                    _report_ui_error(
                        error,
                        workflow="kis",
                        stage="feedback",
                        batch_id=(
                            st.session_state.kis_last_request.batch_filters[0]
                            if st.session_state.kis_last_request.batch_filters
                            else None
                        ),
                        query_or_session_id=st.session_state.kis_last_request.query_id,
                    )

    st.divider()
    st.subheader("Các ứng viên KIS đã ghim")
    if not st.session_state.kis_pinned:
        st.caption("Ghim kết quả để tạo danh sách cuối cùng đã được kiểm tra.")
        return
    st.dataframe(
        [
            {
                "rank": index,
                "video_id": item["video_id"],
                "source_frame_idx": item["source_frame_idx"],
            }
            for index, item in enumerate(st.session_state.kis_pinned, start=1)
        ],
        width="stretch",
        hide_index=True,
    )
    batch_ids = {
        str(item["batch_id"]) for item in st.session_state.kis_pinned if item.get("batch_id")
    }
    if len(batch_ids) != 1 or context.artifact_repository is None:
        st.error(
            "Trước khi xuất, các ứng viên đã ghim phải thuộc cùng một batch đang hoạt động "
            "và đã lập chỉ mục."
        )
    else:
        batch_id = next(iter(batch_ids))
        snapshot = _kis_export_snapshot(st.session_state.kis_pinned)
        canonical_check, evidence, _ = _submission_validation_context(context, batch_id, snapshot)
        render_export_panel(
            snapshot,
            output_dir=context.config["competition"]["output_dir"],
            spec=CompetitionSpec.from_config(context.config.get("competition", {})),
            batch_id=batch_id,
            is_batch_active=context.ledger.is_batch_active,
            is_canonical_source_frame=canonical_check,
            frame_index_evidence=evidence,
            result_fingerprint=_result_fingerprint(
                workflow="kis",
                batch_id=batch_id,
                rows=snapshot,
            ),
        )
    st.button(
        "Xóa các ứng viên KIS đã ghim",
        key="kis_clear_pinned",
        on_click=_clear_kis_pins,
    )


def _qa_failure_metadata(value: dict[str, Any]) -> tuple[str, str, str]:
    """Return a stable failure status/code/stage without exposing legacy text."""

    code = value.get("failure_code")
    if code not in _QA_FAILURE_CODES:
        return "internal_failed", "internal_reasoning_error", "internal"
    if code.startswith("media_"):
        return "media_failed", str(code), "media"
    if code.startswith("provider_"):
        return "provider_failed", str(code), "provider"
    return "internal_failed", "internal_reasoning_error", "internal"


def _qa_frame_refs(
    values: Any,
    *,
    batch_id: Any = None,
    video_id: Any = None,
) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    frames: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for fallback_position, raw in enumerate(values, 1):
        if not isinstance(raw, dict):
            continue
        try:
            source_frame_idx = int(raw["source_frame_idx"])
            timestamp_ms = int(raw.get("timestamp_ms", 0))
            position = int(raw.get("position", fallback_position))
        except (KeyError, TypeError, ValueError):
            continue
        frame_batch = str(raw.get("batch_id") or batch_id or "").strip()
        frame_video = str(raw.get("video_id") or video_id or "").strip()
        identity = (frame_batch, frame_video, source_frame_idx)
        if not frame_batch or not frame_video or source_frame_idx < 0 or identity in seen:
            continue
        frames.append(
            {
                "position": max(1, position),
                "batch_id": frame_batch,
                "video_id": frame_video,
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": max(0, timestamp_ms),
            }
        )
        seen.add(identity)
    return frames


def _qa_anchor_frame(value: dict[str, Any]) -> list[dict[str, Any]]:
    return _qa_frame_refs(
        [
            {
                "position": 1,
                "source_frame_idx": value.get("source_frame_idx"),
                "timestamp_ms": value.get("timestamp_ms", 0),
            }
        ],
        batch_id=value.get("batch_id"),
        video_id=value.get("video_id"),
    )


def _qa_outcomes(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return every safe attempted-window outcome, including legacy sessions."""

    if snapshot is None:
        return []
    state = snapshot.get("state", {})
    state = state if isinstance(state, dict) else {}
    predictions = [
        dict(item)
        for item in state.get("predictions", snapshot.get("predictions", ()))
        if isinstance(item, dict)
    ]
    predictions_by_window: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        predictions_by_window.setdefault(str(prediction.get("window_id", "")), []).append(
            prediction
        )
    targets = [dict(item) for item in snapshot.get("targets", ()) if isinstance(item, dict)]
    targets_by_window = {
        str(
            item.get("window_id")
            or f"{item.get('batch_id')}:{item.get('video_id')}:{item.get('source_frame_idx')}"
        ): item
        for item in targets
    }
    raw = snapshot.get("window_outcomes", state.get("window_outcomes", ()))
    if isinstance(raw, dict):
        raw = list(raw.values())
    if isinstance(raw, (list, tuple)) and raw:
        normalized: list[dict[str, Any]] = []
        for raw_item in raw:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            window_id = str(item.get("window_id", ""))
            target = targets_by_window.get(window_id, {})
            for name in (
                "batch_id",
                "video_id",
                "source_frame_idx",
                "timestamp_ms",
                "shot_id",
            ):
                if item.get(name) is None and target.get(name) is not None:
                    item[name] = target[name]
            if str(item.get("status", "")).endswith("_failed"):
                status, code, stage = _qa_failure_metadata(item)
                item.update(status=status, failure_code=code, failure_stage=stage)
            frames = _qa_frame_refs(
                item.get("frames", ()),
                batch_id=item.get("batch_id"),
                video_id=item.get("video_id"),
            )
            if not frames:
                for prediction in predictions_by_window.get(window_id, ()):
                    evidence = prediction.get("evidence", {})
                    evidence = evidence if isinstance(evidence, dict) else {}
                    frames = _qa_frame_refs(
                        evidence.get("window_frames", ()),
                        batch_id=item.get("batch_id"),
                        video_id=item.get("video_id"),
                    )
                    if frames:
                        break
            item["frames"] = frames or _qa_anchor_frame(item)
            normalized.append(item)
        return normalized

    outcomes: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if prediction.get("window_id"):
            window_id = str(prediction["window_id"])
            evidence = prediction.get("evidence", {})
            evidence = evidence if isinstance(evidence, dict) else {}
            frames = evidence.get("window_frames", ())
            if not isinstance(frames, (list, tuple)):
                frames = ()
            outcomes.setdefault(
                window_id,
                {
                    "window_id": window_id,
                    "status": "answered",
                    "batch_id": prediction.get("batch_id"),
                    "video_id": prediction.get("video_id"),
                    "source_frame_idx": prediction.get("source_frame_idx"),
                    "timestamp_ms": prediction.get("timestamp_ms", 0),
                    "frames": _qa_frame_refs(
                        frames,
                        batch_id=prediction.get("batch_id"),
                        video_id=prediction.get("video_id"),
                    ),
                    "raw_answer": prediction.get("raw_answer"),
                    "normalized_answer": prediction.get("answer"),
                    "answerable": True,
                    "evidence_frame_indices": evidence.get("evidence_frame_indices", ()),
                    "provider": evidence.get("provider"),
                    "model_name": prediction.get("model_name"),
                    "provider_latency_ms": evidence.get("latency_ms"),
                    "attempts": evidence.get("attempts"),
                    "cache_status": evidence.get("cache_status"),
                },
            )
    results = state.get("results", {})
    if isinstance(results, dict):
        for window_id, result in results.items():
            if str(window_id) in outcomes or not isinstance(result, dict):
                continue
            reasoning = result.get("reasoning", result)
            reasoning = reasoning if isinstance(reasoning, dict) else {}
            answerable = bool(reasoning.get("answerable", False))
            target = targets_by_window.get(str(window_id), {})
            outcomes[str(window_id)] = {
                "window_id": str(window_id),
                "status": "answered" if answerable else "unanswerable",
                **target,
                "frames": _qa_anchor_frame(target),
                "raw_answer": reasoning.get("raw_answer"),
                "normalized_answer": reasoning.get("normalized_answer", reasoning.get("answer")),
                "answerable": answerable,
                "evidence_frame_indices": reasoning.get("evidence_frame_indices", ()),
                "provider": reasoning.get("provider"),
                "model_name": reasoning.get("model_name"),
                "provider_latency_ms": reasoning.get("latency_ms"),
                "attempts": reasoning.get("attempts"),
                "cache_status": reasoning.get("cache_status"),
            }
    failures = state.get("failures", {})
    if isinstance(failures, dict):
        for window_id, failure_code in failures.items():
            target = targets_by_window.get(str(window_id), {})
            status, code, stage = _qa_failure_metadata({"failure_code": failure_code})
            outcomes.setdefault(
                str(window_id),
                {
                    "window_id": str(window_id),
                    **target,
                    "status": status,
                    "frames": _qa_anchor_frame(target),
                    "failure_code": code,
                    "failure_stage": stage,
                },
            )
    return list(outcomes.values())


def _qa_summary(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    outcomes = _qa_outcomes(snapshot)
    counts = {
        "answerable": sum(item.get("status") == "answered" for item in outcomes),
        "unanswerable": sum(item.get("status") == "unanswerable" for item in outcomes),
        "failed": sum(str(item.get("status", "")).endswith("_failed") for item in outcomes),
        "pending": sum(item.get("status") == "pending" for item in outcomes),
    }
    if snapshot is None:
        return {**counts, "reasoned": 0, "status": "incomplete"}
    aliases = {
        "answerable": "answerable_count",
        "unanswerable": "unanswerable_count",
        "failed": "failed_count",
        "pending": "pending_count",
    }
    for name, field in aliases.items():
        if not outcomes and field in snapshot:
            counts[name] = int(snapshot[field])
    status = snapshot.get("status")
    if status is None:
        if counts["failed"] and not counts["answerable"] and not counts["unanswerable"]:
            status = "failed"
        elif counts["failed"]:
            status = "partial"
        elif counts["answerable"]:
            status = "completed"
        elif counts["unanswerable"]:
            status = "completed_no_answer"
        else:
            status = "incomplete"
    return {
        **counts,
        "reasoned": counts["answerable"] + counts["unanswerable"],
        "status": str(status),
    }


def _apply_qa_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Apply one fetched snapshot without causing another persistence read."""

    st.session_state.qa_snapshot = snapshot
    if snapshot is None:
        st.session_state.qa_predictions = []
        st.session_state.qa_candidate_windows = []
        st.session_state.qa_candidates = []
        st.session_state.qa_answers = []
        st.session_state.qa_result = None
        return None
    st.session_state.qa_search_id = str(
        snapshot.get("search_id") or snapshot.get("session_id") or ""
    ) or None
    candidate_values = snapshot.get("candidates", snapshot.get("candidate_windows", ()))
    st.session_state.qa_candidate_windows = [
        dict(item)
        for item in candidate_values
        if isinstance(item, dict)
    ]
    st.session_state.qa_candidates = list(st.session_state.qa_candidate_windows)
    st.session_state.qa_answers = [
        dict(item) for item in snapshot.get("answers", ()) if isinstance(item, dict)
    ]
    selected_candidate = snapshot.get("selected_candidate_id")
    if (
        isinstance(selected_candidate, str)
        and selected_candidate
        and st.session_state.get("qa_selected_candidate_id") != selected_candidate
    ):
        st.session_state.qa_selected_candidate_id = selected_candidate
    state = snapshot.get("state", {})
    state = state if isinstance(state, dict) else {}
    persisted_edits = snapshot.get("operator_edits", ())
    if isinstance(persisted_edits, list):
        st.session_state.qa_edits = [dict(item) for item in persisted_edits]
    raw_predictions = snapshot.get("predictions", state.get("predictions", ()))
    predictions = [dict(item) for item in raw_predictions if isinstance(item, dict)]
    latest_edits = {
        str(edit["window_id"]): edit
        for edit in st.session_state.qa_edits
        if isinstance(edit, dict) and edit.get("window_id")
    }
    for item in predictions:
        edit = latest_edits.get(str(item.get("window_id")))
        if edit is None:
            continue
        evidence = dict(item.get("evidence", {}))
        evidence["operator_edit"] = dict(edit)
        item["answer"] = edit["normalized_answer"]
        item["answer_source"] = "operator"
        item["evidence"] = evidence
    st.session_state.qa_predictions = predictions
    st.session_state.qa_result = predictions[0] if predictions else None
    return snapshot


def _load_qa_snapshot(context: RuntimeContext, search_id: str) -> dict[str, Any] | None:
    if context.qa_pipeline is None:
        return _apply_qa_snapshot(None)
    return _apply_qa_snapshot(context.qa_pipeline.get_search(search_id))


def _apply_qa_edit(context: RuntimeContext, window_id: str, edited: str) -> None:
    request = st.session_state.qa_request
    if request is None:
        raise ValueError("the QA request is unavailable")
    spec = QACompetitionSpec.for_spec_id(request.answer_spec_id)
    normalized = spec.normalize_answer(edited, request.question)
    predictions = [dict(item) for item in st.session_state.qa_predictions]
    original_answers = {
        str(item["answer"]) for item in predictions if item.get("window_id") == window_id
    }
    if not original_answers:
        raise ValueError("the selected QA window is unavailable")
    edited_at = time.time()
    original_item = next(item for item in predictions if item.get("window_id") == window_id)
    audit = {
        "previous_normalized_answers": sorted(original_answers),
        "original_raw_answer": original_item.get("raw_answer"),
        "provider": dict(original_item.get("evidence", {})).get("provider"),
        "model_name": original_item.get("model_name"),
        "edited_raw_answer": edited,
        "normalized_answer": normalized,
        "edited_at": edited_at,
    }
    if context.qa_session_store is None or not st.session_state.qa_search_id:
        raise ValueError("the QA audit store is unavailable")
    persisted_edit = context.qa_session_store.record_operator_edit(
        str(st.session_state.qa_search_id), window_id, audit
    )
    for item in predictions:
        if item.get("window_id") != window_id:
            continue
        evidence = dict(item.get("evidence", {}))
        evidence["operator_edit"] = dict(persisted_edit)
        item["answer"] = normalized
        item["answer_source"] = "operator"
        item["evidence"] = evidence
    st.session_state.qa_predictions = predictions
    for item in st.session_state.qa_pinned:
        if item.get("window_id") == window_id:
            item["answer"] = normalized
            item["answer_source"] = "operator"
            item["evidence"] = next(
                value["evidence"] for value in predictions if value.get("window_id") == window_id
            )
    st.session_state.qa_edits.append(persisted_edit)
    st.session_state.qa_export_cache = None


def _pin_qa_prediction(prediction: dict) -> None:
    identity = (prediction["video_id"], prediction["source_frame_idx"])
    pinned = {
        (item["video_id"], item["source_frame_idx"]): item for item in st.session_state.qa_pinned
    }
    pinned[identity] = dict(prediction)
    st.session_state.qa_pinned = list(pinned.values())[:100]
    st.session_state.qa_export_cache = None


def _qa_window_frames(outcome: dict[str, Any], predictions: list[dict]) -> list[dict]:
    frames = _qa_frame_refs(
        outcome.get("frames", ()),
        batch_id=outcome.get("batch_id"),
        video_id=outcome.get("video_id"),
    )
    if frames:
        return frames
    for prediction in predictions:
        evidence = prediction.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        frames = evidence.get("window_frames", ())
        if isinstance(frames, (list, tuple)) and frames:
            resolved = _qa_frame_refs(
                frames,
                batch_id=outcome.get("batch_id") or prediction.get("batch_id"),
                video_id=outcome.get("video_id") or prediction.get("video_id"),
            )
            if resolved:
                return resolved
    resolved = _qa_frame_refs(
        [
            {
                "position": index,
                "batch_id": item.get("batch_id"),
                "video_id": item.get("video_id"),
                "source_frame_idx": item.get("source_frame_idx"),
                "timestamp_ms": item.get("timestamp_ms", 0),
            }
            for index, item in enumerate(
                sorted(predictions, key=lambda value: value.get("timestamp_ms", 0)), 1
            )
        ],
        batch_id=outcome.get("batch_id"),
        video_id=outcome.get("video_id"),
    )
    return resolved or _qa_anchor_frame(outcome)


def _render_qa_evidence_summary(
    context: RuntimeContext,
    outcomes: list[dict[str, Any]],
    predictions: list[dict],
) -> None:
    """Highlight only provider-selected evidence frames, never inferred boxes."""

    answered = next((item for item in outcomes if item.get("status") == "answered"), None)
    if answered is None:
        return
    window_id = str(answered.get("window_id", ""))
    window_predictions = [
        item for item in predictions if str(item.get("window_id", "")) == window_id
    ]
    frames = _qa_window_frames(answered, window_predictions)
    positions = {
        int(value)
        for value in answered.get("evidence_frame_indices", ())
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1
    }
    evidence_frames = [
        frame for frame in frames if int(frame.get("position", 0)) in positions
    ]
    if not evidence_frames:
        return
    answer = window_predictions[0].get("answer") if window_predictions else answered.get("answer")
    st.markdown("#### Bằng chứng chính")
    st.success(f"Câu trả lời: `{answer or ''}`")
    columns = st.columns(len(evidence_frames))
    for column, frame_info in zip(columns, evidence_frames, strict=True):
        with column:
            batch_id = frame_info.get("batch_id") or answered.get("batch_id")
            video_id = frame_info.get("video_id") or answered.get("video_id")
            source_frame_idx = int(frame_info["source_frame_idx"])
            timestamp_ms = int(frame_info.get("timestamp_ms", 0))
            try:
                getter = getattr(context.artifact_repository, "get_source_frame", None)
                if not callable(getter):
                    getter = context.artifact_repository.get_frame
                frame = getter(batch_id, video_id, source_frame_idx)
                if frame is None:
                    raise ValueError("missing evidence frame")
                image = context.qa_image_resolver.resolve(frame)
                st.image(image.data, width="stretch")
            except Exception:
                st.warning("Không thể xem trước bằng chứng")
            st.caption(
                f"{video_id} · {timestamp_ms / 1000:.2f}s · source frame {source_frame_idx}"
            )


def _render_qa_window(
    context: RuntimeContext,
    outcomes: list[dict[str, Any]],
    predictions: list[dict],
) -> None:
    windows: dict[str, list[dict]] = {}
    for prediction in predictions:
        windows.setdefault(str(prediction["window_id"]), []).append(prediction)
    outcome_by_id = {str(item["window_id"]): item for item in outcomes if item.get("window_id")}
    if not outcome_by_id:
        return
    window_ids = list(outcome_by_id)
    if st.session_state.qa_selected_window_id not in window_ids:
        st.session_state.qa_selected_window_id = window_ids[0]

    def _window_label(window_id: str) -> str:
        outcome = outcome_by_id[window_id]
        video_id = outcome.get("video_id", "video không rõ")
        timestamp = float(outcome.get("timestamp_ms", 0)) / 1000
        return f"{_vi_status(outcome.get('status', 'pending'))} · {video_id} · {timestamp:.2f}s"

    window_id = st.selectbox(
        "Cửa sổ ứng viên",
        window_ids,
        format_func=_window_label,
        key="qa_selected_window_id",
    )
    outcome = outcome_by_id[window_id]
    items = windows.get(window_id, [])
    by_frame = {int(item["source_frame_idx"]): item for item in items}
    window_frames = _qa_window_frames(outcome, items)
    selected_positions = set(outcome.get("evidence_frame_indices", ()))
    if not selected_positions and items:
        evidence = items[0].get("evidence", {})
        if isinstance(evidence, dict):
            selected_positions = set(evidence.get("evidence_frame_indices", ()))
    columns = st.columns(len(window_frames)) if window_frames else []
    for column, frame_info in zip(columns, window_frames, strict=True):
        try:
            source_frame_idx = int(frame_info["source_frame_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        prediction = by_frame.get(source_frame_idx)
        batch_id = frame_info.get("batch_id") or outcome.get("batch_id")
        video_id = frame_info.get("video_id") or outcome.get("video_id")
        with column:
            try:
                if context.artifact_repository is None or context.qa_image_resolver is None:
                    raise ValueError("canonical image service unavailable")
                getter = getattr(context.artifact_repository, "get_source_frame", None)
                if not callable(getter):
                    getter = context.artifact_repository.get_frame
                frame = getter(batch_id, video_id, source_frame_idx)
                if frame is None:
                    raise ValueError("missing frame")
                image = context.qa_image_resolver.resolve(frame)
                st.image(image.data, width="stretch")
            except Exception:
                st.warning("Không thể xem trước")
            position = int(frame_info.get("position", 0))
            marker = "Bằng chứng" if position in selected_positions else "Ngữ cảnh"
            st.caption(f"{marker} · khung hình {source_frame_idx}")
            if (
                outcome.get("status") == "answered"
                and prediction is not None
                and st.button("Ghim", key=f"qa_pin_{window_id}_{source_frame_idx}")
            ):
                _pin_qa_prediction(prediction)
                st.rerun()

    status = outcome.get("status")
    if status == "unanswerable":
        st.info(
            "Không đủ bằng chứng: VLM đã chạy bình thường nhưng chưa tìm thấy đủ căn cứ "
            "trong cửa sổ này. Hãy chỉnh mô tả sự kiện hoặc mở rộng thêm cụm ứng viên."
        )
        return
    if isinstance(status, str) and status.endswith("_failed"):
        code = outcome.get("failure_code", "internal_reasoning_error")
        stage = outcome.get("failure_stage", "internal")
        category = outcome.get("provider_error_category")
        detail = f" · nguyên nhân: {_vi_status(category)}" if category else ""
        st.warning(f"Xử lý cửa sổ thất bại an toàn tại bước {stage}: `{code}`{detail}.")
        return
    if status == "pending":
        st.info("Cửa sổ ứng viên này đang chờ suy luận hình ảnh.")
        return
    if status != "answered" or not items:
        st.caption("Cửa sổ này chưa tạo được câu trả lời có thể xuất.")
        return

    first = items[0]
    st.write(f"Câu trả lời đã chuẩn hóa: `{first['answer']}`")
    st.caption(
        "Câu trả lời gốc của mô hình: "
        f"{first.get('raw_answer')!r} · {first.get('answer_source')} · "
        f"điểm xếp hạng đại diện {float(first.get('confidence_score', 0.0)):.3f}"
    )
    edited = st.text_input(
        "Chỉnh sửa câu trả lời của người vận hành",
        value=str(first["answer"]),
        key=f"qa_edit_{st.session_state.qa_search_id}_{window_id}",
    )
    if st.button("Áp dụng chỉnh sửa có lưu vết", key=f"qa_apply_edit_{window_id}"):
        try:
            _apply_qa_edit(context, window_id, edited)
            st.rerun()
        except (TypeError, ValueError):
            st.error("Chỉnh sửa bị từ chối do không đúng đặc tả câu trả lời đã chọn.")


def _qa_trace_rows(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only safe execution metadata; never model reasoning text."""

    rows: list[dict[str, Any]] = []
    for item in outcomes:
        row = {
            "window_id": item.get("window_id"),
            "outcome": item.get("status"),
            "provider": item.get("provider") or "not recorded",
            "model": item.get("model_name") or "not recorded",
            "cache": item.get("cache_status"),
            "provider_latency_ms": item.get("provider_latency_ms"),
            "served_latency_ms": item.get("served_latency_ms"),
            "attempts": item.get("attempts"),
        }
        if str(item.get("status", "")).endswith("_failed"):
            row["failure_code"] = item.get("failure_code", "internal_reasoning_error")
            row["failure_stage"] = item.get("failure_stage", "internal")
            category = item.get("provider_error_category")
            if category in _QA_PROVIDER_ERROR_CATEGORIES:
                row["provider_error_category"] = category
            provider_status = item.get("provider_status_code")
            if (
                isinstance(provider_status, int)
                and not isinstance(provider_status, bool)
                and 100 <= provider_status <= 599
            ):
                row["provider_status_code"] = provider_status
        cache_layers = item.get("cache_layers")
        if isinstance(cache_layers, dict):
            row["cache_layers"] = ", ".join(
                f"{name}={value}" for name, value in sorted(cache_layers.items())
            )
        rows.append(row)
    return rows


def _qa_prediction_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project predictions onto the fixed operator-facing result columns."""

    allowed = (
        "rank",
        "batch_id",
        "video_id",
        "source_frame_idx",
        "timestamp_ms",
        "answer",
        "answer_source",
        "confidence_score",
        "window_id",
    )
    return [
        {field: item.get(field) for field in allowed}
        for item in predictions
        if isinstance(item, dict)
    ]


def _qa_operator_edit_rows(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project persisted answer edits without exposing session or extra fields."""

    allowed = (
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
    )
    return [
        {field: item.get(field) for field in allowed} for item in edits if isinstance(item, dict)
    ]


def _qa_channel_status_rows(
    channel_statuses: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project session retrieval status onto documented safe columns."""

    allowed = (
        "applicability",
        "state",
        "reason_code",
        "coverage",
        "count",
        "collection_name",
        "index_name",
        "expected_dimension",
        "remediation_id",
    )
    return [
        {"channel": str(name), **{field: status.get(field) for field in allowed}}
        for name, status in sorted(channel_statuses.items(), key=lambda item: str(item[0]))
        if isinstance(status, dict)
    ]


def _qa_exportable_pins(pinned: list[dict], answered_window_ids: set[str] | None) -> list[dict]:
    return [
        item
        for item in pinned
        if answered_window_ids is None or str(item.get("window_id")) in answered_window_ids
    ]


def _render_qa_export(
    context: RuntimeContext,
    *,
    answered_window_ids: set[str] | None = None,
) -> None:
    st.subheader("Các dự đoán Hỏi đáp hình ảnh đã ghim")
    pinned = _qa_exportable_pins(st.session_state.qa_pinned, answered_window_ids)
    if not pinned:
        st.caption("Ghim các khung hình có bằng chứng để chuẩn bị bài nộp ba cột.")
        return
    rows = [dict(item, rank=index) for index, item in enumerate(pinned, 1)]
    st.dataframe(
        [
            {
                "rank": item["rank"],
                "video_id": item["video_id"],
                "source_frame_idx": item["source_frame_idx"],
                "answer": item["answer"],
                "answer_source": item["answer_source"],
            }
            for item in rows
        ],
        width="stretch",
        hide_index=True,
    )
    batch_ids = {str(item["batch_id"]) for item in rows if item.get("batch_id")}
    if len(batch_ids) != 1 or context.artifact_repository is None:
        st.error("Các dự đoán đã ghim phải thuộc cùng một batch đang hoạt động và đã lập chỉ mục.")
        st.session_state.qa_export_cache = None
    else:
        batch_id = next(iter(batch_ids))
        fingerprint = _result_fingerprint(workflow="qa", batch_id=batch_id, rows=rows)
        cached = st.session_state.get("qa_export_cache")
        if not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint:
            st.session_state.qa_export_cache = None
            cached = None
        if st.button("Kiểm tra và chuẩn bị CSV hỏi đáp", key="qa_export"):
            started = time.monotonic()
            try:
                request = st.session_state.qa_request
                if request is None:
                    raise ValueError("the QA answer specification is unavailable")
                spec = QACompetitionSpec.for_spec_id(request.answer_spec_id)
                canonical_check, evidence, _ = _submission_validation_context(
                    context, batch_id, rows
                )
                filename = f"visual-qa-{str(st.session_state.qa_search_id)[:8]}.csv"
                data = serialize_qa_submission(
                    rows,
                    spec=spec,
                    batch_id=batch_id,
                    is_batch_active=context.ledger.is_batch_active,
                    is_canonical_source_frame=canonical_check,
                    frame_index_evidence=evidence,
                )
                cached = {
                    "fingerprint": fingerprint,
                    "filename": filename,
                    "data": data,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                st.session_state.qa_export_cache = cached
            except Exception as error:
                st.session_state.qa_export_cache = None
                _report_ui_error(
                    error,
                    workflow="qa",
                    stage="export",
                    batch_id=batch_id,
                    query_or_session_id=st.session_state.qa_search_id,
                    started_at=started,
                )
                cached = None
        if isinstance(cached, dict) and isinstance(cached.get("data"), bytes):
            st.download_button(
                "Tải CSV hỏi đáp đã kiểm tra",
                data=cached["data"],
                file_name=str(cached["filename"]),
                mime="text/csv",
                key="qa_download",
            )
            if st.button("Lưu CSV hỏi đáp trên máy chủ", key="qa_save_server"):
                output_dir = Path(context.config["competition"]["output_dir"])
                if not output_dir.is_absolute():
                    output_dir = PROJECT_ROOT / output_dir
                try:
                    target = atomic_write_bytes(
                        output_dir / str(cached["filename"]),
                        cached["data"],
                    )
                except OSError as error:
                    _report_ui_error(
                        error,
                        workflow="qa",
                        stage="server_save",
                        batch_id=batch_id,
                        query_or_session_id=st.session_state.qa_search_id,
                    )
                else:
                    st.success(f"Đã lưu `{target.name}`")
    if st.button("Xóa các dự đoán hỏi đáp đã ghim", key="qa_clear_pinned"):
        st.session_state.qa_pinned = []
        st.session_state.qa_export_cache = None
        st.rerun()


def _render_qa_legacy(context: RuntimeContext) -> None:
    st.subheader("Hỏi đáp dựa trên hình ảnh")
    if context.qa_pipeline is None:
        st.error("Hỏi đáp hình ảnh không khả dụng vì dữ liệu chuẩn chưa được cấu hình.")
        return
    capabilities = getattr(context, "capabilities", {})
    refresh_capabilities = getattr(context, "refresh_capabilities", None)
    if callable(refresh_capabilities):
        try:
            capabilities = refresh_capabilities()
        except Exception:
            pass
    qa_capabilities = capabilities.get("qa", {}) if isinstance(capabilities, dict) else {}
    local_status = qa_capabilities.get("local", {}).get("status", {})
    cloud_status = qa_capabilities.get("cloud", {}).get("status", {})
    engine = st.selectbox(
        "Bộ máy suy luận",
        ("gemini", "local"),
        format_func=lambda value: (
            "Gemini Flash API (Cloud — consent required)"
            if value == "gemini"
            else "Local Qwen2.5-VL (Ollama — 100% Offline)"
        ),
        key="qa_engine",
    )
    local_state = str(local_status.get("state", "ready"))
    cloud_state = str(cloud_status.get("state", "blocked_asset"))
    engine_state = local_state if engine == "local" else cloud_state
    engine_ready = engine_state in {"ready", "configured_unverified"}
    if engine_state in {"ready", "configured_unverified"}:
        st.success(
            "🟢 Engine: Gemini Flash"
            if engine == "gemini"
            else "🟢 Engine: Local Qwen2.5-VL (Offline)"
        )
    elif engine_state in {"warming", "idle", "evicted"}:
        st.warning("🟡 Engine Visual Q&A đang khởi động; giao diện sẽ tự cập nhật.")
    elif engine_state == "service_unavailable":
        st.warning("🟡 Ollama chưa chạy hoặc chưa phản hồi trên loopback.")
    else:
        st.error("🔴 Bộ máy Visual Q&A đã chọn chưa sẵn sàng.")
    cloud_requested = engine == "gemini"
    consent = False
    if cloud_requested:
        consent = st.checkbox(
            "Tôi đồng ý gửi đúng một frame đã chọn tới Gemini trong phiên này",
            key="qa_external_transfer_consent",
        )
        st.warning("Chế độ này có thể truyền ảnh ra dịch vụ cloud bên ngoài.")
    reasoning_policy = "gemini_then_local" if engine == "gemini" else "local_only"
    st.caption(
        "Qwen2.5-VL chỉ nhận các cửa sổ frame canonical sau bước truy xuất; "
        "câu trả lời được chuẩn hóa ngắn gọn và giữ provenance frame."
    )
    event_description = st.text_area("Mô tả sự kiện dùng để truy xuất", key="qa_event_description")
    question = st.text_input("Câu hỏi cần trả lời từ các khung hình đã chọn", key="qa_question")
    with st.expander("Chẩn đoán nhanh bằng một ảnh (không dùng để export)"):
        uploaded = st.file_uploader(
            "Ảnh PNG, JPEG hoặc WebP",
            type=("png", "jpg", "jpeg", "webp"),
            key="qa_diagnostic_image",
        )
        diagnostic_question = st.text_input(
            "Câu hỏi cho ảnh chẩn đoán",
            value="Có bao nhiêu hình tròn màu đỏ?",
            key="qa_diagnostic_question",
        )
        diagnostic_call = getattr(context, "diagnostic_image", None)
        diagnostic_clicked = st.button(
            "Phân tích ảnh chẩn đoán",
            key="qa_diagnostic_submit",
            disabled=(
                uploaded is None
                or not diagnostic_question.strip()
                or not callable(diagnostic_call)
                or not engine_ready
                or (cloud_requested and not consent)
            ),
        )
        if diagnostic_clicked and uploaded is not None and callable(diagnostic_call):
            raw = uploaded.getvalue()
            mime_type = uploaded.type
            try:
                st.session_state.qa_diagnostic_result = diagnostic_call(
                    {
                        "image_base64": base64.b64encode(raw).decode("ascii"),
                        "mime_type": mime_type,
                        "question": diagnostic_question,
                        "reasoning_policy": reasoning_policy,
                        "allow_external_frame_transfer": consent,
                    }
                )
            except Exception as error:
                st.session_state.qa_diagnostic_result = None
                _report_ui_error(
                    error,
                    workflow="qa",
                    stage="diagnostic_image",
                    batch_id=_selected_batch_id(context),
                )
        diagnostic_result = st.session_state.get("qa_diagnostic_result")
        if isinstance(diagnostic_result, dict):
            st.success(f"Kết quả: {diagnostic_result.get('answer', '')}")
            st.caption(
                f"Provider: {diagnostic_result.get('actual_provider', 'unknown')} · "
                f"Latency: {float(diagnostic_result.get('latency_ms', 0)):.0f} ms"
            )
    if not st.session_state.qa_batch_filters_seeded:
        if not str(st.session_state.get("qa_batch_filters", "")).strip():
            selected_batch = _selected_batch_id(context)
            if selected_batch:
                st.session_state.qa_batch_filters = selected_batch
        st.session_state.qa_batch_filters_seeded = True
    controls = st.columns((1, 1, 1))
    with controls[0]:
        mode = st.radio(
            "Chế độ hỏi đáp",
            ("interactive", "automatic"),
            format_func=lambda value: "Tương tác" if value == "interactive" else "Tự động",
            horizontal=True,
            key="qa_mode",
        )
    with controls[1]:
        answer_spec_id = st.selectbox(
            "Đặc tả câu trả lời",
            (
                QACompetitionSpec.conservative_general_text().spec_id,
                QACompetitionSpec.historical_numeric().spec_id,
            ),
            format_func=lambda value: (
                "Khớp chính xác dạng số theo thể lệ cũ"
                if value == QACompetitionSpec.historical_numeric().spec_id
                else "Khớp chính xác văn bản tổng quát (tạm thời)"
            ),
            key="qa_answer_spec",
        )
    with controls[2]:
        batch_text = st.text_input("Bộ lọc batch (phân cách bằng dấu phẩy)", key="qa_batch_filters")
    if (
        answer_spec_id == QACompetitionSpec.historical_numeric().spec_id
        and question.strip()
        and not any(
            marker in question.casefold()
            for marker in ("how many", "how much", "number", "bao nhiêu", "mấy", "số lượng")
        )
    ):
        st.warning(
            "Câu hỏi này không có dấu hiệu hỏi số lượng. Hãy dùng đặc tả văn bản tổng quát "
            "cho câu trả lời có/không, màu sắc hoặc tên vật thể."
        )

    action_col, restore_col = st.columns((1, 2))
    with action_col:
        policy_ready = engine_ready and (not cloud_requested or consent)
        qa_retrieval_warmup = _warmup_workflow_snapshot(_warmup_snapshot(context), "kis")
        qa_retrieval_state = str(qa_retrieval_warmup.get("state", "ready"))
        retrieval_actionable = bool(qa_retrieval_warmup.get("usable", False)) or (
            qa_retrieval_state in {"idle", "evicted"}
        )
        search_clicked = st.button(
            "Truy xuất và trả lời",
            type="primary",
            key="qa_search",
            disabled=not (retrieval_actionable and policy_ready),
        )
    with restore_col:
        restore_id = st.text_input("Khôi phục theo mã tìm kiếm", key="qa_restore_id")
        restore_clicked = st.button("Khôi phục", key="qa_restore")

    if search_clicked:
        if not event_description.strip() or not question.strip():
            st.warning("Hãy nhập cả mô tả sự kiện và câu hỏi.")
        else:
            try:
                requested_filters = tuple(
                    value.strip() for value in batch_text.split(",") if value.strip()
                )
                batch_filters = _validated_batch_filters(context, requested_filters or None)
                request = QAQueryRequest(
                    query_id=str(uuid4()),
                    event_description=event_description,
                    question=question,
                    mode=mode,
                    answer_spec_id=answer_spec_id,
                    batch_filters=batch_filters,
                    limit=100,
                    reasoning_policy=reasoning_policy,
                    allow_external_frame_transfer=consent,
                )
            except (TypeError, ValueError):
                st.error(
                    "Lựa chọn batch cho hỏi đáp hình ảnh không hợp lệ. Hãy chọn batch tham "
                    "chiếu đã cấu hình hoặc batch gốc đã đăng ký."
                )
            else:
                search_started = time.monotonic()
                st.session_state.qa_edits = []
                st.session_state.qa_pinned = []
                st.session_state.qa_export_cache = None
                st.session_state.qa_search_id = None
                st.session_state.qa_request = request
                st.session_state.qa_snapshot = None
                st.session_state.qa_candidate_windows = []
                st.session_state.qa_predictions = []
                st.session_state.qa_result = None
                st.session_state.qa_last_failure = None
                st.session_state.qa_selected_window_id = None
                with st.spinner(
                    "Đang truy xuất cửa sổ tốt nhất và suy luận trên ba khung hình..."
                ):
                    try:
                        context.qa_pipeline.search(request)
                        snapshot = getattr(context.qa_pipeline, "last_snapshot", None)
                        if snapshot is None:
                            snapshot = context.qa_pipeline.get_search(request.query_id)
                        if snapshot is None:
                            raise BackendError("backend_response_invalid")
                        _apply_qa_snapshot(snapshot)
                    except BackendError as error:
                        snapshot = error.snapshot
                        if snapshot is not None:
                            _apply_qa_snapshot(snapshot)
                            st.session_state.qa_last_failure = {
                                "code": error.code,
                                "search_id": st.session_state.qa_search_id,
                                "recoverable": True,
                            }
                        elif error.code in {"runtime_warming", "runtime_not_ready"}:
                            st.session_state.qa_last_failure = {
                                "code": error.code,
                                "search_id": None,
                                "recoverable": True,
                            }
                        else:
                            _report_ui_error(
                                error,
                                workflow="qa",
                                stage="search",
                                batch_id=batch_filters[0] if batch_filters else None,
                                query_or_session_id=request.query_id,
                                started_at=search_started,
                            )
                    except Exception as error:
                        _report_ui_error(
                            error,
                            workflow="qa",
                            stage="search",
                            batch_id=batch_filters[0] if batch_filters else None,
                            query_or_session_id=request.query_id,
                            started_at=search_started,
                        )

    if restore_clicked:
        try:
            snapshot = context.qa_pipeline.get_search(restore_id.strip())
            if snapshot is None:
                raise ValueError("missing or expired")
            st.session_state.qa_search_id = restore_id.strip()
            st.session_state.qa_request = QAQueryRequest.from_dict(snapshot["request"])
            st.session_state.qa_pinned = []
            st.session_state.qa_export_cache = None
            st.session_state.qa_selected_window_id = None
            st.session_state.qa_last_failure = None
            _apply_qa_snapshot(snapshot)
        except Exception as error:
            _report_ui_error(
                error,
                workflow="qa",
                stage="restore",
                batch_id=_selected_batch_id(context),
                query_or_session_id=restore_id,
            )

    failure = st.session_state.get("qa_last_failure")
    if isinstance(failure, dict):
        if failure.get("search_id"):
            st.warning(
                "Ollama chưa tạo được câu trả lời hợp lệ, nhưng phiên đã được lưu an toàn. "
                f"Mã lỗi `{failure.get('code', 'qa_reasoning_unavailable')}` · "
                f"mã tìm kiếm `{failure['search_id']}`."
            )
        elif failure.get("code") == "runtime_warming":
            st.info(
                "Bộ truy xuất đang được làm nóng lại sau khi nhường GPU cho Qwen. "
                "Nút truy xuất sẽ tự mở khóa khi sẵn sàng."
            )
        else:
            st.warning(
                "Visual Q&A tạm thời chưa sẵn sàng. "
                f"Mã lỗi `{failure.get('code', 'runtime_not_ready')}`."
            )

    snapshot = st.session_state.qa_snapshot
    if snapshot is None and st.session_state.qa_search_id:
        st.info(f"Mã tìm kiếm: {st.session_state.qa_search_id} · trạng thái đã lưu không khả dụng.")
    if snapshot is not None:
        requested_target = int(
            snapshot.get("requested_target_clusters", snapshot.get("target_count", 0))
        )
        retrieved_clusters = int(snapshot.get("retrieved_clusters", requested_target))
        summary = _qa_summary(snapshot)
        st.info(
            f"Mã tìm kiếm: {snapshot['session_id']} · trạng thái {_vi_status(summary['status'])} · "
            f"đã trả lời {summary['answerable']} · không đủ bằng chứng {summary['unanswerable']} · "
            f"lỗi {summary['failed']} · đang chờ {summary['pending']} · "
            f"kho ứng viên {retrieved_clusters} · đang xét {requested_target} · tối đa 20"
        )
        request = st.session_state.qa_request
        can_expand = (
            request is not None
            and request.mode == "interactive"
            and requested_target < 20
            and not bool(snapshot.get("retrieval_exhausted", False))
        )
        if can_expand:
            target = min(20, requested_target + 1)
            if st.button(
                "Thử nhanh cửa sổ kế tiếp",
                key=f"qa_fast_retry_{target}",
            ):
                with st.spinner("Đang suy luận trên cửa sổ thay thế tiếp theo..."):
                    try:
                        context.qa_pipeline.expand(snapshot["session_id"], target)
                        refreshed = getattr(context.qa_pipeline, "last_snapshot", None)
                        if refreshed is None:
                            refreshed = _load_qa_snapshot(context, snapshot["session_id"])
                        if refreshed is None:
                            raise RuntimeError("expanded search state is unavailable")
                        _apply_qa_snapshot(refreshed)
                        st.session_state.qa_last_failure = None
                        st.rerun()
                    except BackendError as error:
                        try:
                            recovered = _load_qa_snapshot(context, snapshot["session_id"])
                        except BackendError:
                            recovered = None
                        if recovered is not None:
                            st.session_state.qa_last_failure = {
                                "code": error.code,
                                "search_id": snapshot["session_id"],
                                "recoverable": True,
                            }
                        else:
                            _report_ui_error(
                                error,
                                workflow="qa",
                                stage="expand",
                                batch_id=(
                                    request.batch_filters[0]
                                    if request is not None and request.batch_filters
                                    else None
                                ),
                                query_or_session_id=snapshot["session_id"],
                            )
                    except Exception as error:
                        _report_ui_error(
                            error,
                            workflow="qa",
                            stage="expand",
                            batch_id=(
                                request.batch_filters[0]
                                if request is not None and request.batch_filters
                                else None
                            ),
                            query_or_session_id=snapshot["session_id"],
                        )
        if snapshot.get("retrieval_exhausted") and requested_target < 20:
            st.caption("Đã truy xuất hết; không còn cửa sổ ứng viên duy nhất nào khác.")
        retrieval_errors = snapshot.get("retrieval_errors", {})
        if isinstance(retrieval_errors, dict) and retrieval_errors:
            st.caption(
                "Các kênh truy xuất không khả dụng: "
                + ", ".join(f"{name} ({code})" for name, code in sorted(retrieval_errors.items()))
            )
        elif isinstance(retrieval_errors, (list, tuple)) and retrieval_errors:
            st.caption(
                "Các kênh truy xuất không khả dụng: "
                + ", ".join(str(item) for item in retrieval_errors)
            )
        channel_statuses = snapshot.get("channel_statuses", {})
        if isinstance(channel_statuses, dict) and channel_statuses:
            with st.expander("Trạng thái các kênh truy xuất của phiên"):
                st.dataframe(
                    _qa_channel_status_rows(channel_statuses),
                    width="stretch",
                    hide_index=True,
                )

    predictions = st.session_state.qa_predictions
    outcomes = _qa_outcomes(snapshot)
    if outcomes:
        _render_qa_evidence_summary(context, outcomes, predictions)
        _render_qa_window(context, outcomes, predictions)
        with st.expander("Nhật ký chạy VLM (chỉ siêu dữ liệu an toàn)"):
            st.caption(
                "Bảng này chỉ chứa siêu dữ liệu thực thi; không hiển thị chuỗi suy luận ẩn "
                "hoặc dữ liệu yêu cầu gửi tới nhà cung cấp."
            )
            st.dataframe(_qa_trace_rows(outcomes), width="stretch", hide_index=True)
            if predictions:
                st.caption("Dự đoán xếp hạng có bằng chứng")
                st.dataframe(_qa_prediction_rows(predictions), width="stretch", hide_index=True)
    if st.session_state.qa_edits:
        with st.expander("Nhật ký chỉnh sửa câu trả lời của người vận hành"):
            st.dataframe(
                _qa_operator_edit_rows(st.session_state.qa_edits),
                width="stretch",
                hide_index=True,
            )
    st.divider()
    answered_window_ids = {
        str(item["window_id"])
        for item in outcomes
        if item.get("window_id") and item.get("status") == "answered"
    }
    _render_qa_export(context, answered_window_ids=answered_window_ids)


def _render_qa(context: RuntimeContext) -> None:
    """Render the decoupled retrieval then single-frame reasoning workflow."""

    st.subheader("Hỏi đáp hình ảnh · Hai bước độc lập")
    if context.qa_pipeline is None:
        st.error("Visual Q&A chưa khả dụng trong runtime hiện tại.")
        return

    event_description = st.text_area(
        "Mô tả sự kiện dùng cho KIS Retrieval",
        key="qa_event_description",
        placeholder="Ví dụ: A man is broadcasting TV news.",
    )
    qa_language_col, qa_override_col = st.columns((1, 2))
    with qa_language_col:
        qa_query_language = st.selectbox(
            "Ngôn ngữ mô tả sự kiện",
            ("auto", "vi", "en"),
            format_func=lambda value: {
                "auto": "Tự động",
                "vi": "Tiếng Việt",
                "en": "English",
            }[value],
            key="qa_query_language",
        )
    with qa_override_col:
        qa_edit_translation = st.checkbox(
            "Chỉnh bản dịch sự kiện",
            key="qa_translation_edit",
        )
        qa_event_description_en = (
            st.text_input(
                "Mô tả sự kiện bằng tiếng Anh",
                key="qa_event_description_en",
                placeholder="Để trống để dùng bộ dịch offline",
            )
            if qa_edit_translation
            else ""
        )
    question = st.text_input(
        "Câu hỏi cho khung hình đã chọn",
        key="qa_question",
        placeholder="Ví dụ: Does the man wear glasses?",
    )
    batch_id = _selected_batch_id(context)
    st.caption(f"Batch hiện hành: `{batch_id or 'chưa chọn'}`")

    retrieve_clicked = st.button(
        "1. Truy xuất khung hình – KIS Search",
        type="primary",
        key="qa_retrieve_candidates",
        disabled=not bool(event_description.strip() and batch_id),
    )
    if retrieve_clicked:
        request = QACandidateRequest(
            query_id=str(uuid4()),
            event_description=event_description,
            batch_filters=(batch_id,),
            top_k=5,
            query_language=qa_query_language,
            event_description_en=qa_event_description_en or None,
        )
        st.session_state.qa_search_id = None
        st.session_state.qa_request = request
        st.session_state.qa_candidates = []
        st.session_state.qa_answers = []
        st.session_state.qa_predictions = []
        st.session_state.qa_result = None
        st.session_state.qa_selected_candidate_id = None
        st.session_state.qa_pinned = []
        st.session_state.qa_export_cache = None
        with st.spinner("Đang truy xuất Top-5 khung hình bằng KIS..."):
            try:
                context.qa_pipeline.retrieve(request)
                snapshot = getattr(context.qa_pipeline, "last_snapshot", None)
                if snapshot is None:
                    raise BackendError("backend_response_invalid")
                _apply_qa_snapshot(snapshot)
            except BackendError as error:
                _report_ui_error(
                    error,
                    workflow="qa",
                    stage="retrieve",
                    batch_id=batch_id,
                    query_or_session_id=request.query_id,
                )
            except Exception as error:
                _report_ui_error(
                    error,
                    workflow="qa",
                    stage="retrieve",
                    batch_id=batch_id,
                    query_or_session_id=request.query_id,
                )

    candidates = [
        dict(value)
        for value in st.session_state.get("qa_candidates", ())
        if isinstance(value, dict)
    ]
    _render_query_processing(
        dict(st.session_state.get("qa_snapshot") or {}).get("query_processing", {})
    )
    if candidates:
        candidate_ids = tuple(str(value["candidate_id"]) for value in candidates)
        selected = st.session_state.get("qa_selected_candidate_id")
        if selected not in candidate_ids:
            selected = candidate_ids[0]
            st.session_state.qa_selected_candidate_id = selected
        retrieval_ms = float(
            dict(st.session_state.qa_snapshot or {})
            .get("stage_timings", {})
            .get("candidate_retrieval_ms", 0.0)
        )
        st.success(
            f"Đã truy xuất {len(candidates)} candidate trong {retrieval_ms:.0f} ms; chưa gọi VLM."
        )
        for offset in range(0, len(candidates), 3):
            columns = st.columns(min(3, len(candidates) - offset))
            for column, candidate in zip(columns, candidates[offset : offset + 3], strict=True):
                with column:
                    with st.container(
                        key=hud_container_key("qa-card", candidate["candidate_id"]),
                        border=True,
                    ):
                        image = _source_frame_image(
                            context,
                            str(candidate["batch_id"]),
                            str(candidate["video_id"]),
                            int(candidate["source_frame_idx"]),
                        )
                        if image is not None:
                            st.image(image, width="stretch")
                        is_selected = str(candidate["candidate_id"]) == str(selected)
                        st.markdown(
                            render_hud_readout(
                                (
                                    "SELECTED TARGET"
                                    if is_selected
                                    else f"CANDIDATE {int(candidate['retrieval_rank']):02d}"
                                ),
                                candidate["video_id"],
                                tone="green" if is_selected else "cyan",
                            ),
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"Frame {candidate['source_frame_idx']} · "
                            f"{int(candidate['timestamp_ms']) / 1000:.2f}s"
                        )
                        if st.button(
                            "Chọn frame này" if not is_selected else "✓ Frame đang chọn",
                            key=hud_container_key("qa-select", candidate["candidate_id"]),
                            disabled=is_selected,
                            width="stretch",
                        ):
                            st.session_state.qa_selected_candidate_id = str(
                                candidate["candidate_id"]
                            )
                            st.rerun()
        selected = st.radio(
            "Chọn đúng một candidate để suy luận",
            candidate_ids,
            index=candidate_ids.index(selected),
            format_func=lambda candidate_id: next(
                (
                    f"#{value['retrieval_rank']} · {value['video_id']} · "
                    f"frame {value['source_frame_idx']} · {int(value['timestamp_ms']) / 1000:.2f}s"
                    for value in candidates
                    if value["candidate_id"] == candidate_id
                ),
                candidate_id,
            ),
            key="qa_selected_candidate_id",
        )

        primary_engine = st.selectbox(
            "Bộ máy trả lời single-frame",
            ("gemini", "local"),
            format_func=lambda value: (
                "⚡ Gemini Flash API — Cloud, nhanh"
                if value == "gemini"
                else "🛡️ Qwen2.5-VL — Ollama, 100% offline"
            ),
            key="qa_engine",
        )
        answer_spec_id = st.selectbox(
            "Chuẩn hóa câu trả lời",
            (
                QACompetitionSpec.conservative_general_text().spec_id,
                QACompetitionSpec.historical_numeric().spec_id,
            ),
            format_func=lambda value: (
                "Số nguyên" if value == QACompetitionSpec.historical_numeric().spec_id else "Văn bản"
            ),
            key="qa_answer_spec",
        )
        reasoning_policy = (
            "gemini_then_local" if primary_engine == "gemini" else "local_only"
        )
        consent = False
        answer_engine = "vlm"
        florence_task = target_label = None
        if primary_engine == "gemini":
            consent = st.checkbox(
                "Tôi đồng ý truyền đúng một frame đã chọn tới Gemini trong phiên này",
                key="qa_external_transfer_consent",
            )
            st.caption("⚡ Gemini primary · Qwen local chỉ fallback khi lỗi tạm thời")
        else:
            st.caption("🛡️ Offline mode · không thực hiện request VLM ra Internet")

        with st.expander("Florence-2 thử nghiệm · OCR / đếm object"):
            use_florence = st.checkbox(
                "Dùng Florence-2 CPU thay cho VLM chính",
                key="qa_use_florence",
            )
            if use_florence:
                answer_engine = "florence2"
                reasoning_policy = "local_only"
                consent = False
                st.warning(
                    "Florence-2 chỉ hỗ trợ OCR hoặc đếm nhãn object do operator chỉ định."
                )
                florence_task = st.selectbox(
                    "Tác vụ Florence-2",
                    ("ocr", "object_count"),
                    format_func=lambda value: "OCR" if value == "ocr" else "Đếm object",
                    key="qa_florence_task",
                )
                if florence_task == "object_count":
                    target_label = st.text_input(
                        "Nhãn object chính xác cần đếm",
                        key="qa_florence_target_label",
                    )

        answer_disabled = (
            not question.strip()
            or (reasoning_policy == "gemini_then_local" and not consent)
            or (answer_engine == "florence2" and florence_task == "object_count" and not str(target_label or "").strip())
        )
        answer_label = (
            "⚡ Trả lời câu hỏi bằng Gemini Flash"
            if reasoning_policy == "gemini_then_local"
            else "🛡️ Trả lời offline bằng Qwen"
            if answer_engine == "vlm"
            else "🧪 Trả lời bằng Florence-2"
        )
        answer_clicked = st.button(
            f"2. {answer_label}",
            type="primary",
            key="qa_answer_selected_candidate",
            disabled=answer_disabled,
        )
        if answer_clicked:
            answer_request = QASingleFrameAnswerRequest(
                candidate_id=selected,
                question=question,
                answer_spec_id=answer_spec_id,
                reasoning_policy=reasoning_policy,
                allow_external_frame_transfer=consent,
                answer_engine=answer_engine,
                florence_task=florence_task,
                florence_target_label=target_label,
            )
            selected_candidate = next(
                value for value in candidates if value["candidate_id"] == selected
            )
            st.session_state.qa_request = QAQueryRequest(
                query_id=str(st.session_state.qa_search_id),
                event_description=event_description,
                question=question,
                mode="interactive",
                answer_spec_id=answer_spec_id,
                batch_filters=(str(selected_candidate["batch_id"]),),
                limit=100,
                reasoning_policy=reasoning_policy,
                allow_external_frame_transfer=consent,
            )
            with st.spinner("Đang suy luận trên đúng một khung hình..."):
                try:
                    context.qa_pipeline.answer_candidate(
                        str(st.session_state.qa_search_id), answer_request
                    )
                    snapshot = getattr(context.qa_pipeline, "last_snapshot", None)
                    if snapshot is None:
                        raise BackendError("backend_response_invalid")
                    _apply_qa_snapshot(snapshot)
                except BackendError as error:
                    code = error.code
                    if code == "florence_not_provisioned":
                        st.warning("Florence-2 chưa được provision offline trên máy này.")
                    else:
                        _report_ui_error(
                            error,
                            workflow="qa",
                            stage="single_frame_answer",
                            batch_id=batch_id,
                            query_or_session_id=st.session_state.qa_search_id,
                        )
                except Exception as error:
                    _report_ui_error(
                        error,
                        workflow="qa",
                        stage="single_frame_answer",
                        batch_id=batch_id,
                        query_or_session_id=st.session_state.qa_search_id,
                    )

    result = st.session_state.get("qa_result")
    if isinstance(result, dict):
        st.subheader("Kết quả single-frame")
        with st.container(key="hud-qa-evidence", border=True):
            st.markdown(
                render_hud_readout("VLM ANSWER", result["answer"], tone="green"),
                unsafe_allow_html=True,
            )
            st.code(
                f"{result['video_id']}, {result['source_frame_idx']}, {result['answer']}",
                language=None,
            )
            evidence = dict(result.get("evidence", {}))
            provider = evidence.get("provider", result.get("answer_source", "unknown"))
            response_ms = float(
                evidence.get("answer_call_ms") or evidence.get("latency_ms") or 0
            )
            st.caption(
                f"Response Time: {response_ms / 1000:.2f}s via {provider}"
            )
            if bool(evidence.get("fallback_used", False)):
                st.warning(
                    "Gemini tạm thời không khả dụng; kết quả này được trả bởi Qwen local "
                    f"({evidence.get('fallback_reason_code') or 'provider_unavailable'})."
                )
            image = _source_frame_image(
                context,
                str(result["batch_id"]),
                str(result["video_id"]),
                int(result["source_frame_idx"]),
            )
            if image is not None:
                st.image(image, caption="Frame bằng chứng duy nhất", width=640)
            if st.button("Ghim kết quả này", key="qa_pin_single_frame"):
                _pin_qa_prediction(result)
                st.rerun()
    elif candidates:
        selected_id = str(st.session_state.get("qa_selected_candidate_id") or "")
        selected_answer = next(
            (
                dict(value)
                for value in reversed(st.session_state.get("qa_answers", ()))
                if isinstance(value, dict) and value.get("candidate_id") == selected_id
            ),
            None,
        )
        if selected_answer is not None:
            status = str(selected_answer.get("status", "unanswerable"))
            if status == "unanswerable":
                st.warning(
                    "Frame đã chọn không cung cấp đủ bằng chứng để trả lời. "
                    "Hãy chọn candidate khác và bấm lại bước 2; không cần chạy lại KIS."
                )
            elif status == "provider_failed":
                st.error("Bộ máy VLM chưa trả lời được frame đã chọn; bạn có thể thử lại.")
            st.caption(
                f"Provider: {selected_answer.get('provider', 'unknown')} · "
                f"latency {float(selected_answer.get('latency_ms') or 0):.0f} ms"
            )

    answered_ids = {
        str(value.get("window_id"))
        for value in st.session_state.get("qa_predictions", ())
        if isinstance(value, dict) and value.get("window_id")
    }
    _render_qa_export(context, answered_window_ids=answered_ids)


def _trake_readiness(
    context: RuntimeContext,
    batch_id: str | None,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Return sanitized TRAKE readiness without loading either dense encoder."""

    warmup_snapshot = _warmup_snapshot(context)
    trake_warmup = _warmup_workflow_snapshot(warmup_snapshot, "trake")
    if warmup_snapshot is not None and not bool(trake_warmup.get("usable", False)):
        return False, {
            "runtime": {
                "state": str(trake_warmup.get("state", "warming")),
                "reason_code": str(trake_warmup.get("reason_code", "runtime_warming")),
            }
        }

    pipeline = getattr(context, "trake_pipeline", None)
    if pipeline is None or batch_id is None:
        return False, {
            "visual_vector": {
                "state": "not_provisioned",
                "reason_code": "trake_pipeline_unavailable",
            }
        }
    probe = getattr(pipeline, "channel_readiness", None)
    if not callable(probe):
        return True, {}
    try:
        usable, statuses = probe(batch_id)
    except Exception:
        return False, {
            "visual_vector": {
                "state": "service_unavailable",
                "reason_code": "readiness_probe_failed",
            }
        }
    safe: dict[str, dict[str, Any]] = {}
    if isinstance(statuses, dict):
        for name, status in statuses.items():
            if not isinstance(status, dict):
                continue
            safe[str(name)] = {
                field: status.get(field)
                for field in (
                    "state",
                    "applicability",
                    "reason_code",
                    "count",
                    "coverage",
                    "expected_dimension",
                    "remediation_id",
                )
                if field in status
            }
    return bool(usable), safe


def _trake_response_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", by_alias=True)
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        return {"results": [], "channel_statuses": {}}
    results = payload.get("results")
    payload["results"] = list(results) if isinstance(results, (list, tuple)) else []
    statuses = payload.get("channel_statuses")
    payload["channel_statuses"] = dict(statuses) if isinstance(statuses, dict) else {}
    return payload


def _trake_result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project alignments onto safe Vietnamese table columns."""

    rows: list[dict[str, Any]] = []
    for item in results[:100]:
        matches = item.get("stage_matches", ()) if isinstance(item, dict) else ()
        if not isinstance(matches, (list, tuple)):
            matches = ()
        frames = [
            int(match["source_frame_idx"])
            for match in matches
            if isinstance(match, dict)
            and isinstance(match.get("source_frame_idx"), int)
            and not isinstance(match.get("source_frame_idx"), bool)
        ]
        rows.append(
            {
                "Hạng": item.get("rank"),
                "Video": item.get("video_id"),
                "Khung nguồn theo giai đoạn": " → ".join(str(frame) for frame in frames),
                "Độ tin cậy": item.get("confidence_score"),
            }
        )
    return rows


def _trake_frame_image(
    context: RuntimeContext,
    batch_id: str,
    video_id: str,
    source_frame_idx: int,
) -> bytes | None:
    try:
        return _source_frame_image(context, batch_id, video_id, source_frame_idx)
    except Exception:
        return None


def _trake_csv_bytes(
    context: RuntimeContext,
    *,
    query_id: str,
    batch_id: str,
    stage_count: int,
    results: list[dict[str, Any]],
) -> tuple[str, bytes]:
    """Validate canonical source frames and build the no-header download in memory."""

    repository = getattr(context, "artifact_repository", None)
    if repository is None:
        raise TRAKESubmissionValidationError("không có ánh xạ khung hình chuẩn để xuất")
    canonical_check, evidence, timestamp_lookup = _submission_validation_context(
        context, batch_id, results
    )

    data = serialize_trake_submission(
        results,
        stage_count=stage_count,
        batch_id=batch_id,
        is_batch_active=context.ledger.is_batch_active,
        is_canonical_source_frame=canonical_check,
        get_canonical_timestamp_ms=timestamp_lookup,
        frame_index_evidence=TRAKEFrameIndexEvidence(evidence.ordinal_to_source),
    )
    return trake_submission_filename(query_id), data


def _invalidate_trake_results(state: Any | None = None) -> None:
    """Clear a TRAKE result whenever its query or stage definition changes."""

    target = st.session_state if state is None else state
    target["trake_response"] = None
    target["trake_results"] = []
    target["trake_query_id"] = None
    target["trake_result_stage_count"] = None
    target["trake_export_cache"] = None
    target.pop("trake_selected_candidate", None)


def _render_trake(context: RuntimeContext) -> None:
    """Render the Vietnamese TRAKE stage-alignment workflow."""

    st.subheader("TRAKE · Căn chỉnh chuỗi sự kiện theo thời gian")
    st.caption(
        "Mô tả toàn cảnh và từ 2 đến 20 giai đoạn theo đúng thứ tự xảy ra. "
        "Mỗi video chỉ giữ một chuỗi khung hình tốt nhất."
    )
    batch_id = _selected_batch_id(context)
    st.caption(f"Batch hiện hành: `{batch_id or 'chưa chọn'}`")
    st.text_area(
        "Mô tả toàn bộ sự kiện",
        key="trake_global_description",
        on_change=_invalidate_trake_results,
        placeholder="Ví dụ: Một người bước vào cửa hàng, chọn sản phẩm rồi thanh toán và rời đi.",
    )
    trake_language_col, trake_override_col = st.columns((1, 2))
    with trake_language_col:
        trake_query_language = st.selectbox(
            "Ngôn ngữ mô tả TRAKE",
            ("auto", "vi", "en"),
            format_func=lambda value: {
                "auto": "Tự động",
                "vi": "Tiếng Việt",
                "en": "English",
            }[value],
            key="trake_query_language",
            on_change=_invalidate_trake_results,
        )
    with trake_override_col:
        trake_edit_translation = st.checkbox(
            "Chỉnh từng bản dịch tiếng Anh",
            key="trake_translation_edit",
            on_change=_invalidate_trake_results,
        )
    if trake_edit_translation:
        st.text_input(
            "Mô tả toàn cảnh bằng tiếng Anh",
            key="trake_global_description_en",
            on_change=_invalidate_trake_results,
            placeholder="Để trống để dùng bộ dịch offline",
        )

    stage_count = max(2, min(20, int(st.session_state.trake_stage_count)))
    st.session_state.trake_stage_count = stage_count
    st.markdown("**Các giai đoạn theo thứ tự**")
    for stage_index in range(1, stage_count + 1):
        key = f"trake_stage_{stage_index}"
        st.session_state.setdefault(key, "")
        st.text_input(
            f"Giai đoạn {stage_index}",
            key=key,
            on_change=_invalidate_trake_results,
            placeholder=f"Mô tả hành động ở giai đoạn {stage_index}",
        )
        if trake_edit_translation:
            english_key = f"trake_stage_en_{stage_index}"
            st.session_state.setdefault(english_key, "")
            st.text_input(
                f"Stage {stage_index} · English",
                key=english_key,
                on_change=_invalidate_trake_results,
                placeholder="Để trống để dùng bộ dịch offline",
            )
    add_col, remove_col, top_k_col = st.columns((1, 1, 2))
    with add_col:
        if st.button(
            "Thêm giai đoạn",
            key="trake_add_stage",
            disabled=stage_count >= 20,
        ):
            _invalidate_trake_results()
            st.session_state.trake_stage_count = stage_count + 1
            st.session_state.setdefault(f"trake_stage_{stage_count + 1}", "")
            st.session_state.setdefault(f"trake_stage_en_{stage_count + 1}", "")
            st.rerun()
    with remove_col:
        if st.button(
            "Xóa giai đoạn cuối",
            key="trake_remove_stage",
            disabled=stage_count <= 2,
        ):
            _invalidate_trake_results()
            st.session_state.pop(f"trake_stage_{stage_count}", None)
            st.session_state.pop(f"trake_stage_en_{stage_count}", None)
            st.session_state.trake_stage_count = stage_count - 1
            st.rerun()
    with top_k_col:
        top_k = st.slider(
            "Số video trả về",
            1,
            100,
            100,
            key="trake_top_k",
            on_change=_invalidate_trake_results,
        )

    warm_ready = _warmup_workflow_usable(context, "trake")
    trake_ready, channel_statuses = _trake_readiness(context, batch_id)
    search_enabled = warm_ready and trake_ready and batch_id is not None
    if not warm_ready:
        st.info("Bộ máy truy xuất đang được làm nóng; nút tìm TRAKE sẽ tự mở khóa khi sẵn sàng.")
    elif not trake_ready:
        reasons = [
            _vi_status(status.get("reason_code") or status.get("state"))
            for status in channel_statuses.values()
        ]
        st.warning("TRAKE chưa sẵn sàng" + (f": {', '.join(reasons)}" if reasons else "."))

    if st.button(
        "Tìm chuỗi TRAKE",
        key="trake_search",
        type="primary",
        disabled=not search_enabled,
    ):
        # A failed replacement query must not leave an old result exportable.
        _invalidate_trake_results()
        global_description = str(st.session_state.trake_global_description).strip()
        descriptions = [
            str(st.session_state.get(f"trake_stage_{index}", "")).strip()
            for index in range(1, stage_count + 1)
        ]
        if not global_description or any(not value for value in descriptions):
            st.warning("Hãy nhập mô tả toàn cảnh và đầy đủ nội dung cho mọi giai đoạn.")
        else:
            search_started = time.monotonic()
            request: TRAKEQueryRequest | None = None
            try:
                batches = _validated_batch_filters(context, (batch_id,))
                if len(batches) != 1:
                    raise ValueError("TRAKE requires exactly one batch")
                request = TRAKEQueryRequest(
                    query_id=str(uuid4()),
                    global_description=global_description,
                    stages=tuple(
                        TRAKEStage(
                            stage_index=index,
                            description=description,
                            description_en=(
                                str(
                                    st.session_state.get(
                                        f"trake_stage_en_{index}", ""
                                    )
                                ).strip()
                                or None
                            ),
                        )
                        for index, description in enumerate(descriptions, start=1)
                    ),
                    batch_filters=batches,
                    top_k=top_k,
                    query_language=trake_query_language,
                    global_description_en=(
                        str(st.session_state.get("trake_global_description_en", "")).strip()
                        or None
                    ),
                )
                with st.spinner("Đang truy xuất video và căn chỉnh chuỗi giai đoạn..."):
                    response = context.trake_pipeline.search(
                        request,
                        default_batch_id=batch_id,
                    )
                public = _trake_response_dict(response)
                st.session_state.trake_response = public
                st.session_state.trake_results = public["results"]
                st.session_state.trake_query_id = request.query_id
                st.session_state.trake_result_stage_count = len(request.stages)
            except Exception as error:
                _report_ui_error(
                    error,
                    workflow="trake",
                    stage="search",
                    batch_id=batch_id,
                    query_or_session_id=(request.query_id if request is not None else None),
                    started_at=search_started,
                )

    _render_query_processing(
        getattr(context.trake_pipeline, "last_query_processing", {})
    )

    results = list(st.session_state.trake_results or [])
    if not results:
        if st.session_state.trake_query_id is None:
            st.caption("Chưa có truy vấn TRAKE nào được thực hiện.")
        else:
            st.info("Không tìm thấy video có chuỗi giai đoạn phù hợp.")
        return

    st.dataframe(_trake_result_rows(results), width="stretch", hide_index=True)
    choices = {
        f"{item.get('batch_id', batch_id)}:{item.get('video_id')}": item
        for item in results
        if isinstance(item, dict) and item.get("video_id")
    }
    if not choices:
        st.warning("Kết quả TRAKE không chứa ứng viên hợp lệ để hiển thị.")
        return
    selected_key = st.session_state.get("trake_selected_candidate")
    if selected_key not in choices:
        st.session_state["trake_selected_candidate"] = next(iter(choices))
    selected_id = st.selectbox(
        "Ứng viên để xem dòng thời gian",
        tuple(choices),
        key="trake_selected_candidate",
        format_func=lambda value: (
            f"#{choices[value].get('rank')} · {choices[value].get('video_id')}"
        ),
    )
    selected = choices[selected_id]
    matches = selected.get("stage_matches", ())
    if isinstance(matches, (list, tuple)) and matches:
        st.markdown("**Dòng thời gian ngang**")
        valid_matches = [match for match in matches if isinstance(match, dict)]
        st.markdown(
            render_hud_timeline_strip(
                [
                    int(match.get("stage_index", index + 1))
                    for index, match in enumerate(valid_matches)
                ]
            ),
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(valid_matches), 3):
            row_matches = valid_matches[row_start : row_start + 3]
            columns = st.columns(len(row_matches))
            for absolute_index, (column, match) in enumerate(
                zip(columns, row_matches, strict=True), start=row_start
            ):
                stage_index = int(match.get("stage_index", absolute_index + 1))
                source_frame_idx = int(match.get("source_frame_idx", 0))
                timestamp_ms = int(match.get("timestamp_ms", 0))
                arrow = " →" if absolute_index < len(valid_matches) - 1 else ""
                with column:
                    with st.container(
                        key=hud_container_key(
                            "trake-stage",
                            selected.get("batch_id", batch_id),
                            selected.get("video_id", ""),
                            stage_index,
                        ),
                        border=True,
                    ):
                        st.markdown(f"**E{stage_index} · Giai đoạn {stage_index}{arrow}**")
                        image = _trake_frame_image(
                            context,
                            str(selected.get("batch_id", batch_id)),
                            str(selected.get("video_id", "")),
                            source_frame_idx,
                        )
                        if image is not None:
                            st.image(image, width="stretch")
                        else:
                            st.caption("Không có thumbnail")
                        st.markdown(
                            render_hud_readout(
                                "SOURCE FRAME",
                                source_frame_idx,
                                tone="amber",
                            ),
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{timestamp_ms / 1000:.2f} giây")

    query_id = str(st.session_state.trake_query_id or "trake")
    export_fingerprint = _result_fingerprint(
        workflow="trake",
        batch_id=str(batch_id),
        rows=results,
    )
    cached_export = st.session_state.get("trake_export_cache")
    try:
        if (
            not isinstance(cached_export, dict)
            or cached_export.get("fingerprint") != export_fingerprint
        ):
            filename, data = _trake_csv_bytes(
                context,
                query_id=query_id,
                batch_id=str(batch_id),
                stage_count=int(st.session_state.trake_result_stage_count or stage_count),
                results=results,
            )
            cached_export = {
                "fingerprint": export_fingerprint,
                "filename": filename,
                "data": data,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            st.session_state.trake_export_cache = cached_export
        filename = str(cached_export["filename"])
        data = cached_export["data"]
    except Exception as error:
        st.session_state.trake_export_cache = None
        _report_ui_error(
            error,
            workflow="trake",
            stage="export",
            batch_id=batch_id,
            query_or_session_id=query_id,
        )
    else:
        st.download_button(
            "Tải CSV TRAKE đã kiểm tra",
            data=data,
            file_name=filename,
            mime="text/csv",
            key="trake_download",
        )
        if st.button("Lưu CSV TRAKE trên máy chủ", key="trake_save_server"):
            output_dir = Path(context.config["competition"]["output_dir"])
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir
            try:
                target = atomic_write_bytes(output_dir / filename, data)
            except OSError as error:
                _report_ui_error(
                    error,
                    workflow="trake",
                    stage="server_save",
                    batch_id=batch_id,
                    query_or_session_id=query_id,
                )
            else:
                st.success(f"Đã lưu `{target.name}`")


def render_app(context: RuntimeContext) -> None:
    """Render all three workflows over an injected, already-constructed runtime."""

    st.set_page_config(
        page_title="AIC 2026 - Truy xuất video đa phương thức",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_cyberpunk_theme()
    try:
        _initialize_state()
        st.markdown(
            render_hud_header(
                "SYSTEM ONLINE // AIC 2026",
                "Truy xuất video đa phương thức",
                "Textual KIS · Visual Q&A · TRAKE temporal alignment",
            ),
            unsafe_allow_html=True,
        )
        _render_warmup_sidebar(context)
        _render_launch_identity(context)
        _render_readiness(context)
        workflows = ("kis", "qa", "trake")
        if st.session_state.get("active_workflow") not in workflows:
            st.session_state.active_workflow = "kis"
        active = st.segmented_control(
            "Quy trình",
            workflows,
            key="active_workflow",
            format_func=lambda value: {
                "kis": "[ KIS ENGINE ]",
                "qa": "[ VISUAL QA ]",
                "trake": "[ TRAKE TIMELINE ]",
            }[value],
            selection_mode="single",
        )
        if active == "qa":
            _render_qa(context)
        elif active == "trake":
            _render_trake(context)
        else:
            _render_kis(context)
    except Exception as error:
        _report_ui_error(
            error,
            workflow=_active_workflow(),
            stage="render",
            batch_id=_selected_batch_id(context),
        )


def main() -> None:
    """Run the app, allowing AppTest to inject a side-effect-free runtime."""

    _install_windows_asyncio_log_filter()
    injected = st.session_state.get("_aic_runtime_override")
    context = injected if injected is not None else _runtime()
    if injected is None:
        logging_config = context.config.get("logging", {})
        configure_app_error_logging(
            logging_config.get("app_error_file", "outputs/logs/app_error.log"),
            rotation=logging_config.get("rotation", "10 MB"),
        )
    render_app(context)


if __name__ == "__main__":
    main()
