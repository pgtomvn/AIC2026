"""Interactive Textual KIS, Visual Q&A, and TRAKE competition console."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
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
    pin_widget_key,
    render_result_gallery,
    result_identity,
)
from app.http_backend import BackendError, build_http_runtime  # noqa: E402
from src import PROJECT_ROOT  # noqa: E402
from src.codabench_query_id import (  # noqa: E402
    codabench_query_filename,
    codabench_query_id_collision_key,
    codabench_query_id_sort_key,
    normalize_codabench_query_id,
    parse_codabench_query_filename,
)
from src.contracts import (  # noqa: E402
    CompetitionSpec,
    QACandidateRequest,
    QACompetitionSpec,
    QAFrameInjectionRequest,
    QAQueryRequest,
    QASingleFrameAnswerRequest,
    QueryRequest,
    TRAKEQueryRequest,
    TRAKEStage,
)
from src.qa_submission import serialize_qa_submission  # noqa: E402
from src.query.entity_extractor import (  # noqa: E402
    EntityConstraint,
    EntityExtractor,
)
from src.runtime import RuntimeContext  # noqa: E402
from src.runtime_contract import source_contract_fingerprint  # noqa: E402
from src.submission import (  # noqa: E402
    FrameIndexEvidence,
    format_qa_submission_row,
)
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

_ENTITY_EXTRACTOR = EntityExtractor()
_EXPECTED_RUNTIME_CONTRACT_VERSION = 6
_EXPECTED_RUNTIME_CONTRACT_FEATURES = frozenset(
    {
        "entity_constraints.kis",
        "entity_constraints.qa",
        "entity_constraints.trake",
        "qa.frame_injection",
        "qa.top_k_50",
        "kis.adaptive_rrf",
        "kis.spatial_color_rerank",
        "kis.temporal_adjacency",
        "kis.relative_relation_graph",
        "kis.soft_negation",
        "kis.strict_color_binding",
        "kis.experimental_cross_modal_calibration",
        "dataset.multimodal_coverage",
        "kis.full_spatial_ontology",
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
    intent = str(value.get("intent") or "visual_dense")
    effective_k = value.get("effective_rrf_k")
    weights = value.get("effective_channel_weights")
    if intent in {"visual_dense", "lexical_dominant", "spatial_color"}:
        detail = f"Intent: {intent}"
        if isinstance(effective_k, int) and not isinstance(effective_k, bool):
            detail += f" · RRF k={effective_k}"
        if isinstance(weights, dict) and weights:
            detail += " · " + ", ".join(
                f"{name}={float(weight):.2f}"
                for name, weight in sorted(weights.items())
                if isinstance(weight, (int, float)) and not isinstance(weight, bool)
            )
        st.caption("⌁ " + detail)
    constraints = [
        *(
            value.get("spatial_constraints")
            if isinstance(value.get("spatial_constraints"), list)
            else []
        ),
        *(
            value.get("color_constraints")
            if isinstance(value.get("color_constraints"), list)
            else []
        ),
    ]
    if constraints:
        labels = []
        for constraint in constraints[:16]:
            if not isinstance(constraint, dict):
                continue
            subject = str(constraint.get("subject") or "?")
            target = constraint.get("zone", constraint.get("color", "?"))
            state = "BOUND" if constraint.get("bound") else "DIAGNOSTIC"
            labels.append(f"[{subject} → {target} · {state}]")
        if labels:
            st.caption("▦ Spatial/Color: " + " ".join(labels))
    relation_labels = []
    for constraint in value.get("relative_constraints", ()):
        if not isinstance(constraint, dict):
            continue
        third = (
            f" & {constraint.get('second_object', '?')}"
            if constraint.get("arity") == 3
            else ""
        )
        kind = (
            " · HEURISTIC 2D"
            if constraint.get("verification_type") == "heuristic_2d"
            else ""
        )
        relation_labels.append(
            f"[{constraint.get('subject', '?')} {constraint.get('relation', '?')} "
            f"{constraint.get('object', '?')}{third}{kind}]"
        )
    for constraint in value.get("negation_constraints", ()):
        if not isinstance(constraint, dict):
            continue
        color = f" {constraint['color']}" if constraint.get("color") else ""
        relation_labels.append(f"[KHÔNG:{color} {constraint.get('subject', '?')}]")
    if relation_labels:
        st.caption("⌖ Relation/Negation: " + " ".join(relation_labels[:16]))
    ambiguous = value.get("ambiguous_relations")
    if isinstance(ambiguous, list) and ambiguous:
        st.caption(
            "⌁ Quan hệ BETWEEN cần nêu rõ vật ở giữa và đủ hai vật mốc; "
            "cụm mơ hồ hiện chỉ dùng cho truy vấn dense/lexical."
        )
    unsupported = value.get("unsupported_relations")
    if isinstance(unsupported, list) and unsupported:
        names = ", ".join(
            str(item.get("relation"))
            for item in unsupported
            if isinstance(item, dict) and item.get("relation")
        )
        if names:
            st.warning(
                "Quan hệ chưa hỗ trợ chỉ được ghi diagnostics, không tác động ranking: " + names
            )

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
    entities = value.get("entities")
    if isinstance(entities, list) and entities:
        labels = []
        for entity in entities[:16]:
            if not isinstance(entity, dict) or not isinstance(entity.get("text"), str):
                continue
            mode = "BẮT BUỘC" if entity.get("mode") == "required" else "TÙY CHỌN"
            matched = " · MATCH" if entity.get("boost_applied") else ""
            labels.append(f"[{entity['text']} · {mode}{matched}]")
        if labels:
            st.caption("🏷️ Phân tích thực thể: " + " ".join(labels))
    required_status = value.get("required_entity_status")
    if required_status == "required_entity_evidence_unavailable":
        st.warning("Không có kênh OCR/ASR/metadata khả dụng để xác minh thực thể bắt buộc.")
    elif required_status == "required_entity_no_match":
        st.info("Không có video thỏa đồng thời tất cả thực thể bắt buộc.")


def _render_entity_controls(
    text: str,
    *,
    state_key: str,
    scope: str,
    on_change: Any = None,
) -> tuple[EntityConstraint, ...]:
    """Render stable Optional/Required controls for deterministic entities."""

    entities = _ENTITY_EXTRACTOR.extract(text) if isinstance(text, str) else ()
    raw_modes = st.session_state.get(state_key, {})
    modes = dict(raw_modes) if isinstance(raw_modes, dict) else {}
    prefix = f"{scope}:"
    active_keys = {f"{prefix}{entity.identity}" for entity in entities}
    modes = {
        key: value
        for key, value in modes.items()
        if not key.startswith(prefix) or key in active_keys
    }
    constraints: list[EntityConstraint] = []
    if entities:
        st.caption("🏷️ Phân tích thực thể · nhấn badge để đổi Tùy chọn/Bắt buộc")
        columns = st.columns(min(4, len(entities)))
        for index, entity in enumerate(entities):
            identity_key = f"{prefix}{entity.identity}"
            mode = "required" if modes.get(identity_key) == "required" else "optional"
            constraints.append(EntityConstraint(entity.text, mode))
            digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:12]
            label = (
                f"🏷️ {entity.text} · {'BẮT BUỘC' if mode == 'required' else 'TÙY CHỌN'}"
            )
            with columns[index % len(columns)]:
                if st.button(label, key=f"entity_mode_{digest}", width="stretch"):
                    modes[identity_key] = "optional" if mode == "required" else "required"
                    st.session_state[state_key] = modes
                    if callable(on_change):
                        on_change()
                    st.rerun()
    st.session_state[state_key] = modes
    return tuple(constraints)


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
    try:
        image = resolver.resolve(frame)
    except Exception:
        return None
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
        "kis_working_fingerprint": None,
        "kis_scrub_anchor": None,
        "kis_scrub_frames": [],
        "kis_scrub_offset": 0,
        "kis_last_failure": None,
        "kis_query_language": "auto",
        "kis_translation_edit": False,
        "kis_english_override": "",
        "kis_entity_modes": {},
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
        "qa_entity_modes": {},
        "qa_answer_drafts": {},
        "qa_final_answer_candidate_id": None,
        "qa_pending_answer_update": None,
        "vqa_final_answer_input": "",
        "vqa_injected_frame": None,
        "vqa_selected_candidate": None,
        "vqa_manual_video_id": "",
        "vqa_manual_frame_idx": 0,
        "qa_top_k": 5,
        "qa_candidate_page": 1,
        "qa_injection_notice": None,
        "trake_results": [],
        "trake_pinned": [],
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
        "trake_entity_modes": {},
        "trake_frame_overrides": {},
        "trake_adjust_stage": 1,
        "codabench_staging": {},
        "codabench_staging_revision": 0,
        "codabench_staging_updated_at": None,
        "codabench_staging_hydrated": False,
        "codabench_staging_error": None,
        "codabench_entry_errors": {},
        "codabench_bundle_cache": None,
        "codabench_bundle_mode": "all",
        "codabench_query_id_kis": "",
        "codabench_query_id_qa": "",
        "codabench_query_id_trake": "",
    }
    for name, value in defaults.items():
        st.session_state.setdefault(name, value)


def _request_contract_fields(value: type[Any]) -> frozenset[str]:
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, dict):
        return frozenset(str(name) for name in dataclass_fields)
    model_fields = getattr(value, "model_fields", None)
    if isinstance(model_fields, dict):
        return frozenset(str(name) for name in model_fields)
    return frozenset()


def _runtime_contract_issues(context: RuntimeContext) -> tuple[str, ...]:
    """Return stable reasons why this process must be restarted."""

    issues: list[str] = []
    required_fields = (
        (QueryRequest, "entity_constraints"),
        (QAQueryRequest, "entity_constraints"),
        (QACandidateRequest, "entity_constraints"),
        (QAFrameInjectionRequest, "source_frame_idx"),
        (TRAKEQueryRequest, "global_entity_constraints"),
        (TRAKEStage, "entity_constraints"),
    )
    for request_type, field_name in required_fields:
        if field_name not in _request_contract_fields(request_type):
            issues.append(f"request_field_missing:{request_type.__name__}.{field_name}")

    if not bool(getattr(context, "is_remote", False)):
        return tuple(issues)

    capabilities = getattr(context, "capabilities", None)
    runtime_contract = (
        capabilities.get("runtime_contract") if isinstance(capabilities, dict) else None
    )
    if not isinstance(runtime_contract, dict):
        issues.append("api_contract_missing")
        return tuple(issues)
    if runtime_contract.get("version") != _EXPECTED_RUNTIME_CONTRACT_VERSION:
        issues.append("api_contract_version_mismatch")
    features = runtime_contract.get("features")
    feature_set = {
        str(value) for value in features
    } if isinstance(features, list) else set()
    if not _EXPECTED_RUNTIME_CONTRACT_FEATURES.issubset(feature_set):
        issues.append("api_contract_feature_missing")

    launch_hash = os.environ.get("AIC_SOURCE_CONTRACT_SHA256", "").casefold()
    api_hash = str(runtime_contract.get("source_contract_sha256") or "").casefold()
    if launch_hash:
        current_hash = source_contract_fingerprint().casefold()
        if current_hash != launch_hash:
            issues.append("source_contract_drift")
        if api_hash != launch_hash:
            issues.append("api_source_contract_mismatch")
    elif os.environ.get("AIC_LAUNCH_CONTRACT") == "run_app-v2":
        issues.append("launch_source_contract_missing")
    return tuple(dict.fromkeys(issues))


def _render_runtime_contract_block(context: RuntimeContext) -> bool:
    """Block mixed-version processes without logging a false application crash."""

    issues = _runtime_contract_issues(context)
    if not issues:
        return False
    st.markdown(
        render_hud_header(
            "[ SYSTEM LOCK ] // RUNTIME CONTRACT",
            "Runtime cần được khởi động lại",
            "UI, API hoặc request schema đang thuộc các phiên bản khác nhau.",
        ),
        unsafe_allow_html=True,
    )
    st.error(
        "Mã `runtime_contract_mismatch`: yêu cầu chưa được gửi và dữ liệu phiên/staging "
        "không bị xóa. Hãy dừng stack AIC hiện tại rồi chạy lại launcher chuẩn."
    )
    profile = str(getattr(context, "config", {}).get("profile", "laptop"))
    batch_id = str(getattr(context, "selected_batch_id", "aic25-b1-reference"))
    st.code(
        f".\\scripts\\run_app.ps1 -Profile {profile} -BatchId {batch_id} "
        "-ApiPort 8080 -Port 8504",
        language="powershell",
    )
    st.caption("Chẩn đoán: " + ", ".join(f"`{value}`" for value in issues))
    return True


_RECOVERY_PREFIXES = ("kis_", "qa_", "trake_", "query_", "vqa_")


def _reset_operator_session() -> None:
    """Clear volatile workflow state while preserving audited submission artifacts."""

    for key in tuple(st.session_state):
        if key.startswith(_RECOVERY_PREFIXES):
            st.session_state.pop(key, None)
    st.session_state.active_workflow = "kis"
    st.session_state.qa_external_transfer_consent = False
    st.session_state["_aic_recovery_notice"] = True


def _render_recovery_control() -> None:
    with st.sidebar:
        st.button(
            "🔄 Khôi phục hệ thống (Reset Session State)",
            key="aic_reset_session_state",
            on_click=_reset_operator_session,
            width="stretch",
            help=(
                "Xóa state tìm kiếm tạm thời nhưng giữ Codabench staging và bundle. "
                "Thao tác này không khởi động lại backend hoặc CUDA."
            ),
        )
        if st.session_state.pop("_aic_recovery_notice", False):
            st.success("Đã khôi phục state giao diện; dữ liệu nộp bài vẫn được giữ nguyên.")


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
    st.session_state.kis_working_fingerprint = _kis_working_source_fingerprint()


def _clear_kis_pins() -> None:
    """Clear the pin source of truth and every associated checkbox key."""

    for key in tuple(st.session_state.get("kis_pin_widget_keys", ())):
        st.session_state.pop(str(key), None)
    st.session_state.kis_pin_widget_keys = []
    st.session_state.kis_pinned = []
    st.session_state.kis_export_cache = None
    st.session_state.kis_working_fingerprint = None


def _open_kis_scrubber(item: dict[str, Any]) -> None:
    """Select one authoritative KIS result for on-demand ordinal navigation."""

    try:
        identity = result_identity(item)
    except (KeyError, TypeError, ValueError):
        return
    st.session_state.kis_scrub_anchor = {
        "batch_id": identity[0],
        "video_id": identity[1],
        "source_frame_idx": identity[2],
        "rank": int(item.get("rank", 0) or 0),
    }
    st.session_state.kis_scrub_frames = []
    st.session_state.kis_scrub_offset = 0


def _pin_temporal_neighbor(
    anchor: dict[str, Any],
    frame: Any,
    offset: int,
) -> None:
    item = {
        "batch_id": str(frame.batch_id),
        "video_id": str(frame.video_id),
        "source_frame_idx": int(frame.source_frame_idx),
        "timestamp_ms": int(frame.timestamp_ms),
        "shot_id": str(frame.shot_id or f"frame:{frame.source_frame_idx}"),
        "provenance": {
            "source": "temporal_neighbor",
            "offset": int(offset),
            "anchor_video_id": str(anchor["video_id"]),
            "anchor_source_frame_idx": int(anchor["source_frame_idx"]),
        },
        "evidence": {
            "items": [
                {
                    "channel": "temporal_neighbor",
                    "identity_method": "canonical_ordinal",
                }
            ]
        },
    }
    identity = result_identity(item)
    widget_key = pin_widget_key(identity)
    st.session_state[widget_key] = True
    _change_kis_pin(identity, item, True, widget_key)


def _render_kis_temporal_scrubber(context: RuntimeContext) -> None:
    anchor = st.session_state.get("kis_scrub_anchor")
    if not isinstance(anchor, dict):
        return
    identity = (
        str(anchor.get("batch_id", "")),
        str(anchor.get("video_id", "")),
        int(anchor.get("source_frame_idx", -1)),
    )
    digest = hashlib.sha256(
        "\0".join((identity[0], identity[1], str(identity[2]))).encode("utf-8")
    ).hexdigest()[:16]
    with st.container(key=f"kis-temporal-scrubber-{digest}", border=True):
        st.markdown("### Temporal scrubber · ordinal canonical ±5")
        st.caption(
            f"Anchor {identity[1]} · source frame {identity[2]} · "
            "không dùng phép cộng/trừ source frame."
        )
        values = st.session_state.get("kis_scrub_frames")
        if not isinstance(values, list) or not values:
            try:
                with st.spinner("Đang tải metadata khung hình lân cận..."):
                    pairs = context.artifact_repository.adjacent_frames(
                        identity[0],
                        identity[1],
                        identity[2],
                        radius=5,
                    )
                values = [
                    {
                        "offset": int(offset),
                        "frame": frame,
                    }
                    for offset, frame in pairs
                ]
                st.session_state.kis_scrub_frames = values
            except Exception:
                st.warning("Không thể tải vùng lân cận canonical cho khung hình này.")
                values = []
        quick = st.columns(5)
        for column, offset in zip(quick, (-2, -1, 0, 1, 2), strict=True):
            with column:
                if st.button(
                    f"{offset:+d}",
                    key=f"kis_scrub_quick_{digest}_{offset}",
                    width="stretch",
                ):
                    st.session_state.kis_scrub_offset = offset
                    st.rerun()
        selected_offset = st.slider(
            "Offset ordinal",
            min_value=-5,
            max_value=5,
            step=1,
            key="kis_scrub_offset",
        )
        selected = next(
            (
                value
                for value in values
                if isinstance(value, dict) and value.get("offset") == selected_offset
            ),
            None,
        )
        if selected is None:
            st.info("Offset này là alias trùng hoặc nằm ngoài biên video; không có frame riêng.")
        else:
            frame = selected["frame"]
            st.code(
                f"{frame.video_id},{frame.source_frame_idx} · "
                f"{frame.timestamp_ms / 1000:.3f}s · offset {selected_offset:+d}",
                language=None,
            )
            image = _source_frame_image(
                context,
                frame.batch_id,
                frame.video_id,
                frame.source_frame_idx,
            )
            if image is not None:
                st.image(image, width="stretch")
                if st.button(
                    "📌 Ghim khung lân cận này",
                    key=f"kis_scrub_pin_{digest}_{selected_offset}",
                    width="stretch",
                ):
                    _pin_temporal_neighbor(anchor, frame, selected_offset)
                    st.success("Đã thêm khung lân cận vào danh sách KIS đang ghim.")
            else:
                st.warning("Ảnh canonical của khung lân cận hiện không khả dụng.")
        if st.button("Đóng scrubber", key=f"kis_scrub_close_{digest}"):
            st.session_state.kis_scrub_anchor = None
            st.session_state.kis_scrub_frames = []
            st.rerun()



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


def _kis_working_source_fingerprint() -> str | None:
    pins = [
        dict(value)
        for value in st.session_state.get("kis_pinned", ())
        if isinstance(value, dict)
    ]
    batch_ids = {
        str(value.get("batch_id"))
        for value in pins
        if isinstance(value.get("batch_id"), str) and value.get("batch_id")
    }
    if not pins or len(batch_ids) != 1:
        return None
    batch_id = next(iter(batch_ids))
    return _result_fingerprint(
        workflow="kis",
        batch_id=batch_id,
        rows=_kis_export_snapshot(pins),
    )



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

    multimodal = report.get("multimodal", {})
    if isinstance(multimodal, dict) and multimodal:
        status_label = str(multimodal.get("status_label") or "").strip()
        full_multimodal = multimodal.get("full_multimodal", {})
        if status_label and isinstance(full_multimodal, dict):
            unavailable = tuple(
                str(value)
                for value in full_multimodal.get("unavailable_channels", ())
                if str(value).strip()
            )
            detail = status_label
            if unavailable:
                detail += " · Kênh chưa sẵn sàng: " + ", ".join(unavailable)
            if bool(full_multimodal.get("healthy", False)):
                st.success(detail)
            else:
                st.warning(detail)
        mm_status = str(multimodal.get("status", "unknown"))
        siglip = multimodal.get("siglip2", {})
        ocr = multimodal.get("ocr", {})
        asr = multimodal.get("asr", {})
        if all(isinstance(value, dict) for value in (siglip, ocr, asr)):
            st.caption(
                "MULTIMODAL // "
                f"{mm_status.upper()} · SigLIP2 "
                f"{int(siglip.get('canonical_frames', 0) or 0):,}/"
                f"{int(siglip.get('expected_frames', 0) or 0):,} · OCR scanned "
                f"{int(ocr.get('scanned_canonical_frames', 0) or 0):,}/"
                f"{int(ocr.get('expected_frames', 0) or 0):,} · ASR source "
                f"{int(asr.get('source_videos', 0) or 0):,}/"
                f"{int(multimodal.get('expected_videos', 0) or 0):,} · processed "
                f"{int(asr.get('completed_videos', 0) or 0):,} · zero-transcript "
                f"{int(asr.get('zero_transcript_videos', asr.get('no_audio_videos', 0)) or 0):,} · missing "
                f"{int(asr.get('missing_source_videos', 0) or 0):,}"
            )

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
    if previous_usability is not None:
        became_usable = any(
            not bool(previous[1]) and bool(current[1])
            for previous, current in zip(previous_usability, usability, strict=False)
        )
        if became_usable:
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
    results_col, refinement_col = st.columns((1, 1))
    with results_col:
        top_k = st.slider("Số kết quả KIS", 1, 100, 20, key="kis_top_k")
    with refinement_col:
        refine_raw = st.checkbox("Tinh chỉnh cục bộ từ video gốc", value=True, key="kis_refine_raw")

    runtime_ready = _warmup_search_usable(context)
    values = render_query_input(disabled=not runtime_ready)
    kis_entity_constraints = _render_entity_controls(
        str(values.get("text", "")),
        state_key="kis_entity_modes",
        scope="kis",
    )
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
                    mode=str(values.get("mode", "automatic")),
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
                    entity_constraints=kis_entity_constraints,
                )
                with st.spinner("Đang tìm trên các kênh vector và từ khóa..."):
                    st.session_state.kis_results = context.pipeline.search(
                        request, limit=top_k, refine_raw=refine_raw
                    )
                    st.session_state.kis_last_request = request
                    st.session_state.kis_history.append(values["text"])
                    st.session_state.kis_last_failure = None
            except (TypeError, ValueError):
                st.session_state.kis_last_failure = {
                    "code": "kis_request_invalid",
                    "recoverable": True,
                }
            except BackendError as error:
                st.session_state.kis_last_failure = {
                    "code": error.code,
                    "correlation_id": error.detail.get("correlation_id"),
                    "recoverable": error.recoverable,
                }
            except Exception as error:
                event_id = _report_ui_error(
                    error,
                    workflow="kis",
                    stage="search",
                    batch_id=(
                        batch_filters[0] if "batch_filters" in locals() and batch_filters else None
                    ),
                    query_or_session_id=(request.query_id if request is not None else None),
                    started_at=search_started,
                )
                st.session_state.kis_last_failure = {
                    "code": getattr(error, "code", "kis_search_failed"),
                    "correlation_id": (
                        getattr(error, "detail", {}).get("correlation_id")
                        if isinstance(getattr(error, "detail", None), dict)
                        else event_id
                    ),
                }

    failure = st.session_state.get("kis_last_failure")
    if isinstance(failure, dict):
        correlation = str(failure.get("correlation_id") or "")
        st.warning(
            "Yêu cầu KIS chưa hoàn tất. Bạn có thể sửa truy vấn và thử lại"
            + (f" · Mã: `{correlation}`" if correlation else ".")
        )
        if st.button("Đóng cảnh báo", key="kis_dismiss_failure"):
            st.session_state.kis_last_failure = None
            st.rerun()

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
            on_send_to_qa=lambda item: _bridge_kis_result_to_qa(context, item),
            on_open_scrubber=_open_kis_scrubber,
        )

    if st.session_state.kis_results and st.session_state.kis_last_request is not None:
        if context.artifact_repository is not None:
            _render_kis_temporal_scrubber(context)

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
    unpin_columns = st.columns(min(4, len(st.session_state.kis_pinned)))
    for index, item in enumerate(list(st.session_state.kis_pinned), start=1):
        identity = result_identity(item)
        digest = hashlib.sha256(
            "\0".join((identity[0], identity[1], str(identity[2]))).encode("utf-8")
        ).hexdigest()[:12]
        with unpin_columns[(index - 1) % len(unpin_columns)]:
            if st.button(
                f"Bỏ ghim #{index}",
                key=f"kis_unpin_{digest}",
                width="stretch",
            ):
                _change_kis_pin(
                    identity,
                    item,
                    False,
                    pin_widget_key(identity),
                )
                st.rerun()

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
    selected_id = str(st.session_state.get("qa_selected_candidate_id") or "")
    selected_prediction = next(
        (
            item
            for item in predictions
            if str(item.get("window_id") or "") == selected_id
        ),
        None,
    )
    st.session_state.qa_result = selected_prediction or (predictions[0] if predictions else None)
    return snapshot


def _apply_injected_qa_snapshot(
    snapshot: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    """Replace volatile candidate state only after FastAPI accepted an injection."""

    _apply_qa_snapshot(snapshot)
    candidates = [
        dict(value)
        for value in snapshot.get("candidates", ())
        if isinstance(value, dict)
    ]
    if len(candidates) != 1:
        raise BackendError("backend_response_invalid")
    candidate = candidates[0]
    st.session_state.qa_selected_candidate_id = str(candidate["candidate_id"])
    st.session_state.vqa_selected_candidate = dict(candidate)
    st.session_state.vqa_injected_frame = {
        "batch_id": str(candidate["batch_id"]),
        "video_id": str(candidate["video_id"]),
        "source_frame_idx": int(candidate["source_frame_idx"]),
        "source": source,
    }
    st.session_state.qa_request = None
    st.session_state.qa_pinned = []
    st.session_state.qa_export_cache = None
    st.session_state.qa_answer_drafts = {}
    st.session_state.qa_final_answer_candidate_id = None
    st.session_state.qa_pending_answer_update = None
    st.session_state.vqa_final_answer_input = ""
    st.session_state.qa_candidate_page = 1
    st.session_state.qa_last_failure = None
    st.session_state.qa_injection_notice = {
        "video_id": str(candidate["video_id"]),
        "source_frame_idx": int(candidate["source_frame_idx"]),
        "source": source,
    }
    return candidate


def _inject_qa_frame(
    context: RuntimeContext,
    request: QAFrameInjectionRequest,
) -> dict[str, Any]:
    """Call the authoritative injection endpoint and apply its returned snapshot."""

    injector = getattr(context.qa_pipeline, "inject", None)
    if not callable(injector):
        raise BackendError("runtime_contract_mismatch")
    injector(request)
    snapshot = getattr(context.qa_pipeline, "last_snapshot", None)
    if not isinstance(snapshot, dict):
        raise BackendError("backend_response_invalid")
    return _apply_injected_qa_snapshot(snapshot, source=request.source)


def _bridge_kis_result_to_qa(context: RuntimeContext, result: dict[str, Any]) -> None:
    """Bridge one KIS result through FastAPI, then switch the active workflow."""

    last_request = st.session_state.get("kis_last_request")
    event_description = str(getattr(last_request, "original_text", "") or "").strip()
    batch_id = str(result.get("batch_id") or _selected_batch_id(context) or "")
    try:
        request = QAFrameInjectionRequest(
            query_id=str(uuid4()),
            batch_id=batch_id,
            video_id=str(result.get("video_id") or ""),
            source_frame_idx=int(result["source_frame_idx"]),
            source="kis_bridge",
            event_description=event_description,
        )
        _inject_qa_frame(context, request)
        st.session_state.qa_event_description = event_description
        st.session_state._pending_active_workflow = "qa"
        st.rerun()
    except (BackendError, TypeError, ValueError, KeyError) as error:
        st.session_state.kis_last_failure = {
            "code": getattr(error, "code", "qa_frame_injection_invalid"),
            "recoverable": True,
        }
        st.rerun()


def _qa_persisted_answer(candidate_id: str) -> str:
    for item in reversed(st.session_state.get("qa_predictions", ())):
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence", {})
        evidence = evidence if isinstance(evidence, dict) else {}
        if str(item.get("window_id") or evidence.get("candidate_id")) == candidate_id:
            answer = item.get("answer")
            return answer if isinstance(answer, str) else ""
    return ""


def _sync_qa_answer_draft(candidate_id: str) -> None:
    """Switch the editable answer without losing another candidate's draft."""

    drafts = dict(st.session_state.get("qa_answer_drafts", {}))
    pending = st.session_state.get("qa_pending_answer_update")
    if (
        isinstance(pending, dict)
        and pending.get("candidate_id") == candidate_id
        and isinstance(pending.get("answer"), str)
    ):
        answer = str(pending["answer"])
        drafts[candidate_id] = answer
        st.session_state.vqa_final_answer_input = answer
        st.session_state.qa_final_answer_candidate_id = candidate_id
        st.session_state.qa_pending_answer_update = None
        st.session_state.qa_answer_drafts = drafts
        return
    previous = st.session_state.get("qa_final_answer_candidate_id")
    if isinstance(previous, str) and previous:
        current = st.session_state.get("vqa_final_answer_input")
        if isinstance(current, str):
            drafts[previous] = current
    if previous != candidate_id:
        st.session_state.vqa_final_answer_input = drafts.get(
            candidate_id, _qa_persisted_answer(candidate_id)
        )
        st.session_state.qa_final_answer_candidate_id = candidate_id
    st.session_state.qa_answer_drafts = drafts


def _capture_qa_answer_draft() -> None:
    candidate_id = st.session_state.get("qa_final_answer_candidate_id")
    answer = st.session_state.get("vqa_final_answer_input")
    if not isinstance(candidate_id, str) or not isinstance(answer, str):
        return
    drafts = dict(st.session_state.get("qa_answer_drafts", {}))
    drafts[candidate_id] = answer
    st.session_state.qa_answer_drafts = drafts


def _set_qa_answer_draft(candidate_id: str, answer: str) -> None:
    drafts = dict(st.session_state.get("qa_answer_drafts", {}))
    drafts[candidate_id] = answer
    st.session_state.qa_answer_drafts = drafts
    st.session_state.qa_final_answer_candidate_id = candidate_id
    st.session_state.vqa_final_answer_input = answer


def _queue_qa_answer_draft(candidate_id: str, answer: str) -> None:
    """Queue a widget value produced after its current-run instantiation."""

    drafts = dict(st.session_state.get("qa_answer_drafts", {}))
    drafts[candidate_id] = answer
    st.session_state.qa_answer_drafts = drafts
    st.session_state.qa_final_answer_candidate_id = candidate_id
    st.session_state.qa_pending_answer_update = {
        "candidate_id": candidate_id,
        "answer": answer,
    }


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
    identity = (
        prediction["video_id"],
        prediction["source_frame_idx"],
        prediction["answer"],
    )
    pinned = {
        (item["video_id"], item["source_frame_idx"], item["answer"]): item
        for item in st.session_state.qa_pinned
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


def _build_qa_export_cache(
    context: RuntimeContext,
    *,
    rows: list[dict[str, Any]],
    batch_id: str,
    filename: str,
) -> dict[str, Any]:
    """Build one immutable QA CSV snapshot through the canonical validator."""

    request = st.session_state.qa_request
    if request is None or not hasattr(request, "answer_spec_id"):
        raise ValueError("the QA answer specification is unavailable")
    ranked = [dict(item, rank=index) for index, item in enumerate(rows, 1)]
    spec = QACompetitionSpec.for_spec_id(request.answer_spec_id)
    canonical_check, evidence, _ = _submission_validation_context(context, batch_id, ranked)
    data = serialize_qa_submission(
        ranked,
        spec=spec,
        batch_id=batch_id,
        is_batch_active=context.ledger.is_batch_active,
        is_canonical_source_frame=canonical_check,
        frame_index_evidence=evidence,
    )
    return {
        "fingerprint": _result_fingerprint(workflow="qa", batch_id=batch_id, rows=ranked),
        "filename": filename,
        "data": data,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


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
                filename = f"visual-qa-{str(st.session_state.qa_search_id)[:8]}.csv"
                cached = _build_qa_export_cache(
                    context,
                    rows=rows,
                    batch_id=batch_id,
                    filename=filename,
                )
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
    qa_entity_constraints = _render_entity_controls(
        event_description,
        state_key="qa_entity_modes",
        scope="qa",
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
    submission_query_id = st.text_input(
        "Query ID nộp bài · Visual Q&A",
        key="codabench_query_id_qa",
        placeholder="Ví dụ: p1-15, 01, round1_01",
    )
    batch_id = _selected_batch_id(context)
    st.caption(f"Batch hiện hành: `{batch_id or 'chưa chọn'}`")

    top_k = st.select_slider(
        "Số lượng ứng viên Stage 1 (Top-K)",
        options=(5, 10, 15, 20, 30, 50),
        key="qa_top_k",
    )

    with st.expander("🛠️ Nạp khung hình thủ công / Nhập ID trực tiếp"):
        manual_video_id = st.text_input(
            "Tên Video (Video ID)",
            key="vqa_manual_video_id",
            placeholder="L01_V028",
        )
        manual_source_frame_idx = st.number_input(
            "Source Frame Index",
            min_value=0,
            step=1,
            key="vqa_manual_frame_idx",
            help="Đây là source_frame_idx canonical, không phải keyframe ordinal.",
        )
        manual_load = st.button(
            "🔍 Nạp & Xem trước khung hình",
            key="qa_inject_manual_frame",
            disabled=not bool(batch_id and manual_video_id.strip()),
            width="stretch",
        )
        if manual_load:
            try:
                injection = QAFrameInjectionRequest(
                    query_id=str(uuid4()),
                    batch_id=str(batch_id),
                    video_id=manual_video_id.strip(),
                    source_frame_idx=int(manual_source_frame_idx),
                    source="manual",
                    event_description=(
                        event_description.strip()
                        or "operator-selected canonical frame"
                    ),
                )
                with st.spinner("Đang xác minh canonical frame và ảnh nguồn..."):
                    _inject_qa_frame(context, injection)
                st.rerun()
            except (TypeError, ValueError):
                st.session_state.qa_last_failure = {
                    "code": "qa_frame_injection_invalid",
                    "recoverable": True,
                }
            except BackendError as error:
                st.session_state.qa_last_failure = {
                    "code": error.code,
                    "correlation_id": error.detail.get("correlation_id"),
                    "recoverable": error.recoverable,
                }

    injection_notice = st.session_state.get("qa_injection_notice")
    if isinstance(injection_notice, dict):
        source_label = (
            "KIS" if injection_notice.get("source") == "kis_bridge" else "nhập thủ công"
        )
        st.success(
            "Đã nạp frame "
            f"{injection_notice.get('video_id')} "
            f"(source frame {injection_notice.get('source_frame_idx')}) từ {source_label}."
        )

    retrieve_clicked = st.button(
        "1. Truy xuất khung hình – KIS Search",
        type="primary",
        key="qa_retrieve_candidates",
        disabled=not bool(event_description.strip() and batch_id),
    )
    if retrieve_clicked:
        request: QACandidateRequest | None = None
        try:
            request = QACandidateRequest(
                query_id=str(uuid4()),
                event_description=event_description,
                batch_filters=(batch_id,),
                top_k=int(top_k),
                query_language=qa_query_language,
                event_description_en=qa_event_description_en or None,
                entity_constraints=qa_entity_constraints,
            )
            with st.spinner(f"Đang truy xuất Top-{int(top_k)} khung hình bằng KIS..."):
                context.qa_pipeline.retrieve(request)
                snapshot = getattr(context.qa_pipeline, "last_snapshot", None)
                if snapshot is None:
                    raise BackendError("backend_response_invalid")
            st.session_state.qa_request = request
            st.session_state.qa_pinned = []
            st.session_state.qa_export_cache = None
            st.session_state.qa_answer_drafts = {}
            st.session_state.qa_final_answer_candidate_id = None
            st.session_state.qa_pending_answer_update = None
            st.session_state.vqa_final_answer_input = ""
            st.session_state.vqa_injected_frame = None
            st.session_state.vqa_selected_candidate = None
            st.session_state.qa_injection_notice = None
            st.session_state.qa_candidate_page = 1
            st.session_state.qa_last_failure = None
            _apply_qa_snapshot(snapshot)
        except (TypeError, ValueError):
            st.session_state.qa_last_failure = {
                "code": "qa_retrieval_request_invalid",
                "recoverable": True,
            }
        except BackendError as error:
            st.session_state.qa_last_failure = {
                "code": error.code,
                "correlation_id": error.detail.get("correlation_id"),
                "recoverable": error.recoverable,
            }
        except Exception as error:
            event_id = _report_ui_error(
                error,
                workflow="qa",
                stage="retrieve",
                batch_id=batch_id,
                query_or_session_id=(request.query_id if request is not None else None),
            )
            st.session_state.qa_last_failure = {
                "code": "qa_retrieval_unavailable",
                "correlation_id": event_id,
                "recoverable": True,
            }

    active_failure = st.session_state.get("qa_last_failure")
    if isinstance(active_failure, dict):
        failure_code = str(active_failure.get("code") or "qa_retrieval_unavailable")
        correlation = str(active_failure.get("correlation_id") or "")
        if failure_code == "qa_injected_frame_not_found":
            st.warning(
                "Không tìm thấy video/source frame canonical đã nhập trong batch hiện hành."
            )
        elif failure_code == "qa_candidate_media_unavailable":
            st.warning(
                "Canonical frame tồn tại nhưng ảnh nguồn đang thiếu hoặc hỏng. "
                "Không có frame lân cận nào được thay thế âm thầm."
            )
        else:
            st.warning(
                "Visual Q&A chưa hoàn tất (`"
                + failure_code
                + "`). Candidate và câu trả lời trước đó vẫn được giữ"
                + (f" · Mã: `{correlation}`" if correlation else ".")
            )
        if st.button("Đóng cảnh báo QA", key="qa_dismiss_active_failure"):
            st.session_state.qa_last_failure = None
            st.rerun()

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
        page_size = 12
        total_pages = max(1, (len(candidates) + page_size - 1) // page_size)
        current_page = int(st.session_state.get("qa_candidate_page", 1))
        current_page = min(max(current_page, 1), total_pages)
        st.session_state.qa_candidate_page = current_page
        if total_pages > 1:
            current_page = st.select_slider(
                "Trang candidate",
                options=tuple(range(1, total_pages + 1)),
                key="qa_candidate_page",
                format_func=lambda value: f"Trang {value}/{total_pages}",
            )
        else:
            st.caption("Trang candidate 1/1")
        page_start = (int(current_page) - 1) * page_size
        visible_candidates = candidates[page_start : page_start + page_size]
        selected_candidate = next(
            value for value in candidates if str(value["candidate_id"]) == str(selected)
        )
        image_candidates = list(visible_candidates)
        if all(
            str(value["candidate_id"]) != str(selected)
            for value in visible_candidates
        ):
            image_candidates.append(selected_candidate)
        candidate_images = {
            str(candidate["candidate_id"]): _source_frame_image(
                context,
                str(candidate["batch_id"]),
                str(candidate["video_id"]),
                int(candidate["source_frame_idx"]),
            )
            for candidate in image_candidates
        }
        snapshot_state = dict(st.session_state.get("qa_snapshot") or {})
        candidate_failures = snapshot_state.get("failures", {})
        if not isinstance(candidate_failures, dict):
            state_value = snapshot_state.get("state", {})
            candidate_failures = (
                dict(state_value.get("failures", {}))
                if isinstance(state_value, dict)
                else {}
            )
        retrieval_ms = float(
            dict(st.session_state.qa_snapshot or {})
            .get("stage_timings", {})
            .get("candidate_retrieval_ms", 0.0)
        )
        st.success(
            f"Đã truy xuất {len(candidates)} candidate trong {retrieval_ms:.0f} ms; chưa gọi VLM."
        )
        selected_origin = str(
            dict(selected_candidate.get("evidence", {})).get("origin") or "kis_retrieval"
        )
        with st.container(
            key=hud_container_key("qa-selected-preview", str(selected)),
            border=True,
        ):
            st.markdown(
                render_hud_readout(
                    "KHUNG HÌNH ĐƯỢC CHỌN ĐỂ TRẢ LỜI",
                    str(selected_candidate["video_id"]),
                    tone="green",
                ),
                unsafe_allow_html=True,
            )
            selected_image = candidate_images.get(str(selected))
            if selected_image is not None:
                st.image(selected_image, width="stretch")
            else:
                st.warning("Ảnh canonical của candidate đang chọn hiện không khả dụng.")
            st.caption(
                f"Nguồn: {selected_origin} · source frame "
                f"{selected_candidate['source_frame_idx']} · "
                f"{int(selected_candidate['timestamp_ms']) / 1000:.2f}s"
            )
        for offset in range(0, len(visible_candidates), 4):
            columns = st.columns(min(4, len(visible_candidates) - offset))
            for column, candidate in zip(
                columns, visible_candidates[offset : offset + 4], strict=True
            ):
                with column:
                    with st.container(
                        key=hud_container_key("qa-card", candidate["candidate_id"]),
                        border=True,
                    ):
                        candidate_id = str(candidate["candidate_id"])
                        image = candidate_images.get(candidate_id)
                        if image is not None:
                            st.image(image, width="stretch")
                        else:
                            st.warning("Ảnh canonical của candidate này hiện không khả dụng.")
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
                            disabled=is_selected or image is None,
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
        selected_candidate = next(
            value for value in candidates if str(value["candidate_id"]) == str(selected)
        )
        st.session_state.vqa_selected_candidate = dict(selected_candidate)

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

        selected_media_available = candidate_images.get(str(selected)) is not None
        if not selected_media_available or candidate_failures.get(str(selected)) == "media_unavailable":
            st.warning(
                "Candidate đang chọn không có ảnh canonical khả dụng. "
                "Hãy chủ động chọn candidate khác; hệ thống sẽ không tự thay frame."
            )
        answer_disabled = (
            not question.strip()
            or not selected_media_available
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
                query_language=qa_query_language,
                event_description_en=qa_event_description_en or None,
                entity_constraints=qa_entity_constraints,
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
                    answered = st.session_state.get("qa_result")
                    if (
                        isinstance(answered, dict)
                        and str(answered.get("window_id")) == str(selected)
                        and isinstance(answered.get("answer"), str)
                    ):
                        _set_qa_answer_draft(str(selected), str(answered["answer"]))
                except BackendError as error:
                    if error.snapshot is not None:
                        _apply_qa_snapshot(error.snapshot)
                    code = error.code
                    if code == "qa_candidate_media_unavailable":
                        st.warning(
                            "Ảnh canonical của candidate này không còn khả dụng. "
                            "Hãy chọn candidate khác để tiếp tục."
                        )
                    elif code == "florence_not_provisioned":
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

        _sync_qa_answer_draft(str(selected))
        st.text_input(
            "✍️ Câu trả lời chính thức (Tùy chỉnh / Xem trước)",
            key="vqa_final_answer_input",
            max_chars=256,
            on_change=_capture_qa_answer_draft,
            help=(
                "Câu trả lời sẽ được chuẩn hóa NFC, gộp khoảng trắng, bỏ dấu câu ngoài "
                "và chuyển thành chữ thường trước khi stage."
            ),
        )
        normalized_submission_id = _normalized_codabench_query_id(submission_query_id)
        manual_answer = str(st.session_state.get("vqa_final_answer_input", ""))
        selected_candidate_for_preview = next(
            (
                value
                for value in candidates
                if str(value.get("candidate_id")) == str(selected)
            ),
            None,
        )
        if manual_answer.strip() and selected_candidate_for_preview is not None:
            try:
                preview_answer = QACompetitionSpec.for_spec_id(
                    answer_spec_id
                ).normalize_answer(manual_answer, question)
                manual_preview = format_qa_submission_row(
                    str(selected_candidate_for_preview["video_id"]),
                    int(selected_candidate_for_preview["source_frame_idx"]),
                    preview_answer,
                )
            except (TypeError, ValueError):
                manual_preview = None
            if manual_preview is not None:
                st.caption("Submission Line Preview")
                st.code(manual_preview, language=None)
        staged_qa_entry: dict[str, Any] | None = None
        staged_qa_rows: list[list[str]] = []
        if normalized_submission_id is not None:
            staged_name = _codabench_filename(normalized_submission_id, "qa")
            candidate_entry = st.session_state.get("codabench_staging", {}).get(staged_name)
            if isinstance(candidate_entry, dict):
                staged_qa_entry = candidate_entry
                staged_qa_rows = _staged_row_values(candidate_entry)
        qa_stage_action = st.radio(
            "Cách ghi vào bảng xếp hạng QA",
            ("append", "update"),
            key="qa_stage_action",
            horizontal=True,
            format_func=lambda value: (
                "Ghim vào Rank tiếp theo" if value == "append" else "Cập nhật một Rank"
            ),
            disabled=not staged_qa_rows,
        )
        qa_update_rank: int | None = None
        if qa_stage_action == "update" and staged_qa_rows:
            qa_update_rank = st.selectbox(
                "Rank QA cần cập nhật",
                tuple(range(1, len(staged_qa_rows) + 1)),
                key="qa_stage_update_rank",
                format_func=lambda value: f"Rank #{value}",
            )
        manual_disabled = (
            normalized_submission_id is None
            or not question.strip()
            or not manual_answer.strip()
            or (qa_stage_action == "append" and len(staged_qa_rows) >= 100)
        )
        manual_label_id = normalized_submission_id or "<Query ID chưa hợp lệ>"
        manual_action_label = (
            f"Cập nhật Rank #{qa_update_rank}"
            if qa_stage_action == "update" and qa_update_rank is not None
            else "Ghim vào Rank tiếp theo"
        )
        if st.button(
            f"📌 {manual_action_label} (Query ID: {manual_label_id})",
            key="qa_confirm_manual_and_stage",
            disabled=manual_disabled,
            width="stretch",
        ):
            assert normalized_submission_id is not None
            selected_candidate = next(
                value for value in candidates if value["candidate_id"] == selected
            )
            manual_request = QASingleFrameAnswerRequest(
                candidate_id=str(selected),
                question=question,
                answer_spec_id=answer_spec_id,
                reasoning_policy="local_only",
                allow_external_frame_transfer=False,
                answer_engine="manual",
                manual_answer=manual_answer,
            )
            st.session_state.qa_request = QAQueryRequest(
                query_id=str(st.session_state.qa_search_id),
                event_description=event_description,
                question=question,
                mode="interactive",
                answer_spec_id=answer_spec_id,
                batch_filters=(str(selected_candidate["batch_id"]),),
                limit=1,
                reasoning_policy="local_only",
                allow_external_frame_transfer=False,
                query_language=qa_query_language,
                event_description_en=qa_event_description_en or None,
                entity_constraints=qa_entity_constraints,
            )
            try:
                context.qa_pipeline.answer_candidate(
                    str(st.session_state.qa_search_id), manual_request
                )
                snapshot = getattr(context.qa_pipeline, "last_snapshot", None)
                if snapshot is None:
                    raise BackendError("backend_response_invalid")
                _apply_qa_snapshot(snapshot)
                authoritative = st.session_state.get("qa_result")
                if (
                    not isinstance(authoritative, dict)
                    or str(authoritative.get("window_id")) != str(selected)
                    or not isinstance(authoritative.get("answer"), str)
                ):
                    raise BackendError("backend_response_invalid")
                _pin_qa_prediction(dict(authoritative))
                batch = str(authoritative["batch_id"])
                single_row_cache = _build_qa_export_cache(
                    context,
                    rows=[dict(authoritative)],
                    batch_id=batch,
                    filename=f"visual-qa-{str(st.session_state.qa_search_id)[:8]}.csv",
                )
                st.session_state.qa_export_cache = single_row_cache
                current_entry = {
                    "task": "qa",
                    "batch_id": batch,
                    "data": single_row_cache["data"],
                    "source_fingerprint": single_row_cache["fingerprint"],
                    "stage_count": None,
                    "row_count": 1,
                    "source_ranks": None,
                }
                staged_filename = _append_or_update_codabench_row(
                    context,
                    task="qa",
                    query_id=normalized_submission_id,
                    current=current_entry,
                    row_data=single_row_cache["data"],
                    update_rank=qa_update_rank if qa_stage_action == "update" else None,
                )
                st.success(
                    f"Đã {manual_action_label.lower()} trong `{staged_filename}` "
                    "mà không gọi VLM."
                )
            except BackendError as error:
                if error.snapshot is not None:
                    _apply_qa_snapshot(error.snapshot)
                if error.code == "qa_manual_answer_invalid":
                    st.warning("Câu trả lời manual không hợp lệ hoặc vượt quá 100 ký tự.")
                else:
                    _report_ui_error(
                        error,
                        workflow="qa",
                        stage="manual_answer",
                        batch_id=batch_id,
                        query_or_session_id=st.session_state.qa_search_id,
                    )
            except ValueError as error:
                message = (
                    "Query ID đã trùng với entry khác trong bundle."
                    if str(error) == "bundle_query_id_collision"
                    else "Không thể chuẩn bị CSV manual từ candidate đã chọn."
                )
                st.warning(message)
            except Exception as error:
                _report_ui_error(
                    error,
                    workflow="qa",
                    stage="manual_answer",
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
            try:
                output_preview = format_qa_submission_row(
                    str(result["video_id"]),
                    int(result["source_frame_idx"]),
                    str(result["answer"]),
                )
            except (TypeError, ValueError):
                output_preview = "Không thể tạo preview CSV đã xác minh"
            st.caption("Submission Line Preview")
            st.code(output_preview, language=None)
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
    target["trake_pinned"] = []
    target["trake_query_id"] = None
    target["trake_result_stage_count"] = None
    target["trake_export_cache"] = None
    target["trake_frame_overrides"] = {}
    target.pop("trake_selected_candidate", None)


def _trake_sequence_with_overrides(
    context: RuntimeContext,
    selected: dict[str, Any],
    overrides: dict[int, int],
) -> dict[str, Any]:
    """Resolve an operator-edited sequence back to canonical timestamps."""

    batch_id = str(selected.get("batch_id", ""))
    video_id = str(selected.get("video_id", ""))
    raw_matches = selected.get("stage_matches", ())
    if not batch_id or not video_id or not isinstance(raw_matches, (list, tuple)):
        raise ValueError("invalid TRAKE candidate")
    repository = getattr(context, "artifact_repository", None)
    getter = getattr(repository, "get_source_frame", None)
    if not callable(getter):
        raise ValueError("canonical repository unavailable")
    matches: list[dict[str, Any]] = []
    for expected_stage, raw in enumerate(raw_matches, start=1):
        if not isinstance(raw, dict):
            raise ValueError("invalid TRAKE stage match")
        stage_index = int(raw.get("stage_index", expected_stage))
        if stage_index != expected_stage:
            raise ValueError("TRAKE stages are not contiguous")
        source_frame_idx = int(
            overrides.get(stage_index, int(raw.get("source_frame_idx", -1)))
        )
        frame = getter(batch_id, video_id, source_frame_idx)
        if frame is None:
            raise ValueError("selected frame is not canonical")
        match = {
            "stage_index": stage_index,
            "source_frame_idx": source_frame_idx,
            "timestamp_ms": int(frame.timestamp_ms),
        }
        similarity = raw.get("similarity")
        if (
            source_frame_idx == int(raw.get("source_frame_idx", -1))
            and isinstance(similarity, (int, float))
            and not isinstance(similarity, bool)
        ):
            match["similarity"] = float(raw["similarity"])
        matches.append(match)
    frames = [int(value["source_frame_idx"]) for value in matches]
    timestamps = [int(value["timestamp_ms"]) for value in matches]
    if any(left >= right for left, right in zip(frames, frames[1:])) or any(
        left >= right for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("edited TRAKE sequence must remain strictly monotonic")
    return {**selected, "stage_matches": matches}


def _pin_trake_sequence(sequence: dict[str, Any]) -> None:
    """Keep one explicit operator-selected sequence per video in local state."""

    video_id = str(sequence.get("video_id", ""))
    pinned = [
        dict(value)
        for value in st.session_state.get("trake_pinned", ())
        if isinstance(value, dict) and str(value.get("video_id")) != video_id
    ]
    pinned.append(dict(sequence))
    st.session_state.trake_pinned = [
        {**value, "rank": rank} for rank, value in enumerate(pinned[:100], start=1)
    ]
    st.session_state.trake_export_cache = None


def _render_trake(context: RuntimeContext) -> None:
    """Render the Vietnamese TRAKE stage-alignment workflow."""

    st.subheader("TRAKE · Căn chỉnh chuỗi sự kiện theo thời gian")
    st.caption(
        "Mô tả toàn cảnh và từ 2 đến 20 giai đoạn theo đúng thứ tự xảy ra. "
        "Mỗi video chỉ giữ một chuỗi khung hình tốt nhất."
    )
    batch_id = _selected_batch_id(context)
    st.caption(f"Batch hiện hành: `{batch_id or 'chưa chọn'}`")
    st.text_input(
        "Query ID nộp bài TRAKE",
        key="codabench_query_id_trake",
        placeholder="Ví dụ: p1-4",
    )
    global_description_input = st.text_area(
        "Mô tả toàn bộ sự kiện",
        key="trake_global_description",
        on_change=_invalidate_trake_results,
        placeholder="Ví dụ: Một người bước vào cửa hàng, chọn sản phẩm rồi thanh toán và rời đi.",
    )
    global_entity_constraints = _render_entity_controls(
        global_description_input,
        state_key="trake_entity_modes",
        scope="global",
        on_change=_invalidate_trake_results,
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
    stage_entity_constraints: dict[int, tuple[EntityConstraint, ...]] = {}
    for stage_index in range(1, stage_count + 1):
        key = f"trake_stage_{stage_index}"
        st.session_state.setdefault(key, "")
        stage_text = st.text_input(
            f"Giai đoạn {stage_index}",
            key=key,
            on_change=_invalidate_trake_results,
            placeholder=f"Mô tả hành động ở giai đoạn {stage_index}",
        )
        stage_entity_constraints[stage_index] = _render_entity_controls(
            stage_text,
            state_key="trake_entity_modes",
            scope=f"stage:{stage_index}",
            on_change=_invalidate_trake_results,
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
                            entity_constraints=stage_entity_constraints.get(index, ()),
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
                    global_entity_constraints=global_entity_constraints,
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
    all_overrides = dict(st.session_state.get("trake_frame_overrides", {}))
    selected_overrides = {
        int(stage_index): int(frame_idx)
        for stage_index, frame_idx in dict(all_overrides.get(selected_id, {})).items()
    }
    try:
        selected_sequence = _trake_sequence_with_overrides(
            context,
            selected,
            selected_overrides,
        )
    except (TypeError, ValueError):
        selected_overrides = {}
        all_overrides.pop(selected_id, None)
        st.session_state.trake_frame_overrides = all_overrides
        selected_sequence = _trake_sequence_with_overrides(context, selected, {})
    matches = selected_sequence.get("stage_matches", ())
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

        st.markdown("**Điều chỉnh một stage bằng các frame lân cận**")
        if st.session_state.get("trake_adjust_stage") not in range(
            1, len(valid_matches) + 1
        ):
            st.session_state.trake_adjust_stage = 1
        stage_choice = st.selectbox(
            "Stage cần điều chỉnh",
            tuple(range(1, len(valid_matches) + 1)),
            key="trake_adjust_stage",
            format_func=lambda value: f"E{value}",
        )
        anchor = valid_matches[int(stage_choice) - 1]
        repository = getattr(context, "artifact_repository", None)
        neighbor_values: tuple[Any, ...] = ()
        neighbor_loader = getattr(repository, "neighbors", None)
        if callable(neighbor_loader):
            try:
                neighbor_values = tuple(
                    neighbor_loader(
                        str(selected_sequence.get("batch_id", batch_id)),
                        str(selected_sequence.get("video_id", "")),
                        int(anchor.get("timestamp_ms", 0)),
                        radius_ms=4_000,
                        limit=5,
                    )
                )
            except Exception:
                neighbor_values = ()
        previous = valid_matches[int(stage_choice) - 2] if int(stage_choice) > 1 else None
        following = (
            valid_matches[int(stage_choice)]
            if int(stage_choice) < len(valid_matches)
            else None
        )
        compatible_neighbors = [
            frame
            for frame in neighbor_values
            if (
                previous is None
                or (
                    int(frame.source_frame_idx) > int(previous["source_frame_idx"])
                    and int(frame.timestamp_ms) > int(previous["timestamp_ms"])
                )
            )
            and (
                following is None
                or (
                    int(frame.source_frame_idx) < int(following["source_frame_idx"])
                    and int(frame.timestamp_ms) < int(following["timestamp_ms"])
                )
            )
        ]
        if compatible_neighbors:
            neighbor_columns = st.columns(len(compatible_neighbors))
            for column, frame in zip(
                neighbor_columns,
                compatible_neighbors,
                strict=True,
            ):
                with column:
                    image = _trake_frame_image(
                        context,
                        str(frame.batch_id),
                        str(frame.video_id),
                        int(frame.source_frame_idx),
                    )
                    if image is not None:
                        st.image(image, width="stretch")
                    if st.button(
                        f"Chọn {int(frame.source_frame_idx)}",
                        key=(
                            f"trake_swap_{hashlib.sha256(selected_id.encode()).hexdigest()[:8]}_"
                            f"{stage_choice}_{int(frame.source_frame_idx)}"
                        ),
                    ):
                        selected_overrides[int(stage_choice)] = int(frame.source_frame_idx)
                        all_overrides[selected_id] = selected_overrides
                        st.session_state.trake_frame_overrides = all_overrides
                        st.session_state.trake_export_cache = None
                        st.rerun()
        else:
            st.caption("Không có frame lân cận nào giữ được thứ tự thời gian nghiêm ngặt.")
        if st.button(
            "Khôi phục chuỗi gốc",
            key=f"trake_reset_sequence_{hashlib.sha256(selected_id.encode()).hexdigest()[:8]}",
            disabled=not selected_overrides,
        ):
            all_overrides.pop(selected_id, None)
            st.session_state.trake_frame_overrides = all_overrides
            st.session_state.trake_export_cache = None
            st.rerun()

        submission_id = _normalized_codabench_query_id(
            st.session_state.get("codabench_query_id_trake", "")
        )
        staged_rank: int | None = None
        if submission_id is not None:
            staged_name = _codabench_filename(submission_id, "trake")
            staged_entry = st.session_state.get("codabench_staging", {}).get(staged_name)
            if isinstance(staged_entry, dict):
                for index, row in enumerate(_staged_row_values(staged_entry), start=1):
                    if row and row[0] == str(selected_sequence.get("video_id", "")):
                        staged_rank = index
                        break
        pin_label = (
            f"📌 Cập nhật chuỗi ở Rank #{staged_rank}"
            if staged_rank is not None
            else "📌 Ghim chuỗi vào Submission Hub"
        )
        if st.button(
            pin_label,
            key="trake_pin_selected",
            disabled=submission_id is None,
            width="stretch",
        ):
            assert submission_id is not None
            try:
                single = {**selected_sequence, "rank": 1}
                _, row_data = _trake_csv_bytes(
                    context,
                    query_id=str(st.session_state.trake_query_id or "trake"),
                    batch_id=str(selected_sequence["batch_id"]),
                    stage_count=len(valid_matches),
                    results=[single],
                )
                current = {
                    "task": "trake",
                    "batch_id": str(selected_sequence["batch_id"]),
                    "data": row_data,
                    "source_fingerprint": hashlib.sha256(row_data).hexdigest(),
                    "stage_count": len(valid_matches),
                    "row_count": 1,
                    "source_ranks": None,
                }
                staged_filename = _append_or_update_codabench_row(
                    context,
                    task="trake",
                    query_id=submission_id,
                    current=current,
                    row_data=row_data,
                    update_rank=staged_rank,
                )
                _pin_trake_sequence(selected_sequence)
            except (BackendError, TypeError, ValueError) as error:
                if isinstance(error, BackendError) and error.code == "staging_revision_conflict":
                    st.warning("Staging đã đổi ở tab khác; hãy bấm ghim lại trên snapshot mới.")
                else:
                    st.warning("Chuỗi chưa được ghim vì frame hoặc thứ tự thời gian không hợp lệ.")
            else:
                st.success(f"Đã ghi chuỗi vào `{staged_filename}`.")

    query_id = str(st.session_state.trake_query_id or "trake")
    submission_results = list(st.session_state.get("trake_pinned", ()))
    if not submission_results:
        st.caption("Ghim ít nhất một chuỗi để chuẩn bị CSV TRAKE cục bộ.")
        return
    export_fingerprint = _result_fingerprint(
        workflow="trake",
        batch_id=str(batch_id),
        rows=submission_results,
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
                results=submission_results,
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


def _codabench_row_count(data: bytes) -> int:
    try:
        text = data.decode("utf-8")
        return sum(1 for _ in csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error):
        return 0


def _codabench_records(data: bytes) -> list[bytes]:
    """Return exact logical CSV records from trusted staged bytes."""

    text = data.decode("utf-8", errors="strict")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    line_numbers: list[int] = []
    for row in reader:
        if not row:
            raise ValueError("empty CSV record")
        line_numbers.append(reader.line_num)
    lines = text.splitlines(keepends=True)
    records: list[bytes] = []
    start = 0
    for stop in line_numbers:
        records.append("".join(lines[start:stop]).encode("utf-8"))
        start = stop
    if not records or start != len(lines) or b"".join(records) != data:
        raise ValueError("invalid staged CSV records")
    return records


def _staged_row_values(entry: dict[str, Any]) -> list[list[str]]:
    data = entry.get("data")
    if not isinstance(data, bytes):
        return []
    try:
        return [
            list(row)
            for row in csv.reader(
                io.StringIO(data.decode("utf-8"), newline=""),
                strict=True,
            )
        ]
    except (UnicodeDecodeError, csv.Error):
        return []


def _codabench_current_entry(
    workflow: str,
) -> dict[str, Any] | None:
    """Return the currently validated CSV cache as one stageable entry."""

    cache_name = {
        "kis": "kis_export_cache",
        "qa": "qa_export_cache",
        "trake": "trake_export_cache",
    }[workflow]
    cache = st.session_state.get(cache_name)
    if not isinstance(cache, dict):
        return None
    data = cache.get("data")
    fingerprint = cache.get("fingerprint")
    if not isinstance(data, bytes) or not isinstance(fingerprint, str):
        return None
    source_rows = {
        "kis": list(st.session_state.get("kis_pinned", ())),
        "qa": list(st.session_state.get("qa_pinned", ())),
        "trake": list(st.session_state.get("trake_pinned", ())),
    }[workflow]
    batch_ids = {
        str(row.get("batch_id"))
        for row in source_rows
        if isinstance(row, dict) and isinstance(row.get("batch_id"), str)
    }
    if len(batch_ids) != 1:
        return None
    stage_count = None
    if workflow == "trake":
        value = st.session_state.get("trake_result_stage_count")
        if isinstance(value, int) and not isinstance(value, bool):
            stage_count = value
        else:
            return None
    row_count = _codabench_row_count(data)
    return {
        "task": workflow,
        "batch_id": next(iter(batch_ids)),
        "data": data,
        "source_fingerprint": fingerprint,
        "stage_count": stage_count,
        "row_count": row_count,
        "source_ranks": (
            tuple(
                int(rank)
                for rank in cache.get(
                    "source_ranks", tuple(range(1, row_count + 1))
                )
            )
            if workflow == "kis"
            else None
        ),
    }


def _codabench_filename(query_id: str, task: str) -> str:
    return codabench_query_filename(query_id, task)  # type: ignore[arg-type]


def _normalized_codabench_query_id(value: Any) -> str | None:
    try:
        return normalize_codabench_query_id(value)
    except (TypeError, ValueError):
        return None


def _stage_codabench_entry(
    context: RuntimeContext,
    *,
    task: str,
    query_id: str,
    current: dict[str, Any],
) -> str:
    """Validate and persist one immutable entry through the FastAPI owner."""

    normalized = normalize_codabench_query_id(query_id)
    filename = _codabench_filename(normalized, task)
    collision_id = codabench_query_id_collision_key(normalized)
    staging = dict(st.session_state.get("codabench_staging", {}))
    for existing_name, item in staging.items():
        existing_id = item.get("query_id")
        if not isinstance(existing_id, str):
            continue
        same_id = codabench_query_id_collision_key(existing_id) == collision_id
        same_filename = existing_name.casefold() == filename.casefold()
        if existing_name == filename:
            continue
        if same_id or same_filename:
            raise ValueError("bundle_query_id_collision")
    pending = {
        **current,
        "filename": filename,
        "query_id": normalized,
        "sha256": hashlib.sha256(current["data"]).hexdigest(),
    }
    stager = getattr(context, "stage_submission_entry", None)
    if callable(stager):
        try:
            snapshot = stager(
                {
                    "expected_revision": int(
                        st.session_state.get("codabench_staging_revision", 0)
                    ),
                    "batch_id": pending["batch_id"],
                    "entry": {
                        "task": pending["task"],
                        "query_id": normalized,
                        "csv_base64": base64.b64encode(pending["data"]).decode("ascii"),
                        "source_fingerprint": pending["source_fingerprint"],
                        "stage_count": pending.get("stage_count"),
                        "source_ranks": pending.get("source_ranks"),
                    },
                }
            )
        except BackendError as error:
            if error.code == "staging_revision_conflict":
                loader = getattr(context, "load_submission_staging", None)
                if callable(loader):
                    _apply_codabench_staging_snapshot(loader())
            raise
        _apply_codabench_staging_snapshot(snapshot)
    else:
        staging[filename] = pending
        st.session_state.codabench_staging = staging
    errors = dict(st.session_state.get("codabench_entry_errors", {}))
    errors.pop(filename, None)
    st.session_state.codabench_entry_errors = errors
    st.session_state.codabench_bundle_cache = None
    return filename


def _mutate_codabench_entry_rows(
    context: RuntimeContext,
    *,
    filename: str,
    action: str,
    row_data: bytes | None = None,
    rank: int | None = None,
    target_rank: int | None = None,
) -> None:
    """Mutate one durable physical rank through the FastAPI trust boundary."""

    staging = dict(st.session_state.get("codabench_staging", {}))
    entry = staging.get(filename)
    if not isinstance(entry, dict):
        raise ValueError("staging_entry_not_found")
    mutator = getattr(context, "mutate_submission_rows", None)
    payload: dict[str, Any] = {
        "action": action,
        "expected_revision": int(st.session_state.get("codabench_staging_revision", 0)),
    }
    if row_data is not None:
        if len(_codabench_records(row_data)) != 1:
            raise ValueError("staging_row_invalid")
        payload["row_csv_base64"] = base64.b64encode(row_data).decode("ascii")
    if rank is not None:
        payload["rank"] = rank
    if target_rank is not None:
        payload["target_rank"] = target_rank
    if callable(mutator):
        try:
            snapshot = mutator(filename, payload)
        except BackendError as error:
            if error.code == "staging_revision_conflict":
                loader = getattr(context, "load_submission_staging", None)
                if callable(loader):
                    _apply_codabench_staging_snapshot(loader())
            raise
        _apply_codabench_staging_snapshot(snapshot)
    else:
        records = _codabench_records(entry["data"])
        incoming = None if row_data is None else _codabench_records(row_data)[0]
        if action == "append" and incoming is not None:
            if len(records) >= 100:
                raise ValueError("staging_row_limit_exceeded")
            records.append(incoming)
        elif (
            action == "update"
            and incoming is not None
            and isinstance(rank, int)
            and not isinstance(rank, bool)
            and 1 <= rank <= len(records)
        ):
            records[rank - 1] = incoming
        elif (
            action == "delete"
            and isinstance(rank, int)
            and not isinstance(rank, bool)
            and 1 <= rank <= len(records)
        ):
            records.pop(rank - 1)
        elif (
            action == "move"
            and isinstance(rank, int)
            and not isinstance(rank, bool)
            and isinstance(target_rank, int)
            and not isinstance(target_rank, bool)
            and 1 <= rank <= len(records)
            and 1 <= target_rank <= len(records)
        ):
            record = records.pop(rank - 1)
            records.insert(target_rank - 1, record)
        else:
            raise ValueError("staging_row_invalid")
        if not records:
            deleter = getattr(context, "delete_submission_entry", None)
            if callable(deleter):
                snapshot = deleter(
                    filename,
                    expected_revision=int(
                        st.session_state.get("codabench_staging_revision", 0)
                    ),
                )
                _apply_codabench_staging_snapshot(snapshot)
            else:
                staging.pop(filename, None)
                st.session_state.codabench_staging = staging
        else:
            data = b"".join(records)
            _stage_codabench_entry(
                context,
                task=str(entry["task"]),
                query_id=str(entry["query_id"]),
                current={
                    "task": entry["task"],
                    "batch_id": entry["batch_id"],
                    "data": data,
                    "source_fingerprint": hashlib.sha256(data).hexdigest(),
                    "stage_count": entry.get("stage_count"),
                    "row_count": len(records),
                    "source_ranks": None,
                },
            )
    errors = dict(st.session_state.get("codabench_entry_errors", {}))
    errors.pop(filename, None)
    st.session_state.codabench_entry_errors = errors
    st.session_state.codabench_bundle_cache = None


def _append_or_update_codabench_row(
    context: RuntimeContext,
    *,
    task: str,
    query_id: str,
    current: dict[str, Any],
    row_data: bytes,
    update_rank: int | None = None,
) -> str:
    """Create a first row or append/replace one rank in an existing entry."""

    normalized = normalize_codabench_query_id(query_id)
    filename = _codabench_filename(normalized, task)
    staged = st.session_state.get("codabench_staging", {}).get(filename)
    if not isinstance(staged, dict):
        if update_rank is not None:
            raise ValueError("staging_entry_not_found")
        first = dict(current)
        first["data"] = row_data
        first["row_count"] = 1
        first["source_fingerprint"] = hashlib.sha256(row_data).hexdigest()
        return _stage_codabench_entry(
            context,
            task=task,
            query_id=normalized,
            current=first,
        )
    if staged.get("task") != task or staged.get("batch_id") != current.get("batch_id"):
        raise ValueError("bundle_query_id_collision")
    if task == "trake" and staged.get("stage_count") != current.get("stage_count"):
        raise ValueError("submission_stage_count_mismatch")
    _mutate_codabench_entry_rows(
        context,
        filename=filename,
        action="append" if update_rank is None else "update",
        row_data=row_data,
        rank=update_rank,
    )
    return filename


def _apply_codabench_staging_snapshot(snapshot: dict[str, Any]) -> None:
    """Hydrate safe session-state projections from an authoritative API snapshot."""

    revision = snapshot.get("revision")
    raw_entries = snapshot.get("entries")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(raw_entries, dict)
    ):
        raise BackendError("backend_response_invalid")
    hydrated: dict[str, dict[str, Any]] = {}
    for filename, value in raw_entries.items():
        if not isinstance(filename, str) or not isinstance(value, dict):
            raise BackendError("backend_response_invalid")
        try:
            query_id, task = parse_codabench_query_filename(filename)
            encoded = value["csv_base64"]
            if not isinstance(encoded, str):
                raise ValueError("invalid bytes")
            data = base64.b64decode(encoded, validate=True)
        except (KeyError, TypeError, ValueError):
            raise BackendError("backend_response_invalid") from None
        digest = value.get("sha256")
        if (
            value.get("query_id") != query_id
            or value.get("task") != task
            or not isinstance(digest, str)
            or hashlib.sha256(data).hexdigest() != digest
        ):
            raise BackendError("backend_response_invalid")
        hydrated[filename] = {
            "filename": filename,
            "task": task,
            "query_id": query_id,
            "batch_id": value.get("batch_id"),
            "data": data,
            "source_fingerprint": value.get("source_fingerprint"),
            "stage_count": value.get("stage_count"),
            "row_count": value.get("row_count"),
            "source_ranks": (
                tuple(value["source_ranks"])
                if isinstance(value.get("source_ranks"), list)
                else None
            ),
            "sha256": digest,
            "updated_at_utc": value.get("updated_at_utc"),
        }
    st.session_state.codabench_staging = hydrated
    st.session_state.codabench_staging_revision = revision
    st.session_state.codabench_staging_updated_at = snapshot.get("updated_at_utc")
    st.session_state.codabench_staging_hydrated = True
    st.session_state.codabench_staging_error = None


def _hydrate_submission_staging(context: RuntimeContext) -> None:
    """Load disk-backed staging once for this browser session."""

    if st.session_state.get("codabench_staging_hydrated"):
        return
    loader = getattr(context, "load_submission_staging", None)
    if not callable(loader):
        st.session_state.codabench_staging_hydrated = True
        return
    try:
        _apply_codabench_staging_snapshot(loader())
    except BackendError as error:
        st.session_state.codabench_staging_hydrated = True
        st.session_state.codabench_staging_error = error.code


def _codabench_entry_status(
    entry: dict[str, Any],
    current_by_task: dict[str, dict[str, Any] | None],
) -> str:
    filename = str(entry.get("filename", ""))
    if filename in st.session_state.get("codabench_entry_errors", {}):
        return "INVALID"
    task = str(entry.get("task", ""))
    query_id = str(entry.get("query_id", ""))
    active_query_id = str(st.session_state.get(f"codabench_query_id_{task}", "")).strip()
    if active_query_id != query_id:
        return "READY"
    if task == "kis":
        working = st.session_state.get("kis_working_fingerprint")
        if not isinstance(working, str):
            working = _kis_working_source_fingerprint()
            st.session_state.kis_working_fingerprint = working
        return (
            "READY"
            if isinstance(working, str) and working == entry.get("source_fingerprint")
            else "STALE"
        )

    if task in {"qa", "trake"}:
        # These workflows are managed as durable ranked lists. Their validated
        # disk snapshot remains authoritative across F5 and candidate changes.
        return "READY"
    current = current_by_task.get(task)
    if not isinstance(current, dict):
        return "READY"
    current_data = current.get("data")
    if not isinstance(current_data, bytes) or hashlib.sha256(current_data).hexdigest() != entry.get(
        "sha256"
    ):
        return "STALE"
    return "READY"


def _render_staged_rank_manager(
    context: RuntimeContext,
    *,
    task: str,
    query_id: str | None,
) -> None:
    """Render granular controls for an active QA or TRAKE ranked CSV."""

    if task not in {"qa", "trake"} or query_id is None:
        return
    filename = _codabench_filename(query_id, task)
    entry = st.session_state.get("codabench_staging", {}).get(filename)
    if not isinstance(entry, dict):
        return
    rows = _staged_row_values(entry)
    if not rows:
        return
    st.markdown(f"**Quản lý thứ hạng · `{filename}`**")
    digest = str(entry.get("sha256", ""))[:10]
    for rank, values in enumerate(rows, start=1):
        if task == "qa" and len(values) == 3:
            identity = f"{values[0]} · frame {values[1]}"
            detail = f'“{values[2]}”'
        elif task == "trake" and len(values) >= 3:
            identity = values[0]
            detail = " → ".join(values[1:])
        else:
            identity = "Dòng không hợp lệ"
            detail = ""
        rank_col, identity_col, detail_col, up_col, down_col, delete_col = st.columns(
            (0.7, 1.7, 3.2, 0.55, 0.55, 0.8)
        )
        rank_col.markdown(f"`#{rank}`")
        identity_col.caption(identity)
        detail_col.caption(detail)
        operation: tuple[str, int | None] | None = None
        with up_col:
            if st.button(
                "⬆",
                key=f"{task}_staged_up_{digest}_{rank}",
                disabled=rank == 1,
                help="Đưa lên một rank",
            ):
                operation = ("move", rank - 1)
        with down_col:
            if st.button(
                "⬇",
                key=f"{task}_staged_down_{digest}_{rank}",
                disabled=rank == len(rows),
                help="Đưa xuống một rank",
            ):
                operation = ("move", rank + 1)
        with delete_col:
            if st.button(
                "🗑️",
                key=f"{task}_staged_delete_{digest}_{rank}",
                help="Xóa riêng rank này",
            ):
                operation = ("delete", None)
        if operation is None:
            continue
        try:
            action, destination = operation
            _mutate_codabench_entry_rows(
                context,
                filename=filename,
                action=action,
                rank=rank,
                target_rank=destination,
            )
        except BackendError as error:
            if error.code == "staging_revision_conflict":
                st.warning("Staging đã đổi ở tab khác; đã tải lại snapshot mới nhất.")
            else:
                st.warning("Không thể thay đổi rank; staging hợp lệ cũ được giữ nguyên.")
        except (TypeError, ValueError):
            st.warning("Rank được chọn không hợp lệ; staging chưa bị thay đổi.")
        else:
            st.rerun()


def _render_codabench_submission_hub(context: RuntimeContext, active_workflow: str) -> None:
    """Stage validated CSV bytes and request server-owned Codabench packaging."""

    st.divider()
    st.subheader("Codabench Submission Hub")
    st.caption(
        "Profile `codabench_10187_2025`: ZIP chỉ chứa thư mục `submission/`. "
        "Không tự động tải dữ liệu lên Codabench."
    )
    if st.session_state.get("codabench_staging_error"):
        st.warning("Không thể tải staging bền vững; dữ liệu trên đĩa chưa bị thay đổi.")
    else:
        st.success("💾 Persistent State: Active · Preserved on F5 / Refresh")
    columns = st.columns(3)
    for column, task, label in zip(
        columns,
        ("kis", "qa", "trake"),
        ("KIS", "Visual Q&A", "TRAKE"),
    ):
        with column:
            if task == active_workflow and task in {"qa", "trake"}:
                active_id = str(
                    st.session_state.get(f"codabench_query_id_{task}", "")
                ).strip()
                st.caption(
                    f"Query ID nộp bài · {label}: `{active_id or 'chưa nhập'}`"
                )
            else:
                st.text_input(
                    f"Query ID nộp bài · {label}",
                    key=f"codabench_query_id_{task}",
                    placeholder="Ví dụ: p1-1, p1-15, 01, round1_01",
                )

    current_by_task = {
        task: _codabench_current_entry(task) for task in ("kis", "qa", "trake")
    }
    raw_query_id = str(
        st.session_state.get(f"codabench_query_id_{active_workflow}", "")
    ).strip()
    query_id = _normalized_codabench_query_id(raw_query_id)
    current = current_by_task[active_workflow]
    stage_disabled = query_id is None or current is None
    if active_workflow == "kis":
        if st.button(
            "Thêm/cập nhật truy vấn KIS hiện tại",
            key="codabench_stage_current",
            disabled=stage_disabled,
            width="stretch",
        ):
            assert current is not None
            assert query_id is not None
            try:
                _stage_codabench_entry(
                    context,
                    task=active_workflow,
                    query_id=query_id,
                    current=current,
                )
            except BackendError as error:
                if error.code == "staging_revision_conflict":
                    st.warning("Staging đã đổi ở tab khác; snapshot mới nhất đã được tải lại.")
                else:
                    st.warning("CSV chưa được lưu vì preflight hoặc staging backend từ chối.")
            except ValueError:
                st.error("Query ID này đã được dùng cho một workflow khác trong bundle.")
            else:
                st.rerun()
    else:
        st.caption(
            "Dùng nút ghim trong workflow để nối thêm hoặc cập nhật từng rank; "
            "Hub không ghi đè toàn bộ bảng xếp hạng QA/TRAKE."
        )
    if active_workflow == "kis" and stage_disabled:
        st.caption(
            "Nhập Query ID gồm 1–32 chữ ASCII, số, dấu gạch dưới hoặc gạch nối; "
            "đồng thời chuẩn bị CSV hợp lệ trước khi stage."
        )

    staging = dict(st.session_state.get("codabench_staging", {}))
    statuses = {
        name: _codabench_entry_status(entry, current_by_task)
        for name, entry in staging.items()
    }
    matrix_rows = [
        {
            "File": name,
            "Workflow": str(entry.get("task", "")).upper(),
            "Trạng thái": statuses[name],
            "Số dòng": entry.get("row_count"),
            "Số stage": entry.get("stage_count"),
            "Preview": (
                entry["data"].decode("utf-8").splitlines()[0]
                if entry.get("task") == "qa"
                and isinstance(entry.get("data"), bytes)
                and entry["data"]
                else None
            ),
        }
        for name, entry in sorted(
            staging.items(),
            key=lambda item: (
                codabench_query_id_sort_key(str(item[1]["query_id"])),
                item[0].casefold(),
                item[0],
            ),
        )
    ]
    for task in ("kis", "qa", "trake"):
        task_query_id = _normalized_codabench_query_id(
            st.session_state.get(f"codabench_query_id_{task}", "")
        )
        expected_name = (
            _codabench_filename(task_query_id, task)
            if task_query_id is not None
            else f"query-<id>-{task}.csv"
        )
        if expected_name in staging:
            continue
        current_task_entry = current_by_task[task]
        matrix_rows.append(
            {
                "File": expected_name,
                "Workflow": task.upper(),
                "Trạng thái": "MISSING",
                "Số dòng": (
                    current_task_entry.get("row_count")
                    if isinstance(current_task_entry, dict)
                    else None
                ),
                "Số stage": (
                    current_task_entry.get("stage_count")
                    if isinstance(current_task_entry, dict)
                    else None
                ),
                "Preview": None,
            }
        )
    st.dataframe(matrix_rows, width="stretch", hide_index=True)
    _render_staged_rank_manager(
        context,
        task=active_workflow,
        query_id=query_id,
    )
    if staging:
        remove_name = st.selectbox(
            "Entry cần xóa khỏi staging",
            tuple(sorted(staging)),
            key="codabench_remove_entry",
        )
        remove_column, clear_column = st.columns(2)
        with remove_column:
            if st.button("Xóa entry đã chọn", key="codabench_remove", width="stretch"):
                deleter = getattr(context, "delete_submission_entry", None)
                try:
                    if callable(deleter):
                        snapshot = deleter(
                            remove_name,
                            expected_revision=int(
                                st.session_state.get("codabench_staging_revision", 0)
                            ),
                        )
                        _apply_codabench_staging_snapshot(snapshot)
                    else:
                        staging.pop(remove_name, None)
                        st.session_state.codabench_staging = staging
                    errors = dict(st.session_state.get("codabench_entry_errors", {}))
                    errors.pop(remove_name, None)
                    st.session_state.codabench_entry_errors = errors
                    st.session_state.codabench_bundle_cache = None
                    st.rerun()
                except BackendError as error:
                    if error.code == "staging_revision_conflict":
                        loader = getattr(context, "load_submission_staging", None)
                        if callable(loader):
                            _apply_codabench_staging_snapshot(loader())
                    st.warning("Không thể xóa entry; staging mới nhất đã được giữ nguyên.")
        with clear_column:
            if st.button(
                "🗑️ Xóa tất cả & Khôi phục mặc định",
                key="codabench_clear_all",
                width="stretch",
            ):
                clearer = getattr(context, "clear_submission_staging", None)
                try:
                    if callable(clearer):
                        snapshot = clearer(
                            expected_revision=int(
                                st.session_state.get("codabench_staging_revision", 0)
                            )
                        )
                        _apply_codabench_staging_snapshot(snapshot)
                    else:
                        st.session_state.codabench_staging = {}
                    st.session_state.codabench_entry_errors = {}
                    st.session_state.codabench_bundle_cache = None
                    st.rerun()
                except BackendError as error:
                    if error.code == "staging_revision_conflict":
                        loader = getattr(context, "load_submission_staging", None)
                        if callable(loader):
                            _apply_codabench_staging_snapshot(loader())
                    st.warning("Không thể xóa staging; dữ liệu hiện có vẫn được giữ nguyên.")
    else:
        st.info("Chưa có CSV nào trong staging.")

    mode = st.radio(
        "Phạm vi bundle",
        ("all", "current"),
        key="codabench_bundle_mode",
        horizontal=True,
        format_func=lambda value: (
            "Bundle All · bắt buộc KIS + QA + TRAKE"
            if value == "all"
            else "Chỉ workflow hiện tại"
        ),
    )
    selected: list[dict[str, Any]] = []
    if mode == "all":
        selected = list(staging.values())
    elif query_id is not None:
        selected_name = _codabench_filename(query_id, active_workflow)
        if selected_name in staging:
            selected = [staging[selected_name]]
    tasks = {str(entry.get("task")) for entry in selected}
    selected_names = {str(entry.get("filename")) for entry in selected}
    can_build = bool(selected) and all(statuses.get(name) == "READY" for name in selected_names)
    if mode == "all":
        can_build = can_build and tasks == {"kis", "qa", "trake"}
    batch_ids = {str(entry.get("batch_id")) for entry in selected}
    can_build = can_build and len(batch_ids) == 1

    if st.button(
        "📦 Tạo gói nộp bài Codabench (.zip)",
        key="codabench_build_bundle",
        disabled=not can_build,
        width="stretch",
    ):
        builder = getattr(context, "create_submission_bundle", None)
        if not callable(builder):
            st.error("Backend hiện tại chưa hỗ trợ tạo Codabench bundle.")
        else:
            persistent_backend = callable(
                getattr(context, "load_submission_staging", None)
            )
            payload_entries = None
            if not persistent_backend:
                payload_entries = [
                    {
                        "task": entry["task"],
                        "query_id": entry["query_id"],
                        "csv_base64": base64.b64encode(entry["data"]).decode("ascii"),
                        "source_fingerprint": entry["source_fingerprint"],
                        "stage_count": entry.get("stage_count"),
                        "source_ranks": entry.get("source_ranks"),
                    }
                    for entry in selected
                ]
            try:
                with st.spinner("Đang kiểm tra CSV và tạo Codabench archive..."):
                    bundle_payload: dict[str, Any] = {
                        "mode": mode,
                        "batch_id": next(iter(batch_ids)),
                    }
                    if persistent_backend:
                        bundle_payload["entry_filenames"] = sorted(selected_names)
                        bundle_payload["staging_revision"] = int(
                            st.session_state.get("codabench_staging_revision", 0)
                        )
                    else:
                        bundle_payload["entries"] = payload_entries
                    result = builder(bundle_payload)
                st.session_state.codabench_bundle_cache = result
                st.session_state.codabench_entry_errors = {}
            except BackendError as error:
                issues = error.detail.get("issues", [])
                entry_errors: dict[str, list[dict[str, Any]]] = {}
                for issue in issues if isinstance(issues, list) else ():
                    if isinstance(issue, dict) and isinstance(issue.get("filename"), str):
                        entry_errors.setdefault(issue["filename"], []).append(issue)
                st.session_state.codabench_entry_errors = entry_errors
                st.session_state.codabench_bundle_cache = None
                st.error("Bundle không hợp lệ; chưa tạo file ZIP.")
                for issue in issues if isinstance(issues, list) else ():
                    if not isinstance(issue, dict):
                        continue
                    location = ""
                    if issue.get("row") is not None:
                        location = f" · dòng dữ liệu {issue['row']}"
                    st.warning(
                        f"{issue.get('filename', 'CSV')}{location}: "
                        f"{issue.get('message', issue.get('code', 'không hợp lệ'))}. "
                        f"{issue.get('suggestion', '')}".strip()
                    )
            except Exception as error:
                _report_ui_error(
                    error,
                    workflow="submission",
                    stage="bundle",
                    batch_id=next(iter(batch_ids)),
                )
    if not can_build:
        st.caption("Bundle chỉ được mở khóa khi mọi entry đã chọn ở trạng thái READY.")

    bundle = st.session_state.get("codabench_bundle_cache")
    if isinstance(bundle, dict) and isinstance(bundle.get("data"), bytes):
        st.success("100% VALID · Bundle đã vượt qua pre-submission gate.")
        st.download_button(
            "Tải Codabench bundle",
            data=bundle["data"],
            file_name=str(bundle.get("filename", "AIC2026_submission.zip")),
            mime="application/zip",
            key="codabench_download_bundle",
            width="stretch",
        )
        st.caption(
            f"Kích thước: {int(bundle.get('byte_count', len(bundle['data']))):,} bytes · "
            f"Số CSV: {int(bundle.get('file_count', 0))} · Trạng thái: 100% VALID"
        )
        st.code(str(bundle.get("sha256", "")), language=None)


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
        pending_workflow = st.session_state.pop("_pending_active_workflow", None)
        if pending_workflow in {"kis", "qa", "trake"}:
            st.session_state.active_workflow = pending_workflow
        _render_recovery_control()
        if _render_runtime_contract_block(context):
            return
        _hydrate_submission_staging(context)
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
        _render_codabench_submission_hub(context, str(active or "kis"))
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
    if injected is not None:
        context = injected
    else:
        try:
            context = _runtime()
        except BackendError:
            st.set_page_config(
                page_title="AIC 2026 - Backend chưa sẵn sàng",
                page_icon="⚠️",
                layout="wide",
                initial_sidebar_state="auto",
            )
            inject_cyberpunk_theme()
            _initialize_state()
            _render_recovery_control()
            st.error(
                "Không thể kết nối FastAPI tại thời điểm này. "
                "Hãy kiểm tra stack rồi tải lại trang; dữ liệu Codabench trong phiên không bị xóa."
            )
            return
    if injected is None:
        logging_config = context.config.get("logging", {})
        configure_app_error_logging(
            logging_config.get("app_error_file", "outputs/logs/app_error.log"),
            rotation=logging_config.get("rotation", "10 MB"),
        )
    render_app(context)


if __name__ == "__main__":
    main()
