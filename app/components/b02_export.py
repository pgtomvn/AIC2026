"""Explicit B02 CSV/ZIP download, never relabel milliseconds as source frames."""
from __future__ import annotations

import io
import zipfile
from typing import Any

import streamlit as st

from src.codabench_query_id import codabench_query_filename


def combined_enabled(context: Any) -> bool:
    return bool(getattr(context, "config", {}).get("retrieval", {}).get("all_batches", {}).get("enabled", False))


def export_rows(context: Any, rows: list[dict[str, Any]], *, task: str,
                answer_spec_id: str | None = None) -> bytes:
    if not combined_enabled(context):
        return context.artifact_repository.b02_export(rows, task=task, answer_spec_id=answer_spec_id)
    exporter = getattr(context.artifact_repository, "submission_export", None)
    if callable(exporter):
        return exporter(rows, task=task, answer_spec_id=answer_spec_id)
    from src.submission.combined import serialize_combined_submission

    if any(not context.ledger.is_batch_active(row.get("batch_id")) for row in rows):
        raise ValueError("inactive batch")
    return serialize_combined_submission(context.artifact_repository, rows, task=task, answer_spec_id=answer_spec_id)


def stage_b02_rows(context: Any, state: Any, filename: str, rows: list[dict[str, Any]], *,
                   task: str, answer_spec_id: str | None = None,
                   replace_existing: bool = False, append: bool = False,
                   update_rank: int | None = None) -> None:
    """Validate the full candidate list before atomically replacing session exports."""
    files = dict(state.get("b02_submission_files", {}))
    manifests = dict(state.get("b02_submission_rows", {}))
    if filename in files and not (replace_existing or append or update_rank is not None):
        raise ValueError("confirm replacement of the existing CSV")
    if (append or update_rank is not None) and filename in files and filename not in manifests:
        raise ValueError("existing CSV has no canonical identity manifest")
    combined = [dict(row) for row in manifests.get(filename, [])] if append or update_rank else []
    if update_rank is not None:
        if len(rows) != 1 or not 1 <= update_rank <= len(combined):
            raise ValueError("invalid update rank")
        combined[update_rank - 1] = dict(rows[0])
    else:
        combined.extend(dict(row) for row in rows)
    ranked = [dict(row, rank=n) for n, row in enumerate(combined, 1)]
    data = export_rows(context, ranked, task=task, answer_spec_id=answer_spec_id)
    files[filename], manifests[filename] = data, ranked
    state["b02_submission_files"], state["b02_submission_rows"] = files, manifests


def render_b02_export(context: Any, rows: list[dict[str, Any]], *, task: str,
                      answer_spec_id: str | None = None) -> None:
    """Keep B02 exports outside the historical L-only Codabench staging adapter."""
    label = "B01 + B02" if combined_enabled(context) else "B02"
    if combined_enabled(context):
        st.caption("B01 + B02 · N: milliseconds PTS (KIS/QA) · L/M/S: source frame")
    else:
        st.caption("B02 · TRAKE: M/S, source frame tăng dần · loại N và M08_V001"
                   if task == "trake" else "B02 · N: milliseconds từ PTS · M/S: source frame")
    query_id = st.text_input("Mã câu", key=f"b02_export_query_{task}")
    replace_existing = st.checkbox("Cho phép thay CSV cùng mã câu", key=f"b02_export_replace_{task}")
    if st.button(f"Chuẩn bị CSV {label}", key=f"b02_export_prepare_{task}"):
        try:
            filename = codabench_query_filename(query_id, task)
            stage_b02_rows(context, st.session_state, filename, rows, task=task,
                           answer_spec_id=answer_spec_id, replace_existing=replace_existing)
        except Exception:
            st.warning("Chưa xuất được: kiểm tra mã câu, ordinal, đáp án, tọa độ trùng hoặc xác nhận thay CSV.")
    render_prepared_downloads(context, key=task)


def render_prepared_downloads(context: Any, *, key: str) -> None:
    """Download the existing canonical session staging without rebuilding rows."""
    task = key
    label = "B01 + B02" if combined_enabled(context) else "B02"
    files = st.session_state.get("b02_submission_files", {})
    if files:
        selected = st.selectbox(f"CSV {label} đã chuẩn bị", list(files), key=f"b02_export_file_{task}")
        st.download_button(f"Tải CSV {label}", files[selected], selected, mime="text/csv", key=f"b02_csv_{task}")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, data in sorted(files.items()):
                archive.writestr(filename, data)
        st.download_button(f"Tải ZIP {label}", output.getvalue(), "all_batches_submission.zip" if combined_enabled(context) else "B02_submission.zip",
                           mime="application/zip", key=f"b02_zip_{task}")
