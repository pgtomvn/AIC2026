"""Validated Textual KIS competition export controls."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Callable

import streamlit as st

from src.contracts import CompetitionSpec
from src.submission import (
    CanonicalSourceFrameCheck,
    FrameIndexEvidence,
    SubmissionValidationError,
    serialize_submission,
)
from src.utils.io_helpers import atomic_write_bytes


def render_export_panel(
    results: list[dict],
    query_type: str = "Textual KIS",
    output_dir: str = "outputs/submissions",
    spec: CompetitionSpec | None = None,
    batch_id: str | None = None,
    is_batch_active: Callable[[str], bool] | None = None,
    is_canonical_source_frame: CanonicalSourceFrameCheck | None = None,
    frame_index_evidence: FrameIndexEvidence | None = None,
    result_fingerprint: str | None = None,
    session_cache_key: str = "kis_export_cache",
) -> None:
    """Validate ranked source frames and offer download/server-side export."""

    if query_type != "Textual KIS" or not results:
        return
    active_spec = spec or CompetitionSpec.historical_2025()
    cached = st.session_state.get(session_cache_key)
    if not isinstance(cached, dict) or cached.get("fingerprint") != result_fingerprint:
        try:
            data = serialize_submission(
                results,
                spec=active_spec,
                batch_id=batch_id,
                is_batch_active=is_batch_active,
                is_canonical_source_frame=is_canonical_source_frame,
                frame_index_evidence=frame_index_evidence,
            )
        except SubmissionValidationError:
            st.error(
                "Không thể xuất kết quả. Hãy kiểm tra batch đang hoạt động, nguồn gốc khung "
                "hình chuẩn, dữ liệu trùng lặp và giới hạn số kết quả."
            )
            st.session_state[session_cache_key] = None
            return
        cached = {
            "fingerprint": result_fingerprint,
            "filename": f"textual-kis-{datetime.now():%Y%m%d-%H%M%S}.csv",
            "data": data,
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_ranks": tuple(int(item["rank"]) for item in results),
        }
        st.session_state[session_cache_key] = cached
    data = cached["data"]
    filename = str(cached["filename"])
    st.download_button(
        "Tải CSV đã kiểm tra",
        data=data,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )
    if st.button("Lưu một bản trên máy chủ", width="stretch"):
        try:
            target = atomic_write_bytes(Path(output_dir) / filename, data)
        except OSError:
            st.error("Không thể lưu bản đã kiểm tra trên máy chủ.")
        else:
            st.success(f"Đã lưu `{target.name}`")
