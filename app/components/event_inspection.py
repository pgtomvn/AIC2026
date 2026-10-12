"""On-demand evidence controls, with human confirmation separate from pin/export."""
from __future__ import annotations

from time import monotonic
from typing import Any

import streamlit as st


def render_event_inspection(context: Any, neighbor: dict[str, Any], *, key: str) -> None:
    if neighbor.get("batch_id") == "aic26-b2":
        st.caption("B02: dùng Visual Q&A để kiểm tra đúng ảnh ordinal; kiểm tra nhiều ảnh chưa được bật.")
        return
    identity = {name: neighbor[name] for name in ("batch_id", "video_id", "source_frame_idx")}
    request = st.session_state.get("kis_last_request")
    repo = context.artifact_repository
    if context.config.get("retrieval", {}).get("interactive", {}).get("event_inspection_enabled", False):
        with st.popover("Kiểm tra sự kiện"):
            st.caption("Ảnh tĩnh; không xác nhận chuyển động giữa các ảnh.")
            question = st.text_input("Nội dung cần kiểm tra", value=request.original_text if request else "",
                                     key=f"{key}_event_question", max_chars=2000)
            nearby = {f.source_frame_idx: {"batch_id": f.batch_id, "video_id": f.video_id,
                      "source_frame_idx": f.source_frame_idx}
                      for _, f in st.session_state.get("kis_scrub_frames", [])
                      if f.batch_id == identity["batch_id"] and f.video_id == identity["video_id"]
                      and f.source_frame_idx != identity["source_frame_idx"]}
            extra = st.multiselect("Thêm khung lân cận (tối đa 2)", list(nearby),
                                   format_func=lambda f: f"Frame {f}", max_selections=2,
                                   key=f"{key}_{identity['source_frame_idx']}_event_extra")
            selected = [identity, *(nearby[f] for f in extra)]
            if st.button("Kiểm tra ảnh này", key=f"{key}_event_run", disabled=not question.strip()):
                try:
                    with st.spinner("Đang kiểm tra bằng Qwen local (tối đa 60 giây)..."):
                        value = repo.inspect_event(selected, question)
                    st.session_state[f"{key}_event_result"] = (selected, question, value)
                except Exception:
                    st.warning("Không kiểm tra được lúc này; kết quả truy xuất giữ nguyên.")
            saved = st.session_state.get(f"{key}_event_result")
            if saved and saved[:2] == (selected, question):
                value = saved[2]
                if value.get("state") == "ready":
                    st.write({"evidence_present": "Có bằng chứng", "insufficient_evidence": "Không đủ bằng chứng",
                              "contradiction": "Mâu thuẫn"}.get(value.get("verdict"), "Không đủ bằng chứng"))
                    st.write(value.get("explanation", ""))
                else:
                    st.warning(f"Chưa kiểm tra: {value.get('reason_code', 'unavailable')}")
    if request is not None and callable(getattr(repo, "confirm_answer", None)):
        with st.popover("Xác nhận đáp án"):
            st.caption("Chỉ xác nhận sau khi bạn đã kiểm tra; ghim/xuất CSV không tạo nhãn đúng.")
            if st.button("Tôi xác nhận khung này đúng", key=f"{key}_confirm_answer"):
                try:
                    elapsed = max(0., monotonic()-st.session_state.get("kis_interaction_started", monotonic()))*1000
                    repo.confirm_answer(request, identity, elapsed)
                    st.success("Đã ghi xác nhận riêng cho phiên này.")
                except Exception:
                    st.warning("Chưa lưu được xác nhận; không thay đổi bài nộp.")
