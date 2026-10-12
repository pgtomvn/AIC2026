"""Session-local explicit refinement controls; no autonomous LLM or inferred labels."""

from __future__ import annotations

from dataclasses import replace
from time import monotonic
from typing import Any
from uuid import uuid4

import streamlit as st

from src.contracts import FrameReference, QueryRequest
from src.retrieval.interactive import SearchConversation, edit_conditions
from src.retrieval.refinement_preview import preview_edit


def remember_search(request: QueryRequest, results: list[dict[str, Any]], *, initial: bool = False) -> None:
    if initial or "kis_conversation" not in st.session_state:
        st.session_state.kis_conversation = SearchConversation()
        st.session_state.kis_conditions = (request.original_text,)
        st.session_state.kis_reference = None
        st.session_state.kis_positive_frames = []
        st.session_state.kis_negative_frames = []
        st.session_state.kis_interaction_session = request.interaction_session_id or str(uuid4())
        st.session_state.kis_interaction_turn = 1
        st.session_state.kis_interaction_started = monotonic()
    st.session_state.kis_conversation.push({
        "request": request, "results": results,
        "conditions": st.session_state.get("kis_conditions", (request.original_text,)),
        "reference": st.session_state.get("kis_reference"),
        "positive": st.session_state.get("kis_positive_frames", []),
        "negative": st.session_state.get("kis_negative_frames", []),
    })


def select_feedback(item: dict[str, Any], kind: str) -> None:
    identity = FrameReference(str(item["batch_id"]), str(item["video_id"]), int(item["source_frame_idx"]),
                              keyframe_ordinal=item.get("keyframe_ordinal"))
    if kind == "reference":
        st.session_state.kis_reference = identity
        st.session_state.kis_negative_frames = [v for v in st.session_state.get("kis_negative_frames", []) if v != identity]
    else:
        target = "kis_positive_frames" if kind == "positive" else "kis_negative_frames"
        other = "kis_negative_frames" if kind == "positive" else "kis_positive_frames"
        values = list(st.session_state.get(target, []))
        if identity not in values:
            values.append(identity)
        st.session_state[target] = values[-20:]
        st.session_state[other] = [v for v in st.session_state.get(other, []) if v != identity]
        if kind == "negative" and st.session_state.get("kis_reference") == identity:
            st.session_state.kis_reference = None
    st.rerun()


def restore_search(snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        return
    st.session_state.kis_last_request = snapshot["request"]
    st.session_state.kis_results = snapshot["results"]
    st.session_state.kis_conditions = snapshot["conditions"]
    st.session_state.kis_reference = snapshot["reference"]
    st.session_state.kis_positive_frames = snapshot["positive"]
    st.session_state.kis_negative_frames = snapshot["negative"]
    # Clear drafts on the next run; never modify instantiated widget state.
    st.session_state.kis_interactive_reset_draft = True
    st.rerun()


def render_interactive_search(context: Any, *, ready: bool, limit: int) -> None:
    request = st.session_state.get("kis_last_request")
    if request is None:
        return
    flags = context.config.get("retrieval", {}).get("interactive", {})
    conversation_enabled = flags.get("conversation_enabled", True)
    feedback_enabled = flags.get("feedback_enabled", True)
    if not any(flags.get(name, True) for name in ("conversation_enabled", "feedback_enabled", "temporal_enabled")):
        return
    if "kis_conversation" not in st.session_state:
        remember_search(request, st.session_state.kis_results, initial=True)
    if st.session_state.pop("kis_interactive_reset_draft", False):
        for key in ("kis_desired_text", "kis_condition_text", "kis_sequence_steps", "kis_sequence_target", "kis_sequence_enabled", "kis_image_only", "kis_edit_preview", "kis_edit_instruction"):
            st.session_state.pop(key, None)
    conversation = st.session_state.kis_conversation
    has_feedback = any(st.session_state.get(key) for key in (
        "kis_reference", "kis_positive_frames", "kis_negative_frames",
    ))
    with st.expander(f"Tìm tiếp · Lượt {len(conversation.turns)}/20", expanded=has_feedback or bool(st.session_state.get("kis_sequence_enabled"))):
        left, right = st.columns(2)
        if left.button("Hoàn tác", key="kis_turn_undo", disabled=not conversation_enabled or len(conversation.turns) < 2):
            restore_search(conversation.undo())
        if right.button("Về kết quả ban đầu", key="kis_turn_reset"):
            restore_search(conversation.reset())
        conditions = tuple(st.session_state.get("kis_conditions", (request.original_text,)))
        with st.popover("Sửa từng điều kiện", disabled=not conversation_enabled):
            instruction = st.text_input("Câu sửa English", placeholder="replace red shirt with blue shirt",
                                        key="kis_edit_instruction")
            if st.button("Xem bản nháp", key="kis_preview_edit"):
                current = st.session_state.get("kis_desired_text", ". ".join(conditions))
                st.session_state.kis_edit_preview = preview_edit(current, instruction) if instruction.strip() else None
            preview = st.session_state.get("kis_edit_preview")
            if preview:
                if preview["applied"]:
                    st.text(preview["draft_text"])
                    for change in preview["changes"]:
                        st.caption(f"{change['operation']}: {change['old']} → {change['new']}")
                    if st.button("Dùng bản nháp", key="kis_accept_edit"):
                        if preview["original_text"] == st.session_state.get("kis_desired_text", ". ".join(conditions)):
                            st.session_state.kis_desired_text = preview["draft_text"]
                            st.session_state.kis_conditions = (preview["draft_text"],)
                        st.session_state.kis_edit_preview = None
                        st.rerun()
                else:
                    st.warning("Chưa sửa: cụm từ thiếu hoặc mơ hồ. Chọn điều kiện bên dưới.")
            operation = st.selectbox("Sửa điều kiện", ("add", "replace", "remove"),
                                     format_func=lambda v: {"add": "Bổ sung", "replace": "Thay điều kiện", "remove": "Bỏ điều kiện"}[v])
            index = st.selectbox("Điều kiện cần sửa", range(len(conditions)),
                                 format_func=lambda i: f"{i + 1}. {conditions[i][:120]}")
            text = st.text_input("Nội dung điều kiện", key="kis_condition_text")
            if st.button("Áp dụng vào bản nháp", key="kis_edit_condition", disabled=not conversation_enabled):
                try:
                    updated = edit_conditions(conditions, operation, text, index)
                    st.session_state.kis_conditions = updated
                    st.session_state.kis_desired_text = ". ".join(updated)
                    st.rerun()
                except ValueError:
                    st.warning("Chọn đúng điều kiện và nhập nội dung rõ ràng; bản cũ chưa bị đổi.")
        if "kis_desired_text" not in st.session_state:
            st.session_state.kis_desired_text = ". ".join(conditions)
        desired = st.text_area("Truy vấn tiếp theo", key="kis_desired_text",
                               help="Nội dung này sẽ được gửi khi bấm Tìm tiếp.")
        reference = st.session_state.get("kis_reference") if feedback_enabled else None
        if reference is not None:
            st.caption(f"Ảnh mẫu: {reference.batch_id} / {reference.video_id} / frame {reference.source_frame_idx}")
            if st.button("Bỏ ảnh mẫu"):
                st.session_state.kis_reference = None
                st.rerun()
        positive = tuple(st.session_state.get("kis_positive_frames", ())) if feedback_enabled else ()
        negative = tuple(st.session_state.get("kis_negative_frames", ())) if feedback_enabled else ()
        if positive or negative:
            st.caption(f"Gần đúng: {len(positive)} · Không phù hợp: {len(negative)}")
            if st.button("Xóa lựa chọn phản hồi"):
                st.session_state.kis_positive_frames = []
                st.session_state.kis_negative_frames = []
                st.rerun()
        image_only = st.checkbox("Chỉ tìm bằng ảnh mẫu", disabled=reference is None, key="kis_image_only") and reference is not None
        temporal = st.checkbox("Chuỗi sự kiện", key="kis_sequence_enabled",
                               disabled=not flags.get("temporal_enabled", True) or image_only)
        steps: tuple[str, ...] = ()
        target = None
        if temporal:
            defaults = tuple(v for v in (request.before_clause, request.current_clause, request.after_clause) if v)
            steps = tuple(v.strip() for v in st.text_area("Các bước trước → sau, mỗi dòng một bước (2–6)",
                                                        value="\n".join(request.temporal_steps or defaults), key="kis_sequence_steps").splitlines() if v.strip())
            if steps:
                default_target = steps.index(request.current_clause) if request.current_clause in steps else len(steps) - 1
                target = st.selectbox("Bước cần trả keyframe", range(len(steps)), index=default_target,
                                      format_func=lambda i: f"Bước {i + 1}: {steps[i]}", key="kis_sequence_target")
        if st.button("Tìm tiếp", key="kis_find_next", disabled=not ready, type="primary", width="stretch"):
            try:
                next_request = replace(
                    request, query_id=str(uuid4()), original_text=desired.strip(), limit=limit,
                    english_text_override=request.english_text_override if desired.strip() == request.original_text else None,
                    before_clause=None, current_clause=None, after_clause=None,
                    quoted_text=None, named_entities=None, objects=None, actions=None, attributes=None,
                    entity_constraints=(), feedback_text=None,
                    positive_feedback_ids=(), negative_feedback_ids=(),
                    reference_frame=reference, positive_frames=positive, negative_frames=negative,
                    image_only=image_only, use_temporal_sequence=temporal and not image_only,
                    temporal_mode="manual" if temporal and not image_only else "off",
                    temporal_steps=steps, temporal_target=target, refine_raw=False,
                    interaction_session_id=st.session_state.get("kis_interaction_session"),
                    interaction_turn=st.session_state.get("kis_interaction_turn", 1) + 1,
                    event_source=request.event_source if request.event_source != "unknown" else "user",
                    mode="automatic" if temporal or reference or positive or negative else request.mode,
                )
                with st.spinner("Đang tìm tiếp..."):
                    hits = (context.pipeline.refine(next_request, limit=limit)
                            if reference or positive or negative else context.pipeline.search(next_request, limit=limit, refine_raw=False))
                st.session_state.kis_conditions = (desired.strip(),) if desired.strip() != ". ".join(conditions) else conditions
                st.session_state.kis_results = hits
                st.session_state.kis_last_request = next_request
                st.session_state.kis_interaction_turn = next_request.interaction_turn
                if conversation_enabled:
                    remember_search(next_request, hits)
                st.rerun()
            except Exception:
                st.warning("Chưa thể tìm tiếp. Kiểm tra câu/bước/ảnh mẫu và trạng thái kênh; kết quả cũ được giữ nguyên.")
