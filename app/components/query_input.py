"""Streamlit query controls shared by the interactive Textual KIS UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.retrieval.query_plan import build_query_plan


def _comma_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def render_text_encoding_diagnostics(statuses: dict[str, dict[str, Any]]) -> None:
    """Explain dense-query chunking without exposing tokenizer internals."""

    for channel_name, status in sorted(statuses.items()):
        if not isinstance(status, dict) or not status.get("query_chunked"):
            continue
        chunk_count = int(status.get("chunk_count", 1))
        dropped_tokens = int(status.get("dropped_tokens", 0))
        if dropped_tokens > 0:
            st.warning(
                f"Kênh {channel_name} đã chia truy vấn thành {chunk_count} phần; "
                f"{dropped_tokens} token vượt giới hạn chưa được mã hóa. "
                "Phần đầu và cuối vẫn được giữ."
            )
        else:
            st.caption(
                f"Kênh {channel_name} đã chia truy vấn thành {chunk_count} phần "
                "để giữ đủ nội dung."
            )


def render_query_input(
    query_type: str = "Textual KIS",
    key_prefix: str = "query",
    *,
    disabled: bool = False,
) -> dict:
    """Render original and temporal-clause inputs without rewriting the query."""

    if query_type != "Textual KIS":
        return {"type": query_type, "submitted": False}
    text = st.text_area(
        "Mô tả sự kiện bằng ngôn ngữ tự nhiên",
        key=f"{key_prefix}_text",
        placeholder="Một người mặc áo đỏ đi bộ qua cây cầu...",
        height=110,
    )
    plan = None
    if text.strip():
        plan = build_query_plan(text)
        source_key = f"{key_prefix}_plan_source"
        if st.session_state.get(source_key) != text:
            for name in ("quoted_text", "named_entities", "objects", "actions", "attributes"):
                st.session_state[f"{key_prefix}_{name}"] = ", ".join(getattr(plan, name))
            for name in ("before", "current", "after"):
                st.session_state[f"{key_prefix}_{name}"] = getattr(plan, name) or ""
            st.session_state[source_key] = text

    with st.expander("Kế hoạch truy xuất có thể chỉnh sửa", expanded=False):
        if plan is not None:
            st.caption(
                f"Chuẩn hóa: {plan.normalized_text} · Bỏ dấu: {plan.accent_folded_text}"
            )
        quoted_text = st.text_input("Văn bản trích dẫn", key=f"{key_prefix}_quoted_text")
        named_entities = st.text_input(
            "Thực thể được nhắc đến", key=f"{key_prefix}_named_entities"
        )
        objects = st.text_input("Đối tượng", key=f"{key_prefix}_objects")
        actions = st.text_input("Hành động", key=f"{key_prefix}_actions")
        attributes = st.text_input("Thuộc tính", key=f"{key_prefix}_attributes")
        before = st.text_input("Trước đó", key=f"{key_prefix}_before")
        current = st.text_input("Hiện tại", key=f"{key_prefix}_current")
        after = st.text_input("Sau đó", key=f"{key_prefix}_after")
    submitted = st.button(
        "Tìm kiếm",
        type="primary",
        width="stretch",
        key=f"{key_prefix}_submit",
        disabled=disabled,
    )
    return {
        "type": query_type,
        "text": text,
        "before_clause": before or None,
        "current_clause": current or None,
        "after_clause": after or None,
        "quoted_text": _comma_values(quoted_text),
        "named_entities": _comma_values(named_entities),
        "objects": _comma_values(objects),
        "actions": _comma_values(actions),
        "attributes": _comma_values(attributes),
        "submitted": submitted,
    }
