"""Evidence-rich Streamlit result gallery."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import streamlit as st

ResultIdentity = tuple[str, str, int]
ImageLoader = Callable[[dict[str, Any]], object | None]
PinChangeHandler = Callable[[ResultIdentity, dict[str, Any], bool, str], None]


def result_identity(result: dict[str, Any]) -> ResultIdentity:
    """Return the stable batch/video/source-frame identity used by widgets."""

    return (
        str(result.get("batch_id", "")),
        str(result["video_id"]),
        int(result["source_frame_idx"]),
    )


def pin_widget_key(identity: ResultIdentity) -> str:
    """Return a bounded widget key without embedding user-controlled text."""

    encoded = "\0".join((identity[0], identity[1], str(identity[2]))).encode("utf-8")
    return "kis_pin_" + hashlib.sha256(encoded).hexdigest()[:20]


def _dispatch_pin_change(
    identity: ResultIdentity,
    result: dict[str, Any],
    widget_key: str,
    callback: PinChangeHandler,
) -> None:
    callback(identity, dict(result), bool(st.session_state.get(widget_key)), widget_key)


def render_result_gallery(
    results: list[dict[str, Any]],
    columns: int = 4,
    show_metadata: bool = True,
    *,
    image_loader: ImageLoader | None = None,
    pinned_identities: set[ResultIdentity] | None = None,
    on_pin_change: PinChangeHandler | None = None,
) -> list[dict[str, Any]]:
    """Render ranked results and return frames pinned by the operator."""

    selected: list[dict[str, Any]] = []
    if not results:
        st.info("Không có kết quả đã lập chỉ mục phù hợp với truy vấn này.")
        return selected
    grid = st.columns(columns)
    for index, result in enumerate(results):
        with grid[index % columns]:
            rank = int(result.get("rank", index + 1))
            st.markdown(f"**#{rank} · {result['video_id']}**")
            if image_loader is not None:
                try:
                    image = image_loader(result)
                except Exception:
                    image = None
                if image is not None:
                    st.image(image, width="stretch")
            st.code(f"khung hình nguồn {result['source_frame_idx']}", language=None)
            if show_metadata:
                timestamp = int(result.get("timestamp_ms", 0)) / 1000
                st.caption(
                    f"{timestamp:.3f}s · cảnh {result.get('shot_id', '?')} · "
                    f"điểm {float(result.get('score', 0.0)):.6f}"
                )
                channels = result.get("channel_ranks", {})
                if channels:
                    st.caption(" · ".join(f"{name} #{rank}" for name, rank in channels.items()))
            identity = result_identity(result)
            widget_key = pin_widget_key(identity)
            if widget_key not in st.session_state:
                st.session_state[widget_key] = identity in (pinned_identities or set())
            checkbox_kwargs: dict[str, Any] = {"key": widget_key}
            if on_pin_change is not None:
                checkbox_kwargs.update(
                    {
                        "on_change": _dispatch_pin_change,
                        "args": (identity, dict(result), widget_key, on_pin_change),
                    }
                )
            if st.checkbox("Ghim", **checkbox_kwargs):
                selected.append(result)
            with st.expander("Bằng chứng"):
                st.json(result.get("evidence", []), expanded=False)
    return selected
