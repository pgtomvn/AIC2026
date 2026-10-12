"""Evidence-rich Streamlit result gallery."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import streamlit as st

from app.components.cyberpunk_theme import hud_container_key, render_hud_readout

ResultIdentity = tuple[str, str, int]
ImageLoader = Callable[[dict[str, Any]], object | None]
PinChangeHandler = Callable[[ResultIdentity, dict[str, Any], bool, str], None]
FrameActionHandler = Callable[[dict[str, Any]], None]


def result_identity(result: dict[str, Any]) -> ResultIdentity:
    """Return the stable batch/video/source-frame identity used by widgets."""

    if result.get("batch_id") == "aic26-b2":
        from src.ingest.frame_identity import image_identity

        return image_identity(result)
    return (
        str(result.get("batch_id", "")),
        str(result["video_id"]),
        int(result["source_frame_idx"]),
    )


def pin_widget_key(identity: ResultIdentity) -> str:
    """Return a bounded widget key without embedding user-controlled text."""

    encoded = "\0".join((identity[0], identity[1], str(identity[2]))).encode("utf-8")
    return "kis_pin_" + hashlib.sha256(encoded).hexdigest()[:20]


def result_score(result: dict[str, Any]) -> float | None:
    """Return the first finite score exposed by current or legacy KIS contracts."""

    for field in ("fused_score", "score", "retrieval_score"):
        value = result.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
    return None


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
    on_send_to_qa: FrameActionHandler | None = None,
    on_open_scrubber: FrameActionHandler | None = None,
    on_reference: FrameActionHandler | None = None,
    on_positive: FrameActionHandler | None = None,
    on_negative: FrameActionHandler | None = None,
) -> list[dict[str, Any]]:
    """Render ranked results and return frames pinned by the operator."""

    selected: list[dict[str, Any]] = []
    if not results:
        st.info("Không có kết quả đã lập chỉ mục phù hợp với truy vấn này.")
        return selected
    grid = st.columns(columns)
    for index, result in enumerate(results):
        with grid[index % columns]:
            identity = result_identity(result)
            with st.container(
                key=hud_container_key("kis-card", *identity),
                border=True,
            ):
                rank = int(result.get("rank", index + 1))
                st.markdown(
                    render_hud_readout(f"TARGET {rank:03d} · {'B02' if identity[0] == 'aic26-b2' else 'B01'}", result["video_id"]),
                    unsafe_allow_html=True,
                )
                if image_loader is not None:
                    try:
                        image = image_loader(result)
                    except Exception:
                        image = None
                    if image is not None:
                        st.image(image, width="stretch")
                coordinate = result.get("submission_value", result["source_frame_idx"])
                st.code(f"{result['video_id']},{coordinate}", language=None)
                if show_metadata:
                    timestamp = int(result.get("timestamp_ms", 0)) / 1000
                    suffix = f"ảnh {identity[2]} · {result.get('submission_unit', 'source_frame_idx')}" if identity[0] == "aic26-b2" else f"source frame {result['source_frame_idx']}"
                    st.caption(f"{timestamp:.3f}s · {suffix}")
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
                action_digest = hashlib.sha256(
                    "\0".join((identity[0], identity[1], str(identity[2]))).encode(
                        "utf-8"
                    )
                ).hexdigest()[:20]
                if on_reference is not None and st.button(
                    "Dùng làm ảnh mẫu", key=f"kis_reference_{action_digest}", width="stretch",
                ):
                    on_reference(dict(result))
                if on_positive is not None or on_negative is not None:
                    for column, label, action, callback in zip(
                        st.columns(2), ("Gần đúng", "Không phù hợp"),
                        ("positive", "negative"), (on_positive, on_negative), strict=True,
                    ):
                        if callback is not None and column.button(label, key=f"kis_{action}_{action_digest}", width="stretch"):
                            callback(dict(result))
                if on_open_scrubber is not None or on_send_to_qa is not None:
                    neighbors, qa = st.columns(2)
                    if on_open_scrubber is not None and neighbors.button(
                        "±5 khung", key=f"kis_scrub_{action_digest}", width="stretch",
                    ):
                        on_open_scrubber(dict(result))
                    if on_send_to_qa is not None and qa.button(
                        "Visual Q&A", key=f"kis_to_qa_{action_digest}", width="stretch",
                    ):
                        on_send_to_qa(dict(result))
                with st.expander("Bằng chứng"):
                    if show_metadata:
                        score = result_score(result)
                        st.caption(f"Batch {identity[0]} · cảnh {result.get('shot_id', '?')}")
                        if score is not None:
                            st.caption(f"Điểm hợp nhất {score:.6f}")
                        channels = result.get("channel_ranks", {})
                        if channels:
                            st.caption(" · ".join(f"{name} #{rank}" for name, rank in channels.items()))
                    st.json(result.get("evidence", []), expanded=False)
                    evidence = result.get("evidence", {})
                    items = evidence.get("items", ()) if isinstance(evidence, dict) else evidence
                    for proof in items:
                        for step in proof.get("temporal_steps", ()):
                            st.caption(f"Bước {step['step'] + 1}: frame {step['source_frame_idx']} · {step['timestamp_ms'] / 1000:.3f}s")
                            if on_open_scrubber is not None and st.button(
                                f"Xem bước {step['step'] + 1}", key=f"kis_proof_{action_digest}_{step['step']}"
                            ):
                                on_open_scrubber({**result, **step})
    return selected
