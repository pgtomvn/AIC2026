"""
AIC 2026 — Streamlit GUI
=========================
Main entry point for the interactive human-in-the-loop search interface.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Main Streamlit application."""

    # ── Page Config ──
    st.set_page_config(
        page_title="AIC 2026 — Video Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ──
    with st.sidebar:
        st.title("🎯 AIC 2026")
        st.markdown("**Multimodal Video Search**")
        st.divider()

        query_type = st.radio(
            "Query Type",
            options=["Textual KIS", "Visual Q&A", "TRAKE"],
            index=0,
        )

        st.divider()
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Top-K Results", min_value=1, max_value=100, value=20)
        use_reranker = st.checkbox("Enable Reranking", value=True)

    # ── Main Content ──
    st.title("🔍 Multimodal Video Search & Retrieval")
    st.caption("AIC 2026 — Ho Chi Minh City AI Challenge")

    # Query Input
    if query_type == "Textual KIS":
        query = st.text_area(
            "Enter your text description:",
            placeholder="A person wearing a red shirt walking across a bridge...",
            height=100,
        )
        search_btn = st.button("🔎 Search", type="primary", use_container_width=True)

        if search_btn and query:
            with st.spinner("Searching..."):
                # TODO: Integrate TextualKISPipeline
                st.info("⚠️ Pipeline not yet connected. Showing placeholder.")
                _show_placeholder_results()

    elif query_type == "Visual Q&A":
        question = st.text_input(
            "Enter your visual question:",
            placeholder="What color is the car in the foreground?",
        )
        search_btn = st.button("🔎 Answer", type="primary", use_container_width=True)

        if search_btn and question:
            with st.spinner("Processing..."):
                # TODO: Integrate VisualQAPipeline
                st.info("⚠️ Pipeline not yet connected. Showing placeholder.")
                _show_placeholder_results()

    elif query_type == "TRAKE":
        st.markdown("Enter sequential event descriptions (one per line):")
        events_text = st.text_area(
            "Events:",
            placeholder="Event 1: A car enters the frame\nEvent 2: The car stops at a red light\nEvent 3: A pedestrian crosses",
            height=150,
        )
        search_btn = st.button("🔎 Align", type="primary", use_container_width=True)

        if search_btn and events_text:
            with st.spinner("Aligning events..."):
                # TODO: Integrate TRAKEPipeline
                st.info("⚠️ Pipeline not yet connected. Showing placeholder.")

    # ── Footer ──
    st.divider()
    st.caption("AIC 2026 Competition — Multimodal Video Search & Retrieval Engine")


def _show_placeholder_results() -> None:
    """Display placeholder results for development/testing."""
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Result {i + 1}**")
            st.markdown(f"Video: `L21_V{i+1:04d}`")
            st.markdown(f"Frame: `{(i+1)*100}`")
            st.markdown(f"Score: `{0.95 - i*0.05:.2f}`")


if __name__ == "__main__":
    main()
