"""
Export Panel Component
======================
Streamlit component for exporting selected results as CSV
in the competition submission format.
"""

from __future__ import annotations


def render_export_panel(
    results: list[dict],
    query_type: str = "Textual KIS",
    output_dir: str = "outputs/submissions",
) -> None:
    """Render the export / submission panel.

    Args:
        results: Ranked list of result dicts to export.
        query_type: Query type for format selection.
        output_dir: Directory to save exported files.
    """
    # TODO: Implement CSV export with proper submission formatting:
    #   - Textual KIS: video_id, frame_id
    #   - Visual Q&A: video_id, frame_id, answer
    #   - TRAKE: video_id, frame_id_1, ..., frame_id_N
    raise NotImplementedError("render_export_panel")
