"""
Result Gallery Component
========================
Streamlit component for displaying search results as a visual grid
with keyframe thumbnails, metadata, and selection controls.
"""

from __future__ import annotations


def render_result_gallery(
    results: list[dict],
    columns: int = 4,
    show_metadata: bool = True,
) -> list[dict]:
    """Render a visual gallery of search results.

    Args:
        results: List of result dicts with video_id, frame_id, score,
                 and optionally image_path, answer.
        columns: Number of gallery columns.
        show_metadata: Whether to show score/metadata overlays.

    Returns:
        List of user-selected result dicts.
    """
    # TODO: Implement visual grid with thumbnails, selection checkboxes,
    #       score badges, and click-to-expand detail view
    raise NotImplementedError("render_result_gallery")
