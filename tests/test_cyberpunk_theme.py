from __future__ import annotations

import hashlib
from pathlib import Path

from app.components.cyberpunk_theme import (
    CYBERPUNK_CSS,
    THEME_MARKER,
    hud_container_key,
    render_hud_header,
    render_hud_readout,
    render_hud_status_panel,
    render_hud_timeline_strip,
)
from src import PROJECT_ROOT


def test_theme_is_offline_accessible_and_uses_stable_hooks() -> None:
    assert THEME_MARKER in CYBERPUNK_CSS
    assert "https://" not in CYBERPUNK_CSS
    assert "http://" not in CYBERPUNK_CSS
    assert "/app/static/fonts/" in CYBERPUNK_CSS
    assert "prefers-reduced-motion: reduce" in CYBERPUNK_CSS
    assert "pointer-events: none" in CYBERPUNK_CSS
    for token in ("#05050d", "#0a0a12", "#00f0ff", "#00ff66", "#ffb700", "#ff003c"):
        assert token in CYBERPUNK_CSS
    for hook in (
        'data-testid="stSegmentedControl"',
        'st-key-hud-kis-card-',
        'st-key-hud-qa-card-',
        'st-key-hud-trake-stage-',
    ):
        assert hook in CYBERPUNK_CSS


def test_dynamic_hud_values_are_escaped() -> None:
    hostile = '<img src=x onerror="alert(1)">'
    outputs = (
        render_hud_header(hostile, hostile, hostile),
        render_hud_readout(hostile, hostile),
        render_hud_status_panel(hostile, [(hostile, hostile, hostile)]),
    )
    for output in outputs:
        assert hostile not in output
        assert "&lt;img" in output
        assert "onerror=&quot;alert(1)&quot;" in output


def test_hud_keys_are_stable_bounded_and_hide_source_identity() -> None:
    first = hud_container_key("kis-card", "batch", "video/user-controlled", 42)
    second = hud_container_key("kis-card", "batch", "video/user-controlled", 42)
    assert first == second
    assert first.startswith("hud-kis-card-")
    assert "video" not in first
    assert len(first) < 48


def test_timeline_preserves_stage_order() -> None:
    timeline = render_hud_timeline_strip([1, 2, 3, 4])
    assert timeline.index(">E1<") < timeline.index(">E2<") < timeline.index(">E4<")
    assert timeline.count("hud-timeline__line") == 3


def test_vendored_font_hashes_and_static_serving_config() -> None:
    expected = {
        "ChakraPetch-Regular-latin.woff2": "7d75be85dd1627e27ec151e9a3701e661bc4f8c2a93611bdd92f3a05d6bbd942",
        "ChakraPetch-Regular-vietnamese.woff2": "54aa4ca0e3ae0ace71b1ec1390b59441426857e0e4f23aa2dd1cef92f38f5793",
        "ChakraPetch-SemiBold-latin.woff2": "a5888696e9eb1b4bbbecc8eb3922b8369f49d4bddb72263e033cbd17f399be76",
        "ChakraPetch-SemiBold-vietnamese.woff2": "0d405c8eaad8bf29383d68552d17ce01b2abc704f2c66753625bb38f5307a436",
        "ShareTechMono-Regular-latin.woff2": "41e6b9f297f7d9a2df2aaa274092f76d2f72711a15ca455f7f4f4f92caf16b72",
    }
    font_root = Path(PROJECT_ROOT) / "app" / "static" / "fonts"
    for name, digest in expected.items():
        assert hashlib.sha256((font_root / name).read_bytes()).hexdigest() == digest
    config = (Path(PROJECT_ROOT) / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "enableStaticServing = true" in config
    assert (font_root / "OFL-ChakraPetch.txt").is_file()
    assert (font_root / "OFL-ShareTechMono.txt").is_file()
