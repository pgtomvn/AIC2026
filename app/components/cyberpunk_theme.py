"""Offline Cyberpunk HUD presentation helpers for the Streamlit console."""

from __future__ import annotations

import hashlib
import html
from collections.abc import Iterable, Sequence

THEME_MARKER = "aic-cyberpunk-hud-v1"


CYBERPUNK_CSS = r"""
<style data-aic-theme="aic-cyberpunk-hud-v1">
@font-face {
  font-family: "Chakra Petch AIC";
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url("/app/static/fonts/ChakraPetch-Regular-vietnamese.woff2") format("woff2");
  unicode-range: U+0102-0103, U+0110-0111, U+0128-0129, U+0168-0169,
    U+01A0-01A1, U+01AF-01B0, U+0300-0301, U+0303-0304, U+0308-0309,
    U+0323, U+0329, U+1EA0-1EF9, U+20AB;
}
@font-face {
  font-family: "Chakra Petch AIC";
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url("/app/static/fonts/ChakraPetch-Regular-latin.woff2") format("woff2");
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6,
    U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122,
    U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}
@font-face {
  font-family: "Chakra Petch AIC";
  font-style: normal;
  font-weight: 600;
  font-display: swap;
  src: url("/app/static/fonts/ChakraPetch-SemiBold-vietnamese.woff2") format("woff2");
  unicode-range: U+0102-0103, U+0110-0111, U+0128-0129, U+0168-0169,
    U+01A0-01A1, U+01AF-01B0, U+0300-0301, U+0303-0304, U+0308-0309,
    U+0323, U+0329, U+1EA0-1EF9, U+20AB;
}
@font-face {
  font-family: "Chakra Petch AIC";
  font-style: normal;
  font-weight: 600;
  font-display: swap;
  src: url("/app/static/fonts/ChakraPetch-SemiBold-latin.woff2") format("woff2");
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6,
    U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122,
    U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}
@font-face {
  font-family: "Share Tech Mono AIC";
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url("/app/static/fonts/ShareTechMono-Regular-latin.woff2") format("woff2");
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6,
    U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122,
    U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}

:root {
  --hud-bg-0: #05050d;
  --hud-bg-1: #0a0a12;
  --hud-panel: rgba(7, 16, 25, 0.88);
  --hud-panel-strong: rgba(5, 10, 18, 0.96);
  --hud-cyan: #00f0ff;
  --hud-green: #00ff66;
  --hud-amber: #ffb700;
  --hud-red: #ff003c;
  --hud-text: #e6fbff;
  --hud-muted: #86a7ad;
  --hud-line: rgba(0, 240, 255, 0.32);
  --hud-font-ui: "Chakra Petch AIC", Bahnschrift, "Segoe UI", sans-serif;
  --hud-font-data: "Share Tech Mono AIC", "Cascadia Mono", Consolas, monospace;
}

html,
body,
.stApp,
[data-testid="stSidebar"] {
  font-family: var(--hud-font-ui);
  color: var(--hud-text);
}

.stApp p,
.stApp label,
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"],
.stApp [role="radiogroup"] {
  font-family: var(--hud-font-ui) !important;
}

.stApp {
  background-color: var(--hud-bg-0);
  background-image:
    linear-gradient(rgba(0, 240, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 240, 255, 0.035) 1px, transparent 1px),
    radial-gradient(circle at 18% 10%, rgba(0, 240, 255, 0.08), transparent 32rem),
    radial-gradient(circle at 92% 76%, rgba(255, 0, 60, 0.05), transparent 30rem),
    linear-gradient(145deg, var(--hud-bg-0), var(--hud-bg-1));
  background-size: 42px 42px, 42px 42px, auto, auto, auto;
}

[data-testid="stAppViewContainer"]::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 99;
  pointer-events: none;
  opacity: 0.075;
  background: repeating-linear-gradient(
    180deg,
    transparent 0,
    transparent 4px,
    rgba(0, 240, 255, 0.22) 5px,
    transparent 6px
  );
  animation: hud-scanline 14s linear infinite;
}

[data-testid="stHeader"] {
  background: rgba(5, 5, 13, 0.72);
  border-bottom: 1px solid rgba(0, 240, 255, 0.15);
  backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] > div:first-child {
  background:
    linear-gradient(180deg, rgba(0, 240, 255, 0.035), transparent 18rem),
    var(--hud-panel-strong);
  border-right: 1px solid var(--hud-line);
  box-shadow: 8px 0 32px rgba(0, 0, 0, 0.42);
}

h1,
h2,
h3,
[data-testid="stHeadingWithActionElements"] {
  color: var(--hud-text);
  font-family: var(--hud-font-ui);
  font-weight: 600;
  letter-spacing: 0.035em;
  text-shadow: 0 0 9px rgba(0, 240, 255, 0.46);
}

code,
kbd,
pre,
[data-testid="stMetricValue"],
.hud-readout,
.hud-status__state,
.hud-timeline__node {
  font-family: var(--hud-font-data);
}

[data-testid="stMarkdownContainer"] a {
  color: var(--hud-cyan);
}

div.stButton > button,
div.stDownloadButton > button {
  min-height: 2.65rem;
  border: 1px solid var(--hud-cyan);
  border-radius: 0;
  clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
  background: linear-gradient(135deg, rgba(0, 240, 255, 0.13), rgba(5, 12, 20, 0.94));
  color: var(--hud-text);
  font-family: var(--hud-font-ui);
  font-weight: 600;
  letter-spacing: 0.025em;
  box-shadow: inset 0 0 16px rgba(0, 240, 255, 0.07), 0 0 10px rgba(0, 240, 255, 0.08);
  transition: border-color 140ms ease, box-shadow 140ms ease, color 140ms ease, transform 140ms ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
  border-color: #80f9ff;
  color: #ffffff;
  box-shadow: inset 0 0 18px rgba(0, 240, 255, 0.14), 0 0 18px rgba(0, 240, 255, 0.34);
  transform: translateY(-1px);
}

div.stButton > button:focus-visible,
div.stDownloadButton > button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="radio"]:focus-visible,
[role="tab"]:focus-visible {
  outline: 2px solid var(--hud-cyan) !important;
  outline-offset: 3px !important;
}

div.stButton > button:disabled,
div.stDownloadButton > button:disabled {
  border-color: rgba(134, 167, 173, 0.38);
  color: rgba(230, 251, 255, 0.48);
  box-shadow: none;
  transform: none;
}

[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div,
[data-testid="stNumberInputContainer"] {
  border-radius: 0 !important;
  border-color: rgba(0, 240, 255, 0.34) !important;
  background: rgba(7, 13, 22, 0.88) !important;
  box-shadow: inset 0 0 14px rgba(0, 240, 255, 0.035);
}

[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-testid="stNumberInputContainer"]:focus-within {
  border-color: var(--hud-cyan) !important;
  box-shadow: 0 0 0 1px var(--hud-cyan), 0 0 14px rgba(0, 240, 255, 0.2) !important;
}

[data-testid="stSegmentedControl"] {
  padding: 0.32rem;
  border: 1px solid rgba(0, 240, 255, 0.28);
  background: rgba(5, 10, 18, 0.9);
  clip-path: polygon(9px 0, 100% 0, 100% calc(100% - 9px), calc(100% - 9px) 100%, 0 100%, 0 9px);
}

[data-testid="stSegmentedControl"] button {
  border-radius: 0 !important;
  font-family: var(--hud-font-data);
  letter-spacing: 0.04em;
}

[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
  color: var(--hud-cyan) !important;
  background: rgba(0, 240, 255, 0.12) !important;
  box-shadow: inset 0 -2px 0 var(--hud-cyan), 0 0 12px rgba(0, 240, 255, 0.14);
}

[data-testid="stAlert"] {
  position: relative;
  border-radius: 0;
  border: 1px solid rgba(0, 240, 255, 0.46);
  background: linear-gradient(100deg, rgba(0, 240, 255, 0.08), rgba(5, 10, 18, 0.94));
  box-shadow: inset 4px 0 0 rgba(0, 240, 255, 0.72);
}

[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
  border-color: rgba(0, 255, 102, 0.52);
  background: linear-gradient(100deg, rgba(0, 255, 102, 0.09), rgba(5, 10, 18, 0.94));
  box-shadow: inset 4px 0 0 rgba(0, 255, 102, 0.74);
}

[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
  border-color: rgba(255, 183, 0, 0.52);
  background: linear-gradient(100deg, rgba(255, 183, 0, 0.09), rgba(5, 10, 18, 0.94));
  box-shadow: inset 4px 0 0 rgba(255, 183, 0, 0.74);
}

[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
  border-color: rgba(255, 0, 60, 0.58);
  background: linear-gradient(100deg, rgba(255, 0, 60, 0.1), rgba(5, 10, 18, 0.94));
  box-shadow: inset 4px 0 0 rgba(255, 0, 60, 0.8);
}

[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"])::after,
[data-testid="stAlert"]:has([data-testid="stAlertContentError"])::after {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  width: 54px;
  height: 5px;
  pointer-events: none;
  background: repeating-linear-gradient(135deg, var(--hud-amber) 0 5px, #090909 5px 10px);
  opacity: 0.7;
}

[data-testid="stAlert"]:has([data-testid="stAlertContentError"])::after {
  background: repeating-linear-gradient(135deg, var(--hud-red) 0 5px, #090909 5px 10px);
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stExpander"],
[data-testid="stFileUploader"] {
  border-color: rgba(0, 240, 255, 0.24) !important;
  background: rgba(5, 12, 20, 0.72);
}

[data-testid="stImage"] img {
  object-fit: contain;
  border: 1px solid rgba(0, 240, 255, 0.31);
  box-shadow: 0 0 18px rgba(0, 240, 255, 0.08);
}

[class*="st-key-hud-kis-card-"],
[class*="st-key-hud-qa-card-"],
[class*="st-key-hud-qa-evidence"],
[class*="st-key-hud-trake-stage-"] {
  position: relative;
  padding: 0.72rem;
  border: 1px solid rgba(0, 240, 255, 0.38) !important;
  border-radius: 0 !important;
  background: linear-gradient(145deg, rgba(0, 240, 255, 0.055), rgba(5, 10, 18, 0.92));
  box-shadow: inset 0 0 24px rgba(0, 240, 255, 0.025), 0 0 12px rgba(0, 240, 255, 0.08);
  clip-path: polygon(13px 0, 100% 0, 100% calc(100% - 13px), calc(100% - 13px) 100%, 0 100%, 0 13px);
}

[class*="st-key-hud-kis-card-"]::before,
[class*="st-key-hud-qa-card-"]::before,
[class*="st-key-hud-qa-evidence"]::before,
[class*="st-key-hud-trake-stage-"]::before {
  content: "+";
  position: absolute;
  top: 2px;
  right: 8px;
  color: var(--hud-cyan);
  font-family: var(--hud-font-data);
  text-shadow: 0 0 8px var(--hud-cyan);
  pointer-events: none;
}

.hud-titlebar {
  position: relative;
  font-family: var(--hud-font-ui);
  margin: 0.2rem 0 1.15rem;
  padding: 1rem 1.15rem 0.95rem 1.3rem;
  border: 1px solid rgba(0, 240, 255, 0.48);
  background: linear-gradient(100deg, rgba(0, 240, 255, 0.08), rgba(5, 10, 18, 0.82) 55%, rgba(255, 0, 60, 0.035));
  box-shadow: inset 4px 0 0 var(--hud-cyan), 0 0 18px rgba(0, 240, 255, 0.12);
  clip-path: polygon(18px 0, 100% 0, 100% calc(100% - 18px), calc(100% - 18px) 100%, 0 100%, 0 18px);
}

.hud-titlebar::after {
  content: "SYS/AIC26 // MULTIMODAL OPS";
  position: absolute;
  top: 7px;
  right: 15px;
  color: rgba(134, 167, 173, 0.62);
  font: 0.68rem/1 var(--hud-font-data);
  letter-spacing: 0.08em;
}

.hud-titlebar__kicker {
  margin-bottom: 0.3rem;
  color: var(--hud-green);
  font: 0.75rem/1.1 var(--hud-font-data);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.hud-titlebar__title {
  color: var(--hud-text);
  font-size: clamp(1.55rem, 3vw, 2.45rem);
  font-weight: 600;
  line-height: 1.08;
  letter-spacing: 0.035em;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.58);
}

.hud-titlebar__subtitle {
  margin-top: 0.45rem;
  color: var(--hud-muted);
  font-size: 0.94rem;
}

.hud-readout {
  font-family: var(--hud-font-data) !important;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
  margin: 0 0 0.5rem;
  padding: 0.44rem 0.55rem;
  border-left: 2px solid var(--hud-tone, var(--hud-cyan));
  border-bottom: 1px solid color-mix(in srgb, var(--hud-tone, var(--hud-cyan)) 35%, transparent);
  background: linear-gradient(90deg, color-mix(in srgb, var(--hud-tone, var(--hud-cyan)) 8%, transparent), transparent);
}

.hud-readout__label {
  color: var(--hud-muted);
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.hud-readout__value {
  min-width: 0;
  overflow-wrap: anywhere;
  color: var(--hud-tone, var(--hud-cyan));
  font-size: 0.91rem;
  text-align: right;
  text-shadow: 0 0 7px color-mix(in srgb, var(--hud-tone, var(--hud-cyan)) 55%, transparent);
}

.hud-status-panel {
  font-family: var(--hud-font-ui);
  margin: 0.3rem 0 0.8rem;
  padding: 0.65rem;
  border: 1px solid rgba(0, 240, 255, 0.28);
  background: rgba(5, 12, 20, 0.82);
  clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
}

.hud-status-panel__title {
  margin-bottom: 0.48rem;
  color: var(--hud-cyan);
  font: 0.74rem/1.1 var(--hud-font-data);
  letter-spacing: 0.12em;
}

.hud-status {
  --hud-tone: var(--hud-muted);
  display: grid;
  grid-template-columns: 8px minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.48rem;
  min-height: 1.7rem;
  border-bottom: 1px solid rgba(134, 167, 173, 0.13);
}

.hud-status:last-child { border-bottom: 0; }
.hud-tone--cyan { --hud-tone: var(--hud-cyan); }
.hud-tone--green { --hud-tone: var(--hud-green); }
.hud-tone--amber { --hud-tone: var(--hud-amber); }
.hud-tone--red { --hud-tone: var(--hud-red); }
.hud-tone--muted { --hud-tone: var(--hud-muted); }

.hud-status__lamp {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--hud-tone);
  box-shadow: 0 0 8px var(--hud-tone);
  animation: hud-status-pulse 2.8s ease-in-out infinite;
}

.hud-status__label {
  min-width: 0;
  overflow: hidden;
  color: var(--hud-text);
  font-size: 0.82rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hud-status__state {
  color: var(--hud-tone);
  font-family: var(--hud-font-data) !important;
  font-size: 0.72rem;
  letter-spacing: 0.05em;
}

.hud-status__detail {
  grid-column: 2 / 4;
  margin: -0.25rem 0 0.28rem;
  color: var(--hud-muted);
  font-size: 0.67rem;
  overflow-wrap: anywhere;
}

.hud-timeline-strip {
  display: flex;
  align-items: center;
  width: 100%;
  margin: 0.55rem 0 1rem;
  padding: 0.75rem;
  overflow-x: auto;
  border: 1px solid rgba(255, 183, 0, 0.42);
  background: linear-gradient(90deg, rgba(255, 183, 0, 0.07), rgba(5, 10, 18, 0.92));
  clip-path: polygon(11px 0, 100% 0, 100% calc(100% - 11px), calc(100% - 11px) 100%, 0 100%, 0 11px);
  scrollbar-color: rgba(255, 183, 0, 0.55) rgba(5, 10, 18, 0.8);
}

.hud-timeline__node {
  flex: 0 0 auto;
  min-width: 3.25rem;
  padding: 0.42rem 0.62rem;
  border: 1px solid var(--hud-amber);
  color: var(--hud-amber);
  background: rgba(255, 183, 0, 0.07);
  text-align: center;
  letter-spacing: 0.06em;
  box-shadow: 0 0 10px rgba(255, 183, 0, 0.12);
}

.hud-timeline__line {
  flex: 1 0 1.7rem;
  min-width: 1.7rem;
  height: 1px;
  background: linear-gradient(90deg, var(--hud-amber), rgba(255, 183, 0, 0.23));
  position: relative;
}

.hud-timeline__line::after {
  content: "›";
  position: absolute;
  right: -1px;
  top: -0.68rem;
  color: var(--hud-amber);
  font: 1.2rem/1 var(--hud-font-data);
}

@keyframes hud-scanline {
  from { transform: translateY(-32px); }
  to { transform: translateY(32px); }
}

@keyframes hud-status-pulse {
  0%, 100% { opacity: 0.72; }
  50% { opacity: 1; }
}

@media (max-width: 1024px) {
  [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
  .hud-titlebar::after { display: none; }
}

@media (min-width: 601px) and (max-width: 1024px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: calc(50% - 0.75rem) !important;
    flex: 1 1 calc(50% - 0.75rem) !important;
  }
}

@media (max-width: 600px) {
  [data-testid="stMainBlockContainer"] { padding-left: 0.72rem; padding-right: 0.72rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: min(100%, 18rem) !important;
    flex: 1 1 100% !important;
  }
  .hud-titlebar { padding: 0.85rem 0.8rem 0.78rem 1rem; }
  .hud-titlebar__title { font-size: 1.48rem; }
  .hud-status { grid-template-columns: 8px minmax(0, 1fr); }
  .hud-status__state { grid-column: 2; }
  .hud-status__detail { grid-column: 2; }
  [data-testid="stSegmentedControl"] button { font-size: 0.76rem; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.001ms !important;
  }
  [data-testid="stAppViewContainer"]::after { transform: none !important; }
}
</style>
"""


_STATUS_TONES = {
    "ONLINE": "green",
    "READY": "green",
    "STANDBY": "cyan",
    "WARMING": "cyan",
    "DEGRADED": "amber",
    "UNKNOWN": "amber",
    "OFFLINE": "red",
    "FAILED": "red",
}
_ALLOWED_TONES = frozenset({"cyan", "green", "amber", "red", "muted"})


def inject_cyberpunk_theme() -> None:
    """Inject the static theme; calling this function performs the only UI side effect."""

    import streamlit as st

    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


def hud_container_key(prefix: str, *identity_parts: object) -> str:
    """Return a stable bounded Streamlit key without exposing dynamic identifiers."""

    normalized = "\0".join(str(value) for value in identity_parts).encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()[:16]
    safe_prefix = "".join(character for character in prefix.lower() if character.isalnum() or character == "-")
    return f"hud-{safe_prefix or 'panel'}-{digest}"


def render_hud_header(kicker: object, title: object, subtitle: object | None = None) -> str:
    """Return a self-contained, escaped HUD title block."""

    subtitle_html = (
        f'<div class="hud-titlebar__subtitle">{html.escape(str(subtitle))}</div>'
        if subtitle is not None
        else ""
    )
    return (
        f'<div class="hud-titlebar" data-hud-component="title">'
        f'<div class="hud-titlebar__kicker">{html.escape(str(kicker))}</div>'
        f'<div class="hud-titlebar__title">{html.escape(str(title))}</div>'
        f"{subtitle_html}</div>"
    )


def render_hud_readout(label: object, value: object, *, tone: str = "cyan") -> str:
    """Return an escaped two-column digital readout."""

    safe_tone = tone if tone in _ALLOWED_TONES else "cyan"
    return (
        f'<div class="hud-readout hud-tone--{safe_tone}" data-hud-component="readout">'
        f'<span class="hud-readout__label">{html.escape(str(label))}</span>'
        f'<span class="hud-readout__value">{html.escape(str(value))}</span>'
        "</div>"
    )


def render_hud_status_panel(
    title: object,
    statuses: Iterable[tuple[object, object, object | None]],
) -> str:
    """Return an escaped status panel for already-sanitized service states."""

    rows: list[str] = []
    for label, state, detail in statuses:
        normalized = str(state).strip().upper() or "UNKNOWN"
        tone = _STATUS_TONES.get(normalized, "amber")
        detail_html = (
            f'<div class="hud-status__detail">{html.escape(str(detail))}</div>'
            if detail
            else ""
        )
        rows.append(
            f'<div class="hud-status hud-tone--{tone}">'
            '<span class="hud-status__lamp" aria-hidden="true"></span>'
            f'<span class="hud-status__label">{html.escape(str(label))}</span>'
            f'<span class="hud-status__state">[{html.escape(normalized)}]</span>'
            f"{detail_html}</div>"
        )
    return (
        '<div class="hud-status-panel" data-hud-component="status-panel">'
        f'<div class="hud-status-panel__title">[{html.escape(str(title))}]</div>'
        f"{''.join(rows)}</div>"
    )


def render_hud_timeline_strip(stage_indices: Sequence[int]) -> str:
    """Return an escaped, horizontally scrollable E1-to-EN timeline strip."""

    nodes: list[str] = []
    for offset, stage_index in enumerate(stage_indices):
        if offset:
            nodes.append('<span class="hud-timeline__line" aria-hidden="true"></span>')
        nodes.append(
            f'<span class="hud-timeline__node" aria-label="Giai đoạn {int(stage_index)}">'
            f"E{int(stage_index)}</span>"
        )
    return (
        '<div class="hud-timeline-strip" data-hud-component="timeline">'
        f"{''.join(nodes)}</div>"
    )


__all__ = [
    "CYBERPUNK_CSS",
    "THEME_MARKER",
    "hud_container_key",
    "inject_cyberpunk_theme",
    "render_hud_header",
    "render_hud_readout",
    "render_hud_status_panel",
    "render_hud_timeline_strip",
]
