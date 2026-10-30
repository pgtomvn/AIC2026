# EviBind Browser implementation handoff

Updated: 2026-10-23 (Asia/Saigon). Update this file before ending a session or approaching the
usage limit; the user explicitly requested a durable progress record for the next agent.

## Goal

Title: **EviBind Browser: Interactive Multimodal Video Retrieval with Query Refinement and
Temporal Browsing**. The user prioritized system/UI work before report editing. Requirements
from the user, in their words where possible:

- Replace Streamlit with a local web UI where "một click là trả kết quả ngay".
- The UI must be as clean as possible and **show no system status** ("không cần hiện trạng thái
  gì hết"); it should look like the two supplied mockups: left *Search stages* (Stage 1 semantic
  query, Stage 2 optional graph refinement, *Search graph*, *Visible relations* checkboxes
  NEXT / MENTIONS / SHARES_ENTITY / GRAPH_RELATION), centre *Retrieved relationship graph*
  ("N final matches in a M-topic subgraph", legend Stage 1 anchor / Final match / Related topic),
  right *Topic inspector* (video - topic k, time range, shots, keyframe, transcript) and an
  *AI Assistant* chat with a *Graph mode* toggle.
- Use the GRASP architecture: an agent interprets each message (Video Search, Question
  Answering, Search Assistance); with Graph mode on, event-order/narrated queries go to the ASR
  topic graph and entity/attribute/relation queries go to the object graph; conventional
  retrieval takes over when graph evidence is missing; every result keeps its path/evidence.
  Topic score `S(u_i,q) = cos(z_q,z_i) + min(Σ_{j∈C} b(r_ij), τ)`, K = 50, τ = 0.20.
  Object score `S(f,q) = |C_sat| / (|V_q| + |E_q|)`; unmentioned ≠ absent.
- The user prefers no clarifying-question rounds; communicate in Vietnamese.

## Later user decisions (2026-10-23)

- UI language: **English**; colour: **green**; friendly wording chosen by the agent.
- One search workflow only: no KIS / Q&A / TRAKE tabs. Q&A is asked inside the Details panel
  (or the chat); TRAKE is hidden from the browser (API and legacy console keep it).
- Frame captioning is **not** needed (the object graph uses organizer detections).
- Drive C: is full (~0.1 GB free). Install or write anything new on D: only; set
  `TEMP`/`TMP` to `D:\AIC2026\data\cache\scratch` for heavy runs.
- Integrate DRES submission. The organizer's real calls look like
  `POST https://eventretrieval.one/api/v2/submit/<evaluationId>?session=<id>` with body
  `{"answerSets":[{"answers":[{"mediaItemName":"L21_V013","start":758933,"end":758933}]}]}`;
  each query has its own evaluation id.

## Current state (branch `feat/evibind-browser`, nothing committed yet)

Verified 2026-10-23:
- `.\scripts\check.ps1 -PytestTempRoot D:\AIC2026\data\test-temp\check2`: ruff clean,
  **1793 passed, 3 skipped**.
- Full B1 graphs built in `data/graphs/aic25-b1-reference/` (36 MB): topic 8,935 topics /
  859 videos / 3,699 entities / 8,076 NEXT + 30,305 semantic edges; object 168,470 frames /
  1.12 M detections / 514 classes. Topic search ≈ 27 ms after load; object search ≈ 1.2 s.
- Live browser check on port 8091 (Qdrant/Meilisearch/Ollama were not running): graph search
  with stage 2, timeline view with thumbnails, Details (transcript, previous/next parts, nearby
  frames), DRES connect + submit against a scratch fake DRES server: the server received
  `{"answerSets":[{"answers":[{"mediaItemName":"L26_V218","start":103320,"end":103320}]}]}` on
  the chosen evaluation with the session as a query value. Not verified live: conventional KIS
  fallback (needs Qdrant), Q&A answers (needs Ollama), the LLM router (needs Ollama), a real DRES.

Components:
- `src/graph/topic_graph.py`, `src/graph/object_graph.py`; builder `scripts/build_topic_graph.py`
  (`--inspect` default, `--sample VIDEO`, `--full`, `--kind topic|object|both`; B1 only; never
  overwrites; prints progress on stderr). Edge weights from float rounding just above 1 are
  clamped (the first full artifact failed to load because of this).
- `src/browser_api.py`: assistant (`assistant.router: auto` → local Ollama with rules fallback),
  graph routing (event/narration → topic; counts, colours, clothing, spatial words → object;
  otherwise topic), graph search (conventional fallback only when no batch has evidence),
  topic NEXT chain endpoint, and installs the DRES routes.
- `src/submission/dres.py`: DRES client API v2 (login or pasted session ID, evaluation list,
  current task, submit, optional result log), server-side canonical time/frame resolution,
  preview endpoint, per-request evaluation id, JSONL log at `logs/dres-submissions.jsonl`.
  Config `services.dres` (formats configurable; Q&A `QA-{answer}-{video_id}-{time_ms}`,
  TRAKE `TR-{video_id}-{frames}` are community conventions, not yet confirmed by the BTC guide).
  Tests: `tests/test_dres.py` (fake server, no network).
- `src/embedding/text_encoder.py`: `_input_limit()` (PhoBERT 256 usable positions).
- Runtime contract v9 (features include `browser.dres_submission`); legacy Streamlit expects 9.
- `app/web/*`: English, green, three panels; see `docs/guides/evibind_browser.md`.

Live system (2026-10-23 21:30): `run_app.ps1 -StartServices` runs in a Terminal tab; Docker
Qdrant + Meilisearch healthy; readiness "Full Multi-Modal" (organizer CLIP 177,321, SigLIP2
176,707, OCR 1,167,264, ASR 131,370, metadata); warm-up ready. Measured warm: KIS ≈ 1.7–3.8 s,
topic graph ≈ 0.35 s, object graph ≈ 2 s, Q&A inject ≈ 0.4 s.

VLM findings: NVIDIA driver 546.18 (CUDA 12.3) is too old for Ollama 0.32.5's CUDA runtimes
(`cuda_v13` and forced `cuda_v12` both crash with "device kernel image is invalid"); Vulkan
discovers no device. Local `.env` now has `AIC_QA_DEVICE=cpu` and `OLLAMA_LLM_LIBRARY=vulkan`
(user-owned file, not in Git). CPU Q&A: first question per frame ≈ 75 s, follow-ups ≈ 4 s.
Added: background VLM prewarm on page load, LLM router only for questions without a frame,
shared Ollama options (no model reload), opt-in Gemini fast answer in Details.

Update 2026-10-23 22:15 (user decisions: Gemini only; no separate graph mode):
- One search: `POST /v1/search` (`src/browser_api.py`) fuses conventional KIS with the topic
  graph (parallel) and, for object cues, the object graph; `src/graph/fusion.py`, weights in
  `graph.fusion`. UI has no Graph switch; *Graph* view appears when topic evidence exists.
  Stage 2 is combined with stage 1 for conventional retrieval (legacy refine replaced the query
  and rejected text-only feedback across batches). Evaluation (scratch harness, user qrels):
  tune R@20 0.28→0.44, MRR 0.07→0.15; locked test (5) R@20 0.20→0.20, MRR 0.20→0.10.
- Gemini is the Q&A model; `.env` has `AIC_LOCAL_VLM=off` (`services.ollama_vlm.enabled_env`).
  Disabled local VLM never touches the network (`OllamaQwenReasoner(enabled=False)`, readiness
  `disabled_by_profile`); this removed ~2 s from every search and 2–4 s from every answer.
  Launcher skips Ollama and passes `--allow-degraded` in that mode. Florence disabled.
- Recycled (Windows Recycle Bin, restorable): `models/ollama`, `models/huggingface/florence-2-base`.
  Kept: search encoders (open_clip, siglip2, vietnamese-embedding, opus-mt/translation) and
  extraction models (whisper, paddleocr, rapidocr, vietocr) pending the user's decision.
- NVIDIA driver update was not performed (system software change; user action). With Gemini
  only it is optional (SigLIP2 on CUDA already works with driver 546.18).
- check.ps1: 1810 passed, 3 skipped.

Update 2026-10-24 (UI round 3, user feedback: graph not useful, preview too small, `>>`):
- Graph view rewritten as one timeline row per video, rows in result order, pinned and zoomed
  on the selected video (also when a grid frame is selected, via the part containing it);
  arcs for non-NEXT links, entity names under parts; no force layout or entity nodes.
- Resizable columns (`--left-w`, `--right-w`) and Details/assistant split (`--inspector-h`),
  stored in localStorage `evibind.layout`; Details flex children no longer shrink (the photo was
  squashed). Large frame viewer dialog with ← / → over nearby frames.
- `a >> b >> c` in stage 1 calls `/v1/trake/search` (≈7 s warm) and renders sequence rows; DRES
  *Submit sequence* sends `task: trake` with the ordered frames.

Update 2026-10-24 (round 4):
- Removed the ">>" tip from the search panel and the assistant greeting (the syntax still works).
- Details no longer has its own question box: questions go through the AI chat, which shows
  "Asking about <video · time> ×" above the input for the selected frame.
- Fixed a hang: a Gemini call on a stale pooled connection blocked forever while holding the
  reasoner lock, so every later question hung. `GeminiFlashReasoner._call_with_deadline` runs the
  SDK call on a daemon thread with a hard deadline (timeout + 2 s), then drops the client.
  Test: `tests/test_gemini_deadline.py`. check.ps1: 1812 passed, 3 skipped.

Update 2026-10-24 (round 5, "qa_all_providers_failed"):
- Root causes: (1) this network resolves Google over IPv6 but cannot route it, so the SDK waited
  on every IPv6 address; (2) the API key is on the Gemini free tier: 20 generateContent requests
  per model per day (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), and the Interactions
  SDK call stalls instead of failing on HTTP 429.
- `src/vlm/gemini.py` now calls REST `generateContent` through httpx (IPv4 with
  `services.gemini_vlm.ip_family: ipv4`), tries `model` then `fallback_models`
  (gemini-3.1-flash-lite → gemini-3.5-flash-lite → gemini-3.6-flash), skips a 429'd model until
  its RetryInfo reset, retries once without `thinkingConfig` if a model rejects it, and keeps the
  hard deadline. Timeout 15 s. With the local VLM disabled the router no longer appends it, so
  the UI shows the real Gemini reason (quota, timeout, network, key).
- Chat: with a selected frame every message is a question about that frame (no search).
  Dataset selector removed (searches use every dataset the runtime serves).
- Measured in the UI: 1.9-5.7 s per answer. check.ps1: 1818 passed, 3 skipped.
- For competition use enable billing on the API key's Google project (paid tier limits are far
  above 20/day/model); otherwise answers stop once all three models hit their daily quota.

Remaining:
0. Ask the user to update the NVIDIA driver (≥ 570); then remove `AIC_QA_DEVICE=cpu` and
   `OLLAMA_LLM_LIBRARY` from `.env` and re-measure GPU Q&A.
1. Exercise with services up (`.\scripts\run_app.ps1 -StartServices`): conventional search,
   Q&A via Ollama, LLM routing, CSV export, and a real DRES warm-up round.
2. Confirm Q&A/TRAKE DRES text formats with the organizer guide; adjust `services.dres.formats`.
3. Decide when to remove Streamlit; then commit on `feat/evibind-browser` when the user asks
   (no Claude co-author trailer — user preference).

Update 2026-10-25 (paper screenshots for the VBS camera-ready):
- Fixed: the Details photo showed only the top-left of the frame (the 768 px image had no size
  rule inside the 16:9 `.photo` box); `app.css` now sizes `.inspector .photo img` to the box.
- Fixed: the *Before and after* / *Previous and next parts* strips went blank after any
  re-render (a Q&A answer, Select). `renderInspector()` now refills them through
  `loadInspectorContext(item, seq)` (cached requests); `inspect()` only sets the selection.
- Verified in headless Chrome: lions query, frame selected, Q&A answer "2", 9 strip images kept,
  no page errors.

Update 2026-10-27 (search turns, VBS camera-ready):
- Added *Search turns* with *Undo* / *First results* to the web UI (`app.js` `saveTurn`,
  `restoreTurn`, `renderRefine`). Turns are kept client-side (last 20 plus the first) and
  restored without a new request; a new first description resets them. Verified in the browser:
  Similar frames creates turn 2, Undo restores turn 1's exact list in a few milliseconds.
- Not added: "Not relevant" (negative) feedback buttons. A simulated feedback round on the 23
  labelled queries (B1, top 20 seen) improved 3 and worsened 5 queries against a control that
  runs the same refine path without marks (marking 1 frame: 0 better, 7 worse). Marking frames
  of the answer video as relevant also hurt. Scripts and results are in the local paper
  workspace (`eval_feedback.mjs`, `control_refine.mjs`).
- Observation to follow up: `/v1/kis/refine` without any marks ranked the answers better than
  `/v1/kis/search` on the same queries (B1 only: R@20 0.48 vs 0.26). The difference is in the
  search path (e.g. `refine_raw`), not in feedback; it was not investigated.
- Follow-up resolved (same day): the refine path won because it searches with the normalized
  mean of all semantic views (global, action, subjects, attributes), while `/v1/kis/search`
  used the global view only; raw refinement is `not_applicable` on reference-only B1 and
  changed nothing, and averaging text chunks instead of views made results worse. Added
  `semantic_views.fusion: mean_vector` (`textual_kis.py`) and enabled it in
  `config/settings.yaml`; numbers and costs are in `docs/design/kis_finegrained_retrieval.md`.

Update 2026-10-30 (B2 object graph, final paper numbers):
- `scripts/build_topic_graph.py --batch-id aic26-b2 --kind object --full` now builds the B2
  object graph from the three organizer ZIPs, keyed by `canonical_ordinal_frames`; the result
  is `data/graphs/aic26-b2/object` (217,570 frames, 2,459,579 boxes, 509 classes, 40 MB).
- B2 has no topic graph: no B2 ASR is on this machine. The builder refuses it until the ASR is
  imported and its timing is checked against B2's mixed ordinal/VFR identities.
- `GraphService.object_hits` now alternates batches on equal fractions; before, B1 filled the
  limit and B2 object hits never reached an all-batch search.
- `services.gemini_vlm.fallback_models` is empty: Q&A uses one Gemini model, so at about 20 free
  requests per day a demo needs billing enabled.
