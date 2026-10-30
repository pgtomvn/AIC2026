# EviBind Browser

EviBind Browser is the local HTML/CSS/JavaScript interface for **Interactive Multimodal Video
Retrieval with Query Refinement and Temporal Browsing**. FastAPI serves the page and handles
search in the same process and on the same loopback port. No Node installation, frontend build,
CDN, or separate UI server is required.

## Launch

Prepare the existing runtime, models, indexes, and local services as described in the
[operations guide](operations.md) or [Vietnamese setup guide](team_setup_vi.md). Then run:

```powershell
.\scripts\run_app.ps1 -Profile laptop -BatchId aic25-b1-reference -ApiPort 8080
```

Open [EviBind Browser](http://127.0.0.1:8080/). Keep the terminal open. `Ctrl+C` cleans up only
the API and Ollama processes owned by this launcher. Existing Ollama and Docker services remain
under the operator's control. `-StartServices` remains an explicit option for starting the two
database containers; the launcher never imports data or downloads models.

`-ApiPort` now selects the shared browser/API port. The old `-Port` option is accepted for
compatibility but applies only with `-LegacyStreamlit`:

```powershell
.\scripts\run_app.ps1 -Profile laptop -BatchId aic25-b1-reference -LegacyStreamlit -ApiPort 8080 -Port 8504
```

This opens the legacy UI at port 8504 while the API remains on 8080. Existing runtime stamps,
profile/batch resolution, loopback binding, port ownership checks, startup doctor, and search
readiness gates still apply to both modes. Streamlit remains in the lockfile and its regression
checks remain available during migration.

For API development, `python scripts/serve.py --port 8080` also serves the browser. It bypasses
launcher provisioning checks; it is not a replacement for the canonical competition launch.

## Layout

The page has three panels, following the GRASP-style mockups used for the project:

- **Search**: *Describe the moment* (stage 1) and *Narrow it down* (optional stage 2); in the
  *Graph* view, *Show links* filters (`NEXT`, `MENTIONS`, `SHARES_ENTITY`, `GRAPH_RELATION`).
  Count/colour/spatial queries show *Looking for* (the parsed objects and relations). The footer
  holds the *Selected* export dialog and the *DRES* connection; searches cover every dataset
  the runtime serves.
- **Results**: one fused frame grid with *said*/*objects* evidence tags and detection boxes.
  *Graph* shows one timeline row per video (rows follow the result ranking): the matched
  transcript parts in time order joined by `NEXT` arrows, `···` where parts are not consecutive,
  dashed arcs for `GRAPH_RELATION`/`SHARES_ENTITY`, and entity names under parts for
  `MENTIONS`. The selected video's row is pinned, highlighted, and zoomed; click a row name or
  part to pin it, double-click the background to see every row.
- **Sequences**: separating steps with `>>` (`bật bếp >> xào rau >> cho thịt ra đĩa`) searches
  for videos where the steps happen in that order (the TRAKE alignment API). Each result row
  shows step 1 → step 2 → step 3 frames; *Submit sequence* sends the frame list to DRES. With only `NEXT` shown, the six best videos are drawn as
  timelines; otherwise a deterministic force layout is used. Drag to pan, scroll to zoom,
  double-click to fit; clicking an entity copies it into stage 2.
- **Details** and **AI Assistant**: the selected part's video, time range, shots, keyframe and
  transcript; *Select*, *Submit to DRES*, and *Similar frames*; strips of previous/next
  parts and nearby frames for temporal browsing. Click the picture for a large viewer (← / →
  walk the nearby frames). Drag the dividers between the three columns and between Details and
  the assistant to resize them; sizes are remembered per browser, double-click resets.

Visual Q&A is asked from the Details panel or the chat; the TRAKE workflow stays available
through the API and the legacy console but is not shown in the browser. The page deliberately
shows no service or warm-up status; use `/v1/runtime/warmup`, `/readyz`, and batch readiness
from the API when diagnosing a blocked search.

## Interaction and performance

The browser keeps view state and updates only affected components through HTTP calls. Selecting
a result, filtering relations, panning, or switching views never reruns a Python application.
Search, refinement, uncached images, and VLM answers still require backend work; the UI change
alone does not change retrieval quality or inference time. A topic-graph search is an
in-memory NumPy search over the loaded artifact plus one CPU text encoding of the query.

## Search

There is one search. `POST /v1/search` runs conventional retrieval (visual CLIP/SigLIP2 vectors
in Qdrant, OCR/ASR/metadata text in Meilisearch, already fused by the pipeline) and, in
parallel, the ASR topic graph (about 0.2 s). Results are combined by weighted reciprocal rank
(`graph.fusion`): a frame whose time falls inside a matched transcript topic of the same video
gains the topic's rank term, and a topic keyframe with no conventional frame nearby is added on
its own. For count, colour, clothing, or spatial queries the object graph scores the
conventional candidates plus object-index candidates by the satisfied share of constraints.
Without graph evidence the conventional ranking is returned unchanged. Cards show *said* or
*objects* when that evidence supported the rank; *Graph* switches to the relationship view.

Stage 2 (*Narrow it down*) is searched together with stage 1 by conventional retrieval and
re-ranks the stage-1 topic subgraph. *Similar frames* adds visual feedback from the selected
frame. Every search is a turn: *Search turns* in the left column appears from the second turn,
and *Undo* / *First results* restore any of the last 20 turns at once without searching again.
A new first description starts a new set of turns.

On the 23 user-labelled queries in `outputs/evaluation/user-qrels-20260905-v1` (B1, correct =
same video within ±5 s of the anchor): train+validation (18) R@5 0.17 → 0.33, R@20 0.28 → 0.44,
MRR 0.07 → 0.15; locked test (5) R@5/R@20 unchanged at 0.20, MRR 0.20 → 0.10 (one query moved
from rank 1 into the top 5). The set is small; treat this as a diagnostic, not a benchmark.

Since 2026-10-27 conventional search averages the semantic views of a query
(`retrieval.semantic_views.fusion: mean_vector`). On the same 23 queries searched over B1+B2,
the deployed search moved from MRR 0.155 to 0.236 (R@1 0.09 → 0.17, R@20 0.30 → 0.35); on top
of the averaged views the graphs add about 0.02 MRR. Details and costs are in
[`../design/kis_finegrained_retrieval.md`](../design/kis_finegrained_retrieval.md).

## Assistant and Visual Q&A

Questions are typed in the AI chat. While a frame is selected the chat shows "Asking about
<video · time>" above the input and every message is a question about that frame (clear it with
× to search from the chat). Answers come from **Gemini** `generateContent` on that frame only
(single image, structured JSON, minimal thinking where the model supports it). The deployment
uses one model, `services.gemini_vlm.model`; optional `fallback_models` are tried in order, and a
model that returns HTTP 429 is skipped until the reset time Gemini reports. The free tier allows about 20 requests per model
per day, so enable billing on the key's Google project for competition use. `ip_family: ipv4`
avoids networks whose IPv6 route to Google is broken. Each call has a hard deadline. Answers took about 2–8 s in testing, dominated by
the Gemini call. Selecting frames sends them to Google; this is the configured choice for this
deployment.

The local Qwen2.5-VL fallback is optional. `AIC_LOCAL_VLM=off` (process or `.env`) starts no
Ollama process and makes the runtime skip every Ollama probe, unload, and fallback call; on
Windows each call to a closed loopback port otherwise costs the 2 s connect timeout. The
launcher then accepts the Q&A readiness state "degraded" (no local fallback). The assistant's
`router: rules` answers each chat message instantly: search, question about the selected frame
(or the top result), or search help with entity suggestions.

Local GPU inference on the reference laptop failed with NVIDIA driver 546.18 (CUDA 12.3): the
CUDA runtimes bundled with Ollama 0.32 need a newer driver, and Vulkan found no device. CPU
inference took about 75 s for the first question per frame.

## DRES submission

The browser submits through FastAPI, which proxies the DRES client API v2, so the session never
reaches page scripts and cross-origin rules do not apply.

1. Click **DRES**, enter the server address (for example `https://eventretrieval.one`) and either
   a username/password or a session ID already issued by the DRES web client. The address may
   be preset with `DRES_URL`; `DRES_USERNAME`/`DRES_PASSWORD` in `.env` are used when the form
   fields are empty. Passwords are not stored; the session lives in API memory until
   *Disconnect* or a restart.
2. The evaluation list comes from `GET /api/v2/client/evaluation/list`. Each AI Challenge query
   can be its own evaluation, so the evaluation is chosen again in every submit dialog.
3. **Submit to DRES** opens a confirmation with the frame, the evaluation, a one-line summary,
   and the exact request body. Times and frame numbers are resolved on the server from the
   canonical frame record, never from browser values.
4. The verdict (`CORRECT`, `WRONG`, `INDETERMINATE`, or the rejection reason) is shown and
   echoed in the chat. Every attempt is appended to `logs/dres-submissions.jsonl`.

Request bodies (`POST /api/v2/submit/{evaluationId}?session=…`):

| Task | Answer |
|---|---|
| KIS | `{"mediaItemName": "L21_V013", "start": 758933, "end": 758933}` (milliseconds) |
| Q&A | `{"text": "QA-<answer>-<video>-<ms>"}` |
| TRAKE | `{"text": "TR-<video>-<frame_1>,...,<frame_n>"}` |

each wrapped as `{"answerSets": [{"answers": [<answer>]}]}`. The KIS shape matches the
organizer's DRES examples; the Q&A/TRAKE text conventions follow other AI Challenge clients and
are configurable in `services.dres.formats`. Check them against the round's official guide.
VBS-style result logs (`POST /api/v2/log/result/{evaluationId}`) are sent after each search only
when `services.dres.log_results` is true.

| Endpoint | Contract |
|---|---|
| `GET /v1/dres/state` | Connection, user, evaluations, selected evaluation |
| `POST /v1/dres/login` | `server_url` plus `username`/`password` or `session_id` |
| `POST /v1/dres/evaluation`, `/refresh`, `/logout`; `GET /v1/dres/task` | Session management |
| `POST /v1/dres/preview` | Exact body and summary; sends nothing |
| `POST /v1/dres/submit` | Sends one answer and returns the verdict |
| `POST /v1/dres/log` | Optional result log |

## Graph evidence and routing

With Graph mode on, `auto` routing sends event-order and narrated queries (sau đó, trước khi,
nói về, phóng viên, ...) to the **ASR topic graph**, and counts, colours, clothing, or spatial
relations (bên trái, bên cạnh, bao nhiêu, ...) to the **object graph**. Other descriptive
queries use the topic graph because detections cover a closed object vocabulary. If the chosen
graph does not exist for the requested batches, the existing one is used. Conventional retrieval
takes over only when no batch produces graph evidence; batches without a graph are listed in
`batch_statuses`.

Topic graph: adjacent timed ASR segments are grouped into topics (no LLM summary by default),
embedded with the configured Vietnamese text encoder (not BGE-M3), and linked by `NEXT` and
semantic `GRAPH_RELATION` edges; entity mentions add `MENTIONS`/`SHARES_ENTITY`. Score:
`S(u_i, q) = cos(z_q, z_i) + min(Σ_{j∈C} b(r_ij), τ)` with K = 50 and τ = 0.20. Stage 2 re-ranks
inside the Stage 1 subgraph. Speech does not by itself prove what is visible.

Object graph: organizer detections are frame-local nodes; image-plane predicates (left/right,
above/below, near, inside) are verified from boxes. Frames are ranked by the satisfied fraction
`S(f, q) = |C_sat| / (|V_q| + |E_q|)`. An undetected object is unobserved, not absent. There are
no frame captions, so colour, `wears`, `holds`, `behind`, `on`, and `in_front_of` are reported
as unverified constraints rather than proven.

| Endpoint | Contract |
|---|---|
| `POST /v1/assistant/route` | Intent, rewritten query and question, graph route, suggestions, `router: llm` or `rules` |
| `GET /v1/graph/status` | Topic/object manifest presence for active batches |
| `POST /v1/graph/search` | Stage 1 query with optional Stage 2 refinement; `auto`, `topic`, or `object`; evidence or an explicit fallback reason |
| `GET /v1/graph/topic/{batch}/{topic_id}/chain` | The `NEXT` chain around a topic (radius ≤ 10) |

Artifacts are loaded lazily from `data/graphs/<batch>/topic/` and `data/graphs/<batch>/object/`
(`graph.root`). Setting `graph.enabled` does not build them, and the browser never builds or
rebuilds them. Coverage is batch-specific: graphs currently build for Batch 1 only.

### Inspect and build a bounded graph sample

The graph builder defaults to read-only inspection of canonical ASR/frame coverage:

```powershell
.\.venv-runtime\Scripts\python.exe scripts/build_topic_graph.py --batch-id aic25-b1-reference --inspect
```

After checking the selected video has canonical transcripts and frames, an explicit sample
build creates derived artifacts under `data/graphs/samples/<video>/<batch>/`:

```powershell
.\.venv-runtime\Scripts\python.exe scripts/build_topic_graph.py --batch-id aic25-b1-reference --sample L21_V001 --kind topic
```

This requires the configured Vietnamese text embedding model already prepared locally. The
builder does not download it. It groups adjacent timed ASR segments without LLM summarization
and embeds them with that configured encoder. Sample output is outside the active graph path
and does not silently replace production search coverage.

`--kind object` reads organizer object detections and canonical ordinal/source-frame mappings;
`--objects` selects an archive explicitly. The builder supports B01 graph builds only at this
stage; B02 identity validation is pending. Existing graph directories are never overwritten.
Use a new `--output` for another build. `--full` is an explicit whole-batch operation and should
only be used when that preprocessing scope has been requested and its storage/runtime cost
reviewed. Generated artifacts remain outside Git.

## Verification before competition use

Run the normal offline checks first. After explicitly starting local services and preparing
data, exercise all three workflows in the browser, inspect returned canonical frames and
evidence, refine one search, browse neighboring frames, and download validated CSVs. Compare
those exports with the legacy console on the same batch and query.

Measure browser interaction time separately from cold and warm API search time. Verify a
missing graph artifact falls back transparently, an unavailable backend gives an actionable
error, and a failed QA call does not fabricate an answer. No instantaneous search or live-data
acceptance claim follows from offline/UI-only verification.
