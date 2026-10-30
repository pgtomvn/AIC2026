/* EviBind Browser: a buildless, same-origin client. No retrieval model runs here. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const QA_SPEC = "aic-2026-visual-qa-general-text-provisional-v1";
  const FRAME_TOP_K = 50;
  const SEQUENCE_TOP_K = 20;

  const state = {
    mode: "kis",
    hits: [],            // frames from conventional or graph search
    graph: null,         // {nodes, edges, subgraph_size, query_graphs}
    graphKind: null,     // "topic" | "object" | null
    view: "graph",
    relations: new Set(["NEXT"]),
    selected: null,      // inspected item (topic node or frame)
    answers: new Map(),  // frameKey -> answer record
    pins: [],
    history: [],         // saved turns of the current search, restored by Undo without searching
    turn: -1,
    capabilities: null,
    searchSeq: 0,
    inspectSeq: 0,
    request: null,
    viewBox: null,
    cache: new Map(),
  };

  // ---------- small helpers ----------
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function svg(tag, attributes = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function button(text, className, action, label) {
    const node = el("button", className, text);
    node.type = "button";
    if (label) node.setAttribute("aria-label", label);
    node.addEventListener("click", action);
    return node;
  }
  function clock(ms) {
    if (!Number.isFinite(ms)) return "";
    const seconds = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }
  function seconds(ms) { return Number.isFinite(ms) ? (ms / 1000).toFixed(2) : "?"; }
  function now() { return new Date().toLocaleTimeString(); }
  function frameKey(frame) {
    return [frame.batch_id, frame.video_id, frame.source_frame_idx, frame.keyframe_ordinal ?? ""].join(":");
  }
  function identity(frame) {
    const value = {batch_id: frame.batch_id, video_id: frame.video_id, source_frame_idx: frame.source_frame_idx};
    if (frame.keyframe_ordinal != null) value.keyframe_ordinal = frame.keyframe_ordinal;
    return value;
  }
  function framePath(frame) {
    if (!frame?.batch_id || !frame.video_id) return null;
    const batch = encodeURIComponent(frame.batch_id), video = encodeURIComponent(frame.video_id);
    if (frame.batch_id === "aic26-b2") {
      return Number.isInteger(frame.keyframe_ordinal) && frame.keyframe_ordinal > 0
        ? `/v1/keyframes/${batch}/${video}/${frame.keyframe_ordinal}` : null;
    }
    return Number.isInteger(frame.source_frame_idx) && frame.source_frame_idx >= 0
      ? `/v1/frames/${batch}/${video}/${frame.source_frame_idx}` : null;
  }
  function imageUrl(frame) { const path = framePath(frame); return path ? `${path}/image` : ""; }
  function image(frame) {
    const node = el("img");
    node.alt = `${frame.video_id ?? ""} ${clock(frame.timestamp_ms ?? frame.start_ms)}`;
    node.loading = "lazy";
    node.decoding = "async";
    const url = imageUrl(frame);
    if (url) node.src = url;
    return node;
  }
  function frameOf(item) {
    // Topic nodes and frames share the canonical identity fields.
    return {
      batch_id: item.batch_id, video_id: item.video_id, source_frame_idx: item.source_frame_idx,
      keyframe_ordinal: item.keyframe_ordinal ?? undefined,
      timestamp_ms: item.timestamp_ms ?? item.start_ms,
    };
  }
  let toastTimer;
  function toast(message) {
    $("toast").textContent = message;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
  }
  const FRIENDLY_ERRORS = {
    qa_all_providers_failed: "Gemini could not answer. Please ask again.",
    gemini_timeout: "Gemini did not answer in time. Please ask again.",
    rate_limited: "The Gemini quota for today is used up. Enable billing for the API key or try again later.",
    server_unavailable: "Gemini is temporarily unavailable. Please ask again.",
    network_unreachable: "Gemini cannot be reached from this network.",
    auth_or_permission: "The Gemini API key was rejected.",
    runtime_warming: "The search engine is still starting. Try again in a moment.",
    manual_local_inference_busy: "Another request is still running. Try again in a moment.",
  };

  function errorText(error) {
    if (error?.name === "AbortError") return "Cancelled.";
    if (error instanceof TypeError) return "The local server is not reachable.";
    const code = error?.detail?.code || (typeof error?.detail === "string" ? error.detail : null);
    if (code && FRIENDLY_ERRORS[code]) return FRIENDLY_ERRORS[code];
    const detail = error?.detail;
    if (detail && typeof detail === "object") return detail.message || detail.code || error.message;
    return error?.message || "Something went wrong.";
  }
  async function api(path, {method = "GET", body, signal, blob = false} = {}) {
    const response = await fetch(path, {
      method, signal,
      headers: body === undefined ? {} : {"Content-Type": "application/json"},
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      const value = await response.json().catch(() => ({}));
      const detail = value.detail ?? value;
      const message = typeof detail === "string" ? detail : detail.message || detail.code || `HTTP ${response.status}`;
      const error = new Error(message);
      error.detail = detail;
      error.status = response.status;
      throw error;
    }
    return blob ? response.blob() : response.json();
  }
  async function cached(key, loader) {
    if (state.cache.has(key)) return state.cache.get(key);
    const value = await loader();
    state.cache.set(key, value);
    if (state.cache.size > 120) state.cache.delete(state.cache.keys().next().value);
    return value;
  }
  function queryId() { return `browser-${Date.now()}`; }
  function batches() { return []; }  // every dataset the runtime serves
  // ---------- search ----------
  function syncSearchButton() {
    $("search-button").textContent = "Search";
  }
  function clearResults() {
    state.hits = []; state.sequences = null; state.graph = null; state.graphKind = null; state.selected = null;
    render();
    renderInspector();
  }
  function busy(active) {
    $("progress").hidden = !active;
    $("search-button").disabled = active;
    if (active) $("search-button").textContent = "Searching…";
    else syncSearchButton();
  }

  function sequenceSteps(text) {
    const steps = text.split(">>").map((step) => step.trim()).filter(Boolean);
    return text.includes(">>") && steps.length >= 2 ? steps.slice(0, 20) : null;
  }

  async function search() {
    const query = $("stage1").value.trim();
    if (!query) { $("stage1").focus(); return 0; }
    const steps = sequenceSteps(query);
    // A new first description starts a new search: earlier turns no longer apply.
    if (state.history[0]?.query !== query) { state.history = []; state.turn = -1; }
    const seq = ++state.searchSeq;
    state.request?.abort();
    const controller = new AbortController();
    state.request = controller;
    busy(true);
    try {
      let value;
      if (steps) {
        // "a >> b >> c": the events happen in this order within one video (TRAKE alignment).
        const context = $("stage2").value.trim();
        value = await api("/v1/trake/search", {method: "POST", signal: controller.signal, body: {
          query_id: queryId(), global_description: [steps.join(", "), context].filter(Boolean).join(". "),
          stages: steps.map((description, index) => ({stage_index: index + 1, description})),
          batch_filters: batches(), top_k: SEQUENCE_TOP_K}});
      } else {
        // One search: conventional retrieval fused with transcript-topic and object evidence.
        const body = {query, refinement: $("stage2").value.trim(), top_k: FRAME_TOP_K, batch_filters: batches()};
        if (state.reference) body.reference_frame = identity(state.reference);
        value = await api("/v1/search", {method: "POST", signal: controller.signal, body});
      }
      if (seq !== state.searchSeq) return 0;
      state.sequences = steps ? value.results || [] : null;
      state.steps = steps;
      state.hits = steps ? state.sequences.flatMap((row) => (row.stage_matches || []).map((m) => ({...m, batch_id: row.batch_id, video_id: row.video_id}))) : value.hits || [];
      state.graph = steps ? null : value.graph || null;
      state.graphKind = state.graph?.nodes?.length ? "topic" : null;
      state.view = "grid";
      state.graphFocus = undefined;
      state.selected = null;
      state.viewBox = null;
      saveTurn(query);
      render();
      renderInspector();
      logResults(query);
      return steps ? state.sequences.length : state.hits.length;
    } catch (error) {
      if (seq === state.searchSeq && error.name !== "AbortError") {
        state.hits = []; state.sequences = null; state.graph = null; state.graphKind = null;
        render(errorText(error));
      }
      throw error;
    } finally {
      if (seq === state.searchSeq) { busy(false); state.request = null; }
    }
  }

  // ---------- search turns ----------
  const MAX_TURNS = 20;
  function saveTurn(query) {
    state.history = state.history.slice(0, state.turn + 1);
    state.history.push({query, refinement: $("stage2").value, reference: state.reference || null, hits: state.hits, sequences: state.sequences, steps: state.steps,
      graph: state.graph, graphKind: state.graphKind});
    if (state.history.length > MAX_TURNS + 1) state.history.splice(1, 1);  // keep the first results
    state.turn = state.history.length - 1;
    renderRefine();
  }
  function restoreTurn(index) {
    const turn = state.history[index];
    if (!turn) return;
    state.request?.abort();
    state.turn = index;
    Object.assign(state, {hits: turn.hits, sequences: turn.sequences, steps: turn.steps, graph: turn.graph,
      graphKind: turn.graphKind, reference: turn.reference, selected: null, view: "grid", graphFocus: undefined, viewBox: null});
    $("stage1").value = turn.query;
    $("stage2").value = turn.refinement;
    render();
    renderInspector();
    renderRefine();
  }
  function renderRefine() {
    // Undo restores a saved turn instantly, e.g. after a refinement that made results worse.
    $("refine").hidden = state.history.length < 2;
    $("turn-label").textContent = state.history.length > 1 ? `turn ${state.turn + 1} of ${state.history.length}` : "";
    $("undo").disabled = state.turn < 1;
    $("first-turn").disabled = state.turn < 1;
  }

  // ---------- center rendering ----------
  function render(message) {
    const topic = Boolean(state.graphKind === "topic" && state.graph?.nodes?.length);
    const showGraph = topic && state.view === "graph";
    $("view-toggle").hidden = !topic;
    document.querySelectorAll("[data-view]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === state.view)));
    $("relations").hidden = !showGraph;
    $("legend").hidden = !showGraph;
    renderQueryGraph();
    const sequences = state.sequences;
    $("view-title").textContent = showGraph ? "Relationship graph" : sequences ? "Sequences" : "Results";
    const matches = sequences ? sequences.length : state.hits.length;
    $("view-meta").textContent = !matches ? ""
      : showGraph ? "Click a video name or part to pin it · double-click to see all"
      : sequences ? `${matches} videos where ${state.steps.length} steps happen in order`
      : `${matches} ${matches === 1 ? "result" : "results"}`;
    $("empty").textContent = message || (state.searchSeq ? "Nothing matched. Try a shorter or different description." : "Describe a moment on the left, or ask the assistant.");
    $("empty").hidden = matches > 0;
    $("graph").toggleAttribute("hidden", Boolean(!showGraph || !matches));
    $("grid").hidden = showGraph || !matches;
    if (showGraph && matches) renderGraph();
    else if (matches) renderGrid();
  }

  function renderQueryGraph() {
    const graphs = Object.values(state.graph?.query_graphs || {});
    const query = graphs.find((g) => g?.nodes?.length);
    $("query-graph").hidden = !query;
    const target = $("query-graph-items");
    target.replaceChildren();
    if (!query) return;
    const names = new Map(query.nodes.map((n) => [n.id, n.label]));
    for (const node of query.nodes) {
      const text = `${node.count ? node.count + " × " : ""}${node.attributes?.length ? node.attributes.join(" ") + " " : ""}${node.label}`;
      target.append(el("span", "chip", text));
    }
    for (const edge of query.edges || []) {
      target.append(el("span", "chip edge", `${names.get(edge.subject)} ${edge.predicate.replaceAll("_", " ")} ${names.get(edge.object)}`));
    }
  }

  function renderGrid() {
    $("grid").classList.toggle("sequences", Boolean(state.sequences));
    if (state.sequences) {
      $("grid").replaceChildren(...state.sequences.map((row, index) => sequenceRow(row, index)));
      return;
    }
    $("grid").replaceChildren(...state.hits.map((hit, index) => frameCard(hit, `#${index + 1}`)));
  }

  function sequenceRow(row, index) {
    const box = el("section", "sequence");
    const head = el("div", "sequence-head");
    const frames = (row.stage_matches || []).map((m) => ({...m, batch_id: row.batch_id, video_id: row.video_id}));
    head.append(el("strong", "", `#${index + 1} ${row.video_id}`),
      el("span", "hint", frames.length ? `${clock(frames[0].timestamp_ms)} → ${clock(frames[frames.length - 1].timestamp_ms)}` : ""),
      button("Submit sequence", "ghost small", () => openSubmit(frames, "trake")));
    const track = el("div", "sequence-track");
    frames.forEach((frame, i) => {
      if (i) track.append(el("span", "arrow", "→"));
      const step = el("div", "step");
      step.append(el("div", "step-label", `${i + 1}. ${state.steps?.[i] ?? ""}`), frameCard(frame, ""));
      track.append(step);
    });
    box.append(head, track);
    return box;
  }

  function frameCard(frame, label) {
    const card = button("", "card", () => inspect({...frame, kind: "frame"}), `${frame.video_id} ${clock(frame.timestamp_ms)}`);
    card.dataset.key = frameKey(frame);
    const thumb = el("div", "thumb");
    thumb.append(image(frame));
    const boxes = frame.object_evidence || (frame.graph_evidence?.nodes ? frame.graph_evidence : null);
    if (boxes?.nodes?.length) drawBoxes(thumb, boxes.nodes, boxes.box_format);
    const tags = el("div", "tags");
    if ((frame.sources || []).includes("topic")) tags.append(el("span", "tag", "said"));
    if ((frame.sources || []).includes("object")) tags.append(el("span", "tag", "objects"));
    if (tags.childElementCount) thumb.append(tags);
    const caption = el("div", "caption");
    caption.append(el("span", "", `${frame.video_id} · ${clock(frame.timestamp_ms)}`), el("span", "", label));
    card.append(thumb, caption);
    card.classList.toggle("selected", state.selected?.kind === "frame" && frameKey(state.selected) === card.dataset.key);
    card.classList.toggle("pinned", isPinned(frame));
    return card;
  }

  function drawBoxes(container, nodes, format) {
    const layer = svg("svg", {viewBox: "0 0 100 100", preserveAspectRatio: "none", class: "boxes", "aria-hidden": "true"});
    let drawn = 0;
    for (const node of nodes) {
      const box = node.box || node.bbox;
      if (!node.matched && node.matched !== undefined) continue;
      if (!Array.isArray(box) || box.length !== 4 || box.some((v) => !Number.isFinite(v))) continue;
      const [x1, y1, x2, y2] = format === "ymin,xmin,ymax,xmax" ? [box[1], box[0], box[3], box[2]] : box;
      if (x2 <= x1 || y2 <= y1) continue;
      layer.append(svg("rect", {x: x1 * 100, y: y1 * 100, width: (x2 - x1) * 100, height: (y2 - y1) * 100}),
        svg("text", {x: x1 * 100 + 1, y: Math.max(4, y1 * 100 - 1)}, node.label || node.class || ""));
      drawn++;
    }
    if (drawn) {
      container.querySelector("img").style.objectFit = "fill";
      container.append(layer);
    }
  }

  // ---------- relationship graph: one timeline row per video ----------
  // Each row is a video's matched transcript parts in time order, rows follow the result
  // ranking, and the selected video is pinned and zoomed. Other links are drawn as arcs.
  const NODE_W = 168, NODE_H = 94, GAP_X = 44, ROW_H = 150, LABEL_W = 132, MAX_ROWS = 8;

  function videoKey(item) { return item ? `${item.batch_id}:${item.video_id}` : null; }

  function graphRows() {
    const rows = new Map();
    for (const node of state.graph.nodes) {
      if (node.kind === "entity") continue;
      const key = videoKey(node);
      if (!rows.has(key)) rows.set(key, []);
      rows.get(key).push(node);
    }
    const order = [...new Set(state.hits.map(videoKey))];
    const rank = (key) => { const i = order.indexOf(key); return i < 0 ? order.length : i; };
    const sorted = [...rows.entries()].sort((a, b) => rank(a[0]) - rank(b[0]));
    const shown = sorted.slice(0, MAX_ROWS);
    const focus = videoKey(state.selected);
    if (focus && !shown.some(([key]) => key === focus)) {
      const extra = sorted.find(([key]) => key === focus);
      if (extra) shown.push(extra);
    }
    return shown.map(([key, nodes], index) => ({
      key, rank: rank(key) + 1, y: index * ROW_H,
      nodes: nodes.sort((a, b) => a.start_ms - b.start_ms),
    }));
  }

  function selectedNodeId(rows) {
    const item = state.selected;
    if (!item) return null;
    if (item.kind === "topic") return item.id;
    // A frame selected in the grid highlights the transcript part that contains it.
    const row = rows.find((r) => r.key === videoKey(item));
    const inside = row?.nodes.find((n) => n.start_ms <= item.timestamp_ms && item.timestamp_ms <= n.end_ms);
    return inside ? inside.id : null;
  }

  function renderGraph() {
    const canvas = $("graph");
    canvas.replaceChildren();
    const rows = graphRows();
    const positions = new Map();
    let width = LABEL_W;
    for (const row of rows) {
      row.nodes.forEach((node, index) => positions.set(node.id, {x: LABEL_W + index * (NODE_W + GAP_X), y: row.y}));
      width = Math.max(width, LABEL_W + row.nodes.length * (NODE_W + GAP_X));
    }
    const focus = videoKey(state.selected);
    const selectedId = selectedNodeId(rows);

    const defs = svg("defs");
    const marker = svg("marker", {id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"});
    marker.append(svg("path", {d: "M0 0L10 5L0 10z", fill: "#7d8a83"}));
    defs.append(marker);
    const world = svg("g");
    canvas.append(defs, world);

    for (const row of rows) {
      const band = svg("g", {class: `row${row.key === focus ? " focused" : ""}`});
      band.append(svg("rect", {class: "row-bg", x: -12, y: row.y - 14, width: width + 24, height: NODE_H + 46, rx: 10}));
      const label = svg("g", {class: "row-label", tabindex: 0, role: "button", "aria-label": `Focus ${row.nodes[0].video_id}`});
      label.append(svg("text", {x: 4, y: row.y + 30, class: "row-rank"}, `#${row.rank}`),
        svg("text", {x: 4, y: row.y + 52, class: "row-video"}, row.nodes[0].video_id));
      label.addEventListener("click", () => focusRow(row.key));
      band.append(label);
      world.append(band);
    }

    const byId = new Map(state.graph.nodes.map((n) => [n.id, n]));
    const edges = (state.graph.edges || []).filter((e) => state.relations.has(e.relation) && positions.has(e.source) && positions.has(e.target));
    const linked = new Set();
    for (const edge of edges) {
      const a = positions.get(edge.source), b = positions.get(edge.target);
      if (edge.relation === "NEXT") {
        linked.add(`${edge.source}>${edge.target}`);
        const forward = a.x <= b.x;
        const [left, right] = forward ? [a, b] : [b, a];
        world.append(svg("line", {x1: left.x + NODE_W + 3, y1: left.y + NODE_H / 2, x2: right.x - 3, y2: right.y + NODE_H / 2,
          class: "edge NEXT", "marker-end": "url(#arrow)"}));
        continue;
      }
      const ax = a.x + NODE_W / 2, ay = a.y + NODE_H, bx = b.x + NODE_W / 2, by = b.y + NODE_H;
      const lift = a.y === b.y ? 38 : 0;
      const path = svg("path", {d: `M${ax} ${ay} C${ax} ${ay + 40 + lift}, ${bx} ${by + 40 + lift}, ${bx} ${by}`,
        class: `edge arc ${edge.relation}`});
      path.append(svg("title", {}, edge.entity ? `${edge.relation} · ${edge.entity}` : edge.relation));
      world.append(path);
    }
    // A gap marker where two shown parts of one video are not consecutive.
    if (state.relations.has("NEXT")) {
      for (const row of rows) {
        row.nodes.forEach((node, i) => {
          const next = row.nodes[i + 1];
          if (!next || linked.has(`${node.id}>${next.id}`) || linked.has(`${next.id}>${node.id}`)) return;
          const p = positions.get(node.id);
          world.append(svg("text", {x: p.x + NODE_W + GAP_X / 2, y: p.y + NODE_H / 2 + 4, class: "gap", "text-anchor": "middle"}, "···"));
        });
      }
    }

    const entities = new Map();
    if (state.relations.has("MENTIONS")) {
      for (const edge of state.graph.edges || []) {
        if (edge.relation !== "MENTIONS") continue;
        const entity = byId.get(edge.target);
        if (!entity) continue;
        if (!entities.has(edge.source)) entities.set(edge.source, []);
        entities.get(edge.source).push(entity.label);
      }
    }
    for (const row of rows) {
      for (const node of row.nodes) world.append(topicNode(node, positions.get(node.id), node.id === selectedId, entities.get(node.id)));
    }
    state.graphLayout = {rows, positions, width};
    applyGraphView(canvas);
  }

  function topicNode(node, p, selected, entityNames) {
    const group = svg("g", {class: `node ${node.role || "related"}${selected ? " selected" : ""}`,
      transform: `translate(${p.x} ${p.y})`, tabindex: 0, role: "button",
      "aria-label": `${node.video_id} part ${node.ordinal + 1}, ${clock(node.start_ms)}`});
    const clipId = `clip-${node.id.replace(/[^A-Za-z0-9_-]/g, "_")}`;
    const clip = svg("clipPath", {id: clipId});
    clip.append(svg("rect", {width: NODE_W, height: NODE_H, rx: 7}));
    group.append(clip, svg("rect", {class: "frame", width: NODE_W, height: NODE_H, rx: 7}));
    const url = imageUrl(frameOf(node));
    if (url) group.append(svg("image", {href: url, width: NODE_W, height: NODE_H, preserveAspectRatio: "xMidYMid slice", "clip-path": `url(#${clipId})`}));
    group.append(svg("rect", {class: "frame border", width: NODE_W, height: NODE_H, rx: 7}));
    group.append(svg("text", {class: "caption", x: NODE_W / 2, y: NODE_H + 16, "text-anchor": "middle"},
      `${clock(node.start_ms)}–${clock(node.end_ms)}`));
    if (entityNames?.length) {
      group.append(svg("text", {class: "entities", x: NODE_W / 2, y: NODE_H + 30, "text-anchor": "middle"},
        entityNames.slice(0, 2).join(" · ")));
    }
    group.append(svg("title", {}, node.text || node.transcript || ""));
    const open = () => { inspect({...node, kind: "topic"}); };
    group.addEventListener("click", open);
    group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); } });
    return group;
  }

  function canvasAspect(canvas) {
    const rect = canvas.getBoundingClientRect();
    return rect.width && rect.height ? rect.width / rect.height : 1.6;
  }

  function fitAll(canvas) {
    const layout = state.graphLayout;
    if (!layout?.rows.length) return;
    const aspect = canvasAspect(canvas);
    const height = layout.rows.length * ROW_H + 40;
    let w = Math.max(layout.width + 60, height * aspect, 900), h = w / aspect;
    if (h < height) { h = height; w = h * aspect; }
    state.viewBox = {x: -30, y: -30, w, h};
  }

  function fitRow(canvas, key) {
    const layout = state.graphLayout;
    const row = layout?.rows.find((r) => r.key === key);
    if (!row) return fitAll(canvas);
    const aspect = canvasAspect(canvas);
    // Show the whole focused row, video name included, unless it is longer than about five
    // parts; then centre on the selected part. Never zoom in past ~2.4 rows of height.
    const rowWidth = LABEL_W + row.nodes.length * (NODE_W + GAP_X) + 40;
    const minW = ROW_H * 2.4 * aspect, maxW = Math.max(minW, LABEL_W + 5 * (NODE_W + GAP_X) + 40);
    const w = Math.min(Math.max(rowWidth, minW), maxW), h = w / aspect;
    const selectedId = selectedNodeId(layout.rows);
    const target = layout.positions.get(selectedId) || layout.positions.get(row.nodes[0].id);
    const x = rowWidth <= w ? -20 : Math.max(-20, target.x + NODE_W / 2 - w / 2);
    state.viewBox = {x, y: row.y + NODE_H / 2 - h / 2, w, h};
  }

  function applyGraphView(canvas) {
    const focus = videoKey(state.selected);
    if (!state.viewBox || state.graphFocus !== focus) {
      state.graphFocus = focus;
      if (focus && state.graphLayout.rows.some((r) => r.key === focus)) fitRow(canvas, focus);
      else fitAll(canvas);
    }
    const v = state.viewBox;
    canvas.setAttribute("viewBox", `${v.x} ${v.y} ${v.w} ${v.h}`);
  }

  function focusRow(key) {
    const row = state.graphLayout?.rows.find((r) => r.key === key);
    if (!row) return;
    state.graphFocus = key;
    if (videoKey(state.selected) !== key) {
      const best = row.nodes.find((n) => n.role === "match") || row.nodes[0];
      inspect({...best, kind: "topic"});
      return;
    }
    fitRow($("graph"), key);
    $("graph").setAttribute("viewBox", `${state.viewBox.x} ${state.viewBox.y} ${state.viewBox.w} ${state.viewBox.h}`);
  }

  function enablePanZoom(canvas) {
    let drag = null;
    canvas.addEventListener("wheel", (event) => {
      if (!state.viewBox) return;
      event.preventDefault();
      const rect = canvas.getBoundingClientRect(), v = state.viewBox;
      const factor = event.deltaY > 0 ? 1.12 : 1 / 1.12;
      const px = v.x + ((event.clientX - rect.left) / rect.width) * v.w;
      const py = v.y + ((event.clientY - rect.top) / rect.height) * v.h;
      const w = Math.min(Math.max(v.w * factor, 300), 30000), h = (w * v.h) / v.w;
      state.viewBox = {x: px - ((px - v.x) * w) / v.w, y: py - ((py - v.y) * h) / v.h, w, h};
      canvas.setAttribute("viewBox", `${state.viewBox.x} ${state.viewBox.y} ${w} ${h}`);
    }, {passive: false});
    canvas.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".node, .row-label") || !state.viewBox) return;
      drag = {x: event.clientX, y: event.clientY, start: {...state.viewBox}};
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("panning");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const rect = canvas.getBoundingClientRect();
      const v = drag.start;
      state.viewBox = {...v, x: v.x - ((event.clientX - drag.x) / rect.width) * v.w, y: v.y - ((event.clientY - drag.y) / rect.height) * v.h};
      canvas.setAttribute("viewBox", `${state.viewBox.x} ${state.viewBox.y} ${v.w} ${v.h}`);
    });
    const stop = () => { drag = null; canvas.classList.remove("panning"); };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("dblclick", (event) => {
      if (event.target.closest(".node")) return;
      fitAll(canvas);
      canvas.setAttribute("viewBox", `${state.viewBox.x} ${state.viewBox.y} ${state.viewBox.w} ${state.viewBox.h}`);
    });
  }

  // ---------- inspector ----------
  function topicIdOf(node) {
    const match = /:t(\d+)$/.exec(node.id || "");
    return match ? Number(match[1]) : null;
  }

  function inspect(item) {
    state.selected = item;
    ensureQA(frameOf(item), $("stage1").value.trim() || item.video_id).catch(() => {});
    highlightSelection();
    renderInspector();
  }

  // Fills the strips of a freshly rendered inspector; every re-render calls it again (cached).
  async function loadInspectorContext(item, seq) {
    const strip = $("inspector").querySelector(".strip");
    const extra = $("inspector").querySelector(".extra");
    if (!strip) return;
    try {
      if (item.kind === "topic" && topicIdOf(item) !== null) {
        const value = await cached(`chain:${item.id}`, () => api(`/v1/graph/topic/${encodeURIComponent(item.batch_id)}/${topicIdOf(item)}/chain?radius=3`));
        if (seq !== state.inspectSeq) return;
        strip.replaceChildren(...value.topics.map((topic) => stripItem({...topic, kind: "topic"}, topic.id === item.id, `${seconds(topic.start_ms)}s`)));
        const nearby = $("inspector").querySelector(".nearby");
        const path = framePath(frameOf(item));
        if (nearby && path) {
          const frames = await cached(`adj:${path}`, () => api(`${path}/${item.batch_id === "aic26-b2" ? "neighbors" : "adjacent"}?radius=4`));
          if (seq !== state.inspectSeq) return;
          const list = frames.frames || frames.neighbors || [];
          nearby.replaceChildren(...list.map((neighbor) => {
            const frame = {...identity(frameOf(item)), ...neighbor, kind: "frame"};
            return stripItem(frame, frame.source_frame_idx === item.source_frame_idx, clock(frame.timestamp_ms));
          }));
        }
      } else {
        const path = framePath(item);
        if (!path) return;
        const neighbors = item.batch_id === "aic26-b2" ? "neighbors" : "adjacent";
        const [frames, evidence] = await Promise.allSettled([
          cached(`adj:${path}`, () => api(`${path}/${neighbors}?radius=4`)),
          cached(`ev:${path}`, () => api(`${path}/evidence?limit=12`)),
        ]);
        if (seq !== state.inspectSeq) return;
        if (frames.status === "fulfilled") {
          const list = frames.value.frames || frames.value.neighbors || [];
          strip.replaceChildren(...list.map((neighbor) => {
            const frame = {...identity(item), ...neighbor, kind: "frame"};
            return stripItem(frame, frameKey(frame) === frameKey(item), clock(frame.timestamp_ms));
          }));
        }
        if (evidence.status === "fulfilled" && extra) {
          const lines = (evidence.value.segments || []).slice(0, 4).map((s) => `${String(s.kind).toUpperCase()} ${clock(s.start_ms)} · ${s.text}`);
          extra.textContent = lines.join("\n");
        }
      }
    } catch { /* The inspector stays usable without the temporal strip. */ }
  }

  function stripItem(item, current, label) {
    const node = button("", current ? "current" : "", () => inspect(item), label);
    node._frame = {...frameOf(item), kind: "frame"};
    node.append(image(frameOf(item)), el("span", "", label));
    return node;
  }

  function highlightSelection() {
    // Redrawing pins and zooms the graph onto the selected video.
    if (state.view === "graph" && state.graphKind === "topic" && state.graph?.nodes?.length) renderGraph();
    $("grid").querySelectorAll(".card").forEach((card) => card.classList.toggle("selected",
      state.selected?.kind === "frame" && card.dataset.key === frameKey(state.selected)));
  }

  function renderChatContext() {
    const target = $("chat-context");
    const item = state.selected;
    target.hidden = !item;
    target.replaceChildren();
    if (!item) return;
    const frame = frameOf(item);
    target.append(image(frame), el("span", "", `Asking about ${item.video_id} · ${clock(frame.timestamp_ms ?? item.start_ms)}`),
      button("×", "clear", () => { state.selected = null; highlightSelection(); renderInspector(); }, "Stop asking about this frame"));
  }

  function renderInspector() {
    renderChatContext();
    const box = $("inspector");
    const item = state.selected;
    box.hidden = !item;
    box.replaceChildren();
    if (!item) return;
    const topic = item.kind === "topic";
    const frame = frameOf(item);
    box.append(el("h2", "", topic ? `${item.video_id} · part ${item.ordinal + 1}` : item.video_id));
    const shots = item.shot_start != null ? ` · shots ${item.shot_start}–${item.shot_end}` : "";
    box.append(el("div", "meta", topic ? `${seconds(item.start_ms)}–${seconds(item.end_ms)} s${shots}`
      : `${clock(item.timestamp_ms)} · frame ${item.source_frame_idx ?? "?"}`));
    const photo = el("div", "photo zoomable");
    photo.title = "Click to enlarge";
    photo.append(image(frame));
    photo.addEventListener("click", () => openViewer(frame));
    const boxes = item.object_evidence || (item.graph_evidence?.nodes ? item.graph_evidence : null);
    if (boxes?.nodes?.length) drawBoxes(photo, boxes.nodes, boxes.box_format);
    box.append(photo);

    const answer = state.answers.get(frameKey(frame));
    const actions = el("div", "actions");
    actions.append(button(isPinned(frame, "kis") ? "Selected ✓" : "Select", isPinned(frame, "kis") ? "on" : "", () => {
      togglePin(item, "kis");
      renderInspector();
      highlightPins();
    }));
    actions.append(button("Submit to DRES", "", () => openSubmit(frame, "kis")));
    actions.append(button("Similar frames", "", () => { state.reference = frame; search().catch(() => {}); }));
    box.append(actions);

    if (answer) {
      const row = el("div", "answer-tag");
      row.append(document.createTextNode(`Answer: ${answer.answer} `));
      row.append(button(isPinned(frame, "qa") ? "Selected ✓" : "Select", "link", () => { togglePin(item, "qa"); renderInspector(); }),
        button("Submit", "link", () => openSubmit(frame, "qa", answer.answer)));
      box.append(row);
    }

    const text = topic ? item.text || item.transcript : item.graph_evidence?.text || "";
    if (text) box.append(el("div", "section-label", "What is said"), el("div", "text", text));
    box.append(el("div", "section-label", topic ? "Previous and next parts" : "Before and after"), el("div", "strip"));
    if (topic) box.append(el("div", "section-label", "Nearby frames · pick the exact one to submit"), el("div", "strip nearby"));
    if (!topic) box.append(el("div", "text extra hint"));
    loadInspectorContext(item, ++state.inspectSeq);
  }

  // ---------- large image viewer ----------
  const viewer = {frames: [], index: 0};
  function viewerFrames(current) {
    const strips = [...$("inspector").querySelectorAll(".strip")];
    const nearby = strips.find((strip) => strip.classList.contains("nearby")) || strips[0];
    const list = [...(nearby?.querySelectorAll("button") || [])].map((b) => b._frame).filter(Boolean);
    return list.some((f) => frameKey(f) === frameKey(current)) ? list : [current, ...list];
  }
  function openViewer(frame) {
    viewer.frames = viewerFrames(frame);
    viewer.index = Math.max(0, viewer.frames.findIndex((f) => frameKey(f) === frameKey(frame)));
    showViewerFrame();
    $("viewer").showModal();
  }
  function showViewerFrame() {
    const frame = viewer.frames[viewer.index];
    if (!frame) return;
    $("viewer-image").src = imageUrl(frame);
    $("viewer-caption").textContent = `${frame.video_id} · ${clock(frame.timestamp_ms)} · frame ${frame.source_frame_idx ?? "?"}   (${viewer.index + 1}/${viewer.frames.length})`;
    $("viewer-prev").disabled = viewer.index === 0;
    $("viewer-next").disabled = viewer.index >= viewer.frames.length - 1;
  }
  function stepViewer(delta) {
    viewer.index = Math.min(Math.max(viewer.index + delta, 0), viewer.frames.length - 1);
    showViewerFrame();
  }

  // ---------- selection and export ----------
  function pinKey(item, task) { return `${task}:${frameKey(frameOf(item))}`; }
  function isPinned(item, task = "kis") { return state.pins.some((p) => p.key === pinKey(item, task)); }
  function togglePin(item, task) {
    const key = pinKey(item, task);
    const index = state.pins.findIndex((p) => p.key === key);
    if (index >= 0) state.pins.splice(index, 1);
    else {
      let row = {...frameOf(item), ...(item.kind === "frame" ? item : {})};
      if (task === "qa") {
        const answer = state.answers.get(frameKey(frameOf(item)));
        if (!answer) { toast("Ask a question about this frame first."); return; }
        row = {...row, answer: answer.answer};
      }
      delete row.kind;
      state.pins.push({key, task, item: row});
    }
    $("selection-count").textContent = String(state.pins.length);
  }
  function highlightPins() {
    $("grid").querySelectorAll(".card").forEach((card) => {
      const hit = state.hits.find((h) => frameKey(h) === card.dataset.key);
      if (hit) card.classList.toggle("pinned", isPinned(hit));
    });
  }
  function renderSelection() {
    const task = $("export-task").value;
    const list = $("selected-items");
    list.replaceChildren();
    const pins = state.pins.filter((p) => p.task === task);
    if (!pins.length) list.append(el("li", "hint", "Nothing selected for this task."));
    pins.forEach((pin, rank) => {
      const frame = pin.item;
      const row = el("li");
      const move = (delta) => {
        const other = pins[rank + delta];
        if (!other) return;
        const a = state.pins.indexOf(pin), b = state.pins.indexOf(other);
        [state.pins[a], state.pins[b]] = [state.pins[b], state.pins[a]];
        renderSelection();
      };
      const up = button("↑", "", () => move(-1), "Move up"); up.disabled = rank === 0;
      const down = button("↓", "", () => move(1), "Move down"); down.disabled = rank === pins.length - 1;
      row.append(el("span", "", String(rank + 1)), image(frame),
        el("span", "grow", `${pin.item.video_id} · ${clock(frame.timestamp_ms)}${pin.item.answer ? ` · ${pin.item.answer}` : ""}`),
        up, down, button("×", "", () => {
          state.pins.splice(state.pins.indexOf(pin), 1);
          $("selection-count").textContent = String(state.pins.length);
          renderSelection();
          highlightPins();
        }, "Remove"));
      list.append(row);
    });
    $("export-status").textContent = "";
  }
  function download(blob, name) {
    const url = URL.createObjectURL(blob), link = el("a");
    link.href = url; link.download = name;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function base64Blob(encoded) { return new Blob([Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0))], {type: "text/csv;charset=utf-8"}); }
  async function blobBase64(blob) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let text = "";
    for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
    return btoa(text);
  }
  async function prepareExport() {
    const task = $("export-task").value;
    const chosen = state.pins.filter((p) => p.task === task);
    if (!chosen.length) throw new Error("Nothing selected for this task yet.");
    const rows = chosen.map((p, index) => ({...p.item, rank: index + 1}));
    const batchIds = [...new Set(rows.map((row) => row.batch_id))];
    const id = $("export-query-id").value.trim() || "01";
    if (!/^[A-Za-z0-9_-]{1,32}$/.test(id)) throw new Error("Query id: 1-32 letters, digits, - or _.");
    const filename = `query-${id}-${task}.csv`;
    let blob;
    if ((state.capabilities?.default_search_batches || []).length > 1) {
      const result = await api("/v1/submissions/preview", {method: "POST", body: {task, rows, answer_spec_id: QA_SPEC}});
      blob = base64Blob(result.csv_base64);
    } else if (batchIds.length === 1 && batchIds[0] === "aic26-b2") {
      const result = await api("/v1/b02/submissions/preview", {method: "POST", body: {task, rows, answer_spec_id: QA_SPEC}});
      blob = base64Blob(result.csv_base64);
    } else {
      if (batchIds.length !== 1) throw new Error("Select results from one dataset.");
      const body = {filename, batch_id: batchIds[0], query_id: id, answer_spec_id: QA_SPEC};
      if (task === "qa") body.predictions = rows;
      else body.hits = rows;
      blob = await api(task === "kis" ? "/v1/submissions/export" : `/v1/${task}/submissions/export`, {method: "POST", body, blob: true});
    }
    return {blob, filename, task, id, batchIds, rows};
  }
  async function exportSelection(stage) {
    const status = $("export-status");
    status.textContent = "Checking…";
    $("export-csv").disabled = $("stage-selection").disabled = true;
    try {
      const value = await prepareExport();
      if (stage) {
        if (value.batchIds.length !== 1) throw new Error("Staging needs results from one dataset.");
        const snapshot = await api("/v1/submissions/staging");
        const entry = {task: value.task, query_id: value.id, csv_base64: await blobBase64(value.blob)};
        state.staging = await api("/v1/submissions/staging/entries", {method: "POST", body: {expected_revision: snapshot.revision, batch_id: value.batchIds[0], entry}});
        renderStaging();
        $("staging-details").open = true;
        status.textContent = "Saved on the server.";
      } else {
        download(value.blob, value.filename);
        status.textContent = "Downloaded.";
      }
    } catch (error) {
      status.textContent = errorText(error);
    } finally {
      $("export-csv").disabled = $("stage-selection").disabled = false;
    }
  }
  async function loadStaging() {
    try { state.staging = await api("/v1/submissions/staging"); renderStaging(); }
    catch (error) { $("staging-entries").textContent = errorText(error); }
  }
  function renderStaging() {
    const target = $("staging-entries");
    target.replaceChildren();
    const entries = Object.values(state.staging?.entries || {});
    if (!entries.length) target.append(el("p", "hint", "No staged files."));
    for (const entry of entries) {
      const row = el("div", "staging-row");
      row.append(el("span", "", `${entry.filename} · ${entry.row_count ?? ""} rows`));
      if (entry.csv_base64) row.append(button("CSV", "", () => download(base64Blob(entry.csv_base64), entry.filename)));
      row.append(button("ZIP", "", async () => {
        try {
          const bundle = await api("/v1/submissions/bundles", {method: "POST", body: {mode: "current", batch_id: entry.batch_id, entry_filenames: [entry.filename], staging_revision: state.staging.revision}});
          if (!String(bundle.download_url).startsWith("/v1/submissions/bundles/")) throw new Error("Invalid download link.");
          download(await api(bundle.download_url, {blob: true}), bundle.filename || "submission.zip");
        } catch (error) { $("export-status").textContent = errorText(error); }
      }));
      target.append(row);
    }
  }

  // ---------- assistant ----------
  function chatMessage(role, content, {answer = false} = {}) {
    const wrap = el("div", `msg ${role}`);
    const bubble = el("div", `bubble${answer ? " answer" : ""}`);
    if (content instanceof Node) bubble.append(content); else bubble.textContent = content;
    wrap.append(bubble, el("time", "", now()));
    $("chat").append(wrap);
    $("chat").scrollTop = $("chat").scrollHeight;
    return wrap;
  }
  function entityLabels() {
    return (state.graph?.nodes || []).filter((n) => n.kind === "entity").map((n) => n.label).slice(0, 24);
  }

  async function ensureQA(frame, description) {
    const known = state.answers.get(`session:${frameKey(frame)}`);
    if (known) return known;
    const value = await api("/v1/qa/inject", {method: "POST", body: {...identity(frame), query_id: queryId(), event_description: description, source: "kis_bridge"}});
    const candidate = (value.candidates || [])[0];
    if (!candidate) throw new Error("This frame cannot be opened for question answering.");
    const record = {session: value.search_id, candidate: candidate.candidate_id};
    state.answers.set(`session:${frameKey(frame)}`, record);
    return record;
  }

  // Answers come only from the verified Visual Q&A workflow on one concrete frame.
  async function answerOn(target, question) {
    const description = $("stage1").value.trim() || question;
    const looking = el("div", "typing", "Looking at the frame…");
    $("chat").append(looking);
    $("chat").scrollTop = $("chat").scrollHeight;
    try {
      return await answerWithSession(target, question, description);
    } finally {
      looking.remove();
    }
  }

  async function answerWithSession(target, question, description) {
    const {session, candidate} = await ensureQA(target, description);
    // Gemini answers the selected frame in seconds; the user chose it as the Q&A model.
    const value = await api(`/v1/qa/search/${encodeURIComponent(session)}/answer`, {method: "POST", body: {
      candidate_id: candidate, question, answer_spec_id: QA_SPEC, answer_engine: "vlm",
      reasoning_policy: "gemini_then_local", allow_external_frame_transfer: true}});
    const result = (value.answers || []).find((a) => a.candidate_id === candidate);
    if (!result?.answerable || !result.answer) {
      return chatMessage("bot", "I can't tell from this frame. Try a nearby frame from “Before and after”.");
    }
    state.answers.set(frameKey(target), {answer: result.answer});
    const reply = chatMessage("bot", result.answer, {answer: true});
    const used = button("", "used-frame", () => inspect({...target, kind: "frame"}), `Frame ${target.video_id}`);
    used.append(image(target));
    reply.insertBefore(used, reply.lastChild);
    if (state.selected) renderInspector();
    return reply;
  }

  async function answerQuestion(route, message) {
    let target = state.selected ? frameOf(state.selected) : state.hits[0] ? frameOf(state.hits[0]) : null;
    if (!target) {
      $("stage1").value = route.query || message;
      await search();
      target = state.hits[0] ? frameOf(state.hits[0]) : null;
    }
    if (!target) return chatMessage("bot", "I could not find a frame for this question.");
    return answerOn(target, route.question || message);
  }

  async function sendChat(event) {
    event.preventDefault();
    const input = $("chat-input");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    chatMessage("user", message);
    const typing = el("div", "typing", "…");
    $("chat").append(typing);
    const send = $("chat-form").querySelector("button");
    send.disabled = true;
    try {
      if (state.selected) {
        // A selected frame owns the conversation: every message is a question about it.
        typing.remove();
        await answerOn(frameOf(state.selected), message);
        return;
      }
      const route = await api("/v1/assistant/route", {method: "POST", body: {
        message, context_query: $("stage1").value.trim(), context_entities: entityLabels(),
        has_frame: Boolean(state.selected || state.hits.length)}});
      typing.remove();
      if (route.intent === "search_assistance") {
        const content = el("div");
        content.append(document.createTextNode(route.message));
        if (route.suggestions?.length) {
          const chips = el("div", "suggestions");
          for (const text of route.suggestions) chips.append(button(text, "", () => { $("stage2").value = text; $("stage2").focus(); }));
          content.append(chips);
        }
        chatMessage("bot", content);
      } else if (route.intent === "question_answering") {
        await answerQuestion(route, message);
      } else {
        $("stage1").value = route.query || message;
        $("stage2").value = "";
        const count = await search();
        chatMessage("bot", count ? `Found ${count} ${count === 1 ? "match" : "matches"}. Click one to see details.` : "Nothing matched. Try describing the scene differently.");
      }
    } catch (error) {
      typing.remove();
      if (error.name !== "AbortError") chatMessage("bot", errorText(error));
    } finally {
      send.disabled = false;
    }
  }

  // ---------- DRES submission ----------
  const dres = {state: null, pending: null, previewSeq: 0};
  const VERDICTS = {CORRECT: "Correct ✓", WRONG: "Wrong ✗", INDETERMINATE: "Submitted · waiting for a judge", UNDECIDABLE: "Submitted · undecidable"};

  function fillEvaluations(select) {
    const s = dres.state || {};
    const options = (s.evaluations || []).map((evaluation) => {
      const status = evaluation.status && evaluation.status !== "ACTIVE" ? ` (${evaluation.status.toLowerCase()})` : "";
      const option = el("option", "", `${evaluation.name}${status}`);
      option.value = evaluation.id;
      return option;
    });
    if (!options.length) {
      const none = el("option", "", "No evaluations available");
      none.value = "";
      options.push(none);
    }
    select.replaceChildren(...options);
    select.value = s.evaluation_id || select.options[0]?.value || "";
  }
  function renderDres() {
    const s = dres.state || {};
    $("dres-open").classList.toggle("connected", Boolean(s.connected));
    $("dres-open").textContent = s.connected ? "DRES ✓" : "DRES";
    $("dres-login").hidden = Boolean(s.connected);
    $("dres-session").hidden = !s.connected;
    if (!s.connected) {
      if (!$("dres-server").value) $("dres-server").value = s.server_url || "";
      return;
    }
    $("dres-who").textContent = `Connected to ${s.server_url} as ${s.username}.`;
    fillEvaluations($("dres-evaluation"));
  }
  async function loadDresTask() {
    $("dres-task").textContent = "";
    try {
      const value = await api("/v1/dres/task");
      $("dres-task").textContent = value.task ? `Current task: ${value.task.name}` : "No task is running right now.";
    } catch { /* The task name is a convenience only. */ }
  }
  async function dresCall(path, body, status = $("dres-status")) {
    status.textContent = "";
    try {
      dres.state = await api(path, {method: "POST", body});
      renderDres();
      if (dres.state.connected) loadDresTask();
      return true;
    } catch (error) {
      status.textContent = errorText(error);
      return false;
    }
  }
  async function selectEvaluation(id) {
    if (id && dres.state?.connected && id !== dres.state.evaluation_id) await dresCall("/v1/dres/evaluation", {evaluation_id: id});
  }

  async function openSubmit(target, task, answer = "") {
    const frames = Array.isArray(target) ? target : [target];
    const frame = frames[0];
    if (!dres.state?.connected) {
      $("dres-status").textContent = "Connect to DRES first, then submit again.";
      renderDres();
      $("dres-dialog").showModal();
      return;
    }
    dres.pending = {frame, frames, task};
    const preview = $("submit-preview");
    preview.replaceChildren(...frames.slice(0, 4).map((f) => image(f)), el("div", "", frame.video_id));
    fillEvaluations($("submit-evaluation"));
    $("submit-answer-field").hidden = task !== "qa";
    $("submit-answer").value = answer;
    $("submit-status").textContent = "";
    $("submit-confirm").disabled = true;
    $("submit-dialog").showModal();
    await refreshPreview();
  }
  function submitRequest() {
    const {frame, frames, task} = dres.pending;
    const body = {task, evaluation_id: $("submit-evaluation").value || null};
    if (task === "trake") body.frames = frames.map(identity);
    else body.frame = identity(frame);
    if (task === "qa") body.answer = $("submit-answer").value;
    return body;
  }
  async function refreshPreview() {
    if (!dres.pending) return;
    const seq = ++dres.previewSeq;
    try {
      const value = await api("/v1/dres/preview", {method: "POST", body: submitRequest()});
      if (seq !== dres.previewSeq) return;
      $("submit-summary").textContent = value.summary;
      $("submit-body").textContent = JSON.stringify(value.body, null, 2);
      $("submit-confirm").disabled = !$("submit-evaluation").value;
    } catch (error) {
      if (seq !== dres.previewSeq) return;
      $("submit-summary").textContent = errorText(error);
      $("submit-body").textContent = "";
      $("submit-confirm").disabled = true;
    }
  }
  async function confirmSubmit() {
    if (!dres.pending) return;
    const confirm = $("submit-confirm");
    confirm.disabled = true;
    $("submit-status").textContent = "Submitting…";
    try {
      await selectEvaluation($("submit-evaluation").value);
      const result = await api("/v1/dres/submit", {method: "POST", body: submitRequest()});
      const verdict = result.accepted ? VERDICTS[result.verdict] || "Submitted" : `Rejected: ${result.description || "no reason given"}`;
      const status = $("submit-status");
      status.replaceChildren(el("span", `verdict ${result.verdict === "CORRECT" ? "correct" : result.verdict === "WRONG" || !result.accepted ? "wrong" : ""}`, verdict));
      chatMessage("bot", `DRES · ${result.summary}\n${verdict}`);
      dres.pending = null;
      setTimeout(() => $("submit-dialog").close(), 1600);
    } catch (error) {
      $("submit-status").textContent = errorText(error);
      confirm.disabled = false;
    }
  }
  function logResults(query) {
    if (!dres.state?.connected || !dres.state.log_results || !state.hits.length) return;
    api("/v1/dres/log", {method: "POST", body: {query, frames: state.hits.slice(0, 100).map((hit) => identity(frameOf(hit)))}}).catch(() => {});
  }

  // ---------- resizable panels ----------
  const LAYOUT_KEY = "evibind.layout";
  function loadLayout() {
    try { return JSON.parse(localStorage.getItem(LAYOUT_KEY) || "{}"); } catch { return {}; }
  }
  function saveLayout(layout) {
    try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); } catch { /* sizes last for this page */ }
  }
  function applyLayout(layout) {
    const app = document.querySelector(".app");
    if (layout.left) app.style.setProperty("--left-w", `${layout.left}px`);
    if (layout.right) app.style.setProperty("--right-w", `${layout.right}px`);
    if (layout.inspector) document.querySelector(".right").style.setProperty("--inspector-h", `${layout.inspector}px`);
  }
  function enableGutters() {
    const app = document.querySelector(".app");
    const layout = loadLayout();
    applyLayout(layout);
    for (const gutter of document.querySelectorAll("[data-gutter]")) {
      gutter.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        gutter.setPointerCapture(event.pointerId);
        gutter.classList.add("dragging");
        const bounds = app.getBoundingClientRect();
        const panel = document.querySelector(".right").getBoundingClientRect();
        const move = (e) => {
          if (gutter.dataset.gutter === "left") layout.left = Math.round(Math.min(Math.max(e.clientX - bounds.left - 12, 200), 480));
          else if (gutter.dataset.gutter === "right") layout.right = Math.round(Math.min(Math.max(bounds.right - e.clientX - 12, 280), Math.max(320, bounds.width * 0.6)));
          else layout.inspector = Math.round(Math.min(Math.max(e.clientY - panel.top - 40, 160), panel.height - 180));
          applyLayout(layout);
        };
        const up = () => {
          gutter.classList.remove("dragging");
          gutter.removeEventListener("pointermove", move);
          gutter.removeEventListener("pointerup", up);
          saveLayout(layout);
          if (state.view === "graph" && state.graph?.nodes?.length) { state.graphFocus = undefined; renderGraph(); }
        };
        gutter.addEventListener("pointermove", move);
        gutter.addEventListener("pointerup", up);
      });
      gutter.addEventListener("dblclick", () => {
        delete layout[gutter.dataset.gutter === "inspector" ? "inspector" : gutter.dataset.gutter];
        app.style.removeProperty(`--${gutter.dataset.gutter}-w`);
        if (gutter.dataset.gutter === "inspector") document.querySelector(".right").style.removeProperty("--inspector-h");
        saveLayout(layout);
      });
    }
  }

  // ---------- wiring ----------
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
    state.view = b.dataset.view;
    state.graphFocus = undefined;
    render();
  }));
  $("search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    state.reference = null;
    search().catch(() => {});
  });
  $("undo").addEventListener("click", () => restoreTurn(state.turn - 1));
  $("first-turn").addEventListener("click", () => restoreTurn(0));
  $("relations").querySelectorAll("input").forEach((box) => box.addEventListener("change", () => {
    if (box.checked) state.relations.add(box.value); else state.relations.delete(box.value);
    state.viewBox = null;
    render();
  }));
  $("chat-form").addEventListener("submit", sendChat);
  $("selection-open").addEventListener("click", () => { renderSelection(); loadStaging(); $("selection-dialog").showModal(); });
  document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => b.closest("dialog").close()));
  $("dres-open").addEventListener("click", () => { $("dres-status").textContent = ""; renderDres(); if (dres.state?.connected) loadDresTask(); $("dres-dialog").showModal(); });
  $("dres-login").addEventListener("submit", async (event) => {
    event.preventDefault();
    const ok = await dresCall("/v1/dres/login", {server_url: $("dres-server").value.trim(), username: $("dres-user").value.trim(),
      password: $("dres-password").value, session_id: $("dres-token").value.trim()});
    $("dres-password").value = "";
    $("dres-token").value = "";
    if (ok) $("dres-status").textContent = "Connected.";
  });
  $("dres-evaluation").addEventListener("change", () => selectEvaluation($("dres-evaluation").value));
  $("dres-refresh").addEventListener("click", () => dresCall("/v1/dres/refresh"));
  $("dres-logout").addEventListener("click", () => dresCall("/v1/dres/logout"));
  $("submit-evaluation").addEventListener("change", refreshPreview);
  let answerTimer;
  $("submit-answer").addEventListener("input", () => { clearTimeout(answerTimer); answerTimer = setTimeout(refreshPreview, 250); });
  $("submit-confirm").addEventListener("click", confirmSubmit);
  $("export-task").addEventListener("change", renderSelection);
  $("export-csv").addEventListener("click", () => exportSelection(false));
  $("stage-selection").addEventListener("click", () => exportSelection(true));
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !document.querySelector("dialog[open]")) {
      event.preventDefault();
      $("search-form").requestSubmit();
    }
  });
  enablePanZoom($("graph"));
  enableGutters();
  $("viewer-prev").addEventListener("click", () => stepViewer(-1));
  $("viewer-next").addEventListener("click", () => stepViewer(1));
  $("viewer-open").addEventListener("click", () => { const f = viewer.frames[viewer.index]; $("viewer").close(); if (f) inspect(f); });
  $("viewer").addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") { event.preventDefault(); stepViewer(-1); }
    if (event.key === "ArrowRight") { event.preventDefault(); stepViewer(1); }
  });
  $("viewer").addEventListener("click", (event) => { if (event.target === $("viewer")) $("viewer").close(); });

  // Capabilities decide the CSV export route (single or combined datasets).
  api("/v1/runtime/capabilities").then((value) => {
    state.capabilities = value;
  }).catch(() => {});
  // Start loading the local VLM now so the first question does not wait for it.
  api("/v1/assistant/prewarm", {method: "POST"}).catch(() => {});
  api("/v1/dres/state").then((value) => { dres.state = value; renderDres(); }).catch(() => {});
  const welcome = el("div", "welcome", "Hi! Describe a moment to search, ask a question about the selected frame, or ask me how to search.");
  $("chat").append(welcome);
  syncSearchButton();
  render();
})();
