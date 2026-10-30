"""Local browser transport and inspectable, optional graph retrieval.

No model, artifact, or service is opened at import or construction time.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import PROJECT_ROOT
from src.ingest.frame_identity import resolve_image_frame

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn").replace("đ", "d")


class AssistantRequest(BaseModel):
    """One chat message; answering still uses the verified QA workflow."""

    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=8192)
    graph_mode: bool = False
    context_query: str = Field(default="", max_length=8192)
    context_entities: list[str] = Field(default_factory=list, max_length=24)
    has_frame: bool = False

    @field_validator("message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message is empty")
        return value.strip()

    @field_validator("context_entities")
    @classmethod
    def short_entities(cls, values: list[str]) -> list[str]:
        return [" ".join(v.split())[:64] for v in values if v.strip()]


INTENTS = ("video_search", "question_answering", "search_assistance")
GRAPH_ROUTES = ("topic", "object", "conventional")

_ASSISTANCE_CUES = (
    "help me search", "how to search", "search tips", "suggest", "goi y", "huong dan tim",
    "cach tim", "giup toi tim", "tro giup", "nen tim", "viet lai truy van",
)
_QUESTION_CUES = (
    "bao nhieu", "may nguoi", "may con", "mau gi", "la gi", "la ai", "ten gi", "o dau",
    "khi nao", "how many", "what color", "what colour", "who is", "what is", "which ",
    "where is", "when ", "co phai", "hay khong", "khong?",
)
_TEMPORAL_CUES = (
    "sau do", "truoc khi", "sau khi", "tiep theo", "tiep den", "roi ", "then ", "before ",
    "after ", "narrat", "loi noi", "thuyet minh", "noi ve", "noi rang", "phat bieu",
    "ke ve", "gioi thieu", "transcript", "mc ", "phong vien", "ban tin", "said", "says",
)
_OBJECT_CUES = (
    "ben trai", "ben phai", "phia tren", "phia duoi", "ben canh", "ben trong", "dang sau",
    "phia truoc", "left of", "right of", "next to", "in front of", "behind", "above",
    "below", "under", "wearing", "holding", "mac ao", "cam ", "deo ", "doi mu",
    "bao nhieu", "how many", "mau gi", "what color", "what colour",
)
# Matched on accented text: folding merges words such as "nấu" (cook) and "nâu" (brown).
_COUNT_OR_COLOR = re.compile(
    r"(?<!\w)(?:\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|two|three|four|five|six)\s+"
    r"(?:người|con|chiếc|cái|quả|trái|people|persons|men|women|cars|dogs|cats)(?!\w)"
    r"|(?<!\w)(?:đỏ|xanh|vàng|trắng|đen|hồng|tím|nâu|xám|red|blue|green|yellow|white|black|pink)(?!\w)"
)


def _has(folded: str, cues: tuple[str, ...]) -> bool:
    return any(cue in folded for cue in cues)


def graph_route(message: str) -> tuple[str, str]:
    """Pick the graph that can hold evidence for the message.

    Event order and narration live in the ASR topic graph; entities, counts, attributes,
    and spatial relations live in the frame object graph.  Descriptive scenes, even ones that
    name objects, default to the topic graph because detections cover a closed vocabulary.
    """
    folded = " " + _fold(message) + " "
    if _has(folded, _TEMPORAL_CUES):
        return "topic", "temporal_or_narrated"
    accented = " ".join(unicodedata.normalize("NFC", message).casefold().split())
    if _has(folded, _OBJECT_CUES) or _COUNT_OR_COLOR.search(accented):
        return "object", "entity_attribute_or_relation"
    return "topic", "descriptive_scene"


def _suggestions(message: str, entities: list[str]) -> list[str]:
    seen = {_fold(message)}
    picked = []
    for name in entities:
        if _fold(name) not in seen and len(picked) < 4:
            picked.append(name)
            seen.add(_fold(name))
    return picked


def route_message(message: str, graph_mode: bool = False,
                  entities: list[str] | None = None) -> dict[str, Any]:
    """Return a deterministic VI/EN intent and an editable graph route."""
    folded = " " + _fold(message) + " "
    assistance = _has(folded, _ASSISTANCE_CUES)
    question = message.rstrip().endswith("?") or _has(folded, _QUESTION_CUES)
    intent = "search_assistance" if assistance else "question_answering" if question else "video_search"
    mode, reason = graph_route(message) if graph_mode else ("conventional", "graph_mode_off")
    return {"intent": intent, "mode": mode, "query": message, "question": message,
            "suggestions": _suggestions(message, entities or []),
            "message": _ASSISTANT_REPLIES[intent], "router": "rules", "reason": reason}


_ASSISTANT_REPLIES = {
    "search_assistance": "Describe the main scene first, then add an object, a spoken phrase, or "
                         "what happens next as a Stage 2 refinement.",
    "question_answering": "Answering from the selected frame.",
    "video_search": "Searching for the described moment.",
}

_LLM_PROMPT = """You route messages for a Vietnamese news-video retrieval system.
Return only JSON: {"intent": "...", "route": "...", "search_query": "...", "question": "..."}
intent: "video_search" (find a moment), "question_answering" (a question about what a frame shows),
or "search_assistance" (the user asks how to search or wants query suggestions).
route: "topic" for event order, speech, narration, or what someone says; "object" for visible
objects, counts, colors, clothing, or spatial relations; "conventional" otherwise.
search_query: a short Vietnamese description of the scene to retrieve, without question words.
question: the question to answer about the frame, or "" if not a question."""


def _ollama_url(settings: dict[str, Any]) -> str | None:
    import os

    url = os.environ.get(settings.get("url_env", "OLLAMA_URL")) or settings.get("url", "http://127.0.0.1:11434")
    return url.rstrip("/") if url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")) else None


def _ollama_options(settings: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Options matching the Visual Q&A reasoner, so Ollama never reloads the shared model."""
    options: dict[str, Any] = {"temperature": 0, "num_ctx": int(settings.get("num_ctx", 4096)), **extra}
    if settings.get("execution_mode") == "cpu":
        # Mirrors OllamaQwenReasoner's CPU profile.
        options.update(num_ctx=min(options["num_ctx"], 2048), num_gpu=0, num_batch=64)
    return options


def llm_route(message: str, settings: dict[str, Any]) -> dict[str, Any] | None:
    """Ask the local Ollama model for intent and a rewritten query; None on any failure."""
    import urllib.error
    import urllib.request

    url = _ollama_url(settings)
    if url is None:
        return None
    body = json.dumps({
        "model": settings["model"], "stream": False, "format": "json",
        "keep_alive": settings.get("keep_alive", "15m"),
        "options": _ollama_options(settings, num_predict=96),
        "messages": [{"role": "system", "content": _LLM_PROMPT}, {"role": "user", "content": message}],
    }).encode("utf-8")
    request = urllib.request.Request(url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=float(settings.get("timeout_seconds", 8))) as response:
            content = json.loads(response.read().decode("utf-8"))["message"]["content"]
        value = json.loads(content)
    except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("intent") not in INTENTS:
        return None
    route = value.get("route") if value.get("route") in GRAPH_ROUTES else None
    query = value.get("search_query")
    question = value.get("question")
    return {"intent": value["intent"], "route": route,
            "query": " ".join(query.split())[:512] if isinstance(query, str) and query.strip() else None,
            "question": " ".join(question.split())[:512] if isinstance(question, str) and question.strip() else None}


def assistant_reply(request: AssistantRequest, config: dict[str, Any]) -> dict[str, Any]:
    """Interpret one chat message; the LLM is optional and rules are the fallback."""
    result = route_message(request.message, request.graph_mode, request.context_entities)
    settings = config.get("assistant", {})
    if settings.get("router", "rules") not in {"auto", "llm"}:
        return result
    # Rules answer instantly. In "auto" the model is consulted only for a question with no
    # frame in view, where it must split the scene to search for from the question to ask.
    if settings.get("router") == "auto" and not (
            result["intent"] == "question_answering" and not request.has_frame):
        return result
    ollama = {**config.get("services", {}).get("ollama_vlm", {}), **settings}
    if not ollama.get("model"):
        return result
    decided = llm_route(request.message, ollama)
    if decided is None:
        return result
    result.update(intent=decided["intent"], router="llm", message=_ASSISTANT_REPLIES[decided["intent"]])
    if decided["query"]:
        result["query"] = decided["query"]
    if decided["question"]:
        result["question"] = decided["question"]
    if request.graph_mode and decided["route"] in {"topic", "object"}:
        result.update(mode=decided["route"], reason="llm_route")
    return result


class VlmPrewarm:
    """Load the local VLM in the background so the first question is answered quickly.

    The first Ollama request loads ~3 GB of weights and, on Vulkan, compiles shaders; that
    takes up to a minute on the laptop. Nothing runs until :meth:`trigger` is called.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.settings = dict(config.get("services", {}).get("ollama_vlm", {}) or {})
        self.enabled = bool(config.get("assistant", {}).get("prewarm_vlm", True))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._warmed_at = 0.0
        self.state = "idle"

    def trigger(self) -> str:
        with self._lock:
            if not self.enabled or not self.settings.get("model") or _ollama_url(self.settings) is None:
                self.state = "skipped"
            elif self._thread is not None and self._thread.is_alive():
                pass
            elif time.monotonic() - self._warmed_at > 600 or self.state != "warm":
                self.state = "warming"
                self._thread = threading.Thread(target=self._run, name="vlm-prewarm", daemon=True)
                self._thread.start()
            return self.state

    def _run(self) -> None:
        import base64
        import io
        import urllib.request

        try:
            from PIL import Image

            buffer = io.BytesIO()
            Image.new("RGB", (64, 64), "white").save(buffer, format="JPEG")
            body = json.dumps({
                "model": self.settings["model"], "stream": False,
                "keep_alive": self.settings.get("keep_alive", "15m"),
                "options": _ollama_options(self.settings, num_predict=1),
                "messages": [{"role": "user", "content": "OK",
                              "images": [base64.b64encode(buffer.getvalue()).decode("ascii")]}],
            }).encode("utf-8")
            request = urllib.request.Request(f"{_ollama_url(self.settings)}/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                response.read()
            with self._lock:
                self.state, self._warmed_at = "warm", time.monotonic()
        except Exception:  # noqa: BLE001 - a failed warm-up only means a slower first answer
            with self._lock:
                self.state = "failed"


class GraphRequest(BaseModel):
    """Bounded two-stage retrieval with explicit dataset scope."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8192)
    refinement: str = Field(default="", max_length=8192)
    mode: Literal["auto", "topic", "object"] = "auto"
    top_k: int = Field(default=30, ge=1, le=100, strict=True)
    batch_filters: list[str] = Field(default_factory=list, max_length=8)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    query_id: str = Field(default_factory=lambda: f"graph-{uuid4().hex}", max_length=128)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query is empty")
        return value.strip()

    @field_validator("batch_filters", "video_ids")
    @classmethod
    def identifiers(cls, values: list[str]) -> list[str]:
        if any(not _IDENTIFIER.fullmatch(v) or v in {".", ".."} for v in values):
            raise ValueError("invalid dataset or video identifier")
        return list(dict.fromkeys(values))


class GraphService:
    """Lazy per-runtime artifact cache; artifact replacement invalidates the cache."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.settings = context.config.get("graph", {})
        root = Path(self.settings.get("root", "data/graphs"))
        self.root = root if root.is_absolute() else PROJECT_ROOT / root
        self._cache: dict[tuple[str, str], tuple[int, Any]] = {}
        self._lock = threading.RLock()
        self._fallback_encoder: Any = None

    def status(self, batches: tuple[str, ...]) -> dict[str, Any]:
        """Report artifact presence without loading models or opening services."""
        return {batch: {mode: {"state": "available" if self.settings.get("enabled", True)
                               and (self.root / batch / mode / "manifest.json").is_file()
                               else "not_provisioned"}
                        for mode in ("topic", "object")} for batch in batches}

    def _load(self, batch: str, mode: str) -> Any:
        path = self.root / batch / mode
        stamp = (path / "manifest.json").stat().st_mtime_ns
        cached = self._cache.get((batch, mode))
        if cached and cached[0] == stamp:
            return cached[1]
        if mode == "topic":
            from src.graph.topic_graph import TopicGraph
            graph = TopicGraph.load(path)
        else:
            from src.graph.object_graph import ObjectGraphIndex
            graph = ObjectGraphIndex.load(path)
        if graph.batch_id != batch:
            raise ValueError("graph batch mismatch")
        self._cache[(batch, mode)] = (stamp, graph)
        return graph

    def _encoder(self) -> Any:
        for channel in getattr(self.context.pipeline, "dense_channels", ()):
            if channel.name == "semantic_text":
                return channel.encoder
        # The semantic text channel may be disabled; the topic graph still needs its encoder.
        model = self.context.config.get("models", {}).get("text")
        if not model:
            raise RuntimeError("topic encoder unavailable")
        with self._lock:
            if self._fallback_encoder is None:
                from src.embedding.text_encoder import VietnameseTextEncoder

                cache = Path(model.get("cache_dir", "models/huggingface"))
                self._fallback_encoder = VietnameseTextEncoder(
                    model["model_name"], revision=model.get("revision"),
                    device=model.get("device", "cpu"), batch_size=int(model.get("batch_size", 16)),
                    cache_dir=cache if cache.is_absolute() else PROJECT_ROOT / cache,
                    local_files_only=True)
            return self._fallback_encoder

    def auto_mode(self, query: str, batches: tuple[str, ...]) -> str:
        """Route by query cues, then prefer a graph that actually exists for these batches."""
        mode, _ = graph_route(query)
        available = {m for b in batches for m in ("topic", "object")
                     if (self.root / b / m / "manifest.json").is_file()}
        if mode not in available and available:
            return next(iter(sorted(available)))
        return mode

    def object_hits(self, text: str, batches: tuple[str, ...], prior_hits: list[dict[str, Any]],
                    limit: int, max_candidates: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Rank frames by satisfied object constraints, conventional candidates first."""
        hits: list[dict[str, Any]] = []
        query_graphs: dict[str, Any] = {}
        with self._lock:
            for batch in batches:
                try:
                    graph = self._load(batch, "object")
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                query = graph.parse(text)
                if not query.nodes:
                    continue
                count = len(prior_hits)
                prior = {(h["video_id"], h["source_frame_idx"]): 1.0 - i / max(count, 1)
                         for i, h in enumerate(prior_hits) if h.get("batch_id") == batch
                         and isinstance(h.get("source_frame_idx"), int)}
                for match in graph.search(query, limit=limit, prior=prior, max_candidates=max_candidates):
                    hit = self._hit(batch, match, "object")
                    if hit is not None:
                        hit["fraction"] = float(match["score"])
                        hits.append(hit)
                query_graphs[batch] = query.to_dict(graph.classes)
        # Equal fractions alternate by per-batch rank, so one batch cannot fill the limit.
        hits.sort(key=lambda h: (-h["fraction"], h["channel_ranks"]["graph_object"]))
        return hits[:limit], query_graphs

    def warm(self, batches: tuple[str, ...]) -> None:
        """Load graph artifacts and the topic encoder so the first search is not slower."""
        with self._lock:
            for batch in batches:
                for mode in ("topic", "object"):
                    try:
                        self._load(batch, mode)
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        pass
        try:
            self._encoder().encode_text(["warm"])
        except Exception:  # noqa: BLE001 - warming is best effort
            pass

    def topic_chain(self, batch: str, topic_id: int, radius: int) -> list[dict[str, Any]]:
        """Return the NEXT chain around a topic for temporal browsing."""
        graph = self._load(batch, "topic")
        if not 0 <= topic_id < len(graph):
            raise KeyError(topic_id)
        return [{**graph.topic_payload(t.topic_id), "id": f"{batch}:t{t.topic_id}"}
                for t in graph.chain(topic_id, radius, radius)]

    def _hit(self, batch: str, match: dict[str, Any], mode: str) -> dict[str, Any] | None:
        reference = {"batch_id": batch, "video_id": match["video_id"],
                     "source_frame_idx": match["source_frame_idx"],
                     "keyframe_ordinal": match.get("keyframe_ordinal")}
        frame = resolve_image_frame(self.context.artifact_repository, reference)
        if frame is None:
            return None
        evidence = dict(match.get("evidence", {})) if mode == "object" else {
            key: match.get(key) for key in ("text", "transcript", "start_ms", "end_ms",
                "shot_start", "shot_end", "path", "cosine", "graph_bonus", "entities")}
        evidence["source"] = "detections" if mode == "object" else "asr"
        return {**reference, "keyframe_ordinal": frame.keyframe_ordinal,
                "timestamp_ms": frame.timestamp_ms, "shot_id": frame.shot_id,
                "fused_score": float(match["score"]), "graph_evidence": evidence,
                "channel_ranks": {f"graph_{mode}": match["rank"]}, "evidence": {}}

    def search(self, request: GraphRequest, batches: tuple[str, ...], mode: str) -> dict[str, Any]:
        """Return graph evidence or a reason to run conventional retrieval."""
        hits: list[dict[str, Any]] = []
        display: dict[str, Any] = {"nodes": [], "edges": []}
        statuses: dict[str, Any] = {}
        with self._lock:
            for batch in batches:
                reason = note = None
                try:
                    if not self.settings.get("enabled", True):
                        raise FileNotFoundError("disabled")
                    graph = self._load(batch, mode)
                    if mode == "topic":
                        encoder = self._encoder()
                        if graph.encoder_fingerprint != encoder.fingerprint:
                            raise ValueError("encoder mismatch")
                        texts = [request.query] + ([request.refinement] if request.refinement.strip() else [])
                        vectors = encoder.encode_text(texts)
                        result = graph.search(vectors[0], vectors[1] if len(texts) > 1 else None,
                                              final_k=request.top_k, video_ids=request.video_ids or None)
                    else:
                        result = graph.search_with_evidence(
                            " ".join((request.query, request.refinement)).strip(),
                            limit=request.top_k, video_ids=request.video_ids or None)
                    # Partial evidence is still ranked evidence (fraction of constraints met);
                    # the note stays visible in the batch status.
                    note = result.get("fallback_reason")
                    batch_hits = [hit for match in result.get("matches", ())
                                  if (hit := self._hit(batch, match, mode)) is not None]
                    if not batch_hits:
                        reason = note or "graph_evidence_missing"
                    hits.extend(batch_hits)
                    display["subgraph_size"] = display.get("subgraph_size", 0) + int(result.get("subgraph_size", 0))
                    for node in result.get("nodes", ()):
                        display["nodes"].append({**node, "id": f"{batch}:{node['id']}"})
                    for edge in result.get("edges", ()):
                        display["edges"].append({**edge, "source": f"{batch}:{edge['source']}",
                                                 "target": f"{batch}:{edge['target']}"})
                    if mode == "object":
                        display.setdefault("query_graphs", {})[batch] = result.get("query_graph", {})
                except FileNotFoundError:
                    reason = "graph_not_provisioned"
                except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                    reason = "graph_artifact_incompatible"
                except (RuntimeError, ImportError):
                    reason = "graph_encoder_unavailable"
                statuses[batch] = {"state": "fallback" if reason else "ready", "reason": reason}
                if not reason and mode == "object" and note:
                    statuses[batch]["note"] = note
        # Conventional retrieval takes over only when no batch produced graph evidence;
        # batches without a graph stay listed in batch_statuses.
        reason = None if hits else next((v["reason"] for v in statuses.values() if v["reason"]),
                                        "graph_evidence_missing")
        hits.sort(key=lambda hit: (-hit["fused_score"], hit["batch_id"], hit["video_id"], hit["source_frame_idx"]))
        for rank, hit in enumerate(hits, 1):
            hit["rank"] = rank
        return {"hits": hits[:request.top_k], "graph": display,
                "batch_statuses": statuses, "fallback_reason": reason}


class UnifiedRequest(BaseModel):
    """One search: conventional retrieval fused with whatever graph evidence exists."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8192)
    refinement: str = Field(default="", max_length=8192)
    top_k: int = Field(default=50, ge=1, le=100, strict=True)
    batch_filters: list[str] = Field(default_factory=list, max_length=8)
    reference_frame: dict[str, Any] | None = None
    query_id: str = Field(default_factory=lambda: f"search-{uuid4().hex}", max_length=128)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query is empty")
        return value.strip()

    @field_validator("batch_filters")
    @classmethod
    def identifiers(cls, values: list[str]) -> list[str]:
        if any(not _IDENTIFIER.fullmatch(v) or v in {".", ".."} for v in values):
            raise ValueError("invalid dataset identifier")
        return list(dict.fromkeys(values))


def install_browser_routes(
    app: Any, context: Any, *, search: Callable[..., dict[str, Any]],
    validate_batches: Callable[..., tuple[str, ...]],
    public_hits: Callable[..., list[dict[str, Any]]],
    refine: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Attach explicit asset routes and graph/assistant APIs to the shared runtime."""
    router = APIRouter()
    service = GraphService(context)
    app.state.graph_service = service
    web = PROJECT_ROOT / "app" / "web"

    @router.get("/", include_in_schema=False)
    def browser() -> FileResponse:
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-cache"})

    @router.get("/assets/{filename}", include_in_schema=False)
    def asset(filename: str) -> FileResponse:
        if filename not in {"app.js", "app.css"}:
            raise HTTPException(status_code=404, detail="asset_not_found")
        return FileResponse(web / filename, headers={"Cache-Control": "no-cache"})

    prewarm = VlmPrewarm(context.config)
    app.state.vlm_prewarm = prewarm

    graph_warm = {"started": False}

    @router.post("/v1/assistant/prewarm")
    def assistant_prewarm() -> dict[str, Any]:
        if not graph_warm["started"]:
            graph_warm["started"] = True
            try:
                batches = validate_batches(context, ())
            except (TypeError, ValueError):
                batches = ()
            threading.Thread(target=service.warm, args=(batches,), name="graph-warm", daemon=True).start()
        return {"state": prewarm.trigger()}

    @router.post("/v1/assistant/route")
    def assistant(request: AssistantRequest) -> dict[str, Any]:
        return assistant_reply(request, context.config)

    @router.get("/v1/graph/topic/{batch_id}/{topic_id}/chain")
    def topic_chain(batch_id: str, topic_id: int, radius: int = 3) -> dict[str, Any]:
        if not _IDENTIFIER.fullmatch(batch_id) or not 0 <= radius <= 10:
            raise HTTPException(status_code=422, detail="graph_request_invalid")
        try:
            with service._lock:
                return {"topics": service.topic_chain(batch_id, topic_id, radius)}
        except (FileNotFoundError, KeyError):
            raise HTTPException(status_code=404, detail="topic_not_found") from None
        except (OSError, ValueError):
            raise HTTPException(status_code=409, detail="graph_artifact_incompatible") from None

    @router.get("/v1/graph/status")
    def graph_status() -> dict[str, Any]:
        return {"batches": service.status(validate_batches(context, ())), "router": "rules"}

    from src.graph.fusion import FusionWeights, fuse

    fusion = FusionWeights.from_mapping(context.config.get("graph", {}).get("fusion"))

    @router.post("/v1/search")
    def unified_search(request: UnifiedRequest) -> dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        started = time.monotonic()
        try:
            batches = validate_batches(context, tuple(request.batch_filters))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="search_batch_invalid") from None
        refinement = request.refinement.strip()
        # Stage 2 narrows stage 1: conventional retrieval searches both descriptions together
        # (the legacy refine endpoint would replace the query), and the topic graph re-ranks
        # the stage-1 subgraph with stage 2. A reference frame uses visual feedback refinement.
        combined = f"{request.query}. {refinement}" if refinement else request.query
        payload: dict[str, Any] = {"query_id": request.query_id, "original_text": combined,
                                   "batch_filters": list(request.batch_filters), "limit": request.top_k}

        def conventional_call(body: dict[str, Any]) -> dict[str, Any]:
            if request.reference_frame and refine is not None:
                try:
                    return refine({**body, "original_text": request.query,
                                   "reference_frame": request.reference_frame,
                                   **({"feedback_text": refinement} if refinement else {})})
                except HTTPException as error:
                    if error.status_code not in {409, 422}:
                        raise
            return search(body)
        graph_request = GraphRequest(query=request.query, refinement=request.refinement,
                                     mode="topic", top_k=fusion.topic_matches,
                                     batch_filters=list(batches))
        object_route = graph_route(f"{request.query} {request.refinement}")[0] == "object"
        with ThreadPoolExecutor(max_workers=1) as pool:
            # The topic graph runs beside conventional retrieval; it is far faster.
            topic_future = pool.submit(service.search, graph_request, batches, "topic")
            conventional_error: HTTPException | None = None
            try:
                conventional = conventional_call(payload)
            except HTTPException as error:
                conventional, conventional_error = {"hits": []}, error
            topic = topic_future.result()
        topic_hits: list[dict[str, Any]] = []
        if not topic["fallback_reason"]:
            evidence = {(h["batch_id"], h["video_id"], h["source_frame_idx"]): h["graph_evidence"]
                        for h in topic["hits"]}
            for hit in public_hits(topic["hits"]):
                hit["graph_evidence"] = evidence[(hit["batch_id"], hit["video_id"], hit["source_frame_idx"])]
                topic_hits.append(hit)
        if conventional_error is not None and not topic_hits:
            raise conventional_error
        object_hits: list[dict[str, Any]] = []
        query_graphs: dict[str, Any] = {}
        if object_route:
            object_hits, query_graphs = service.object_hits(
                f"{request.query} {request.refinement}".strip(), batches, conventional.get("hits", []),
                limit=request.top_k, max_candidates=600)
        hits = fuse(conventional.get("hits", []), topic_hits, object_hits, weights=fusion, limit=request.top_k)
        graph = dict(topic["graph"]) if topic_hits else {"nodes": [], "edges": []}
        graph["query_graphs"] = query_graphs
        return {"query_id": request.query_id, "hits": hits, "graph": graph,
                "evidence": {"topic_matches": len(topic_hits), "object_matches": len(object_hits),
                             "conventional": len(conventional.get("hits", [])),
                             "conventional_available": conventional_error is None},
                "latency_ms": round((time.monotonic() - started) * 1000, 1)}

    @router.post("/v1/graph/search")
    def graph_search(request: GraphRequest) -> dict[str, Any]:
        started = time.monotonic()
        try:
            batches = validate_batches(context, tuple(request.batch_filters))
            if any(not _IDENTIFIER.fullmatch(batch) for batch in batches):
                raise ValueError("invalid batch")
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="graph_batch_invalid") from None
        mode = service.auto_mode(request.query, batches) if request.mode == "auto" else request.mode
        result = service.search(request, batches, mode)
        if result["fallback_reason"]:
            # Conventional retrieval has no equivalent exact video scope filter.
            # Do not silently broaden a caller's explicit scope.
            if request.video_ids:
                result.update(hits=[], graph={"nodes": [], "edges": []}, mode="conventional",
                              fallback_reason="scoped_graph_unavailable")
            else:
                try:
                    fallback = search({"query_id": request.query_id,
                        "original_text": " ".join((request.query, request.refinement)).strip(),
                        "batch_filters": list(batches), "limit": request.top_k})
                except HTTPException as error:
                    raise HTTPException(status_code=error.status_code, detail={
                        "code": "graph_and_conventional_unavailable",
                        "fallback_reason": result["fallback_reason"],
                        "conventional": error.detail}) from None
                result.update(fallback, mode="conventional", graph={"nodes": [], "edges": []})
        else:
            clean = public_hits(result["hits"])
            evidence = {(hit["batch_id"], hit["video_id"], hit["source_frame_idx"], hit.get("keyframe_ordinal")):
                        hit["graph_evidence"] for hit in result["hits"]}
            for hit in clean:
                hit["graph_evidence"] = evidence[(hit["batch_id"], hit["video_id"], hit["source_frame_idx"], hit.get("keyframe_ordinal"))]
            result.update(hits=clean, mode=mode)
        return {**result, "query_id": request.query_id, "requested_mode": request.mode,
                "latency_ms": round((time.monotonic() - started) * 1000, 1)}

    app.include_router(router)

    from src.submission.dres import install_dres_routes

    install_dres_routes(app, context)
