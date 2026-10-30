"""Browser/graph HTTP contract tests with offline in-memory evidence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api import _public_hits, create_app
from src.browser_api import GraphRequest, GraphService, install_browser_routes, route_message


def context(tmp_path):
    frame = SimpleNamespace(keyframe_ordinal=2, timestamp_ms=1000, shot_id="s1")
    return SimpleNamespace(config={"graph": {"root": str(tmp_path)}},
                           pipeline=SimpleNamespace(dense_channels=[]),
                           artifact_repository=SimpleNamespace(get_frame=lambda *args: frame))


def browser_client(ctx, search=None):
    app = FastAPI()
    calls = []

    def conventional(payload):
        calls.append(payload)
        return {"query_id": payload["query_id"], "hits": [{"video_id": "fallback"}]}

    install_browser_routes(app, ctx, search=search or conventional,
                           validate_batches=lambda c, batches: batches or ("b1",),
                           public_hits=_public_hits)
    return TestClient(app), app.state.graph_service, calls


def test_static_assets_are_bounded_and_real(tmp_path):
    client, _, _ = browser_client(context(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert "EviBind Browser" in page.text
    assert "text/html" in page.headers["content-type"]
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/app.css").status_code == 200
    assert client.get("/assets/settings.yaml").status_code == 404
    assert client.get("/assets/%2e%2e%2f.env").status_code == 404


def test_real_app_installs_browser_routes(tmp_path):
    app = create_app(context(tmp_path))
    assert TestClient(app).get("/").status_code == 200


@pytest.mark.parametrize("message,intent,mode", [
    ("Người đầu bếp cắt rau sau đó cho vào chảo", "video_search", "topic"),
    ("Có bao nhiêu người trong ảnh?", "question_answering", "object"),
    ("Gợi ý truy vấn tìm cảnh nấu ăn", "search_assistance", "topic"),
    ("Hai người đàn ông bên cạnh chiếc xe hơi", "video_search", "object"),
    ("Phóng viên nói về giá xăng", "video_search", "topic"),
    ("A dog to the left of a car", "video_search", "object"),
])
def test_assistant_routes_without_inventing_answers(message, intent, mode):
    result = route_message(message, True)
    assert (result["intent"], result["mode"], result["router"]) == (intent, mode, "rules")
    assert result["query"] == message
    assert "answer" not in result


@pytest.mark.parametrize("payload", [
    {"query": " "}, {"query": "a", "batch_filters": ["../models"]},
    {"query": "a", "top_k": True}, {"query": "a", "top_k": 101},
    {"query": "a", "mode": "invented"}, {"query": "a", "video_ids": ["a/b"]},
])
def test_invalid_graph_request_is_rejected(tmp_path, payload):
    client, _, calls = browser_client(context(tmp_path))
    assert client.post("/v1/graph/search", json=payload).status_code == 422
    assert not calls


def test_missing_graph_falls_back_preserving_scope_and_refinement(tmp_path):
    client, _, calls = browser_client(context(tmp_path))
    value = client.post("/v1/graph/search", json={"query": "dog", "refinement": "by a tree",
                        "batch_filters": ["b1", "b2"], "top_k": 20}).json()
    assert value["mode"] == "conventional"
    assert value["fallback_reason"] == "graph_not_provisioned"
    assert value["hits"] == [{"video_id": "fallback"}]
    assert calls[0]["batch_filters"] == ["b1", "b2"]
    assert calls[0]["original_text"] == "dog by a tree"
    assert set(value["batch_statuses"]) == {"b1", "b2"}


def test_fallback_failure_stays_an_error_with_a_reason(tmp_path):
    def unavailable(payload):
        raise HTTPException(status_code=503, detail={"code": "kis_search_unavailable"})
    client, _, _ = browser_client(context(tmp_path), unavailable)
    response = client.post("/v1/graph/search", json={"query": "dog"})
    assert response.status_code == 503
    assert response.json()["detail"]["fallback_reason"] == "graph_not_provisioned"


def test_video_filter_never_silently_broadens_on_fallback(tmp_path):
    client, _, calls = browser_client(context(tmp_path))
    result = client.post("/v1/graph/search", json={"query": "dog", "video_ids": ["v1"]}).json()
    assert result["hits"] == []
    assert result["fallback_reason"] == "scoped_graph_unavailable"
    assert calls == []


def test_object_evidence_uses_canonical_frame_and_survives_public_filter(tmp_path, monkeypatch):
    client, service, calls = browser_client(context(tmp_path))
    evidence = {"nodes": [{"id": "n0", "label": "dog", "box": [0, 0, 1, 1]}], "edges": []}
    graph = SimpleNamespace(search_with_evidence=lambda *args, **kwargs: {
        "matches": [{"rank": 1, "score": 1.0, "video_id": "v1", "source_frame_idx": 10,
                     "keyframe_ordinal": 2, "evidence": evidence}], "fallback_reason": None})
    monkeypatch.setattr(service, "_load", lambda *args: graph)
    value = client.post("/v1/graph/search", json={"query": "dog", "mode": "object"}).json()
    assert value["mode"] == "object"
    hit = value["hits"][0]
    assert hit["timestamp_ms"] == 1000
    assert hit["thumbnail_uri"] == "/v1/frames/b1/v1/10/image"
    assert hit["graph_evidence"]["nodes"] == evidence["nodes"]
    assert hit["graph_evidence"]["source"] == "detections"
    assert not calls


def test_topic_encoder_mismatch_falls_back_before_inference(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.pipeline.dense_channels = [SimpleNamespace(name="semantic_text", encoder=SimpleNamespace(fingerprint="new"))]
    client, service, calls = browser_client(ctx)
    monkeypatch.setattr(service, "_load", lambda *args: SimpleNamespace(encoder_fingerprint="old"))
    value = client.post("/v1/graph/search", json={"query": "then cook", "mode": "topic"}).json()
    assert value["fallback_reason"] == "graph_artifact_incompatible"
    assert len(calls) == 1


def test_topic_paths_have_unique_batch_namespaces(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.pipeline.dense_channels = [SimpleNamespace(name="semantic_text", encoder=SimpleNamespace(
        fingerprint="encoder1", encode_text=lambda texts: np.array([[1, 0]] * len(texts))))]
    service = GraphService(ctx)
    graph = SimpleNamespace(encoder_fingerprint="encoder1", search=lambda *args, **kwargs: {
        "matches": [{"rank": 1, "score": .8, "video_id": "v1", "source_frame_idx": 10}],
        "nodes": [{"id": "t0"}, {"id": "t1"}],
        "edges": [{"source": "t0", "target": "t1", "relation": "NEXT"}]})
    monkeypatch.setattr(service, "_load", lambda *args: graph)
    result = service.search(GraphRequest(query="cook"), ("b1", "b2"), "topic")
    assert {node["id"] for node in result["graph"]["nodes"]} == {"b1:t0", "b1:t1", "b2:t0", "b2:t1"}
    assert result["graph"]["edges"][1]["source"] == "b2:t0"


def test_graph_construction_is_side_effect_free(tmp_path):
    service = GraphService(context(tmp_path / "absent"))
    assert not service.root.exists()
    assert service.status(("b1",))["b1"]["topic"]["state"] == "not_provisioned"
    assert not service.root.exists()


def test_llm_router_is_optional_and_falls_back_to_rules(monkeypatch):
    import src.browser_api as browser

    request = browser.AssistantRequest(message="có bao nhiêu quả chuối trong món bún", graph_mode=True)
    calls = []
    monkeypatch.setattr(browser, "llm_route", lambda message, settings: calls.append(settings) or None)
    rules = browser.assistant_reply(request, {"assistant": {"router": "rules"}})
    assert rules["router"] == "rules" and not calls
    fallback = browser.assistant_reply(request, {"assistant": {"router": "auto"},
                                                 "services": {"ollama_vlm": {"model": "m"}}})
    assert fallback["router"] == "rules" and len(calls) == 1
    monkeypatch.setattr(browser, "llm_route", lambda message, settings: {
        "intent": "question_answering", "route": "object", "query": "món bún sườn non om chuối đậu",
        "question": "có bao nhiêu quả chuối"})
    value = browser.assistant_reply(request, {"assistant": {"router": "auto"},
                                              "services": {"ollama_vlm": {"model": "m"}}})
    assert (value["router"], value["intent"], value["mode"]) == ("llm", "question_answering", "object")
    assert value["query"] == "món bún sườn non om chuối đậu"
    assert "answer" not in value


def test_llm_route_rejects_remote_hosts_and_invalid_output(monkeypatch):
    import src.browser_api as browser

    assert browser.llm_route("x", {"model": "m", "url": "http://example.com"}) is None

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            import json
            return json.dumps({"message": {"content": self.content}}).encode()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response('{"intent": "delete_all"}'))
    assert browser.llm_route("x", {"model": "m"}) is None
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response(
        '{"intent": "video_search", "route": "topic", "search_query": " cảnh  nấu ăn ", "question": ""}'))
    assert browser.llm_route("x", {"model": "m"}) == {
        "intent": "video_search", "route": "topic", "query": "cảnh nấu ăn", "question": None}


def test_auto_mode_prefers_an_existing_graph(tmp_path):
    service = GraphService(context(tmp_path))
    assert service.auto_mode("hai người bên cạnh xe hơi", ("b1",)) == "object"
    (tmp_path / "b1" / "topic").mkdir(parents=True)
    (tmp_path / "b1" / "topic" / "manifest.json").write_text("{}", encoding="utf-8")
    assert service.auto_mode("hai người bên cạnh xe hơi", ("b1",)) == "topic"
    assert service.auto_mode("phóng viên nói về thời tiết", ("b1",)) == "topic"


def test_batches_without_a_graph_do_not_discard_graph_evidence(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.pipeline.dense_channels = [SimpleNamespace(name="semantic_text", encoder=SimpleNamespace(
        fingerprint="e", encode_text=lambda texts: np.array([[1, 0]] * len(texts))))]
    client, service, calls = browser_client(ctx)
    graph = SimpleNamespace(encoder_fingerprint="e", search=lambda *args, **kwargs: {
        "matches": [{"rank": 1, "score": .8, "video_id": "v1", "source_frame_idx": 10}],
        "nodes": [], "edges": [], "subgraph_size": 7})

    def load(batch, mode):
        if batch == "b2":
            raise FileNotFoundError(batch)
        return graph
    monkeypatch.setattr(service, "_load", load)
    value = client.post("/v1/graph/search", json={"query": "cook", "mode": "topic",
                                                   "batch_filters": ["b1", "b2"]}).json()
    assert value["mode"] == "topic" and value["fallback_reason"] is None
    assert value["batch_statuses"]["b2"]["reason"] == "graph_not_provisioned"
    assert value["graph"]["subgraph_size"] == 7
    assert len(value["hits"]) == 1 and not calls


def test_topic_chain_route(tmp_path, monkeypatch):
    client, service, _ = browser_client(context(tmp_path))
    from src.graph.topic_graph import Topic, TopicGraph

    topics = [Topic(i, "b1", "v1", i, i * 1000, i * 1000 + 900, f"t{i}", i * 10) for i in range(5)]
    graph = TopicGraph("b1", topics, np.eye(5, 4, dtype=np.float32) + .1,
                       [(i, i + 1, "NEXT", 1.0) for i in range(4)], [])
    monkeypatch.setattr(service, "_load", lambda *args: graph)
    value = client.get("/v1/graph/topic/b1/2/chain?radius=1").json()
    assert [t["ordinal"] for t in value["topics"]] == [1, 2, 3]
    assert value["topics"][0]["id"] == "b1:t1"
    assert client.get("/v1/graph/topic/b1/9/chain").status_code == 404
    assert client.get("/v1/graph/topic/b1/1/chain?radius=99").status_code == 422


def test_auto_router_skips_the_model_when_rules_suffice(monkeypatch):
    import src.browser_api as browser

    calls = []
    monkeypatch.setattr(browser, "llm_route", lambda message, settings: calls.append(message) or None)
    config = {"assistant": {"router": "auto"}, "services": {"ollama_vlm": {"model": "m"}}}
    browser.assistant_reply(browser.AssistantRequest(message="cảnh nấu ăn trong bếp"), config)
    browser.assistant_reply(browser.AssistantRequest(message="có bao nhiêu người?", has_frame=True), config)
    assert calls == []
    browser.assistant_reply(browser.AssistantRequest(message="có bao nhiêu người?"), config)
    assert calls == ["có bao nhiêu người?"]


def test_llm_route_shares_the_vlm_context_size(monkeypatch):
    import json
    import urllib.request

    import src.browser_api as browser

    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"content": '{"intent": "video_search"}'}}).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: sent.append(json.loads(request.data)) or Response())
    browser.llm_route("x", {"model": "m", "num_ctx": 4096, "execution_mode": "cpu"})
    assert sent[0]["options"]["num_ctx"] == 2048 and sent[0]["options"]["num_gpu"] == 0


def test_vlm_prewarm_runs_once_in_the_background(monkeypatch):
    import urllib.request

    import src.browser_api as browser

    assert browser.VlmPrewarm({}).trigger() == "skipped"
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: calls.append(request) or Response())
    warm = browser.VlmPrewarm({"services": {"ollama_vlm": {"model": "m", "url": "http://127.0.0.1:11434"}}})
    assert warm.trigger() == "warming"
    warm._thread.join(5)
    assert warm.state == "warm" and len(calls) == 1
    assert warm.trigger() == "warm" and len(calls) == 1


def test_unified_search_fuses_topic_evidence_and_survives_missing_graphs(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.pipeline.dense_channels = [SimpleNamespace(name="semantic_text", encoder=SimpleNamespace(
        fingerprint="e", encode_text=lambda texts: np.array([[1, 0]] * len(texts))))]
    calls = []

    def conventional(payload):
        calls.append(payload)
        return {"hits": [{"batch_id": "b1", "video_id": "v1", "source_frame_idx": 1, "timestamp_ms": 1000},
                         {"batch_id": "b1", "video_id": "v2", "source_frame_idx": 10, "timestamp_ms": 1000}]}

    client, service, _ = browser_client(ctx, conventional)
    # No graph artifacts: the conventional order is returned unchanged.
    plain = client.post("/v1/search", json={"query": "dog"}).json()
    assert [h["video_id"] for h in plain["hits"]] == ["v1", "v2"]
    assert plain["evidence"]["topic_matches"] == 0
    graph = SimpleNamespace(encoder_fingerprint="e", search=lambda *args, **kwargs: {
        "matches": [{"rank": 1, "score": .9, "video_id": "v2", "source_frame_idx": 10,
                     "start_ms": 0, "end_ms": 5000, "text": "a dog barks"}],
        "nodes": [{"id": "t0"}], "edges": [], "subgraph_size": 3})
    monkeypatch.setattr(service, "_load", lambda batch, mode: graph if mode == "topic" else (_ for _ in ()).throw(FileNotFoundError()))
    fused = client.post("/v1/search", json={"query": "dog", "refinement": "barking"}).json()
    assert fused["hits"][0]["video_id"] == "v2"
    assert fused["hits"][0]["graph_evidence"]["text"] == "a dog barks"
    assert fused["graph"]["nodes"] == [{"id": "b1:t0"}]
    assert calls[-1]["original_text"] == "dog. barking" and "feedback_text" not in calls[-1]



def test_object_hits_alternate_batches_when_fractions_tie(tmp_path, monkeypatch):
    _, service, _ = browser_client(context(tmp_path))
    query = SimpleNamespace(nodes=[1], to_dict=lambda classes: {})
    graph = SimpleNamespace(classes=[], parse=lambda text: query, search=lambda *args, **kwargs: [
        {"rank": rank, "score": 1.0, "video_id": f"v{rank}", "source_frame_idx": rank} for rank in (1, 2, 3)])
    monkeypatch.setattr(service, "_load", lambda batch, mode: graph)
    monkeypatch.setattr(service, "_hit", lambda batch, match, mode: {
        "batch_id": batch, "video_id": match["video_id"], "fused_score": match["score"],
        "channel_ranks": {"graph_object": match["rank"]}})
    hits, _ = service.object_hits("two cars", ("b1", "b2"), [], limit=4, max_candidates=10)
    assert [(h["batch_id"], h["video_id"]) for h in hits] == [
        ("b1", "v1"), ("b2", "v1"), ("b1", "v2"), ("b2", "v2")]
