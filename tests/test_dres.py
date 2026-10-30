"""DRES client API v2 integration, exercised against an in-memory fake server."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.submission.dres import (
    DresClient,
    DresError,
    ResolvedFrame,
    install_dres_routes,
    kis_answer,
    normalize_server_url,
    qa_answer,
    submission_body,
    trake_answer,
)


class FakeDres:
    """Records requests and answers like DRES 2.x."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict, object]] = []
        self.verdict = (200, {"status": True, "submission": "CORRECT", "description": "Submission correct!"})

    def __call__(self, request, timeout):
        parts = urlsplit(request.full_url)
        body = json.loads(request.data) if request.data else None
        self.requests.append((request.get_method(), parts.path, parse_qs(parts.query), body))
        if parts.path == "/api/v2/login":
            if body["password"] != "pw":
                return 401, json.dumps({"status": False, "description": "Invalid credentials"}).encode()
            return 200, json.dumps({"id": "u1", "username": body["username"], "role": "PARTICIPANT",
                                    "sessionId": "S1"}).encode()
        if parse_qs(parts.query).get("session") not in (["S1"], ["pasted-session-123"]):
            return 401, json.dumps({"status": False, "description": "Unauthorized"}).encode()
        if parts.path == "/api/v2/client/evaluation/list":
            return 200, json.dumps([
                {"id": "e-old", "name": "Warm-up", "type": "SYNCHRONOUS", "status": "TERMINATED",
                 "templateId": "t", "teams": [], "taskTemplates": []},
                {"id": "e1", "name": "Final", "type": "SYNCHRONOUS", "status": "ACTIVE", "templateId": "t",
                 "teams": ["team"], "taskTemplates": [{"name": "kis-01", "taskGroup": "kis", "taskType": "KIS"}]},
            ]).encode()
        if parts.path == "/api/v2/client/evaluation/currentTask/e1":
            return 200, json.dumps({"name": "kis-01", "taskGroup": "kis", "taskType": "KIS", "duration": 300}).encode()
        if parts.path == "/api/v2/submit/e1":
            status, value = self.verdict
            return status, json.dumps(value).encode()
        if parts.path == "/api/v2/log/result/e1":
            return 200, json.dumps({"status": True, "description": "ok"}).encode()
        if parts.path == "/api/v2/logout":
            return 200, json.dumps({"status": True, "description": "bye"}).encode()
        return 404, json.dumps({"status": False, "description": "not found"}).encode()


class Repository:
    def get_frame(self, batch, video, source):
        if video == "L21_V001" and source in {100, 250, 400}:
            return SimpleNamespace(batch_id=batch, video_id=video, source_frame_idx=source,
                                   keyframe_ordinal=source // 50, timestamp_ms=source * 40)
        return None

    def get_ordinal_frame(self, batch, video, ordinal):
        return None


def client(tmp_path, fake=None, settings=None):
    fake = fake or FakeDres()
    app = FastAPI()
    config = {"services": {"dres": {"submission_log": str(tmp_path / "dres.jsonl"), **(settings or {})}}}
    service = install_dres_routes(app, SimpleNamespace(config=config, artifact_repository=Repository()),
                                  transport=fake)
    return TestClient(app), fake, service


FRAME = {"batch_id": "aic25-b1-reference", "video_id": "L21_V001", "source_frame_idx": 250}


def login(http):
    return http.post("/v1/dres/login", json={"server_url": "http://dres.local:8080/", "username": "team",
                                              "password": "pw"})


def test_payload_shapes_follow_the_client_api():
    frame = ResolvedFrame("b1", "L21_V001", 250, 5, 10_000)
    assert submission_body([kis_answer(frame, {}, None)]) == {
        "answerSets": [{"answers": [{"mediaItemName": "L21_V001", "start": 10_000, "end": 10_000}]}]}
    assert kis_answer(frame, {"kis_end_offset_ms": 500}, "AIC")["end"] == 10_500
    assert kis_answer(frame, {}, "AIC")["mediaItemCollectionName"] == "AIC"
    assert qa_answer(frame, "3", {"qa_text": "QA-{answer}-{video_id}-{time_ms}"}) == {"text": "QA-3-L21_V001-10000"}
    frames = [ResolvedFrame("b1", "L21_V001", n, None, n * 40) for n in (100, 250, 400)]
    assert trake_answer(frames, {"trake_text": "TR-{video_id}-{frames}"}) == {"text": "TR-L21_V001-100,250,400"}
    with pytest.raises(ValueError):
        trake_answer(list(reversed(frames)), {})


@pytest.mark.parametrize("url", ["ftp://x", "dres.local", "http://user:pw@dres.local", "http://h/?a=1"])
def test_server_url_is_validated(url):
    with pytest.raises(ValueError):
        normalize_server_url(url)


def test_login_selects_the_single_active_evaluation_and_keeps_no_password(tmp_path):
    http, fake, service = client(tmp_path)
    state = login(http).json()
    assert state["connected"] and state["evaluation_id"] == "e1" and state["evaluation_name"] == "Final"
    assert [e["id"] for e in state["evaluations"]] == ["e-old", "e1"]
    assert "pw" not in json.dumps(service.state())
    assert http.get("/v1/dres/task").json()["task"]["name"] == "kis-01"


def test_wrong_password_is_an_actionable_error(tmp_path):
    http, _, _ = client(tmp_path)
    response = http.post("/v1/dres/login", json={"server_url": "http://dres.local", "username": "t", "password": "x"})
    assert response.status_code == 401
    assert response.json()["detail"]["message"] == "Invalid credentials"


def test_submit_uses_canonical_time_not_browser_values(tmp_path):
    http, fake, _ = client(tmp_path)
    login(http)
    result = http.post("/v1/dres/submit", json={"task": "kis", "frame": {**FRAME, "timestamp_ms": 1}}).json()
    method, path, query, body = fake.requests[-1]
    assert (method, path, query["session"]) == ("POST", "/api/v2/submit/e1", ["S1"])
    assert body == {"answerSets": [{"answers": [{"mediaItemName": "L21_V001", "start": 10_000, "end": 10_000}]}]}
    assert result["accepted"] and result["verdict"] == "CORRECT"
    logged = [json.loads(line) for line in (tmp_path / "dres.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged[0]["body"] == body and logged[0]["result"]["submission"] == "CORRECT"


def test_qa_and_trake_text_answers(tmp_path):
    http, fake, _ = client(tmp_path, settings={"qa_answer_spec": ""})
    login(http)
    http.post("/v1/dres/submit", json={"task": "qa", "frame": FRAME, "answer": "  3   quả "})
    assert fake.requests[-1][3]["answerSets"][0]["answers"][0]["text"] == "QA-3 quả-L21_V001-10000"
    frames = [{**FRAME, "source_frame_idx": n} for n in (100, 250, 400)]
    http.post("/v1/dres/submit", json={"task": "trake", "frames": frames})
    assert fake.requests[-1][3]["answerSets"][0]["answers"][0]["text"] == "TR-L21_V001-100,250,400"


def test_rejected_submission_is_a_verdict_not_a_crash(tmp_path):
    fake = FakeDres()
    fake.verdict = (412, {"status": False, "description": "Duplicate submission"})
    http, _, _ = client(tmp_path, fake)
    login(http)
    result = http.post("/v1/dres/submit", json={"task": "kis", "frame": FRAME}).json()
    assert result == {**result, "accepted": False, "http_status": 412, "description": "Duplicate submission"}


@pytest.mark.parametrize("payload", [
    {"task": "kis"},
    {"task": "kis", "frame": {**FRAME, "source_frame_idx": 7}},
    {"task": "qa", "frame": FRAME, "answer": " "},
    {"task": "qa", "frame": FRAME, "answer": "a\u0000b"},
    {"task": "kis", "frame": {**FRAME, "video_id": "../x"}},
])
def test_invalid_answers_never_reach_dres(tmp_path, payload):
    http, fake, _ = client(tmp_path)
    login(http)
    before = len(fake.requests)
    assert http.post("/v1/dres/submit", json=payload).status_code == 422
    assert len(fake.requests) == before


def test_submit_requires_a_session_and_preview_sends_nothing(tmp_path):
    http, fake, _ = client(tmp_path)
    assert http.post("/v1/dres/submit", json={"task": "kis", "frame": FRAME}).status_code == 409
    preview = http.post("/v1/dres/preview", json={"task": "kis", "frame": FRAME}).json()
    assert preview["body"]["answerSets"][0]["answers"][0]["start"] == 10_000
    assert "L21_V001" in preview["summary"]
    assert fake.requests == []


def test_result_logging_is_opt_in(tmp_path):
    http, fake, _ = client(tmp_path)
    login(http)
    assert http.post("/v1/dres/log", json={"query": "q", "frames": [FRAME]}).json() == {"sent": False}
    http, fake, _ = client(tmp_path, settings={"log_results": True})
    login(http)
    assert http.post("/v1/dres/log", json={"query": "q", "frames": [FRAME]}).json() == {"sent": True}
    log = fake.requests[-1][3]
    assert log["results"][0] == {"answer": {"mediaItemName": "L21_V001", "start": 10_000, "end": 10_000}, "rank": 1}
    assert log["events"][0]["category"] == "TEXT"


def test_transport_failure_is_reported(tmp_path):
    def offline(request, timeout):
        raise OSError("connection refused")
    with pytest.raises(DresError) as error:
        DresClient("http://dres.local", transport=offline).login("a", "b")
    assert error.value.status == 0


def test_logout_clears_the_session(tmp_path):
    http, fake, _ = client(tmp_path)
    login(http)
    assert http.post("/v1/dres/logout").json()["connected"] is False
    assert fake.requests[-1][1] == "/api/v2/logout"


def test_a_pasted_session_id_is_validated_without_a_password_login(tmp_path):
    http, fake, _ = client(tmp_path)
    good = http.post("/v1/dres/login", json={"server_url": "https://eventretrieval.one",
                                             "session_id": "pasted-session-123"}).json()
    assert good["connected"] and good["evaluation_id"] == "e1"
    assert all(path != "/api/v2/login" for _, path, _, _ in fake.requests)
    http.post("/v1/dres/submit", json={"task": "kis", "frame": FRAME})
    assert fake.requests[-1][2]["session"] == ["pasted-session-123"]
    stale = http.post("/v1/dres/login", json={"server_url": "https://eventretrieval.one", "session_id": "expired-session"})
    assert stale.status_code == 401
    assert http.post("/v1/dres/login", json={"server_url": "https://h", "session_id": "bad token!"}).status_code == 422


def test_each_query_can_target_its_own_evaluation(tmp_path):
    fake = FakeDres()
    http, _, _ = client(tmp_path, fake)
    login(http)
    http.post("/v1/dres/submit", json={"task": "kis", "frame": FRAME, "evaluation_id": "e-old"})
    assert fake.requests[-1][1] == "/api/v2/submit/e-old"
