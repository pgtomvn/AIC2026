"""DRES (Distributed Retrieval Evaluation Server) client API v2 submission.

DRES is the evaluation server used by VBS and by the HCMC AI Challenge. The client flow is:
``POST /api/v2/login`` → ``GET /api/v2/client/evaluation/list`` → (optionally
``GET /api/v2/client/evaluation/currentTask/{id}``) → ``POST /api/v2/submit/{id}`` with an
``ApiClientSubmission`` body ``{"answerSets": [{"answers": [...]}]}``.

Answer shapes follow the HCMC AI Challenge convention and are configurable under
``services.dres.formats``:

* KIS: ``{"mediaItemName": video, "start": ms, "end": ms}``
* Q&A: ``{"text": "QA-<answer>-<video>-<ms>"}``
* TRAKE: ``{"text": "TR-<video>-<frame_1>,...,<frame_n>"}``

Times and frame numbers are always resolved from canonical frame records, never from browser
values. Nothing connects to a server at import or construction time.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import PROJECT_ROOT
from src.ingest.frame_identity import resolve_image_frame

Task = Literal["kis", "qa", "trake"]
_VIDEO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SESSION = re.compile(r"[A-Za-z0-9_.~-]{8,256}\Z")
_DEFAULT_FORMATS = {
    "qa_text": "QA-{answer}-{video_id}-{time_ms}",
    "trake_text": "TR-{video_id}-{frames}",
    "kis_end_offset_ms": 0,
}


class DresError(RuntimeError):
    """A DRES call failed; ``status`` is the HTTP status (0 for transport errors)."""

    def __init__(self, status: int, description: str) -> None:
        super().__init__(description)
        self.status = status
        self.description = description


def normalize_server_url(value: str) -> str:
    """Accept only an absolute http(s) origin with an optional path prefix."""
    url = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("DRES server must be an http(s) URL such as http://192.168.1.10:8080")
    if parsed.username or parsed.password:
        raise ValueError("put credentials in the login form, not in the URL")
    return url


def clean_answer(value: str, limit: int = 256) -> str:
    """Collapse whitespace and reject control characters in a free-text answer."""
    text = " ".join(unicodedata.normalize("NFC", str(value)).split())
    if not text or len(text) > limit or any(unicodedata.category(c)[0] == "C" for c in text):
        raise ValueError(f"answer must be 1-{limit} printable characters")
    return text


@dataclass(frozen=True, slots=True)
class ResolvedFrame:
    """Canonical coordinates used for a DRES answer."""

    batch_id: str
    video_id: str
    source_frame_idx: int
    keyframe_ordinal: int | None
    time_ms: int


def kis_answer(frame: ResolvedFrame, formats: Mapping[str, Any], collection: str | None) -> dict[str, Any]:
    """A temporal KIS answer: the frame time as a (possibly widened) interval."""
    offset = int(formats.get("kis_end_offset_ms", 0))
    answer: dict[str, Any] = {"mediaItemName": frame.video_id, "start": frame.time_ms,
                              "end": frame.time_ms + max(0, offset)}
    if collection:
        answer["mediaItemCollectionName"] = collection
    return answer


def qa_answer(frame: ResolvedFrame, answer: str, formats: Mapping[str, Any]) -> dict[str, Any]:
    template = str(formats.get("qa_text", _DEFAULT_FORMATS["qa_text"]))
    return {"text": template.format(answer=answer, video_id=frame.video_id, time_ms=frame.time_ms,
                                    frame=frame.source_frame_idx)}


def trake_answer(frames: Sequence[ResolvedFrame], formats: Mapping[str, Any]) -> dict[str, Any]:
    if not 1 <= len(frames) <= 20 or len({f.video_id for f in frames}) != 1:
        raise ValueError("TRAKE needs 1-20 frames from one video")
    numbers = [f.source_frame_idx for f in frames]
    if any(a >= b for a, b in zip(numbers, numbers[1:])):
        raise ValueError("TRAKE frames must be in increasing time order")
    template = str(formats.get("trake_text", _DEFAULT_FORMATS["trake_text"]))
    return {"text": template.format(video_id=frames[0].video_id, frames=",".join(map(str, numbers)))}


def submission_body(answers: Sequence[Mapping[str, Any]], task_name: str | None = None) -> dict[str, Any]:
    """Wrap answers in an ``ApiClientSubmission`` with one answer set."""
    answer_set: dict[str, Any] = {"answers": [dict(a) for a in answers]}
    if task_name:
        answer_set["taskName"] = task_name
    return {"answerSets": [answer_set]}


Transport = Callable[[urllib.request.Request, float], tuple[int, bytes]]


def _urllib_transport(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - validated URL
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class DresClient:
    """Minimal JSON client for the DRES client API v2 (session passed as a query value)."""

    def __init__(self, server_url: str, *, timeout: float = 10.0, transport: Transport | None = None) -> None:
        self.server_url = normalize_server_url(server_url)
        self.timeout = float(timeout)
        self._transport = transport or _urllib_transport

    def _call(self, method: str, path: str, *, session: str | None = None,
              body: Any = None) -> tuple[int, Any]:
        query = f"?{urllib.parse.urlencode({'session': session})}" if session else ""
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.server_url}{path}{query}", data=data,
                                         headers=headers, method=method)
        try:
            status, raw = self._transport(request, self.timeout)
        except (OSError, urllib.error.URLError) as error:
            raise DresError(0, f"cannot reach DRES server: {getattr(error, 'reason', error)}") from None
        try:
            value = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        if status >= 400:
            description = value.get("description") if isinstance(value, dict) else None
            raise DresError(status, str(description or f"DRES returned HTTP {status}"))
        return status, value

    def login(self, username: str, password: str) -> dict[str, Any]:
        _, user = self._call("POST", "/api/v2/login", body={"username": username, "password": password})
        if not isinstance(user, dict) or not user.get("sessionId"):
            raise DresError(502, "DRES login returned no session")
        return user

    def logout(self, session: str) -> None:
        self._call("GET", "/api/v2/logout", session=session)

    def evaluations(self, session: str) -> list[dict[str, Any]]:
        _, value = self._call("GET", "/api/v2/client/evaluation/list", session=session)
        return [item for item in value or [] if isinstance(item, dict) and item.get("id")]

    def current_task(self, session: str, evaluation_id: str) -> dict[str, Any] | None:
        try:
            _, value = self._call("GET", f"/api/v2/client/evaluation/currentTask/{urllib.parse.quote(evaluation_id, safe='')}",
                                  session=session)
        except DresError as error:
            if error.status == 404:
                return None
            raise
        return value if isinstance(value, dict) else None

    def submit(self, session: str, evaluation_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        status, value = self._call("POST", f"/api/v2/submit/{urllib.parse.quote(evaluation_id, safe='')}",
                                   session=session, body=body)
        result = dict(value) if isinstance(value, dict) else {}
        result["http_status"] = status
        return result

    def log_results(self, session: str, evaluation_id: str, log: Mapping[str, Any]) -> None:
        self._call("POST", f"/api/v2/log/result/{urllib.parse.quote(evaluation_id, safe='')}",
                   session=session, body=log)


@dataclass(slots=True)
class _Session:
    client: DresClient
    session_id: str
    username: str
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    evaluation_id: str | None = None


class DresService:
    """Holds one operator session in server memory; the password is never stored."""

    def __init__(self, config: Mapping[str, Any], repository: Any,
                 *, transport: Transport | None = None, log_path: Path | None = None) -> None:
        self.settings = dict(config.get("services", {}).get("dres", {}) or {})
        self.formats = {**_DEFAULT_FORMATS, **dict(self.settings.get("formats", {}) or {})}
        self.repository = repository
        self._transport = transport
        self._session: _Session | None = None
        self._lock = threading.RLock()
        configured = self.settings.get("submission_log", "logs/dres-submissions.jsonl")
        path = Path(log_path or configured)
        self.log_path = path if path.is_absolute() else PROJECT_ROOT / path

    # ---- session -------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            evaluation = None
            if session and session.evaluation_id:
                evaluation = next((e for e in session.evaluations if e.get("id") == session.evaluation_id), None)
            return {
                "connected": session is not None,
                "server_url": session.client.server_url if session else str(self.settings.get("url", "") or ""),
                "username": session.username if session else "",
                "evaluations": [_evaluation_summary(e) for e in (session.evaluations if session else [])],
                "evaluation_id": session.evaluation_id if session else None,
                "evaluation_name": evaluation.get("name") if evaluation else None,
                "log_results": bool(self.settings.get("log_results", False)),
            }

    def login(self, server_url: str, username: str, password: str, session_id: str = "") -> dict[str, Any]:
        """Log in with credentials, or adopt a session ID issued by the DRES web client."""
        url = server_url or str(self.settings.get("url", "") or "")
        if not url:
            raise ValueError("enter the DRES server address")
        client = DresClient(url, timeout=float(self.settings.get("timeout_seconds", 10)), transport=self._transport)
        token = session_id.strip()
        if token:
            if not _SESSION.fullmatch(token):
                raise ValueError("the session ID has unexpected characters")
            session = _Session(client, token, username or "session")
        else:
            user = username or _secret(self.settings.get("username_env", "DRES_USERNAME"))
            secret = password or _secret(self.settings.get("password_env", "DRES_PASSWORD"))
            if not user or not secret:
                raise ValueError("enter the DRES username and password, or a session ID")
            account = client.login(user, secret)
            session = _Session(client, str(account["sessionId"]), str(account.get("username") or user))
        # Listing evaluations also proves that an adopted session is valid.
        session.evaluations = client.evaluations(session.session_id)
        active = [e for e in session.evaluations if e.get("status") == "ACTIVE"]
        if len(active) == 1:
            session.evaluation_id = str(active[0]["id"])
        with self._lock:
            self._session = session
        return self.state()

    def logout(self) -> dict[str, Any]:
        with self._lock:
            session, self._session = self._session, None
        if session:
            try:
                session.client.logout(session.session_id)
            except DresError:
                pass  # The local session is cleared either way.
        return self.state()

    def _require(self) -> _Session:
        with self._lock:
            if self._session is None:
                raise PermissionError("not connected to DRES")
            return self._session

    def refresh(self) -> dict[str, Any]:
        session = self._require()
        evaluations = session.client.evaluations(session.session_id)
        with self._lock:
            session.evaluations = evaluations
            if session.evaluation_id and not any(e.get("id") == session.evaluation_id for e in evaluations):
                session.evaluation_id = None
        return self.state()

    def select(self, evaluation_id: str) -> dict[str, Any]:
        session = self._require()
        if not any(e.get("id") == evaluation_id for e in session.evaluations):
            raise ValueError("unknown evaluation")
        with self._lock:
            session.evaluation_id = evaluation_id
        return self.state()

    def current_task(self) -> dict[str, Any] | None:
        session = self._require()
        if not session.evaluation_id:
            return None
        return session.client.current_task(session.session_id, session.evaluation_id)

    # ---- answers -------------------------------------------------------------
    def resolve(self, reference: Mapping[str, Any]) -> ResolvedFrame:
        frame = resolve_image_frame(self.repository, reference)
        if frame is None:
            raise ValueError("frame is not in the canonical catalog")
        timestamp = getattr(frame, "timestamp_ms", None)
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("frame has no canonical timestamp")
        return ResolvedFrame(str(reference["batch_id"]), frame.video_id, int(frame.source_frame_idx),
                             getattr(frame, "keyframe_ordinal", None), timestamp)

    def build(self, request: DresAnswerRequest) -> tuple[dict[str, Any], str]:
        """Return the exact submission body and a one-line human summary."""
        collection = self.settings.get("collection") or None
        if request.task == "trake":
            frames = [self.resolve(item.model_dump()) for item in request.frames]
            answer = trake_answer(frames, self.formats)
            summary = f"TRAKE {frames[0].video_id}: frames {', '.join(str(f.source_frame_idx) for f in frames)}"
        else:
            if request.frame is None:
                raise ValueError("choose a frame to submit")
            frame = self.resolve(request.frame.model_dump())
            if request.task == "qa":
                from src.contracts import QACompetitionSpec

                text = clean_answer(request.answer or "")
                spec_id = self.settings.get("qa_answer_spec")
                if spec_id:
                    text = QACompetitionSpec.for_spec_id(str(spec_id)).normalize_answer(text)
                answer = qa_answer(frame, text, self.formats)
                summary = f"Q&A “{text}” at {frame.video_id} {_clock(frame.time_ms)} ({frame.time_ms} ms)"
            else:
                answer = kis_answer(frame, self.formats, collection)
                summary = f"KIS {frame.video_id} at {_clock(frame.time_ms)} ({frame.time_ms} ms)"
        return submission_body([answer]), summary

    def submit(self, request: DresAnswerRequest) -> dict[str, Any]:
        session = self._require()
        evaluation_id = request.evaluation_id or session.evaluation_id
        if not evaluation_id:
            raise ValueError("choose an evaluation first")
        body, summary = self.build(request)
        started = time.time()
        error: DresError | None = None
        try:
            result = session.client.submit(session.session_id, evaluation_id, body)
        except DresError as caught:
            error = caught
            result = {"status": False, "description": caught.description, "http_status": caught.status}
        record = {"time": round(started, 3), "evaluation_id": evaluation_id, "username": session.username,
                  "task": request.task, "summary": summary, "body": body, "result": result}
        self._append_log(record)
        if error is not None and error.status in {0, 401, 404}:
            # Transport/session problems are operator-actionable errors, not verdicts.
            raise error
        return {"accepted": bool(result.get("status")), "verdict": result.get("submission"),
                "description": result.get("description", ""), "http_status": result.get("http_status"),
                "summary": summary, "body": body}

    def log_results(self, query: str, frames: Sequence[Mapping[str, Any]]) -> bool:
        """Send a VBS-style result log for the current evaluation when enabled."""
        if not self.settings.get("log_results", False):
            return False
        session = self._require()
        if not session.evaluation_id:
            return False
        now = int(time.time() * 1000)
        ranked = []
        for rank, item in enumerate(frames[:100], 1):
            try:
                frame = self.resolve(item)
            except (ValueError, KeyError, TypeError):
                continue
            ranked.append({"answer": {"mediaItemName": frame.video_id, "start": frame.time_ms,
                                      "end": frame.time_ms}, "rank": rank})
        log = {"timestamp": now, "sortType": "score", "resultSetAvailability": "top",
               "results": ranked,
               "events": [{"timestamp": now, "category": "TEXT", "type": "query", "value": query[:1000]}]}
        session.client.log_results(session.session_id, session.evaluation_id, log)
        return True

    def _append_log(self, record: Mapping[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Logging must never block a competition submission.


def _secret(env_name: Any) -> str:
    import os

    from dotenv import dotenv_values

    if not isinstance(env_name, str) or not env_name:
        return ""
    value = os.environ.get(env_name)
    if not value:
        path = PROJECT_ROOT / ".env"
        value = dotenv_values(path).get(env_name) if path.is_file() else None
    return str(value or "").strip()


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def _evaluation_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"id": item.get("id"), "name": item.get("name"), "status": item.get("status"),
            "type": item.get("type"),
            "tasks": [t.get("name") for t in item.get("taskTemplates", []) if isinstance(t, dict)][:200]}


class FrameReference(BaseModel):
    model_config = ConfigDict(extra="ignore")
    batch_id: str = Field(min_length=1, max_length=128)
    video_id: str = Field(min_length=1, max_length=128)
    source_frame_idx: int | None = Field(default=None, ge=0, strict=True)
    keyframe_ordinal: int | None = Field(default=None, ge=1, strict=True)

    @field_validator("batch_id", "video_id")
    @classmethod
    def identifier(cls, value: str) -> str:
        if not _VIDEO.fullmatch(value):
            raise ValueError("invalid identifier")
        return value


class DresAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Task = "kis"
    frame: FrameReference | None = None
    frames: list[FrameReference] = Field(default_factory=list, max_length=20)
    answer: str | None = Field(default=None, max_length=1000)
    evaluation_id: str | None = Field(default=None, max_length=128)


class DresLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_url: str = Field(default="", max_length=512)
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=1024)
    session_id: str = Field(default="", max_length=256)


class DresSelectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluation_id: str = Field(min_length=1, max_length=128)


class DresLogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8192)
    frames: list[FrameReference] = Field(default_factory=list, max_length=100)


def _http(error: Exception) -> HTTPException:
    if isinstance(error, PermissionError):
        return HTTPException(status_code=409, detail={"code": "dres_not_connected", "message": "Connect to DRES first."})
    if isinstance(error, DresError):
        status = 401 if error.status == 401 else 502
        return HTTPException(status_code=status, detail={"code": "dres_error", "message": error.description,
                                                          "dres_status": error.status})
    return HTTPException(status_code=422, detail={"code": "dres_request_invalid", "message": str(error)})


def install_dres_routes(app: Any, context: Any, *, transport: Transport | None = None) -> DresService:
    """Expose a same-origin DRES proxy so the browser never handles the session or CORS."""
    service = DresService(context.config, getattr(context, "artifact_repository", None), transport=transport)
    app.state.dres_service = service
    router = APIRouter(prefix="/v1/dres")

    def guarded(action: Callable[[], Any]) -> Any:
        try:
            return action()
        except (PermissionError, DresError, ValueError, KeyError, TypeError) as error:
            raise _http(error) from None

    @router.get("/state")
    def state() -> dict[str, Any]:
        return service.state()

    @router.post("/login")
    def login(request: DresLoginRequest) -> dict[str, Any]:
        return guarded(lambda: service.login(request.server_url, request.username, request.password,
                                             request.session_id))

    @router.post("/logout")
    def logout() -> dict[str, Any]:
        return service.logout()

    @router.post("/refresh")
    def refresh() -> dict[str, Any]:
        return guarded(service.refresh)

    @router.post("/evaluation")
    def select(request: DresSelectRequest) -> dict[str, Any]:
        return guarded(lambda: service.select(request.evaluation_id))

    @router.get("/task")
    def task() -> dict[str, Any]:
        return {"task": guarded(service.current_task)}

    @router.post("/preview")
    def preview(request: DresAnswerRequest) -> dict[str, Any]:
        body, summary = guarded(lambda: service.build(request))
        return {"body": body, "summary": summary}

    @router.post("/submit")
    def submit(request: DresAnswerRequest) -> dict[str, Any]:
        return guarded(lambda: service.submit(request))

    @router.post("/log")
    def log(request: DresLogRequest) -> dict[str, Any]:
        sent = guarded(lambda: service.log_results(request.query, [f.model_dump() for f in request.frames]))
        return {"sent": sent}

    app.include_router(router)
    return service
