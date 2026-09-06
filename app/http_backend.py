"""Thin Streamlit-side client for the sole FastAPI runtime."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import requests

_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class BackendError(RuntimeError):
    """Sanitized backend failure safe for UI logging."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.detail = dict(detail or {})
        self.search_id = self.detail.get("search_id")
        self.recoverable = bool(self.detail.get("recoverable", False))
        self.snapshot: dict[str, Any] | None = None


def _safe_error(response: Any) -> BackendError:
    code = "backend_request_failed"
    safe_detail: dict[str, Any] = {}
    try:
        body = response.json()
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, dict) and isinstance(detail.get("code"), str):
            code = detail["code"]
            for name in (
                "code",
                "search_id",
                "candidate_id",
                "recoverable",
                "status",
                "reasoned_clusters",
                "answerable_clusters",
                "unanswerable_clusters",
                "failed_clusters",
                "pending_clusters",
                "target_clusters",
                "requested_target_clusters",
                "retrieved_clusters",
                "retrieval_exhausted",
                "correlation_id",
            ):
                value = detail.get(name)
                if isinstance(value, (bool, int, float, str)) or value is None:
                    safe_detail[name] = value
            issues = detail.get("issues")
            if isinstance(issues, list):
                safe_issues: list[dict[str, Any]] = []
                for issue in issues[:32]:
                    if not isinstance(issue, dict):
                        continue
                    sanitized: dict[str, Any] = {}
                    for name in (
                        "filename",
                        "task",
                        "code",
                        "message",
                        "suggestion",
                        "row",
                        "line",
                    ):
                        value = issue.get(name)
                        if isinstance(value, str) and len(value) <= 256:
                            sanitized[name] = value
                        elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            sanitized[name] = value
                    if sanitized:
                        safe_issues.append(sanitized)
                safe_detail["issues"] = safe_issues
        elif isinstance(detail, str) and _SAFE_ERROR_CODE.fullmatch(detail):
            code = detail
    except Exception:
        pass
    return BackendError(
        code,
        status_code=int(getattr(response, "status_code", 0)) or None,
        detail=safe_detail,
    )


class _Client:
    def __init__(self, base_url: str, http: Any | None = None) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.http = http or requests.Session()

    def json(
        self,
        method: str,
        path: str,
        *,
        payload: Any = None,
        params: Any = None,
        timeout_seconds: float = 180,
    ) -> Any:
        try:
            response = self.http.request(
                method,
                f"{self.base_url}{path}",
                json=payload,
                params=params,
                timeout=(3, float(timeout_seconds)),
            )
        except requests.RequestException:
            raise BackendError("backend_unreachable") from None
        if not 200 <= int(response.status_code) < 300:
            raise _safe_error(response)
        try:
            return response.json()
        except Exception:
            raise BackendError("backend_response_invalid") from None

    def bytes(self, path: str) -> bytes:
        try:
            response = self.http.get(f"{self.base_url}{path}", timeout=30)
        except requests.RequestException:
            raise BackendError("backend_unreachable") from None
        if not 200 <= int(response.status_code) < 300:
            raise _safe_error(response)
        return bytes(response.content)


class _Pipeline:
    def __init__(self, client: _Client) -> None:
        self.client = client
        self.last_channel_statuses: dict[str, Any] = {}
        self.last_query_processing: dict[str, Any] = {}

    def search(self, request: Any, *, limit: int, refine_raw: bool = False) -> list[dict[str, Any]]:
        self.last_channel_statuses = {}
        self.last_query_processing = {}
        payload = request.to_dict()
        payload["limit"] = limit
        payload["refine_raw"] = bool(refine_raw)
        value = self.client.json("POST", "/v1/kis/search", payload=payload, timeout_seconds=30)
        self.last_channel_statuses = dict(value.get("channel_statuses", {}))
        self.last_query_processing = dict(value.get("query_processing", {}))
        return [dict(item) for item in value.get("hits", ())]

    def refine(
        self,
        request: Any,
        *,
        positive_ids: tuple[tuple[str, int], ...],
        negative_ids: tuple[tuple[str, int], ...],
        limit: int,
    ) -> list[dict[str, Any]]:
        payload = request.to_dict()
        payload["positive_feedback_ids"] = [
            f"{video_id}:{source_frame_idx}" for video_id, source_frame_idx in positive_ids
        ]
        payload["negative_feedback_ids"] = [
            f"{video_id}:{source_frame_idx}" for video_id, source_frame_idx in negative_ids
        ]
        payload["limit"] = limit
        value = self.client.json("POST", "/v1/kis/refine", payload=payload)
        self.last_channel_statuses = dict(value.get("channel_statuses", {}))
        self.last_query_processing = dict(value.get("query_processing", {}))
        return [dict(item) for item in value.get("hits", ())]


class _QAPipeline:
    def __init__(self, client: _Client) -> None:
        self.client = client
        self.aliases: dict[str, str] = {}
        self.last_snapshot: dict[str, Any] | None = None

    @staticmethod
    def _normalize(value: dict[str, Any], request: Any | None = None) -> dict[str, Any]:
        result = dict(value)
        search_id = str(result.get("search_id") or result.get("session_id") or "")
        result["session_id"] = search_id
        if request is not None:
            result["request"] = request.to_dict()
        result.setdefault(
            "state",
            {
                "predictions": list(result.get("predictions", ())),
                "window_outcomes": list(result.get("window_outcomes", ())),
            },
        )
        return result

    def _actual(self, search_id: str) -> str:
        return self.aliases.get(search_id, search_id)

    def search(self, request: Any, *, session_id: str | None = None) -> list[dict[str, Any]]:
        """Deprecated compatibility wrapper for retrieval-only QA search."""

        self.retrieve(request, session_id=session_id)
        return []

    def retrieve(self, request: Any, *, session_id: str | None = None) -> list[dict[str, Any]]:
        try:
            value = self.client.json(
                "POST",
                "/v1/qa/retrieve",
                payload=request.to_dict(),
                timeout_seconds=30,
            )
        except BackendError as error:
            actual = error.search_id if isinstance(error.search_id, str) else None
            if actual and error.recoverable:
                if session_id:
                    self.aliases[session_id] = actual
                try:
                    error.snapshot = self.get_search(actual)
                    self.last_snapshot = error.snapshot
                except BackendError:
                    error.snapshot = None
            raise
        normalized = self._normalize(value, request)
        self.last_snapshot = normalized
        if session_id:
            self.aliases[session_id] = normalized["session_id"]
        return list(normalized.get("candidates", ()))

    def inject(self, request: Any, *, session_id: str | None = None) -> list[dict[str, Any]]:
        """Create an authoritative one-frame QA session without running retrieval."""

        value = self.client.json(
            "POST",
            "/v1/qa/inject",
            payload=request.to_dict(),
            timeout_seconds=30,
        )
        normalized = self._normalize(value, request)
        self.last_snapshot = normalized
        if session_id:
            self.aliases[session_id] = normalized["session_id"]
        return list(normalized.get("candidates", ()))

    def answer_candidate(self, search_id: str, request: Any) -> list[dict[str, Any]]:
        actual = self._actual(search_id)
        try:
            value = self.client.json(
                "POST",
                f"/v1/qa/search/{actual}/answer",
                payload=request.to_dict(),
                timeout_seconds=75,
            )
        except BackendError as error:
            if error.recoverable:
                try:
                    error.snapshot = self.get_search(actual)
                    self.last_snapshot = error.snapshot
                except BackendError:
                    error.snapshot = None
            raise
        normalized = self._normalize(value)
        self.last_snapshot = normalized
        return list(normalized.get("predictions", ()))

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        actual = self._actual(search_id)
        try:
            value = self.client.json("GET", f"/v1/qa/search/{actual}")
        except BackendError as error:
            if str(error) == "Visual Q&A search is missing or expired":
                return None
            raise
        normalized = self._normalize(value)
        self.last_snapshot = normalized
        self.aliases[search_id] = normalized["session_id"]
        return normalized

    def expand(self, search_id: str, target: int) -> list[dict[str, Any]]:
        del search_id, target
        raise BackendError("qa_expand_removed", status_code=410)


class _QASessionStore:
    def __init__(self, client: _Client, pipeline: _QAPipeline) -> None:
        self.client = client
        self.pipeline = pipeline

    def initialize(self) -> None:
        return None

    def record_operator_edit(
        self, search_id: str, window_id: str, edit: dict[str, Any]
    ) -> dict[str, Any]:
        actual = self.pipeline._actual(search_id)
        value = self.client.json(
            "POST",
            f"/v1/qa/search/{actual}/edits",
            payload={"window_id": window_id, "answer": edit.get("edited_raw_answer")},
        )
        edits = value.get("operator_edits", ())
        return dict(edits[-1]) if isinstance(edits, list) and edits else dict(edit)


@dataclass(frozen=True)
class _Frame:
    batch_id: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int = 0
    shot_id: str | None = None


class _Repository:
    def __init__(self, client: _Client) -> None:
        self.client = client
        self._validation: dict[str, Any] = {}

    def initialize(self) -> None:
        return None

    def _validate(self, batch_id: str, identities: Any) -> dict[str, Any]:
        values = [
            {"video_id": str(video), "source_frame_idx": int(frame)} for video, frame in identities
        ]
        result = self.client.json(
            "POST",
            "/v1/frames/validate",
            payload={"batch_id": batch_id, "identities": values},
        )
        self._validation[batch_id] = result
        return result

    def existing_source_frames(
        self, batch_id: str, identities: Any, *, include_refined: bool = True
    ) -> set[tuple[str, int]]:
        del include_refined
        value = self._validate(batch_id, identities)
        return {
            (str(item["video_id"]), int(item["source_frame_idx"]))
            for item in value.get("existing", ())
        }

    def frame_index_mappings_for_sources(self, batch_id: str, identities: Any) -> dict[str, Any]:
        value = self._validate(batch_id, identities)
        return dict(value.get("ordinal_mappings", {}))

    def frame_index_mappings(self, batch_id: str) -> dict[str, Any]:
        return dict(self._validation.get(batch_id, {}).get("ordinal_mappings", {}))

    def get_source_frame(
        self, batch_id: str, video_id: str, source_frame_idx: int
    ) -> _Frame | None:
        value = self._validate(batch_id, ((video_id, source_frame_idx),))
        existing = value.get("existing", ())
        if not existing:
            return None
        timestamp = int(value.get("timestamps_ms", {}).get(f"{video_id}:{source_frame_idx}", 0))
        return _Frame(batch_id, video_id, source_frame_idx, timestamp)

    get_frame = get_source_frame

    def neighbors(
        self,
        batch_id: str,
        video_id: str,
        timestamp_ms: int,
        *,
        radius_ms: int,
        limit: int,
    ) -> tuple[_Frame, ...]:
        value = self.client.json(
            "GET",
            f"/v1/frames/{batch_id}/{video_id}/neighbors",
            params={"timestamp_ms": timestamp_ms, "radius_ms": radius_ms, "limit": limit},
        )
        return tuple(
            _Frame(
                str(item.get("batch_id", batch_id)),
                str(item.get("video_id", video_id)),
                int(item["source_frame_idx"]),
                int(item.get("timestamp_ms", 0)),
                item.get("shot_id"),
            )
            for item in value.get("neighbors", ())
        )

    def adjacent_frames(
        self,
        batch_id: str,
        video_id: str,
        source_frame_idx: int,
        *,
        radius: int = 5,
    ) -> tuple[tuple[int, _Frame], ...]:
        value = self.client.json(
            "GET",
            f"/v1/frames/{batch_id}/{video_id}/{source_frame_idx}/adjacent",
            params={"radius": radius},
        )
        return tuple(
            (
                int(item["offset"]),
                _Frame(
                    str(item.get("batch_id", batch_id)),
                    str(item.get("video_id", video_id)),
                    int(item["source_frame_idx"]),
                    int(item.get("timestamp_ms", 0)),
                    item.get("shot_id"),
                ),
            )
            for item in value.get("frames", ())
        )



class _ImageResolver:
    def __init__(self, client: _Client) -> None:
        self.client = client

    def resolve(self, frame: _Frame) -> Any:
        data = self.client.bytes(
            f"/v1/frames/{frame.batch_id}/{frame.video_id}/{frame.source_frame_idx}/image"
        )
        return SimpleNamespace(data=data, mime_type="image/webp")


class _Ledger:
    def __init__(self, repository: _Repository) -> None:
        self.repository = repository

    def initialize(self) -> None:
        return None

    def is_batch_active(self, batch_id: str) -> bool:
        value = self.repository._validation.get(batch_id)
        return bool(value and value.get("batch_active"))


class _Report:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return dict(self.value)


class _Readiness:
    def __init__(self, client: _Client) -> None:
        self.client = client

    def check(self, batch_id: str, *, mode: str = "app") -> _Report:
        return _Report(
            self.client.json(
                "GET",
                "/v1/readiness",
                params={"batch_id": batch_id, "mode": mode},
                timeout_seconds=3,
            )
        )

    def workflow_readiness(self, report: _Report, workflow: str) -> _Report:
        selected = report.value.get(workflow)
        if isinstance(selected, dict):
            return _Report(dict(selected))
        warmup = report.value.get("warmup")
        workflows = warmup.get("workflows") if isinstance(warmup, dict) else None
        selected = workflows.get(workflow) if isinstance(workflows, dict) else None
        return _Report(dict(selected) if isinstance(selected, dict) else {})


class _Warmup:
    def __init__(self, client: _Client) -> None:
        self.client = client

    def start(self) -> bool:
        value = self.client.json("POST", "/v1/runtime/warmup/start", timeout_seconds=3)
        return str(value.get("state", "idle")) == "warming"

    def snapshot(self) -> dict[str, Any]:
        return dict(self.client.json("GET", "/v1/runtime/warmup", timeout_seconds=3))

    def readiness_report(self) -> None:
        return None


class _Trake:
    def __init__(self, client: _Client) -> None:
        self.client = client
        self.last_query_processing: dict[str, Any] = {}

    def search(self, request: Any, *, default_batch_id: str | None = None) -> dict[str, Any]:
        del default_batch_id
        payload = request.model_dump(mode="json") if hasattr(request, "model_dump") else request
        response = dict(self.client.json("POST", "/v1/trake/search", payload=payload))
        self.last_query_processing = dict(response.get("query_processing", {}))
        return response


class HttpRuntimeContext:
    """Runtime-shaped HTTP adapter used only by the Streamlit process."""

    is_remote = True

    def __init__(self, base_url: str, *, http: Any | None = None) -> None:
        client = _Client(base_url, http)
        capabilities = client.json("GET", "/v1/runtime/capabilities", timeout_seconds=3)
        self.client = client
        self.capabilities = capabilities
        self.selected_batch_id = str(capabilities["default_batch_id"])
        self.config = {
            "profile": str(capabilities.get("profile", "laptop")),
            "runtime": {"default_batch_id": self.selected_batch_id},
            "readiness": {"batches": {self.selected_batch_id: {}}},
            "retrieval": {"visual_qa": {}},
            "competition": {"output_dir": "outputs/submissions"},
            "logging": {
                "app_error_file": "outputs/logs/app_error.log",
                "rotation": "10 MB",
            },
        }
        self.launch_identity = {
            "canonical": True,
            "contract": "fastapi-thin-client-v1",
            "batch_id": self.selected_batch_id,
            "profile": self.config["profile"],
            "port": 8504,
            "launch_id": "api-backed",
        }
        self.pipeline = _Pipeline(client)
        self.qa_pipeline = _QAPipeline(client)
        self.artifact_repository = _Repository(client)
        self.qa_image_resolver = _ImageResolver(client)
        self.qa_session_store = _QASessionStore(client, self.qa_pipeline)
        self.ledger = _Ledger(self.artifact_repository)
        self.readiness_service = _Readiness(client)
        self.warmup = _Warmup(client)
        self.trake_pipeline = _Trake(client)

    def refresh_capabilities(self) -> dict[str, Any]:
        """Refresh provider status without constructing any local model runtime."""

        self.capabilities = dict(
            self.client.json("GET", "/v1/runtime/capabilities", timeout_seconds=3)
        )
        return self.capabilities

    def validate_batch_filters(self, values: Any = None) -> tuple[str, ...]:
        selected = tuple(str(value).strip() for value in (values or ()) if str(value).strip())
        return selected or (self.selected_batch_id,)

    def diagnostic_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(
            self.client.json(
                "POST",
                "/v1/qa/diagnostics/image",
                payload=payload,
                timeout_seconds=75,
            )
        )

    def create_submission_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a validated server-side bundle and fetch its immutable bytes."""

        metadata = dict(
            self.client.json(
                "POST",
                "/v1/submissions/bundles",
                payload=payload,
                timeout_seconds=30,
            )
        )
        download_url = metadata.get("download_url")
        if not isinstance(download_url, str) or not download_url.startswith(
            "/v1/submissions/bundles/"
        ):
            raise BackendError("backend_response_invalid")
        data = self.client.bytes(download_url)
        digest = metadata.get("sha256")
        byte_count = metadata.get("byte_count")
        if (
            not isinstance(digest, str)
            or hashlib.sha256(data).hexdigest() != digest
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count != len(data)
        ):
            raise BackendError("bundle_download_integrity_failed")
        metadata["data"] = data
        return metadata

    def load_submission_staging(self) -> dict[str, Any]:
        """Load the machine-global staged CSV snapshot from FastAPI."""

        return dict(
            self.client.json(
                "GET",
                "/v1/submissions/staging",
                timeout_seconds=5,
            )
        )

    def stage_submission_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist one staged CSV through the server trust boundary."""

        return dict(
            self.client.json(
                "POST",
                "/v1/submissions/staging/entries",
                payload=payload,
                timeout_seconds=30,
            )
        )

    def delete_submission_entry(
        self,
        filename: str,
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Delete one exact staged filename under optimistic concurrency."""

        safe_filename = quote(str(filename), safe="")
        return dict(
            self.client.json(
                "DELETE",
                f"/v1/submissions/staging/entries/{safe_filename}",
                params={"expected_revision": expected_revision},
                timeout_seconds=10,
            )
        )

    def mutate_submission_rows(
        self,
        filename: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Append, replace, delete, or move one staged QA/TRAKE rank."""

        safe_filename = quote(str(filename), safe="")
        return dict(
            self.client.json(
                "POST",
                f"/v1/submissions/staging/entries/{safe_filename}/rows",
                payload=payload,
                timeout_seconds=30,
            )
        )

    def clear_submission_staging(self, *, expected_revision: int) -> dict[str, Any]:
        """Publish an empty staging revision without deleting prior bundles."""

        return dict(
            self.client.json(
                "DELETE",
                "/v1/submissions/staging",
                params={"expected_revision": expected_revision},
                timeout_seconds=10,
            )
        )


def build_http_runtime(base_url: str, *, http: Any | None = None) -> HttpRuntimeContext:
    return HttpRuntimeContext(base_url, http=http)


__all__ = ["BackendError", "HttpRuntimeContext", "build_http_runtime"]
