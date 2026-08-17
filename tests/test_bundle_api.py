from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from src.api import create_app


class _Ledger:
    def is_batch_active(self, batch_id: str) -> bool:
        return batch_id == "batch"


class _Repository:
    timestamps = {
        ("L21_V001", 10): 100,
        ("L21_V001", 20): 200,
    }

    def get_source_frame(self, batch_id: str, video_id: str, frame_idx: int):
        timestamp = self.timestamps.get((video_id, frame_idx)) if batch_id == "batch" else None
        return None if timestamp is None else SimpleNamespace(timestamp_ms=timestamp)

    get_frame = get_source_frame


def _context(tmp_path: Path):
    return SimpleNamespace(
        config={"competition": {"output_dir": str(tmp_path)}},
        ledger=_Ledger(),
        artifact_repository=_Repository(),
        pipeline=SimpleNamespace(),
    )


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def _entry(
    task: str,
    query_id: str,
    data: bytes,
    *,
    stage_count=None,
    source_ranks=None,
) -> dict:
    value = {
        "task": task,
        "query_id": query_id,
        "csv_base64": base64.b64encode(data).decode("ascii"),
        "source_fingerprint": hashlib.sha256(data).hexdigest(),
        "stage_count": stage_count,
    }
    if source_ranks is not None:
        value["source_ranks"] = source_ranks
    return value


def test_bundle_api_builds_metadata_and_download(tmp_path: Path) -> None:
    app = create_app(_context(tmp_path))
    create = _endpoint(app, "/v1/submissions/bundles", "POST")
    metadata = create(
        {
            "mode": "all",
            "batch_id": "batch",
            "entries": [
                _entry("kis", "1", b"L21_V001,10\n"),
                _entry("qa", "2", b"L21_V001,20,yes\n"),
                _entry("trake", "3", b"L21_V001,10,20\n", stage_count=2),
            ],
        }
    )
    assert metadata["status"] == "valid"
    assert metadata["validation_status"] == "valid"
    assert metadata["file_count"] == 3
    assert metadata["download_url"].endswith("/download")

    download = _endpoint(app, "/v1/submissions/bundles/{run_id}/download", "GET")
    response = download(metadata["run_id"])
    assert isinstance(response, FileResponse)
    with zipfile.ZipFile(response.path) as archive:
        assert archive.namelist()[0] == "submission/"

    inspect = _endpoint(app, "/v1/submissions/bundles/{run_id}", "GET")
    assert inspect(metadata["run_id"])["sha256"] == metadata["sha256"]


def test_bundle_api_returns_sanitized_validation_issues(tmp_path: Path) -> None:
    create = _endpoint(create_app(_context(tmp_path)), "/v1/submissions/bundles", "POST")
    with pytest.raises(HTTPException) as caught:
        create(
            {
                "mode": "current",
                "batch_id": "batch",
                "entries": [_entry("kis", "1", b"L21_V001.mp4,10\n")],
            }
        )
    assert caught.value.status_code == 422
    assert caught.value.detail["code"] == "bundle_validation_failed"
    issue = caught.value.detail["issues"][0]
    assert issue["code"] == "submission_video_extension_forbidden"
    assert issue["row"] == 1
    assert "path" not in issue
    assert not list(tmp_path.glob("run_*"))


def test_bundle_download_rejects_path_like_run_id(tmp_path: Path) -> None:
    download = _endpoint(
        create_app(_context(tmp_path)),
        "/v1/submissions/bundles/{run_id}/download",
        "GET",
    )
    with pytest.raises(HTTPException) as caught:
        download("../manifest.json")
    assert caught.value.status_code == 404
    assert caught.value.detail == {"code": "bundle_not_found"}


def test_bundle_api_rejects_kis_source_rank_gap(tmp_path: Path) -> None:
    create = _endpoint(create_app(_context(tmp_path)), "/v1/submissions/bundles", "POST")
    with pytest.raises(HTTPException) as caught:
        create(
            {
                "mode": "current",
                "batch_id": "batch",
                "entries": [
                    _entry(
                        "kis",
                        "9",
                        b"L21_V001,10\nL21_V001,20\n",
                        source_ranks=[1, 3],
                    )
                ],
            }
        )
    assert caught.value.status_code == 422
    assert caught.value.detail["issues"][0]["code"] == "submission_rank_sequence_invalid"
