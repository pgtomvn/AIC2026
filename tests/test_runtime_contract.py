from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import app.streamlit_app as streamlit_app
from src.runtime_contract import (
    RUNTIME_CONTRACT_FEATURES,
    RUNTIME_CONTRACT_SURFACE,
    RUNTIME_CONTRACT_VERSION,
    source_contract_fingerprint,
)


def _remote_context(contract: dict[str, object] | None) -> SimpleNamespace:
    return SimpleNamespace(
        is_remote=True,
        capabilities={"runtime_contract": contract} if contract is not None else {},
    )


def test_runtime_contract_fingerprint_changes_with_surface(tmp_path: Path) -> None:
    for relative in RUNTIME_CONTRACT_SURFACE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"surface:{relative}\n", encoding="utf-8")

    before = source_contract_fingerprint(tmp_path)
    assert before == source_contract_fingerprint(tmp_path)

    changed = tmp_path / RUNTIME_CONTRACT_SURFACE[0]
    changed.write_text("changed\n", encoding="utf-8")
    assert source_contract_fingerprint(tmp_path) != before


def test_runtime_contract_guard_accepts_fresh_local_and_remote_contracts(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AIC_LAUNCH_CONTRACT", raising=False)
    monkeypatch.delenv("AIC_SOURCE_CONTRACT_SHA256", raising=False)
    assert streamlit_app._runtime_contract_issues(SimpleNamespace(is_remote=False)) == ()

    context = _remote_context(
        {
            "version": RUNTIME_CONTRACT_VERSION,
            "features": list(RUNTIME_CONTRACT_FEATURES),
            "source_contract_sha256": source_contract_fingerprint(),
        }
    )
    assert streamlit_app._runtime_contract_issues(context) == ()


def test_runtime_contract_guard_detects_stale_request_class(monkeypatch) -> None:
    @dataclass(frozen=True)
    class LegacyQACandidateRequest:
        query_id: str
        event_description: str

    monkeypatch.setattr(streamlit_app, "QACandidateRequest", LegacyQACandidateRequest)
    issues = streamlit_app._runtime_contract_issues(SimpleNamespace(is_remote=False))
    assert "request_field_missing:LegacyQACandidateRequest.entity_constraints" in issues


def test_runtime_contract_guard_detects_api_and_source_drift(monkeypatch) -> None:
    launched = "a" * 64
    monkeypatch.setenv("AIC_LAUNCH_CONTRACT", "run_app-v2")
    monkeypatch.setenv("AIC_SOURCE_CONTRACT_SHA256", launched)
    monkeypatch.setattr(streamlit_app, "source_contract_fingerprint", lambda: "b" * 64)
    context = _remote_context(
        {
            "version": RUNTIME_CONTRACT_VERSION + 1,
            "features": [],
            "source_contract_sha256": "c" * 64,
        }
    )

    issues = streamlit_app._runtime_contract_issues(context)

    assert "api_contract_version_mismatch" in issues
    assert "api_contract_feature_missing" in issues
    assert "source_contract_drift" in issues
    assert "api_source_contract_mismatch" in issues


def test_correlation_reproducer_redacts_payload_and_absolute_paths(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    correlation_id = "ERR-QA-47022AFB"
    frame_path = project_root / "app" / "streamlit_app.py"
    event = {
        "text": (
            "Traceback (most recent call last):\n"
            f'  File "{frame_path}", line 2653, in _render_qa\n'
            "TypeError: QACandidateRequest.__init__() got an unexpected keyword "
            "argument 'entity_constraints'\n"
        ),
        "record": {
            "extra": {
                "event_id": correlation_id,
                "workflow": "qa",
                "stage": "render",
                "batch_id": "aic25-b1-reference",
                "query_text": "SECRET QUERY MUST NOT APPEAR",
            },
            "exception": {
                "type": "TypeError",
                "value": (
                    "QACandidateRequest.__init__() got an unexpected keyword argument "
                    "'entity_constraints'"
                ),
            },
        },
    }
    log_path = tmp_path / "app_error.log"
    log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "debug_qa_correlation_error.py"),
            "--correlation-id",
            correlation_id,
            "--log",
            str(log_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["event"]["payload_available"] is False
    assert report["event"]["project_frames"] == [
        {"file": "app/streamlit_app.py", "line": 2653, "function": "_render_qa"}
    ]
    assert "SECRET QUERY" not in completed.stdout
    assert str(project_root) not in completed.stdout
    assert report["offline_component_probes"]["provider_called"] is False
