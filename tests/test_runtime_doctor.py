"""Offline tests for bounded runtime-doctor waiting and launcher contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.runtime_doctor as doctor_module
from src.readiness import WorkflowState


class _Report:
    def __init__(
        self, kis_state: WorkflowState, qa_state: WorkflowState | None = None
    ) -> None:
        self.kis = SimpleNamespace(state=kis_state)
        self.qa = SimpleNamespace(state=qa_state or kis_state)
        self.checks = ()
        self.state = kis_state

    def to_dict(self) -> dict:
        return {
            "kis": {"state": self.kis.state},
            "qa": {"state": self.qa.state},
            "checks": [],
        }


class _Service:
    def __init__(self, reports: list[_Report]) -> None:
        self.reports = reports
        self.calls = 0

    def check(self, *_args, **_kwargs) -> _Report:
        report = self.reports[min(self.calls, len(self.reports) - 1)]
        self.calls += 1
        return report


def _configure(monkeypatch, tmp_path: Path, service: _Service) -> None:
    monkeypatch.setattr(
        doctor_module,
        "load_config",
        lambda: {
            "profile": "laptop",
            "runtime": {"default_batch_id": "b1"},
            "data": {"catalog_db": str(tmp_path / "missing.sqlite3")},
        },
    )
    monkeypatch.setattr(doctor_module, "validate_runtime_batch", lambda *_args: "b1")
    monkeypatch.setattr(doctor_module, "build_readiness_service", lambda _config: service)


def test_doctor_waits_until_local_readiness_recovers(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    service = _Service([_Report(WorkflowState.BLOCKED), _Report(WorkflowState.READY)])
    _configure(monkeypatch, tmp_path, service)
    times = iter((0.0, 0.0, 0.1))
    monkeypatch.setattr(doctor_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(doctor_module.time, "sleep", lambda _seconds: None)

    result = doctor_module.main(
        [
            "--batch-id",
            "b1",
            "--wait-seconds",
            "2",
            "--poll-seconds",
            "0.1",
            "--compact",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert service.calls == 2
    assert payload["wait"] == {
        "requested_seconds": 2.0,
        "attempts": 2,
        "timed_out": False,
    }


def test_doctor_timeout_is_bounded_and_sanitized(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    service = _Service([_Report(WorkflowState.BLOCKED)])
    _configure(monkeypatch, tmp_path, service)
    times = iter((0.0, 1.0))
    monkeypatch.setattr(doctor_module.time, "monotonic", lambda: next(times))

    result = doctor_module.main(
        ["--batch-id", "b1", "--wait-seconds", "0.5", "--compact"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert service.calls == 1
    assert payload["wait"]["timed_out"] is True


def test_startup_only_validates_batch_without_running_full_readiness(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    service = _Service([_Report(WorkflowState.READY)])
    _configure(monkeypatch, tmp_path, service)

    result = doctor_module.main(
        ["--batch-id", "b1", "--startup-only", "--compact"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert service.calls == 0
    assert payload["status"] == "ready"
    assert payload["check_scope"] == "launcher_preflight"
    assert payload["readiness"] == "deferred_to_gated_warmup"


def test_doctor_allows_qa_block_only_with_explicit_degraded_opt_in(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    service = _Service([_Report(WorkflowState.READY, WorkflowState.BLOCKED)])
    _configure(monkeypatch, tmp_path, service)

    result = doctor_module.main(
        ["--batch-id", "b1", "--allow-degraded", "--compact"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["kis"]["state"] == WorkflowState.READY.value
    assert payload["qa"]["state"] == WorkflowState.BLOCKED.value


def test_launcher_contract_propagates_identity_and_never_kills_or_switches_ports() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_app.ps1").read_text(encoding="utf-8")

    for name in (
        "AIC_BATCH_ID",
        "AIC_LAUNCH_CONTRACT",
        "AIC_LAUNCH_ID",
        "AIC_LAUNCH_PORT",
        "AIC_LAUNCH_BATCH",
        "AIC_LAUNCH_PROFILE",
        "AIC_LAUNCHED_AT_UTC",
        "AIC_LAUNCH_LOCK_SHA256",
        "AIC_RUNTIME_STAMP_SHA256",
    ):
        assert f"$env:{name}" in source
    assert source.count("Assert-PortAvailable -RequestedPort") == 2
    assert '"--startup-only"' in source
    assert '"--compact"' in source
    assert '"--wait-seconds", "60"' in source
    assert "if ($StartServices)" in source
    assert "Write-OtherAicListeners" in source
    assert "Stop-Process" not in source
    assert '[string]$RuntimeProfile = "laptop"' not in source
    assert '[string]$BatchId = "aic25-b1-reference"' not in source
    assert '$PSBoundParameters.ContainsKey("RuntimeProfile")' in source
    assert '$PSBoundParameters.ContainsKey("BatchId")' in source
    assert 'Get-LocalEnvValue -Name "AIC_PROFILE"' in source
    assert 'Get-LocalEnvValue -Name "AIC_BATCH_ID"' in source
    assert 'Get-YamlRuntimeDefault -Name "profile"' in source
    assert 'Get-YamlRuntimeDefault -Name "batch"' in source


def test_streamlit_runtime_check_scans_logs_for_forbidden_diagnostics() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "check.ps1").read_text(encoding="utf-8")

    for marker in (
        "Traceback \\(most recent call last\\):",
        "Uncaught app execution",
        "use_container_width",
        "Api key is used with an insecure connection",
    ):
        assert marker in source
    assert "Competition Streamlit runtime logs contain no forbidden" in source


def test_live_gemini_readiness_uses_exactly_one_provider_call_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    doctor_source = (root / "scripts" / "runtime_doctor.py").read_text(encoding="utf-8")
    checker_source = (root / "scripts" / "check_gemini.py").read_text(encoding="utf-8")

    assert "check_five_image_once" in doctor_source
    assert "max_attempts=1" in checker_source
    function = checker_source.split("def check_five_image_once", 1)[1]
    assert function.count("reasoner.reason(") == 1
