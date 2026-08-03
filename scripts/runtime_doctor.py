"""Report safe, read-only readiness for a selected competition batch."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.ledger import JobLedger, LedgerError
from src.utils.io_helpers import load_config, validate_batch_identifier


def _state_value(value: object) -> str:
    """Normalize enum-like workflow states without importing model backends."""

    raw = getattr(value, "value", value)
    return str(raw)


def _readiness_exit_code(report, *, allow_degraded: bool) -> int:
    if any(not check.usable for check in report.checks):
        return 1
    if _state_value(report.kis.state) == "blocked":
        return 1
    # The app remains useful for KIS while an optional QA provider is awaiting
    # operator configuration.  Starting in that state requires the launcher's
    # explicit degraded-mode opt-in; KIS/index failures always remain fatal.
    if _state_value(report.qa.state) == "blocked":
        return 0 if allow_degraded else 1
    states = {_state_value(report.kis.state), _state_value(report.qa.state)}
    if "degraded" in states and not allow_degraded:
        return 2
    return 0


def validate_runtime_batch(
    batch_id: str,
    config: dict,
    ledger: JobLedger,
) -> str:
    """Validate a configured or registered batch without importing model code."""

    batch_id = validate_batch_identifier(batch_id)
    readiness = config.get("readiness", {})
    batches = readiness.get("batches", {}) if isinstance(readiness, dict) else {}
    if isinstance(batches, dict) and batch_id in batches:
        return batch_id
    try:
        if ledger.is_batch_registered(batch_id):
            return batch_id
    except LedgerError as error:
        raise ValueError("registered batches could not be validated") from error
    raise ValueError("batch is not configured or registered")


def build_readiness_service(config: dict):
    """Lazily construct readiness so startup-only stays dependency-light."""

    from src.readiness import build_readiness_service as build

    return build(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--mode", choices=("app", "ingest", "full"), default="app")
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help=(
            "validate only the launch contract and selected batch; the application "
            "performs exact readiness auditing in its gated background warm-up"
        ),
    )
    parser.add_argument(
        "--live-gemini",
        action="store_true",
        help="explicitly send one tiny synthetic request to the configured Gemini provider",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="return success for a degraded runtime (the JSON still reports degraded)",
    )
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="bounded time to wait for local readiness (used after explicit service startup)",
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not 0 <= args.wait_seconds <= 300:
        parser.error("--wait-seconds must be between 0 and 300")
    if not 0.1 <= args.poll_seconds <= 10:
        parser.error("--poll-seconds must be between 0.1 and 10")
    if args.startup_only and args.live_gemini:
        parser.error("--startup-only cannot be combined with --live-gemini")

    config = load_config()
    catalog_value = config.get("data", {}).get("catalog_db")
    catalog_path = Path(str(catalog_value))
    if not catalog_path.is_absolute():
        catalog_path = PROJECT_ROOT / catalog_path
    try:
        validate_runtime_batch(args.batch_id, config, JobLedger(catalog_path))
    except (TypeError, ValueError):
        print(
            json.dumps(
                {
                    "profile": config.get("profile"),
                    "batch_id": args.batch_id,
                    "selected_batch_id": config.get("runtime", {}).get(
                        "default_batch_id"
                    ),
                    "status": "blocked",
                    "reason_code": "batch_not_configured_or_registered",
                },
                ensure_ascii=False,
            )
        )
        return 1
    if args.startup_only:
        print(
            json.dumps(
                {
                    "profile": config.get("profile"),
                    "batch_id": args.batch_id,
                    "selected_batch_id": config.get("runtime", {}).get(
                        "default_batch_id"
                    ),
                    "status": "ready",
                    "check_scope": "launcher_preflight",
                    "readiness": "deferred_to_gated_warmup",
                },
                ensure_ascii=False,
                separators=(",", ":") if args.compact else None,
                indent=None if args.compact else 2,
            )
        )
        return 0

    live_verified = False
    live_status = "not_requested"
    if args.live_gemini:
        # Keep the doctor output valid JSON and never include provider exception text.
        try:
            from scripts.check_gemini import check_five_image_once

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                live_verified = check_five_image_once() == 0
            live_status = "passed" if live_verified else "failed"
        except Exception:
            live_status = "failed"

    service = build_readiness_service(config)
    started = time.monotonic()
    deadline = started + args.wait_seconds
    attempts = 0
    while True:
        attempts += 1
        report = service.check(
            args.batch_id,
            mode=args.mode,
            gemini_live_verified=live_verified,
            force=True,
        )
        preliminary_exit = _readiness_exit_code(
            report, allow_degraded=args.allow_degraded
        )
        remaining = deadline - time.monotonic()
        if preliminary_exit == 0 or remaining <= 0:
            break
        time.sleep(min(args.poll_seconds, remaining))
    payload = report.to_dict()
    payload["live_gemini"] = live_status
    payload["wait"] = {
        "requested_seconds": args.wait_seconds,
        "attempts": attempts,
        "timed_out": bool(args.wait_seconds and preliminary_exit != 0),
    }
    warmup_failed = False
    if args.mode == "full" and all(check.usable for check in report.checks):
        try:
            from src.runtime import build_pipeline

            pipeline = build_pipeline(config)
            warmed: list[str] = []
            for channel in pipeline.dense_channels:
                try:
                    channel.encoder.encode_text("a person walking")
                    warmed.append(channel.name)
                finally:
                    unload = getattr(channel.encoder, "unload", None)
                    if callable(unload):
                        unload()
            payload["encoder_warmup"] = {"state": "ready", "channels": warmed}
        except Exception:
            warmup_failed = True
            payload["encoder_warmup"] = {
                "state": "blocked_asset",
                "reason_code": "encoder_warmup_failed",
                "remediation_id": "prepare_models",
            }
    elif args.mode == "full":
        payload["encoder_warmup"] = {
            "state": "not_run",
            "reason_code": "runtime_prerequisite_failed",
        }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            separators=(",", ":") if args.compact else None,
        )
    )

    if args.live_gemini and not live_verified:
        return 1
    if warmup_failed:
        return 1
    return _readiness_exit_code(report, allow_degraded=args.allow_degraded)


if __name__ == "__main__":
    raise SystemExit(main())
