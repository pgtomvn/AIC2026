"""
Logger
======
Structured logging setup using Loguru.
Provides a pre-configured logger instance for the entire project.

Usage:
    from src.utils.logger import logger
    logger.info("Processing started", video_id="L21_V001")
"""

from __future__ import annotations

import hashlib
import re
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger as _loguru_logger

from src import PROJECT_ROOT

_SAFE_EVENT_VALUE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_APP_ERROR_LOCK = threading.RLock()
_APP_ERROR_HANDLER_ID: int | None = None
_APP_ERROR_HANDLER_PATH: Path | None = None


def setup_logger(
    level: str = "INFO",
    log_file: str | None = "logs/aic2026.log",
    rotation: str = "10 MB",
) -> None:
    """Configure the global Loguru logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file. None = no file logging.
        rotation: Log rotation size/time.
    """
    # Remove default handlers
    _loguru_logger.remove()

    # Console handler (colorized)
    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler (structured)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_path),
            level=level,
            rotation=rotation,
            retention="30 days",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
        )


def get_logger():
    """Get the configured logger instance.

    Returns:
        Loguru logger instance.
    """
    return _loguru_logger


def setup_logger_from_config(config: dict[str, Any]) -> None:
    """Apply the logging section after an application explicitly starts."""
    logging_config = config.get("logging", {})
    log_file = logging_config.get("log_file", "logs/aic2026.log")
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        log_file = str(log_path)
    setup_logger(
        level=logging_config.get("level", "INFO"),
        log_file=log_file,
        rotation=logging_config.get("rotation", "10 MB"),
    )


def configure_app_error_logging(
    log_file: str | Path = "outputs/logs/app_error.log",
    *,
    rotation: str = "10 MB",
) -> Path:
    """Install one explicit application-error sink and return its absolute path.

    This function is deliberately opt-in: importing :mod:`src.utils.logger`
    never creates directories or files. Repeated Streamlit reruns reuse the
    same sink instead of duplicating log entries.
    """

    global _APP_ERROR_HANDLER_ID, _APP_ERROR_HANDLER_PATH
    path = Path(log_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    with _APP_ERROR_LOCK:
        if _APP_ERROR_HANDLER_ID is not None and _APP_ERROR_HANDLER_PATH == path:
            return path
        if _APP_ERROR_HANDLER_ID is not None:
            try:
                _loguru_logger.remove(_APP_ERROR_HANDLER_ID)
            except ValueError:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        _APP_ERROR_HANDLER_ID = _loguru_logger.add(
            str(path),
            level="ERROR",
            rotation=rotation,
            retention="30 days",
            enqueue=True,
            backtrace=True,
            diagnose=False,
            serialize=True,
        )
        _APP_ERROR_HANDLER_PATH = path
    return path


def _safe_event_value(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if _SAFE_EVENT_VALUE.fullmatch(normalized):
        return normalized
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def record_application_error(
    error: BaseException,
    *,
    workflow: str,
    stage: str,
    batch_id: object = None,
    query_or_session_id: object = None,
    latency_ms: float | None = None,
    context: Mapping[str, object] | None = None,
) -> str:
    """Log one server-side exception and return a safe correlation ID.

    Only allowlisted operational fields are attached. Query/session identifiers
    are always hashed, and logging failures are swallowed so the production
    error boundary can still display a sanitized message.
    """

    safe_workflow = (_safe_event_value(workflow) or "APP").upper()[:16]
    correlation_id = f"ERR-{safe_workflow}-{uuid4().hex[:8].upper()}"
    fields: dict[str, object] = {
        "event_id": correlation_id,
        "workflow": safe_workflow.casefold(),
        "stage": _safe_event_value(stage) or "unknown",
        "batch_id": _safe_event_value(batch_id),
    }
    if query_or_session_id is not None:
        fields["request_id_hash"] = hashlib.sha256(
            str(query_or_session_id).encode("utf-8")
        ).hexdigest()[:12]
    if isinstance(latency_ms, (int, float)):
        fields["latency_ms"] = round(float(latency_ms), 3)
    for name, value in (context or {}).items():
        if name in {"operation", "reason_code", "provider", "profile"}:
            fields[name] = _safe_event_value(value)
    try:
        _loguru_logger.bind(**fields).opt(
            exception=(type(error), error, error.__traceback__)
        ).error("application_error")
    except Exception:
        pass
    return correlation_id


# Importing this module must not create log directories or files. Applications
# that want project file logging call ``setup_logger_from_config`` at startup.
logger = get_logger()
