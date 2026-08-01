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

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

from src import PROJECT_ROOT


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


# Importing this module must not create log directories or files. Applications
# that want project file logging call ``setup_logger_from_config`` at startup.
logger = get_logger()
