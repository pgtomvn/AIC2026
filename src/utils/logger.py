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

from loguru import logger as _loguru_logger

from src.utils.io_helpers import load_config


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
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )


def get_logger():
    """Get the configured logger instance.

    Returns:
        Loguru logger instance.
    """
    return _loguru_logger


# Auto-configure on import using settings if available
try:
    _config = load_config()
    _log_config = _config.get("logging", {})
    setup_logger(
        level=_log_config.get("level", "INFO"),
        log_file=_log_config.get("log_file", "logs/aic2026.log"),
        rotation=_log_config.get("rotation", "10 MB"),
    )
except Exception:
    # Fallback: basic console logging if config not available
    setup_logger()

# Convenience export
logger = get_logger()
