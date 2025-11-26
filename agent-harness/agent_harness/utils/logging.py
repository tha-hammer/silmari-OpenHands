"""Standard logging setup for agent harness."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "agent_harness",
    level: str = "INFO",
    format_string: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """Setup standard Python logger for harness.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        json_format: Whether to use JSON formatting (requires pythonjsonlogger)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if json_format:
            try:
                from pythonjsonlogger import jsonlogger
                formatter = jsonlogger.JsonFormatter(format_string)
            except ImportError:
                # Fallback to standard formatter if pythonjsonlogger not available
                format_string = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                formatter = logging.Formatter(format_string)
        else:
            format_string = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            formatter = logging.Formatter(format_string)

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

