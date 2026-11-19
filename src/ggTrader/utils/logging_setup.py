"""Configure application logging for the CLI."""

import logging
import os
import sys


class _StderrHandler(logging.StreamHandler):
    """Resolve stderr when emitting, including after CLI capture changes it."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record to the current stderr stream."""
        self.stream = sys.stderr
        super().emit(record)


def configure(level: str | None = None) -> None:
    """Configure the ggTrader logger without changing third-party logging."""
    logger = logging.getLogger("ggTrader")
    if not any(isinstance(handler, _StderrHandler) for handler in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    selected = level if level is not None else os.environ.get("GGT_LOG_LEVEL", "INFO")
    try:
        logger.setLevel(selected.upper())
    except ValueError:
        logger.setLevel(logging.INFO)
        logger.warning("Invalid log level %r; using INFO", selected)
