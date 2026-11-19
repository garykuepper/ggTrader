"""Tests for application logging configuration."""

import io
import logging
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def restore_logger():
    """Isolate the application logger for each test."""
    logger = logging.getLogger("ggTrader")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    yield
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_configure_routes_ggtrader_info_to_stderr(capsys):
    from ggTrader.utils.logging_setup import configure

    configure()
    logging.getLogger("ggTrader.example").info("hello")
    assert "INFO ggTrader.example: hello" in capsys.readouterr().err


def test_no_duplicate_with_root_handler(capsys):
    from ggTrader.utils.logging_setup import configure

    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    try:
        configure()
        logging.getLogger("ggTrader.example").warning("once")
        assert capsys.readouterr().err.count("once") == 1
    finally:
        root.removeHandler(handler)


def test_third_party_levels_untouched():
    from ggTrader.utils.logging_setup import configure

    loggers = [logging.getLogger(name) for name in ("urllib3", "yfinance")]
    levels = [logger.level for logger in loggers]
    configure()
    assert [logger.level for logger in loggers] == levels


def test_invalid_level_falls_back(monkeypatch, capsys):
    from ggTrader.utils.logging_setup import configure

    monkeypatch.setenv("GGT_LOG_LEVEL", "LOUD")
    configure()
    assert logging.getLogger("ggTrader").level == logging.INFO
    assert "LOUD" in capsys.readouterr().err


def test_configure_is_idempotent():
    from ggTrader.utils.logging_setup import configure

    configure()
    configure()
    assert len(logging.getLogger("ggTrader").handlers) == 1


def test_handler_follows_stderr_swap(monkeypatch):
    from ggTrader.utils.logging_setup import configure

    configure()
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    logging.getLogger("ggTrader").info("swapped")
    assert "swapped" in stream.getvalue()


def test_env_level_respected(monkeypatch):
    from ggTrader.utils.logging_setup import configure

    monkeypatch.setenv("GGT_LOG_LEVEL", "DEBUG")
    configure()
    assert logging.getLogger("ggTrader").level == logging.DEBUG


def test_cli_callback_configures():
    from ggTrader.cli.main import app

    with patch("ggTrader.cli.main.configure") as configure:
        result = CliRunner().invoke(app, ["db", "--help"])
    assert result.exit_code == 0
    configure.assert_called_once_with()


def test_cli_style_configuration_does_not_leak_into_the_next_test_a() -> None:
    """Part 1 of a pair: configure the logger and leave it configured on purpose."""
    from ggTrader.utils.logging_setup import configure

    configure("INFO")
    assert logging.getLogger("ggTrader").level == logging.INFO


def test_cli_style_configuration_does_not_leak_into_the_next_test_b() -> None:
    """Part 2: the root conftest restored the logger, so INFO is not enabled by a prior test."""
    logger = logging.getLogger("ggTrader")
    assert logger.level == logging.NOTSET
    assert logger.propagate is True
    assert not any(type(h).__name__ == "_StderrHandler" for h in logger.handlers)
