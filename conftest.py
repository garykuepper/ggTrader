"""Repo-root pytest configuration: it applies to every test, wherever it lives.

It sits at the repository root (not in tests/) so that tests collected from outside tests/, such
as study tests under research/<category>/<study>/code/tests, get the same safety fixture, the same
plotting backend and the same options. tests/test_root_conftest.py proves it.
"""

import logging
import os

import matplotlib
import pytest

from tests import tiers
from tests.timeout_guard import UNTIMED_TESTS, time_limit

#: The S&P 500 list every test reads unless it says otherwise: the frozen vintage-1 fixture, so
#: no test needs the database (spec 3.2 and 7; slice M3).
TEST_MEMBERSHIP = "fixture:sp500_v1_2026-06-02"

matplotlib.use("Agg")


#: Seconds one test may run before it fails as a presumed hang. The slowest test in the recorded
#: baseline takes 21 s, so this is generous; GGT_TEST_TIMEOUT=0 turns it off, and a test that
#: really needs longer says so with ``@pytest.mark.timeout_seconds(n)``.
DEFAULT_TEST_TIMEOUT_SECONDS = 180.0


@pytest.fixture(autouse=True)
def _per_test_timeout(request: pytest.FixtureRequest):
    """Fail a test that runs past its wall-clock limit (tests/timeout_guard.py)."""
    seconds = float(os.environ.get("GGT_TEST_TIMEOUT", DEFAULT_TEST_TIMEOUT_SECONDS))
    marker = request.node.get_closest_marker("timeout_seconds")
    if marker is not None and marker.args:
        seconds = float(marker.args[0])
    if request.node.nodeid in UNTIMED_TESTS:
        seconds = 0.0
    with time_limit(seconds):
        yield


@pytest.fixture(autouse=True)
def _restore_ggtrader_logger():
    """Give every test the ggTrader logger it started with.

    ``ggt`` configures that logger (a handler, level INFO) in its CLI callback, and a test that
    runs the CLI would otherwise leave INFO enabled for the rest of the process. On Python 3.12 a
    log record reads ``time.time()``, and paper tests that stub the global clock with a finite
    sequence then lose values to those records: the order-polling loop never reaches its
    deadline and the run allocates until the memory cap kills it (found 2026-11-18).
    """
    logger = logging.getLogger("ggTrader")
    handlers, level, propagate, disabled = (
        list(logger.handlers),
        logger.level,
        logger.propagate,
        logger.disabled,
    )
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
    logger.disabled = disabled


def pytest_addoption(parser: pytest.Parser) -> None:
    """Golden regeneration for tests/engine (explicit, reasoned, changelogged) and the tiers."""
    parser.getgroup("test tiers").addoption(
        "--tier",
        choices=sorted(tiers.TIER_MARKEXPR),
        default=None,
        help="fast = developer loop, full = merge/deploy gate, studies = study tests only "
        "(scripts/run_tests.sh picks one; a bare pytest is the full tier)",
    )
    parser.getgroup("test tiers").addoption(
        "--min-collected",
        type=int,
        default=0,
        help="fail with a usage error if fewer tests than this are collected "
        "(the study half of scripts/run_tests.sh sets it from tests/tiers.py)",
    )
    group = parser.getgroup("engine goldens")
    group.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite tests/engine/golden/*.json (needs --golden-reason)",
    )
    group.addoption(
        "--golden-reason",
        default="",
        help="why the goldens change; must also appear in docs/changelog.md",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Apply the chosen tier's marker expression (it replaces the pyproject default)."""
    tiers.apply_tier(config.option, config.getoption("--tier"))


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail loudly when a run collected fewer tests than ``--min-collected`` promised."""
    minimum = session.config.getoption("--min-collected")
    if len(session.items) < minimum:
        raise pytest.UsageError(
            f"collected {len(session.items)} tests but --min-collected is {minimum}: "
            "a study test folder is missing or failed to collect"
        )


@pytest.fixture(autouse=True, scope="session")
def _no_dotenv():
    """Never load the developer's real `.env` during tests.

    `utils/config._load_env()` calls `load_dotenv`, which writes into
    `os.environ` for the rest of the session. Once any test triggered it, live
    flags like `CASH_SWEEP_ENABLED=true` leaked into every later test, making
    results order-dependent (and unlike CI, which has no `.env`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ggTrader.utils.config.load_dotenv", lambda *a, **k: False)
        yield


@pytest.fixture(autouse=True)
def _membership_is_a_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the frozen membership fixture for every test, so none can reach the database.

    Library code that is handed no `membership=` reads the source named by `GGT_MEMBERSHIP`
    (`ggTrader.data.membership.selected_source`); with this set that is a file. Tests of the
    selection itself clear or change the variable.
    """
    monkeypatch.setenv("GGT_MEMBERSHIP", TEST_MEMBERSHIP)
