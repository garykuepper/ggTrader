"""Repo-root pytest configuration: it applies to every test, wherever it lives.

It sits at the repository root (not in tests/) so that tests collected from outside tests/, such
as study tests under research/<category>/<study>/code/tests, get the same safety fixture, the same
plotting backend and the same options. tests/test_root_conftest.py proves it.
"""

import matplotlib
import pytest

from tests import tiers

matplotlib.use("Agg")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Golden regeneration for tests/engine (explicit, reasoned, changelogged) and the tiers."""
    parser.getgroup("test tiers").addoption(
        "--tier",
        choices=sorted(tiers.TIER_MARKEXPR),
        default=None,
        help="fast = developer loop, full = merge/deploy gate, studies = study tests only "
        "(scripts/run_tests.sh picks one; a bare pytest is the full tier)",
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
