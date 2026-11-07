import matplotlib
import pytest

matplotlib.use("Agg")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Golden regeneration for tests/engine: explicit, reasoned and changelogged."""
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
