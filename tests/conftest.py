import matplotlib
import pytest

matplotlib.use("Agg")


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
