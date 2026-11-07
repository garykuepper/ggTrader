"""L1b: tripwire on vectorbt's DEFAULT parameters.

Every preset in ggTrader.engine spells out the flags that matter, because vectorbt's defaults
are a different strategy (Cutler RSI, SMA MACD, ddof=0 Bollinger, span-EWM ATR, 365-day
year). This test fails the moment an upgrade changes any default of an API we call, listing
each change, so the agent can decide whether a preset needs a new explicit flag.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from . import tripwire
from .golden_io import check_or_update
from .panels import make_panel, ref_bbands, ref_ema, ref_sma, ref_true_range

PROBE = Path(__file__).with_name("defaults_probe.py")
#: Keys of the probe output that describe the installed build, not a default. Every bump changes
#: them by definition, so comparing them made this test fail on every upgrade with nothing but a
#: version line (P8, R1). The versions are recorded in provenance (``engine/provenance.py``).
BUILD_KEYS = ("vectorbt_version",)


@pytest.fixture(scope="module")
def upstream_defaults() -> dict:
    """Defaults as vectorbt ships them: a clean interpreter that never imports ggTrader."""
    proc = subprocess.run(
        [sys.executable, "-I", str(PROBE)], capture_output=True, text=True, timeout=300, check=True
    )
    return json.loads(proc.stdout)


def _diff(actual: object, expected: object, path: str = "") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        out = []
        for key in sorted(set(actual) | set(expected)):
            out += _diff(
                actual.get(key, "<missing>"), expected.get(key, "<missing>"), f"{path}/{key}"
            )
        return out
    return [] if actual == expected else [f"{path}: {expected!r} -> {actual!r}"]


def definitional(payload: dict) -> dict:
    """The probe payload without the build keys: only these are compared with the golden."""
    return {key: value for key, value in payload.items() if key not in BUILD_KEYS}


def test_vectorbt_defaults_have_not_changed(request, upstream_defaults: dict) -> None:
    golden = check_or_update(request, "vbt_defaults", upstream_defaults)
    changes = _diff(definitional(upstream_defaults), definitional(golden))
    assert not changes, (
        f"vectorbt defaults changed ({tripwire.installed()} against the golden made in "
        f"{golden.get('vectorbt_version', 'an unknown version')}):\n  "
        + "\n  ".join(changes)
        + "\nDecide per default whether an engine preset must now set an explicit flag "
        "(vbt-upgrade triage item 2)."
    )


def test_a_version_only_difference_is_not_a_default_change(upstream_defaults: dict) -> None:
    """A bump alone must leave the tripwire green (P8, R1); provenance carries the versions."""
    bumped = json.loads(json.dumps(upstream_defaults))
    bumped["vectorbt_version"] = "9.9.9"
    assert _diff(definitional(bumped), definitional(upstream_defaults)) == []
    raw = _diff(bumped, upstream_defaults)  # the old comparison: this is what failed every bump
    assert len(raw) == 1 and raw[0].startswith("/vectorbt_version")


def test_a_changed_default_is_still_reported_next_to_a_version_bump(
    upstream_defaults: dict,
) -> None:
    bumped = json.loads(json.dumps(upstream_defaults))
    bumped["vectorbt_version"] = "9.9.9"
    bumped["signatures"]["BBANDS.run"]["ddof"] = "1"
    changes = _diff(definitional(bumped), definitional(upstream_defaults))
    assert len(changes) == 1 and changes[0].startswith("/signatures/BBANDS.run/ddof")


def test_the_golden_still_records_the_version_it_was_made_with() -> None:
    """The version stays in the probe output and the golden, only the comparison skips it."""
    from .golden_io import load_golden

    assert load_golden("vbt_defaults")["vectorbt_version"]


def test_the_tripwire_notices_a_changed_default(upstream_defaults: dict) -> None:
    mutated = json.loads(json.dumps(upstream_defaults))
    mutated["settings"]["returns"]["year_freq"] = "252 days"
    mutated["signatures"]["BBANDS.run"]["ddof"] = "1"
    assert len(_diff(upstream_defaults, mutated)) >= 1


def test_the_headline_defaults_are_the_ones_the_presets_override(upstream_defaults: dict) -> None:
    """Literal values, so a reader of the failure sees what is being protected."""
    assert upstream_defaults["settings"]["returns"]["year_freq"] == "'365 days'"
    assert upstream_defaults["settings"]["engine"] == "'auto'"
    assert upstream_defaults["settings"]["returns"]["defaults"]["ddof"] == "1"
    assert upstream_defaults["signatures"]["RSI.run"]["ewm"] == "False"
    assert upstream_defaults["signatures"]["MACD.run"]["macd_ewm"] == "False"
    assert upstream_defaults["signatures"]["MACD.run"]["signal_ewm"] == "False"
    assert upstream_defaults["signatures"]["ATR.run"]["ewm"] == "True"
    assert upstream_defaults["signatures"]["BBANDS.run"]["ewm"] == "False"


# --- behavioural defaults (the kwargs_to_args ones are invisible in the signature) -------


@pytest.fixture(scope="module")
def close() -> pd.DataFrame:
    return make_panel(seed=5, n_bars=300, n_syms=2)["close"]


def test_default_bollinger_std_is_population_ddof_0(close: pd.DataFrame) -> None:
    out = vbt.BBANDS.run(close, window=20, alpha=2.0).lower.iloc[:, 0].to_numpy()
    _, _, lower0 = ref_bbands(close.iloc[:, 0].to_numpy(), 20, 2.0, ddof=0)
    np.testing.assert_allclose(out[19:], lower0[19:], rtol=1e-9)


def test_default_mstd_is_ddof_0(close: pd.DataFrame) -> None:
    out = vbt.MSTD.run(close, window=20).mstd.iloc[:, 0].to_numpy()
    expected = close.iloc[:, 0].rolling(20).std(ddof=0).to_numpy()
    np.testing.assert_allclose(out[19:], expected[19:], rtol=1e-9)


def test_default_rsi_is_the_simple_average_form_not_wilder(close: pd.DataFrame) -> None:
    out = vbt.RSI.run(close, window=14).rsi.iloc[:, 0].to_numpy()
    x = close.iloc[:, 0].to_numpy()
    delta = np.diff(x)
    gain, loss = (
        ref_sma(np.concatenate([[np.nan], np.clip(delta, 0, None)]), 14),
        ref_sma(np.concatenate([[np.nan], np.clip(-delta, 0, None)]), 14),
    )
    ref = 100 - 100 / (1 + gain / loss)
    np.testing.assert_allclose(out[15:], ref[15:], rtol=1e-9)


def test_default_macd_is_the_sma_form(close: pd.DataFrame) -> None:
    out = vbt.MACD.run(close).macd.iloc[:, 0].to_numpy()
    x = close.iloc[:, 0].to_numpy()
    ref = ref_sma(x, 12) - ref_sma(x, 26)
    np.testing.assert_allclose(out[25:], ref[25:], rtol=1e-9)


def test_default_atr_is_span_ewm_not_a_simple_average(close: pd.DataFrame) -> None:
    high, low = close * 1.01, close * 0.99
    out = vbt.ATR.run(high, low, close, window=14).atr.iloc[:, 0].to_numpy()
    tr = ref_true_range(
        high.iloc[:, 0].to_numpy(), low.iloc[:, 0].to_numpy(), close.iloc[:, 0].to_numpy()
    )
    ewm = ref_ema(tr, 14)
    np.testing.assert_allclose(out[14:], ewm[14:], rtol=1e-6)
    assert np.nanmax(np.abs(out[14:] - ref_sma(tr, 14)[14:])) > 1e-4


def test_default_ma_is_a_simple_average(close: pd.DataFrame) -> None:
    out = vbt.MA.run(close, window=20).ma.iloc[:, 0].to_numpy()
    np.testing.assert_allclose(out[19:], ref_sma(close.iloc[:, 0].to_numpy(), 20)[19:], rtol=1e-9)


def test_default_sharpe_uses_a_365_day_year(upstream_defaults: dict) -> None:
    """In a clean interpreter, daily returns are annualised with sqrt(365)."""
    code = (
        "import numpy as np, pandas as pd, vectorbt as vbt\n"
        "r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300),"
        " index=pd.bdate_range('2021-01-04', periods=300))\n"
        "s = r.vbt.returns(freq='1D').sharpe_ratio()\n"
        "print(s / (r.mean() / r.std(ddof=1)))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert float(out.stdout) == pytest.approx(365**0.5, rel=1e-9)


def _flat_returns() -> pd.Series:
    return pd.Series(np.zeros(100), index=pd.bdate_range("2021-01-04", periods=100))


def test_flat_curve_sharpe_is_still_inf_upstream() -> None:
    """Why stats.sharpe has a finite guard. If this fails vectorbt fixed it; keep the guard."""
    value = _flat_returns().vbt.returns(freq="1D", year_freq="252 days").sharpe_ratio()
    assert np.isinf(value), tripwire.explain(
        "Sharpe ratio of an all-zero (flat) return series",
        installed_value=float(value),
        reference_value=float("inf"),
        reference="the behaviour recorded in 1.1.0 to 1.2.0: zero volatility gives +inf",
        cause="upstream fixed zero-volatility Sharpe (polakowo/vectorbt #897: NaN instead of inf)",
        triage="ggTrader is unaffected (stats._nan_if_nonfinite maps inf to NaN, so the "
        "application value does not move). Keep the guard, change this assertion to accept NaN "
        "with a recorded reason (vbt-upgrade, the 'flat curve' row of the failure table).",
    )


def test_flat_curve_sortino_is_still_inf_upstream() -> None:
    """The Sortino sibling of the test above: zero downside deviation gives +inf upstream."""
    value = _flat_returns().vbt.returns(freq="1D", year_freq="252 days").sortino_ratio()
    assert np.isinf(value), tripwire.explain(
        "Sortino ratio of an all-zero (flat) return series",
        installed_value=float(value),
        reference_value=float("inf"),
        reference="the behaviour recorded in 1.1.0 to 1.2.0: zero downside gives +inf",
        cause="upstream fixed zero-volatility ratios (polakowo/vectorbt #897 changed Sharpe and "
        "Sortino to NaN)",
        triage="ggTrader is unaffected (stats.sortino maps non-finite to NaN). Keep the guard, "
        "change this assertion to accept NaN with a recorded reason.",
    )
