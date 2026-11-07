"""Upstream tripwires (P8b): tests that fire when a vectorbt upgrade changes a definition.

Each test feeds the ggTrader adapter (``ggTrader.engine``) an input where the *installed
vectorbt* and an *independent reference* could disagree, and says on failure which side is
which, which vectorbt is installed and what upstream change it probably is
(``tests/engine/tripwire.py``). Only APIs the adapter actually calls are covered; changes in
APIs it never calls are listed in ``.claude/skills/vbt-upgrade/SKILL.md`` ("not relevant").

Found by the P8 rehearsal (research/programme/vectorbt-engine/P8-completion.md, section 6):

* G1/G2 rolling-std (Welford) change of 1.1.1 was invisible to the synthetic-panel tests.
* G3 the deflated Sharpe with NaN returns and fat tails (1.1.1 "Results change").
* G4 numba and Rust disagreeing on degenerate inputs (master with an older Rust wheel).
* G5 a window that ends in NaN.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import config, stats
from ggTrader.engine import indicators as ind

from . import tripwire
from .panels import ref_max_drawdown

TAPE = Path(__file__).resolve().parents[2] / "data" / "snapshots" / "ohlcv_20261106.parquet"
EULER_GAMMA = 0.5772156649015329
DAILY = pd.bdate_range("2021-01-04", periods=500)


def _fail_unless_close(
    actual: object,
    expected: object,
    *,
    quantity: str,
    reference: str,
    cause: str,
    triage: str,
    rtol: float,
    atol: float = 0.0,
) -> None:
    """``np.allclose`` with the tripwire message (installed side, reference side, cause)."""
    a = np.atleast_1d(np.asarray(actual, dtype=float)).ravel()
    e = np.atleast_1d(np.asarray(expected, dtype=float)).ravel()
    if np.allclose(a, e, rtol=rtol, atol=atol, equal_nan=True):
        return
    with np.errstate(invalid="ignore"):
        gap = np.abs(a - e) / np.maximum(np.abs(e), 1e-300)
    gap = np.where(np.isnan(a) != np.isnan(e), np.inf, np.nan_to_num(gap))
    worst = int(np.argmax(gap))
    pytest.fail(
        tripwire.explain(
            quantity,
            installed_value=float(a[worst]),
            reference_value=float(e[worst]),
            reference=reference,
            cause=cause,
            triage=triage,
        )
        + f"\n  tolerance:  rtol {rtol:g} atol {atol:g}; {int(a.size)} value(s) compared",
        pytrace=False,
    )


# --- G3: the deflated Sharpe ratio, from the Bailey and Lopez de Prado formulas ---------------


def ref_dsr(returns: np.ndarray, n_trials: int, *, null_moments: str = "pandas") -> float:
    """Deflated Sharpe ratio written from the paper, sharing no code with vectorbt or the lab.

    * NaN observations are omitted; T is the count of the others.
    * Sharpe is per period, ``mean/std(ddof=1)``.
    * Skewness and *excess* kurtosis are the unbiased estimators (what ``pandas`` returns).
    * The standard error of the Sharpe at ``sr`` is
      ``sqrt((1 - skew*sr + (excess_kurtosis + 2)/4 * sr**2) / (T - 1))``.
    * The benchmark is the expected maximum Sharpe of ``n_trials`` null trials,
      ``sqrt(1/(T-1)) * ((1-g) * z(1-1/N) + g * z(1-1/(N*e)))``.

    ``null_moments="wrong_kurtosis"`` returns what a wrong kurtosis convention (non-excess
    value used where excess is meant) would give; it exists only to prove a case is sensitive.
    """
    x = np.asarray(returns, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mean, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr = mean / sd
    m2 = np.mean((x - mean) ** 2)
    g1 = np.mean((x - mean) ** 3) / m2**1.5
    g2 = np.mean((x - mean) ** 4) / m2**2 - 3.0
    skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0) if n > 3 else float("nan")
    if null_moments == "wrong_kurtosis":
        excess += 3.0
    normal = NormalDist()
    variance = max(1.0 - skew * sr + (excess + 2.0) / 4.0 * sr * sr, 1e-12) / (n - 1)
    benchmark = math.sqrt(1.0 / (n - 1)) * (
        (1.0 - EULER_GAMMA) * normal.inv_cdf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * normal.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    return normal.cdf((sr - benchmark) / math.sqrt(variance))


def _fat_tailed(seed: int = 3, drift: float = 0.0030, n: int = 500) -> pd.Series:
    """Student-t(3) returns with occasional crashes: skewed, kurtosis well above normal."""
    rng = np.random.default_rng(seed)
    values = rng.standard_t(3, n) * 0.008 + drift
    values[::37] -= 0.05
    return pd.Series(values, index=pd.bdate_range("2021-01-04", periods=n))


DSR_CAUSE = (
    "vectorbt changed the deflated Sharpe definition (1.1.1 release notes: 'non-excess "
    "kurtosis, omit NaNs, Results change')"
)
DSR_TRIAGE = (
    "the independent formula is the reference. If the installed value is the one that is "
    "wrong, the old definition is not reachable by a flag: RE-BASELINE (owner decision, "
    "triage item 2). If it agrees with the reference, update this test with the reason and "
    "ACCEPT (triage item 2b)."
)


@pytest.mark.parametrize("n_trials", [10, 48, 80])
def test_dsr_on_fat_tailed_returns_matches_an_independent_formula(n_trials: int) -> None:
    returns = _fat_tailed()
    expected = ref_dsr(returns.to_numpy(), n_trials)
    assert 0.05 < expected < 0.95, "the case must not saturate, or a definition change hides"
    _fail_unless_close(
        stats.dsr(returns, n_trials),
        expected,
        quantity=f"stats.dsr, Student-t(3) returns with crashes, {n_trials} trials",
        reference="independent Bailey/Lopez de Prado formula in this file (pandas moments)",
        cause=DSR_CAUSE,
        triage=DSR_TRIAGE,
        rtol=1e-9,
    )


def test_the_fat_tailed_case_is_sensitive_to_the_kurtosis_convention() -> None:
    """Test power: a wrong kurtosis convention moves this case far above the 1e-9 tolerance."""
    values = _fat_tailed().to_numpy()
    right, wrong = ref_dsr(values, 48), ref_dsr(values, 48, null_moments="wrong_kurtosis")
    assert abs(right - wrong) > 1e-6 * right  # a million times the 1e-9 tolerance


def _frame_with_nans() -> pd.DataFrame:
    base = _fat_tailed(seed=8, drift=0.0036)
    rng = np.random.default_rng(5)
    scattered = base.copy()
    scattered.iloc[rng.choice(len(base), 70, replace=False)] = np.nan
    late_start = base.copy()
    late_start.iloc[:200] = np.nan
    nearly_empty = base.copy()
    nearly_empty.iloc[2:] = np.nan
    return pd.DataFrame(
        {
            "clean": base,
            "scattered": scattered,
            "late_start": late_start,
            "nearly_empty": nearly_empty,
        }
    )


def test_dsr_with_nan_returns_matches_an_independent_formula_per_column() -> None:
    frame = _frame_with_nans()
    expected = [ref_dsr(frame[c].to_numpy(), 48) for c in frame]
    assert np.isnan(expected[-1]) and all(0.05 < v < 0.95 for v in expected[:-1])
    out = stats.dsr(frame, 48)
    assert list(out.index) == list(frame.columns)
    _fail_unless_close(
        out.to_numpy(),
        expected,
        quantity="stats.dsr per column, returns with NaN (scattered, leading, almost all)",
        reference="independent formula in this file: NaN omitted, T = count of the rest",
        cause=DSR_CAUSE + "; NaN used to be zero-filled and counted in T before 1.1.1",
        triage=DSR_TRIAGE,
        rtol=1e-9,
    )


def test_dsr_nan_returns_are_omitted_not_zero_filled() -> None:
    """The same returns with the NaN rows deleted give the same DSR (Series path)."""
    series = _frame_with_nans()["scattered"]
    assert stats.dsr(series, 48) == pytest.approx(stats.dsr(series.dropna(), 48), rel=1e-9)


# --- G1/G2: rolling standard deviation behind the Bollinger bands ----------------------------


def _tape_closes(n_symbols: int = 60, min_bars: int = 1500) -> pd.DataFrame:
    long = pd.read_parquet(TAPE, columns=["timestamp", "symbol", "close"])
    counts = long.groupby("symbol")["close"].count()
    symbols = sorted(counts[counts >= min_bars].index)[:n_symbols]
    wide = long[long["symbol"].isin(symbols)].pivot(
        index="timestamp", columns="symbol", values="close"
    )
    return wide[symbols]


@pytest.mark.skipif(
    not TAPE.exists(),
    reason=f"frozen tape {TAPE.name} is absent (git-ignored; it lives on Encom, not on runners)",
)
def test_bollinger_bands_match_pandas_rolling_std_on_the_real_tape() -> None:
    """60 real symbols against ``rolling(20).std(ddof=1)``: the 1.1.1 Welford change shows here.

    On the synthetic panels vectorbt 1.1.0 and 1.1.1 agree to ~1e-13, below every tolerance;
    on the tape the naive running-variance formula of 1.1.0 is off by 1.4e-7 relative (5e-5
    absolute on a 200-dollar stock). Cells inside the 200-bar burn-in are not compared.
    """
    close = _tape_closes()
    bands = ind.bbands(close, 20, 2.0)
    mean, sd = close.rolling(20).mean(), close.rolling(20).std(ddof=1)
    first = close.notna().to_numpy().argmax(axis=0)
    settled = np.arange(len(close))[:, None] >= (first + 200)[None, :]
    compared = 0
    for name, got, want in (
        ("lower", bands.lower, mean - 2.0 * sd),
        ("upper", bands.upper, mean + 2.0 * sd),
    ):
        a, b = got.to_numpy(), want.to_numpy()
        ok = np.isfinite(a) & np.isfinite(b) & settled
        compared += int(ok.sum())
        rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-12)
        worst = int(np.argmax(rel))
        assert rel[worst] < 1e-10, tripwire.explain(
            f"Bollinger {name} band (window 20, alpha 2, ddof 1), worst of {int(ok.sum())} cells",
            installed_value=float(a[ok][worst]),
            reference_value=float(b[ok][worst]),
            reference="pandas rolling(20).std(ddof=1) on the frozen tape",
            cause="vectorbt changed its rolling standard deviation (1.1.0 had the naive running "
            "variance, 1.1.1 'numerical stability' switched to Welford)",
            triage="run scripts/engine_parity_evidence.py (runbook step 3b) and compare with the "
            "committed JSON; a signal flip is a RE-BASELINE (owner decision), a change under "
            "1e-9 with no flip is ACCEPT with a recorded reason.",
        )
    assert compared > 150_000, "too few comparable cells: the tape or the burn-in changed"


def _longdouble_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for end in range(window, len(values) + 1):
        chunk = values[end - window : end].astype(np.longdouble)
        out[end - 1] = float(np.sqrt(np.sum((chunk - chunk.mean()) ** 2) / (window - 1)))
    return out


def test_rolling_std_is_numerically_stable_for_a_large_offset_with_a_small_spread() -> None:
    """A level of 1e6 with a spread of 1e-2: the naive sum-of-squares form loses every digit."""
    rng = np.random.default_rng(1)
    close = pd.Series(
        1e6 + rng.normal(0.0, 1e-2, 400), index=pd.bdate_range("2021-01-04", periods=400)
    )
    bands = ind.bbands(close, 20, 1.0)
    got = (bands.upper - bands.middle).to_numpy()
    want = _longdouble_rolling_std(close.to_numpy(), 20)
    ok = np.isfinite(got) & np.isfinite(want)
    # (upper - middle) subtracts two 1e6-sized numbers, so it carries ~2e-10 of rounding itself.
    rel = np.abs(got[ok] - want[ok]) / want[ok]
    assert rel.max() < 1e-5, tripwire.explain(
        "rolling std (via Bollinger upper - middle) at level 1e6, spread 1e-2",
        installed_value=float(got[ok][int(rel.argmax())]),
        reference_value=float(want[ok][int(rel.argmax())]),
        reference="two-pass np.longdouble standard deviation of each window",
        cause="vectorbt's rolling std lost the Welford update (1.1.0 had the naive formula; "
        "1.1.1 fixed it) or it was refactored back",
        triage="a regression of numerical stability: REJECT unless the new form is exact too.",
    )


# --- G4: numba and Rust must agree on degenerate return series --------------------------------


def _degenerate_cases() -> dict[str, pd.Series]:
    rng = np.random.default_rng(2)
    n = 100
    positive = np.abs(rng.normal(0.001, 0.0005, n))
    trailing = np.r_[rng.normal(0.0, 0.01, 80), np.full(20, np.nan)]
    cases = {
        "flat_zero": np.zeros(n),
        "constant_positive": np.full(n, 0.001),
        "no_downside": positive,
        "single_value": np.r_[0.01, np.full(n - 1, np.nan)],
        "all_nan": np.full(n, np.nan),
        "trailing_nan": trailing,
    }
    index = pd.bdate_range("2021-01-04", periods=n)
    return {name: pd.Series(values, index=index) for name, values in cases.items()}


RAW_STATS = ("sharpe_ratio", "sortino_ratio", "annualized_volatility", "annualized", "max_drawdown")


def _under(engine: str, func):
    config.set_engine(engine)
    return func()


def _same(a: float, b: float, rtol: float = 1e-12) -> bool:
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    if np.isinf(a) or np.isinf(b):
        return bool(a == b)
    return bool(abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300) or abs(a - b) < 1e-15)


@pytest.mark.parametrize("case", list(_degenerate_cases()))
def test_the_two_engines_agree_on_every_statistic_of_a_degenerate_series(case: str) -> None:
    """The vectorbt accessor itself, NaN and inf compared as values: numba == Rust."""
    returns = _degenerate_cases()[case]
    for name in RAW_STATS:

        def value(name=name) -> float:
            accessor = returns.vbt.returns(freq=config.BAR_FREQ, year_freq=config.YEAR_FREQ)
            return float(getattr(accessor, name)())

        rust, numba = _under("rust", value), _under("numba", value)
        assert _same(rust, numba), tripwire.explain(
            f"returns accessor {name}() on the '{case}' series: Rust engine against numba engine",
            installed_value=rust,
            reference_value=numba,
            reference="the numba engine on the same input (the Rust wheel is the one under test)",
            cause="the vectorbt and vectorbt-rust halves differ in a zero-volatility rule (seen "
            "on unreleased master with the 1.2.0 wheel: numba NaN, Rust inf, polakowo #897)",
            triage="never ship a Rust path that disagrees with numba (triage item 4): REJECT, "
            "or pin GGT_VBT_ENGINE=numba for research and file an upstream issue.",
        )


@pytest.mark.parametrize("case", list(_degenerate_cases()))
def test_the_adapter_gives_the_same_answer_on_both_engines_for_a_degenerate_series(
    case: str,
) -> None:
    """ggTrader's own statistics (guards included) must be engine-independent, and never inf."""
    returns = _degenerate_cases()[case]
    funcs = {
        "sharpe": stats.sharpe,
        "sortino": stats.sortino,
        "volatility": stats.volatility,
        "max_drawdown": stats.max_drawdown,
        "cagr": stats.cagr,
    }
    for name, func in funcs.items():
        rust = _under("rust", lambda f=func: float(f(returns)))
        numba = _under("numba", lambda f=func: float(f(returns)))
        assert not np.isinf(rust) and not np.isinf(numba), f"stats.{name} leaked an inf"
        assert _same(rust, numba), tripwire.explain(
            f"stats.{name}() on the '{case}' series: Rust against numba",
            installed_value=rust,
            reference_value=numba,
            reference="the numba engine on the same input",
            cause="an engine-dependent degenerate value got past ggTrader's finite guard",
            triage="triage item 4: fix the guard in engine/stats.py or REJECT.",
        )


# --- G5: a window that ends in NaN -------------------------------------------------------------


def _ref_window_drawdown(window_returns: np.ndarray) -> float:
    """Drawdown of the compounded path that starts at the base bar (1.0), NaN tail dropped."""
    values = window_returns[: len(window_returns) - int(np.argmax(~np.isnan(window_returns[::-1])))]
    curve = np.concatenate([[1.0], np.cumprod(1.0 + values)])
    return float(np.min(curve / np.maximum.accumulate(curve) - 1.0))


def _ends_in_nan() -> pd.Series:
    rng = np.random.default_rng(12)
    values = rng.normal(0.0004, 0.012, 160)
    values[130:] = np.nan  # a delisted symbol: the window's returns stop and stay NaN
    return pd.Series(values, index=pd.bdate_range("2021-01-04", periods=160))


def test_max_drawdown_of_a_series_that_ends_in_nan_equals_the_explicit_path(engine: str) -> None:
    returns = _ends_in_nan()
    expected = ref_max_drawdown(returns.dropna().to_numpy())
    assert expected < -0.05, "the case needs a real drawdown"
    _fail_unless_close(
        stats.max_drawdown(returns),
        expected,
        quantity=f"stats.max_drawdown of returns ending in NaN ({engine} engine)",
        reference="explicit compounded path of the non-NaN prefix (panels.ref_max_drawdown)",
        cause="vectorbt changed how a trailing NaN enters the drawdown (1.2.0 'drawdowns when a "
        "series ends in NaN' fixed the value path; the returns path must not move)",
        triage="check which accessor changed; the returns-accessor path is the one ggTrader uses.",
        rtol=1e-9,
    )


def test_window_stats_of_a_window_that_ends_in_nan_equal_the_explicit_reference(
    engine: str,
) -> None:
    returns = _ends_in_nan()
    start, end = 60, 160  # bars 61..159: 70 real returns then 29 NaN
    window = returns.iloc[start + 1 : end].to_numpy()
    table = stats.window_stats(returns, start, end)
    column = table.iloc[:, 0]
    real = window[~np.isnan(window)]
    assert column["n_obs"] == len(real)
    _fail_unless_close(
        column["max_drawdown"],
        _ref_window_drawdown(window),
        quantity=f"window_stats max_drawdown, window ends in NaN ({engine} engine)",
        reference="explicit path from the base bar over the non-NaN returns",
        cause="trailing-NaN handling in vectorbt's drawdown or returns accessor changed (1.2.0)",
        triage="compare the returns-accessor path with the value-path drawdown in the P8 probe.",
        rtol=1e-9,
    )
    _fail_unless_close(
        column["sharpe"],
        real.mean() / real.std(ddof=1) * math.sqrt(config.TRADING_DAYS),
        quantity=f"window_stats sharpe, window ends in NaN ({engine} engine)",
        reference="mean/std(ddof=1)*sqrt(252) over the non-NaN returns",
        cause="vectorbt stopped skipping NaN in the returns moments",
        triage="REJECT or RE-BASELINE: NaN handling is a definition.",
        rtol=1e-9,
    )
