"""Oracle: the pre-rebuild hand-written Deflated Sharpe Ratio (numpy + scipy, no vectorbt).

Verbatim copy of ``DsrResult``, ``psr``, ``expected_max_sr`` and ``dsr_check`` from
``ggTrader.lab.gates`` (removed in P10: vectorbt's DSR through ``engine.stats.dsr`` is the only
implementation). It is the independent reference the engine statistic is compared with
(``tests/engine/test_stats.py``, ``tests/lab/wfo/test_logic_units.py``). Not part of the
application.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.stats import norm


class DsrResult(NamedTuple):
    """Result of the Deflated Sharpe Ratio check."""

    passed: bool
    dsr_value: float
    expected_max_sr: float
    threshold: float


_EULER_MASCHERONI = 0.5772156649015329


def _sr_standard_error(sr: float, n_obs: int, skew: float, kurtosis_excess: float) -> float:
    """Per-period standard error of the Sharpe Ratio estimate (Bailey & López de Prado 2014).

    ``sr`` must be a per-period (non-annualized) Sharpe ratio.
    σ̂(SR) = sqrt( (1 - γ₃·SR + (γ₄-1)/4 · SR²) / (T-1) ), where γ₄ = kurtosis_excess + 3.
    """
    if n_obs <= 1:
        return float("inf")
    numerator = 1.0 - skew * sr + (kurtosis_excess + 2.0) / 4.0 * sr**2
    numerator = max(numerator, 1e-12)
    return float(np.sqrt(numerator / (n_obs - 1)))


def psr(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis_excess: float,
) -> float:
    """Per-period Probabilistic Sharpe Ratio: P(true SR > benchmark_sr).

    Both ``observed_sr`` and ``benchmark_sr`` are per-period (non-annualized)
    Sharpe ratios.
    PSR(SR*) = Φ( (SR_obs - SR*) / σ̂(SR_obs) )
    """
    se = _sr_standard_error(observed_sr, n_obs, skew, kurtosis_excess)
    if se <= 0 or not np.isfinite(se):
        return 0.0
    z = (observed_sr - benchmark_sr) / se
    return float(norm.cdf(z))


def expected_max_sr(
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis_excess: float,
) -> float:
    """Per-period expected maximum Sharpe across N independent trials.

    Returns a per-period (non-annualized) Sharpe threshold.
    E[max(SR)] ≈ σ̂(SR) · ( (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)) )

    where γ is the Euler-Mascheroni constant and σ̂(SR) is evaluated at SR=0
    (the null hypothesis that the true SR is zero).
    """
    if n_trials < 2:
        return 0.0
    se = _sr_standard_error(0.0, n_obs, skew, kurtosis_excess)
    gamma = _EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(se * ((1.0 - gamma) * z1 + gamma * z2))


def dsr_check(
    observed_sr_annual: float,
    n_obs: int,
    n_trials: int,
    skew: float,
    kurtosis_excess: float,
    threshold: float = 0.80,
    periods_per_year: int = 252,
) -> DsrResult:
    """Deflated Sharpe Ratio: PSR evaluated against E[max(SR)] as the benchmark.

    Takes an annualized Sharpe (as returned by ``metrics.curve_stats``),
    converts to per-period via ``sqrt(periods_per_year)``, and reports
    ``DsrResult.expected_max_sr`` annualized.
    Passes if DSR >= threshold (default 0.80, per spec Section 6).
    """
    scale = math.sqrt(periods_per_year)
    sr = observed_sr_annual / scale
    e_max = expected_max_sr(n_trials, n_obs, skew, kurtosis_excess)
    dsr_value = psr(sr, e_max, n_obs, skew, kurtosis_excess)
    return DsrResult(
        passed=dsr_value >= threshold,
        dsr_value=dsr_value,
        expected_max_sr=e_max * scale,
        threshold=threshold,
    )
