"""Numba-compiled quantitative factor functions for Strategy 1.

Includes:
  - compute_momentum_nb: Volatility-adjusted idiosyncratic momentum.
  - compute_liquidity_shock_nb: Amihud-proxy price impact per dollar volume.
  - strip_btc_beta_nb: Rolling OLS to strip systematic BTC beta.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def compute_momentum_nb(close_arr: np.ndarray, window: int, gap: int) -> np.ndarray:
    """Computes the volatility-adjusted idiosyncratic momentum factor.

    For each asset, it calculates the log return over the window (skipping the
    last gap bars) and normalizes it by the realized volatility of log returns
    over that lookback window.

    Args:
        close_arr: 2D array of close prices of shape (n_timestamps, n_assets).
        window: Total lookback window in bars.
        gap: Exclusion gap (short-term reversal skip) in bars.

    Returns:
        2D array of shape (n_timestamps, n_assets) containing factor values.
        Warmup rows (where index < window) are filled with np.nan.
    """
    n_t, n_assets = close_arr.shape
    mom_out = np.empty((n_t, n_assets), dtype=np.float64)
    mom_out[:] = np.nan

    if window <= gap or window <= 1:
        return mom_out

    slice_len = window - gap
    if slice_len <= 1:
        return mom_out

    for i in range(window, n_t):
        for col in range(n_assets):
            p_start = close_arr[i - window, col]
            p_end = close_arr[i - gap - 1, col]

            if p_start <= 0.0 or p_end <= 0.0 or np.isnan(p_start) or np.isnan(p_end):
                continue

            log_return = np.log(p_end) - np.log(p_start)

            # Volatility (standard deviation of daily log returns over the lookback window)
            sum_lr = 0.0
            sum_sq_lr = 0.0
            valid = True
            n_lr = slice_len - 1

            for k in range(n_lr):
                p1 = close_arr[i - window + k, col]
                p2 = close_arr[i - window + k + 1, col]
                if p1 <= 0.0 or p2 <= 0.0 or np.isnan(p1) or np.isnan(p2):
                    valid = False
                    break
                lr = np.log(p2) - np.log(p1)
                sum_lr += lr
                sum_sq_lr += lr * lr

            if not valid:
                continue

            mean_lr = sum_lr / n_lr
            var_lr = (sum_sq_lr / n_lr) - (mean_lr * mean_lr)
            vol = np.sqrt(max(var_lr, 0.0))

            mom_out[i, col] = log_return / (vol + 1e-8)

    return mom_out


@njit(cache=True)
def compute_liquidity_shock_nb(close_arr: np.ndarray, vol_arr: np.ndarray) -> np.ndarray:
    """Computes the Amihud-proxy liquidity shock factor.

    Calculates the absolute price impact per dollar volume:
        |log(close[t]/close[t-1])| / (volume[t] + 1e-8)

    Args:
        close_arr: 2D array of close prices of shape (n_timestamps, n_assets).
        vol_arr: 2D array of dollar volumes of shape (n_timestamps, n_assets).

    Returns:
        2D array of shape (n_timestamps, n_assets) containing factor values.
        The first row is filled with np.nan.
    """
    n_t, n_assets = close_arr.shape
    liq_out = np.empty((n_t, n_assets), dtype=np.float64)
    liq_out[:] = np.nan

    for i in range(1, n_t):
        for col in range(n_assets):
            c_prev = close_arr[i - 1, col]
            c_curr = close_arr[i, col]
            v_curr = vol_arr[i, col]

            if (
                c_prev <= 0.0
                or c_curr <= 0.0
                or v_curr < 0.0
                or np.isnan(c_prev)
                or np.isnan(c_curr)
                or np.isnan(v_curr)
            ):
                continue

            log_ret = np.log(c_curr) - np.log(c_prev)
            liq_out[i, col] = np.abs(log_ret) / (v_curr + 1e-8)

    return liq_out


@njit(cache=True)
def strip_btc_beta_nb(
    alt_log_ret: np.ndarray, btc_log_ret: np.ndarray, window: int
) -> np.ndarray:
    """Strips systematic BTC beta from altcoin log returns via rolling OLS.

    For each altcoin, it performs a rolling window OLS: alt ~ btc, and returns
    the residual value at the current timestamp.

    Args:
        alt_log_ret: 2D array of altcoin log returns of shape (n_timestamps, n_alts).
        btc_log_ret: 1D array of BTC log returns of shape (n_timestamps,).
        window: Rolling regression lookback window in bars.

    Returns:
        2D array of shape (n_timestamps, n_alts) containing residual returns.
        Warmup rows (where index < window) are filled with np.nan.
    """
    n_t, n_alts = alt_log_ret.shape
    residual = np.empty((n_t, n_alts), dtype=np.float64)
    residual[:] = np.nan

    if window <= 1 or n_t <= window:
        return residual

    for i in range(window, n_t):
        # Calculate mean and variance of BTC returns over the window [i - window + 1, i]
        sum_btc = 0.0
        valid_btc = True
        for k in range(i - window + 1, i + 1):
            val = btc_log_ret[k]
            if np.isnan(val):
                valid_btc = False
                break
            sum_btc += val

        if not valid_btc:
            continue

        mean_btc = sum_btc / window

        var_btc = 0.0
        for k in range(i - window + 1, i + 1):
            dev = btc_log_ret[k] - mean_btc
            var_btc += dev * dev
        var_btc = var_btc / window

        if var_btc < 1e-12:
            continue

        for col in range(n_alts):
            sum_alt = 0.0
            sum_cov = 0.0
            valid_alt = True
            for k in range(i - window + 1, i + 1):
                val_alt = alt_log_ret[k, col]
                if np.isnan(val_alt):
                    valid_alt = False
                    break
                sum_alt += val_alt
                sum_cov += (btc_log_ret[k] - mean_btc) * val_alt

            if not valid_alt:
                continue

            mean_alt = sum_alt / window
            cov_btc_alt = sum_cov / window

            beta = cov_btc_alt / var_btc
            alpha = mean_alt - beta * mean_btc

            # Residual for the current timestamp
            residual[i, col] = alt_log_ret[i, col] - (alpha + beta * btc_log_ret[i])

    return residual
