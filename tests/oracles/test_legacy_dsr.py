"""The legacy DSR oracle keeps its reference values (``engine.stats.dsr`` is checked against it)."""

import math

import pytest

from tests.oracles.legacy_dsr import DsrResult, _sr_standard_error, dsr_check, expected_max_sr, psr


def test_psr_known_values():
    """PSR with zero skew/kurtosis and large T should match basic z-test."""
    # SR=1.0 vs benchmark=0.0, T=252, normal returns
    result = psr(
        observed_sr=1.0,
        benchmark_sr=0.0,
        n_obs=252,
        skew=0.0,
        kurtosis_excess=0.0,
    )
    # SE(SR) = sqrt(1/251) ≈ 0.0631, z = 1.0/0.0631 ≈ 15.85 -> PSR ≈ 1.0
    assert result > 0.99


def test_psr_low_sr_low_probability():
    """PSR with SR barely above benchmark should give ~0.5."""
    result = psr(
        observed_sr=0.5,
        benchmark_sr=0.5,
        n_obs=252,
        skew=0.0,
        kurtosis_excess=0.0,
    )
    assert abs(result - 0.5) < 0.01


def test_psr_negative_skew_reduces_confidence():
    """Negative skew inflates SR variance -> lower PSR."""
    psr_normal = psr(
        observed_sr=1.0,
        benchmark_sr=0.0,
        n_obs=252,
        skew=0.0,
        kurtosis_excess=0.0,
    )
    psr_skewed = psr(
        observed_sr=1.0,
        benchmark_sr=0.0,
        n_obs=252,
        skew=-2.0,
        kurtosis_excess=5.0,
    )
    assert psr_skewed < psr_normal


def test_expected_max_sr_increases_with_trials():
    """More trials -> higher expected max Sharpe."""
    sr10 = expected_max_sr(n_trials=10, n_obs=252, skew=0.0, kurtosis_excess=0.0)
    sr1000 = expected_max_sr(n_trials=1000, n_obs=252, skew=0.0, kurtosis_excess=0.0)
    sr10000 = expected_max_sr(n_trials=10000, n_obs=252, skew=0.0, kurtosis_excess=0.0)
    assert sr10 < sr1000 < sr10000
    # With 10 trials and T=252, expected max SR should be modest
    assert 0.05 < sr10 < 0.5


def test_expected_max_sr_longer_track_record_narrows():
    """Longer T -> smaller SE -> smaller expected max SR."""
    sr_short = expected_max_sr(n_trials=100, n_obs=60, skew=0.0, kurtosis_excess=0.0)
    sr_long = expected_max_sr(n_trials=100, n_obs=1000, skew=0.0, kurtosis_excess=0.0)
    assert sr_long < sr_short


def test_sr_standard_error_normal_matches_lo_2002():
    # Normal returns: Var(SR) = (1 + SR^2/2) / (T-1); pins the (excess + 2)/4 kurtosis term.
    assert _sr_standard_error(0.1, 253, 0.0, 0.0) == pytest.approx(
        math.sqrt(1.005 / 252), rel=1e-12
    )


def test_dsr_check_reference_value():
    r = dsr_check(observed_sr_annual=1.0, n_obs=252, n_trials=48, skew=0.0, kurtosis_excess=0.0)
    assert r.dsr_value == pytest.approx(0.1036, abs=0.002)
    assert r.expected_max_sr == pytest.approx(2.265, abs=0.01)  # annualized
    assert r.passed is False


def test_dsr_check_strong_signal_passes():
    r = dsr_check(observed_sr_annual=3.0, n_obs=504, n_trials=20, skew=0.0, kurtosis_excess=0.0)
    assert isinstance(r, DsrResult)
    assert r.passed is True
    assert r.dsr_value == pytest.approx(0.9898, abs=0.002)


def test_dsr_check_annual_1p5_over_two_years_is_not_significant():
    r = dsr_check(observed_sr_annual=1.5, n_obs=504, n_trials=20, skew=0.0, kurtosis_excess=0.0)
    assert r.passed is False
    assert r.dsr_value == pytest.approx(0.586, abs=0.005)


def test_dsr_check_data_mined_fails():
    """Mediocre SR found after thousands of trials should fail."""
    result = dsr_check(
        observed_sr_annual=0.15,
        n_obs=252,
        n_trials=10000,
        skew=-1.0,
        kurtosis_excess=3.0,
    )
    assert result.passed is False
    assert result.dsr_value < 0.80


def test_dsr_check_threshold_respected():
    kw = dict(observed_sr_annual=1.0, n_obs=252, n_trials=48, skew=0.0, kurtosis_excess=0.0)
    strict, relaxed = dsr_check(**kw, threshold=0.5), dsr_check(**kw, threshold=0.05)
    assert relaxed.passed and not strict.passed
    assert relaxed.dsr_value == strict.dsr_value
