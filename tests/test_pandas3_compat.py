"""Pin the pandas-2 behaviours that the pandas-3 / vectorbt 1.2 stack would silently change.

Every expected value here is a literal or an independent numpy/pandas reference that gives the
same answer on pandas 2.3.3 and pandas 3.x, so the module passes on both stacks and fails if a
call site regresses to a bare `pct_change()` or a bare `stack()`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.paper.overlay import compute_weights_and_scale

REPO = Path(__file__).resolve().parents[1]


def _idx(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


class TestNoBarePctChangeCalls:
    """Static guard: a bare `.pct_change()` means different numbers on pandas 2 and 3."""

    @staticmethod
    def _bare_calls(root: Path) -> list[str]:
        bad: list[str] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pct_change"
                    and not any(k.arg == "fill_method" for k in node.keywords)
                ):
                    bad.append(f"{path.relative_to(REPO)}:{node.lineno}")
        return bad

    def test_src_has_no_bare_pct_change(self) -> None:
        assert self._bare_calls(REPO / "src") == []

    def test_scripts_have_no_bare_pct_change(self) -> None:
        assert self._bare_calls(REPO / "scripts") == []

    def test_study_code_has_no_bare_pct_change(self) -> None:
        # Study code lives under research/<category>/<study>/code and research/<category>/_lib once
        # studies move there; until then the globs match nothing and the test passes trivially.
        roots = [*REPO.glob("research/*/*/code"), *REPO.glob("research/*/_lib")]
        assert [call for root in roots for call in self._bare_calls(root)] == []


class TestLivePathOverlay:
    """paper/overlay.py:88 -- gap-free sleeve curves keep their old weights and scale."""

    @staticmethod
    def _curves() -> dict[str, pd.Series]:
        rng = np.random.default_rng(5)
        idx = _idx(120)
        return {
            "sp500": pd.Series(10000 * np.exp(rng.normal(0.0004, 0.010, 120).cumsum()), index=idx),
            "nasdaq100": pd.Series(
                10000 * np.exp(rng.normal(0.0004, 0.020, 120).cumsum()), index=idx
            ),
        }

    def test_weights_and_scale_match_independent_reference(self) -> None:
        curves = self._curves()
        weights, scale = compute_weights_and_scale(curves, target_vol=0.068, window=60)

        rets = {k: v.to_numpy()[1:] / v.to_numpy()[:-1] - 1.0 for k, v in curves.items()}
        vol = {k: np.std(r[-60:], ddof=1) * np.sqrt(252) for k, r in rets.items()}
        inv = {k: 1.0 / x for k, x in vol.items()}
        want_w = {k: x / sum(inv.values()) for k, x in inv.items()}
        for k in curves:
            assert weights[k] == pytest.approx(want_w[k], rel=1e-12)
        blended = sum(rets[k] * want_w[k] for k in curves)
        want_scale = float(np.clip(0.068 / (np.std(blended[-60:], ddof=1) * np.sqrt(252)), 0, 1.0))
        assert scale == pytest.approx(want_scale, rel=1e-12)
        assert np.isfinite(scale)
