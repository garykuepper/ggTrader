"""Write or verify the frozen synthetic test inputs behind the bit-exact golden tests.

The seeded generators in ``tests/engine/panels.py`` and ``tests/lab/strategies/helpers.py`` build
prices with ``np.exp``/``np.sin``, whose float64 loops differ in the last bit between CPU
families (AVX2 versus AVX-512). The goldens were frozen on an AVX2 machine, so the parameter sets
they are compared on are stored in ``tests/engine/golden/panels.npz`` and
``tests/lab/strategies/golden/bars.npz`` and read from there.

Examples:
    python scripts/freeze_test_inputs.py            # verify: regenerate here and compare
    python scripts/freeze_test_inputs.py --write    # rewrite both files from the live generators

Only ``--write`` on the machine that produced the goldens keeps the inputs and the goldens in
step; a changed golden needs a changed freeze (and a changelog line). ``--check`` exits 1 when the
live generator no longer reproduces the frozen arrays bit for bit, which is expected on a CPU with
different SIMD kernels and is reported as the size of the difference, not as a test failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tests.engine import panels  # noqa: E402
from tests.frozen_inputs import load_frozen, save_frozen  # noqa: E402
from tests.lab.strategies import helpers  # noqa: E402


def _panel_entries() -> dict[str, dict[str, np.ndarray]]:
    """Generate every frozen panel live."""
    return {
        panels.panel_key(seed, n_bars, n_syms, holes, panels.START): panels.generate_panel_arrays(
            seed, n_bars, n_syms, holes=holes
        )
        for seed, n_bars, n_syms, holes in panels.FROZEN_PANELS
    }


def _bars_entries() -> dict[str, dict[str, np.ndarray]]:
    """Generate every frozen bar set live."""
    return {
        helpers.bars_key(seed, n_bars, n_syms, holes, symbols): helpers.generate_bar_arrays(
            seed, n_bars, n_syms, holes=holes
        )
        for seed, n_bars, n_syms, holes, symbols in helpers.FROZEN_BARS
    }


def _compare(path: Path, names: tuple[str, ...], live: dict[str, dict[str, np.ndarray]]) -> bool:
    """Print how each frozen entry differs from the live generator; True when all are equal."""
    all_equal = True
    for key, arrays in live.items():
        frozen = load_frozen(path, key, names)
        if frozen is None:
            print(f"MISSING  {key}")
            all_equal = False
            continue
        worst = 0.0
        for name in names:
            a, b = frozen[name], arrays[name]
            if not np.array_equal(a, b, equal_nan=True):
                worst = max(worst, float(np.nanmax(np.abs(a / b - 1.0))))
        status = "equal" if worst == 0.0 else f"DIFFERS max rel {worst:.2e}"
        all_equal &= worst == 0.0
        print(f"{status:24s} {key}")
    return all_equal


def main(argv: list[str] | None = None) -> int:
    """Verify (default) or rewrite the frozen inputs; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite both .npz files from the live generators (run on the golden machine)",
    )
    args = parser.parse_args(argv)

    from ggTrader.lab.strategies.cross_asset_sleeve import UNIVERSE

    if tuple(sorted(UNIVERSE)) != helpers.SLEEVE_SYMBOLS:
        print(f"SLEEVE_SYMBOLS {helpers.SLEEVE_SYMBOLS} != sorted(UNIVERSE) {sorted(UNIVERSE)}")
        return 2

    live_panels, live_bars = _panel_entries(), _bars_entries()
    if args.write:
        save_frozen(panels.PANELS_FILE, live_panels)
        save_frozen(helpers.BARS_FILE, live_bars)
        for path in (panels.PANELS_FILE, helpers.BARS_FILE):
            print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
        return 0
    ok_panels = _compare(panels.PANELS_FILE, panels.PANEL_ARRAYS, live_panels)
    ok_bars = _compare(helpers.BARS_FILE, helpers.BAR_ARRAYS, live_bars)
    return 0 if ok_panels and ok_bars else 1


if __name__ == "__main__":
    sys.exit(main())
