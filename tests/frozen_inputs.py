"""Read and write the frozen synthetic inputs that bit-exact golden tests are held to.

The seeded generators in ``tests/engine/panels.py`` and ``tests/lab/strategies/helpers.py`` build
prices with ``np.exp`` (and ``np.sin``). NumPy dispatches those float64 loops by CPU: on an
AVX-512 machine (the GitHub runners) the SIMD/SVML kernels can differ from the AVX2 ones in the
last bit. The goldens were frozen on an AVX2 machine, so a bit-for-bit comparison of the engine's
equity curves against them fails on the runner although the engine behaves identically. Freezing
the generated arrays makes those comparisons CPU-independent; every other parameter set keeps
using the live generator.

One ``.npz`` per generator family, one ``<key>|<array>`` entry per array, keyed by the full
parameter set. ``scripts/freeze_test_inputs.py`` writes and verifies them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_frozen(path: Path, key: str, names: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    """The frozen arrays stored under ``key``, or None when that parameter set is not frozen."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if f"{key}|{names[0]}" not in data.files:
            return None
        return {name: data[f"{key}|{name}"] for name in names}


def save_frozen(path: Path, entries: dict[str, dict[str, np.ndarray]]) -> None:
    """Write ``{key: {name: array}}`` as one compressed archive (sorted for stable bytes)."""
    flat = {f"{key}|{name}": arr for key in sorted(entries) for name, arr in entries[key].items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
