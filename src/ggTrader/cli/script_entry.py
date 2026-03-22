"""Load and run ``scripts/*.py`` ``main()`` for setuptools console scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def scripts_dir() -> Path:
    """Repository ``scripts/`` directory (package lives under ``src/ggTrader/``)."""
    return Path(__file__).resolve().parents[3] / "scripts"


def run_script(filename: str) -> None:
    """Execute ``main()`` from a file in ``scripts/``."""
    path = scripts_dir() / filename
    if not path.is_file():
        sys.stderr.write(f"ggTrader: script not found: {path}\n")
        sys.exit(1)
    mod_name = f"_ggtrader_script_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        sys.stderr.write(f"ggTrader: could not load {path}\n")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    main_fn = getattr(mod, "main", None)
    if not callable(main_fn):
        sys.stderr.write(f"ggTrader: no main() in {path}\n")
        sys.exit(1)
    main_fn()
