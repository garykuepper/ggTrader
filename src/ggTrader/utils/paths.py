import os
from pathlib import Path


def find_project_root() -> Path:
    """
    Find the project root directory by looking for pyproject.toml.

    Returns:
        Path: The absolute path to the project root.
    """
    # Start looking from the directory this file is in
    current = Path(__file__).absolute().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback to current working directory hierarchy
    current_cwd = Path(os.getcwd()).absolute()
    for parent in [current_cwd] + list(current_cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Final absolute fallback (relative to src/ggTrader/utils/paths.py)
    return Path(__file__).resolve().parent.parent.parent.parent
