"""L0: the installed engine stack is the one uv.lock records, and pyproject pins the pair."""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENGINE_STACK = ("vectorbt", "vectorbt-rust", "pandas", "numpy", "numba", "llvmlite", "plotly")


@pytest.fixture(scope="module")
def locked() -> dict[str, str]:
    data = tomllib.loads((ROOT / "uv.lock").read_text())
    return {pkg["name"]: pkg["version"] for pkg in data["package"]}


@pytest.mark.parametrize("name", ENGINE_STACK)
def test_installed_version_equals_the_lock(locked: dict[str, str], name: str) -> None:
    assert name in locked, f"{name} missing from uv.lock"
    assert metadata.version(name) == locked[name], "venv is stale: run uv sync --frozen"


def test_the_vectorbt_pair_is_pinned_exactly_and_together() -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pins = {
        m.group(1): m.group(2)
        for d in deps
        if (m := re.match(r"^(vectorbt(?:\[rust\]|-rust)?)==([\d.]+)$", d))
    }
    assert set(pins) == {"vectorbt[rust]", "vectorbt-rust"}
    assert pins["vectorbt[rust]"] == pins["vectorbt-rust"]


def test_the_engine_stack_has_bounded_ranges() -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    for name in ("numpy", "pandas", "numba", "plotly"):
        spec = next(d for d in deps if re.match(rf"^{name}[<>=!~]", d))
        assert "<" in spec, f"{name} must carry an upper bound so a transitive cap cannot move it"


def test_python_is_pinned_to_the_image_minor() -> None:
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    assert requires == "==3.12.*"
