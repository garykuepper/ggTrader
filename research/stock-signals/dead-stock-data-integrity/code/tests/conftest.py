"""Put the study's code on sys.path and share the file-read counter between test files."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ggTrader.utils.study import bootstrap, study_dir

bootstrap(__file__)


class OpenCounter:
    """Counts every way the test can see a file being read: bytes, text and ``open``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
        self.reads: list[str] = []
        real_bytes, real_text, real_open = Path.read_bytes, Path.read_text, Path.open

        inside = [False]  # read_bytes and read_text call open themselves: count them once

        def note(path: Path, how: str) -> None:
            if path == target:
                self.reads.append(how)

        def read_bytes(path: Path) -> bytes:
            note(path, "read_bytes")
            inside[0] = True
            try:
                return real_bytes(path)
            finally:
                inside[0] = False

        def read_text(path: Path, *a: Any, **k: Any) -> str:
            note(path, "read_text")
            inside[0] = True
            try:
                return real_text(path, *a, **k)
            finally:
                inside[0] = False

        def open_(path: Path, *a: Any, **k: Any) -> Any:
            if not inside[0]:
                note(path, "open")
            return real_open(path, *a, **k)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
        monkeypatch.setattr(Path, "read_text", read_text)
        monkeypatch.setattr(Path, "open", open_)


@pytest.fixture
def open_counter(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A factory: ``open_counter(path)`` starts counting reads of ``path`` and returns the count."""
    return lambda target: OpenCounter(monkeypatch, target)


RUN_MODULE_NAME = "dead_stock_data_integrity_run"


@pytest.fixture(scope="session")
def run_module() -> ModuleType:
    """This study's ``code/run.py``, loaded by file path under a name no other study uses.

    Every study has a ``run.py``; a bare ``import run`` picks up whichever one is first in
    ``sys.modules`` when the study tier collects all studies together.
    """
    path = study_dir(__file__) / "code" / "run.py"
    spec = importlib.util.spec_from_file_location(RUN_MODULE_NAME, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[RUN_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
