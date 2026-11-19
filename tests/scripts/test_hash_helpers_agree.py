"""File hash helpers produce the same digest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ggTrader.data.membership_export import file_sha256
from ggTrader.data.snapshot import sha256_file
from ggTrader.utils.manifest import sha256_file as manifest_sha256_file

ROOT = Path(__file__).resolve().parents[2]


def _study_sha256_file():
    spec = importlib.util.spec_from_file_location("study_script", ROOT / "scripts/study.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.sha256_file


@pytest.mark.parametrize("size", [0, 1, 3 * 1024 * 1024])
def test_file_hash_helpers_agree(tmp_path: Path, size: int) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"x" * size)
    functions = (sha256_file, file_sha256, manifest_sha256_file, _study_sha256_file())
    assert len({function(path) for function in functions}) == 1
