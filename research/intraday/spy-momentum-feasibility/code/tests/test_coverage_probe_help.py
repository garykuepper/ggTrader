"""The Stage 0A coverage-probe driver prints help without credentials, network or artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study

CODE = study_dir(__file__) / "code"


def test_help_has_no_credentials_network_or_artifacts(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("APCA_")}
    result = subprocess.run(
        [sys.executable, str(CODE / "intraday_coverage_probe.py"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout
    assert list(tmp_path.iterdir()) == []
