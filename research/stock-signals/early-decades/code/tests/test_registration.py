"""Check that the registered brief contains the study's fixed parameters."""

import tomllib

import pytest

from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study


def test_brief_registers_the_actual_study() -> None:
    study = study_dir(__file__)
    card = tomllib.loads((study / "study.toml").read_text())
    text = (study / card["brief"]).read_text()
    assert "TODO" not in text
    for value in (
        "eval_start: 1969-02-03",
        "eval_end: 1998-07-31",
        "family_size: 18",
        "0.183%",
        "0.5%",
    ):
        assert value in text
    assert "TODO" not in card["question"]
