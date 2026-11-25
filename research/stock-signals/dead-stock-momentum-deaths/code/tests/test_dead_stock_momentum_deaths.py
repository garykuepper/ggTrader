"""Tests for the dead-stock-momentum-deaths study: write the failing test first."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.study

MODULE = importlib.import_module("dead_stock_momentum_deaths")


def test_run_study_is_defined() -> None:
    assert callable(MODULE.run_study)
