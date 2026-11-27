"""Tests for the early-decades study: write the failing test first."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.study

MODULE = importlib.import_module("early_decades")


def test_run_study_is_defined() -> None:
    assert callable(MODULE.run_study)
