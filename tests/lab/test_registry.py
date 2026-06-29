"""The strategy registry is a single source of truth; names/builders derive from it."""

import pytest

from ggTrader.lab.strategies import (
    STRATEGY_REGISTRY,
    all_strategy_names,
    build_strategy,
    signal_strategy_names,
    weight_strategy_names,
)
from ggTrader.lab.strategy import LabConfig


def test_registry_keys_match_class_name_and_kind():
    for name, cls in STRATEGY_REGISTRY.items():
        assert cls.name == name, f"key {name!r} != cls.name {cls.name!r}"
        assert cls.target_kind in {"signals", "weights"}


def test_name_views_partition_the_registry():
    sig, wt = set(signal_strategy_names()), set(weight_strategy_names())
    assert sig.isdisjoint(wt)
    assert sig | wt == set(STRATEGY_REGISTRY)
    assert set(all_strategy_names()) == set(STRATEGY_REGISTRY)


def test_build_strategy_builds_every_registered_name():
    cfg = LabConfig()
    for name in all_strategy_names():
        assert build_strategy(name, cfg).name == name


def test_build_strategy_unknown_raises():
    with pytest.raises(ValueError):
        build_strategy("nope", LabConfig())
