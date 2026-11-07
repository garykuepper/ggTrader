"""``ggt lab``: the parser, the mode and strategy guards, and the removed flags."""

import inspect

import pytest

import ggTrader.lab.blend as blend_mod
from ggTrader.lab.cli import _parse_blend_sleeves, build_arg_parser, run_lab
from ggTrader.lab.strategies import registry


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args(["--strategy", "xs_momentum", "--wfo"])
    assert args.strategy == "xs_momentum"
    assert args.wfo and args.blend is None
    assert args.market == "equity"
    assert args.universe == "sp500"
    assert args.top_n == 50
    assert args.snapshot is None and args.result_json is None


def test_a_mode_is_required():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--strategy", "ensemble"])


def test_blend_is_mutually_exclusive_with_wfo():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["--strategy", "ensemble", "--wfo", "--blend", "ensemble@sp500"]
        )


@pytest.mark.parametrize("flag", [["--engine", "vbt"], ["--sweep"], ["--max-sector-count", "2"]])
def test_the_removed_flags_are_gone(flag):
    """--engine (the vectorbt engine is the only path), --sweep and --max-sector-count."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--strategy", "ensemble", "--wfo", *flag])


def test_max_leverage_defaults_to_the_deployed_value():
    assert build_arg_parser().parse_args(["--wfo"]).max_leverage == 1.0
    assert inspect.signature(blend_mod.run_blend).parameters["max_leverage"].default == 1.0
    assert inspect.signature(blend_mod.blend_curves).parameters["max_leverage"].default == 1.0


def test_wfo_without_a_strategy_exits():
    with pytest.raises(SystemExit, match="--strategy"):
        run_lab(["--wfo"])


def test_a_strategy_that_is_not_ported_names_the_legacy_tag():
    with pytest.raises(SystemExit, match="legacy-engine-vbt0285"):
        run_lab(["--strategy", "pead", "--wfo", "--eval-end", "2022-07-31"])


def test_an_unknown_strategy_lists_the_ported_ones():
    with pytest.raises(SystemExit, match="ensemble"):
        run_lab(["--strategy", "bogus", "--wfo"])


def test_every_legacy_only_name_is_outside_the_registry_and_there_are_28():
    assert len(registry.LEGACY_ONLY_STRATEGIES) == 28
    assert not set(registry.LEGACY_ONLY_STRATEGIES) & set(registry.REGISTRY)
    assert {"ensemble", "xs_momentum", "cross_asset_sleeve"} <= set(registry.REGISTRY)


def test_parse_blend_sleeves_ok():
    assert _parse_blend_sleeves("ensemble@sp500, xs_momentum@nasdaq100") == [
        ("ensemble", "sp500"),
        ("xs_momentum", "nasdaq100"),
    ]


def test_parse_blend_sleeves_unknown_strategy():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("nope@sp500")


def test_parse_blend_sleeves_unported_strategy_names_the_tag():
    with pytest.raises(SystemExit, match="legacy-engine-vbt0285"):
        _parse_blend_sleeves("pead@sp500")


def test_parse_blend_sleeves_unknown_universe():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("ensemble@mars")


def test_parse_blend_sleeves_bad_format():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("ensemble_sp500")


def test_arg_parser_market_crypto():
    args = build_arg_parser().parse_args(
        ["--strategy", "xs_momentum", "--wfo", "--market", "crypto"]
    )
    assert args.market == "crypto"
