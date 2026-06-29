import pytest

from ggTrader.lab.cli import _parse_blend_sleeves, build_arg_parser


def test_arg_parser_defaults():
    p = build_arg_parser()
    args = p.parse_args(["--strategy", "xs_momentum"])
    assert args.strategy == "xs_momentum"
    assert args.market == "equity"
    assert args.top_n == 50


def test_arg_parser_rejects_unknown_strategy():
    p = build_arg_parser()
    try:
        p.parse_args(["--strategy", "bogus"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_parse_blend_sleeves_ok():
    assert _parse_blend_sleeves("ensemble@sp500, xs_momentum@nasdaq100") == [
        ("ensemble", "sp500"),
        ("xs_momentum", "nasdaq100"),
    ]


def test_parse_blend_sleeves_unknown_strategy():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("nope@sp500")


def test_parse_blend_sleeves_unknown_universe():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("ensemble@mars")


def test_parse_blend_sleeves_bad_format():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("ensemble_sp500")


def test_blend_is_mutually_exclusive_with_wfo():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--strategy", "ensemble", "--wfo", "--blend", "ensemble@sp500"])
