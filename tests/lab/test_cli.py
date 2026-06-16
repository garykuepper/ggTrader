from ggTrader.lab.cli import build_arg_parser


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
