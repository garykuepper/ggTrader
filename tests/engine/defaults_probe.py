"""Dump vectorbt's DEFAULT parameters as JSON; run in a clean interpreter (no ggTrader import).

``python tests/engine/defaults_probe.py`` prints the defaults of every vectorbt API ggTrader
calls, as upstream ships them, before ``ggTrader.engine`` overrides any setting. The tripwire
test compares this to a checked-in golden, so an upgrade lists each changed default.
"""

from __future__ import annotations

import inspect
import json
import re
import sys

import vectorbt as vbt


def _clean(value: object) -> object:
    """Stable text for a default: strip memory addresses and unwrap vectorbt's Default."""
    inner = getattr(value, "value", value) if type(value).__name__ == "Default" else value
    return re.sub(r" at 0x[0-9a-f]+", "", repr(inner))


def signature_defaults(func: object) -> dict[str, object]:
    """name -> default for every parameter with a default (``<required>`` otherwise)."""
    out = {}
    for name, param in inspect.signature(func).parameters.items():
        if name in ("kwargs", "args", "self"):
            continue
        out[name] = (
            "<required>" if param.default is inspect.Parameter.empty else _clean(param.default)
        )
    return out


def collect() -> dict[str, object]:
    from vectorbt.generic.accessors import GenericAccessor
    from vectorbt.returns.accessors import ReturnsAccessor

    apis = {
        "RSI.run": vbt.RSI.run,
        "MA.run": vbt.MA.run,
        "MSTD.run": vbt.MSTD.run,
        "BBANDS.run": vbt.BBANDS.run,
        "MACD.run": vbt.MACD.run,
        "ATR.run": vbt.ATR.run,
        "Portfolio.from_signals": vbt.Portfolio.from_signals,
        "Portfolio.from_orders": vbt.Portfolio.from_orders,
    }
    for name in (
        "sharpe_ratio",
        "sortino_ratio",
        "annualized",
        "annualized_volatility",
        "max_drawdown",
        "deflated_sharpe_ratio",
        "calmar_ratio",
    ):
        apis[f"ReturnsAccessor.{name}"] = getattr(ReturnsAccessor, name)
    for name in ("crossed_above", "crossed_below", "fshift", "range_split", "pct_change"):
        apis[f"GenericAccessor.{name}"] = getattr(GenericAccessor, name)
    returns = {k: _clean(v) for k, v in vbt.settings["returns"].items() if k != "stats"}
    returns["defaults"] = {k: _clean(v) for k, v in vbt.settings["returns"]["defaults"].items()}
    return {
        "vectorbt_version": vbt.__version__,
        "signatures": {name: signature_defaults(func) for name, func in apis.items()},
        "settings": {
            "engine": _clean(vbt.settings["engine"]),
            "returns": returns,
            "array_wrapper_freq": _clean(vbt.settings["array_wrapper"]["freq"]),
            "portfolio": {
                k: _clean(vbt.settings["portfolio"][k])
                for k in ("init_cash", "cash_sharing", "call_seq", "fees", "slippage", "freq")
                if k in vbt.settings["portfolio"]
            },
        },
    }


if __name__ == "__main__":
    json.dump(collect(), sys.stdout, indent=1, sort_keys=True)
