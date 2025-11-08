import pandas as pd
import numpy as np

from ggTrader.Signals import Signals
from ggTrader.Backtest import Backtest

def make_df(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="4h", tz="UTC")
    df = pd.DataFrame({
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": 1.0
    }, index=idx)
    return df

def inject_signals(df, entries, exits, ce_l=None):
    # entries/exits are boolean lists aligned to df
    s = df.copy()
    s["filtered_entry"] = entries
    s["filtered_exit"]  = exits
    # build in_position from entries/exits
    in_pos = []
    pos = False
    for e, x in zip(entries, exits):
        if not pos and e: pos = True
        elif pos and x:   pos = False
        in_pos.append(pos)
    s["in_position"] = in_pos
    # ce_l exit level (use close if not provided)
    s["ce_l"] = df["close"] if ce_l is None else ce_l
    return s

def approx(a, b, atol=1e-12, rtol=1e-9):
    # Reliable float comparison tolerant to tiny FP noise
    return np.allclose(a, b, atol=atol, rtol=rtol)

# Case A: Win trade: enter at t1 close=100, exit at t3 ce_l(next)=120
prices = [95, 100, 110, 120, 120]
df = make_df(prices)
entries = [False, True,  False, False, False]  # enter at t1 close 100
exits   = [False, False, False, True,  False]  # exit at t3 using ce_l at t4 (or next)
signals = inject_signals(df, entries, exits, ce_l=df["close"])  # ce_l equals close for simplicity

bt = Backtest(signals, interval="4h", transaction_fee=0.0, start_equity=1000.0)
stats, pf = bt.run()

# Expected: position spans t1..t3 with multiplicative returns 100->110->120 => 20% total
expected_total = 0.20
observed_total = (pf["Equity"].iloc[-1] / pf["Equity"].iloc[0]) - 1.0
print("Case A total:", observed_total, "OK?", approx(observed_total, expected_total))

# Case B: Loss trade: 100->95 (−5%)
prices = [100, 95, 95]
df = make_df(prices)
entries = [True,  False, False]
exits   = [False, True,  False]
signals = inject_signals(df, entries, exits, ce_l=df["close"])

bt = Backtest(signals, interval="4h", transaction_fee=0.0, start_equity=1000.0)
stats, pf = bt.run()
expected_total = -0.05
observed_total = (pf["Equity"].iloc[-1] / pf["Equity"].iloc[0]) - 1.0
print("Case B total:", observed_total, "OK?", approx(observed_total, expected_total))

# Case C: Two trades with fees
prices = [100, 102, 104, 103, 101, 100, 99]
df = make_df(prices)
entries = [True,  False, False, False, True,  False, False]
exits   = [False, True,  False, False, False, True,  False]
signals = inject_signals(df, entries, exits, ce_l=df["close"])

fee = 0.004
bt = Backtest(signals, interval="4h", transaction_fee=fee, start_equity=1000.0)
stats, pf = bt.run()

# Manually compute expected:
# Trade1: 100->102 with entry+exit fee: gross 1.02, fees factor=(1-fee)^2
# Trade2: 101->100 with fees: gross 100/101, fees factor=(1-fee)^2
gross1 = 102/100
gross2 = 100/101
fee_factor = (1-fee)**2
expected_total_factor = gross1*fee_factor * gross2*fee_factor
expected_total = expected_total_factor - 1.0  # equals ~0.03584

observed_total = (pf["Equity"].iloc[-1] / pf["Equity"].iloc[0]) - 1.0
print("Case C total:", observed_total, "OK?", approx(observed_total, expected_total))