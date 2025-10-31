import pandas as pd
import vectorbt as vbt
from numba import njit
import numpy as np
from vectorbt.portfolio.enums import StopExitPrice, StopEntryPrice
from utils.KrakenHistoricalData import KrakenHistoricalData


def resample_df(df, tickers, interval='4h'):
    # Resample each ticker's data to 4-hour intervals
    resampled_data = {}
    for ticker in tickers:
        # Extract the data for the current ticker
        ticker_data = df.xs(key=ticker, axis=1, level=0)  # Selects sub-columns for the specific ticker
        # Resample the data to 4-hour intervals and aggregate OHLCV
        resampled_data[ticker] = ticker_data.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

    # Combine all resampled data into a single DataFrame with a MultiIndex
    df_4h = pd.concat(resampled_data, axis=1)
    df_4h = df_4h.interpolate(method='linear', axis=0)
    return df_4h

# Define tickers to download.  Replaced BNB with AVAX
tickers = ['BTC', 'ETH', 'AVAX', 'ADA', 'SOL', 'XRP', 'DOGE', 'LINK', 'DOT',
           'LTC']
cols = ['open', 'high', 'low', 'close', 'volume']
k = KrakenHistoricalData()

df = k.get_ohlcv_df(tickers, interval='1h')

print(df.info())
df = resample_df(df, tickers)
# Display the structure of the DataFrame
print("Original DataFrame:")
print(df.info())
print(df.head())
df_dict = {}
for col in cols:
    df_dict[col] = df.loc[:, pd.IndexSlice[:, col]].copy()  # Use pd.IndexSlice to select all tickers with 'Open'
    df_dict[col].columns = df_dict[col].columns.droplevel(1)  # Drop the "OHLCV" level, leaving only the ticker names

# Display the resulting DataFrame
print("Open Prices DataFrame:")
print(df_dict['open'].head())



# --- apply function ---
def psar_adx_apply(high: pd.Series, low: pd.Series, close: pd.Series,
                   accel: float = 0.02, maximum: float = 0.2,
                   adx_len: int = 14, adx_thr: float = 25):
    """Return PSAR, ADX, and entry signal = rising edge of (close>psar & adx>thr)."""

    psar = vbt.pandas_ta("PSAR").run(high, low, close=close, acceleration=accel, maximum=maximum).psarl
    adx = vbt.pandas_ta("ADX").run(high, low, close, timeperiod=adx_len).adx
    sar_bull = close > psar
    adx_strong = adx > adx_thr
    combo_now = sar_bull & adx_strong
    combo_prev = combo_now.shift(1)
    combo_prev = combo_prev.where(pd.notna(combo_prev), False).astype(bool)
    entries = combo_now & ~combo_prev
    return psar, adx, entries


def ce_apply(high: pd.Series, low: pd.Series, close: pd.Series,
             multiplier: float = 3.0,
             atr_length: int = 14):
    ce = vbt.pandas_ta("chandelier_exit").run(high, low, close, multiplier=multiplier,
                                              atr_length=atr_length)
    ce_levels = ce.chdlrextl
    ce_signal = ce.chdlrextl_crossed_below(low)
    return ce_levels, ce_signal


# --- register with IndicatorFactory ---
PSAR_ADX = vbt.IndicatorFactory(
    class_name="PSAR_ADX",
    input_names=["high", "low", "close"],
    param_names=["accel", "maximum", "adx_len", "adx_thr"],
    output_names=["psar", "adx", "entries"]
).from_apply_func(psar_adx_apply, keep_pd=True)
# ).from_apply_func(psar_adx_apply, keep_pd=True, param_product=True)
ChandelierExit = vbt.IndicatorFactory(
    class_name="ChandelierExit",
    input_names=["high", "low", "close"],
    param_names=["multiplier", "atr_length"],
    output_names=["ce_levels", "ce_signal"]
).from_apply_func(ce_apply, keep_pd=True)
# ).from_apply_func(ce_apply, keep_pd=True, param_product=True)



#
# run indicators
I = PSAR_ADX.run(df_dict['high'], df_dict['low'], df_dict['close'],
                 accel=0.02, maximum=0.2, adx_thr=25, adx_len=14)

E = ChandelierExit.run(df_dict['high'], df_dict['low'], df_dict['close'], multiplier=3.0, atr_length=14)


entries = I.entries
exits = E.ce_signal
exit_price = E.ce_levels

# Run the backtest
# 3) Build the portfolio using *stops*, not exits
pf = vbt.Portfolio.from_signals(
    close=df_dict['close'],
    entries=entries,
    exits=exits,  # CE exits via stop
    open=df_dict['open'],
    high=df_dict['high'],
    low=df_dict['low'],
    # stop_exit_price=StopExitPrice.StopLimit,
    direction='longonly',
    init_cash=10_000,
    fees=0.001,
    slippage=0.0005,
    cash_sharing=True,
    freq='4h'
)
print(pf.stats(group_by=True))

fig = pf.plot(group_by=True, width=1600, height=800)
fig.show()

# run indicators
# Define parameter grids
# accel_grid = [0.01, 0.02, 0.03]
# maximum_grid = [0.2, 0.3]
# adx_len_grid = [14, 20]
# adx_thr_grid = [20, 25, 30]
#
# ce_multiplier_grid = [2.0, 3.0, 4.0]
# ce_atr_len_grid = [10, 14, 20]
#
# # Vectorbt will create a parameter index for each combination
# I = PSAR_ADX.run(
#     df_dict['high'], df_dict['low'], df_dict['close'],
#     accel=accel_grid,
#     maximum=maximum_grid,
#     adx_len=adx_len_grid,
#     adx_thr=adx_thr_grid
# )
#
# E = ChandelierExit.run(
#     df_dict['high'], df_dict['low'], df_dict['close'],
#     multiplier=ce_multiplier_grid,
#     atr_length=ce_atr_len_grid
# )
#
# # Align shapes by reindexing to the cross-product of param combinations
# # This ensures entries/exits have the same param index ordering
# # Build a common parameter index
# I_idx = I.param_product  # MultiIndex of PSAR_ADX params
# E_idx = E.param_product  # MultiIndex of CE params
#
# # Cross join the two param spaces
# common_param_index = pd.MultiIndex.from_product(
#     [range(len(I_idx)), range(len(E_idx))],
#     names=["psar_adx_idx", "ce_idx"]
# )
#
# # Expand entries/exits to the common_param_index
# # entries has param index I_idx; tile it across CE params
# entries = I.entries.vbt.tile(len(E_idx))
# entries = entries.vbt.slice_by_param(common_param_index)
# # CE outputs have param index E_idx; repeat across PSAR/ADX params
# exits = E.ce_signal.vbt.repeat(len(I_idx))
# exits = exits.vbt.slice_by_param(common_param_index)
# exit_price = E.ce_levels.vbt.repeat(len(I_idx))
# exit_price = exit_price.vbt.slice_by_param(common_param_index)
#
# # Close/Open/High/Low don't depend on params, but need to broadcast
# close_b = vbt.utils.broadcasting.broadcast_to(df_dict['close'], entries.shape, to_pd=True)
# open_b = vbt.utils.broadcasting.broadcast_to(df_dict['open'], entries.shape, to_pd=True)
# high_b = vbt.utils.broadcasting.broadcast_to(df_dict['high'], entries.shape, to_pd=True)
# low_b = vbt.utils.broadcasting.broadcast_to(df_dict['low'], entries.shape, to_pd=True)
#
# # Build portfolios for every param combo per ticker
# pf = vbt.Portfolio.from_signals(
#     close=close_b,
#     entries=entries,
#     exits=exits,
#     open=open_b,
#     high=high_b,
#     low=low_b,
#     direction='longonly',
#     init_cash=10_000,
#     fees=0.001,
#     slippage=0.0005,
#     cash_sharing=True,
#     freq='4h'
# )
#
# # Get performance per param combo, grouped by ticker
# stats = pf.stats(group_by=True)
# print(stats)
#
# # Find best parameter combo by total return (per ticker)
# total_return = pf.total_return(group_by=True)
# # total_return index == common_param_index; columns == tickers
# best_idx_per_ticker = total_return.idxmax(axis=0)  # returns param index for each ticker
# print("Best param indices per ticker:")
# print(best_idx_per_ticker)
#
#
# # Map indices back to actual params
# def idx_to_params(psar_adx_idx, ce_idx):
#     pa = I_idx[psar_adx_idx]
#     ce = E_idx[ce_idx]
#     return {
#         "accel": pa[I.param_names.index("accel")],
#         "maximum": pa[I.param_names.index("maximum")],
#         "adx_len": pa[I.param_names.index("adx_len")],
#         "adx_thr": pa[I.param_names.index("adx_thr")],
#         "ce_multiplier": ce[E.param_names.index("multiplier")],
#         "ce_atr_length": ce[E.param_names.index("atr_length")],
#     }
#
#
# best_params = {}
# for ticker, tup in best_idx_per_ticker.items():
#     psar_adx_idx, ce_idx = tup
#     best_params[ticker] = idx_to_params(psar_adx_idx, ce_idx)
#
# print("Best params per ticker:")
# for t, p in best_params.items():
#     print(t, p)
#
# # Optionally plot the best run only (example for BTC)
# btc_param = best_idx_per_ticker['BTC']
# pf_btc = pf[(btc_param,)].group_by('BTC')  # slice param and ticker
# fig = pf_btc.plot(width=1600, height=800)
# fig.show()