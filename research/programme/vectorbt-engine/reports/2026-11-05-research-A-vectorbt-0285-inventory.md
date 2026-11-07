# vectorbt 0.28.5 (installed) feature inventory

Source root: `/home/flynn/ggTrader/.venv/lib/python3.12/site-packages/vectorbt/` (abbreviated `VBT/` below).
Version confirmed by `importlib.metadata.version('vectorbt')` and `vbt.__version__` = `0.28.5`.
Every item cites file:line from source read in this session. Items marked **UNVERIFIED** were not exercised.

Optional dependencies present in the venv (checked with `importlib.util.find_spec`): numba, plotly, yfinance, ccxt, alpaca, pandas_ta, ta.
Absent: talib, quantstats, ray, dask, binance, telegram (python-telegram-bot), alpaca_trade_api.
Importing `vbt` prints `[i] Requires TA-Lib to use xsidegap3methods` (TA-Lib absent; harmless notice).

---

## 0. Top-level exports (`vbt.*`)

Source: `VBT/__init__.py` and sub-package `__init__.py` re-exports. Full `dir(vbt)` public list was obtained by introspection.

| Group | Names |
|---|---|
| Indicators | `MA`, `MSTD`, `BBANDS`, `RSI`, `STOCH`, `MACD`, `ATR`, `OBV`, `talib`, `pandas_ta`, `ta` (functions) |
| Label indicators | `FMEAN`, `FSTD`, `FMIN`, `FMAX`, `FIXLB`, `MEANLB`, `LEXLB`, `TRENDLB`, `BOLB` |
| Signal generators | `RAND`, `RANDX`, `RANDNX`, `RPROB`, `RPROBX`, `RPROBCX`, `RPROBNX`, `STX`, `STCX`, `OHLCSTX`, `OHLCSTCX` |
| Portfolio / records | `Portfolio`, `Orders`, `Logs`, `Trades`, `EntryTrades`, `ExitTrades`, `Positions`, `Drawdowns`, `Ranges`, `Records`, `MappedArray` |
| Splitters | `RangeSplitter`, `RollingSplitter`, `ExpandingSplitter` |
| Data | `Data`, `YFData`, `BinanceData`, `CCXTData`, `AlpacaData`, `SyntheticData`, `GBMData`, `DataUpdater`, `symbol_dict` |
| Factories / base | `IndicatorFactory`, `SignalFactory`, `ArrayWrapper`, `Config`, `Configured`, `atomic_dict`, `AtomicConfig` |
| Utilities | `settings`, `set_seed`, `cached_method`, `cached_property`, `make_figure`, `make_subplots`, `merge_dicts`, `deep_substitute`, `Rep`, `RepEval`, `RepFunc`, `Sub`, `symbol_dict`, `save_animation` |
| Async / schedule | `AsyncJob`, `AsyncScheduler`, `ScheduleManager`, `CancelledError` |
| Messaging | `messaging` (module; `TelegramBot` at `VBT/messaging/telegram.py:83`) |
| Accessors | `ohlcv_accessors`, `px_accessors`, `root_accessors` (`.vbt` accessor namespace on pandas objects) |

**Absent (verified by grep across all `*.py`):**
- `VWAP` (only a column name is dropped in `VBT/data/custom.py:914`).
- `OLS` / any regression indicator.
- Any time-based stop (no `time_stop` / `max_hold` in source).
- Any engine switch (see section 9).

---

## 1. Public classes/functions for backtesting research (by subpackage)

### 1a. `base/` (pandas/numpy plumbing)

| Name | Import | Purpose | Source |
|---|---|---|---|
| `ArrayWrapper` | `vbt.ArrayWrapper` | Holds index/columns, handles column grouping | `VBT/base/array_wrapper.py:109` |
| `Wrapping` | internal | Base for objects that carry a wrapper | `VBT/base/array_wrapper.py:687` |
| `broadcast`, `broadcast_to`, `to_1d`, `to_2d`, `repeat`, `tile` | `vbt.base.reshape_fns` | Broadcast/reshape arrays with pandas index alignment | `VBT/base/reshape_fns.py:332` (broadcast), :634 (broadcast_to), :30-138 |
| `align_indexes`, `combine_indexes`, `stack_indexes` | `vbt.base.index_fns` | Index alignment helpers; raise `ValueError` when index cannot be aligned (observed in smoke test) | `VBT/base/index_fns.py:313`, :136, :104 |
| `ColumnGrouper` | `vbt.base.column_grouper` | Group-by logic | `VBT/base/column_grouper.py:98` |
| `BaseAccessor`, `BaseSRAccessor`, `BaseDFAccessor` | `vbt.base.accessors` | `.vbt` accessor base classes | `VBT/base/accessors.py:83`, :707, :726 |
| `Loc`, `iLoc`, `ParamLoc` | `vbt.base.indexing` | pandas-style and parameter indexers | `VBT/base/indexing.py:65`, :57, :160 |
| `combine_multiple`, `apply_and_concat_one`, `ray_apply` | `vbt.base.combine_fns` | Column-wise apply/concat; ray variant | `VBT/base/combine_fns.py:186`, :47, :210 |

### 1b. `generic/` (cross-sectional/time-series accessor on pandas)

| Name | Import | Purpose | Source |
|---|---|---|---|
| `GenericAccessor` (`.vbt` on Series/DataFrame) | `df.vbt.*` | rolling/expanding/ewm stats, apply, groupby, drawdown, ranges, split, plotting | `VBT/generic/accessors.py` |
| `rolling_std`, `expanding_std`, `ewm_mean`, `ewm_std` | `.vbt` | | `VBT/generic/accessors.py:404, 410, 416, 422` |
| `rolling_apply`, `expanding_apply`, `apply_along_axis`, `groupby_apply`, `resample_apply`, `applymap`, `apply_and_reduce`, `reduce` | `.vbt` | Custom Numba/pandas apply | `VBT/generic/accessors.py:441, 476, 428, 511, 550, 597, 639, 666` |
| `min/max/mean/median/std/sum/count/idxmin/idxmax/describe/value_counts` | `.vbt` | | `VBT/generic/accessors.py:780-954` |
| `crossed_above`, `crossed_below` | `.vbt` | Crossover signals | `VBT/generic/accessors.py:1225, 1263` |
| `zscore`, `rebase`, `to_returns`, `transform`, `fit_transform` | `.vbt` | | `VBT/generic/accessors.py:1320, 1324, 1219, 1277`, :277 |
| `drawdown`, `ranges`, `drawdowns`, `to_mapped` | `.vbt` | | `VBT/generic/accessors.py:1168, 1174, 1186, 1197` |
| `Ranges` | `vbt.Ranges` | Records of (start,end) ranges | `VBT/generic/ranges.py:192` |
| `Drawdowns` | `vbt.Drawdowns` | Records of drawdown episodes | `VBT/generic/drawdowns.py:251` |
| `StatsBuilderMixin`, `PlotsBuilderMixin` | internal | Build `stats()` / `plot()` from `_metrics` / `_subplots` configs | `VBT/generic/stats_builder.py`, `VBT/generic/plots_builder.py` |
| Heatmaps / plots: `heatmap`, `ts_heatmap`, `qqplot`, `plot_against`, `overlay_with_heatmap` | `.vbt` | Plotly-based | `VBT/generic/accessors.py:1995, 2152, 2312, 1807, 1945` |

### 1c. `records/`

| Name | Import | Source |
|---|---|---|
| `Records` (base class for order/trade-like record arrays) | `vbt.Records` | `VBT/records/base.py:444` |
| `MappedArray` | `vbt.MappedArray` | `VBT/records/mapped_array.py:405` |
| `ColumnMapper` | internal | `VBT/records/col_mapper.py` |

### 1d. `labels/` (forward-looking targets)

Source `VBT/labels/generators.py`. Their `nb` functions compute over future bars: `FMEAN` is `future_mean_apply_nb` with `wait=1` default (`generators.py:16-30`). These are **target/label generators**, not causal features. Do not feed them into signals that trade at the same bar.

| Name | Params (inputs/params/outputs, introspected) | Line |
|---|---|---|
| `FMEAN` | close / window, ewm / fmean | :16 |
| `FSTD` | close / window, ewm / fstd | :32 |
| `FMIN`, `FMAX` | close / window / fmin, fmax | :49, :63 |
| `FIXLB` | close / n / labels | :90 |
| `MEANLB` | close / window, ewm / labels | :110 |
| `LEXLB` | close / pos_th, neg_th / labels | :134 |
| `TRENDLB` | close / pos_th, neg_th, mode / labels | :159 |
| `BOLB` | close / window, pos_th, neg_th / labels | :186 |

### 1e. `indicators/basic.py` (built-in indicators, all via `IndicatorFactory`)

Introspected input / param / output names:

| Indicator | Import | Inputs | Params | Outputs | Line |
|---|---|---|---|---|---|
| `MA` | `vbt.MA` | close | window, ewm | ma | `VBT/indicators/basic.py:76` |
| `MSTD` | `vbt.MSTD` | close | window, ewm | mstd | :166 |
| `BBANDS` | `vbt.BBANDS` | close | window, ewm, alpha | middle, upper, lower | :236 |
| `RSI` | `vbt.RSI` | close | window, ewm | rsi | :363 |
| `STOCH` | `vbt.STOCH` | high, low, close | k_window, d_window, d_ewm | percent_k, percent_d | :461 |
| `MACD` | `vbt.MACD` | close | fast_window, slow_window, signal_window, macd_ewm, signal_ewm | macd, signal | :575 |
| `ATR` | `vbt.ATR` | high, low, close | window, ewm | tr, atr | :689 |
| `OBV` | `vbt.OBV` | close, volume | none | obv | :776 |

**Not present:** `VWAP`, `OLS`, `ADX`/`MFI`/etc. are not built in; they are only reachable through the `ta` / `pandas_ta` / `talib` wrappers (section 2).

### 1f. `portfolio/` (see sections 5 and 8 for detail)

`Portfolio` at `VBT/portfolio/base.py:1496`; `Orders` `VBT/portfolio/orders.py:180`; `Logs` `VBT/portfolio/logs.py:266`; `Trades` `VBT/portfolio/trades.py:607`; `EntryTrades` :1488, `ExitTrades` :1534, `Positions` :1584.

### 1g. `returns/` (section 6)

`ReturnsAccessor` in `VBT/returns/accessors.py`; numba kernels in `VBT/returns/nb.py`; QuantStats bridge `VBT/returns/qs_adapter.py`; metric formulas in `VBT/returns/metrics.py`.

### 1h. `data/` (section 7)

`Data` `VBT/data/base.py:291`; `YFData`, `BinanceData`, `CCXTData`, `AlpacaData`, `SyntheticData`, `GBMData` in `VBT/data/custom.py` (see section 7).

### 1i. `utils/`, `messaging/`, `ohlcv_accessors.py`, `root_accessors.py`

- `OHLCVDFAccessor` `VBT/ohlcv_accessors.py:106`: `open/high/low/close/volume/ohlc` properties (:133-169) and `.plot()` (:268), `.stats()` (:263 metrics). Candle-style plot only.
- `Vbt_DFAccessor` / `Vbt_SRAccessor` `VBT/root_accessors.py:119, 108` are the `.vbt` namespace roots.
- `PXAccessor` (`VBT/px_accessors.py:70`) wraps plotly express helpers.

---

## 2. Indicator factory

### 2a. `IndicatorFactory` (`VBT/indicators/factory.py:2209`)

Creates an indicator class from a calculation function. Named constructors:

| Method | Purpose | Source |
|---|---|---|
| `IndicatorFactory(class_name, input_names, param_names, output_names, ...)` | Base constructor | `factory.py:2209` |
| `from_custom_func(custom_func, ...)` | Full control: you handle caching/concat per param column | `factory.py:2616` |
| `from_apply_func(apply_func_nb, ...)` | Numba-friendly per-column apply; supports `use_ray` (`factory.py:3273`) | `factory.py:3096` |
| `from_talib(func_name, ...)` | Wrap a TA-Lib function. **Needs TA-Lib, which is NOT installed here (UNVERIFIED at runtime).** | `factory.py:3387` |
| `from_pandas_ta(func_name, ...)` | Wrap a `pandas_ta` function (installed; 147 indicators listed by `get_pandas_ta_indicators`). | `factory.py:3605`, list at :3584 |
| `from_ta(cls_name, ...)` | Wrap a `ta` library indicator (installed; 43 listed by `get_ta_indicators`). | `factory.py:3820`, list at :3753 |
| Shortcuts | `vbt.talib`, `vbt.pandas_ta`, `vbt.ta` are functions delegating to the above | `VBT/indicators/__init__.py:23, 28, 33` |

### 2b. Running an indicator and sweeping parameters

- `IndicatorBase.run(...)` (classmethod) `factory.py:2194` -> `_run`.
- `IndicatorBase.run_combs(...)` (classmethod) `factory.py:2204` -> `_run_combs`, generated per indicator in `from_custom_func` (`factory.py:2980-3060`).
- Parameters are **broadcast by default (zipped/aligned)**; pass `param_product=True` to take the Cartesian product. Cite: `run_pipeline` signature has `param_product: bool = False` (`factory.py:1375`; doc at :1428); the Cartesian product is built by `create_param_product` (`VBT/utils/params.py:90`). `prepare_params` at `factory.py:1213`.
- `run_combs` defaults: `param_product=False`, `comb_func=itertools.combinations` (`factory.py:2980-2981`). Passing `comb_func=itertools.product` is special-cased (`factory.py:3049`). `r` is the combination size.
- Parameter broadcasting helpers: `broadcast_params` (`VBT/utils/params.py:73`), `to_typed_list` (:15).
- `.vbt` indexing: `ParamLoc` (`VBT/base/indexing.py:160`) allows `ind.loc[...]` on parameter tuples.
- Smoke-tested: `vbt.MA.run(close, window=10)` returns columns `(10,'A'), (10,'B')`; `vbt.MA.run(close, window=[10,20])` gives a 2-level column MultiIndex (verified).

### 2c. Signal-producing indicators from `SignalFactory`

`SignalFactory(IndicatorFactory)` at `VBT/signals/factory.py:33`; `from_choice_func` at :165 (selects one of several signal sets). Used by all generators in section 4.

---

## 3. Splitters / walk-forward machinery

Source: `VBT/generic/splitters.py`.

| Item | Import | Line | Notes |
|---|---|---|---|
| `split_ranges_into_sets(start_idxs, end_idxs, set_lens, left_to_right)` | `vbt.generic.splitters.split_ranges_into_sets` | :21 | Low-level: splits given ranges into sub-sets by fractional lengths |
| `SplitterT` (Protocol) | internal | :96 | Interface: `split(X, **kwargs)` |
| `BaseSplitter` | internal | :101 | Base |
| `RangeSplitter` | `vbt.RangeSplitter` | :108 | `split(X, n=None, range_len=None, min_len=1, start_idxs=None, end_idxs=None)`; generates fixed-length windows |
| `RollingSplitter` | `vbt.RollingSplitter` | :181 | "Rolling walk-forward splitter" (docstring :182). `split(X, n=None, window_len=None, min_len=1)` |
| `ExpandingSplitter` | `vbt.ExpandingSplitter` | :228 | "Expanding walk-forward splitter" (docstring :229). `split(X, n=None, min_len=1)` |

Accessor methods on `.vbt` (`VBT/generic/accessors.py`):

| Method | Line | Purpose |
|---|---|---|
| `split(splitter, ...)` | :1335 | Apply any splitter; optional plot/heatmap |
| `range_split(**kwargs)` | :1454 | Shortcut over `RangeSplitter` |
| `rolling_split(**kwargs)` | :1538 | Shortcut over `RollingSplitter` |
| `expanding_split(**kwargs)` | :1566 | Shortcut over `ExpandingSplitter` |

**Verified behaviour (smoke test, 300 business days, `n=4`):** `RollingSplitter(n=4, window_len=100)`, `ExpandingSplitter(n=4)`, `RangeSplitter(n=4, range_len=0.3)` each produced 4 folds. `close.vbt.rolling_split(n=3, window_len=100)` returned a 2-tuple (split output form).

**Absent:** no purged / embargoed / combinatorial (CPCV) splitter; the grep for `purg|embargo|combinatorial` only matched docstrings (`factory.py:133`, `splitters.py:182, 229`). Anchored walk-forward with purge gap must be hand-written on top of the index ranges.

**Walk-forward optimisation helpers:** the `IndicatorBase.run_combs` + `Portfolio.from_signals` combination is the closest built-in path to a grid sweep; there is no built-in fold-level optimiser that picks parameters inside each training window. That selection step must be written by hand.

---

## 4. Signal generators and stops

### 4a. Generator indicators (`VBT/signals/generators.py`, each a `SignalFactory`)

Introspected inputs / params / outputs:

| Name | Import | Inputs | Params | Outputs | Factory line |
|---|---|---|---|---|---|
| `RAND` | `vbt.RAND` | none | n | entries | :25 |
| `RANDX` | `vbt.RANDX` | entries | none | exits | :95 |
| `RANDNX` | `vbt.RANDNX` | none | n | entries, exits | :141 |
| `RPROB` | `vbt.RPROB` | none | prob | entries | :204 |
| `RPROBX` | `vbt.RPROBX` | entries | prob | exits | :298 |
| `RPROBCX` | `vbt.RPROBCX` | entries | prob | new_entries, exits | :316 |
| `RPROBNX` | `vbt.RPROBNX` | none | entry_prob, exit_prob | entries, exits | :341 |
| `STX` | `vbt.STX` | entries, ts | stop, trailing | exits | :475 |
| `STCX` | `vbt.STCX` | entries, ts | stop, trailing | new_entries, exits | :495 |
| `OHLCSTX` | `vbt.OHLCSTX` | entries, open, high, low, close | sl_stop, sl_trail, tp_stop, reverse | exits | :572 |
| `OHLCSTCX` | `vbt.OHLCSTCX` | entries, open, high, low, close | sl_stop, sl_trail, tp_stop, reverse | new_entries, exits | :768 |

`_bind_ohlcstx_plot` at `generators.py:579` binds a plot method for `OHLCSTX`.
Stop implementations in Numba: `VBT/signals/nb.py` (not individually inventoried here).

### 4b. Signal accessor (`VBT/signals/accessors.py`)

Boolean-frame methods on `.vbt` (line numbers from `grep`):

| Method | Line | Purpose |
|---|---|---|
| `generate(...)` | :261 | Generate signals from a custom function |
| `generate_both(...)` | :304 | Entries+exits in one pass |
| `generate_exits(...)` | :388 | Exits given entries |
| `clean(...)` | :431 | Remove consecutive duplicate signals (entry/exit de-duplication) |
| `generate_random(...)` / `generate_random_both` / `generate_random_exits` | :467, :538, :636 | Random signal generation |
| `generate_stop_exits(...)` | :694 | Stop-loss/take-profit exits |
| `generate_ohlc_stop_exits(...)` | :789 | OHLC-based stop exits (intrabar) |
| `between_ranges`, `partition_ranges`, `between_partition_ranges` | :1046, :1139, :1163 | Range extraction |
| `rank`, `pos_rank`, `partition_pos_rank`, `pos_rank_mapped`, `partition_pos_rank_mapped` | :1186, :1236, :1286, :1342, :1348 | Rank signals |
| `first`, `nth`, `from_nth`, `nth_index` | :1327, :1332, :1337, :1356 | Pick the n-th signal |
| `norm_avg_index`, `index_mapped` | :1398, :1440 | Index statistics |
| `total`, `rate`, `total_partitions`, `partition_rate` | :1457, :1463, :1471, :1477 | Count/rate of signals |
| `AND`, `OR`, `XOR` | :1487, :1495, :1517 | Boolean combine |
| `bshift`, `fshift` | :240, :244 | Shift signals back/forward with fill |
| `plot_as_markers`, `plot_as_entry_markers`, `plot_as_exit_markers` | :1719, :1759, :1782 | Plot helpers |

### 4c. Stop options available on `Portfolio.from_signals`

From the introspected signature (`VBT/portfolio/base.py:2022`):
`sl_stop`, `sl_trail`, `tp_stop`, `stop_entry_price`, `stop_exit_price`, `upon_stop_exit`, `upon_stop_update`, `adjust_sl_func_nb`, `adjust_sl_args`, `adjust_tp_func_nb`, `adjust_tp_args`, `use_stops`, plus OHLC-based `open/high/low` inputs (for intrabar stop checks).
Stop enums: `StopEntryPriceT` `VBT/portfolio/enums.py:236`, `StopExitPriceT` :262, `StopExitModeT` :312, `StopUpdateModeT` :339.

**Absent:** any time-based exit (no `time_stop` in source).

---

## 5. Portfolio

### 5a. Constructors (`VBT/portfolio/base.py`, class at :1496)

| Constructor | Line | Params (introspected) |
|---|---|---|
| `Portfolio.from_orders` | :1601 | close, size, size_type, direction, price, fees, fixed_fees, slippage, min_size, max_size, size_granularity, reject_prob, lock_cash, allow_partial, raise_reject, log, val_price, init_cash, cash_sharing, call_seq, ffill_val_price, update_value, max_orders, max_logs, seed, group_by, broadcast_kwargs, wrapper_kwargs, freq, attach_call_seq (30) |
| `Portfolio.from_signals` | :2022 | close, entries, exits, short_entries, short_exits, signal_func_nb, signal_args, size, size_type, price, fees, fixed_fees, slippage, min_size, max_size, size_granularity, reject_prob, lock_cash, allow_partial, raise_reject, log, accumulate, upon_long_conflict, upon_short_conflict, upon_dir_conflict, upon_opposite_entry, direction, val_price, open, high, low, sl_stop, sl_trail, tp_stop, stop_entry_price, stop_exit_price, upon_stop_exit, upon_stop_update, adjust_sl_func_nb, adjust_sl_args, adjust_tp_func_nb, adjust_tp_args, use_stops, init_cash, cash_sharing, call_seq, ffill_val_price, update_value, max_orders, max_logs, seed, group_by, broadcast_named_args, broadcast_kwargs, template_mapping, wrapper_kwargs, freq, attach_call_seq (58) |
| `Portfolio.from_order_func` | :3145 | close, order_func_nb, flexible, init_cash, cash_sharing, call_seq, segment_mask, call_pre_segment, call_post_segment, pre/post_sim/group/row/segment/order function hooks + args, ffill_val_price, update_value, fill_pos_record, row_wise, **use_numba**, max_orders, max_logs, seed, group_by, ... (42) |
| `Portfolio.from_holding` | :3024 | close (buy-and-hold benchmark) |
| `Portfolio.from_random_signals` | :3038 | close, n, prob, entry_prob, exit_prob, param_product, seed, run_kwargs |

`use_numba` argument: `base.py:3178`, doc `:3286`.

### 5b. Option enums (`VBT/portfolio/enums.py`)

| Enum | Members (introspected) | Line |
|---|---|---|
| `SizeType` | Amount, Value, Percent, TargetAmount, TargetValue, TargetPercent | :363 |
| `Direction` | LongOnly, ShortOnly, Both | :413 |
| `CallSeqType` | Default, Reversed, Random, Auto | :87 |
| `AccumulationMode` | Disabled, Both, AddOnly, RemoveOnly | :111 |
| `ConflictMode` | Ignore, Entry, Exit, Adjacent, Opposite | :144 |
| `OppositeEntryMode` | Ignore, Close, CloseReduce, Reverse, ReverseReduce | :208 |
| `InitCashMode` | Auto, AutoAlign | :67 |
| `StopEntryPrice` / `StopExitPrice` / `StopExitMode` / `StopUpdateMode` | see section 4c | :236, :262, :312, :339 |
| `OrderStatus` | Filled, Ignored, Rejected | :435 |
| `TradesType` | EntryTrades, ExitTrades, Positions | :558 |

`cash_sharing`, `group_by`, `call_seq`, `size_type`, `direction` are all constructor args (see 5a). Docstring hint that cash sharing with grouping is required for multi-asset portfolios is in the constructor docs (not re-verified line by line).

### 5c. Stats / metric methods

| Method | Line | Notes |
|---|---|---|
| `stats_defaults` | :4581 | Merges generic `StatsBuilderMixin` defaults with `settings['portfolio']['stats']` |
| `_metrics` (config of metric definitions) | :4601 | Keys (introspected): start, end, period, start_value, end_value, total_return, benchmark_return, max_gross_exposure, total_fees_paid, max_dd, max_dd_duration, total_trades, total_closed_trades, total_open_trades, open_trade_pnl, win_rate, best_trade, worst_trade, avg_winning_trade, avg_losing_trade, avg_winning_trade_duration, avg_losing_trade_duration, profit_factor, expectancy, sharpe_ratio, calmar_ratio, omega_ratio, sortino_ratio |
| `pf.stats()` | (generic) | Smoke-tested: returns 28 rows for a 2-column portfolio (verified) |
| `value`, `total_profit`, `final_value`, `total_return` | :4326, :4354, :4381, :4391 | |
| `returns(...)` | :4401 | Per-column returns series |
| `get_returns_acc`, `returns_acc`, `returns_stats` | :4439, :4434, :4778 | Access to `ReturnsAccessor` (section 6) |
| `get_qs`, `qs` | :4473, :4468 | QuantStats adapter (requires quantstats, **absent**; UNVERIFIED that it errors only on use) |
| `benchmark_value`, `benchmark_returns`, `total_benchmark_return` | :4495, :4517, :4528 | Buy-and-hold benchmark |
| `sharpe_ratio()`, `sortino_ratio()`, `calmar_ratio()`, `omega_ratio()` | via `ReturnsAccessor` | Smoke-tested: `pf.sharpe_ratio()` returns per-column values (verified) |
| `deflated_sharpe_ratio()` | via `pf.returns_acc` | Smoke-tested: `pf.returns_acc.deflated_sharpe_ratio()` returns values (verified) |

### 5d. Trades, positions, orders, drawdowns, logs

| Accessor | Line | Notes |
|---|---|---|
| `orders` / `get_orders` | :4033, :4038 | `Orders` (`VBT/portfolio/orders.py:180`) |
| `logs` / `get_logs` | :4050, :4055 | `Logs` (`VBT/portfolio/logs.py:266`) |
| `entry_trades` / `exit_trades` / `trades` | :4062, :4074, :4086 | `EntryTrades` (:1488), `ExitTrades` (:1534) |
| `positions` / `get_positions` | :4091, :4096 | `Positions` (:1584) |
| `drawdowns` / `get_drawdowns` | :4112, :4117 | `Drawdowns` (`VBT/generic/drawdowns.py:251`) |
| `Trades` metrics | `VBT/portfolio/trades.py` | winning :653, losing :659, winning_streak :665, losing_streak :672, win_rate :679, profit_factor :690, expectancy :707, sqn :724 |
| `Drawdowns` metrics | `VBT/generic/drawdowns.py` | avg_drawdown :321, max_drawdown :330, recovery_return :341, decline_duration :372, recovery_duration :383, active_drawdown :410, active_recovery :436 |
| Smoke-tested trade count | (verified) | `pf.trades` on 2 columns returned `ExitTrades` with closed counts 3 and 2 |

Also present: `asset_flow`, `assets`, `position_mask`, `position_coverage`, `cash_flow`, `cash`, `asset_value`, `gross_exposure`, `net_exposure` (`base.py:4129-4317`).

### 5e. Plotting on Portfolio

`plot_orders` :4804, `plot_trades` :4809, `plot_trade_pnl` :4814, `plot_positions` :4818, `plot_position_pnl` :4823, `plot_asset_flow` :4827, `plot_cash_flow` :4874, `plot_assets` :4923, `plot_cash` :4977, `plot_asset_value` :5035, `plot_value` :5091, `plot_cum_returns` :5142, `plot_drawdowns` :5192, `plot_underwater` :5216, `plot_gross_exposure` :5267, `plot_net_exposure` :5323 (Plotly-based).

---

## 6. Returns analytics (`VBT/returns/accessors.py`)

`ReturnsAccessor` (`.vbt.returns`). Methods with line numbers (from `grep`):

| Family | Methods (line) |
|---|---|
| Basics | `from_value` :233, `benchmark_rets` :254, `year_freq` :259, `ann_factor` :272, `daily` :296, `annual` :304, `cumulative` :312, `total` :322 |
| Annualised | `annualized` :341, `annualized_volatility` :360 (+ rolling variants :347, :373) |
| Ratios | `calmar_ratio` :393, `omega_ratio` :412, `sharpe_ratio` :445, `deflated_sharpe_ratio` :477, `sortino_ratio` :541, `information_ratio` :567 (each with `rolling_*` counterpart) |
| Risk | `downside_risk` :515, `beta` :601, `alpha` :629, `tail_ratio` :663, `common_sense_ratio` :682, `value_at_risk` :703, `cond_value_at_risk` :729, `capture` :755, `up_capture` :783, `down_capture` :811 |
| Drawdown | `drawdown` :839, `max_drawdown` :845, `drawdowns` :867 |
| Adapters | `qs` :879 (QuantStats, absent) |
| Metric config | `_metrics` :944 (keys: start, end, period, total_return, benchmark_return, ann_return, ann_volatility, max_dd, max_dd_duration, sharpe_ratio, calmar_ratio, omega_ratio, sortino_ratio, skew, kurtosis, tail_ratio, common_sense_ratio, value_at_risk, alpha, beta) |
| Plot | `plot_cumulative` :1136 |

**Deflated Sharpe** (`returns/accessors.py:477`): signature `(risk_free, ddof, var_sharpe, nb_trials, bias)`. `nb_trials` defaults to the number of columns, so for a single strategy tested across many parameter sets the caller **must** pass the true trial count. Smoke-tested on 2 columns: returned values in (0,1) (verified).

Numba kernels: `VBT/returns/nb.py` (e.g. `sharpe_ratio_nb` :347, `rolling_sharpe_ratio_nb` :359, `information_ratio_nb` :459, `beta_nb` :505, `value_at_risk_nb` :614).

Benchmark-dependent metrics (alpha, beta, information, capture) need `benchmark_rets` supplied (warnings at `returns/accessors.py:73-75`).

---

## 7. Data classes (`VBT/data/`)

| Class | Import | Line | Notes |
|---|---|---|---|
| `Data` | `vbt.Data` | `VBT/data/base.py:291` | Base. Methods: `download` :549, `download_symbol` :544, `update` :601, `update_symbol` :597, `concat` :667, `get` :699, `align_index` :372, `align_columns` :407, `tz_localize` :347, `tz_convert` :352, `from_data` :467, `stats_defaults` :722, `plot` :781 |
| `symbol_dict` | `vbt.symbol_dict` | `VBT/data/base.py:279` | |
| `SyntheticData` | `vbt.SyntheticData` | `VBT/data/custom.py:34` | |
| `GBMData` | `vbt.GBMData` | `VBT/data/custom.py:93` | Geometric Brownian motion simulated data |
| `YFData` | `vbt.YFData` | `VBT/data/custom.py:177` | yfinance (installed). **Use yfinance-sourced data with caution for survivorship (see ggTrader notes); this inventory does not validate it.** |
| `BinanceData` | `vbt.BinanceData` | `VBT/data/custom.py:293` | Needs `python-binance`, **absent** |
| `CCXTData` | `vbt.CCXTData` | `VBT/data/custom.py:549` | ccxt installed |
| `AlpacaData` | `vbt.AlpacaData` | `VBT/data/custom.py:765` | `alpaca` package installed; `alpaca_trade_api` absent. Which client it imports: **UNVERIFIED** |
| `DataUpdater` | `vbt.DataUpdater` | `VBT/data/updater.py:16` | `update` :26; `update_every` :137 (uses `ScheduleManager`) |

Update mechanics: `Data.update` (`base.py:601`) and `update_symbol` (:597) exist; their body was NOT read in detail (UNVERIFIED that it re-downloads from stored `download_kwargs`). `DataUpdater.update_every` (`updater.py:137`) schedules periodic updates via `ScheduleManager` (`VBT/utils/schedule_.py:64`). Scheduler uses `AsyncScheduler` (`schedule_.py:38`) and `AsyncJob` (`schedule_.py:26`).

The data classes listed above are the complete set of `Data` subclasses in `data/custom.py` (grep for `class .*Data` in that file). The package has no point-in-time index-constituent loader (not searched exhaustively; UNVERIFIED beyond that grep).

---

## 8. Utilities: parallelism, caching, settings, messaging

### 8a. Settings (`VBT/_settings.py`)

- `settings` is a `SettingsConfig` object built at `_settings.py:117`.
- Top-level keys (introspected): numba, config, configured, caching, broadcasting, array_wrapper, datetime, data, plotting, stats_builder, plots_builder, generic, ranges, drawdowns, ohlcv, signals, returns, qs_adapter, records, mapped_array, orders, trades, logs, portfolio, messaging.
- `numba` block: `_settings.py:119` (`check_func_type`, `check_func_suffix`).
- `caching` block: `_settings.py:131` (`enabled=True`; whitelist: `ArrayWrapper`, `ColumnGrouper`, `ColumnMapper`; blacklist empty).

### 8b. Caching

- `cached_method` `VBT/utils/decorators.py:397` (`maxsize=128`, `typed=False`); `cached_property` :294; `CacheCondition` :104; `should_cache` :126.
- Cached on `ArrayWrapper`, `ColumnGrouper`, `ColumnMapper` by default (see 8a).

### 8c. Parallelism

- **Ray**: `ray_apply` at `VBT/base/combine_fns.py:210`, and `apply_and_concat_one_ray`/`_multiple_ray` at :253, :259, :265. Enabled per-call with `use_ray=True` in `from_apply_func` (`VBT/indicators/factory.py:3273-3297`). Ray is **not installed**, so this path is unusable here.
- **Dask / joblib / multiprocessing / concurrent.futures**: grep for `dask`, `joblib`, `multiprocess`, `ThreadPool`, `ProcessPool`, `concurrent` returned no matches in the package. **Not present.**
- Numba `parallel` is not separately exposed; the Numba kernels are used via `use_numba` (section 9).

### 8d. Telegram bot (`VBT/messaging/telegram.py`)

`TelegramBot(Configured)` at `:83`; `LogHandler` :41; `send_message` :289; `send_to_all` :284; `send` :266; `send_giphy` :299; `start` :231; `stop` :367; `running` :373. Requires python-telegram-bot, **absent** here, so UNVERIFIED that it imports. Notifications go through `vbt.messaging`, not the ggTrader Telegram code.

### 8e. Reproducibility

- `set_seed(seed)` `VBT/utils/random_.py:18` (plus `set_seed_nb` :13 for Numba RNG).
- Portfolio and generator constructors all take `seed=` (e.g. `from_signals` `seed` arg; `RPROB.run(..., seed=...)`).

### 8f. Other utilities

- `Config` / `Configured` / `atomic_dict` (`VBT/utils/config.py`).
- `merge_dicts`, `deep_substitute`, `Rep`, `RepEval`, `RepFunc`, `Sub`: templating (`VBT/utils/template.py`).
- `make_figure`, `make_subplots`, `FigureWidget`: plotly helpers (`VBT/utils/figure.py`).
- `save_animation` (root export): animation writer.

---

## 9. "Engine" setting (numba vs rust, etc.)

**No engine switch exists in 0.28.5.**
- `grep -rn "engine|rust|dask|joblib|multiprocess|ThreadPool|ProcessPool|concurrent"` across the package found no engine-selection setting (the only "engine" hit is a docstring about giphy in `utils/requests_.py:36`).
- The compute backend is fixed: **Numba** kernels with NumPy/pandas wrappers. Numba is controlled per call, not globally:
  - `Portfolio.from_order_func(..., use_numba=...)` (`VBT/portfolio/base.py:3178`; default resolved at :3679).
  - Settings block `numba` (`_settings.py:119`) only controls `check_func_type` and `check_func_suffix`.
- Ray is an optional per-call parallel path (section 8c), not an engine.

---

## 10. Summary (10 lines, for walk-forward optimisation research)

1. Splitters that exist: `RollingSplitter` (`generic/splitters.py:181`), `ExpandingSplitter` (:228), `RangeSplitter` (:108), exposed as `vbt.*Splitter` and as `.vbt.rolling_split/expanding_split/range_split` (`generic/accessors.py:1538, 1566, 1454`). No purged/embargoed/CPCV splitter; add the purge gap by hand.
2. In-sample parameter selection inside each fold is **not** built in; you must write the selection loop.
3. Parameter sweeps: `IndicatorBase.run` / `run_combs` (`indicators/factory.py:2194, 2204`); zipped by default, `param_product=True` for the grid (`factory.py:1375`), `comb_func` for combinations (`factory.py:2981`).
4. Built-in indicators are few and standard: MA, MSTD, BBANDS, RSI, STOCH, MACD, ATR, OBV (`indicators/basic.py`). No VWAP, no OLS. Wider coverage only via `vbt.ta` (43) and `vbt.pandas_ta` (147) wrappers (`factory.py:3820, 3605`); `vbt.talib` needs TA-Lib, which is not installed.
5. Labels (`labels/generators.py`) are forward-looking. Use them as targets only, never as same-bar features.
6. Signal generators: RAND/RPROB family plus STX/STCX and OHLC intrabar stops `OHLCSTX`/`OHLCSTCX` (`signals/generators.py:475-768`); no time-based exit. Clean duplicates with `.vbt.signals.clean` (`signals/accessors.py:431`).
7. `Portfolio.from_signals` (`portfolio/base.py:2022`) covers sizing, fees, slippage, stops, conflict modes, cash sharing, and call_seq. `from_orders` (:1601) and `from_order_func` (:3145) cover custom order logic.
8. Metrics: `pf.stats()` (28 rows, smoke-tested), `ReturnsAccessor` sharpe/sortino/calmar/omega/information/beta/alpha/VaR/CVaR/capture (`returns/accessors.py:445-811`), and `deflated_sharpe_ratio` (:477). For DSR, pass `nb_trials` = the real number of parameter trials, since the default is the column count.
9. Caveats: no engine switch; Ray, Dask, TA-Lib, QuantStats, Telegram, and Binance are absent or unverified in this venv. Numba is the only live compute backend. Data classes are yfinance/ccxt/alpaca wrappers with no point-in-time index support.
10. Smoke tests on synthetic data confirmed: MA run, crossover signals, `from_signals`, `pf.stats()`, `pf.sharpe_ratio()`, `deflated_sharpe_ratio`, and all three splitters produce folds as documented.
