# Adding a strategy

A strategy is one file in `src/ggTrader/lab/strategies/`. Put it there and
`ggt lab --strategy <name> --wfo` finds it. Nothing else is edited: not
`__init__.py`, not the registry, not a test list. The registry is built by
scanning the package (`registry.discover()`), and the lookahead test runs over
every registered strategy by itself.

Terms: **WFO** = walk-forward optimisation (rolling train and test windows).
**PIT** = point-in-time (which S&P 500 stocks were members on a given day).
**Lookahead** = a signal that uses data from after the day it is for.

## The contract

A strategy is a class with four class attributes and a `build` method.

| Part | What it is |
|---|---|
| `name` | Unique string. This is what `--strategy` takes. |
| `kind` | `"signals"` (entries and exits per stock) or `"weights"` (target weights on rebalance dates). |
| `grid` | A `ParamGrid`: the parameter values to try, plus validity rules such as `Lt("fast", "slow")`. |
| `live` | Signal strategies only. The deployed parameters, or `None`. Optional. |
| `build(bars, grid)` | Signal strategies return a `SignalOut`. Weight strategies take `(bars, grid, rebal, eligible)` and return a target-weight frame. |

`build` composes the presets in `ggTrader.engine` (`indicators`, `signals`). It
never imports vectorbt (ruff and a boundary test enforce this), never calls the
simulator, and never applies the PIT mask. The walk-forward does that.

`SignalOut` columns are `(grid levels..., symbol)`, one block of stocks per
parameter combination. `group_columns(combos, grid.swept(), bars.symbols)`
builds that index.

## Worked example: SMA crossover

This is `docs/examples/sma_cross.py` (about 45 lines). `tests/lab/strategies/test_docs_example.py`
runs it on every test run, so it cannot rot.

```python
class SmaCross:
    name: ClassVar[str] = "sma_cross"
    kind: ClassVar[str] = "signals"
    grid: ClassVar[ParamGrid] = ParamGrid.from_sweep(
        {"fast": [10, 20], "slow": [50, 100]}, rules=(Lt("fast", "slow"),)
    )
    live: ClassVar[dict[str, Any] | None] = None

    def build(self, bars: Bars, grid: ParamGrid | None = None) -> SignalOut:
        grid = grid or self.grid
        combos = resolve_combos(grid, {})
        close = bars.close
        entries, exits = [], []
        for combo in combos:
            fast = ind.values(ind.sma(close, combo["fast"]).ma, like=close)
            slow = ind.values(ind.sma(close, combo["slow"]).ma, like=close)
            entries.append(sig.cross_above(fast, slow))
            exits.append(sig.cross_below(fast, slow))
        columns = group_columns(combos, grid.swept(), bars.symbols)
        return SignalOut(
            entries=pd.concat(entries, axis=1).set_axis(columns, axis=1),
            exits=pd.concat(exits, axis=1).set_axis(columns, axis=1),
            combos=tuple(combos),
        )
```

Two details that trip people up:

* **`ind.values(output, like=close)`** removes the parameter level. A preset such as
  `sma(close, 20).ma` has columns `(ma_window, symbol)`; two of them do not line up for a
  crossover until that level is gone. `values` relabels the output with the columns of
  `close`. It refuses an output that holds several parameter values.
* **Crossovers are strict.** `sig.cross_above` fires on the bar where `a` goes strictly above
  `b` after being below it. A tie does not count.

## Try it

1. Copy the file: `cp docs/examples/sma_cross.py src/ggTrader/lab/strategies/sma_cross.py`.
2. Run it on the frozen tape (no database, nothing written except the result file):

   ```bash
   python ggt.py lab --strategy sma_cross --wfo \
       --eval-start 2023-01-31 --eval-end 2024-10-31 --max-stocks 40 \
       --snapshot data/snapshots/ohlcv_20261106.parquet \
       --result-json /tmp/sma_cross.json
   ```

   The first test of this example on the real tape gave OOS Sharpe 0.80, CAGR 4.2%, gates 0/3
   against SPY 2.14 and 29.3% over those three folds. It is a demonstration, not a finding.
3. Run the checks: `scripts/run_tests.sh tests/lab/strategies -q`. The lookahead test
   (`tests/lab/strategies/test_leak.py`) now includes your strategy on a reduced grid. It
   truncates and perturbs the bars after a date and fails if a signal on that date moves.

If the strategy needs more than price bars (a data table, specific symbols), add an entry for
its name to `SPECIAL` in `test_leak.py`; the test tells you when a strategy is not covered.

## What the registry checks

`discover()` imports every module in the package except the helpers (`contract`, `leak`,
`registry`, `vote`, `voters`, and any file whose name starts with an underscore). It raises
`StrategyModuleError` naming the file when a module

* cannot be imported,
* defines no class with `name`, `kind`, `grid` and `build`,
* defines a class that breaks the contract (the message lists what is missing), or
* reuses a name another file already took.

A helper module in the package must therefore start with an underscore. Order is
deterministic: modules by name, classes in definition order.

A strategy that lives outside the package (a scratch file, a study) can use the
`@register` decorator from `ggTrader.lab.strategies.registry` and then call
`run_vbt_wfo` from Python.

## Provenance and the point-in-time mask

Every result JSON from `ggt lab --wfo` carries a provenance block. It now includes
`strategy_sha256` and `strategy_files` (a SHA-256 of each source file the strategy is made of:
its module, its base classes' modules and the `lab/strategies` siblings it imports, such as
`voters.py` or `contract.py`), so an edited strategy is identified by its result. `--blend` records
`strategy_sources` per sleeve. These keys are informational: `provenance.compare` ignores
them, the same way it ignores `git_sha`.

`run_wfo` raises `ValueError` when a signal strategy gets neither `mask=` nor `universe_fn=`:
without membership data it would trade stocks on days they were not in the index (survivorship
bias). `ggt lab` always passes the PIT mask. For synthetic data or a fixed single-symbol tape,
pass `allow_unmasked=True` to say so explicitly.

## After the run

A number you will cite needs a pinned `--eval-start` and `--eval-end`, the result JSON kept next
to the report, and the report written from `research/TEMPLATE-research-report.md`. See
`AGENTS.md` section 5.
