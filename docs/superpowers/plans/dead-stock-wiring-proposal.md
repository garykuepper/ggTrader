# Dead-stock exit: where it plugs into the lab (proposal, needs owner approval G4)

Status: proposal. Nothing below is built. The pure functions it uses are done and tested
(`src/ggTrader/lab/dead_stocks.py`, commit 7f054b51). Wiring touches existing lab code in `src/`,
so it waits for the owner's say.

## The problem in one paragraph

About 22,000 stocks in the combined bars stop trading before the file ends. The simulator
forward-fills prices, so a stock a strategy holds when it dies would sit at its last price
forever, and a bankruptcy would look like a flat line. The new functions take the true last day
of each series from the release (`market.stock_spans.last_date`), not from empty cells at the end
of a window, so two runs with different end dates agree on their shared history.

## Where it plugs in (four seams, all behind `source == "combined"`)

1. **`Bars` gets an optional `spans` field** (data/bars.py, part of Task 12). `load_bars(source="combined")`
   fills it from the release: one row per column with `symbol, last_date, has_successor, total_loss`.
   Only series that really ended are listed. Two exclusions: (a) a series still trading on the
   newest bar of its own source (current Yahoo columns) is left out; (b) a CRSP-only series whose
   last day is the file's last day (2024-12-31) is left out, because the file stopping is not the
   company dying (otherwise a 50% loss setting would wrongly halve every survivor on the last day).
   `has_successor` is true when another column continues the stock (an earlier spell, or the
   `PERMNO-n` column of a stock whose later days live in a linked Yahoo column): the position is
   still closed, but no price loss is applied.
2. **`lab/wfo/run.py` `run_wfo`**: right after `inputs.as_bars`, if the bars come from `combined`,
   build `forced_exit_frames(bars.close, bars.spans)` and a haircut close
   `haircut_close(bars.close, bars.spans, loss)`. Put both on `ScoringInputs` (new optional
   fields in `lab/wfo/scorers.py`). New keyword `dead_stock_loss: float = 0.0`. The rule is always
   on for `combined`; there is no switch to turn it off.
3. **Signal strategies** (`SignalScorer.prepare` through `simulate_signal_out` in
   `lab/strategies/contract.py`): add an optional `dead` argument. Before the arrays are tiled
   across parameter combinations, apply `apply_signal_rule` (exit on the last bar, drop that
   bar's and later entries) and pass the haircut close to the simulator. Count of removed entries
   goes to the result.
4. **Weight strategies** (`WeightScorer._simulate`): after `strategy.build`, apply
   `apply_weight_rule` (weight 0.0 on the last bar, then "no order"), tiled across combinations;
   the simulator close uses the haircut close before the forward-fill.
5. **CLI and provenance** (`lab/cli.py`, `lab/blend.py`, `engine/provenance.py`): new flag
   `--dead-stock-loss {0,0.5,1}` (default 0). The saved run records
   `dead_stock_rule: {applied, loss, n_ended_in_window, entries_suppressed, biases}` next to
   `data_release`. A run with different loss settings is a different run in provenance. A helper
   script runs a chosen strategy at 0%, 50% and 100% and prints the three results side by side;
   every result on `combined` says in plain words that dead-stock results are optimistic.

Runs on any other source (`snapshot`, `db`) do not change: the new fields are absent, the new code
paths are skipped, and the existing frozen-result tests must still pass byte for byte.

## Tests (written first)

- `test_combined_run_applies_forced_exit`: a bankrupt-stock fixture (position open at the last
  bar, an order submitted after it); expected cash, holdings and returns exact.
- `test_combined_run_weight_strategy_exits_dead_stock`: same for a weight strategy.
- Two end dates agree through their common history.
- A survivor on 2024-12-31 is not haircut; a successor-linked spell is closed with no loss.
- `test_non_combined_run_unchanged`: results identical to today for `snapshot`.

## Fallback if you would rather not touch the weight path

`combined` refuses weight strategies with a clear error ("weight strategies are not supported on
combined bars yet"); signal strategies still get seams 1 to 3 and 5. Weight strategies (such as
`xs_momentum`) would then wait for a later change.

## What you are approving

The edits to `lab/wfo/run.py`, `lab/wfo/scorers.py`, `lab/strategies/contract.py`, `lab/cli.py`,
`lab/blend.py` and `engine/provenance.py` described above, with the always-on rule for `combined`
bars and the three loss settings. Cost is one Terra build and one Opus review.
