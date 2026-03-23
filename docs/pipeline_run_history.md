# Pipeline run history

High-level log of **full pipeline** runs (`run_full_pipeline.py` / `ggtrader-pipeline`) so you can compare outcomes after code, grid, or data changes. Per-run detail stays in `results/pipeline_<timestamp>/pipeline_report.md`; this file is the **index**.

## Pre-flight checklist (before a long run)

1. **Editable install**: `pip install -e .` from repo root (required for `ggTrader` imports and console commands).
2. **Database**: `.env` has `POSTGRES_CONNECTION_STRING`; TimescaleDB has OHLCV for your **interval** (e.g. `4h`), **symbols**, and **date range** from [`run_config.full_pipeline_config`](../src/ggTrader/utils/run_config.py) (or overrides).
3. **Universe size**: `MAX_SYMBOLS` must not exceed the number of rows in your symbols JSON. Default file `data/top_25_USD_2023-01-01_2025-12-31.json` has only **25** names—for **30** coins use e.g. `--max-symbols 30 --symbols-file data/top_50_USD_2023-01-01_2025-12-31.json`. The pipeline prints a **warning** if the file is shorter than `MAX_SYMBOLS`.
4. **Portfolio share**: Full pipeline default is **`PORTFOLIO_SHARE = 0.10`** (10% max per symbol per project policy). Change only in [`run_config.full_pipeline_config`](../src/ggTrader/utils/run_config.py) if you want a different cap (no CLI flag yet).
5. **Train metric**: Override WFO ranking without editing `run_config`: `python scripts/run_full_pipeline.py --train-metric sharpe` (also `sortino`, `calmar`, `composite`).
5b. **Exit tournament**: Default config is **ATR-only** (`atr_trailing`). Use **`--dual-exits`** for both ATR + fixed SL/TP, or **`--exits ...`** (comma-separated, matches `EXIT_REGISTRY`).
5c. **Recent validation (Phase 3B)**: `--recent-validation-start YYYY-MM-DD` and optional `--recent-validation-end` / `--recent-validation-ccxt-tail`; see [Strategy Pipeline Guide — Recent validation](strategy_pipeline_guide.md).
6. **Cost**: Phase 2 scales with coins × entry strategies × `EXIT_TOURNAMENT` × grid size. Use `--dry-run` for a quick wiring check; add `--sensitivity` / `--detailed-sensitivity` only when you intend the extra cost.
7. **Tests**: `pytest` after local code changes.

Example 30-coin command:

```bash
ggtrader-pipeline --max-symbols 30 --symbols-file data/top_50_USD_2023-01-01_2025-12-31.json --no-progress
```

### Monitoring while the pipeline runs

The run writes `results/pipeline_<timestamp>/status.txt` continuously. In **another terminal** (repo root):

- `ggtrader-pipeline-status --watch --interval 10` — follows the **most recently modified** pipeline folder (useful if you only have one active run).
- Or tail the specific folder: `Get-Content -Path 'results/pipeline_YYYYMMDD_HHMMSS/status.txt' -Wait -Tail 40` (PowerShell).

## How to add a manual row

1. Copy the **example** row in the table below (or add a new row above it).
2. Fill **Run folder**, **Git** (`git rev-parse --short HEAD`), **Flags**, metrics from that run’s `pipeline_report.md` (Executive Summary + benchmark), and **Notes** (what changed).
3. Keep **newest runs at the top** of the manual table.

## Portfolio improvement plan (executed 2026-03-22)

Baseline runs used current `full_pipeline_config` (**`TRAIN_METRIC=composite`** unless overridden), Kraken **`FEES` / `SLIPPAGE` unchanged**, **`PORTFOLIO_SHARE=0.10`** per policy. Logs: `logs/plan_exec_baseline20.log`, `plan_exec_btc1.log`, `plan_exec_5coin_no_sens.log`, `plan_exec_5coin_with_sens.log`, `plan_exec_5coin_train_sharpe.log`, `plan_20coin_train_sortino.log`, `plan_20coin_train_calmar.log`.

| Experiment | Run folder | N | Key result (strategy vs EW B&H) |
|------------|------------|---|-------------------------------------|
| 20-coin baseline | `pipeline_20260322_092410` | 20 | +6.68% total, **2.18% CAGR**, 325 trades vs **26.80%** BH CAGR |
| 20-coin `--train-metric sortino` | `pipeline_20260322_100424` | 20 | **Same** as composite: +6.68% total, **2.18% CAGR**, 325 trades |
| 20-coin `--train-metric calmar` | `pipeline_20260322_101334` | 20 | **Worse:** -0.45% total, **-0.15% CAGR**, 353 trades (different fold winners / more churn) |
| BTC-only lab | `pipeline_20260322_093524` | 1 | +4.15% total, **1.37% CAGR**, 65 trades; WFO pick **ema_cross+fixed_sl_tp** (robustness 0.33) vs **74%** BH CAGR |
| 5-coin, no Phase 1 | `pipeline_20260322_093621` | 5 | +16.08% total, **5.10% CAGR**, 179 trades vs **54.51%** BH CAGR |
| 5-coin + `--sensitivity` | `pipeline_20260322_093919` | 5 | **-3.83%** total, **-1.29% CAGR**, 238 trades vs **54.51%** BH CAGR |
| 5-coin A/B train metric | `pipeline_20260322_100009` | 5 | Same as composite run: **5.10% CAGR**, 179 trades with `--train-metric sharpe` |

**Takeaway:** **Sortino** as sole train metric matched **composite** on 20 coins here; **pure Calmar** on train folds **hurt** the combined book (favors volatile train paths that do not generalize). Skip `--sensitivity` until Phase 1 logic is revisited. **10% CAGR** still not met; next levers: **universe**, **strategy design**, optional **composite weight tuning**—not lower fees.

## Universe + exit tournament (executed 2026-03-22)

**`PORTFOLIO_SHARE` left at 0.10.** Coarse grid tweak in [`param_grids.py`](../src/ggTrader/pipeline/param_grids.py): `ema_cross` `ema_slow` `[21, 50]` (was `[21, 100]`), `rsi_reversal` `rsi_oversold` `[30, 40]` (was `[25, 40]`). New slice files: [`data/universe_slices.md`](../data/universe_slices.md). CLI: `--exits` on `run_full_pipeline.py` for exit ablation without editing `run_config`. Logs: `logs/plan_majors8_20260322.log`, `logs/plan_exit_atr_only_5coin.log`, `logs/plan_exit_fixed_only_5coin.log`.

| Experiment | Run folder | N | Key result (strategy vs EW B&H CAGR) |
|------------|------------|---|----------------------------------------|
| Majors top-8, dual exits | `pipeline_20260322_103806` | 8 | **19.04%** vs **38.03%** (68.60% total ret, 385 trades) |
| Top-5 default universe, ATR-only | `pipeline_20260322_104314` | 5 | **28.96%** vs **54.51%** (114.29% total ret, 188 trades) |
| Top-5 default universe, fixed-only | `pipeline_20260322_104527` | 5 | **-4.71%** vs **54.51%** (-13.45% total ret, 476 trades) |

**Takeaway:** **Fixed SL/TP alone** was clearly weaker on this 5-coin slice; **ATR-only** looked strong vs the prior dual-exit 5-coin run (`pipeline_20260322_093621`, older grids), so re-check with the same coarse book before dropping `fixed_sl_tp` from the tournament. **Majors-8 + dual exits** with the new coarse grid produced the strongest combined-book CAGR in this batch (still below equal-weight BH on the window).

## Plan follow-up (ATR default + alt8 / majors, 2026-03-23)

After **`EXIT_TOURNAMENT` default → `atr_trailing` only**, integrated **Phase 3B recent validation** (`--recent-validation-start` …), and report section **Recent validation (frozen WFO params)**.

| Experiment | Run folder | N | Key result (strategy vs EW B&H CAGR) |
|------------|------------|---|----------------------------------------|
| Alt ranks 9–16, ATR default | `pipeline_20260322_120916` | 8 | **3.30%** vs **8.01%** (10.22% total ret, 393 trades; max DD -76.76%) |
| Majors top-8, ATR default | `pipeline_20260322_121218` | 8 | **24.93%** vs **38.03%** (94.88% total ret, 302 trades) |
| Dry-run + Phase 3B sample | `pipeline_20260322_120807` | 3 | Recent window 2025-06-01–2025-12-30 in report (frozen params) |

Logs: `logs/plan_alt8_atr_20260323.log`, `logs/plan_majors8_atr_20260323.log`.

## Manual summary (newest first)

| Run folder | Date (local) | Git | Flags / universe | N sym | PS | Return % | CAGR % | Sharpe | Max DD % | Trades | BH ret % | Excess CAGR | Notes |
|------------|--------------|-----|-------------------|-------|-----|----------|--------|--------|----------|--------|----------|-------------|-------|
| `pipeline_20260321_221917` | 2026-03-21 | — | default top_25, MAX_SYMBOLS=5 | 5 | 0.1 | 43.97 | 12.93 | 0.7721 | -19.08 | 186 | 268.38 | -41.58% | Example row; replace with your run |

*PS = `PORTFOLIO_SHARE`. BH = equal-weight buy-and-hold from report.*

## Automated runs

A **successful** full pipeline **appends** a short bullet block below this section by default (same repo `docs/pipeline_run_history.md`).

To **disable** appending (tests, CI, or quick experiments you do not want in git), set `GGTRADER_APPEND_RUN_HISTORY` to `0`, `false`, `no`, or `off` (case-insensitive):

```powershell
# PowerShell — skip automated append for this session
$env:GGTRADER_APPEND_RUN_HISTORY="0"
ggtrader-pipeline --max-symbols 30 --symbols-file data/top_50_USD_2023-01-01_2025-12-31.json
```

---
*See also [Strategy Pipeline Guide](strategy_pipeline_guide.md). [README](../readme.md).*

### `pipeline_20260321_230422` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --no-save --dry-run`
- **Universe**: 3 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 15.86% | CAGR 5.04% | Sharpe 0.8183 | Max DD -10.88% | Trades 119
- **Benchmark**: BH return 342.26% | Excess CAGR -59.19%


### `pipeline_20260321_232640` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 30 --symbols-file data/top_50_USD_2023-01-01_2025-12-31.json`
- **Universe**: 30 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return -0.74% | CAGR -0.25% | Sharpe 0.0847 | Max DD -37.07% | Trades 568
- **Benchmark**: BH return 102.71% | Excess CAGR -26.84%


### `pipeline_20260322_083256` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 20`
- **Universe**: 20 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return -5.37% | CAGR -1.82% | Sharpe 0.0146 | Max DD -30.60% | Trades 434
- **Benchmark**: BH return 103.71% | Excess CAGR -28.62%


### `pipeline_20260322_085044` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 3`
- **Universe**: 3 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 22.16% | CAGR 6.91% | Sharpe 1.0158 | Max DD -9.60% | Trades 86
- **Benchmark**: BH return 342.26% | Excess CAGR -57.32%


### `pipeline_20260322_085426` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 20`
- **Universe**: 20 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 6.18% | CAGR 2.02% | Sharpe 0.2052 | Max DD -23.39% | Trades 311
- **Benchmark**: BH return 103.71% | Excess CAGR -24.77%


### `pipeline_20260322_092410` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 20`
- **Universe**: 20 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 6.68% | CAGR 2.18% | Sharpe 0.2162 | Max DD -23.32% | Trades 325
- **Benchmark**: BH return 103.71% | Excess CAGR -24.62%


### `pipeline_20260322_093524` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 1`
- **Universe**: 1 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 4.15% | CAGR 1.37% | Sharpe 0.3916 | Max DD -6.13% | Trades 65
- **Benchmark**: BH return 426.04% | Excess CAGR -72.65%


### `pipeline_20260322_093621` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 5`
- **Universe**: 5 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 16.08% | CAGR 5.10% | Sharpe 0.5047 | Max DD -14.99% | Trades 179
- **Benchmark**: BH return 268.38% | Excess CAGR -49.41%


### `pipeline_20260322_093919` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 5 --sensitivity`
- **Universe**: 5 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return -3.83% | CAGR -1.29% | Sharpe -0.0940 | Max DD -19.46% | Trades 238
- **Benchmark**: BH return 268.38% | Excess CAGR -55.80%


### `pipeline_20260322_100009` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 5 --train-metric sharpe`
- **Universe**: 5 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 16.08% | CAGR 5.10% | Sharpe 0.5047 | Max DD -14.99% | Trades 179
- **Benchmark**: BH return 268.38% | Excess CAGR -49.41%


### `pipeline_20260322_100424` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 20 --train-metric sortino`
- **Universe**: 20 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 6.68% | CAGR 2.18% | Sharpe 0.2162 | Max DD -23.32% | Trades 325
- **Benchmark**: BH return 103.71% | Excess CAGR -24.62%


### `pipeline_20260322_101334` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 20 --train-metric calmar`
- **Universe**: 20 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return -0.45% | CAGR -0.15% | Sharpe 0.0635 | Max DD -25.92% | Trades 353
- **Benchmark**: BH return 103.71% | Excess CAGR -26.95%


### `pipeline_20260322_103806` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --max-symbols 8`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 68.60% | CAGR 19.04% | Sharpe 0.7521 | Max DD -27.64% | Trades 385
- **Benchmark**: BH return 162.70% | Excess CAGR -18.98%


### `pipeline_20260322_104314` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 5 --exits atr_trailing`
- **Universe**: 5 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return 114.29% | CAGR 28.96% | Sharpe 0.8027 | Max DD -44.67% | Trades 188
- **Benchmark**: BH return 268.38% | Excess CAGR -25.55%


### `pipeline_20260322_104527` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --max-symbols 5 --exits fixed_sl_tp`
- **Universe**: 5 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['fixed_sl_tp']
- **Strategy**: Total return -13.45% | CAGR -4.71% | Sharpe -0.2975 | Max DD -26.78% | Trades 476
- **Benchmark**: BH return 268.38% | Excess CAGR -59.21%


### `pipeline_20260322_120807` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--dry-run --no-save --recent-validation-start 2025-06-01 --recent-validation-end 2025-12-30`
- **Universe**: 3 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return -7.07% | CAGR -2.42% | Sharpe -0.2289 | Max DD -19.92% | Trades 301
- **Benchmark**: BH return 342.26% | Excess CAGR -66.64%


### `pipeline_20260322_120916` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --symbols-file data/alt_ranks_9_16_usd_2023-01-01_2025-12-31.json --max-symbols 8`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return 10.22% | CAGR 3.30% | Sharpe 0.3651 | Max DD -76.76% | Trades 393
- **Benchmark**: BH return 25.98% | Excess CAGR -4.71%


### `pipeline_20260322_121218` (automated)

- **Git**: `ee6b1f3`
- **CLI / flags**: `--no-progress --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --max-symbols 8`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return 94.88% | CAGR 24.93% | Sharpe 0.7082 | Max DD -46.25% | Trades 302
- **Benchmark**: BH return 162.70% | Excess CAGR -13.09%


### `pipeline_20260322_143117` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--max-symbols 8 --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --recent-validation-start 2025-01-01`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return -10.85% | CAGR -3.76% | Sharpe -0.2891 | Max DD -25.11% | Trades 317
- **Benchmark**: BH return 162.70% | Excess CAGR -41.79%


### `pipeline_20260322_143439` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--max-symbols 8 --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --recent-validation-start 2026-01-01`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return -10.85% | CAGR -3.76% | Sharpe -0.2891 | Max DD -25.11% | Trades 317
- **Benchmark**: BH return 162.70% | Excess CAGR -41.79%


### `pipeline_20260322_205756` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--dry-run --max-symbols 2`
- **Universe**: 2 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return -0.06% | CAGR -0.02% | Sharpe 0.0107 | Max DD -8.11% | Trades 79
- **Benchmark**: BH return 249.38% | Excess CAGR -51.82%


### `pipeline_20260322_210034` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--no-progress --max-symbols 8 --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --recent-validation-start 2026-01-01`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return -10.85% | CAGR -3.76% | Sharpe -0.2891 | Max DD -25.11% | Trades 317
- **Benchmark**: BH return 162.70% | Excess CAGR -41.79%


### `pipeline_20260322_214320` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--no-progress --max-symbols 8 --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --exits atr_trailing`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return 5.65% | CAGR 1.85% | Sharpe 0.3134 | Max DD -10.93% | Trades 102
- **Benchmark**: BH return 358.36% | Excess CAGR -64.34%


### `pipeline_20260322_222728` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--no-progress --dry-run`
- **Universe**: 3 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 34.33% | CAGR 10.35% | Sharpe 1.4168 | Max DD -5.40% | Trades 85
- **Benchmark**: BH return 342.26% | Excess CAGR -53.88%


### `pipeline_20260322_224231` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--no-progress --max-symbols 8 --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --exits atr_trailing`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return -13.17% | CAGR -4.60% | Sharpe -0.7275 | Max DD -18.46% | Trades 120
- **Benchmark**: BH return 148.85% | Excess CAGR -40.16%


### `pipeline_20260322_225700` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--no-progress --dry-run`
- **Universe**: 3 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing', 'fixed_sl_tp']
- **Strategy**: Total return 25.53% | CAGR 7.88% | Sharpe 1.0307 | Max DD -7.96% | Trades 90
- **Benchmark**: BH return 342.26% | Excess CAGR -56.34%


### `pipeline_20260322_225844` (automated)

- **Git**: `5daf853`
- **CLI / flags**: `--no-progress --max-symbols 8 --symbols-file data/majors_top8_usd_2023-01-01_2025-12-31.json --exits atr_trailing`
- **Universe**: 8 symbols | `PORTFOLIO_SHARE`=0.1 | `EXIT_TOURNAMENT`=['atr_trailing']
- **Strategy**: Total return -4.64% | CAGR -1.57% | Sharpe -0.7668 | Max DD -5.47% | Trades 26
- **Benchmark**: BH return 447.16% | Excess CAGR -77.89%

