"""Generate comprehensive markdown report from pipeline results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_SENSITIVITY_METRIC_COLS = frozenset({"Sharpe Ratio", "Closed trades (agg)"})


def _fmt_pct_opt(value: Any, decimals: int = 2) -> str:
    """Format a percentage field; ``None``/invalid -> ``n/a``."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_float_opt(value: Any, decimals: int = 4) -> str:
    """Format a numeric field; ``None``/invalid -> ``n/a``."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_robustness_score(value: Any) -> str:
    """Format WFO robustness (finite, nan, +/-inf)."""
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(x):
        return "nan"
    if np.isneginf(x):
        return "-inf"
    if np.isposinf(x):
        return "inf"
    return f"{x:.4f}"


def _fmt_period(final_stats: Dict[str, Any]) -> str:
    """Human-readable backtest window from orchestrator fields."""
    start = final_stats.get("backtest_start")
    end = final_stats.get("backtest_end")
    if start and end:
        return f"{start} -> {end}"
    return "n/a"


def _sensitivity_param_columns(results_df: pd.DataFrame) -> List[str]:
    """Columns that are strategy parameters (not metrics)."""
    return [c for c in results_df.columns if c not in _SENSITIVITY_METRIC_COLS]


def generate_pipeline_report(
    sensitivity_results: Dict[str, pd.DataFrame],
    wfo_results: Dict[str, Any],
    final_backtest_results: Dict[str, Any],
    output_dir: str,
) -> None:
    """Generate a comprehensive markdown report summarizing the pipeline results.

    Args:
        sensitivity_results: Dict mapping strategy name to sensitivity results DataFrame
        wfo_results: Dict containing per_coin_results, final_portfolio, final_stats
        final_backtest_results: Dict with final_portfolio, per_coin_results, final_stats
        output_dir: Directory to save the report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract data from results
    per_coin_results = wfo_results.get("per_coin_results", {})
    per_coin_final_stats = final_backtest_results.get("per_coin_final_stats", {})
    final_stats = final_backtest_results.get("final_stats", {})

    # Start building report
    lines = []

    lines.append("# Trading Strategy Pipeline Report")
    lines.append("")
    lines.append(f"**Generated**: {timestamp}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    # Phase 2: Full Data Performance
    lines.append("## Phase 2: Result Validation (Full Data)")
    lines.append("")
    lines.append("### Combined Portfolio Performance (Full 3-Year Backtest)")
    lines.append("")

    p2_stats = final_backtest_results.get("phase_2_stats", final_stats)
    
    lines.append("| Metric | Strategy | Buy & Hold | Excess |")
    lines.append("|--------|----------|------------|--------|")
    lines.append(f"| Backtest period | {_fmt_period(p2_stats)} | - | - |")
    lines.append(
        f"| Total Return | {p2_stats.get('profit_pct', 0):.2f}% | "
        f"{_fmt_pct_opt(p2_stats.get('benchmark_profit_pct'))} | - |"
    )
    lines.append(
        f"| CAGR | {_fmt_pct_opt(p2_stats.get('cagr_pct'))} | "
        f"{_fmt_pct_opt(p2_stats.get('benchmark_cagr_pct'))} | "
        f"{_fmt_pct_opt(p2_stats.get('excess_cagr_pct'))} |"
    )
    lines.append(
        f"| Sharpe Ratio | {p2_stats.get('sharpe', 0):.4f} | "
        f"{_fmt_float_opt(p2_stats.get('benchmark_sharpe'))} | - |"
    )
    lines.append(
        f"| Max Drawdown | {p2_stats.get('max_drawdown', 0):.2f}% | "
        f"{_fmt_pct_opt(p2_stats.get('benchmark_max_drawdown'))} | - |"
    )
    lines.append(
        f"| Total Trades | {p2_stats.get('total_trades', 0)} | "
        f"{p2_stats.get('benchmark_total_trades', 0)} | - |"
    )
    lines.append(f"| Win Rate | {p2_stats.get('win_rate', 0):.2f}% | - | - |")
    lines.append("")

    # Phase 3: Recent Performance
    lines.append("## Phase 3: Recent Performance (Past Year)")
    lines.append("")

    p3_stats = final_backtest_results.get("phase_3_stats")
    if p3_stats:
        lines.append(
            "Combined portfolio replay on the **recent-only** window below using the same "
            "entry/exit/params chosen by WFO on the full training range (no re-optimization)."
        )
        lines.append("")
        lines.append("| Metric | Strategy | Buy & Hold | Excess |")
        lines.append("|--------|----------|------------|--------|")
        lines.append(f"| Backtest period | {_fmt_period(p3_stats)} | - | - |")
        lines.append(
            f"| Total Return | {p3_stats.get('profit_pct', 0):.2f}% | "
            f"{_fmt_pct_opt(p3_stats.get('benchmark_profit_pct'))} | - |"
        )
        lines.append(
            f"| CAGR | {_fmt_pct_opt(p3_stats.get('cagr_pct'))} | "
            f"{_fmt_pct_opt(p3_stats.get('benchmark_cagr_pct'))} | "
            f"{_fmt_pct_opt(p3_stats.get('excess_cagr_pct'))} |"
        )
        lines.append(
            f"| Sharpe Ratio | {p3_stats.get('sharpe', 0):.4f} | "
            f"{_fmt_float_opt(p3_stats.get('benchmark_sharpe'))} | - |"
        )
        lines.append(
            f"| Max Drawdown | {p3_stats.get('max_drawdown', 0):.2f}% | "
            f"{_fmt_pct_opt(p3_stats.get('benchmark_max_drawdown'))} | - |"
        )
        lines.append(
            f"| Total Trades | {p3_stats.get('total_trades', 0)} | "
            f"{p3_stats.get('benchmark_total_trades', 0)} | - |"
        )
        lines.append(f"| Win Rate | {p3_stats.get('win_rate', 0):.2f}% | - | - |")
        lines.append("")
    else:
        lines.append("*Phase 3 was not executed.*")
        lines.append("")

    # Phase 0: Sensitivity Analysis
    lines.append("## Phase 0: Sensitivity Analysis Findings")
    lines.append("")
    lines.append("Analysis of parameter impact across expanded ranges for each strategy.")
    lines.append("")

    if not sensitivity_results:
        lines.append(
            "*No sensitivity phase was run (default pipeline omits Phase 0 unless you pass "
            "`--sensitivity`); WFO used the configured parameter grids directly.*"
        )
        lines.append("")

    for strategy_name, results_df in sensitivity_results.items():
        lines.append(f"### {strategy_name}")
        lines.append("")

        if not results_df.empty:
            sr_num = pd.to_numeric(results_df["Sharpe Ratio"], errors="coerce")
            finite_mask = np.isfinite(sr_num.to_numpy(dtype=float, copy=False))
            param_cols = _sensitivity_param_columns(results_df)

            if not finite_mask.any():
                lines.append(
                    "**All sensitivity rows have NaN Sharpe** (typically gated by "
                    "`MIN_CLOSED_TRADES_TRAIN`: no completed round-trip on the window, "
                    "or train drawdown exceeded `MAX_TRAIN_DRAWDOWN_PCT` when set)."
                )
                lines.append("")
                if "Closed trades (agg)" in results_df.columns:
                    ct = results_df["Closed trades (agg)"]
                    lines.append(
                        f"- Closed trades (aggregated per combo): min={ct.min():.0f}, "
                        f"max={ct.max():.0f}, mean={ct.mean():.2f}"
                    )
                    lines.append(
                        f"- Rows with at least one closed trade: "
                        f"{int((ct >= 1).sum())} / {len(results_df)}"
                    )
                    lines.append("")
                lines.append("Showing first 5 parameter rows (for inspection):")
                lines.append("")
                sample = results_df.head(5)
                hdr = "| Params | Sharpe Ratio |"
                sep = "|--------|--------------|"
                if "Closed trades (agg)" in sample.columns:
                    hdr += " Closed trades (agg) |"
                    sep += "----------------------|"
                lines.append(hdr)
                lines.append(sep)
                for _, row in sample.iterrows():
                    params_str = ", ".join(f"{c}={row[c]}" for c in param_cols)
                    sh = row["Sharpe Ratio"]
                    sh_s = f"{sh:.4f}" if pd.notna(sh) and np.isfinite(float(sh)) else "nan"
                    line = f"| {params_str} | {sh_s} |"
                    if "Closed trades (agg)" in sample.columns:
                        line += f" {row['Closed trades (agg)']:.0f} |"
                    lines.append(line)
                lines.append("")
            else:
                top_5 = results_df.loc[finite_mask].nlargest(5, "Sharpe Ratio")
                lines.append("**Top 5 Parameter Combinations (by Sharpe Ratio)**:")
                lines.append("")
                hdr = "| Params | Sharpe Ratio |"
                sep = "|--------|--------------|"
                if "Closed trades (agg)" in top_5.columns:
                    hdr += " Closed trades (agg) |"
                    sep += "----------------------|"
                lines.append(hdr)
                lines.append(sep)

                for _, row in top_5.iterrows():
                    params_str = ", ".join(f"{c}={row[c]}" for c in param_cols)
                    sharpe = float(row["Sharpe Ratio"])
                    line = f"| {params_str} | {sharpe:.4f} |"
                    if "Closed trades (agg)" in top_5.columns:
                        line += f" {row['Closed trades (agg)']:.0f} |"
                    lines.append(line)

                lines.append("")
        else:
            lines.append("No sensitivity results available.")
            lines.append("")

    # Phase 1: Per-Coin Strategy Selection
    lines.append("## Phase 1: Per-Coin Strategy Selection (From WFO)")
    lines.append("")
    lines.append("Best performing strategy per coin based on robustness scores.")
    lines.append("")

    strategy_table_data = []
    for symbol, results in per_coin_results.items():
        strat = results.get("best_strategy")
        strat_disp = strat if strat is not None else "n/a"
        sel = results.get("selection_reason", "wfo_robustness")
        strategy_table_data.append({
            "Symbol": symbol,
            "Strategy": strat_disp,
            "Selection": sel,
            "Robustness Score": _fmt_robustness_score(results.get("robustness_score")),
        })

    if strategy_table_data:
        lines.append("| Symbol | Strategy | Selection | Robustness Score |")
        lines.append("|--------|----------|-----------|------------------|")
        for row in strategy_table_data:
            lines.append(
                f"| {row['Symbol']} | {row['Strategy']} | {row['Selection']} | "
                f"{row['Robustness Score']} |"
            )
    lines.append("")

    # OOS fold breakdown diagnostic table
    if per_coin_results:
        max_folds = max(
            (len(r.get("wfo_stats") or []) for r in per_coin_results.values()),
            default=0,
        )
        if max_folds > 0:
            lines.append("### WFO Out-of-Sample Sharpe — Per Fold Breakdown")
            lines.append("")
            lines.append(
                "Per-fold OOS Sharpe for each coin's winning strategy. "
                "Negative values indicate the strategy did not generalise in that fold."
            )
            lines.append("")
            fold_hdrs = " | ".join(f"Fold {i + 1}" for i in range(max_folds))
            lines.append(
                f"| Symbol | Strategy+Exit | IS Rob | OOS Rob | Consistency | {fold_hdrs} |"
            )
            sep = "|--------|---------------|--------|---------|-------------|"
            sep += "".join("--------|" for _ in range(max_folds))
            lines.append(sep)
            for symbol, r in per_coin_results.items():
                strat = r.get("best_strategy", "n/a")
                exit_ = r.get("best_exit", "n/a")
                label = f"{strat}+{exit_}"
                is_rob = _fmt_robustness_score(r.get("is_robustness_score", r.get("robustness_score")))
                oos_rob = _fmt_robustness_score(r.get("oos_robustness_score"))
                fc = r.get("fold_consistency")
                fc_str = f"{fc:.0%}" if fc is not None and np.isfinite(float(fc)) else "n/a"
                fold_vals: list[str] = []
                for fs in (r.get("wfo_stats") or []):
                    fold_vals.append(_fmt_float_opt(fs.get("oos_sharpe"), decimals=3))
                while len(fold_vals) < max_folds:
                    fold_vals.append("n/a")
                fold_cells = " | ".join(fold_vals)
                lines.append(
                    f"| {symbol} | {label} | {is_rob} | {oos_rob} | {fc_str} | {fold_cells} |"
                )
            lines.append("")

    # Final Full-Period Performance (Per-Coin)
    lines.append("## Final Full-Period Performance (Per-Coin)")
    lines.append("")
    lines.append("Performance metrics from running WFO-selected parameters on full 3-year data.")
    lines.append("")

    if per_coin_final_stats:
        lines.append(
            "| Symbol | Strategy | Selection | Return % | Sharpe | Max DD % | "
            "Trades | Win Rate % |"
        )
        lines.append(
            "|--------|----------|-----------|----------|--------|----------|--------|-----------|"
        )

        for symbol, stats in per_coin_final_stats.items():
            sstrat = stats.get("strategy")
            strat_cell = sstrat if sstrat is not None else "n/a"
            sel = stats.get("selection_reason", "wfo_robustness")
            lines.append(
                f"| {symbol} | {strat_cell} | {sel} | "
                f"{stats.get('profit_pct', 0):.2f}% | "
                f"{stats.get('sharpe', 0):.4f} | "
                f"{stats.get('max_drawdown', 0):.2f}% | "
                f"{stats.get('total_trades', 0)} | "
                f"{stats.get('win_rate', 0):.2f}% |"
            )

        lines.append("")

    # Combined Portfolio Performance
    lines.append("## Combined Portfolio Performance")
    lines.append("")
    lines.append(
        "Aggregate results when all configured symbols trade simultaneously with shared capital."
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append("| Starting Capital | $1,000 |")
    lines.append(f"| Final Value | ${final_stats.get('total_value', 0):,.2f} |")
    lines.append(f"| Total Profit/Loss | ${final_stats.get('total_profit', 0):,.2f} |")
    lines.append(f"| Return % | {final_stats.get('profit_pct', 0):.2f}% |")
    lines.append(f"| Sharpe Ratio | {final_stats.get('sharpe', 0):.4f} |")
    lines.append(f"| Max Drawdown | {final_stats.get('max_drawdown', 0):.2f}% |")
    lines.append(f"| Total Trades | {final_stats.get('total_trades', 0)} |")
    lines.append(f"| Win Rate | {final_stats.get('win_rate', 0):.2f}% |")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Phase 0: Sensitivity Analysis")
    lines.append(
        "- Expanded parameter ranges tested for each entry strategy (PSAR+ADX, EMA Crossover, RSI Reversal)"
    )
    lines.append(
        "- Grid search evaluated all parameter combinations on the configured symbol universe"
    )
    lines.append(
        "- WFO train fold: metric from `TRAIN_METRIC` (sharpe / sortino / calmar); "
        "gates via `MIN_CLOSED_TRADES_TRAIN` and optional `MAX_TRAIN_DRAWDOWN_PCT`"
    )
    lines.append("")

    lines.append("### Phase 1: Per-Coin Multi-Strategy WFO")
    lines.append("- Walk-Forward Optimization with 4 folds, 2:1 train/test ratio")
    lines.append("- Each coin optimized independently with each strategy")
    lines.append("- Best strategy selected per coin based on out-of-sample robustness score")
    lines.append("")

    lines.append("### Phase 2: Result Validation (Full Data)")
    lines.append("- WFO-selected strategy + parameters applied to full 3-year range for each coin")
    lines.append("- Per-coin results combined into single portfolio with shared capital")
    lines.append("- Performance compared against an equal-weight Buy & Hold benchmark")
    lines.append("")

    lines.append("### Phase 3: Recent Performance (Past Year)")
    lines.append("- Frozen parameters applied to the most recent 1-year data window")
    lines.append(
        "- Provides an 'out-of-sample' check on the most recent market conditions"
    )
    lines.append("- Performance compared against a 1-year Buy & Hold benchmark")
    lines.append("")

    lines.append("### Reporting")
    lines.append("- Comprehensive analysis of parameter sensitivity, strategy selection, and final performance")
    lines.append("- Per-coin and combined portfolio metrics")
    lines.append(
        "- **CAGR**: geometric annualized return from total return over the calendar span "
        "between the first and last bar of the combined close matrix"
    )
    lines.append(
        "- **Benchmark**: equal-weight buy-and-hold on the same symbols, first-bar entry and "
        "last-bar exit per leg, using the same `START_CASH`, `FEES`, `SLIPPAGE`, and bar frequency"
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by ggTrader Pipeline*")

    # Write report
    report_file = output_path / "pipeline_report.md"
    with open(report_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    print(f">>> Report generated: {report_file}")
