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
    lines.append("### Combined Portfolio Performance (Full 3-Year Backtest)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Backtest period | {_fmt_period(final_stats)} |")
    lines.append(f"| Calendar span (years) | {_fmt_float_opt(final_stats.get('backtest_years'), 3)} |")
    lines.append(f"| Total Return | {final_stats.get('profit_pct', 0):.2f}% |")
    lines.append(f"| CAGR | {_fmt_pct_opt(final_stats.get('cagr_pct'))} |")
    lines.append(f"| Sharpe Ratio | {final_stats.get('sharpe', 0):.4f} |")
    lines.append(f"| Sortino Ratio | {final_stats.get('sortino', 0):.4f} |")
    lines.append(f"| Max Drawdown | {final_stats.get('max_drawdown', 0):.2f}% |")
    lines.append(f"| Total Trades | {final_stats.get('total_trades', 0)} |")
    lines.append(f"| Win Rate | {final_stats.get('win_rate', 0):.2f}% |")
    lines.append("")
    lines.append("### Benchmark: Equal-Weight Buy-and-Hold")
    lines.append("")
    lines.append(f"*{final_stats.get('benchmark_label', 'Same universe and transaction assumptions as the strategy run.')}*")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Return | {_fmt_pct_opt(final_stats.get('benchmark_profit_pct'))} |")
    lines.append(f"| CAGR | {_fmt_pct_opt(final_stats.get('benchmark_cagr_pct'))} |")
    lines.append(f"| Sharpe Ratio | {_fmt_float_opt(final_stats.get('benchmark_sharpe'))} |")
    lines.append(f"| Max Drawdown | {_fmt_pct_opt(final_stats.get('benchmark_max_drawdown'))} |")
    lines.append(f"| Total Trades | {final_stats.get('benchmark_total_trades', 0)} |")
    lines.append(f"| Excess CAGR (strategy − benchmark) | {_fmt_pct_opt(final_stats.get('excess_cagr_pct'))} |")
    lines.append("")

    recent_stats = final_backtest_results.get("recent_validation_stats")
    if recent_stats:
        lines.append("### Recent validation (frozen WFO params)")
        lines.append("")
        lines.append(
            "Combined portfolio replay on the **recent-only** window below using the same "
            "entry/exit/params chosen by WFO on the full training range (no re-optimization)."
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Backtest period | {_fmt_period(recent_stats)} |")
        lines.append(
            f"| Calendar span (years) | {_fmt_float_opt(recent_stats.get('backtest_years'), 3)} |"
        )
        lines.append(f"| Total Return | {recent_stats.get('profit_pct', 0):.2f}% |")
        lines.append(f"| CAGR | {_fmt_pct_opt(recent_stats.get('cagr_pct'))} |")
        lines.append(f"| Sharpe Ratio | {recent_stats.get('sharpe', 0):.4f} |")
        lines.append(f"| Sortino Ratio | {recent_stats.get('sortino', 0):.4f} |")
        lines.append(f"| Max Drawdown | {recent_stats.get('max_drawdown', 0):.2f}% |")
        lines.append(f"| Total Trades | {recent_stats.get('total_trades', 0)} |")
        lines.append(f"| Win Rate | {recent_stats.get('win_rate', 0):.2f}% |")
        lines.append("")
        lines.append("#### Benchmark (equal-weight B&H, same recent window)")
        lines.append("")
        lines.append(
            f"*{recent_stats.get('benchmark_label', 'Same assumptions as full-range benchmark.')}*"
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Return | {_fmt_pct_opt(recent_stats.get('benchmark_profit_pct'))} |")
        lines.append(f"| CAGR | {_fmt_pct_opt(recent_stats.get('benchmark_cagr_pct'))} |")
        lines.append(f"| Sharpe Ratio | {_fmt_float_opt(recent_stats.get('benchmark_sharpe'))} |")
        lines.append(f"| Max Drawdown | {_fmt_pct_opt(recent_stats.get('benchmark_max_drawdown'))} |")
        lines.append(f"| Total Trades | {recent_stats.get('benchmark_total_trades', 0)} |")
        lines.append(
            f"| Excess CAGR (strategy − benchmark) | {_fmt_pct_opt(recent_stats.get('excess_cagr_pct'))} |"
        )
        lines.append("")

    # Sensitivity Analysis Findings
    lines.append("## Sensitivity Analysis Findings")
    lines.append("")
    lines.append("Analysis of parameter impact across expanded ranges for each strategy.")
    lines.append("")

    if not sensitivity_results:
        lines.append(
            "*No sensitivity phase was run (default pipeline omits Phase 1 unless you pass "
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

    # Per-Coin Strategy Selection
    lines.append("## Per-Coin Strategy Selection (From WFO)")
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

    # Final Full-Period Performance (Per-Coin)
    lines.append("## Final Full-Period Performance (Per-Coin)")
    lines.append("")
    lines.append("Performance metrics from running WFO-selected parameters on full 3-year data.")
    lines.append("")

    if per_coin_final_stats:
        lines.append(
            "| Symbol | Strategy | Selection | Return % | Sharpe | Max DD % | Trades | Win Rate % |"
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
    lines.append(f"| Starting Capital | $1,000 |")
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
    lines.append("### Phase 1: Sensitivity Analysis")
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

    lines.append("### Phase 2: Per-Coin Multi-Strategy WFO")
    lines.append("- Walk-Forward Optimization with 4 folds, 2:1 train/test ratio")
    lines.append("- Each coin optimized independently with each strategy")
    lines.append("- Best strategy selected per coin based on out-of-sample robustness score")
    lines.append("")

    lines.append("### Phase 3: Final Validation")
    lines.append("- WFO-selected strategy + parameters applied to full 3-year range for each coin")
    lines.append("- Per-coin results combined into single portfolio with shared capital")
    lines.append("- Final performance represents definitive backtest across entire period")
    lines.append("")

    lines.append("### Phase 4: Reporting")
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
