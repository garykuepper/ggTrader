"""Generate comprehensive markdown report from pipeline results."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


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
    lines.append(f"| Total Return | {final_stats.get('profit_pct', 0):.2f}% |")
    lines.append(f"| Sharpe Ratio | {final_stats.get('sharpe', 0):.4f} |")
    lines.append(f"| Sortino Ratio | {final_stats.get('sortino', 0):.4f} |")
    lines.append(f"| Max Drawdown | {final_stats.get('max_drawdown', 0):.2f}% |")
    lines.append(f"| Total Trades | {final_stats.get('total_trades', 0)} |")
    lines.append(f"| Win Rate | {final_stats.get('win_rate', 0):.2f}% |")
    lines.append("")

    # Sensitivity Analysis Findings
    lines.append("## Sensitivity Analysis Findings")
    lines.append("")
    lines.append("Analysis of parameter impact across expanded ranges for each strategy.")
    lines.append("")

    for strategy_name, results_df in sensitivity_results.items():
        lines.append(f"### {strategy_name}")
        lines.append("")

        if not results_df.empty:
            # Get top 5 parameter combinations by Sharpe
            top_5 = results_df.nlargest(5, "Sharpe Ratio")
            lines.append("**Top 5 Parameter Combinations (by Sharpe Ratio)**:")
            lines.append("")
            lines.append("| Params | Sharpe Ratio |")
            lines.append("|--------|--------------|")

            for _, row in top_5.iterrows():
                params_str = ", ".join(
                    [f"{col}={row[col]}" for col in top_5.columns if col != "Sharpe Ratio"]
                )
                sharpe = row["Sharpe Ratio"]
                lines.append(f"| {params_str} | {sharpe:.4f} |")

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
        strategy_table_data.append({
            "Symbol": symbol,
            "Strategy": results.get("best_strategy", "N/A"),
            "Robustness Score": f"{results.get('robustness_score', 0):.4f}",
        })

    if strategy_table_data:
        lines.append("| Symbol | Strategy | Robustness Score |")
        lines.append("|--------|----------|-----------------|")
        for row in strategy_table_data:
            lines.append(f"| {row['Symbol']} | {row['Strategy']} | {row['Robustness Score']} |")
    lines.append("")

    # Final Full-Period Performance (Per-Coin)
    lines.append("## Final Full-Period Performance (Per-Coin)")
    lines.append("")
    lines.append("Performance metrics from running WFO-selected parameters on full 3-year data.")
    lines.append("")

    if per_coin_final_stats:
        lines.append("| Symbol | Strategy | Return % | Sharpe | Max DD % | Trades | Win Rate % |")
        lines.append("|--------|----------|----------|--------|----------|--------|-----------|")

        for symbol, stats in per_coin_final_stats.items():
            lines.append(
                f"| {symbol} | {stats.get('strategy', 'N/A')} | "
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
    lines.append("Aggregate results when all 20 coins trade simultaneously with shared capital.")
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
    lines.append("- Grid search evaluated all parameter combinations on top 20 cryptocurrencies")
    lines.append("- Sharpe ratio used as primary optimization metric")
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
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by ggTrader Pipeline*")

    # Write report
    report_file = output_path / "pipeline_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    print(f">>> Report generated: {report_file}")
