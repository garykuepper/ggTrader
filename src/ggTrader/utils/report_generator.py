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
    gate_stats = final_backtest_results.get("selection_gate_stats") or {}

    # Start building report
    lines = []

    p2_stats = final_backtest_results.get("phase_2_stats") or final_stats
    p3_stats = final_backtest_results.get("phase_3_stats")

    lines.append("# Trading Strategy Pipeline Report")
    lines.append("")
    lines.append(f"**Generated**: {timestamp}")
    lines.append("")

    # --- Executive Summary ---
    n_coins = len(per_coin_results)
    p2_period = _fmt_period(p2_stats)
    p3_period = _fmt_period(p3_stats) if p3_stats else "n/a"

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"**WFO Training/Test period:** {p2_period}  "
    )
    if p3_stats:
        lines.append(f"**YTD performance window:** {p3_period}  ")
    lines.append(f"**Coins:** {n_coins}")
    lines.append("")

    # --- WFO Selection Funnel ---
    if gate_stats and gate_stats.get("gates"):
        n_input = gate_stats.get("n_input", n_coins)
        n_kept = gate_stats.get("n_kept", n_coins)
        lines.append(
            f"**WFO Selection Funnel:** {n_input} candidates → "
            f"{n_kept} kept ({gate_stats.get('n_dropped_total', 0)} culled)"
        )
        lines.append("")
        lines.append("| Gate | Threshold | Dropped | Symbols |")
        lines.append("|------|-----------|---------|---------|")
        for g in gate_stats["gates"]:
            syms = g.get("dropped_symbols", []) or []
            sym_str = ", ".join(syms) if syms else "—"
            if len(sym_str) > 80:
                sym_str = sym_str[:77] + "…"
            lines.append(
                f"| {g['name']} | {g.get('threshold', '—')} | "
                f"{g['dropped_count']} | {sym_str} |"
            )
        lines.append("")
    p3_col = "YTD" if p3_stats else "YTD"
    def _beat_emoji(strat_cagr: Any, bench_cagr: Any) -> str:
        """✅ if strategy meets or exceeds benchmark, ❌ if it doesn't."""
        try:
            s = float(strat_cagr)
            b = float(bench_cagr)
        except (TypeError, ValueError):
            return ""
        if not (np.isfinite(s) and np.isfinite(b)):
            return ""
        return " ✅" if s >= b else " ❌"

    p2_strat = p2_stats.get("cagr_pct")
    p3_strat = p3_stats.get("cagr_pct") if p3_stats else None

    lines.append(f"| | WFO Full Range | {p3_col} |")
    lines.append("|-|----------------|--------|")
    lines.append(
        f"| Strategy CAGR | {_fmt_pct_opt(p2_strat)} | "
        f"{_fmt_pct_opt(p3_strat)} |"
    )
    lines.append(
        f"| BTC buy & hold CAGR | "
        f"{_fmt_pct_opt(p2_stats.get('benchmark_cagr_pct'))}"
        f"{_beat_emoji(p2_strat, p2_stats.get('benchmark_cagr_pct'))} | "
        f"{_fmt_pct_opt(p3_stats.get('benchmark_cagr_pct') if p3_stats else None)}"
        f"{_beat_emoji(p3_strat, p3_stats.get('benchmark_cagr_pct') if p3_stats else None)} |"
    )
    lines.append(
        f"| Strategy Sharpe | {_fmt_float_opt(p2_stats.get('sharpe'), decimals=2)} | "
        f"{_fmt_float_opt(p3_stats.get('sharpe') if p3_stats else None, decimals=2)} |"
    )
    lines.append(
        f"| Max Drawdown | {_fmt_pct_opt(p2_stats.get('max_drawdown'))} | "
        f"{_fmt_pct_opt(p3_stats.get('max_drawdown') if p3_stats else None)} |"
    )
    lines.append(
        f"| Total Trades | {p2_stats.get('total_trades', 0)} | "
        f"{p3_stats.get('total_trades', 0) if p3_stats else 'n/a'} |"
    )
    lines.append(f"| Win Rate | {_fmt_pct_opt(p2_stats.get('win_rate'))} | "
                 f"{_fmt_pct_opt(p3_stats.get('win_rate') if p3_stats else None)} |")
    lines.append("")

    # Embed dashboard plots directly in the executive summary
    plots_dir = output_path / "plots"
    full_range_plot = plots_dir / "combined_portfolio_final_dashboard.png"
    ytd_plot = plots_dir / "combined_portfolio_ytd_dashboard.png"
    if full_range_plot.exists():
        lines.append("### Full Range Portfolio")
        lines.append("")
        lines.append("![Full Range Portfolio](plots/combined_portfolio_final_dashboard.png)")
        lines.append("")
    if ytd_plot.exists():
        lines.append("### YTD Portfolio")
        lines.append("")
        lines.append("![YTD Portfolio](plots/combined_portfolio_ytd_dashboard.png)")
        lines.append("")

    def _phase_table(stats: Dict[str, Any]) -> None:
        lines.append("| Metric | Strategy | BTC (buy & hold) |")
        lines.append("|--------|----------|------------------|")
        lines.append(
            f"| Total Return | {_fmt_pct_opt(stats.get('profit_pct'))} | "
            f"{_fmt_pct_opt(stats.get('benchmark_profit_pct'))} |"
        )
        lines.append(
            f"| CAGR | {_fmt_pct_opt(stats.get('cagr_pct'))} | "
            f"{_fmt_pct_opt(stats.get('benchmark_cagr_pct'))} |"
        )
        lines.append(
            f"| Sharpe Ratio | {_fmt_float_opt(stats.get('sharpe'), decimals=4)} | "
            f"{_fmt_float_opt(stats.get('benchmark_sharpe'), decimals=4)} |"
        )
        lines.append(
            f"| Max Drawdown | {_fmt_pct_opt(stats.get('max_drawdown'))} | "
            f"{_fmt_pct_opt(stats.get('benchmark_max_drawdown'))} |"
        )
        lines.append(
            f"| Total Trades | {stats.get('total_trades', 0)} | "
            f"{stats.get('benchmark_total_trades', 1)} |"
        )
        lines.append(f"| Win Rate | {_fmt_pct_opt(stats.get('win_rate'))} | - |")
        lines.append("")

    # --- Training/Test Data ---
    lines.append("## Result Validation (Training/Test Data)")
    lines.append(
        f"**Period: {p2_period}** — WFO-selected parameters replayed on the "
        f"full training/test range."
    )
    lines.append("")
    lines.append("### Combined Portfolio Performance")
    lines.append("")
    _phase_table(p2_stats)

    # --- YTD Performance ---
    lines.append("## YTD Performance")

    if p3_stats:
        lines.append(
            f"**Period: {p3_period}** — WFO-selected parameters replayed on the YTD window, "
            f"no re-optimization."
        )
        lines.append("")
        _phase_table(p3_stats)
    else:
        lines.append("")
        lines.append("*Not executed.*")
        lines.append("")

    # --- Per-Coin Strategy Selection ---
    lines.append("## Per-Coin Strategy Selection (WFO)")
    lines.append("")
    lines.append("Best performing strategy per coin based on robustness scores (highest first).")
    lines.append("")

    def _rob_sort_key(item: tuple) -> float:
        val = item[1].get("robustness_score")
        try:
            v = float(val)
            return v if np.isfinite(v) else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    sorted_coin_results = sorted(per_coin_results.items(), key=_rob_sort_key, reverse=True)

    strategy_table_data = []
    for symbol, results in sorted_coin_results:
        strat = results.get("best_strategy")
        exit_ = results.get("best_exit", "")
        combo = f"{strat}+{exit_}" if exit_ else (strat or "n/a")
        sel = results.get("selection_reason", "wfo_robustness")
        strategy_table_data.append(
            {
                "Symbol": symbol,
                "Strategy": combo,
                "Selection": sel,
                "Robustness Score": _fmt_robustness_score(results.get("robustness_score")),
            }
        )

    if strategy_table_data:
        lines.append("| Symbol | Strategy | Selection | Robustness Score |")
        lines.append("|--------|----------|-----------|------------------|")
        for row in strategy_table_data:
            lines.append(
                f"| {row['Symbol']} | {row['Strategy']} | {row['Selection']} | "
                f"{row['Robustness Score']} |"
            )
    lines.append("")

    # WFO fold breakdown
    if per_coin_results:
        max_folds = max(
            (len(r.get("wfo_stats") or []) for r in per_coin_results.values()),
            default=0,
        )
        if max_folds > 0:
            # --- Mermaid Gantt: fold timeline (use representative coin's dates) ---
            rep_folds: List[Dict] = []
            for _, r in sorted_coin_results:
                folds = r.get("wfo_stats") or []
                if len(folds) == max_folds:
                    rep_folds = folds
                    break

            if rep_folds:
                lines.append("### WFO Fold Timeline")
                lines.append("")
                # Sort folds chronologically by train start (earliest first)
                chrono = sorted(
                    rep_folds,
                    key=lambda f: str(f.get("train_start", "")),
                )
                lines.append("```mermaid")
                lines.append("gantt")
                lines.append("    title Walk-Forward Folds (Step = Test Length)")
                lines.append("    dateFormat  YYYY-MM-DD")
                lines.append("    axisFormat  %Y-%m")
                lines.append("    ")
                for i, fs in enumerate(chrono, 1):
                    ts = str(fs.get("train_start", ""))[:10]
                    te = str(fs.get("test_start", ""))[:10]
                    os_ = str(fs.get("test_start", ""))[:10]
                    oe = str(fs.get("test_end", ""))[:10]
                    if ts and te and os_ and oe:
                        lines.append(f"    section Fold {i}")
                        lines.append(f"    Train  :active, f{i}_tr, {ts}, {te}")
                        lines.append(f"    Test   :crit, f{i}_ts, {os_}, {oe}")
                        lines.append("    ")
                lines.append("```")
                lines.append("")

            # --- OOS Sharpe table ---
            lines.append("### WFO Out-of-Sample Sharpe — Per Fold")
            lines.append("")
            lines.append(
                "Per-fold OOS Sharpe for each coin's winning strategy "
                "(folds ordered as above). Negative = strategy did not generalise."
            )
            lines.append("")
            fold_hdrs = " | ".join(f"Fold {i + 1}" for i in range(max_folds))
            lines.append(
                f"| Symbol | Strategy+Exit | IS Rob | OOS Rob | Consistency | {fold_hdrs} |"
            )
            sep = "|--------|---------------|--------|---------|-------------|"
            sep += "".join("--------|" for _ in range(max_folds))
            lines.append(sep)
            def _oos_sort_key(item: tuple) -> float:
                val = item[1].get("oos_robustness_score")
                try:
                    v = float(val)
                    return v if np.isfinite(v) else float("-inf")
                except (TypeError, ValueError):
                    return float("-inf")

            oos_sorted = sorted(sorted_coin_results, key=_oos_sort_key, reverse=True)
            for symbol, r in oos_sorted:
                strat = r.get("best_strategy", "n/a")
                exit_ = r.get("best_exit", "n/a")
                label = f"{strat}+{exit_}"
                is_rob = _fmt_robustness_score(
                    r.get("is_robustness_score", r.get("robustness_score"))
                )
                oos_rob = _fmt_robustness_score(r.get("oos_robustness_score"))
                fc = r.get("fold_consistency")
                fc_str = f"{fc:.0%}" if fc is not None and np.isfinite(float(fc)) else "n/a"
                fold_vals: list[str] = []
                for fs in r.get("wfo_stats") or []:
                    fold_vals.append(_fmt_float_opt(fs.get("oos_sharpe"), decimals=3))
                while len(fold_vals) < max_folds:
                    fold_vals.append("n/a")
                fold_cells = " | ".join(fold_vals)
                lines.append(
                    f"| {symbol} | {label} | {is_rob} | {oos_rob} | {fc_str} | {fold_cells} |"
                )
            lines.append("")

    # --- Full-Period Performance (Per-Coin) ---
    lines.append("## Full-Period Performance (Per-Coin)")
    lines.append("")
    lines.append("Performance metrics from running WFO-selected parameters on full-range data.")
    lines.append("")

    if per_coin_final_stats:
        lines.append(
            "| Symbol | Strategy | Exit | Selection | Return % | Sharpe | Max DD % | Trades | Win Rate % |"
        )
        lines.append(
            "|--------|----------|------|-----------|----------|--------|----------|--------|-----------|"
        )

        for symbol, stats in sorted(
            per_coin_final_stats.items(),
            key=lambda x: float(x[1].get("sharpe", 0) or 0),
            reverse=True,
        ):
            sstrat = stats.get("strategy")
            strat_cell = sstrat if sstrat is not None else "n/a"
            exit_cell = stats.get("exit", "n/a")
            sel = stats.get("selection_reason", "wfo_robustness")
            lines.append(
                f"| {symbol} | {strat_cell} | {exit_cell} | {sel} | "
                f"{stats.get('profit_pct', 0):.2f}% | "
                f"{stats.get('sharpe', 0):.4f} | "
                f"{stats.get('max_drawdown', 0):.2f}% | "
                f"{stats.get('total_trades', 0)} | "
                f"{stats.get('win_rate', 0):.2f}% |"
            )

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*For pipeline methodology, fold structure, and benchmark definitions "
        "see [docs/architecture.md](../../../docs/architecture.md).*"
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
