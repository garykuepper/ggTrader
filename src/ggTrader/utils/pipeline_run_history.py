"""Optional append-only logging of full-pipeline summaries to docs/pipeline_run_history.md."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ggTrader.utils.paths import find_project_root

# Set GGTRADER_APPEND_RUN_HISTORY to 0/false/no/off to skip appending (e.g. tests, CI).
_APPEND_HISTORY_DISABLE = frozenset({"0", "false", "no", "off"})


def _git_short_sha() -> str:
    try:
        root = find_project_root()
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "n/a"


def append_automated_run_section(
    *,
    run_folder_name: str,
    final_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    cli_summary: str,
) -> None:
    """Append a dated subsection under ``## Automated runs`` in ``docs/pipeline_run_history.md``.

    **On by default** after a successful pipeline. Set ``GGTRADER_APPEND_RUN_HISTORY`` to ``0``,
    ``false``, ``no``, or ``off`` (case-insensitive) to skip (e.g. local testing, CI).
    """
    flag = os.environ.get("GGTRADER_APPEND_RUN_HISTORY", "").strip().lower()
    if flag in _APPEND_HISTORY_DISABLE:
        return

    root = find_project_root()
    path = root / "docs" / "pipeline_run_history.md"
    if not path.is_file():
        return

    sha = _git_short_sha()
    n_sym = len(config.get("SYMBOLS") or [])
    ps = config.get("PORTFOLIO_SHARE", "n/a")
    exits = config.get("EXIT_TOURNAMENT", "n/a")

    def _pct(key: str, default: str = "n/a") -> str:
        v = final_stats.get(key)
        if v is None:
            return default
        try:
            return f"{float(v):.2f}%"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_metric(key: str, default: str = "n/a") -> str:
        v = final_stats.get(key)
        if v is None:
            return default
        if key == "total_trades":
            try:
                return str(int(v))
            except (TypeError, ValueError):
                return str(v)
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    block = (
        f"\n### `{run_folder_name}` (automated)\n\n"
        f"- **Git**: `{sha}`\n"
        f"- **CLI / flags**: `{cli_summary}`\n"
        f"- **Universe**: {n_sym} symbols | `PORTFOLIO_SHARE`={ps} | `EXIT_TOURNAMENT`={exits}\n"
        f"- **Strategy**: Total return {_pct('profit_pct', '0.00%')} | "
        f"CAGR {_pct('cagr_pct')} | Sharpe {_fmt_metric('sharpe')} | "
        f"Max DD {_pct('max_drawdown')} | Trades {_fmt_metric('total_trades')}\n"
        f"- **Benchmark**: BH return {_pct('benchmark_profit_pct')} | "
        f"Excess CAGR {_pct('excess_cagr_pct')}\n\n"
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(block)
