#!/usr/bin/env python3
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import closing
from typing import Any, Dict, Optional

import psycopg2
from mcp.server.fastmcp import FastMCP

# Ensure the ggTrader src directory is in sys.path so we can do local imports
sys.path.append("/home/flynn/ggTrader/src")

# Initialize FastMCP server
mcp = FastMCP("ggTrader")

DB_URL = os.environ.get("GGTRADER_DB_URL", "postgresql://ggtrader:ggtrader@localhost:5433/ggtrader")


def _format_run_report(run_data: Dict[str, Any]) -> str:
    """Render a backtest run row (metadata + metrics) as a Markdown report."""
    report = [
        f"### Backtest Run: {run_data['run_id']}",
        f"- **Strategy**: {run_data['strategy']}",
        f"- **Status**: {run_data['status']}",
        f"- **Created At**: {run_data['created_at']}",
        f"- **Period**: {run_data['eval_start']} to {run_data['eval_end']}",
        f"- **Market/Freq**: {run_data['market']} ({run_data['freq']})",
        f"- **Params**: {json.dumps(run_data['params'])}",
    ]

    for title, key in (
        ("Strategy Metrics", "metrics"),
        ("Benchmark Metrics", "benchmark_metrics"),
    ):
        section = run_data.get(key)
        if section:
            report.append(f"\n#### {title}")
            for k, v in section.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                report.append(f"- **{k}**: {val_str}")

    diag = run_data.get("diagnostics")
    if diag:
        report.append("\n#### Diagnostics")
        for k, v in diag.items():
            report.append(f"- **{k}**: {v}")

    return "\n".join(report)


@mcp.tool()
def list_strategies() -> str:
    """List all strategy names in ggTrader's STRATEGY_REGISTRY with their class names."""
    try:
        from ggTrader.lab.strategies import STRATEGY_REGISTRY

        result = ["### Registered ggTrader Strategies\n"]
        for name, cls in STRATEGY_REGISTRY.items():
            result.append(f"- **{name}**: `{cls.__name__}`")
        return "\n".join(result)
    except Exception as e:
        return f"Error listing strategies: {str(e)}"


@mcp.tool()
def list_tickers(force_refresh: bool = False) -> str:
    """List all unique ticker symbols in the TimescaleDB ohlcv table (cached 24h).

    Args:
        force_refresh: Set to True to bypass the cache and query the database directly.
    """
    cache_file = "/home/flynn/ggTrader/scripts/mcp_cache.json"
    cache_duration = 86400  # 24 hours

    # Check cache first
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            if time.time() - cache_data.get("timestamp", 0) < cache_duration:
                symbols = cache_data.get("symbols", [])
                return f"Found {len(symbols)} symbols (cached):\n" + ", ".join(symbols)
        except Exception:
            pass  # Fallback to query if cache fails to read or is corrupted

    # Run query
    try:
        with closing(psycopg2.connect(DB_URL)) as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM ohlcv")
            symbols = sorted([r[0] for r in cur.fetchall()])
    except Exception as e:
        return f"Error querying symbols from database: {str(e)}"

    # Save to cache
    try:
        with open(cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "symbols": symbols}, f)
    except Exception as ce:
        # Non-blocking write failure warning
        print(f"Warning: Failed to write cache: {str(ce)}", file=sys.stderr)

    return f"Found {len(symbols)} symbols:\n" + ", ".join(symbols)


@mcp.tool()
def run_backtest(strategy: str, additional_args: str = "") -> str:
    """Run a strategy backtest / walk-forward fold inside the docker compose environment.

    Args:
        strategy: Name of the strategy to run (e.g., 'xs_momentum', 'ema_cross', 'ensemble').
        additional_args: Optional CLI arguments to pass to the lab script
            (e.g., '--eval-start 2023-01-01 --eval-end 2023-12-31').
    """
    try:
        from ggTrader.lab.strategies import STRATEGY_REGISTRY

        if strategy not in STRATEGY_REGISTRY:
            return (
                f"Error: Strategy '{strategy}' is not registered. "
                f"Registered strategies: {list(STRATEGY_REGISTRY.keys())}"
            )
    except Exception:
        pass  # Proceed if import check fails, allowing running anyway

    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "ggtrader_live",
        "python",
        "ggt.py",
        "lab",
        "--strategy",
        strategy,
    ]
    if additional_args:
        # Safely tokenize the additional arguments to prevent command-injection
        cmd.extend(shlex.split(additional_args))

    try:
        # Limit backtest runs to a maximum of 300 seconds to prevent resource locks
        res = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/home/flynn/ggTrader", timeout=300
        )
        output = f"Exit code: {res.returncode}\n\n=== STDOUT ===\n{res.stdout or '(no stdout)'}\n"
        if res.stderr:
            output += f"\n=== STDERR ===\n{res.stderr}\n"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Backtest execution timed out (300 seconds limit)."
    except Exception as e:
        return f"Error executing backtest command: {str(e)}"


_RUN_QUERY = """
    SELECT r.run_id, r.strategy, r.market, r.freq, r.eval_start, r.eval_end,
           r.params, r.status, r.created_at,
           s.metrics, s.benchmark_metrics, s.diagnostics
    FROM lab_runs r
    LEFT JOIN lab_summary s ON r.run_id = s.run_id
"""


@mcp.tool()
def get_latest_backtest_run(strategy: Optional[str] = None) -> str:
    """Retrieve the most recent backtest run's metadata and performance KPIs.

    Args:
        strategy: Optional filter by strategy name (e.g., 'xs_momentum').
    """
    try:
        query = _RUN_QUERY
        params: list = []
        if strategy:
            query += " WHERE r.strategy = %s"
            params.append(strategy)
        query += " ORDER BY r.created_at DESC LIMIT 1"

        with closing(psycopg2.connect(DB_URL)) as conn, conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            if not row:
                suffix = f" for strategy '{strategy}'" if strategy else ""
                return f"No backtest runs found{suffix}"
            colnames = [desc[0] for desc in cur.description]

        return _format_run_report(dict(zip(colnames, row)))
    except Exception as e:
        return f"Error retrieving latest backtest: {str(e)}"


@mcp.tool()
def get_backtest_metrics(run_id: str) -> str:
    """Query TimescaleDB for detailed KPIs and diagnostics of a specific backtest run.

    Args:
        run_id: The unique ID of the backtest run (e.g. 'xs_momentum_ffead534').
    """
    try:
        query = _RUN_QUERY + " WHERE r.run_id = %s"
        with closing(psycopg2.connect(DB_URL)) as conn, conn.cursor() as cur:
            cur.execute(query, [run_id])
            row = cur.fetchone()
            if not row:
                return f"Error: Run ID '{run_id}' not found."
            colnames = [desc[0] for desc in cur.description]

        return _format_run_report(dict(zip(colnames, row)))
    except Exception as e:
        return f"Error retrieving backtest metrics for run {run_id}: {str(e)}"


@mcp.tool()
def get_ops_status() -> str:
    """Check health/status of live/paper trading (Docker containers, DB snapshots, states)."""
    report = ["## ggTrader Operational Status\n"]

    # 1. Check Docker Containers
    try:
        res = subprocess.run(
            ["docker", "ps", "--filter", "name=ggtrader", "--format", "{{.Names}}: {{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        docker_status = res.stdout.strip()
        report.append("### Docker Containers")
        if docker_status:
            for line in docker_status.splitlines():
                report.append(f"- {line}")
        else:
            report.append("No active ggTrader docker containers found.")
    except Exception as de:
        report.append(f"### Docker Containers\nError checking docker status: {str(de)}")

    # 2. Query DB snapshots
    try:
        with closing(psycopg2.connect(DB_URL)) as conn, conn.cursor() as cur:
            # System State
            report.append("\n### System State")
            cur.execute("SELECT key, value, updated_at FROM system_state ORDER BY updated_at DESC")
            states = cur.fetchall()
            if states:
                for key, val, updated_at in states:
                    report.append(f"- **{key}** (updated {updated_at}):")
                    report.append(f"  `{json.dumps(val)}`")
            else:
                report.append("No entries in system_state table.")

            # Paper Snapshots
            report.append("\n### Paper Trading Latest Snapshot")
            cur.execute(
                "SELECT run_date, portfolio_value, cash, positions, created_at "
                "FROM paper_snapshots ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                run_date, portfolio_value, cash, positions, created_at = row
                report.append(f"- **Run Date**: {run_date}")
                report.append(f"- **Created At**: {created_at}")
                report.append(f"- **Portfolio Value**: ${portfolio_value:,.2f}")
                report.append(f"- **Cash**: ${cash:,.2f}")
                report.append(f"- **Positions**: `{json.dumps(positions)}`")
            else:
                report.append("No paper trading snapshots found.")

            # Live Balance Snapshots
            report.append("\n### Live Balance Latest Snapshot")
            cur.execute(
                "SELECT asset_class, timestamp, total_usd, free_usd, positions_usd, num_positions "
                "FROM live_balance_snapshots ORDER BY timestamp DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                asset_class, timestamp, total_usd, free_usd, positions_usd, num_positions = row
                report.append(f"- **Asset Class**: {asset_class}")
                report.append(f"- **Timestamp**: {timestamp}")
                report.append(f"- **Total USD**: ${total_usd:,.2f}")
                report.append(f"- **Free USD**: ${free_usd:,.2f}")
                report.append(f"- **Positions USD**: ${positions_usd:,.2f}")
                report.append(f"- **Num Positions**: {num_positions}")
            else:
                report.append("No live balance snapshots found.")

            # Live Positions Snapshot
            report.append("\n### Live Positions Latest Snapshot")
            cur.execute(
                "SELECT asset_class, timestamp, positions, circuit_breaker_triggered "
                "FROM live_positions_snapshot ORDER BY timestamp DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                asset_class, timestamp, positions, circuit_breaker_triggered = row
                report.append(f"- **Asset Class**: {asset_class}")
                report.append(f"- **Timestamp**: {timestamp}")
                report.append(f"- **Circuit Breaker Triggered**: {circuit_breaker_triggered}")
                report.append(f"- **Positions**: `{json.dumps(positions)}`")
            else:
                report.append("No live positions snapshots found.")
    except Exception as e:
        report.append(f"\n### Database Query Error\nError querying operations tables: {str(e)}")

    return "\n".join(report)


if __name__ == "__main__":
    mcp.run()
