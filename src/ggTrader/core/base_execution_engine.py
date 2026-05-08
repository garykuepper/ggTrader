"""Base class for the live crypto trading execution engine."""

from __future__ import annotations

import html
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from ggTrader.utils.result_db_manager import ResultDBManager

from ggTrader.core.trade_tracker import TradeTracker
from ggTrader.utils.notifier import build_notifiers_from_env


def setup_live_logger(log_name: str = "live_trader.log") -> logging.Logger:
    """Configures a thread-safe logger writing to both Console and logs/log_name"""
    logger = logging.getLogger("ggTraderLive")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = RotatingFileHandler(log_dir / log_name, maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _format_hold_time(entry_time: Optional[str]) -> str:
    """Render ISO8601-UTC entry_time as a compact hold-duration string."""
    if not entry_time:
        return "?"
    try:
        t0 = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - t0
        total_min = int(delta.total_seconds() // 60)
    except Exception:
        return "?"
    if total_min < 0:
        return "?"
    days, rem_min = divmod(total_min, 1440)
    hours, minutes = divmod(rem_min, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_entry_alert(
    symbol: str,
    strategy: str,
    exit_name: str,
    fill_price: float,
    amount: float,
    cost_usd: float,
    stop_pct: float,
    open_positions: int,
    universe_size: int,
) -> str:
    """HTML-formatted alert for a successful entry fill."""
    base = symbol.split("-")[0].split("/")[0]
    sym_e = html.escape(symbol)
    strat_e = html.escape(strategy or "?")
    exit_e = html.escape(exit_name or "?")
    return (
        f"🟢 <b>BUY {sym_e}</b> @ <code>${fill_price:,.6g}</code>\n"
        f"strategy: <code>{strat_e}</code> / <code>{exit_e}</code>\n"
        f"size: <code>{amount:,.6g} {html.escape(base)}</code> "
        f"(~${cost_usd:,.2f})\n"
        f"stop: -{stop_pct:.2f}%\n"
        f"open positions: {open_positions}/{universe_size}"
    )


def _format_exit_alert(
    symbol: str,
    strategy: str,
    exit_reason: str,
    entry_price: Optional[float],
    exit_price: float,
    amount: float,
    entry_fee: float,
    exit_fee: float,
    entry_time: Optional[str],
    open_positions: int,
    universe_size: int,
) -> str:
    """HTML-formatted alert for an exit fill."""
    sym_e = html.escape(symbol)
    strat_e = html.escape(strategy or "?")
    reason_e = html.escape(exit_reason or "?")
    hold_str = _format_hold_time(entry_time)

    if entry_price and entry_price > 0:
        pnl_usd = (exit_price - entry_price) * amount - (entry_fee + exit_fee)
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        win = pnl_usd >= 0
        marker = "✅" if win else "❌"
        sign = "+" if pnl_usd >= 0 else "-"
        pnl_line = (
            f"PnL: <code>{sign}${abs(pnl_usd):,.2f}</code> "
            f"({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%) {marker}"
        )
        entry_line = (
            f"entry: <code>${entry_price:,.6g}</code> (held {hold_str})"
        )
    else:
        marker = "⚠️"
        pnl_line = "PnL: <code>?</code> (no stored entry price)"
        entry_line = f"entry: <code>?</code> (held {hold_str})"

    return (
        f"🔴 <b>SELL {sym_e}</b> @ <code>${exit_price:,.6g}</code>  {marker}\n"
        f"{entry_line}\n"
        f"{pnl_line}\n"
        f"reason: <code>{reason_e}</code>\n"
        f"strategy: <code>{strat_e}</code>\n"
        f"open positions: {open_positions}/{universe_size}"
    )


class BaseExecutionEngine(ABC):
    """Abstract base for live trading bots with shared risk and state logic."""

    def __init__(
        self,
        config: Dict[str, Any],
        results_path: Optional[str] = None,
        db_manager: Optional[ResultDBManager] = None,
        run_id: str = "LIVE",
    ):
        self.config = config
        self.results_path = results_path
        self.db_manager = db_manager
        self.run_id = run_id
        
        self.asset_class = "crypto"
        self.logger = setup_live_logger("live_trader.log")

        self.persistence_path = config.get("PERSISTENCE_PATH", "data/active_positions.json")

        self.state = "INITIALIZED"
        self.interval = config.get("INTERVAL", "4h")
        self.symbols = config.get("SYMBOLS", [])
        self.per_coin_params = {}
        self.allocation_weights: Dict[str, float] = {}
        self.active_positions = {}

        self.tracker = TradeTracker(
            db_manager=self.db_manager,
            run_id=self.run_id,
        )
        self.notifiers = build_notifiers_from_env()

        # Daily loss circuit breaker state
        self.daily_start_equity: Optional[float] = None
        self.circuit_breaker_triggered = False
        self._last_check_date: Optional[datetime.date] = None

    def _notify(self, message: str) -> None:
        """Fan-out a fill alert to every configured notifier."""
        if not self.notifiers:
            return
        for n in self.notifiers:
            try:
                n.send_text(message)
            except Exception as e:
                self.logger.debug(f"  [Notify] {n.name} send failed: {e!r}")

    def load_optimized_params(self, source: Any) -> None:
        """Load per-symbol parameters and apply selection gates.

        ``source`` may be:
          - a ``LatestResearchRun`` dataclass (preferred — DB-backed)
          - a path to a ``run_results.json`` (legacy)
          - a path to a results directory containing ``run_results.json``
        """
        from ggTrader.utils.state_manager import LatestResearchRun, from_run_dir

        per_coin: Dict[str, Any] = {}
        if isinstance(source, LatestResearchRun):
            per_coin = source.per_coin or {}
        else:
            path_str = str(source)
            p = Path(path_str)
            if p.is_dir():
                latest = from_run_dir(p)
                per_coin = latest.per_coin
            elif p.exists():
                with open(p, "r") as f:
                    data = json.load(f)
                sp = data.get("strategy_parameters", {})
                per_coin = sp.get("per_coin", data.get("per_coin_results", {}))
            else:
                self.logger.warning(f"Results source {path_str} not found.")
                return
        
        min_rob = float(self.config.get("MIN_ROBUSTNESS_SCORE", 0.0))
        max_per_strategy = self.config.get("MAX_COINS_PER_STRATEGY", None)
        blacklist = {s.upper() for s in (self.config.get("SYMBOL_BLACKLIST", []) or [])}

        strategy_counts: Dict[str, int] = defaultdict(int)
        
        for symbol, result in per_coin.items():
            if symbol.upper() in blacklist: continue
            
            robustness = float(result.get("robustness_score", 0.0))
            if min_rob > 0 and robustness < min_rob: continue
            
            strategy = result.get("best_strategy")
            if max_per_strategy is not None and strategy_counts[strategy] >= max_per_strategy:
                continue

            strategy_counts[strategy] += 1
            self.per_coin_params[symbol] = {
                "strategy_name": strategy,
                "exit_name": result.get("best_exit"),
                "params": result.get("best_params", {}),
                "robustness_score": robustness,
                "oos_robustness_score": float(result.get("oos_robustness_score", 0.0)),
            }
        self.logger.info(f"Loaded optimized parameters for {len(self.per_coin_params)} symbols.")
        self._compute_research_allocation_weights()

    def _compute_research_allocation_weights(self) -> None:
        """Compute per-symbol portfolio weights from OOS robustness scores.

        Mirrors ``orchestrator._compute_allocation_weights``: weights ∝ max(0, oos_score),
        normalized to sum to 1.0, capped at ``MAX_COIN_ALLOCATION`` per coin with
        excess redistributed to under-cap coins. Stored on ``self.allocation_weights``.
        Coins with non-positive OOS scores receive 0% — they're in the gate-passing
        universe but the live trader will refuse to open new entries on them when
        ``WEIGHTED_SIZING`` is enabled.
        """
        import numpy as np
        syms = list(self.per_coin_params.keys())
        if not syms:
            self.allocation_weights = {}
            return
        scores = [float(self.per_coin_params[s].get("oos_robustness_score", 0.0)) for s in syms]
        raw = np.array([max(0.0, x) for x in scores], dtype=float)
        total = raw.sum()
        if total <= 0.0:
            raw = np.ones(len(scores), dtype=float)
            total = raw.sum()
        w = raw / total
        max_alloc = float(self.config.get("MAX_COIN_ALLOCATION", 0.25))
        for _ in range(len(w) + 1):
            over = w > max_alloc + 1e-12
            if not over.any():
                break
            excess = (w[over] - max_alloc).sum()
            w[over] = max_alloc
            under = ~over
            if not under.any():
                w += excess / len(w)
                break
            w[under] += excess / under.sum()
        # Zero-out any coins whose raw OOS was non-positive so they aren't allocated
        # via the redistribution loop.
        for i, sc in enumerate(scores):
            if sc <= 0.0:
                w[i] = 0.0
        self.allocation_weights = {s: float(wt) for s, wt in zip(syms, w)}
        nz = sum(1 for v in self.allocation_weights.values() if v > 0)
        self.logger.info(
            f"Allocation weights computed: {nz}/{len(syms)} coins receive capital "
            f"(cap={max_alloc:.0%}/coin). Top: "
            + ", ".join(f"{s}={self.allocation_weights[s]*100:.1f}%"
                        for s in sorted(self.allocation_weights, key=self.allocation_weights.get, reverse=True)[:5])
        )

    def load_state(self) -> None:
        """Load state from persistence file."""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "positions" in data:
                    self.active_positions = data["positions"]
                    self.daily_start_equity = data.get("daily_start_equity")
                    self.circuit_breaker_triggered = data.get("circuit_breaker_triggered", False)
                    lcd = data.get("last_check_date")
                    if lcd:
                        self._last_check_date = datetime.fromisoformat(lcd).date()
                else:
                    self.active_positions = data
                self.logger.info(f"Loaded {len(self.active_positions)} active positions.")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e!r}")

    def save_state(self) -> None:
        """Save state to persistence file atomically."""
        state_data = {
            "positions": self.active_positions,
            "daily_start_equity": self.daily_start_equity,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "last_check_date": self._last_check_date.isoformat() if self._last_check_date else None,
        }
        dir_name = os.path.dirname(self.persistence_path) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state_data, f, indent=4)
            os.replace(tmp_path, self.persistence_path)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e!r}")
            os.unlink(tmp_path)

    def _check_circuit_breaker(self) -> None:
        """Evaluate the intraday drawdown circuit breaker."""
        now = datetime.now(timezone.utc)
        current_date = now.date()

        # Reset on new day
        if self._last_check_date != current_date:
            self.logger.info(f"--- New Day Started: {current_date} ---")
            self.daily_start_equity = self._get_total_portfolio_usd()
            self.circuit_breaker_triggered = False
            self._last_check_date = current_date
            self.save_state()
            if self.daily_start_equity:
                self.logger.info(f"  [CircuitBreaker] Start-of-day equity: ${self.daily_start_equity:,.2f}")

        # Check limit
        limit = self.config.get("DAILY_LOSS_LIMIT_PCT")
        if limit and self.daily_start_equity and not self.circuit_breaker_triggered:
            current_equity = self._get_total_portfolio_usd()
            if current_equity:
                drawdown = (current_equity / self.daily_start_equity) - 1
                if drawdown < -limit:
                    self.logger.warning(f"🛑 [CircuitBreaker] TRIGGERED: Loss {drawdown*100:.2f}% > {limit*100:.2f}%")
                    self.circuit_breaker_triggered = True
                    self.save_state()
                    self._notify(
                        f"🛑 <b>Circuit Breaker Triggered ({self.asset_class})</b>\n"
                        f"Intraday Loss: <code>{drawdown*100:.2f}%</code>\n"
                        f"Entries halted until tomorrow."
                    )

    @abstractmethod
    def run_event_loop(self) -> None:
        """Main operational loop."""
        pass

    @abstractmethod
    def _execute_trade_logic(self, signals_dict: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """Evaluate signals and place orders."""
        pass

    @abstractmethod
    def _get_total_portfolio_usd(self) -> Optional[float]:
        """Query total account equity in USD."""
        pass
