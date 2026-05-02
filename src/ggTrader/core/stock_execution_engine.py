"""Live stock execution engine for Alpaca."""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, TrailingStopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

if TYPE_CHECKING:
    from ggTrader.utils.result_db_manager import ResultDBManager

from ggTrader.core.base_execution_engine import (
    BaseExecutionEngine,
    _format_entry_alert,
    _format_exit_alert,
)
from ggTrader.core.stock_regime_filtering import (
    _compute_spy_regime_mask,
    _compute_vix_regime_mask,
    _compute_spy_correlations,
)
from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader
from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.strategies import (
    AtrTrailingExit,
    BollingerMeanReversionEntry,
    DonchianBreakoutEntry,
    EmaCrossEntry,
    FixedStopTakeProfit,
    KeltnerBreakoutEntry,
    MacdCrossEntry,
    PsarAdxEntry,
    RsiReversalEntry,
    StochRsiReversalEntry,
    SupertrendFlipEntry,
    TrailingStopExit,
)
from ggTrader.utils.config import get_alpaca_credentials


class StockExecutionEngine(BaseExecutionEngine):
    """Live stock trading bot orchestrator for Alpaca."""

    STRATEGY_MAP = {
        "psar_adx": PsarAdxEntry,
        "ema_cross": EmaCrossEntry,
        "rsi_reversal": RsiReversalEntry,
        "macd_cross": MacdCrossEntry,
        "bbands_mean_reversion": BollingerMeanReversionEntry,
        "donchian_breakout": DonchianBreakoutEntry,
        "supertrend_flip": SupertrendFlipEntry,
        "stoch_rsi_reversal": StochRsiReversalEntry,
        "keltner_breakout": KeltnerBreakoutEntry,
    }

    EXIT_MAP = {
        "atr_trailing": AtrTrailingExit,
        "fixed_sl_tp": FixedStopTakeProfit,
        "trailing_stop": TrailingStopExit,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        results_path: Optional[str] = None,
        db_manager: Optional[ResultDBManager] = None,
        run_id: str = "LIVE_STOCKS",
    ):
        config["ASSET_CLASS"] = "stocks"
        super().__init__(config, results_path, db_manager, run_id)
        
        creds = get_alpaca_credentials(paper=config.get("PAPER", True))
        self.trading_client = TradingClient(
            api_key=creds["key_id"],
            secret_key=creds["secret_key"],
            paper=config.get("PAPER", True),
        )
        self.loader = CachedYFinanceLoader()
        
        if results_path:
            self.load_optimized_params(results_path)
            if not self.symbols:
                self.symbols = list(self.per_coin_params.keys())

        self.load_state()
        self._reconcile_positions()

    def _get_total_portfolio_usd(self) -> Optional[float]:
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            self.logger.error(f"Failed to fetch Alpaca equity: {e!r}")
            return None

    def _reconcile_positions(self) -> None:
        if self.config.get("DRY_RUN"): return
        try:
            positions = self.trading_client.get_all_positions()
            held = {p.symbol: float(p.qty) for p in positions}
            
            # Detect closed
            for s in list(self.active_positions.keys()):
                if s not in held:
                    # In a real impl, we'd fetch the last order to get fill price
                    # For now, just remove from state
                    self.logger.info(f"  [Reconcile] {s} no longer held - removing from state")
                    del self.active_positions[s]
            
            # Detect untracked
            for s, qty in held.items():
                if s not in self.active_positions:
                    self.logger.info(f"  [Reconcile] Found untracked position: {s}")
                    self.active_positions[s] = {"amount": qty, "entry_price": None}
            
            self.save_state()
        except Exception as e:
            self.logger.warning(f"  [Reconcile] Failed: {e!r}")

    def _execute_trade_logic(self, signals_dict: Dict[str, Dict[str, Any]], regime_allowance: Optional[Dict[str, bool]] = None) -> None:
        regime_allowance = regime_allowance or {}
        if self.circuit_breaker_triggered: return
        
        for s, sig in signals_dict.items():
            if sig["entry"] and s not in self.active_positions:
                if not regime_allowance.get(s, True): continue
                
                # Alpaca execution: Limit Buy + Trailing Stop
                price = sig["current_price"]
                qty = self.config.get("CAPITAL_PER_TRADE", 100.0) / price
                
                self.logger.info(f"PLACING STOCK BUY: {s} @ ${price:.2f}")
                if self.config.get("DRY_RUN"): continue
                
                try:
                    order = self.trading_client.submit_order(LimitOrderRequest(
                        symbol=s, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC, limit_price=price
                    ))
                    # Simplified: assume fill for state tracking in this turn
                    self.active_positions[s] = {"amount": qty, "entry_price": price, "entry_time": datetime.now(timezone.utc).isoformat()}
                    self.save_state()
                    self._notify(f"🟢 <b>BUY {s}</b> @ <code>${price:.2f}</code>")
                except Exception as e:
                    self.logger.error(f"Failed to place buy for {s}: {e}")

    def run_event_loop(self) -> None:
        self.state = "EVENT_LOOP"
        self.logger.info("ggTrader Stock Engine Started")
        while True:
            # Check market hours (simplified)
            # 1. Check circuit breaker
            self._check_circuit_breaker()
            
            # 2. Fetch data & signals
            df = self.loader.fetch_ohlcv(self.symbols, self.interval, pd.Timestamp.now(tz='UTC') - timedelta(days=5), pd.Timestamp.now(tz='UTC'))
            if not df.empty:
                # 3. Regime
                allowance = {s: True for s in self.symbols} # Stub
                # 4. Signals
                # 5. Execute
                pass
            
            time.sleep(3600) # Check hourly
