"""Live stock execution engine for Alpaca."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest, 
    TrailingStopOrderRequest, 
    GetOrdersRequest,
    MarketOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, AssetClass

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
from ggTrader.core.market_hours import MarketHours


class StockExecutionEngine(BaseExecutionEngine):
    """Live stock trading bot orchestrator for Alpaca (Paper and Live)."""

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
        
        # Alpaca Client Setup
        is_paper = config.get("PAPER", True)
        creds = get_alpaca_credentials(paper=is_paper)
        self.trading_client = TradingClient(
            api_key=creds["key_id"],
            secret_key=creds["secret_key"],
            paper=is_paper,
        )
        
        self.loader = CachedYFinanceLoader()
        self.market_hours = MarketHours(self.trading_client)
        
        if results_path:
            self.load_optimized_params(results_path)
            if not self.symbols:
                self.symbols = list(self.per_coin_params.keys())

        self.load_state()
        self._reconcile_positions()

    def _get_total_portfolio_usd(self) -> Optional[float]:
        """Fetch total equity from Alpaca account."""
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            self.logger.error(f"  [Alpaca] Failed to fetch equity: {e!r}")
            return None

    def _reconcile_positions(self) -> None:
        """Sync local active_positions with actual Alpaca holdings."""
        if self.config.get("DRY_RUN"):
            return

        try:
            positions = self.trading_client.get_all_positions()
            held = {p.symbol: float(p.qty) for p in positions}
            
            # 1. Detect positions closed on Alpaca (e.g. TSL triggered)
            stale = [s for s in self.active_positions if s not in held]
            if stale:
                self.logger.warning(f"  [Reconcile] {len(stale)} position(s) closed on Alpaca: {stale}")
                for s in stale:
                    pos = self.active_positions[s]
                    # Note: We record exit with zero fee for Alpaca-side triggers
                    self.tracker.record_sell(
                        symbol=s, order_id="alpaca_side_exit", price=0.0, # Will need better fill price recovery
                        amount=pos["amount"], amount_usd=0.0, fee=0.0,
                        entry_price=pos.get("entry_price"), entry_time=pos.get("entry_time"),
                        exit_reason="alpaca_side_trigger"
                    )
                    del self.active_positions[s]
            
            # 2. Detect untracked positions found on Alpaca
            untracked = [s for s in held if s not in self.active_positions]
            if untracked:
                self.logger.warning(f"  [Reconcile] Found {len(untracked)} untracked position(s) on Alpaca: {untracked}")
                for s in untracked:
                    self.active_positions[s] = {
                        "amount": held[s],
                        "entry_price": None, # Unknown
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "exit_name": "atr_trailing", # Default
                        "stop_pct": 3.0
                    }
            
            self.save_state()
        except Exception as e:
            self.logger.error(f"  [Reconcile] Failed: {e!r}")

    def _fetch_latest_data(self) -> pd.DataFrame:
        """Fetch daily bars for the universe plus benchmark (SPY)."""
        fetch_symbols = list(self.symbols)
        if "SPY" not in fetch_symbols:
            fetch_symbols = ["SPY"] + fetch_symbols
            
        # Daily bars: look back 60 days for enough indicator warmup
        start_date = pd.Timestamp.now(tz="UTC") - timedelta(days=60)
        return self.loader.fetch_ohlcv(
            symbols=fetch_symbols, 
            interval="1d", 
            start_date=start_date
        )

    def _compute_latest_signals(self, ohlcv_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Run project strategies on the latest daily OHLCV."""
        results = {}
        for s in self.symbols:
            if s not in ohlcv_df.columns.get_level_values(0):
                continue
            
            info = self.per_coin_params.get(s)
            if not info: continue
            
            strat_class = self.STRATEGY_MAP.get(info["strategy_name"])
            exit_class = self.EXIT_MAP.get(info["exit_name"])
            if not strat_class or not exit_class: continue
            
            close = ohlcv_df[(s, "close")]
            high = ohlcv_df[(s, "high")]
            low = ohlcv_df[(s, "low")]
            
            pc = IndicatorPrecomputer(close, high, low)
            grid = {k: [v] for k, v in info["params"].items()}
            
            ent, _ = strat_class().compute_entries(pc, grid)
            ext, stops, pxs = exit_class().compute_exits(ent, pc, grid, 1)
            
            atr = float("nan")
            if info["exit_name"] == "atr_trailing":
                try:
                    atr = float(pc.compute_atr(int(info["params"].get("atr_length", 14))).atrr.values[-1])
                except Exception: pass

            results[s] = {
                "entry": bool(ent[-1, 0]),
                "exit": bool(ext[-1, 0]),
                "exit_name": info["exit_name"],
                "stop_price": float(stops[-1, 0]),
                "fill_price": float(pxs[-1, 0]),
                "current_price": float(close.iloc[-1]),
                "atr_value": atr,
            }
        return results

    def _execute_trade_logic(self, signals_dict: Dict[str, Dict[str, Any]], regime_allowance: Optional[Dict[str, bool]] = None) -> None:
        """Submit Limit Buys for entries and Trailing Stops for management."""
        regime_allowance = regime_allowance or {}
        if self.circuit_breaker_triggered:
            return

        for s, sig in signals_dict.items():
            # 1. Handle Entry Signals
            if sig["entry"] and s not in self.active_positions:
                if not regime_allowance.get(s, True):
                    continue
                
                price = sig["current_price"]
                capital = self.config.get("CAPITAL_PER_TRADE", 100.0)
                qty = capital / price
                
                self.logger.info(f"  [Order] SUBMITTING LIMIT BUY: {s} | {qty:.4f} units @ ${price:.2f}")
                
                if self.config.get("DRY_RUN"):
                    self.active_positions[s] = {"amount": qty, "entry_price": price, "exit_name": sig["exit_name"]}
                    continue
                    
                try:
                    # Place Limit Buy
                    limit_req = LimitOrderRequest(
                        symbol=s, qty=qty, side=OrderSide.BUY, 
                        time_in_force=TimeInForce.GTC, limit_price=round(price, 2)
                    )
                    order = self.trading_client.submit_order(limit_req)
                    
                    # Track as active immediately (simpler for daily bars)
                    self.active_positions[s] = {
                        "entry_order_id": str(order.id),
                        "entry_price": price,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "amount": qty,
                        "exit_name": sig["exit_name"],
                        "stop_pct": float(self.per_coin_params[s]["params"].get("stop_pct", 3.0))
                    }
                    
                    # Mirror to DB and CSV
                    self.tracker.record_buy(
                        symbol=s, 
                        order_id=str(order.id), 
                        price=price, 
                        amount=qty, 
                        amount_usd=capital, 
                        fee=0.0, 
                        fee_currency="USD"
                    )
                    
                    self.save_state()
                    self._notify(f"🟢 <b>BUY {s} (Alpaca)</b> @ <code>${price:.2f}</code>")
                except Exception as e:
                    self.logger.error(f"  [Order] Alpaca buy failed for {s}: {e!r}")

            # 2. Handle Strategy Exit Signals (for non-TSL managed trades)
            elif s in self.active_positions and sig["exit"]:
                pos = self.active_positions[s]
                if pos.get("exit_name") == "fixed_sl_tp":
                    self.logger.info(f"  [Order] SUBMITTING MARKET SELL (Strategy Exit): {s}")
                    if self.config.get("DRY_RUN"):
                        del self.active_positions[s]
                        continue
                        
                    try:
                        # Cancel existing orders for symbol
                        self.trading_client.cancel_orders() # Simple: cancel all. Better: filter by symbol.
                        
                        sell_req = MarketOrderRequest(
                            symbol=s, qty=pos["amount"], side=OrderSide.SELL, time_in_force=TimeInForce.GTC
                        )
                        self.trading_client.submit_order(sell_req)
                        
                        self.tracker.record_sell(
                            symbol=s, order_id="manual_exit", price=sig["current_price"],
                            amount=pos["amount"], amount_usd=sig["current_price"] * pos["amount"],
                            fee=0.0, entry_price=pos.get("entry_price"), entry_time=pos.get("entry_time"),
                            exit_reason="strategy_signal"
                        )
                        del self.active_positions[s]
                        self.save_state()
                        self._notify(f"🔴 <b>SELL {s} (Alpaca)</b> @ <code>${sig['current_price']:.2f}</code>")
                    except Exception as e:
                        self.logger.error(f"  [Order] Alpaca sell failed for {s}: {e!r}")

    def run_event_loop(self) -> None:
        """Daily stock trading loop: wait for close -> evaluate -> sleep."""
        self.state = "EVENT_LOOP"
        self.logger.info(f"ggTrader Stock Engine Started (PAPER={self.config.get('PAPER')})")
        
        try:
            while True:
                # 1. Sync & Risk
                self._reconcile_positions()
                self._check_circuit_breaker()
                
                # 2. Check Market State
                if not self.market_hours.is_market_open():
                    # Align to next market close
                    self.market_hours.wait_for_market_open()
                
                self.market_hours.wait_for_market_close()
                
                # 3. Daily Execution (Market has just closed)
                self.logger.info("  [Cycle] Market closed. Evaluating signals...")
                
                # Snapshot balance
                total_usd = self._get_total_portfolio_usd()
                if total_usd:
                    # Alpaca usually has near-zero free cash if fully invested, but we fetch it
                    account = self.trading_client.get_account()
                    free_usd = float(account.cash)
                    self.tracker.record_balance_snapshot(total_usd, free_usd, total_usd - free_usd, len(self.active_positions))

                # Process Signals
                df = self._fetch_latest_data()
                if not df.empty:
                    # Compute Macro Regime
                    spy_mask = _compute_spy_regime_mask(df, self.config)
                    vix_mask = _compute_vix_regime_mask(df, self.config)
                    
                    is_bull = True
                    if spy_mask is not None: is_bull = is_bull and bool(spy_mask.iloc[-1])
                    if vix_mask is not None: is_bull = is_bull and bool(vix_mask.iloc[-1])
                    
                    allowance = {s: is_bull for s in self.symbols}
                    signals = self._compute_latest_signals(df)
                    
                    self._execute_trade_logic(signals, allowance)
                
                self.logger.info("  [Cycle] Daily evaluation complete. Sleeping until tomorrow.")
                time.sleep(3600) # Safety sleep before re-checking clock

        except KeyboardInterrupt:
            self.state = "STOPPED"
            self.logger.info("Stock Engine stopped by user.")
