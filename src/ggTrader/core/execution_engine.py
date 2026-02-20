"""Live execution engine for transition from optimization to real-world trading."""

import time
import pandas as pd
from typing import Dict, Any, List

from ggTrader.data.live.exchange_loader import LiveExchangeLoader
from ggTrader.indicators.Signals import Signals


class ExecutionEngine:
    """
    Live trading bot orchestrator.
    Consumes optimized parameters from the WFO pipeline and executes trades on the exchange.
    """

    def __init__(self, config: Dict[str, Any], optimized_params: Dict[str, Any]):
        self.config = config
        self.params = optimized_params
        self.exchange = LiveExchangeLoader(exchange_id=config.get("EXCHANGE", "kraken"))

        self.state = "INITIALIZED"
        self.symbols = config.get("SYMBOLS", [])
        self.interval = config.get("INTERVAL", "4h")
        self.active_positions = {}

    def _fetch_latest_data(self) -> pd.DataFrame:
        """Fetch the most recent candles for calculating signals."""
        print(
            f"[{time.strftime('%X')}] Fetching latest {self.interval} data for {len(self.symbols)} symbols..."
        )
        # In a real scenario, we limit to the lookback required by our indicators (e.g., ADX + ATR padding)
        lookback_limit = 200
        return self.exchange.fetch_ohlcv(
            symbols=self.symbols, interval=self.interval, limit=lookback_limit
        )

    def _compute_latest_signals(self, ohlcv_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run the vectorbt compiled signals on the latest live data."""
        print(f"[{time.strftime('%X')}] Computing latest signals...")
        return Signals.calculate_ohlcv_signals(
            ohlcv_df=ohlcv_df,
            adx_length=self.params.get("adx_length", 14),
            adx_threshold=self.params.get("adx_threshold", 25),
            sar_acceleration=self.params.get("sar_acceleration", 0.02),
            sar_maximum=self.params.get("sar_maximum", 0.2),
            use_dmp_cross=self.params.get("use_dmp_cross", True),
            atr_length=self.params.get("atr_length", 14),
            atr_multiplier=self.params.get("atr_multiplier", 3.0),
        )

    def _check_balances(self):
        """Placeholder: Query live exchange balances."""
        print(f"[{time.strftime('%X')}] Checking live balances on {self.exchange.exchange_id}...")
        pass

    def _create_market_buy_order(self, symbol: str, amount: float):
        """Placeholder: Execute a live market buy order."""
        print(f"[{time.strftime('%X')}] EXECUTING MARKET BUY: {amount} {symbol}")
        pass

    def _create_trailing_stop_order(self, symbol: str, amount: float, stop_price: float):
        """Placeholder: Place or update a trailing stop order."""
        print(f"[{time.strftime('%X')}] UPDATING TRAILING STOP: {amount} {symbol} at {stop_price}")
        pass

    def _execute_trade_logic(self, signals_dict: Dict[str, pd.DataFrame]):
        """Evaluate signals and execute orders."""
        # Grab the absolute last row of signals for real-time decisions
        for symbol, df in signals_dict.items():
            if df.empty:
                continue

            latest = df.iloc[-1]
            signal = latest.get("signal", 0)
            stop = latest.get("stop_loss", 0.0)

            # Simple State Machine Transition
            if signal == 1 and symbol not in self.active_positions:
                # ENTRY
                self._create_market_buy_order(symbol, 1.0)  # Placeholder amount
                self.active_positions[symbol] = {"entry_price": latest["close"], "stop": stop}
                self._create_trailing_stop_order(symbol, 1.0, stop)

            elif symbol in self.active_positions:
                # IN POSITION - UPDATE OR EXIT
                if signal == -1:
                    print(f"[{time.strftime('%X')}] STOP LOSS HIT OR SIGNAL REVERSAL FOR {symbol}")
                    del self.active_positions[symbol]
                else:
                    current_stop = self.active_positions[symbol]["stop"]
                    if stop > current_stop:
                        self.active_positions[symbol]["stop"] = stop
                        self._create_trailing_stop_order(symbol, 1.0, stop)

    def run_event_loop(self, poll_interval_seconds: int = 60):
        """
        Transition from optimization to real-world event loop.
        Polls for new data, computes signals, and manages orders.
        """
        self.state = "EVENT_LOOP"
        print("=========================================")
        print("  ggTrader Live Execution Engine Started ")
        print("=========================================")

        self._check_balances()

        try:
            while True:
                # 1. Fetch live data
                latest_df = self._fetch_latest_data()

                if not latest_df.empty:
                    # 2. Compute signals based on optimal parameters
                    latest_signals = self._compute_latest_signals(latest_df)

                    # 3. Execute / Manage Trades
                    self._execute_trade_logic(latest_signals)

                # 4. Wait for next interval
                print(
                    f"[{time.strftime('%X')}] Polling loop complete. Sleeping for {poll_interval_seconds}s..."
                )
                time.sleep(poll_interval_seconds)

        except KeyboardInterrupt:
            self.state = "STOPPED"
            print("\nExecution Engine stopped by user.")
