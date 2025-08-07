from typing import Dict, Optional
import pandas as pd
from models import Signal, StrategyParameters
from portfolio_manager import PortfolioManager
from data_manager import DataManager, CryptoDataManager

class TradingStrategy:
    def __init__(self, 
                 symbol: str, 
                 max_position_pct: float,
                 params: StrategyParameters,
                 portfolio_manager: PortfolioManager,
                 data_manager: DataManager):
        self.symbol = symbol
        self.max_position_pct = max_position_pct
        self.params = params
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager
        self.ema_values: Dict[int, float] = {}
        self.current_data: Optional[pd.DataFrame] = None

    def update(self, current_time: pd.Timestamp) -> None:
        """Update strategy with new data and generate signals"""
        self._update_data(current_time)
        self._update_emas()
        signal = self._generate_signal()
        current_price = self.current_data.iloc[-1]['close']
        self._check_stops(current_price)
        self._execute_signal(signal, current_price)

    def _update_data(self, current_time: pd.Timestamp) -> None:
        """Fetch and update market data"""
        start_time = current_time - pd.Timedelta(days=self.params.lookback_days)
        
        if isinstance(self.data_manager, CryptoDataManager):
            self.current_data = self.data_manager.get_crypto_data(
                symbol=self.symbol,
                interval=self.params.interval,
                start_date=start_time,
                end_date=current_time
            )
        else:  # StockDataManager
            self.current_data = self.data_manager.get_stock_data(
                symbol=self.symbol,
                interval=self.params.interval,
                start_date=start_time,
                end_date=current_time
            )

    def _update_emas(self) -> None:
        """Calculate EMAs using pandas"""
        if self.current_data is not None and not self.current_data.empty:
            for window in self.params.ema_windows:
                ema_key = f'ema_{window}'
                self.current_data[ema_key] = self.current_data['close'].ewm(
                    span=window, adjust=False).mean()
                self.ema_values[window] = self.current_data[ema_key].iloc[-1]

    def _generate_signal(self) -> Signal:
        """Generate trading signal based on strategy rules"""
        if len(self.ema_values) < 2:
            return Signal.HOLD
            
        fast_ema = self.ema_values[min(self.params.ema_windows)]
        slow_ema = self.ema_values[max(self.params.ema_windows)]
        
        if fast_ema > slow_ema:
            return Signal.BUY
        elif fast_ema < slow_ema:
            return Signal.SELL
        
        return Signal.HOLD

    def _check_stops(self, current_price: float) -> None:
        """Check and update stop losses"""
        position = self.portfolio_manager.get_position(self.symbol)
        if position:
            if current_price > position.entry_price:
                new_stop = current_price * (1 - self.params.trailing_stop_pct)
                if new_stop > position.stop_loss:
                    self.portfolio_manager.update_stop_loss(self.symbol, new_stop)
            
            if current_price <= position.stop_loss:
                self.portfolio_manager.close_position(self.symbol)

    def _execute_signal(self, signal: Signal, current_price: float) -> None:
        """Execute trading signal"""
        if signal == Signal.BUY and not self.portfolio_manager.get_position(self.symbol):
            position_size = self.portfolio_manager.get_total_value() * self.max_position_pct
            self.portfolio_manager.open_position(self.symbol, position_size, current_price)
        elif signal == Signal.SELL:
            self.portfolio_manager.close_position(self.symbol)
