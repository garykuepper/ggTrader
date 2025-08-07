# main.py
from models import StrategyParameters
from trading_strategy import TradingStrategy
from portfolio_manager import SimulatedPortfolioManager, PortfolioManager
from data_manager import CryptoDataManager
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

class TradingSystem:
    def __init__(self, 
                 symbol: str,
                 strategy_params: StrategyParameters,
                 portfolio_manager: PortfolioManager,
                 data_manager: CryptoDataManager,
                 max_position_pct: float = 0.1):
        
        self.symbol = symbol
        self.strategy = TradingStrategy(
            symbol=symbol,
            max_position_pct=max_position_pct,
            params=strategy_params,
            portfolio_manager=portfolio_manager,
            data_manager=data_manager
        )
        self.portfolio_manager = portfolio_manager
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run_live(self, update_interval: int = 60):
        """
        Run the trading system in live mode
        update_interval: seconds between updates
        """
        self.logger.info(f"Starting live trading for {self.symbol}")
        
        while True:
            try:
                current_time = pd.Timestamp.now()
                self.strategy.update(current_time)
                
                # Log portfolio status
                self._log_portfolio_status()
                
                # Wait for next update
                time.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in live trading: {e}")
                raise

    def run_backtest(self, start_date: datetime, end_date: datetime):
        """
        Run the trading system in backtest mode
        """
        self.logger.info(f"Starting backtest for {self.symbol} from {start_date} to {end_date}")
        
        current_date = start_date
        while current_date <= end_date:
            try:
                self.strategy.update(pd.Timestamp(current_date))
                current_date += timedelta(hours=1)  # Adjust based on your strategy interval
                
            except Exception as e:
                self.logger.error(f"Error in backtesting: {e}")
                raise
        
        self._log_backtest_results()

    def _log_portfolio_status(self):
        """Log current portfolio status"""
        total_value = self.portfolio_manager.get_total_value()
        position = self.portfolio_manager.get_position(self.symbol)
        
        self.logger.info(f"Portfolio Value: ${total_value:,.2f}")
        if position:
            self.logger.info(
                f"Position: {position.symbol} - "
                f"Qty: {position.quantity:.8f}, "
                f"Entry: ${position.entry_price:.2f}, "
                f"Current: ${position.current_price:.2f}, "
                f"Stop: ${position.stop_loss:.2f}"
            )

    def _log_backtest_results(self):
        """Log backtest results"""
        if isinstance(self.portfolio_manager, SimulatedPortfolioManager):
            initial_value = 100000  # Assuming this was the initial value
            final_value = self.portfolio_manager.get_total_value()
            returns_pct = ((final_value - initial_value) / initial_value) * 100
            
            self.logger.info("\n=== Backtest Results ===")
            self.logger.info(f"Initial Value: ${initial_value:,.2f}")
            self.logger.info(f"Final Value: ${final_value:,.2f}")
            self.logger.info(f"Return: {returns_pct:.2f}%")
            
            # Log all trades
            self.logger.info("\n=== Trade History ===")
            for trade in self.portfolio_manager.trades_history:
                self.logger.info(
                    f"{trade['timestamp']} - {trade['action']}: "
                    f"{trade['symbol']} - Qty: {trade['quantity']:.8f} "
                    f"@ ${trade['price']:.2f} = ${trade['value']:.2f}"
                )

def main():
    # Initialize components
    data_manager = CryptoDataManager()
    portfolio_manager = SimulatedPortfolioManager(initial_cash=100000)
    
    # Strategy parameters
    params = StrategyParameters(
        ema_windows=[9, 21],
        trailing_stop_pct=0.02,
        stop_loss_pct=0.05,
        interval='4h',
        lookback_days=30
    )
    
    # Create trading system
    trading_system = TradingSystem(
        symbol="BTCUSDT",
        strategy_params=params,
        portfolio_manager=portfolio_manager,
        data_manager=data_manager
    )
    
    # Example: Run backtest
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 2, 1)
    trading_system.run_backtest(start_date, end_date)
    
    # Example: Run live trading
    # trading_system.run_live(update_interval=60)

if __name__ == "__main__":
    main()
