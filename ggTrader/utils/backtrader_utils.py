import backtrader as bt
import pandas as pd
import numpy as np

class BacktraderUtils:
    """Utility class for common backtrader operations"""

    @staticmethod
    def create_backtrader_data(df):
        """Convert DataFrame to backtrader data format"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        bt_df = df.copy()
        bt_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = bt.feeds.PandasData(dataname=bt_df)
        return data

    @staticmethod
    def calculate_manual_sharpe(total_return, max_drawdown, num_trades, risk_free_rate=0.02):
        """Calculate Sharpe ratio manually when analyzer fails"""
        if num_trades < 2 or total_return <= 0:
            return 0

        estimated_volatility = max(max_drawdown / 100 * 2, 0.01)
        excess_return = total_return - risk_free_rate
        return excess_return / estimated_volatility if estimated_volatility > 0 else 0

    @staticmethod
    def setup_cerebro(initial_cash=10000.0, commission=0.001):
        """Setup basic cerebro configuration"""
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.broker.set_filler(bt.broker.fillers.FixedBarPerc(perc=100.0))
        return cerebro

    @staticmethod
    def add_standard_analyzers(cerebro, risk_free_rate=0.02):
        """Add standard analyzers to cerebro"""
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=risk_free_rate)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        return cerebro

    @staticmethod
    def extract_strategy_results(strategy, initial_value=10000.0):
        """Extract standard results from a strategy"""
        # Get analyzer results
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()
        sqn_analysis = strategy.analyzers.sqn.get_analysis()

        # Extract metrics
        sharpe_ratio = sharpe_analysis.get('sharperatio', 0) or 0
        total_return = returns.get('rtot', 0) or 0
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0) or 0
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        sqn = sqn_analysis.get('sqn', 0) or 0

        # Calculate derived metrics
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        final_value = initial_value * (1 + total_return)

        # Manual Sharpe calculation if needed
        if sharpe_ratio == 0 and total_trades > 0:
            sharpe_ratio = BacktraderUtils.calculate_manual_sharpe(
                total_return * 100, max_drawdown, total_trades
            )

        return {
            'final_value': final_value,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sqn': sqn
        }

    @staticmethod
    def validate_data_for_optimization(df, min_length=100):
        """Validate DataFrame for optimization"""
        if df.empty:
            return False, "No data available for optimization"

        if len(df) < min_length:
            return False, f"Insufficient data for optimization (need {min_length}, got {len(df)})"

        return True, "Data validation passed"

    @staticmethod
    def calculate_parameter_combinations(*param_ranges):
        """Calculate total number of parameter combinations"""
        total = 1
        for param_range in param_ranges:
            if hasattr(param_range, '__len__'):
                total *= len(param_range)
            else:
                total *= 1
        return total