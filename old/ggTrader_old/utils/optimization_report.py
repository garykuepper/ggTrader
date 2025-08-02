import pandas as pd
from tabulate import tabulate
import numpy as np


class OptimizationReport:
    """Generate comprehensive optimization reports"""

    def __init__(self):
        pass

    def print_data_summary(self, df):
        """Print summary of the data being used"""
        print(f"\n## 📊 DATA SUMMARY")

        start_date = df.index[0].strftime('%Y-%m-%d %H:%M')
        end_date = df.index[-1].strftime('%Y-%m-%d %H:%M')
        duration_days = (df.index[-1] - df.index[0]).days
        duration_hours = len(df)

        price_range_min = df['close'].min()
        price_range_max = df['close'].max()

        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * 100

        total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100

        summary_data = [
            ["Data Points", f"{len(df):,}"],
            ["Date Range", f"{start_date} to {end_date}"],
            ["Duration", f"{duration_days} days ({duration_hours:,} candles)"],
            ["Price Range", f"${price_range_min:.2f} - ${price_range_max:.2f}"],
            ["Buy & Hold Return", f"{total_return:+.2f}%"],
            ["Volatility", f"{volatility:.2f}%"]
        ]

        print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="github"))

    def format_results_table(self, results, title="OPTIMIZATION RESULTS", limit=10):
        """Format optimization results table with strategy type detection"""
        if not results:
            print(f"\n## {title}")
            print("No results to display")
            return

        print(f"\n## {title}")

        # Detect strategy type based on available parameters
        first_result = results[0]

        if 'fast_sma' in first_result and 'slow_sma' in first_result:
            # SMA Strategy
            table_data = []
            for i, result in enumerate(results[:limit]):
                table_data.append([
                    i + 1,
                    result['fast_sma'],
                    result['slow_sma'],
                    f"{result['position_pct']:.0%}",
                    f"${result['final_value']:,.0f}",
                    f"{result['total_return']:+.1f}%",
                    f"{result['sharpe_ratio']:.2f}",
                    f"{result['max_drawdown']:.1f}%",
                    result['total_trades'],
                    f"{result['win_rate']:.1f}%"
                ])

            headers = ["#", "Fast", "Slow", "Size", "Final $", "Return", "Sharpe", "DD", "Trades", "Win%"]

        elif 'rsi_period' in first_result and 'bb_period' in first_result:
            # Mean Reversion Strategy
            table_data = []
            for i, result in enumerate(results[:limit]):
                table_data.append([
                    i + 1,
                    result['rsi_period'],
                    result['rsi_oversold'],
                    result['bb_period'],
                    f"{result['position_pct']:.0%}",
                    f"${result['final_value']:,.0f}",
                    f"{result['total_return']:+.1f}%",
                    f"{result['sharpe_ratio']:.2f}",
                    f"{result['max_drawdown']:.1f}%",
                    result['total_trades'],
                    f"{result['win_rate']:.1f}%"
                ])

            headers = ["#", "RSI", "O.Sold", "BB", "Size", "Final $", "Return", "Sharpe", "DD", "Trades", "Win%"]

        elif 'atr_period' in first_result and 'breakout_lookback' in first_result:
            # Momentum Breakout Strategy
            table_data = []
            for i, result in enumerate(results[:limit]):
                table_data.append([
                    i + 1,
                    result['atr_period'],
                    f"{result['atr_multiplier']:.1f}",
                    f"{result['ema_fast']}-{result['ema_slow']}",
                    result['breakout_lookback'],
                    f"{result['position_pct']:.0%}",
                    f"${result['final_value']:,.0f}",
                    f"{result['total_return']:+.1f}%",
                    f"{result['sharpe_ratio']:.2f}",
                    f"{result['max_drawdown']:.1f}%",
                    result['total_trades'],
                    f"{result['win_rate']:.1f}%"
                ])

            headers = ["#", "ATR", "Mult", "EMA", "B.Out", "Size", "Final $", "Return", "Sharpe", "DD", "Trades",
                       "Win%"]

        elif 'trend_ma' in first_result and 'fast_ma' in first_result:
            # Multi-Timeframe Strategy
            table_data = []
            for i, result in enumerate(results[:limit]):
                table_data.append([
                    i + 1,
                    f"{result['fast_ma']}-{result['slow_ma']}",
                    result['trend_ma'],
                    result['rsi_period'],
                    result['rsi_oversold'],
                    f"{result['position_pct']:.0%}",
                    f"${result['final_value']:,.0f}",
                    f"{result['total_return']:+.1f}%",
                    f"{result['sharpe_ratio']:.2f}",
                    f"{result['max_drawdown']:.1f}%",
                    result['total_trades'],
                    f"{result['win_rate']:.1f}%"
                ])

            headers = ["#", "MA", "Trend", "RSI", "O.Sold", "Size", "Final $", "Return", "Sharpe", "DD", "Trades",
                       "Win%"]

        else:
            # Generic strategy - use available parameters
            table_data = []
            param_keys = [k for k in first_result.keys() if
                          k not in ['final_value', 'total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades',
                                    'win_rate', 'sqn']]

            for i, result in enumerate(results[:limit]):
                row = [i + 1]
                # Add parameter values
                for key in param_keys[:3]:  # Limit to first 3 parameters
                    row.append(result[key])

                # Add performance metrics
                row.extend([
                    f"${result['final_value']:,.0f}",
                    f"{result['total_return']:+.1f}%",
                    f"{result['sharpe_ratio']:.2f}",
                    f"{result['max_drawdown']:.1f}%",
                    result['total_trades'],
                    f"{result['win_rate']:.1f}%"
                ])
                table_data.append(row)

            # Create headers dynamically
            headers = ["#"] + [k.title() for k in param_keys[:3]] + ["Final $", "Return", "Sharpe", "DD", "Trades",
                                                                     "Win%"]

        print(tabulate(table_data, headers=headers, tablefmt="github"))

    def format_best_performers_table(self, results):
        """Format table of best performers by different metrics"""
        if not results:
            return

        print(f"\n## 🏆 BEST PERFORMERS BY METRIC")

        # Sort by different metrics
        best_return = max(results, key=lambda x: x['total_return'])
        best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
        best_trades = max(results, key=lambda x: x['total_trades'])
        best_winrate = max(results, key=lambda x: x['win_rate'])
        min_drawdown = min(results, key=lambda x: x['max_drawdown'])

        # Detect strategy type for parameter display
        first_result = results[0]

        if 'ema_fast' in first_result and 'ema_slow' in first_result and 'macd_fast' in first_result:
            # EMA MACD Strategy
            performers_data = [
                ["Highest Return", f"{best_return['total_return']:+.1f}%",
                 f"EMA: {best_return['ema_fast']}-{best_return['ema_slow']}, MACD: {best_return['macd_fast']}-{best_return['macd_slow']}-{best_return['macd_signal']}",
                 f"${best_return['final_value']:,.0f}"],
                ["Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}",
                 f"EMA: {best_sharpe['ema_fast']}-{best_sharpe['ema_slow']}, MACD: {best_sharpe['macd_fast']}-{best_sharpe['macd_slow']}-{best_sharpe['macd_signal']}",
                 f"${best_sharpe['final_value']:,.0f}"],
                ["Most Trades", f"{best_trades['total_trades']}",
                 f"EMA: {best_trades['ema_fast']}-{best_trades['ema_slow']}, MACD: {best_trades['macd_fast']}-{best_trades['macd_slow']}-{best_trades['macd_signal']}",
                 f"${best_trades['final_value']:,.0f}"],
                ["Best Win Rate", f"{best_winrate['win_rate']:.1f}%",
                 f"EMA: {best_winrate['ema_fast']}-{best_winrate['ema_slow']}, MACD: {best_winrate['macd_fast']}-{best_winrate['macd_slow']}-{best_winrate['macd_signal']}",
                 f"${best_winrate['final_value']:,.0f}"],
                ["Min Drawdown", f"{min_drawdown['max_drawdown']:.1f}%",
                 f"EMA: {min_drawdown['ema_fast']}-{min_drawdown['ema_slow']}, MACD: {min_drawdown['macd_fast']}-{min_drawdown['macd_slow']}-{min_drawdown['macd_signal']}",
                 f"${min_drawdown['final_value']:,.0f}"]
            ]

        elif 'fast_sma' in first_result and 'slow_sma' in first_result:
            # SMA Strategy
            performers_data = [
                ["Highest Return", f"{best_return['total_return']:+.1f}%",
                 f"Fast: {best_return['fast_sma']}, Slow: {best_return['slow_sma']}",
                 f"${best_return['final_value']:,.0f}"],
                ["Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}",
                 f"Fast: {best_sharpe['fast_sma']}, Slow: {best_sharpe['slow_sma']}",
                 f"${best_sharpe['final_value']:,.0f}"],
                ["Most Trades", f"{best_trades['total_trades']}",
                 f"Fast: {best_trades['fast_sma']}, Slow: {best_trades['slow_sma']}",
                 f"${best_trades['final_value']:,.0f}"],
                ["Best Win Rate", f"{best_winrate['win_rate']:.1f}%",
                 f"Fast: {best_winrate['fast_sma']}, Slow: {best_winrate['slow_sma']}",
                 f"${best_winrate['final_value']:,.0f}"],
                ["Min Drawdown", f"{min_drawdown['max_drawdown']:.1f}%",
                 f"Fast: {min_drawdown['fast_sma']}, Slow: {min_drawdown['slow_sma']}",
                 f"${min_drawdown['final_value']:,.0f}"],
            ]
        elif 'atr_period' in first_result and 'breakout_lookback' in first_result:
            # Momentum Breakout Strategy
            performers_data = [
                ["Highest Return", f"{best_return['total_return']:+.1f}%",
                 f"ATR: {best_return['atr_period']}, Mult: {best_return['atr_multiplier']:.1f}, EMA: {best_return['ema_fast']}-{best_return['ema_slow']}",
                 f"${best_return['final_value']:,.0f}"],
                ["Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}",
                 f"ATR: {best_sharpe['atr_period']}, Mult: {best_sharpe['atr_multiplier']:.1f}, EMA: {best_sharpe['ema_fast']}-{best_sharpe['ema_slow']}",
                 f"${best_sharpe['final_value']:,.0f}"],
                ["Most Trades", f"{best_trades['total_trades']}",
                 f"ATR: {best_trades['atr_period']}, Mult: {best_trades['atr_multiplier']:.1f}, EMA: {best_trades['ema_fast']}-{best_trades['ema_slow']}",
                 f"${best_trades['final_value']:,.0f}"],
                ["Best Win Rate", f"{best_winrate['win_rate']:.1f}%",
                 f"ATR: {best_winrate['atr_period']}, Mult: {best_winrate['atr_multiplier']:.1f}, EMA: {best_winrate['ema_fast']}-{best_winrate['ema_slow']}",
                 f"${best_winrate['final_value']:,.0f}"],
                ["Min Drawdown", f"{min_drawdown['max_drawdown']:.1f}%",
                 f"ATR: {min_drawdown['atr_period']}, Mult: {min_drawdown['atr_multiplier']:.1f}, EMA: {min_drawdown['ema_fast']}-{min_drawdown['ema_slow']}",
                 f"${min_drawdown['final_value']:,.0f}"],
            ]
        elif 'rsi_period' in first_result and 'bb_period' in first_result:
            # Mean Reversion Strategy
            performers_data = [
                ["Highest Return", f"{best_return['total_return']:+.1f}%",
                 f"RSI: {best_return['rsi_period']}, OS: {best_return['rsi_oversold']}, BB: {best_return['bb_period']}",
                 f"${best_return['final_value']:,.0f}"],
                ["Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}",
                 f"RSI: {best_sharpe['rsi_period']}, OS: {best_sharpe['rsi_oversold']}, BB: {best_sharpe['bb_period']}",
                 f"${best_sharpe['final_value']:,.0f}"],
                ["Most Trades", f"{best_trades['total_trades']}",
                 f"RSI: {best_trades['rsi_period']}, OS: {best_trades['rsi_oversold']}, BB: {best_trades['bb_period']}",
                 f"${best_trades['final_value']:,.0f}"],
                ["Best Win Rate", f"{best_winrate['win_rate']:.1f}%",
                 f"RSI: {best_winrate['rsi_period']}, OS: {best_winrate['rsi_oversold']}, BB: {best_winrate['bb_period']}",
                 f"${best_winrate['final_value']:,.0f}"],
                ["Min Drawdown", f"{min_drawdown['max_drawdown']:.1f}%",
                 f"RSI: {min_drawdown['rsi_period']}, OS: {min_drawdown['rsi_oversold']}, BB: {min_drawdown['bb_period']}",
                 f"${min_drawdown['final_value']:,.0f}"],
            ]
        else:
            # Generic strategy fallback
            param_keys = [k for k in first_result.keys() if
                          k not in ['final_value', 'total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades',
                                    'win_rate', 'sqn']]
            performers_data = [
                ["Highest Return", f"{best_return['total_return']:+.1f}%", "Various",
                 f"${best_return['final_value']:,.0f}"],
                ["Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}", "Various", f"${best_sharpe['final_value']:,.0f}"],
                ["Most Trades", f"{best_trades['total_trades']}", "Various", f"${best_trades['final_value']:,.0f}"],
                ["Best Win Rate", f"{best_winrate['win_rate']:.1f}%", "Various",
                 f"${best_winrate['final_value']:,.0f}"],
                ["Min Drawdown", f"{min_drawdown['max_drawdown']:.1f}%", "Various",
                 f"${min_drawdown['final_value']:,.0f}"],
            ]

        print(tabulate(performers_data, headers=["Metric", "Value", "Parameters", "Final Value"], tablefmt="github"))

    def format_summary_stats_table(self, results):
        """Format summary statistics table"""
        if not results:
            return

        print(f"\n## 📈 SUMMARY STATISTICS")

        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        trades = [r['total_trades'] for r in results]
        win_rates = [r['win_rate'] for r in results]

        stats_data = [
            ["Total Combinations", f"{len(results):,}"],
            ["Avg Return", f"{np.mean(returns):+.2f}%"],
            ["Best Return", f"{np.max(returns):+.2f}%"],
            ["Worst Return", f"{np.min(returns):+.2f}%"],
            ["Std Dev Returns", f"{np.std(returns):.2f}%"],
            ["Avg Sharpe Ratio", f"{np.mean(sharpes):.3f}"],
            ["Best Sharpe", f"{np.max(sharpes):.3f}"],
            ["Avg Max Drawdown", f"{np.mean(drawdowns):.2f}%"],
            ["Worst Drawdown", f"{np.max(drawdowns):.2f}%"],
            ["Avg Trades", f"{np.mean(trades):.1f}"],
            ["Avg Win Rate", f"{np.mean(win_rates):.1f}%"]
        ]

        print(tabulate(stats_data, headers=["Statistic", "Value"], tablefmt="github"))

    def print_performance_distribution(self, results):
        """Print performance distribution analysis"""
        if not results:
            return

        print(f"\n## 📊 PERFORMANCE DISTRIBUTION")

        returns = [r['total_return'] for r in results]

        # Create return buckets
        buckets = [
            ("< -10%", len([r for r in returns if r < -10])),
            ("-10% to -5%", len([r for r in returns if -10 <= r < -5])),
            ("-5% to 0%", len([r for r in returns if -5 <= r < 0])),
            ("0% to 5%", len([r for r in returns if 0 <= r < 5])),
            ("5% to 10%", len([r for r in returns if 5 <= r < 10])),
            ("10% to 20%", len([r for r in returns if 10 <= r < 20])),
            ("> 20%", len([r for r in returns if r >= 20]))
        ]

        bucket_data = []
        for bucket_name, count in buckets:
            percentage = (count / len(results)) * 100
            bar = "█" * int(percentage / 2)  # Scale bar length
            bucket_data.append([bucket_name, count, f"{percentage:.1f}%", bar])

        print(tabulate(bucket_data, headers=["Return Range", "Count", "Percentage", "Distribution"], tablefmt="github"))

    def print_optimization_analysis(self, results):
        """Print optimization analysis and insights"""
        if not results:
            return

        print(f"\n## 🔍 OPTIMIZATION ANALYSIS")

        profitable_count = len([r for r in results if r['total_return'] > 0])
        total_count = len(results)

        if profitable_count > 0:
            avg_profitable_return = np.mean([r['total_return'] for r in results if r['total_return'] > 0])
            best_result = max(results, key=lambda x: x['total_return'])

            print(f"✅ **{profitable_count}/{total_count}** parameter combinations were profitable ({profitable_count / total_count * 100:.1f}%)")
            print(f"📈 **Average profitable return:** {avg_profitable_return:+.1f}%")
            print(f"🎯 **Best strategy return:** {best_result['total_return']:+.1f}%")

            # Strategy-specific parameter analysis
            first_result = results[0]

            if 'ema_fast' in first_result and 'ema_slow' in first_result and 'macd_fast' in first_result:
                # EMA MACD analysis
                print(f"🔧 **Best parameters:** EMA: {best_result['ema_fast']}-{best_result['ema_slow']}, MACD: {best_result['macd_fast']}-{best_result['macd_slow']}-{best_result['macd_signal']}, Position: {best_result['position_pct']:.0%}, Stop: {best_result['stop_loss_pct']:.0%}")
            elif 'fast_sma' in first_result:
                # SMA analysis
                print(f"🔧 **Best parameters:** Fast SMA: {best_result['fast_sma']}, Slow SMA: {best_result['slow_sma']}, Position: {best_result['position_pct']:.0%}")
            elif 'atr_period' in first_result and 'breakout_lookback' in first_result:
                # Momentum Breakout analysis
                print(f"🔧 **Best parameters:** ATR: {best_result['atr_period']}, Mult: {best_result['atr_multiplier']:.1f}, EMA: {best_result['ema_fast']}-{best_result['ema_slow']}, Breakout: {best_result['breakout_lookback']}, Position: {best_result['position_pct']:.0%}")
            elif 'rsi_period' in first_result:
                # Mean reversion analysis
                print(f"🔧 **Best parameters:** RSI: {best_result['rsi_period']}, Oversold: {best_result['rsi_oversold']}, BB: {best_result['bb_period']}, Position: {best_result['position_pct']:.0%}")
            else:
                print("🔧 **Best parameters:** Various")
        else:
            print("❌ **No profitable parameter combinations found**")
            best_result = max(results, key=lambda x: x['total_return'])
            print(f"📉 **Best (least losing) return:** {best_result['total_return']:+.1f}%")

    def print_completion_summary(self):
        """Print completion summary"""
        print(f"\n## ✅ OPTIMIZATION COMPLETE")
        print("*Analysis ready for strategy implementation*")

    def print_optimization_config(self, fast_range, slow_range, position_range):
        """Print optimization configuration for SMA strategy"""
        print(f"\n## ⚙️ OPTIMIZATION CONFIGURATION")
        total_combinations = len(fast_range) * len(slow_range) * len(position_range)

        config_data = [
            ["Strategy", "SMA Crossover"],
            ["Fast SMA Range", f"{min(fast_range)} - {max(fast_range)}"],
            ["Slow SMA Range", f"{min(slow_range)} - {max(slow_range)}"],
            ["Position Sizes", ", ".join([f"{p:.0%}" for p in position_range])],
            ["Total Combinations", f"{total_combinations:,}"]
        ]

        print(tabulate(config_data, headers=["Parameter", "Value"], tablefmt="github"))