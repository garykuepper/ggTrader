from portfolio import PortfolioManager,  BinancePortfolioManager

# Example with real trading
api_key = "your_api_key"
api_secret = "your_api_secret"
portfolio_manager = BinancePortfolioManager(api_key, api_secret)

trading_system = TradingSystem(
    symbol="BTCUSDT",
    strategy_params=params,
    portfolio_manager=portfolio_manager,
    data_manager=data_manager
)

# Run live trading
trading_system.run_live(update_interval=60)
