import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from models import Position

class PortfolioManager(ABC):
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        pass

    @abstractmethod
    def get_total_value(self) -> float:
        pass

    @abstractmethod
    def open_position(self, symbol: str, position_size: float, price: float) -> None:
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> None:
        pass

    @abstractmethod
    def update_stop_loss(self, symbol: str, new_stop: float) -> None:
        pass

class SimulatedPortfolioManager(PortfolioManager):
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades_history: List[Dict] = []

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_total_value(self) -> float:
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def open_position(self, symbol: str, position_size: float, price: float) -> None:
        quantity = position_size / price
        self.cash -= position_size
        
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            stop_loss=price * 0.95  # Default 5% stop loss
        )
        
        self.trades_history.append({
            'timestamp': pd.Timestamp.now(),
            'action': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': position_size
        })

    def close_position(self, symbol: str) -> None:
        if symbol in self.positions:
            position = self.positions[symbol]
            value = position.quantity * position.current_price
            self.cash += value
            
            self.trades_history.append({
                'timestamp': pd.Timestamp.now(),
                'action': 'SELL',
                'symbol': symbol,
                'quantity': position.quantity,
                'price': position.current_price,
                'value': value
            })
            
            del self.positions[symbol]

    def update_stop_loss(self, symbol: str, new_stop: float) -> None:
        if symbol in self.positions:
            self.positions[symbol].stop_loss = new_stop

# portfolio_manager.py
class BinancePortfolioManager(PortfolioManager):
    def __init__(self, api_key: str, api_secret: str):
        from binance.client import Client
        self.client = Client(api_key, api_secret)
        self.positions: Dict[str, Position] = {}

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            account_info = self.client.get_account()
            for balance in account_info['balances']:
                if balance['asset'] == symbol.replace('USDT', ''):
                    quantity = float(balance['free']) + float(balance['locked'])
                    if quantity > 0:
                        # Get current price
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])

                        return Position(
                            symbol=symbol,
                            quantity=quantity,
                            entry_price=0,  # Would need to calculate from trades
                            current_price=current_price,
                            stop_loss=0  # Would need to maintain this separately
                        )
            return None
        except Exception as e:
            logging.error(f"Error getting position: {e}")
            return None

    def get_total_value(self) -> float:
        try:
            account_info = self.client.get_account()
            total_value = 0
            for balance in account_info['balances']:
                if float(balance['free']) + float(balance['locked']) > 0:
                    if balance['asset'] == 'USDT':
                        total_value += float(balance['free']) + float(balance['locked'])
                    else:
                        # Get current price in USDT
                        symbol = f"{balance['asset']}USDT"
                        try:
                            ticker = self.client.get_symbol_ticker(symbol=symbol)
                            price = float(ticker['price'])
                            total_value += (float(balance['free']) + float(balance['locked'])) * price
                        except:
                            pass
            return total_value
        except Exception as e:
            logging.error(f"Error getting total value: {e}")
            return 0

    def open_position(self, symbol: str, position_size: float, price: float) -> None:
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quoteOrderQty=position_size
            )
            logging.info(f"Opened position: {order}")
        except Exception as e:
            logging.error(f"Error opening position: {e}")

    def close_position(self, symbol: str) -> None:
        try:
            position = self.get_position(symbol)
            if position:
                order = self.client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=position.quantity
                )
                logging.info(f"Closed position: {order}")
        except Exception as e:
            logging.error(f"Error closing position: {e}")

    def update_stop_loss(self, symbol: str, new_stop: float) -> None:
        # Implement exchange-specific stop loss order
        pass