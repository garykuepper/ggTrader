"""NYSE market hours awareness via Alpaca clock API."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from alpaca.trading.client import TradingClient


class MarketHours:
    """Utility to track NYSE market state and handle sleep-until-open logic."""

    def __init__(self, trading_client: TradingClient):
        self.client = trading_client
        self.logger = logging.getLogger("ggTraderLive")

    def is_market_open(self) -> bool:
        """Check if the market is currently open for trading."""
        try:
            clock = self.client.get_clock()
            return clock.is_open
        except Exception as e:
            self.logger.error(f"  [Clock] Failed to fetch market clock: {e!r}")
            return False

    def wait_for_market_open(self) -> None:
        """Block execution until the market opens."""
        try:
            clock = self.client.get_clock()
            if clock.is_open:
                return

            next_open = clock.next_open
            # Convert next_open to local/naive for calculation if needed, 
            # but Alpaca returns aware datetimes.
            now = datetime.now(timezone.utc)
            wait_seconds = (next_open - now).total_seconds()

            if wait_seconds > 0:
                # Add 1 minute buffer to ensure the exchange has actually transitioned
                wait_seconds += 60
                hours = int(wait_seconds // 3600)
                minutes = int((wait_seconds % 3600) // 60)
                
                self.logger.info(
                    f"  [Clock] Market is CLOSED. Waiting {hours}h {minutes}m "
                    f"until next open ({next_open.strftime('%Y-%m-%d %H:%M:%S')} UTC)."
                )
                time.sleep(wait_seconds)
        except Exception as e:
            self.logger.error(f"  [Clock] Error during wait: {e!r}")
            # Fallback sleep to avoid tight loops on API failure
            time.sleep(300)

    def wait_for_market_close(self) -> None:
        """Block execution until the market closes (useful for daily bar alignment)."""
        try:
            clock = self.client.get_clock()
            if not clock.is_open:
                return

            next_close = clock.next_close
            now = datetime.now(timezone.utc)
            wait_seconds = (next_close - now).total_seconds()

            if wait_seconds > 0:
                # Add 5 minute buffer to ensure daily candle is finalized by providers
                wait_seconds += 300
                self.logger.info(
                    f"  [Clock] Market is OPEN. Waiting until close at "
                    f"{next_close.strftime('%Y-%m-%d %H:%M:%S')} UTC to evaluate signals."
                )
                time.sleep(wait_seconds)
        except Exception as e:
            self.logger.error(f"  [Clock] Error during wait for close: {e!r}")
            time.sleep(300)
