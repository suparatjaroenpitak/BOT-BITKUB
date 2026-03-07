"""
Market Data Collector - ดึงข้อมูลราคาจาก Bitkub และเก็บเป็น DataFrame
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from exchange.bitkub_client import BitkubClient
from config import TradingConfig
from utils.logger import TradeLogger


class MarketDataCollector:
    """Collects and manages market data from Bitkub."""

    def __init__(self, client: BitkubClient, config: TradingConfig,
                 logger: Optional[TradeLogger] = None):
        self.client = client
        self.config = config
        self.logger = logger or TradeLogger()
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(self, symbol: str = "", timeframe: str = "",
                    limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data and return as DataFrame."""
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.default_timeframe

        raw = self.client.get_ohlcv(symbol, timeframe, limit)
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)

        cache_key = f"{symbol}_{timeframe}"
        self.data_cache[cache_key] = df

        return df

    def fetch_ticker(self, symbol: str = "") -> Dict:
        """Fetch current ticker."""
        symbol = symbol or self.config.symbol
        return self.client.get_ticker(symbol)

    def fetch_orderbook(self, symbol: str = "", limit: int = 10) -> Dict:
        """Fetch current order book."""
        symbol = symbol or self.config.symbol
        return self.client.get_orderbook(symbol, limit)

    def fetch_all_timeframes(self, symbol: str = "",
                             limit: int = 500) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all configured timeframes."""
        symbol = symbol or self.config.symbol
        result = {}

        for tf in self.config.timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit)
            if not df.empty:
                result[tf] = df
            time.sleep(0.5)  # Rate limiting

        return result

    def get_current_price(self, symbol: str = "") -> float:
        """Get the current price."""
        ticker = self.fetch_ticker(symbol)
        return ticker.get("last", 0.0)

    def get_cached_data(self, symbol: str = "", timeframe: str = "") -> pd.DataFrame:
        """Get cached OHLCV data."""
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.default_timeframe
        cache_key = f"{symbol}_{timeframe}"
        return self.data_cache.get(cache_key, pd.DataFrame())

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV."""
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        df.to_csv(filepath)
        self.logger.log_info(f"Data saved to {filepath}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
        return pd.DataFrame()

    def collect_and_save(self, symbol: str = "", timeframe: str = "",
                         limit: int = 500) -> pd.DataFrame:
        """Fetch data and save to CSV."""
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.default_timeframe

        df = self.fetch_ohlcv(symbol, timeframe, limit)
        if not df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.csv"
            self.save_data(df, filename)

        return df
