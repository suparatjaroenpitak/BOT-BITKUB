"""
Technical Indicator Engine - คำนวณ RSI, MACD, Bollinger Bands, MA, EMA
"""
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands


class TechnicalIndicatorEngine:
    """Calculate technical indicators for trading signals."""

    def __init__(self, support_resistance_window: int = 20):
        self.support_resistance_window = max(5, int(support_resistance_window))

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame."""
        df = df.copy()
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_moving_averages(df)
        df = self.add_ema(df)
        df = self.add_support_resistance(df, window=self.support_resistance_window)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        df = df.copy()
        df["rsi"] = RSIIndicator(
            close=df["close"], window=period
        ).rsi()
        return df

    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        df = df.copy()
        macd = MACD(
            close=df["close"], window_slow=slow, window_fast=fast, window_sign=signal
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20,
                            std_dev: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df = df.copy()
        bb = BollingerBands(
            close=df["close"], window=period, window_dev=std_dev
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()
        return df

    def add_moving_averages(self, df: pd.DataFrame,
                            periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        df = df.copy()
        if periods is None:
            periods = [7, 20, 50, 200]
        for period in periods:
            df[f"sma_{period}"] = SMAIndicator(
                close=df["close"], window=period
            ).sma_indicator()
        return df

    def add_ema(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        df = df.copy()
        if periods is None:
            periods = [9, 21, 50, 200]
        for period in periods:
            df[f"ema_{period}"] = EMAIndicator(
                close=df["close"], window=period
            ).ema_indicator()
        return df

    def add_volatility_indicators(self, df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """Add volatility metrics used for adaptive risk controls."""
        df = df.copy()
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=atr_period
        )
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = np.where(df["close"] > 0, (df["atr"] / df["close"]) * 100, 0.0)
        df["price_change_1_pct"] = np.where(
            df["close"].shift(1) > 0,
            ((df["close"] - df["close"].shift(1)) / df["close"].shift(1)) * 100,
            0.0,
        )
        df["price_change_3_pct"] = np.where(
            df["close"].shift(3) > 0,
            ((df["close"] - df["close"].shift(3)) / df["close"].shift(3)) * 100,
            0.0,
        )
        df["candle_body_pct"] = np.where(
            df["open"] > 0,
            ((df["close"] - df["open"]) / df["open"]) * 100,
            0.0,
        )
        df["close_to_low_pct"] = np.where(
            df["close"] > 0,
            ((df["close"] - df["low"]) / df["close"]) * 100,
            0.0,
        )
        return df

    def add_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add rolling support and resistance from prior candles."""
        df = df.copy()
        rolling_low = df["low"].rolling(window=window, min_periods=window).min().shift(1)
        rolling_high = df["high"].rolling(window=window, min_periods=window).max().shift(1)
        df["support_level"] = rolling_low
        df["resistance_level"] = rolling_high
        df["support_distance_pct"] = np.where(
            rolling_low > 0,
            ((df["close"] - rolling_low) / rolling_low) * 100,
            np.nan,
        )
        df["resistance_distance_pct"] = np.where(
            rolling_high > 0,
            ((rolling_high - df["close"]) / rolling_high) * 100,
            np.nan,
        )
        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        df = df.copy()
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        return df

    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """Get a summary of current signals from the latest row."""
        if df.empty:
            return {}

        latest = df.iloc[-1]
        signals = {
            "price": latest.get("close", 0),
            "rsi": latest.get("rsi", 50),
            "macd": latest.get("macd", 0),
            "macd_signal": latest.get("macd_signal", 0),
            "macd_histogram": latest.get("macd_histogram", 0),
            "bb_upper": latest.get("bb_upper", 0),
            "bb_lower": latest.get("bb_lower", 0),
            "bb_middle": latest.get("bb_middle", 0),
            "bb_width": latest.get("bb_width", 0),
            "ema_9": latest.get("ema_9", 0),
            "ema_21": latest.get("ema_21", 0),
            "ema_50": latest.get("ema_50", 0),
            "sma_50": latest.get("sma_50", 0),
            "support_level": latest.get("support_level", 0),
            "resistance_level": latest.get("resistance_level", 0),
            "support_distance_pct": latest.get("support_distance_pct", 0),
            "resistance_distance_pct": latest.get("resistance_distance_pct", 0),
            "atr": latest.get("atr", 0),
            "atr_pct": latest.get("atr_pct", 0),
            "price_change_1_pct": latest.get("price_change_1_pct", 0),
            "price_change_3_pct": latest.get("price_change_3_pct", 0),
            "candle_body_pct": latest.get("candle_body_pct", 0),
            "close_to_low_pct": latest.get("close_to_low_pct", 0),
            "volume_ratio": latest.get("volume_ratio", 1),
        }

        # Signal interpretations
        signals["rsi_oversold"] = signals["rsi"] < 35
        signals["rsi_overbought"] = signals["rsi"] > 70
        signals["price_below_ema"] = signals["price"] < signals["ema_21"]
        signals["price_above_bb_upper"] = signals["price"] > signals["bb_upper"]
        signals["price_below_bb_lower"] = signals["price"] < signals["bb_lower"]
        signals["macd_bullish"] = signals["macd"] > signals["macd_signal"]
        signals["macd_bearish"] = signals["macd"] < signals["macd_signal"]
        signals["high_volume"] = signals["volume_ratio"] > 1.5
        signals["trend_up"] = signals["ema_9"] > signals["ema_21"] > signals["ema_50"] > 0
        signals["trend_down"] = signals["ema_9"] < signals["ema_21"] < signals["ema_50"] if signals["ema_50"] > 0 else False
        signals["bearish_pressure"] = (
            signals["candle_body_pct"] <= -0.8
            and signals["price_change_1_pct"] < 0
            and signals["close_to_low_pct"] <= 0.5
        )

        return signals
