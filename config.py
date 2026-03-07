"""
Configuration for Bitkub Trading Bot
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class BitkubConfig:
    """Bitkub API configuration."""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api.bitkub.com"


@dataclass
class TradingConfig:
    """Trading parameters configuration."""
    symbol: str = "BTC_THB"
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    default_timeframe: str = "15m"
    trading_interval_seconds: int = 30

    # Buy conditions
    rsi_buy_threshold: float = 32.0
    # Sell conditions
    rsi_sell_threshold: float = 68.0
    min_buy_signal_score: float = 2.5
    min_sell_signal_score: float = 1.8
    min_ai_buy_confidence: float = 0.45
    min_ai_sell_confidence: float = 0.40
    strong_ai_buy_confidence: float = 0.65
    strong_ai_sell_confidence: float = 0.60
    min_volume_ratio: float = 1.05

    # Stop Loss / Take Profit (percentage)
    stop_loss_pct: float = 1.6
    take_profit_pct: float = 4.8
    break_even_trigger_pct: float = 0.9
    break_even_buffer_pct: float = 0.15
    trailing_stop_enabled: bool = True
    trailing_stop_trigger_pct: float = 1.4
    trailing_stop_pct: float = 0.8

    # AI-driven cut loss
    ai_cutloss_enabled: bool = True
    ai_cutloss_min_loss_pct: float = 0.45
    ai_cutloss_hard_limit_pct: float = 1.2
    ai_cutloss_min_lstm_confidence: float = 0.35
    ai_cutloss_min_rl_confidence: float = 0.25

    # AI-driven scale-in / averaging down
    ai_scale_in_enabled: bool = False
    ai_scale_in_loss_pct: float = 1.4
    ai_scale_in_min_lstm_confidence: float = 0.55
    ai_scale_in_min_rl_confidence: float = 0.45

    # AI-driven take profit review
    ai_take_profit_enabled: bool = True
    ai_take_profit_min_profit_pct: float = 1.2
    ai_take_profit_min_lstm_confidence: float = 0.35
    ai_take_profit_min_rl_confidence: float = 0.25


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_trade_size_thb: float = 10000.0
    max_daily_loss_thb: float = 5000.0
    max_position_pct: float = 22.0  # max % of balance per position
    max_open_positions: int = 2
    max_daily_trades: int = 10
    max_consecutive_losses: int = 3
    cash_reserve_pct: float = 15.0


@dataclass
class AIConfig:
    """AI model configuration."""
    lstm_sequence_length: int = 60
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 0.001
    lstm_model_path: str = "ai_model/saved/lstm_model.pth"

    rl_episodes: int = 1000
    rl_learning_rate: float = 0.0003
    rl_gamma: float = 0.99
    rl_model_path: str = "ai_model/saved/rl_model.pth"


@dataclass
class LogConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    trade_log_file: str = "logs/trades.log"
    error_log_file: str = "logs/errors.log"
    ai_log_file: str = "logs/ai_predictions.log"
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """Main application configuration."""
    bitkub: BitkubConfig = field(default_factory=BitkubConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.bitkub.api_key = os.environ.get("BITKUB_API_KEY", "")
        config.bitkub.api_secret = os.environ.get("BITKUB_API_SECRET", "")

        if symbol := os.environ.get("TRADING_SYMBOL"):
            config.trading.symbol = symbol
        if sl := os.environ.get("STOP_LOSS_PCT"):
            config.trading.stop_loss_pct = float(sl)
        if tp := os.environ.get("TAKE_PROFIT_PCT"):
            config.trading.take_profit_pct = float(tp)
        if min_buy_score := os.environ.get("MIN_BUY_SIGNAL_SCORE"):
            config.trading.min_buy_signal_score = float(min_buy_score)
        if min_sell_score := os.environ.get("MIN_SELL_SIGNAL_SCORE"):
            config.trading.min_sell_signal_score = float(min_sell_score)
        if min_ai_buy_conf := os.environ.get("MIN_AI_BUY_CONFIDENCE"):
            config.trading.min_ai_buy_confidence = float(min_ai_buy_conf)
        if min_ai_sell_conf := os.environ.get("MIN_AI_SELL_CONFIDENCE"):
            config.trading.min_ai_sell_confidence = float(min_ai_sell_conf)
        if break_even_trigger := os.environ.get("BREAK_EVEN_TRIGGER_PCT"):
            config.trading.break_even_trigger_pct = float(break_even_trigger)
        if trailing_stop_enabled := os.environ.get("TRAILING_STOP_ENABLED"):
            config.trading.trailing_stop_enabled = trailing_stop_enabled.lower() in {
                "1", "true", "yes", "on",
            }
        if trailing_trigger := os.environ.get("TRAILING_STOP_TRIGGER_PCT"):
            config.trading.trailing_stop_trigger_pct = float(trailing_trigger)
        if trailing_stop_pct := os.environ.get("TRAILING_STOP_PCT"):
            config.trading.trailing_stop_pct = float(trailing_stop_pct)
        if ai_cutloss_enabled := os.environ.get("AI_CUTLOSS_ENABLED"):
            config.trading.ai_cutloss_enabled = ai_cutloss_enabled.lower() in {
                "1", "true", "yes", "on",
            }
        if ai_cutloss_min_loss := os.environ.get("AI_CUTLOSS_MIN_LOSS_PCT"):
            config.trading.ai_cutloss_min_loss_pct = float(ai_cutloss_min_loss)
        if ai_cutloss_hard_limit := os.environ.get("AI_CUTLOSS_HARD_LIMIT_PCT"):
            config.trading.ai_cutloss_hard_limit_pct = float(ai_cutloss_hard_limit)
        if ai_cutloss_lstm_conf := os.environ.get("AI_CUTLOSS_MIN_LSTM_CONFIDENCE"):
            config.trading.ai_cutloss_min_lstm_confidence = float(ai_cutloss_lstm_conf)
        if ai_cutloss_rl_conf := os.environ.get("AI_CUTLOSS_MIN_RL_CONFIDENCE"):
            config.trading.ai_cutloss_min_rl_confidence = float(ai_cutloss_rl_conf)
        if ai_scale_in_enabled := os.environ.get("AI_SCALE_IN_ENABLED"):
            config.trading.ai_scale_in_enabled = ai_scale_in_enabled.lower() in {
                "1", "true", "yes", "on",
            }
        if ai_scale_in_loss := os.environ.get("AI_SCALE_IN_LOSS_PCT"):
            config.trading.ai_scale_in_loss_pct = float(ai_scale_in_loss)
        if ai_scale_in_lstm_conf := os.environ.get("AI_SCALE_IN_MIN_LSTM_CONFIDENCE"):
            config.trading.ai_scale_in_min_lstm_confidence = float(ai_scale_in_lstm_conf)
        if ai_scale_in_rl_conf := os.environ.get("AI_SCALE_IN_MIN_RL_CONFIDENCE"):
            config.trading.ai_scale_in_min_rl_confidence = float(ai_scale_in_rl_conf)
        if ai_take_profit_enabled := os.environ.get("AI_TAKE_PROFIT_ENABLED"):
            config.trading.ai_take_profit_enabled = ai_take_profit_enabled.lower() in {
                "1", "true", "yes", "on",
            }
        if ai_take_profit_min_profit := os.environ.get("AI_TAKE_PROFIT_MIN_PROFIT_PCT"):
            config.trading.ai_take_profit_min_profit_pct = float(ai_take_profit_min_profit)
        if ai_take_profit_lstm_conf := os.environ.get("AI_TAKE_PROFIT_MIN_LSTM_CONFIDENCE"):
            config.trading.ai_take_profit_min_lstm_confidence = float(ai_take_profit_lstm_conf)
        if ai_take_profit_rl_conf := os.environ.get("AI_TAKE_PROFIT_MIN_RL_CONFIDENCE"):
            config.trading.ai_take_profit_min_rl_confidence = float(ai_take_profit_rl_conf)
        if max_trade := os.environ.get("MAX_TRADE_SIZE_THB"):
            config.risk.max_trade_size_thb = float(max_trade)
        if max_loss := os.environ.get("MAX_DAILY_LOSS_THB"):
            config.risk.max_daily_loss_thb = float(max_loss)
        if max_daily_trades := os.environ.get("MAX_DAILY_TRADES"):
            config.risk.max_daily_trades = int(max_daily_trades)
        if max_consecutive_losses := os.environ.get("MAX_CONSECUTIVE_LOSSES"):
            config.risk.max_consecutive_losses = int(max_consecutive_losses)
        if cash_reserve_pct := os.environ.get("CASH_RESERVE_PCT"):
            config.risk.cash_reserve_pct = float(cash_reserve_pct)

        return config
