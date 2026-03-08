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
    paper_trade_enabled: bool = False
    paper_trade_start_balance_thb: float = 0.0

    # Buy conditions
    rsi_buy_threshold: float = 32.0
    # Sell conditions
    rsi_sell_threshold: float = 68.0
    min_buy_signal_score: float = 2.8
    min_sell_signal_score: float = 1.8
    min_ai_buy_confidence: float = 0.52
    min_ai_sell_confidence: float = 0.40
    strong_ai_buy_confidence: float = 0.72
    strong_ai_sell_confidence: float = 0.60
    min_volume_ratio: float = 1.10
    buy_zone_enabled: bool = True
    buy_zone_rsi_max: float = 42.0
    buy_zone_ema_gap_pct: float = 1.2
    buy_zone_bb_buffer_pct: float = 0.8
    buy_zone_min_ai_confidence: float = 0.30
    support_resistance_window: int = 20
    support_buffer_pct: float = 0.55
    resistance_breakout_pct: float = 0.25
    reversal_buy_min_ai_confidence: float = 0.80
    reversal_buy_max_trend_gap_pct: float = 0.30
    dip_buy_rsi_max: float = 40.0
    dip_buy_max_ai_down_confidence: float = 0.60
    trend_buy_min_ai_confidence: float = 0.52
    downtrend_recovery_min_ai_confidence: float = 0.74
    downtrend_recovery_min_volume_ratio: float = 1.15
    downtrend_rsi_ceiling: float = 36.0
    quick_profit_sell_pct: float = 1.2
    extended_profit_sell_pct: float = 2.8
    profit_lock_rsi_threshold: float = 66.0
    buy_fee_rate: float = 0.0027
    sell_fee_rate: float = 0.0027

    # Stop Loss / Take Profit (percentage)
    stop_loss_pct: float = 1.40
    take_profit_pct: float = 4.8
    break_even_trigger_pct: float = 0.9
    break_even_buffer_pct: float = 0.15
    trailing_stop_enabled: bool = True
    trailing_stop_trigger_pct: float = 1.4
    trailing_stop_pct: float = 0.8

    # AI-driven cut loss
    ai_cutloss_enabled: bool = True
    ai_cutloss_min_loss_pct: float = 0.75
    ai_cutloss_hard_limit_pct: float = 1.65
    ai_cutloss_min_lstm_confidence: float = 0.35
    ai_cutloss_min_rl_confidence: float = 0.25
    adaptive_risk_enabled: bool = True
    adaptive_cutloss_floor_pct: float = 0.45
    adaptive_cutloss_ceiling_pct: float = 1.20
    adaptive_hard_limit_ceiling_pct: float = 2.20
    adaptive_rebuy_floor_pct: float = 0.70
    adaptive_rebuy_ceiling_pct: float = 1.90
    adaptive_reentry_delay_floor_pct: float = 0.90
    adaptive_reentry_delay_ceiling_pct: float = 2.20
    adaptive_rebuy_min_allocation_pct: float = 35.0
    adaptive_rebuy_max_allocation_pct: float = 80.0
    reentry_cooldown_cycles: int = 2
    reentry_confirm_cycles: int = 3
    reentry_trigger_buffer_pct: float = 0.10
    boss_recovery_cooldown_cycles: int = 2
    boss_recovery_confirm_cycles: int = 3

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
    max_trade_size_thb: float = 7000.0
    max_daily_loss_thb: float = 3000.0
    max_position_pct: float = 16.0  # max % of balance per position
    max_open_positions: int = 2
    max_daily_trades: int = 8
    max_consecutive_losses: int = 3
    cash_reserve_pct: float = 25.0
    downtrend_position_scale_pct: float = 45.0
    high_volatility_atr_pct: float = 2.2
    high_volatility_position_scale_pct: float = 70.0
    loss_streak_position_scale_step_pct: float = 22.0
    downtrend_pause_loss_streak: int = 2


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
    llm_enabled: bool = False
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: float = 8.0
    llm_request_interval_seconds: int = 20
    llm_max_tokens: int = 220
    llm_override_min_confidence: float = 0.68


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
        if paper_trade_enabled := os.environ.get("PAPER_TRADE_ENABLED"):
            config.trading.paper_trade_enabled = paper_trade_enabled.lower() in {
                "1", "true", "yes", "on",
            }
        if paper_trade_balance := os.environ.get("PAPER_TRADE_START_BALANCE_THB"):
            config.trading.paper_trade_start_balance_thb = float(paper_trade_balance)
        if min_buy_score := os.environ.get("MIN_BUY_SIGNAL_SCORE"):
            config.trading.min_buy_signal_score = float(min_buy_score)
        if min_sell_score := os.environ.get("MIN_SELL_SIGNAL_SCORE"):
            config.trading.min_sell_signal_score = float(min_sell_score)
        if min_ai_buy_conf := os.environ.get("MIN_AI_BUY_CONFIDENCE"):
            config.trading.min_ai_buy_confidence = float(min_ai_buy_conf)
        if min_ai_sell_conf := os.environ.get("MIN_AI_SELL_CONFIDENCE"):
            config.trading.min_ai_sell_confidence = float(min_ai_sell_conf)
        if buy_zone_enabled := os.environ.get("BUY_ZONE_ENABLED"):
            config.trading.buy_zone_enabled = buy_zone_enabled.lower() in {
                "1", "true", "yes", "on",
            }
        if buy_zone_rsi_max := os.environ.get("BUY_ZONE_RSI_MAX"):
            config.trading.buy_zone_rsi_max = float(buy_zone_rsi_max)
        if buy_zone_ema_gap := os.environ.get("BUY_ZONE_EMA_GAP_PCT"):
            config.trading.buy_zone_ema_gap_pct = float(buy_zone_ema_gap)
        if buy_zone_bb_buffer := os.environ.get("BUY_ZONE_BB_BUFFER_PCT"):
            config.trading.buy_zone_bb_buffer_pct = float(buy_zone_bb_buffer)
        if buy_zone_ai_conf := os.environ.get("BUY_ZONE_MIN_AI_CONFIDENCE"):
            config.trading.buy_zone_min_ai_confidence = float(buy_zone_ai_conf)
        if sr_window := os.environ.get("SUPPORT_RESISTANCE_WINDOW"):
            config.trading.support_resistance_window = int(sr_window)
        if support_buffer := os.environ.get("SUPPORT_BUFFER_PCT"):
            config.trading.support_buffer_pct = float(support_buffer)
        if resistance_breakout := os.environ.get("RESISTANCE_BREAKOUT_PCT"):
            config.trading.resistance_breakout_pct = float(resistance_breakout)
        if reversal_buy_conf := os.environ.get("REVERSAL_BUY_MIN_AI_CONFIDENCE"):
            config.trading.reversal_buy_min_ai_confidence = float(reversal_buy_conf)
        if reversal_trend_gap := os.environ.get("REVERSAL_BUY_MAX_TREND_GAP_PCT"):
            config.trading.reversal_buy_max_trend_gap_pct = float(reversal_trend_gap)
        if dip_buy_rsi_max := os.environ.get("DIP_BUY_RSI_MAX"):
            config.trading.dip_buy_rsi_max = float(dip_buy_rsi_max)
        if dip_buy_ai_down_conf := os.environ.get("DIP_BUY_MAX_AI_DOWN_CONFIDENCE"):
            config.trading.dip_buy_max_ai_down_confidence = float(dip_buy_ai_down_conf)
        if trend_buy_ai_conf := os.environ.get("TREND_BUY_MIN_AI_CONFIDENCE"):
            config.trading.trend_buy_min_ai_confidence = float(trend_buy_ai_conf)
        if quick_profit_sell := os.environ.get("QUICK_PROFIT_SELL_PCT"):
            config.trading.quick_profit_sell_pct = float(quick_profit_sell)
        if extended_profit_sell := os.environ.get("EXTENDED_PROFIT_SELL_PCT"):
            config.trading.extended_profit_sell_pct = float(extended_profit_sell)
        if profit_lock_rsi := os.environ.get("PROFIT_LOCK_RSI_THRESHOLD"):
            config.trading.profit_lock_rsi_threshold = float(profit_lock_rsi)
        if buy_fee_rate := os.environ.get("BUY_FEE_RATE"):
            config.trading.buy_fee_rate = float(buy_fee_rate)
        if sell_fee_rate := os.environ.get("SELL_FEE_RATE"):
            config.trading.sell_fee_rate = float(sell_fee_rate)
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
        if downtrend_recovery_ai_conf := os.environ.get("DOWNTREND_RECOVERY_MIN_AI_CONFIDENCE"):
            config.trading.downtrend_recovery_min_ai_confidence = float(downtrend_recovery_ai_conf)
        if downtrend_recovery_volume := os.environ.get("DOWNTREND_RECOVERY_MIN_VOLUME_RATIO"):
            config.trading.downtrend_recovery_min_volume_ratio = float(downtrend_recovery_volume)
        if downtrend_rsi_ceiling := os.environ.get("DOWNTREND_RSI_CEILING"):
            config.trading.downtrend_rsi_ceiling = float(downtrend_rsi_ceiling)
        if llm_enabled := os.environ.get("LLM_ENABLED"):
            config.ai.llm_enabled = llm_enabled.lower() in {
                "1", "true", "yes", "on",
            }
        if openai_api_key := os.environ.get("OPENAI_API_KEY"):
            config.ai.openai_api_key = openai_api_key
        if llm_model := os.environ.get("LLM_MODEL"):
            config.ai.llm_model = llm_model
        if llm_timeout := os.environ.get("LLM_TIMEOUT_SECONDS"):
            config.ai.llm_timeout_seconds = float(llm_timeout)
        if llm_interval := os.environ.get("LLM_REQUEST_INTERVAL_SECONDS"):
            config.ai.llm_request_interval_seconds = int(llm_interval)
        if llm_max_tokens := os.environ.get("LLM_MAX_TOKENS"):
            config.ai.llm_max_tokens = int(llm_max_tokens)
        if llm_min_conf := os.environ.get("LLM_OVERRIDE_MIN_CONFIDENCE"):
            config.ai.llm_override_min_confidence = float(llm_min_conf)
        if max_trade := os.environ.get("MAX_TRADE_SIZE_THB"):
            config.risk.max_trade_size_thb = float(max_trade)
        if max_loss := os.environ.get("MAX_DAILY_LOSS_THB"):
            config.risk.max_daily_loss_thb = float(max_loss)
        if max_position_pct := os.environ.get("MAX_POSITION_PCT"):
            config.risk.max_position_pct = float(max_position_pct)
        if max_open_positions := os.environ.get("MAX_OPEN_POSITIONS"):
            config.risk.max_open_positions = int(max_open_positions)
        if max_daily_trades := os.environ.get("MAX_DAILY_TRADES"):
            config.risk.max_daily_trades = int(max_daily_trades)
        if max_consecutive_losses := os.environ.get("MAX_CONSECUTIVE_LOSSES"):
            config.risk.max_consecutive_losses = int(max_consecutive_losses)
        if cash_reserve_pct := os.environ.get("CASH_RESERVE_PCT"):
            config.risk.cash_reserve_pct = float(cash_reserve_pct)
        if downtrend_position_scale := os.environ.get("DOWNTREND_POSITION_SCALE_PCT"):
            config.risk.downtrend_position_scale_pct = float(downtrend_position_scale)
        if high_volatility_atr := os.environ.get("HIGH_VOLATILITY_ATR_PCT"):
            config.risk.high_volatility_atr_pct = float(high_volatility_atr)
        if high_volatility_scale := os.environ.get("HIGH_VOLATILITY_POSITION_SCALE_PCT"):
            config.risk.high_volatility_position_scale_pct = float(high_volatility_scale)
        if loss_streak_scale := os.environ.get("LOSS_STREAK_POSITION_SCALE_STEP_PCT"):
            config.risk.loss_streak_position_scale_step_pct = float(loss_streak_scale)
        if downtrend_pause_streak := os.environ.get("DOWNTREND_PAUSE_LOSS_STREAK"):
            config.risk.downtrend_pause_loss_streak = int(downtrend_pause_streak)

        return config
