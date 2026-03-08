"""
Dynamic Risk Management - จัดการความเสี่ยง max trade size, max daily loss, position size
"""
from datetime import datetime, date
from typing import Dict, List, Optional

from config import RiskConfig
from strategy.trading_strategy import Position
from utils.logger import TradeLogger


class RiskManager:
    """Manages trading risk and position sizing."""

    def __init__(self, config: RiskConfig, logger: Optional[TradeLogger] = None):
        self.config = config
        self.logger = logger or TradeLogger()
        self.daily_loss: float = 0.0
        self.daily_trades: int = 0
        self.consecutive_losses: int = 0
        self.last_reset_date: date = date.today()

    def _reset_daily_if_needed(self):
        """Reset daily counters if it's a new day."""
        today = date.today()
        if today != self.last_reset_date:
            self.daily_loss = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.last_reset_date = today
            self.logger.log_info("Daily risk counters reset")

    def get_market_risk_profile(self, signals: Optional[Dict] = None,
                                ai_prediction: Optional[Dict] = None) -> Dict:
        """Summarize current market risk regime for sizing and UI display."""
        signals = signals or {}
        ai_prediction = ai_prediction or {}

        atr_pct = float(signals.get("atr_pct", 0.0) or 0.0)
        volume_ratio = float(signals.get("volume_ratio", 1.0) or 1.0)
        trend_down = bool(signals.get("trend_down", False))
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = float(ai_prediction.get("confidence", 0.0) or 0.0)

        scale_factor = 1.0
        reasons: List[str] = []
        regime = "normal"

        if trend_down:
            scale_factor *= max(self.config.downtrend_position_scale_pct / 100, 0.10)
            reasons.append("Downtrend active")
            regime = "downtrend"

        if atr_pct >= self.config.high_volatility_atr_pct > 0:
            scale_factor *= max(self.config.high_volatility_position_scale_pct / 100, 0.10)
            reasons.append(f"ATR high {atr_pct:.2f}%")
            regime = "volatile_downtrend" if trend_down else "high_volatility"

        if ai_direction == "down" and ai_confidence >= 0.65:
            scale_factor *= 0.85
            reasons.append(f"AI downside {ai_confidence:.0%}")

        if volume_ratio < 1.0:
            scale_factor *= 0.92
            reasons.append(f"Volume light {volume_ratio:.2f}x")

        scale_factor = max(scale_factor, 0.18)

        label_map = {
            "normal": "Normal",
            "downtrend": "Downtrend Guard",
            "high_volatility": "High Volatility",
            "volatile_downtrend": "Downtrend + Volatility",
        }
        return {
            "regime": regime,
            "regime_label": label_map.get(regime, "Normal"),
            "position_scale_factor": scale_factor,
            "position_scale_pct": scale_factor * 100,
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio,
            "trend_down": trend_down,
            "ai_direction": ai_direction,
            "ai_confidence": ai_confidence,
            "reasons": reasons,
        }

    def can_trade(self, balance: float, positions: List[Position],
                  signals: Optional[Dict] = None,
                  ai_prediction: Optional[Dict] = None) -> Dict:
        """Check if a new trade is allowed."""
        self._reset_daily_if_needed()

        checks = {
            "allowed": True,
            "reasons": [],
        }

        # Check daily loss limit
        if self.daily_loss >= self.config.max_daily_loss_thb:
            checks["allowed"] = False
            checks["reasons"].append(
                f"Daily loss limit reached: {self.daily_loss:.2f} >= "
                f"{self.config.max_daily_loss_thb:.2f} THB"
            )

        if self.daily_trades >= self.config.max_daily_trades:
            checks["allowed"] = False
            checks["reasons"].append(
                f"Daily trade limit reached: {self.daily_trades} >= "
                f"{self.config.max_daily_trades}"
            )

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            checks["allowed"] = False
            checks["reasons"].append(
                f"Consecutive loss limit reached: {self.consecutive_losses} >= "
                f"{self.config.max_consecutive_losses}"
            )

        # Check max open positions
        if len(positions) >= self.config.max_open_positions:
            checks["allowed"] = False
            checks["reasons"].append(
                f"Max open positions reached: {len(positions)} >= "
                f"{self.config.max_open_positions}"
            )

        # Check minimum balance
        if balance < 10:  # Minimum 10 THB
            checks["allowed"] = False
            checks["reasons"].append(f"Insufficient balance: {balance:.2f} THB")

        market_profile = self.get_market_risk_profile(signals, ai_prediction)
        if (
            checks["allowed"]
            and not positions
            and market_profile["trend_down"]
            and self.consecutive_losses >= self.config.downtrend_pause_loss_streak
        ):
            checks["allowed"] = False
            checks["reasons"].append(
                "Downtrend protection active after recent losses"
            )

        return checks

    def calculate_position_size(self, balance: float, current_price: float,
                                signal_strength: float = 1.0,
                                signals: Optional[Dict] = None,
                                ai_prediction: Optional[Dict] = None) -> Dict:
        """Calculate optimal position size based on risk parameters."""
        self._reset_daily_if_needed()

        # Max trade size
        reserve_cash = balance * (self.config.cash_reserve_pct / 100)
        tradable_balance = max(balance - reserve_cash, 0)

        max_trade = min(
            self.config.max_trade_size_thb,
            balance * (self.config.max_position_pct / 100),
            tradable_balance,
        )

        # Adjust by signal strength (0.0 - 1.0)
        loss_step = max(float(self.config.loss_streak_position_scale_step_pct or 0.0), 0.0) / 100
        loss_streak_factor = max(0.35, 1.0 - (loss_step * self.consecutive_losses))
        market_profile = self.get_market_risk_profile(signals, ai_prediction)
        adjusted_size = (
            max_trade
            * max(signal_strength, 0.25)
            * loss_streak_factor
            * market_profile["position_scale_factor"]
        )

        # Ensure we don't exceed balance
        position_size_thb = min(adjusted_size, tradable_balance * 0.95)

        # Calculate crypto amount
        crypto_amount = position_size_thb / current_price if current_price > 0 else 0

        return {
            "position_size_thb": position_size_thb,
            "crypto_amount": crypto_amount,
            "max_trade_size": max_trade,
            "signal_strength": signal_strength,
            "pct_of_balance": (position_size_thb / balance * 100) if balance > 0 else 0,
            "loss_streak_factor": loss_streak_factor,
            "market_scale_factor": market_profile["position_scale_factor"],
            "market_regime": market_profile["regime_label"],
            "risk_notes": market_profile["reasons"],
        }

    def record_trade_result(self, profit_thb: float):
        """Record a trade result for daily tracking."""
        self._reset_daily_if_needed()
        self.daily_trades += 1

        if profit_thb < 0:
            self.daily_loss += abs(profit_thb)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.logger.log_info(
            f"Trade recorded | P/L: {profit_thb:.2f} THB | "
            f"Daily loss: {self.daily_loss:.2f} THB | "
            f"Daily trades: {self.daily_trades} | "
            f"Loss streak: {self.consecutive_losses}"
        )

    def get_risk_status(self, balance: float, positions: List[Position]) -> Dict:
        """Get current risk status overview."""
        self._reset_daily_if_needed()

        total_position_value = sum(p.value for p in positions)
        exposure_pct = (total_position_value / balance * 100) if balance > 0 else 0

        remaining_daily_loss = max(
            self.config.max_daily_loss_thb - self.daily_loss, 0
        )

        return {
            "balance": balance,
            "total_position_value": total_position_value,
            "exposure_pct": exposure_pct,
            "open_positions": len(positions),
            "max_positions": self.config.max_open_positions,
            "daily_loss": self.daily_loss,
            "max_daily_loss": self.config.max_daily_loss_thb,
            "remaining_daily_loss": remaining_daily_loss,
            "daily_trades": self.daily_trades,
            "loss_streak": self.consecutive_losses,
            "can_trade": (
                remaining_daily_loss > 0
                and len(positions) < self.config.max_open_positions
                and self.daily_trades < self.config.max_daily_trades
                and self.consecutive_losses < self.config.max_consecutive_losses
            ),
        }
