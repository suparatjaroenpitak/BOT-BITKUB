"""
Trading Strategy - เงื่อนไข Buy/Sell/Stop Loss/Take Profit
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config import TradingConfig
from utils.logger import TradeLogger


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: str  # "long"
    entry_price: float
    amount: float
    entry_time: datetime = field(default_factory=datetime.now)
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    highest_price: float = 0.0

    @property
    def value(self) -> float:
        return self.entry_price * self.amount


class TradingStrategy:
    """Combined strategy using technical indicators + AI predictions."""

    def __init__(self, config: TradingConfig, logger: Optional[TradeLogger] = None):
        self.config = config
        self.logger = logger or TradeLogger()
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """Clamp a float between two bounds."""
        return max(minimum, min(value, maximum))

    def _evaluate_buy_zone(self, signals: Dict, ai_prediction: Dict) -> Dict:
        """Check whether price is inside a favorable auto-buy accumulation zone."""
        price = float(signals.get("price", 0) or 0)
        rsi = float(signals.get("rsi", 50) or 50)
        ema_21 = float(signals.get("ema_21", 0) or 0)
        bb_lower = float(signals.get("bb_lower", 0) or 0)
        bb_middle = float(signals.get("bb_middle", 0) or 0)
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = float(ai_prediction.get("confidence", 0) or 0)

        if (
            not self.config.buy_zone_enabled
            or price <= 0
            or ema_21 <= 0
            or bb_lower <= 0
        ):
            return {
                "in_zone": False,
                "zone_low": 0.0,
                "zone_high": 0.0,
                "reasons": [],
            }

        zone_high = min(
            ema_21 * (1 - self.config.buy_zone_ema_gap_pct / 100),
            bb_middle if bb_middle > 0 else ema_21,
        )
        zone_low = bb_lower * (1 - self.config.buy_zone_bb_buffer_pct / 100)

        if zone_high <= 0 or zone_low <= 0 or zone_low > zone_high:
            return {
                "in_zone": False,
                "zone_low": zone_low,
                "zone_high": zone_high,
                "reasons": [],
            }

        in_zone = zone_low <= price <= zone_high
        reasons = []
        if in_zone:
            reasons.append(
                f"Price entered buy zone {zone_low:,.2f}-{zone_high:,.2f}"
            )

        zone_allowed = (
            in_zone
            and rsi <= self.config.buy_zone_rsi_max
            and not (ai_direction == "down" and ai_confidence >= self.config.strong_ai_sell_confidence)
            and (ai_direction != "down" or ai_confidence <= self.config.buy_zone_min_ai_confidence)
        )

        if in_zone and rsi <= self.config.buy_zone_rsi_max:
            reasons.append(f"RSI is within buy-zone limit ({rsi:.1f} <= {self.config.buy_zone_rsi_max:.1f})")

        return {
            "in_zone": zone_allowed,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "reasons": reasons,
        }

    def should_buy(self, signals: Dict, ai_prediction: Dict) -> Dict:
        """Determine if we should BUY when we do not currently hold the coin."""
        reasons = []
        score = 0

        rsi = signals.get("rsi", 50)
        price = signals.get("price", 0)
        ema_9 = signals.get("ema_9", 0)
        ema_21 = signals.get("ema_21", 0)
        ema_50 = signals.get("ema_50", 0)
        support_level = float(signals.get("support_level", 0) or 0)
        resistance_level = float(signals.get("resistance_level", 0) or 0)
        support_distance_pct = float(signals.get("support_distance_pct", float("inf")) or float("inf"))
        resistance_distance_pct = float(signals.get("resistance_distance_pct", float("inf")) or float("inf"))
        volume_ratio = float(signals.get("volume_ratio", 1) or 1)
        ai_change_pct = float(ai_prediction.get("price_change_pct", 0) or 0)
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = ai_prediction.get("confidence", 0)
        trend_up = ema_9 > ema_21 > ema_50 > 0
        trend_soft_up = ema_9 > ema_21 > 0
        falling_trend = ema_9 < ema_21 < ema_50 if ema_50 > 0 else ema_9 < ema_21
        trend_down = bool(signals.get("trend_down", False)) or falling_trend
        buy_zone = self._evaluate_buy_zone(signals, ai_prediction)
        if support_distance_pct != support_distance_pct:
            support_distance_pct = float("inf")
        if resistance_distance_pct != resistance_distance_pct:
            resistance_distance_pct = float("inf")

        trend_gap_pct = abs((price - ema_21) / ema_21 * 100) if ema_21 > 0 else 0.0
        near_support = (
            support_level > 0
            and price >= support_level
            and support_distance_pct <= self.config.support_buffer_pct
        )
        breakout_resistance = (
            resistance_level > 0
            and price >= resistance_level * (1 + self.config.resistance_breakout_pct / 100)
        )
        strong_reversal = (
            rsi <= max(self.config.rsi_buy_threshold - 4, 20)
            and signals.get("macd_bullish", False)
            and ai_direction == "up"
            and ai_confidence >= self.config.strong_ai_buy_confidence
        )
        uptrend_pullback = (
            (trend_up or trend_soft_up)
            and near_support
            and ai_direction == "up"
            and ai_confidence >= self.config.min_ai_buy_confidence
        )
        uptrend_breakout = (
            (trend_up or trend_soft_up)
            and breakout_resistance
            and ai_direction == "up"
            and ai_confidence >= self.config.trend_buy_min_ai_confidence
        )
        counter_trend_reversal = (
            trend_down
            and near_support
            and strong_reversal
            and ai_confidence >= self.config.reversal_buy_min_ai_confidence
            and trend_gap_pct <= self.config.reversal_buy_max_trend_gap_pct
        )
        dip_buy = (
            (
                price < ema_21
                or signals.get("price_below_bb_lower", False)
                or rsi <= self.config.dip_buy_rsi_max
                or ai_change_pct < 0
            )
            and (
                near_support
                or buy_zone["in_zone"]
                or signals.get("price_below_bb_lower", False)
            )
            and not (
                ai_direction == "down"
                and ai_confidence >= self.config.dip_buy_max_ai_down_confidence
            )
        )
        trend_buy = (
            (trend_up or trend_soft_up or ai_change_pct > 0)
            and (
                ai_direction == "up"
                or ai_confidence >= self.config.trend_buy_min_ai_confidence
                or breakout_resistance
            )
            and (near_support or breakout_resistance or price > ema_9 > 0)
        )

        # RSI condition
        if rsi < self.config.rsi_buy_threshold:
            score += 1
            reasons.append(f"RSI oversold ({rsi:.1f} < {self.config.rsi_buy_threshold})")

        # Price below EMA
        if price < ema_21 and ema_21 > 0:
            score += 0.5
            reasons.append(f"Price ({price:.2f}) below EMA21 ({ema_21:.2f})")

        if near_support:
            score += 0.7
            reasons.append(
                f"Price is holding near support {support_level:,.2f} ({support_distance_pct:.2f}% above support)"
            )

        if breakout_resistance:
            score += 0.85
            reasons.append(
                f"Price broke resistance {resistance_level:,.2f} and is entering momentum continuation"
            )

        if trend_up:
            score += 0.75
            reasons.append("Trend filter passed: EMA9 > EMA21 > EMA50")
        elif trend_soft_up:
            score += 0.35
            reasons.append("Short-term trend still positive")

        # AI prediction UP
        if ai_direction == "up" and ai_confidence >= self.config.min_ai_buy_confidence:
            score += 1
            reasons.append(f"AI predicts UP (confidence: {ai_confidence:.2f})")

        if ai_change_pct > 0.35:
            score += 0.35
            reasons.append(f"AI expects upside {ai_change_pct:.2f}%")

        # Additional confirmations
        if signals.get("macd_bullish", False):
            score += 0.5
            reasons.append("MACD bullish crossover")

        if signals.get("price_below_bb_lower", False):
            score += 0.5
            reasons.append("Price below Bollinger lower band")

        if buy_zone["in_zone"] and not trend_down:
            score += 0.9
            reasons.extend(buy_zone["reasons"])

        range_auto_buy = buy_zone["in_zone"] and not trend_down and (near_support or trend_up or trend_soft_up)
        if range_auto_buy:
            reasons.append("Auto-buy triggered because price is inside the configured buy zone")

        if dip_buy:
            score += 0.8
            reasons.append("Dip-buy setup detected: price weakened and AI allows accumulation")

        if trend_buy:
            score += 0.8
            reasons.append("Trend-buy setup detected: price is climbing and AI allows momentum entry")

        if uptrend_pullback:
            score += 0.55
            reasons.append("Uptrend pullback entry: AI waits for support, then buys on strength")

        if uptrend_breakout:
            score += 0.7
            reasons.append("Uptrend breakout entry confirmed above resistance")

        if counter_trend_reversal:
            score += 0.95
            reasons.append("Counter-trend reversal allowed: strong AI rebound from support")

        if volume_ratio >= self.config.min_volume_ratio:
            score += 0.35
            reasons.append(f"Volume support confirmed ({volume_ratio:.2f}x)")
        elif ai_confidence < self.config.strong_ai_buy_confidence:
            score -= 0.4
            reasons.append(f"Volume too light ({volume_ratio:.2f}x), skipping weak setup")

        if trend_down and not (dip_buy or counter_trend_reversal):
            score -= 0.9
            reasons.append("Downtrend remains active, so AI only allows dip accumulation near support")

        valid_entry_setup = (
            dip_buy
            or trend_buy
            or uptrend_pullback
            or uptrend_breakout
            or counter_trend_reversal
            or range_auto_buy
        )

        if not valid_entry_setup:
            reasons.append("AI is waiting for either a dip-buy near support or a trend-buy on strength")

        effective_score = max(score, self.config.min_buy_signal_score) if valid_entry_setup else score

        should = valid_entry_setup and effective_score >= self.config.min_buy_signal_score

        return {
            "should_buy": should,
            "score": effective_score,
            "reasons": reasons,
            "signal_strength": min(max(effective_score / 4.0, 0.0), 1.0),
            "buy_zone": buy_zone,
            "range_auto_buy": range_auto_buy,
            "dip_buy": dip_buy,
            "trend_buy": trend_buy,
            "near_support": near_support,
            "breakout_resistance": breakout_resistance,
            "counter_trend_reversal": counter_trend_reversal,
        }

    def should_sell(self, signals: Dict, ai_prediction: Dict,
                    position: Optional[Position] = None) -> Dict:
        """Determine if we should SELL when we already hold the coin."""
        reasons = []
        score = 0

        rsi = signals.get("rsi", 50)
        price = signals.get("price", 0)
        bb_upper = signals.get("bb_upper", float("inf"))
        resistance_level = float(signals.get("resistance_level", 0) or 0)
        ema_9 = signals.get("ema_9", 0)
        ema_21 = signals.get("ema_21", 0)
        volume_ratio = float(signals.get("volume_ratio", 1) or 1)
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = ai_prediction.get("confidence", 0)
        ai_change_pct = float(ai_prediction.get("price_change_pct", 0) or 0)
        trend_weak = price < ema_9 < ema_21 if ema_9 and ema_21 else price < ema_21
        trend_down = bool(signals.get("trend_down", False)) or trend_weak
        near_resistance = resistance_level > 0 and price >= resistance_level * 0.997
        pnl = self.get_position_pnl(position, price) if position else {"profit_pct": 0.0, "profit_thb": 0.0}
        profit_pct = pnl["profit_pct"]
        loss_pct = abs(min(profit_pct, 0.0))
        panic_sell = trend_down and (ai_direction == "down" or ai_change_pct < -0.2 or loss_pct >= self.config.ai_cutloss_min_loss_pct)
        profit_lock_sell = (
            profit_pct >= self.config.quick_profit_sell_pct
            and (
                rsi >= self.config.profit_lock_rsi_threshold
                or price > bb_upper
                or near_resistance
                or ai_direction == "down"
                or ai_change_pct <= 0
            )
        )
        extended_rally_sell = profit_pct >= self.config.extended_profit_sell_pct

        # RSI condition
        if rsi > self.config.rsi_sell_threshold:
            score += 1
            reasons.append(f"RSI overbought ({rsi:.1f} > {self.config.rsi_sell_threshold})")

        # Price above Bollinger upper
        if price > bb_upper and bb_upper > 0:
            score += 1
            reasons.append(f"Price ({price:.2f}) above BB upper ({bb_upper:.2f})")

        # AI prediction DOWN
        if ai_direction == "down" and ai_confidence >= self.config.min_ai_sell_confidence:
            score += 1
            reasons.append(f"AI predicts DOWN (confidence: {ai_confidence:.2f})")

        if trend_weak:
            score += 0.6
            reasons.append("Trend weakened: price < EMA9 < EMA21")

        if ai_change_pct < -0.3:
            score += 0.35
            reasons.append(f"AI expects downside {ai_change_pct:.2f}%")

        if panic_sell:
            score += 1.15
            reasons.append("Holding position while price weakens, so AI wants to reduce downside")

        if profit_lock_sell:
            score += 1.05
            reasons.append(f"Profit lock setup: gain {profit_pct:.2f}% is large enough to secure")

        if extended_rally_sell:
            score += 1.25
            reasons.append(f"Price already rallied {profit_pct:.2f}%, so AI locks the move")

        if near_resistance:
            score += 0.45
            reasons.append(f"Price is testing resistance {resistance_level:,.2f}")

        # Additional confirmations
        if signals.get("macd_bearish", False):
            score += 0.5
            reasons.append("MACD bearish crossover")

        if volume_ratio >= self.config.min_volume_ratio:
            score += 0.2

        if ai_direction == "up" and ai_confidence >= self.config.strong_ai_buy_confidence and not trend_weak:
            score -= 0.75
            reasons.append("Strong bullish AI bias delays sell")

        should = score >= self.config.min_sell_signal_score and (
            panic_sell
            or profit_lock_sell
            or extended_rally_sell
            or ai_direction == "down"
            or trend_weak
            or rsi >= self.config.rsi_sell_threshold + 4
        )

        return {
            "should_sell": should,
            "score": score,
            "reasons": reasons,
            "signal_strength": min(score / 3.0, 1.0),
            "panic_sell": panic_sell,
            "profit_lock_sell": profit_lock_sell,
            "extended_rally_sell": extended_rally_sell,
        }

    @staticmethod
    def get_position_pnl(position: Position, current_price: float) -> Dict[str, float]:
        """Calculate current P/L for a position."""
        profit_pct = (current_price - position.entry_price) / position.entry_price * 100
        profit_thb = (current_price - position.entry_price) * position.amount
        return {
            "profit_pct": profit_pct,
            "profit_thb": profit_thb,
        }

    def get_adaptive_risk_profile(self, signals: Dict, ai_prediction: Dict,
                                  position: Optional[Position] = None,
                                  rl_decision: Optional[Dict] = None,
                                  base_cutloss_pct: Optional[float] = None,
                                  base_recovery_pct: Optional[float] = None) -> Dict:
        """Build adaptive cutloss and rebuy thresholds from live market conditions."""
        base_cutloss = max(base_cutloss_pct or self.config.ai_cutloss_min_loss_pct, 0.05)
        base_recovery = max(base_recovery_pct or self.config.adaptive_rebuy_floor_pct, 0.05)

        if not self.config.adaptive_risk_enabled:
            hard_limit = max(base_cutloss * 1.6, self.config.ai_cutloss_hard_limit_pct)
            delay_pct = max(base_recovery * 1.2, self.config.adaptive_reentry_delay_floor_pct)
            allocation_pct = self._clamp(
                self.config.adaptive_rebuy_max_allocation_pct,
                self.config.adaptive_rebuy_min_allocation_pct,
                self.config.adaptive_rebuy_max_allocation_pct,
            )
            return {
                "cutloss_pct": base_cutloss,
                "hard_limit_pct": hard_limit,
                "recovery_pct": base_recovery,
                "delay_pct": delay_pct,
                "rebuy_allocation_pct": allocation_pct,
                "volatility_pct": 0.0,
                "bearish_pressure": 0.0,
                "rebound_score": 0.0,
                "reasons": ["Adaptive risk disabled; using base thresholds"],
            }

        rsi = float(signals.get("rsi", 50) or 50)
        atr_pct = float(signals.get("atr_pct", 0) or 0)
        bb_width = float(signals.get("bb_width", 0) or 0)
        volume_ratio = float(signals.get("volume_ratio", 1) or 1)
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = float(ai_prediction.get("confidence", 0) or 0)
        ai_change_pct = float(ai_prediction.get("price_change_pct", 0) or 0)

        trend_down = bool(signals.get("trend_down", False))
        trend_up = bool(signals.get("trend_up", False))
        macd_bearish = bool(signals.get("macd_bearish", False))
        macd_bullish = bool(signals.get("macd_bullish", False))
        price_below_bb = bool(signals.get("price_below_bb_lower", False))

        volatility_pct = max(atr_pct, bb_width * 0.25, abs(ai_change_pct) * 0.6, 0.08)
        bearish_pressure = 0.0
        rebound_score = 0.0
        reasons: List[str] = []

        if ai_direction == "down":
            bearish_pressure += 0.45 + (ai_confidence * 0.55)
            reasons.append(f"AI downside pressure {ai_confidence:.2f}")
        elif ai_direction == "up":
            rebound_score += 0.30 + (ai_confidence * 0.45)
            reasons.append(f"AI rebound confidence {ai_confidence:.2f}")

        if ai_change_pct < 0:
            bearish_pressure += min(abs(ai_change_pct) / 1.5, 0.40)
        elif ai_change_pct > 0:
            rebound_score += min(ai_change_pct / 1.5, 0.35)

        if trend_down:
            bearish_pressure += 0.25
            reasons.append("Trend remains down")
        if trend_up:
            rebound_score += 0.20
        if macd_bearish:
            bearish_pressure += 0.20
        if macd_bullish:
            rebound_score += 0.18
        if price_below_bb:
            rebound_score += 0.14
        if rsi <= min(self.config.rsi_buy_threshold + 3, 40):
            rebound_score += 0.16
        elif rsi >= self.config.rsi_sell_threshold:
            bearish_pressure += 0.12
        if volume_ratio >= self.config.min_volume_ratio:
            rebound_score += 0.08
        elif volume_ratio < 0.9:
            bearish_pressure += 0.10

        if position is not None:
            pnl = self.get_position_pnl(position, float(signals.get("price", position.entry_price) or position.entry_price))
            if pnl["profit_pct"] < 0:
                bearish_pressure += min(abs(pnl["profit_pct"]) / 2.5, 0.35)

        if rl_decision:
            rl_action = rl_decision.get("action_name", "HOLD")
            rl_confidence = float(rl_decision.get("confidence", 0) or 0)
            if rl_action == "SELL":
                bearish_pressure += 0.30 + (rl_confidence * 0.35)
                reasons.append(f"RL leans SELL {rl_confidence:.2f}")
            elif rl_action == "BUY":
                rebound_score += 0.22 + (rl_confidence * 0.28)
                reasons.append(f"RL leans BUY {rl_confidence:.2f}")

        cutloss_floor = min(base_cutloss, self.config.adaptive_cutloss_floor_pct)
        cutloss_ceiling = max(base_cutloss, self.config.adaptive_cutloss_ceiling_pct)
        recovery_floor = min(base_recovery, self.config.adaptive_rebuy_floor_pct)
        recovery_ceiling = max(base_recovery, self.config.adaptive_rebuy_ceiling_pct)

        cutloss_pct = max(base_cutloss, volatility_pct * 0.90)
        cutloss_pct *= 1 - min(bearish_pressure * 0.22, 0.38)
        if rebound_score > bearish_pressure:
            cutloss_pct *= 1 + min((rebound_score - bearish_pressure) * 0.06, 0.12)
        cutloss_pct = self._clamp(cutloss_pct, cutloss_floor, cutloss_ceiling)

        hard_limit_pct = max(
            cutloss_pct * 1.55,
            cutloss_pct + max(volatility_pct * 0.55, 0.10),
            self.config.ai_cutloss_hard_limit_pct * (0.75 if bearish_pressure > 1.0 else 1.0),
        )
        hard_limit_pct = self._clamp(
            hard_limit_pct,
            cutloss_pct + 0.05,
            max(cutloss_pct + 0.05, self.config.adaptive_hard_limit_ceiling_pct),
        )

        recovery_pct = max(base_recovery, volatility_pct * 0.75)
        recovery_pct *= 1 + min((bearish_pressure * 0.22) + (volatility_pct * 0.08), 0.60)
        recovery_pct *= 1 - min(rebound_score * 0.16, 0.28)
        recovery_pct = self._clamp(recovery_pct, recovery_floor, recovery_ceiling)

        delay_pct = max(recovery_pct * (1.15 + min(volatility_pct * 0.10, 0.30)), recovery_pct + 0.10)
        delay_pct = self._clamp(
            delay_pct,
            max(recovery_pct, self.config.adaptive_reentry_delay_floor_pct),
            max(recovery_pct, self.config.adaptive_reentry_delay_ceiling_pct),
        )

        allocation_pct = 74.0
        allocation_pct -= min(volatility_pct * 10.0, 18.0)
        allocation_pct -= min(bearish_pressure * 16.0, 22.0)
        allocation_pct += min(rebound_score * 8.0, 10.0)
        rebuy_allocation_pct = self._clamp(
            allocation_pct,
            self.config.adaptive_rebuy_min_allocation_pct,
            self.config.adaptive_rebuy_max_allocation_pct,
        )

        reasons.append(
            f"Adaptive profile: vol {volatility_pct:.2f}% | cutloss {cutloss_pct:.2f}% | recovery {recovery_pct:.2f}%"
        )
        return {
            "cutloss_pct": cutloss_pct,
            "hard_limit_pct": hard_limit_pct,
            "recovery_pct": recovery_pct,
            "delay_pct": delay_pct,
            "rebuy_allocation_pct": rebuy_allocation_pct,
            "volatility_pct": volatility_pct,
            "bearish_pressure": bearish_pressure,
            "rebound_score": rebound_score,
            "reasons": reasons,
        }

    def evaluate_ai_cut_loss(self, position: Position, current_price: float,
                             ai_prediction: Dict,
                             rl_decision: Optional[Dict] = None,
                             min_loss_pct: Optional[float] = None,
                             hard_limit_pct: Optional[float] = None) -> Dict:
        """Evaluate whether AI should force an exit back to THB."""
        pnl = self.get_position_pnl(position, current_price)
        loss_pct = abs(min(pnl["profit_pct"], 0.0))

        if not self.config.ai_cutloss_enabled or pnl["profit_pct"] >= 0:
            return {
                "should_sell": False,
                "score": 0.0,
                "reasons": [],
                "reason": "",
                "loss_pct": loss_pct,
                "profit_thb": pnl["profit_thb"],
            }

        min_loss = max(min_loss_pct or self.config.ai_cutloss_min_loss_pct, 0.0)
        hard_limit = max(hard_limit_pct or self.config.ai_cutloss_hard_limit_pct, min_loss)

        if loss_pct < min_loss:
            return {
                "should_sell": False,
                "score": 0.0,
                "reasons": [],
                "reason": "",
                "loss_pct": loss_pct,
                "profit_thb": pnl["profit_thb"],
            }

        reasons = [
            f"Position loss {loss_pct:.2f}% reached AI review threshold {min_loss:.2f}%"
        ]
        score = 1.0

        if loss_pct >= hard_limit:
            reasons.append(
                f"Hard protection at {hard_limit:.2f}% loss triggered immediate sell"
            )
            return {
                "should_sell": True,
                "score": 99.0,
                "reasons": reasons,
                "reason": "AI_CUTLOSS_TO_THB: " + "; ".join(reasons),
                "loss_pct": loss_pct,
                "profit_thb": pnl["profit_thb"],
            }

        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = float(ai_prediction.get("confidence", 0) or 0)
        ai_change_pct = float(ai_prediction.get("price_change_pct", 0) or 0)

        if ai_direction == "down":
            reasons.append(
                f"LSTM predicts DOWN with confidence {ai_confidence:.2f}"
            )
            score += 0.75
            if ai_confidence >= self.config.ai_cutloss_min_lstm_confidence:
                score += 0.75

        if ai_change_pct <= -min_loss:
            reasons.append(f"LSTM expects another {ai_change_pct:.2f}% move")
            score += 0.5

        if rl_decision:
            rl_action = rl_decision.get("action_name", "HOLD")
            rl_confidence = float(rl_decision.get("confidence", 0) or 0)
            if rl_action == "SELL":
                reasons.append(
                    f"RL recommends SELL with confidence {rl_confidence:.2f}"
                )
                score += 0.75
                if rl_confidence >= self.config.ai_cutloss_min_rl_confidence:
                    score += 0.5

        should_sell = score >= 2.5
        return {
            "should_sell": should_sell,
            "score": score,
            "reasons": reasons,
            "reason": "AI_CUTLOSS_TO_THB: " + "; ".join(reasons) if should_sell else "",
            "loss_pct": loss_pct,
            "profit_thb": pnl["profit_thb"],
        }

    def evaluate_ai_scale_in(self, positions: List[Position], current_price: float,
                             signals: Dict, ai_prediction: Dict,
                             rl_decision: Optional[Dict] = None,
                             min_loss_pct: Optional[float] = None) -> Dict:
        """Evaluate whether AI should average down into an existing losing position."""
        if not self.config.ai_scale_in_enabled or not positions or current_price <= 0:
            return {
                "should_buy": False,
                "score": 0.0,
                "reasons": [],
                "reason": "",
                "avg_loss_pct": 0.0,
                "trigger_loss_pct": 0.0,
                "signal_strength": 0.0,
            }

        pnl_values = [
            self.get_position_pnl(position, current_price)["profit_pct"]
            for position in positions
        ]
        avg_pnl_pct = sum(pnl_values) / len(pnl_values)
        avg_loss_pct = abs(min(avg_pnl_pct, 0.0))
        base_trigger = max(min_loss_pct or self.config.ai_scale_in_loss_pct, 0.0)
        trigger_loss_pct = base_trigger * max(len(positions), 1)

        if avg_loss_pct < trigger_loss_pct:
            return {
                "should_buy": False,
                "score": 0.0,
                "reasons": [],
                "reason": "",
                "avg_loss_pct": avg_loss_pct,
                "trigger_loss_pct": trigger_loss_pct,
                "signal_strength": 0.0,
            }

        reasons = [
            f"Average loss {avg_loss_pct:.2f}% reached AI scale-in trigger {trigger_loss_pct:.2f}%"
        ]
        score = 1.0

        rsi = float(signals.get("rsi", 50) or 50)
        price = float(signals.get("price", current_price) or current_price)
        ema_21 = float(signals.get("ema_21", 0) or 0)
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = float(ai_prediction.get("confidence", 0) or 0)
        ai_change_pct = float(ai_prediction.get("price_change_pct", 0) or 0)

        if rsi <= min(self.config.rsi_buy_threshold + 5, 45):
            reasons.append(f"RSI remains weak at {rsi:.1f}, rebound setup possible")
            score += 0.5

        if price < ema_21 and ema_21 > 0:
            reasons.append(f"Price {price:.2f} still below EMA21 {ema_21:.2f}")
            score += 0.25

        if ai_direction == "up":
            reasons.append(f"LSTM predicts rebound UP with confidence {ai_confidence:.2f}")
            score += 0.75
            if ai_confidence >= self.config.ai_scale_in_min_lstm_confidence:
                score += 0.75
        elif ai_direction == "down" and ai_confidence >= 0.4:
            reasons.append(f"LSTM still bearish ({ai_confidence:.2f}); scale-in blocked")
            score -= 0.75

        if ai_change_pct > 0:
            reasons.append(f"LSTM expects {ai_change_pct:.2f}% upside from here")
            score += 0.5

        if rl_decision:
            rl_action = rl_decision.get("action_name", "HOLD")
            rl_confidence = float(rl_decision.get("confidence", 0) or 0)
            if rl_action == "BUY":
                reasons.append(
                    f"RL recommends BUY with confidence {rl_confidence:.2f}"
                )
                score += 0.75
                if rl_confidence >= self.config.ai_scale_in_min_rl_confidence:
                    score += 0.5

        should_buy = score >= 2.5 and ai_direction != "down"
        return {
            "should_buy": should_buy,
            "score": score,
            "reasons": reasons,
            "reason": "AI_SCALE_IN: " + "; ".join(reasons) if should_buy else "",
            "avg_loss_pct": avg_loss_pct,
            "trigger_loss_pct": trigger_loss_pct,
            "signal_strength": min(max(score / 3.5, 0.0), 1.0),
        }

    def evaluate_ai_take_profit(self, position: Position, current_price: float,
                                signals: Dict, ai_prediction: Dict,
                                rl_decision: Optional[Dict] = None,
                                min_profit_pct: Optional[float] = None) -> Dict:
        """Evaluate whether AI should lock in profits before momentum fades."""
        pnl = self.get_position_pnl(position, current_price)
        profit_pct = max(pnl["profit_pct"], 0.0)

        if not self.config.ai_take_profit_enabled or profit_pct <= 0:
            return {
                "should_sell": False,
                "score": 0.0,
                "reasons": [],
                "reason": "",
                "profit_pct": profit_pct,
                "profit_thb": pnl["profit_thb"],
            }

        min_profit = max(min_profit_pct or self.config.ai_take_profit_min_profit_pct, 0.0)
        if profit_pct < min_profit:
            return {
                "should_sell": False,
                "score": 0.0,
                "reasons": [],
                "reason": "",
                "profit_pct": profit_pct,
                "profit_thb": pnl["profit_thb"],
            }

        reasons = [
            f"Profit {profit_pct:.2f}% reached AI take-profit review threshold {min_profit:.2f}%"
        ]
        score = 1.0

        rsi = float(signals.get("rsi", 50) or 50)
        price = float(signals.get("price", current_price) or current_price)
        bb_upper = float(signals.get("bb_upper", 0) or 0)
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = float(ai_prediction.get("confidence", 0) or 0)
        ai_change_pct = float(ai_prediction.get("price_change_pct", 0) or 0)

        if position.take_profit_price > 0 and current_price >= position.take_profit_price:
            reasons.append(
                f"Price touched configured TP target {position.take_profit_price:.2f}"
            )
            score += 0.75

        if rsi >= self.config.rsi_sell_threshold:
            reasons.append(f"RSI overbought at {rsi:.1f}")
            score += 0.5

        if bb_upper > 0 and price > bb_upper:
            reasons.append(f"Price {price:.2f} is above BB upper {bb_upper:.2f}")
            score += 0.5

        if signals.get("macd_bearish", False):
            reasons.append("MACD bearish crossover appeared")
            score += 0.25

        if ai_direction == "down":
            reasons.append(f"LSTM predicts pullback DOWN with confidence {ai_confidence:.2f}")
            score += 0.75
            if ai_confidence >= self.config.ai_take_profit_min_lstm_confidence:
                score += 0.75
        elif ai_direction == "up" and ai_confidence >= 0.6 and ai_change_pct > 0:
            reasons.append(f"LSTM still bullish ({ai_confidence:.2f}); delaying sell")
            score -= 0.5

        if ai_change_pct < 0:
            reasons.append(f"LSTM expects {ai_change_pct:.2f}% downside ahead")
            score += 0.5

        if rl_decision:
            rl_action = rl_decision.get("action_name", "HOLD")
            rl_confidence = float(rl_decision.get("confidence", 0) or 0)
            if rl_action == "SELL":
                reasons.append(
                    f"RL recommends SELL with confidence {rl_confidence:.2f}"
                )
                score += 0.75
                if rl_confidence >= self.config.ai_take_profit_min_rl_confidence:
                    score += 0.5

        should_sell = score >= 2.5
        return {
            "should_sell": should_sell,
            "score": score,
            "reasons": reasons,
            "reason": "AI_TAKE_PROFIT_TO_THB: " + "; ".join(reasons) if should_sell else "",
            "profit_pct": profit_pct,
            "profit_thb": pnl["profit_thb"],
        }

    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should be triggered."""
        if position.highest_price <= 0:
            position.highest_price = position.entry_price

        position.highest_price = max(position.highest_price, current_price)

        base_stop = position.entry_price * (1 - self.config.stop_loss_pct / 100)
        if position.stop_loss_price <= 0:
            position.stop_loss_price = base_stop

        dynamic_stop = max(position.stop_loss_price, base_stop)
        pnl = self.get_position_pnl(position, current_price)

        if pnl["profit_pct"] >= self.config.break_even_trigger_pct:
            break_even_stop = position.entry_price * (1 + self.config.break_even_buffer_pct / 100)
            dynamic_stop = max(dynamic_stop, break_even_stop)

        if self.config.trailing_stop_enabled and pnl["profit_pct"] >= self.config.trailing_stop_trigger_pct:
            trailing_stop = position.highest_price * (1 - self.config.trailing_stop_pct / 100)
            dynamic_stop = max(dynamic_stop, trailing_stop)

        position.stop_loss_price = dynamic_stop

        if current_price <= position.stop_loss_price:
            loss_pct = (
                (current_price - position.entry_price) / position.entry_price * 100
            )
            self.logger.log_stop_loss(
                position.symbol, position.entry_price, current_price, abs(loss_pct)
            )
            return True
        return False

    def check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit should be triggered."""
        if position.take_profit_price <= 0:
            # Calculate from config
            position.take_profit_price = position.entry_price * (
                1 + self.config.take_profit_pct / 100
            )

        if current_price >= position.take_profit_price:
            profit_pct = (
                (current_price - position.entry_price) / position.entry_price * 100
            )
            self.logger.log_take_profit(
                position.symbol, position.entry_price, current_price, profit_pct
            )
            return True
        return False

    def add_position(self, symbol: str, entry_price: float, amount: float) -> Position:
        """Add a new position."""
        sl_price = entry_price * (1 - self.config.stop_loss_pct / 100)
        tp_price = entry_price * (1 + self.config.take_profit_pct / 100)

        position = Position(
            symbol=symbol,
            side="long",
            entry_price=entry_price,
            amount=amount,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            highest_price=entry_price,
        )
        self.positions.append(position)

        self.logger.log_info(
            f"Position opened: {symbol} @ {entry_price:.2f} | "
            f"Amount: {amount:.8f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}"
        )
        return position

    def close_position(self, position: Position, exit_price: float, reason: str = ""):
        """Close a position and record it."""
        pnl = self.get_position_pnl(position, exit_price)
        profit_pct = pnl["profit_pct"]
        profit_thb = pnl["profit_thb"]

        trade_record = {
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "amount": position.amount,
            "profit_pct": profit_pct,
            "profit_thb": profit_thb,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "reason": reason,
        }
        self.trade_history.append(trade_record)

        if position in self.positions:
            self.positions.remove(position)

        self.logger.log_trade(
            "SELL", position.symbol, exit_price, position.amount,
            f"{reason} | P/L: {profit_pct:.2f}%"
        )
        return trade_record

    def get_open_positions(self, symbol: str = "") -> List[Position]:
        """Get open positions, optionally filtered by symbol."""
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return self.positions

    def get_trade_summary(self) -> Dict:
        """Get summary of all trades."""
        if not self.trade_history:
            return {"total_trades": 0}

        profits = [t["profit_pct"] for t in self.trade_history]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        return {
            "total_trades": len(self.trade_history),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.trade_history) * 100 if self.trade_history else 0,
            "avg_profit_pct": sum(profits) / len(profits) if profits else 0,
            "total_profit_pct": sum(profits),
            "total_profit_thb": sum(t["profit_thb"] for t in self.trade_history),
            "max_win_pct": max(profits) if profits else 0,
            "max_loss_pct": min(profits) if profits else 0,
        }
