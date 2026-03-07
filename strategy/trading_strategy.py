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

    def should_buy(self, signals: Dict, ai_prediction: Dict) -> Dict:
        """
        Determine if we should BUY.

        Conditions:
        - RSI < 35 (oversold)
        - Price below EMA
        - AI prediction is UP
        """
        reasons = []
        score = 0

        rsi = signals.get("rsi", 50)
        price = signals.get("price", 0)
        ema_21 = signals.get("ema_21", 0)
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = ai_prediction.get("confidence", 0)

        # RSI condition
        if rsi < self.config.rsi_buy_threshold:
            score += 1
            reasons.append(f"RSI oversold ({rsi:.1f} < {self.config.rsi_buy_threshold})")

        # Price below EMA
        if price < ema_21 and ema_21 > 0:
            score += 1
            reasons.append(f"Price ({price:.2f}) below EMA21 ({ema_21:.2f})")

        # AI prediction UP
        if ai_direction == "up" and ai_confidence > 0.3:
            score += 1
            reasons.append(f"AI predicts UP (confidence: {ai_confidence:.2f})")

        # Additional confirmations
        if signals.get("macd_bullish", False):
            score += 0.5
            reasons.append("MACD bullish crossover")

        if signals.get("price_below_bb_lower", False):
            score += 0.5
            reasons.append("Price below Bollinger lower band")

        should = score >= 2  # Need at least 2 out of 3 main conditions

        return {
            "should_buy": should,
            "score": score,
            "reasons": reasons,
            "signal_strength": min(score / 3.0, 1.0),
        }

    def should_sell(self, signals: Dict, ai_prediction: Dict,
                    position: Optional[Position] = None) -> Dict:
        """
        Determine if we should SELL.

        Conditions:
        - RSI > 70 (overbought)
        - Price above Bollinger upper band
        - AI prediction is DOWN
        """
        reasons = []
        score = 0

        rsi = signals.get("rsi", 50)
        price = signals.get("price", 0)
        bb_upper = signals.get("bb_upper", float("inf"))
        ai_direction = ai_prediction.get("direction", "unknown")
        ai_confidence = ai_prediction.get("confidence", 0)

        # RSI condition
        if rsi > self.config.rsi_sell_threshold:
            score += 1
            reasons.append(f"RSI overbought ({rsi:.1f} > {self.config.rsi_sell_threshold})")

        # Price above Bollinger upper
        if price > bb_upper and bb_upper > 0:
            score += 1
            reasons.append(f"Price ({price:.2f}) above BB upper ({bb_upper:.2f})")

        # AI prediction DOWN
        if ai_direction == "down" and ai_confidence > 0.3:
            score += 1
            reasons.append(f"AI predicts DOWN (confidence: {ai_confidence:.2f})")

        # Additional confirmations
        if signals.get("macd_bearish", False):
            score += 0.5
            reasons.append("MACD bearish crossover")

        should = score >= 2

        return {
            "should_sell": should,
            "score": score,
            "reasons": reasons,
            "signal_strength": min(score / 3.0, 1.0),
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
        if position.stop_loss_price <= 0:
            # Calculate from config
            position.stop_loss_price = position.entry_price * (
                1 - self.config.stop_loss_pct / 100
            )

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
