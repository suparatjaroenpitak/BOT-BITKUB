"""
Backtesting System - ทดสอบกลยุทธ์กับข้อมูลย้อนหลัง
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy.indicators import TechnicalIndicatorEngine
from strategy.trading_strategy import TradingStrategy, Position
from config import TradingConfig, RiskConfig
from utils.logger import TradeLogger


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """Backtests trading strategies on historical data."""

    def __init__(self, trading_config: Optional[TradingConfig] = None,
                 risk_config: Optional[RiskConfig] = None,
                 logger: Optional[TradeLogger] = None):
        self.trading_config = trading_config or TradingConfig()
        self.risk_config = risk_config or RiskConfig()
        self.logger = logger or TradeLogger()
        self.indicator_engine = TechnicalIndicatorEngine()

    def run(self, df: pd.DataFrame, initial_balance: float = 100000.0,
            commission: float = 0.0025,
            ai_predictions: Optional[List[Dict]] = None) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: OHLCV DataFrame
            initial_balance: Starting balance in THB
            commission: Commission rate (0.25% for Bitkub)
            ai_predictions: Optional list of AI predictions per bar
        """
        # Add indicators
        df_with_indicators = self.indicator_engine.add_all_indicators(df)
        df_with_indicators = df_with_indicators.dropna()

        if df_with_indicators.empty:
            return BacktestResult()

        # Initialize strategy
        strategy = TradingStrategy(self.trading_config, self.logger)

        balance = initial_balance
        position: Optional[Position] = None
        trades = []
        equity_curve = [initial_balance]

        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            current_price = row["close"]

            # Build signal summary
            signals = self._build_signals(row)

            # Build AI prediction (use simple momentum if no AI)
            if ai_predictions and i < len(ai_predictions):
                ai_pred = ai_predictions[i]
            else:
                ai_pred = self._simple_prediction(df_with_indicators, i)

            # Check existing position for SL/TP
            if position:
                # Stop Loss
                if strategy.check_stop_loss(position, current_price):
                    revenue = position.amount * current_price * (1 - commission)
                    profit = revenue - (position.entry_price * position.amount)
                    balance += revenue
                    trade = {
                        "entry_price": position.entry_price,
                        "exit_price": current_price,
                        "amount": position.amount,
                        "profit_pct": (current_price - position.entry_price) / position.entry_price * 100,
                        "profit_thb": profit,
                        "reason": "STOP_LOSS",
                        "bar_index": i,
                    }
                    trades.append(trade)
                    position = None

                # Take Profit
                elif strategy.check_take_profit(position, current_price):
                    revenue = position.amount * current_price * (1 - commission)
                    profit = revenue - (position.entry_price * position.amount)
                    balance += revenue
                    trade = {
                        "entry_price": position.entry_price,
                        "exit_price": current_price,
                        "amount": position.amount,
                        "profit_pct": (current_price - position.entry_price) / position.entry_price * 100,
                        "profit_thb": profit,
                        "reason": "TAKE_PROFIT",
                        "bar_index": i,
                    }
                    trades.append(trade)
                    position = None

                # Sell signal
                else:
                    sell_decision = strategy.should_sell(signals, ai_pred)
                    if sell_decision["should_sell"]:
                        revenue = position.amount * current_price * (1 - commission)
                        profit = revenue - (position.entry_price * position.amount)
                        balance += revenue
                        trade = {
                            "entry_price": position.entry_price,
                            "exit_price": current_price,
                            "amount": position.amount,
                            "profit_pct": (current_price - position.entry_price) / position.entry_price * 100,
                            "profit_thb": profit,
                            "reason": "SELL_SIGNAL",
                            "bar_index": i,
                        }
                        trades.append(trade)
                        position = None

            # Check buy signal
            if position is None and balance > 100:
                buy_decision = strategy.should_buy(signals, ai_pred)
                if buy_decision["should_buy"]:
                    trade_size = min(
                        balance * 0.95,
                        self.risk_config.max_trade_size_thb,
                    )
                    cost = trade_size * (1 + commission)
                    if cost <= balance:
                        amount = trade_size / current_price
                        balance -= cost
                        position = Position(
                            symbol=self.trading_config.symbol,
                            side="long",
                            entry_price=current_price,
                            amount=amount,
                            stop_loss_price=current_price * (1 - self.trading_config.stop_loss_pct / 100),
                            take_profit_price=current_price * (1 + self.trading_config.take_profit_pct / 100),
                        )

            # Track equity
            portfolio_value = balance
            if position:
                portfolio_value += position.amount * current_price
            equity_curve.append(portfolio_value)

        # Close any remaining position at last price
        if position:
            last_price = df_with_indicators.iloc[-1]["close"]
            revenue = position.amount * last_price * (1 - commission)
            balance += revenue
            trades.append({
                "entry_price": position.entry_price,
                "exit_price": last_price,
                "amount": position.amount,
                "profit_pct": (last_price - position.entry_price) / position.entry_price * 100,
                "profit_thb": revenue - (position.entry_price * position.amount),
                "reason": "END_OF_DATA",
                "bar_index": len(df_with_indicators) - 1,
            })

        # Calculate results
        return self._calculate_results(initial_balance, balance, trades, equity_curve)

    def _build_signals(self, row: pd.Series) -> Dict:
        """Build signal dictionary from DataFrame row."""
        return {
            "price": row.get("close", 0),
            "rsi": row.get("rsi", 50),
            "macd": row.get("macd", 0),
            "macd_signal": row.get("macd_signal", 0),
            "macd_histogram": row.get("macd_histogram", 0),
            "bb_upper": row.get("bb_upper", 0),
            "bb_lower": row.get("bb_lower", 0),
            "bb_middle": row.get("bb_middle", 0),
            "ema_9": row.get("ema_9", 0),
            "ema_21": row.get("ema_21", 0),
            "sma_50": row.get("sma_50", 0),
            "volume_ratio": row.get("volume_ratio", 1),
            "rsi_oversold": row.get("rsi", 50) < 35,
            "rsi_overbought": row.get("rsi", 50) > 70,
            "price_below_ema": row.get("close", 0) < row.get("ema_21", 0),
            "price_above_bb_upper": row.get("close", 0) > row.get("bb_upper", float("inf")),
            "macd_bullish": row.get("macd", 0) > row.get("macd_signal", 0),
            "macd_bearish": row.get("macd", 0) < row.get("macd_signal", 0),
        }

    def _simple_prediction(self, df: pd.DataFrame, index: int) -> Dict:
        """Simple momentum-based prediction when AI is not available."""
        if index < 5:
            return {"direction": "unknown", "confidence": 0}

        prices = df["close"].iloc[max(0, index - 5):index + 1].values
        momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0

        return {
            "direction": "up" if momentum > 0 else "down",
            "confidence": min(abs(momentum) * 20, 1.0),
            "predicted_price": prices[-1] * (1 + momentum),
        }

    def _calculate_results(self, initial_balance: float, final_balance: float,
                           trades: List[Dict], equity_curve: List[float]) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        result = BacktestResult(
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_return_pct=(final_balance - initial_balance) / initial_balance * 100,
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_curve,
        )

        if trades:
            profits = [t["profit_pct"] for t in trades]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p <= 0]

            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = len(wins) / len(trades) * 100
            result.avg_profit_pct = sum(profits) / len(profits)

            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Max drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown_pct = max_dd

        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        return result

    def print_results(self, result: BacktestResult):
        """Print backtest results to console."""
        print("\n" + "=" * 60)
        print("           BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Initial Balance:   {result.initial_balance:>12,.2f} THB")
        print(f"  Final Balance:     {result.final_balance:>12,.2f} THB")
        print(f"  Total Return:      {result.total_return_pct:>11.2f}%")
        print(f"  Total Trades:      {result.total_trades:>12}")
        print(f"  Win Rate:          {result.win_rate:>11.1f}%")
        print(f"  Avg Profit/Trade:  {result.avg_profit_pct:>11.2f}%")
        print(f"  Profit Factor:     {result.profit_factor:>12.2f}")
        print(f"  Max Drawdown:      {result.max_drawdown_pct:>11.2f}%")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:>12.2f}")
        print("=" * 60)

        if result.trades:
            print(f"\n  Winning Trades:    {result.winning_trades}")
            print(f"  Losing Trades:     {result.losing_trades}")
            print("\n  Last 5 Trades:")
            for t in result.trades[-5:]:
                sign = "+" if t["profit_pct"] > 0 else ""
                print(
                    f"    {t['reason']:<15} "
                    f"Entry: {t['entry_price']:>10,.2f} → "
                    f"Exit: {t['exit_price']:>10,.2f} "
                    f"({sign}{t['profit_pct']:.2f}%)"
                )
        print()
