"""
Trading Dashboard - แสดงข้อมูล balance, positions, profit/loss, trading history
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

from strategy.trading_strategy import TradingStrategy, Position
from strategy.risk_management import RiskManager


class TradingDashboard:
    """Console-based trading dashboard."""

    def __init__(self, strategy: TradingStrategy, risk_manager: RiskManager):
        self.strategy = strategy
        self.risk_manager = risk_manager

    def display(self, balance: float, current_prices: Dict[str, float],
                signals: Optional[Dict] = None, ai_prediction: Optional[Dict] = None):
        """Display the full dashboard."""
        self._clear_screen()
        self._print_header()
        self._print_balance(balance, current_prices)
        self._print_positions(current_prices)
        self._print_risk_status(balance)
        if signals:
            self._print_signals(signals)
        if ai_prediction:
            self._print_ai_prediction(ai_prediction)
        self._print_trade_history()
        self._print_footer()

    def _clear_screen(self):
        """Clear terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def _print_header(self):
        """Print dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print()
        print("╔" + "═" * 70 + "╗")
        print("║" + "  🤖 BITKUB AUTO TRADING BOT - DASHBOARD".center(70) + "║")
        print("║" + f"  {now}".center(70) + "║")
        print("╠" + "═" * 70 + "╣")

    def _print_balance(self, balance: float, current_prices: Dict[str, float]):
        """Print balance information."""
        total_value = balance
        for pos in self.strategy.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            total_value += pos.amount * price

        print("║" + "  💰 BALANCE".ljust(70) + "║")
        print("║" + f"    THB Balance:    {balance:>15,.2f} THB".ljust(70) + "║")
        print("║" + f"    Total Value:    {total_value:>15,.2f} THB".ljust(70) + "║")
        print("╠" + "═" * 70 + "╣")

    def _print_positions(self, current_prices: Dict[str, float]):
        """Print open positions."""
        positions = self.strategy.get_open_positions()
        print("║" + f"  📊 OPEN POSITIONS ({len(positions)})".ljust(70) + "║")

        if not positions:
            print("║" + "    No open positions".ljust(70) + "║")
        else:
            print("║" + "    Symbol        Entry       Current     P/L%      Amount".ljust(70) + "║")
            print("║" + "    " + "-" * 60 + "".ljust(6) + "║")

            for pos in positions:
                current = current_prices.get(pos.symbol, pos.entry_price)
                pnl_pct = (current - pos.entry_price) / pos.entry_price * 100
                sign = "+" if pnl_pct > 0 else ""
                indicator = "🟢" if pnl_pct > 0 else "🔴"

                line = (
                    f"    {indicator} {pos.symbol:<10} "
                    f"{pos.entry_price:>10,.2f} "
                    f"{current:>10,.2f} "
                    f"{sign}{pnl_pct:>6.2f}%  "
                    f"{pos.amount:.8f}"
                )
                print("║" + line.ljust(70) + "║")

                sl_line = f"       SL: {pos.stop_loss_price:,.2f}  |  TP: {pos.take_profit_price:,.2f}"
                print("║" + sl_line.ljust(70) + "║")

        print("╠" + "═" * 70 + "╣")

    def _print_risk_status(self, balance: float):
        """Print risk management status."""
        positions = self.strategy.get_open_positions()
        risk = self.risk_manager.get_risk_status(balance, positions)

        print("║" + "  ⚠️  RISK STATUS".ljust(70) + "║")
        can_trade_str = "✅ YES" if risk["can_trade"] else "❌ NO"
        print("║" + f"    Can Trade:      {can_trade_str}".ljust(70) + "║")
        print("║" + f"    Exposure:       {risk['exposure_pct']:.1f}%".ljust(70) + "║")
        print("║" + f"    Daily Loss:     {risk['daily_loss']:,.2f} / {risk['max_daily_loss']:,.2f} THB".ljust(70) + "║")
        print("║" + f"    Positions:      {risk['open_positions']} / {risk['max_positions']}".ljust(70) + "║")
        print("╠" + "═" * 70 + "╣")

    def _print_signals(self, signals: Dict):
        """Print current technical signals."""
        print("║" + "  📈 TECHNICAL SIGNALS".ljust(70) + "║")
        print("║" + f"    RSI:            {signals.get('rsi', 0):.1f}".ljust(70) + "║")
        print("║" + f"    MACD:           {signals.get('macd', 0):.4f}".ljust(70) + "║")
        print("║" + f"    BB Upper:       {signals.get('bb_upper', 0):,.2f}".ljust(70) + "║")
        print("║" + f"    BB Lower:       {signals.get('bb_lower', 0):,.2f}".ljust(70) + "║")
        print("║" + f"    EMA 21:         {signals.get('ema_21', 0):,.2f}".ljust(70) + "║")
        print("║" + f"    Volume Ratio:   {signals.get('volume_ratio', 0):.2f}".ljust(70) + "║")
        print("╠" + "═" * 70 + "╣")

    def _print_ai_prediction(self, prediction: Dict):
        """Print AI prediction."""
        print("║" + "  🧠 AI PREDICTION".ljust(70) + "║")
        direction = prediction.get("direction", "unknown")
        arrow = "⬆️" if direction == "up" else "⬇️" if direction == "down" else "➡️"
        print("║" + f"    Direction:      {arrow} {direction.upper()}".ljust(70) + "║")
        print("║" + f"    Predicted:      {prediction.get('predicted_price', 0):,.2f}".ljust(70) + "║")
        print("║" + f"    Confidence:     {prediction.get('confidence', 0):.2%}".ljust(70) + "║")
        change = prediction.get("price_change_pct", 0)
        print("║" + f"    Change:         {'+' if change > 0 else ''}{change:.2f}%".ljust(70) + "║")
        print("╠" + "═" * 70 + "╣")

    def _print_trade_history(self):
        """Print recent trade history."""
        summary = self.strategy.get_trade_summary()
        history = self.strategy.trade_history[-5:]  # Last 5 trades

        print("║" + "  📜 TRADE HISTORY".ljust(70) + "║")
        print("║" + f"    Total Trades:   {summary.get('total_trades', 0)}".ljust(70) + "║")
        print("║" + f"    Win Rate:       {summary.get('win_rate', 0):.1f}%".ljust(70) + "║")
        print("║" + f"    Total P/L:      {summary.get('total_profit_thb', 0):>+,.2f} THB".ljust(70) + "║")

        if history:
            print("║" + "    Recent:".ljust(70) + "║")
            for t in history:
                sign = "+" if t["profit_pct"] > 0 else ""
                icon = "🟢" if t["profit_pct"] > 0 else "🔴"
                line = (
                    f"      {icon} {t['symbol']:<10} "
                    f"{t['entry_price']:>10,.2f} → {t['exit_price']:>10,.2f} "
                    f"({sign}{t['profit_pct']:.2f}%)"
                )
                print("║" + line.ljust(70) + "║")

        print("╠" + "═" * 70 + "╣")

    def _print_footer(self):
        """Print dashboard footer."""
        print("║" + "  Press Ctrl+C to stop the bot".center(70) + "║")
        print("╚" + "═" * 70 + "╝")
        print()

    def print_compact_status(self, balance: float, current_price: float,
                             symbol: str, action: str = "HOLD"):
        """Print a compact one-line status update."""
        positions = self.strategy.get_open_positions()
        total_value = balance
        for pos in positions:
            total_value += pos.amount * current_price

        now = datetime.now().strftime("%H:%M:%S")
        pnl = total_value - balance
        sign = "+" if pnl > 0 else ""

        print(
            f"[{now}] {symbol} @ {current_price:,.2f} | "
            f"Balance: {balance:,.2f} | "
            f"Value: {total_value:,.2f} | "
            f"P/L: {sign}{pnl:,.2f} | "
            f"Positions: {len(positions)} | "
            f"Action: {action}"
        )
