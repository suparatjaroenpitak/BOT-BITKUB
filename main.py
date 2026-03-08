"""
Bitkub Auto Trading Bot - Main Entry Point
ระบบทำงานทุก 30 วินาที: ดึงราคา → คำนวณ indicators → AI วิเคราะห์ → ตัดสินใจ → ส่งคำสั่ง
"""
import sys
import os
import time
import argparse
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from config import AppConfig
from exchange.bitkub_client import BitkubClient
from exchange.data_collector import MarketDataCollector
from strategy.indicators import TechnicalIndicatorEngine
from strategy.trading_strategy import TradingStrategy
from strategy.risk_management import RiskManager
from ai_model.lstm_model import LSTMPredictor
from ai_model.rl_model import RLTradingAgent, TradingEnvironment
from backtest.backtester import Backtester
from dashboard.trading_dashboard import TradingDashboard
from utils.logger import TradeLogger


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = TradeLogger(config.log.log_dir)

        # Initialize components
        self.client = BitkubClient(config.bitkub, self.logger)
        self.data_collector = MarketDataCollector(self.client, config.trading, self.logger)
        self.indicator_engine = TechnicalIndicatorEngine(
            support_resistance_window=config.trading.support_resistance_window
        )
        self.strategy = TradingStrategy(config.trading, self.logger)
        self.risk_manager = RiskManager(config.risk, self.logger)
        self.lstm_predictor = LSTMPredictor(config.ai, self.logger)
        self.rl_agent = RLTradingAgent(config.ai, logger=self.logger)
        self.dashboard = TradingDashboard(self.strategy, self.risk_manager)

        self.running = False
        self.cycle_count = 0

    def start(self):
        """Start the auto trading loop."""
        self.running = True
        self.logger.log_info("=" * 50)
        self.logger.log_info("Bitkub Auto Trading Bot STARTED")
        self.logger.log_info(f"Symbol: {self.config.trading.symbol}")
        self.logger.log_info(f"Interval: {self.config.trading.trading_interval_seconds}s")
        self.logger.log_info(f"Stop Loss: {self.config.trading.stop_loss_pct}%")
        self.logger.log_info(f"Take Profit: {self.config.trading.take_profit_pct}%")
        self.logger.log_info("=" * 50)

        # Load AI models
        self._load_models()

        print("\n🤖 Bitkub Auto Trading Bot started!")
        print(f"   Symbol: {self.config.trading.symbol}")
        print(f"   Interval: {self.config.trading.trading_interval_seconds}s")
        print("   Press Ctrl+C to stop\n")

        try:
            while self.running:
                self._trading_cycle()
                time.sleep(self.config.trading.trading_interval_seconds)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the trading bot."""
        self.running = False
        self.logger.log_info("Trading Bot STOPPED")
        print("\n🛑 Bot stopped.")

    def _load_models(self):
        """Load pre-trained AI models."""
        try:
            self.lstm_predictor.load_model()
            self.logger.log_info("LSTM model loaded")
        except Exception as e:
            self.logger.log_error("Loading LSTM model", e)

        try:
            self.rl_agent.load_model()
            self.logger.log_info("RL model loaded")
        except Exception as e:
            self.logger.log_error("Loading RL model", e)

    def _trading_cycle(self):
        """Execute one trading cycle."""
        self.cycle_count += 1
        symbol = self.config.trading.symbol

        try:
            # Step 1: Fetch market data
            df = self.data_collector.fetch_ohlcv(symbol)
            if df.empty:
                self.logger.log_error("Trading cycle", ValueError("No market data"))
                return

            # Step 2: Calculate indicators
            df_with_indicators = self.indicator_engine.add_all_indicators(df)
            signals = self.indicator_engine.get_signal_summary(df_with_indicators)

            current_price = signals.get("price", 0)
            if current_price <= 0:
                return

            # Step 3: AI analysis
            ai_prediction = self.lstm_predictor.predict(df_with_indicators)

            # Step 4: Get balance and check risk
            balance = self.client.get_thb_balance()
            positions = self.strategy.get_open_positions(symbol)
            risk_check = self.risk_manager.can_trade(balance, positions)

            # Step 5: Check existing positions for SL/TP
            action_taken = self._check_positions(
                symbol, current_price, df_with_indicators, ai_prediction
            )

            # Step 6: Make trading decision
            if not action_taken and risk_check["allowed"]:
                action_taken = self._make_decision(
                    symbol, signals, ai_prediction, balance, current_price
                )

            # Step 7: Display dashboard (every 5 cycles show full, otherwise compact)
            action_name = action_taken or "HOLD"
            if self.cycle_count % 5 == 0:
                self.dashboard.display(
                    balance,
                    {symbol: current_price},
                    signals,
                    ai_prediction,
                )
            else:
                self.dashboard.print_compact_status(
                    balance, current_price, symbol, action_name
                )

        except Exception as e:
            self.logger.log_error("Trading cycle", e)

    def _build_rl_live_decision(self, df_with_indicators: pd.DataFrame, position) -> Dict:
        """Build an RL decision for an open position using the latest market state."""
        try:
            state = self.rl_agent.build_live_state(
                df_with_indicators, has_position=True, entry_price=position.entry_price
            )
            if state is None:
                return {}
            return self.rl_agent.decide(state)
        except Exception as e:
            self.logger.log_error("RL live decision", e)
            return {}

    def _check_positions(self, symbol: str, current_price: float,
                         df_with_indicators: pd.DataFrame,
                         ai_prediction: Dict) -> str:
        """Check existing positions for AI cut loss, stop loss, and take profit."""
        for position in self.strategy.get_open_positions(symbol):
            rl_decision = self._build_rl_live_decision(df_with_indicators, position)
            ai_cutloss = self.strategy.evaluate_ai_cut_loss(
                position, current_price, ai_prediction, rl_decision
            )
            if ai_cutloss["should_sell"]:
                trade_record = self._execute_sell(position, current_price, ai_cutloss["reason"])
                if trade_record:
                    return "AI_CUTLOSS"

            # Stop Loss
            if self.strategy.check_stop_loss(position, current_price):
                trade_record = self._execute_sell(position, current_price, "STOP_LOSS_TO_THB")
                if trade_record:
                    return "STOP_LOSS"

            # Take Profit
            if self.strategy.check_take_profit(position, current_price):
                trade_record = self._execute_sell(position, current_price, "TAKE_PROFIT_TO_THB")
                if trade_record:
                    return "TAKE_PROFIT"

        return ""

    def _make_decision(self, symbol: str, signals: Dict, ai_prediction: Dict,
                       balance: float, current_price: float) -> str:
        """Make buy/sell decision."""
        positions = self.strategy.get_open_positions(symbol)

        # Check SELL first
        if positions:
            sold_any = False
            for position in positions:
                sell_decision = self.strategy.should_sell(signals, ai_prediction, position)
                if not sell_decision["should_sell"]:
                    continue
                trade_record = self._execute_sell(
                    position, current_price,
                    "SELL_SIGNAL_TO_THB: " + "; ".join(sell_decision["reasons"])
                )
                sold_any = sold_any or bool(trade_record)
            if sold_any:
                return "SELL"

        # Check BUY
        if not positions:
            buy_decision = self.strategy.should_buy(signals, ai_prediction)
            if buy_decision["should_buy"]:
                # Calculate position size
                sizing = self.risk_manager.calculate_position_size(
                    balance, current_price, buy_decision["signal_strength"]
                )
                if sizing["position_size_thb"] >= 100:  # Min 100 THB
                    self._execute_buy(
                        symbol, sizing["position_size_thb"], current_price,
                        "BUY_SIGNAL: " + "; ".join(buy_decision["reasons"])
                    )
                    return "BUY"

        return ""

    def _execute_buy(self, symbol: str, amount_thb: float, price: float,
                     reason: str):
        """Execute a buy order."""
        self.logger.log_trade("BUY", symbol, price, amount_thb / price, reason)

        order = self.client.create_buy_order(symbol, amount_thb)
        if "_error" in order:
            self.logger.log_error(
                "execute_buy",
                Exception(f"Buy failed for {symbol}: {order.get('_raw', order)}"),
            )
            return None
        if order:
            buy_fee_rate = max(float(getattr(self.config.trading, "buy_fee_rate", 0.0027) or 0.0), 0.0)
            crypto_amount = order.get("amount", (amount_thb * (1 - buy_fee_rate)) / price if price else 0.0)
            executed_price = order.get("rate", price)
            self.strategy.add_position(symbol, executed_price, crypto_amount)
            self.logger.log_info(f"BUY executed: {symbol} @ {executed_price:.2f}")
            return order
        return None

    def _execute_sell(self, position, current_price: float, reason: str):
        """Execute a sell order back into THB."""
        estimated_value = position.amount * current_price
        if estimated_value < 10:
            if position in self.strategy.positions:
                self.strategy.positions.remove(position)
            self.logger.log_info(
                f"Skipping sell for {position.symbol}: remaining value {estimated_value:.2f} THB is below Bitkub minimum; position removed from bot tracking"
            )
            return None

        order = self.client.create_sell_order(position.symbol, position.amount)
        if "_error" in order:
            self.logger.log_error(
                "execute_sell",
                Exception(f"Sell failed for {position.symbol}: {order.get('_raw', order)}"),
            )
            return None
        if not order:
            self.logger.log_error(
                "execute_sell",
                Exception(f"Sell failed for {position.symbol}: empty exchange response"),
            )
            return None

        exit_price = current_price
        if order:
            exit_price = order.get("rate", current_price)

        trade_record = self.strategy.close_position(position, exit_price, reason)
        if trade_record:
            self.risk_manager.record_trade_result(trade_record["profit_thb"])
        return trade_record

    def train_ai(self):
        """Train AI models on historical data."""
        print("📊 Fetching training data...")
        symbol = self.config.trading.symbol

        # Fetch historical data
        df = self.data_collector.fetch_ohlcv(symbol, "1h", 1000)
        if df.empty:
            print("❌ No data available for training")
            return

        # Add indicators
        df_with_indicators = self.indicator_engine.add_all_indicators(df)
        df_with_indicators = df_with_indicators.dropna()

        print(f"📈 Training data: {len(df_with_indicators)} bars")

        # Train LSTM
        print("\n🧠 Training LSTM model...")
        lstm_history = self.lstm_predictor.train(df_with_indicators)
        if "error" not in lstm_history:
            print("✅ LSTM training complete")

        # Train RL agent
        print("\n🎮 Training RL agent...")
        rl_history = self.rl_agent.train(df_with_indicators)
        print("✅ RL training complete")

        print("\n🎉 All AI models trained and saved!")

    def run_backtest(self):
        """Run backtesting on historical data."""
        print("📊 Fetching backtest data...")
        symbol = self.config.trading.symbol

        df = self.data_collector.fetch_ohlcv(symbol, "1h", 1000)
        if df.empty:
            print("❌ No data available for backtesting")
            return

        print(f"📈 Backtest data: {len(df)} bars")
        print("⏳ Running backtest...")

        backtester = Backtester(self.config.trading, self.config.risk, self.logger)
        result = backtester.run(df)
        backtester.print_results(result)



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bitkub Auto Trading Bot")
    parser.add_argument(
        "mode",
        choices=["trade", "train", "backtest", "dashboard", "gui"],
        default="gui",
        nargs="?",
        help="Operation mode: gui (GUI app), trade (live), train (AI), backtest, dashboard",
    )
    parser.add_argument("--symbol", default="BTC_THB", help="Trading pair")
    parser.add_argument("--interval", type=int, default=30, help="Trading interval (seconds)")
    parser.add_argument("--sl", type=float, help="Stop loss percentage")
    parser.add_argument("--tp", type=float, help="Take profit percentage")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # GUI mode - launch tkinter app
    if args.mode == "gui":
        from gui import TradingBotGUI
        app = TradingBotGUI()
        app.run()
        return

    # Load config
    config = AppConfig.from_env()
    config.trading.symbol = args.symbol
    config.trading.trading_interval_seconds = args.interval

    if args.sl:
        config.trading.stop_loss_pct = args.sl
    if args.tp:
        config.trading.take_profit_pct = args.tp

    # Validate API keys for live trading
    if args.mode == "trade" and (not config.bitkub.api_key or not config.bitkub.api_secret):
        print("❌ Error: BITKUB_API_KEY and BITKUB_API_SECRET environment variables required")
        print("   Set them with:")
        print('   $env:BITKUB_API_KEY = "your_api_key"')
        print('   $env:BITKUB_API_SECRET = "your_api_secret"')
        sys.exit(1)

    # Create bot
    bot = TradingBot(config)

    if args.mode == "trade":
        bot.start()
    elif args.mode == "train":
        bot.train_ai()
    elif args.mode == "backtest":
        bot.run_backtest()
    elif args.mode == "dashboard":
        # Single dashboard display
        try:
            balance = bot.client.get_thb_balance()
            ticker = bot.data_collector.fetch_ticker()
            current_price = ticker.get("last", 0)
            bot.dashboard.display(
                balance,
                {config.trading.symbol: current_price},
            )
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
