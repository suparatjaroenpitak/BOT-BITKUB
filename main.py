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
from ai_model.llm_advisor import LLMBossAdvisor
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
        self.llm_boss_advisor = LLMBossAdvisor(config.ai, config.trading, self.logger)
        self.dashboard = TradingDashboard(self.strategy, self.risk_manager)

        self.running = False
        self.cycle_count = 0
        self.last_trade_llm_advice: Dict = {}
        self.paper_trade_enabled = bool(config.trading.paper_trade_enabled)
        self.paper_balance_thb = float(config.trading.paper_trade_start_balance_thb or 0.0)

    def _resolve_paper_start_balance(self, live_balance: float = 0.0) -> float:
        """Resolve the simulated starting THB balance."""
        configured_balance = float(self.config.trading.paper_trade_start_balance_thb or 0.0)
        if configured_balance > 0:
            return configured_balance
        return max(float(live_balance or 0.0), 0.0)

    def _ensure_paper_balance(self, live_balance: float = 0.0, force_reset: bool = False) -> float:
        """Initialize the paper wallet when needed."""
        if force_reset or self.paper_balance_thb <= 0:
            self.paper_balance_thb = self._resolve_paper_start_balance(live_balance)
        return self.paper_balance_thb

    def start(self):
        """Start the auto trading loop."""
        self.running = True
        self.logger.log_info("=" * 50)
        self.logger.log_info("Bitkub Auto Trading Bot STARTED")
        self.logger.log_info(f"Symbol: {self.config.trading.symbol}")
        self.logger.log_info(f"Interval: {self.config.trading.trading_interval_seconds}s")
        self.logger.log_info(f"Stop Loss: {self.config.trading.stop_loss_pct}%")
        self.logger.log_info(f"Take Profit: {self.config.trading.take_profit_pct}%")
        self.logger.log_info(f"Mode: {'PAPER' if self.paper_trade_enabled else 'LIVE'}")
        self.logger.log_info("=" * 50)

        # Load AI models
        self._load_models()

        print("\n🤖 Bitkub Auto Trading Bot started!")
        print(f"   Symbol: {self.config.trading.symbol}")
        print(f"   Interval: {self.config.trading.trading_interval_seconds}s")
        print(f"   Mode: {'PAPER' if self.paper_trade_enabled else 'LIVE'}")
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
            live_balance = self.client.get_thb_balance()
            if self.paper_trade_enabled:
                balance = self._ensure_paper_balance(live_balance)
            else:
                balance = live_balance
            positions = self.strategy.get_open_positions(symbol)
            risk_check = self.risk_manager.can_trade(balance, positions, signals, ai_prediction)

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
                    mode_label="PAPER" if self.paper_trade_enabled else "LIVE",
                )
            else:
                self.dashboard.print_compact_status(
                    balance,
                    current_price,
                    symbol,
                    action_name,
                    mode_label="PAPER" if self.paper_trade_enabled else "LIVE",
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

    def _request_trade_llm_advice(self, stage: str, symbol: str, current_price: float,
                                  balance: float, signals: Dict, ai_prediction: Dict,
                                  position=None, reasons=None, signal_score: float = 0.0) -> Dict:
        context = {
            "symbol": symbol,
            "current_price": round(float(current_price or 0.0), 8),
            "balance_thb": round(float(balance or 0.0), 2),
            "signal_score": round(float(signal_score or 0.0), 4),
            "position_entry_price": round(float(position.entry_price), 8) if position else None,
            "position_amount": round(float(position.amount), 8) if position else None,
            "signals": {
                "rsi": round(float(signals.get("rsi", 0.0) or 0.0), 2),
                "trend_down": bool(signals.get("trend_down", False)),
                "macd_bearish": bool(signals.get("macd_bearish", False)),
                "support_level": round(float(signals.get("support_level", 0.0) or 0.0), 8),
                "resistance_level": round(float(signals.get("resistance_level", 0.0) or 0.0), 8),
            },
            "ai": {
                "direction": ai_prediction.get("direction", "unknown"),
                "confidence": round(float(ai_prediction.get("confidence", 0.0) or 0.0), 4),
                "price_change_pct": round(float(ai_prediction.get("price_change_pct", 0.0) or 0.0), 4),
                "predicted_price": round(float(ai_prediction.get("predicted_price", 0.0) or 0.0), 8),
            },
            "reasons": reasons or [],
        }
        advice = self.llm_boss_advisor.review_trade_action(stage, context, {"stage": stage})
        self.last_trade_llm_advice = advice
        return advice

    def _llm_is_confident(self, advice: Dict, expected_action: str) -> bool:
        min_conf = float(getattr(self.config.ai, "llm_override_min_confidence", 0.68) or 0.68)
        return (
            advice.get("used")
            and advice.get("action") == expected_action
            and float(advice.get("confidence", 0.0) or 0.0) >= min_conf
        )

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
                trade_record = self._execute_sell(position, current_price, "TAKE_PROFIT_TO_THB", keep_principal=True)
                if trade_record:
                    return "PROFIT_CASHOUT" if trade_record.get("partial_close") else "TAKE_PROFIT"

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
                llm_sell_advice = self._request_trade_llm_advice(
                    "exit",
                    symbol,
                    current_price,
                    balance,
                    signals,
                    ai_prediction,
                    position=position,
                    reasons=sell_decision.get("reasons", []),
                    signal_score=sell_decision.get("score", 0.0),
                )
                should_sell = sell_decision["should_sell"]
                if self._llm_is_confident(llm_sell_advice, "HOLD"):
                    should_sell = False
                elif self._llm_is_confident(llm_sell_advice, "SELL"):
                    should_sell = True

                if not should_sell:
                    continue
                trade_record = self._execute_sell(
                    position, current_price,
                    "SELL_SIGNAL_TO_THB: " + "; ".join(
                        list(sell_decision["reasons"]) + ([f"LLM: {llm_sell_advice.get('reason', '')}"] if llm_sell_advice.get("used") else [])
                    ),
                    keep_principal=bool(
                        getattr(self.config.trading, "profit_cashout_enabled", False)
                        and (sell_decision.get("profit_lock_sell") or sell_decision.get("extended_rally_sell"))
                        and not sell_decision.get("panic_sell")
                    ),
                )
                sold_any = sold_any or bool(trade_record)
            if sold_any:
                return "SELL"

        # Check BUY
        if not positions:
            buy_decision = self.strategy.should_buy(signals, ai_prediction)
            llm_buy_advice = self._request_trade_llm_advice(
                "entry",
                symbol,
                current_price,
                balance,
                signals,
                ai_prediction,
                reasons=buy_decision.get("reasons", []),
                signal_score=buy_decision.get("score", 0.0),
            )
            should_buy = buy_decision["should_buy"]
            if self._llm_is_confident(llm_buy_advice, "SKIP"):
                should_buy = False
            elif self._llm_is_confident(llm_buy_advice, "BUY"):
                should_buy = True

            if should_buy:
                # Calculate position size
                sizing = self.risk_manager.calculate_position_size(
                    balance,
                    current_price,
                    buy_decision["signal_strength"],
                    signals,
                    ai_prediction,
                )
                if sizing["position_size_thb"] >= 100:  # Min 100 THB
                    self._execute_buy(
                        symbol, sizing["position_size_thb"], current_price,
                        "BUY_SIGNAL: " + "; ".join(
                            list(buy_decision["reasons"]) + ([f"LLM: {llm_buy_advice.get('reason', '')}"] if llm_buy_advice.get("used") else [])
                        )
                    )
                    return "BUY"

        return ""

    def _execute_buy(self, symbol: str, amount_thb: float, price: float,
                     reason: str):
        """Execute a buy order."""
        self.logger.log_trade("BUY", symbol, price, amount_thb / price, reason)

        if self.paper_trade_enabled:
            available_balance = self._ensure_paper_balance()
            if amount_thb > available_balance:
                self.logger.log_error(
                    "execute_buy",
                    Exception(
                        f"Paper buy failed for {symbol}: insufficient simulated balance {available_balance:.2f} THB"
                    ),
                )
                return None
            buy_fee_rate = max(float(getattr(self.config.trading, "buy_fee_rate", 0.0027) or 0.0), 0.0)
            crypto_amount = (amount_thb * (1 - buy_fee_rate)) / price if price else 0.0
            self.paper_balance_thb = max(available_balance - amount_thb, 0.0)
            self.strategy.add_position(
                symbol,
                price,
                crypto_amount,
                cost_thb=amount_thb,
                entry_fee_thb=max(amount_thb - (price * crypto_amount), 0.0),
            )
            self.logger.log_info(
                f"PAPER BUY executed: {symbol} @ {price:.2f} | remaining paper balance {self.paper_balance_thb:.2f}"
            )
            return {
                "rate": price,
                "amount": crypto_amount,
                "paper": True,
            }

        order = self.client.create_buy_order(symbol, amount_thb)
        if "_error" in order:
            self.logger.log_error(
                "execute_buy",
                Exception(f"Buy failed for {symbol}: {order.get('_raw', order)}"),
            )
            return None
        if order:
            crypto_amount = order.get("amount", amount_thb / price)
            executed_price = order.get("rate", price)
            self.strategy.add_position(
                symbol,
                executed_price,
                crypto_amount,
                cost_thb=float(order.get("cost", amount_thb) or amount_thb),
            )
            self.logger.log_info(f"BUY executed: {symbol} @ {executed_price:.2f}")
            return order
        return None

    def _execute_sell(self, position, current_price: float, reason: str,
                      keep_principal: bool = False):
        """Execute a sell order back into THB."""
        cashout_plan = None
        sell_amount = position.amount
        if keep_principal:
            cashout_plan = self.strategy.evaluate_profit_cashout(position, current_price)
            if cashout_plan["should_cashout"]:
                sell_amount = float(cashout_plan["sell_amount"] or 0.0)
            else:
                keep_principal = False

        estimated_value = sell_amount * current_price
        if estimated_value < 10:
            if not keep_principal and position in self.strategy.positions:
                self.strategy.positions.remove(position)
            self.logger.log_info(
                (
                    f"Skipping profit cashout for {position.symbol}: sell value {estimated_value:.2f} THB is below Bitkub minimum"
                    if keep_principal else
                    f"Skipping sell for {position.symbol}: remaining value {estimated_value:.2f} THB is below Bitkub minimum; position removed from bot tracking"
                )
            )
            return None

        if self.paper_trade_enabled:
            if keep_principal and cashout_plan and cashout_plan["should_cashout"]:
                trade_record = self.strategy.cash_out_profit(
                    position,
                    current_price,
                    float(cashout_plan["sell_amount"] or 0.0),
                    reason,
                )
            else:
                trade_record = self.strategy.close_position(position, current_price, reason)
            if trade_record:
                self.paper_balance_thb += float(trade_record.get("net_exit_value_thb", 0.0) or 0.0)
                self.risk_manager.record_trade_result(trade_record["profit_thb"])
            return trade_record

        order = self.client.create_sell_order(position.symbol, sell_amount)
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

        if keep_principal and cashout_plan and cashout_plan["should_cashout"]:
            trade_record = self.strategy.cash_out_profit(
                position,
                exit_price,
                float(order.get("amount", sell_amount) or sell_amount),
                reason,
                net_exit_value_thb=float(order.get("received", 0.0) or 0.0),
            )
        else:
            trade_record = self.strategy.close_position(
                position,
                exit_price,
                reason,
                net_exit_value_thb=float(order.get("received", 0.0) or 0.0),
            )
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

        self.lstm_predictor.load_model(df_with_indicators)
        self.rl_agent.load_model()

        print("\n🎉 All AI models trained, saved, and reloaded for future auto trade!")

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
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode")
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
    config.trading.paper_trade_enabled = bool(config.trading.paper_trade_enabled or args.paper)

    if args.sl:
        config.trading.stop_loss_pct = args.sl
    if args.tp:
        config.trading.take_profit_pct = args.tp

    # Validate API keys for live trading
    if (
        args.mode == "trade"
        and not config.trading.paper_trade_enabled
        and (not config.bitkub.api_key or not config.bitkub.api_secret)
    ):
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
            live_balance = bot.client.get_thb_balance()
            balance = bot._ensure_paper_balance(live_balance) if bot.paper_trade_enabled else live_balance
            ticker = bot.data_collector.fetch_ticker()
            current_price = ticker.get("last", 0)
            bot.dashboard.display(
                balance,
                {config.trading.symbol: current_price},
                mode_label="PAPER" if bot.paper_trade_enabled else "LIVE",
            )
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
