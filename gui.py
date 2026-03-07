"""
Bitkub Trading Bot - GUI Dashboard (Tkinter)
หน้า GUI แสดง balance, positions, indicators, AI predictions, trade history
พร้อมปุ่มควบคุม Start/Stop bot, Train AI, Backtest
"""
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
from typing import Dict, Optional

from config import AppConfig
from exchange.bitkub_client import BitkubClient
from exchange.data_collector import MarketDataCollector
from strategy.indicators import TechnicalIndicatorEngine
from strategy.trading_strategy import TradingStrategy, Position
from strategy.risk_management import RiskManager
from ai_model.lstm_model import LSTMPredictor
from ai_model.rl_model import RLTradingAgent
from backtest.backtester import Backtester
from utils.logger import TradeLogger


# ─── Color Theme ──────────────────────────────────────────────
COLORS = {
    "bg": "#1a1a2e",
    "bg_card": "#16213e",
    "bg_input": "#0f3460",
    "accent": "#e94560",
    "accent2": "#533483",
    "green": "#00b894",
    "red": "#e17055",
    "yellow": "#fdcb6e",
    "text": "#dfe6e9",
    "text_dim": "#636e72",
    "text_bright": "#ffffff",
    "border": "#2d3436",
}


class TradingBotGUI:
    """Main GUI application for the trading bot."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Bitkub Auto Trading Bot")
        self.root.geometry("1280x850")
        self.root.minsize(1100, 750)
        self.root.configure(bg=COLORS["bg"])

        # State
        self.config = AppConfig()
        self.bot_running = False
        self.bot_thread: Optional[threading.Thread] = None
        self.logger = TradeLogger("logs")

        # Components (initialized on connect)
        self.client: Optional[BitkubClient] = None
        self.data_collector: Optional[MarketDataCollector] = None
        self.indicator_engine = TechnicalIndicatorEngine(
            support_resistance_window=self.config.trading.support_resistance_window
        )
        self.strategy = TradingStrategy(self.config.trading, self.logger)
        self.risk_manager = RiskManager(self.config.risk, self.logger)
        self.lstm_predictor = LSTMPredictor(self.config.ai, self.logger)
        self.rl_agent = RLTradingAgent(self.config.ai, logger=self.logger)

        # Data variables
        self.current_price = tk.StringVar(value="0.00")
        self.price_change = tk.StringVar(value="0.00%")
        self.balance_thb = tk.StringVar(value="0.00 THB")
        self.total_value = tk.StringVar(value="0.00 THB")
        self.pnl_text = tk.StringVar(value="0.00 THB")
        self.bot_status = tk.StringVar(value="⏹ Stopped")
        self.rsi_val = tk.StringVar(value="-")
        self.macd_val = tk.StringVar(value="-")
        self.bb_val = tk.StringVar(value="-")
        self.ema_val = tk.StringVar(value="-")
        self.ai_direction = tk.StringVar(value="-")
        self.ai_confidence = tk.StringVar(value="-")
        self.ai_predicted = tk.StringVar(value="-")
        self.cycle_count = 0
        self.wallet_balances: Dict[str, Dict] = {}  # detailed balances
        self.wallet_price_map: Dict[str, float] = {}
        self.is_connected = False
        self.initial_portfolio_value = 0.0  # portfolio value when bot started
        self.realtime_pnl = tk.StringVar(value="0.00 THB")
        self.realtime_pnl_pct = tk.StringVar(value="0.00%")
        self.trade_amount_var = tk.StringVar(value="100")

        # Boss Mode state
        self.boss_mode = tk.BooleanVar(value=True)
        self.boss_cutloss_pct_var = tk.StringVar(value="0.6")
        self.boss_recovery_pct_var = tk.StringVar(value="0.8")
        self.boss_last_sell_price = 0.0  # track price at which boss sold
        self.boss_recovery_low_price = 0.0
        self.boss_rebuy_budget_thb = 0.0
        self.boss_waiting_recovery = False  # True = sold, waiting for price to recover before buy-back

        # Auto re-entry state
        self.auto_reentry_enabled = tk.BooleanVar(value=True)
        self.reentry_rise_pct_var = tk.StringVar(value="0.5")
        self.reentry_delay_pct_var = tk.StringVar(value="1.0")
        self.reentry_waiting = False
        self.reentry_symbol = ""
        self.reentry_last_exit_price = 0.0
        self.reentry_recovery_low_price = 0.0
        self.reentry_budget_thb = 0.0
        self.reentry_last_reason = ""

        # Bot performance tracking
        self.bot_start_time: Optional[datetime] = None
        self.bot_total_realized_pnl = 0.0      # cumulative realized P/L (THB)
        self.bot_total_trades = 0               # total completed trades
        self.bot_win_trades = 0                 # winning trades
        self.bot_lose_trades = 0                # losing trades
        self.bot_last_action = tk.StringVar(value="—")
        self.bot_uptime_str = tk.StringVar(value="00:00:00")
        self.bot_cycles_str = tk.StringVar(value="0")
        self.bot_realized_pnl_str = tk.StringVar(value="0.00 THB")
        self.bot_total_pnl_str = tk.StringVar(value="0.00 THB")
        self.bot_winrate_str = tk.StringVar(value="— %")
        self.auto_trade_amount_var = tk.StringVar(value="100")
        self.ai_scale_in_enabled = tk.BooleanVar(value=True)
        self.ai_scale_in_loss_pct_var = tk.StringVar(value="1.0")
        self.ai_take_profit_enabled = tk.BooleanVar(value=True)
        self.ai_take_profit_pct_var = tk.StringVar(value="1.2")
        self._runtime_settings_snapshot: Dict[str, object] = {}
        self._runtime_settings_invalid = False
        self._runtime_badges: Dict[str, tk.Label] = {}
        self._runtime_field_normalizers: Dict[str, object] = {}
        self._connect_in_progress = False
        self._wallet_refresh_in_progress = False
        self._market_refresh_in_progress = False
        self._bot_start_in_progress = False

        self._build_ui()
        self._setup_runtime_badges()
        self._apply_styles()

    # ─── UI Construction ──────────────────────────────────────

    def _build_ui(self):
        """Build the main UI layout."""
        # Main container
        main = tk.Frame(self.root, bg=COLORS["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top bar
        self._build_top_bar(main)

        # Body: left panel + right panel
        body = tk.Frame(main, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Left panel (wider)
        left = tk.Frame(body, bg=COLORS["bg"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel
        right = tk.Frame(body, bg=COLORS["bg"], width=380)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        # Left: Price + Bot Status + Wallet + Indicators + Positions + Log
        self._build_price_card(left)
        self._build_bot_status_card(left)
        self._build_wallet_card(left)
        self._build_indicators_card(left)
        self._build_positions_card(left)
        self._build_log_card(left)

        # Left: Real-time P/L card
        self._build_pnl_card(left)

        # Right: API Settings + Quick Trade + AI + Controls + Trade History
        self._build_api_settings(right)
        self._build_quick_trade_card(right)
        self._build_ai_card(right)
        self._build_controls_card(right)
        self._build_risk_card(right)
        self._build_history_card(right)

    def _build_top_bar(self, parent):
        """Build the top bar with title and status."""
        bar = tk.Frame(parent, bg=COLORS["bg_card"], height=50)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        # Title
        title = tk.Label(bar, text="🤖 BITKUB AUTO TRADING BOT",
                         font=("Segoe UI", 16, "bold"),
                         fg=COLORS["accent"], bg=COLORS["bg_card"])
        title.pack(side=tk.LEFT, padx=15)

        # Status
        status_frame = tk.Frame(bar, bg=COLORS["bg_card"])
        status_frame.pack(side=tk.RIGHT, padx=15)

        tk.Label(status_frame, text="Status:", font=("Segoe UI", 10),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self.status_label = tk.Label(status_frame, textvariable=self.bot_status,
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["yellow"], bg=COLORS["bg_card"])
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))

        # Time
        self.time_label = tk.Label(bar, text="", font=("Segoe UI", 10),
                                   fg=COLORS["text_dim"], bg=COLORS["bg_card"])
        self.time_label.pack(side=tk.RIGHT, padx=10)
        self._update_clock()

    def _build_price_card(self, parent):
        """Build the price display card."""
        card = self._make_card(parent, "📊 MARKET DATA")

        # Price row
        price_row = tk.Frame(card, bg=COLORS["bg_card"])
        price_row.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(price_row, text="Price:", font=("Segoe UI", 11),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)

        self.price_label = tk.Label(price_row, textvariable=self.current_price,
                                    font=("Segoe UI", 22, "bold"),
                                    fg=COLORS["text_bright"], bg=COLORS["bg_card"])
        self.price_label.pack(side=tk.LEFT, padx=(10, 5))

        tk.Label(price_row, text="THB", font=("Segoe UI", 11),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)

        self.change_label = tk.Label(price_row, textvariable=self.price_change,
                                     font=("Segoe UI", 12, "bold"),
                                     fg=COLORS["green"], bg=COLORS["bg_card"])
        self.change_label.pack(side=tk.RIGHT)

        # Balance row
        bal_row = tk.Frame(card, bg=COLORS["bg_card"])
        bal_row.pack(fill=tk.X, padx=10, pady=2)

        for label_text, var in [("Balance:", self.balance_thb),
                                 ("Total Value:", self.total_value),
                                 ("P/L:", self.pnl_text)]:
            f = tk.Frame(bal_row, bg=COLORS["bg_card"])
            f.pack(side=tk.LEFT, expand=True)
            tk.Label(f, text=label_text, font=("Segoe UI", 9),
                     fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack()
            tk.Label(f, textvariable=var, font=("Segoe UI", 11, "bold"),
                     fg=COLORS["text"], bg=COLORS["bg_card"]).pack()

    def _build_bot_status_card(self, parent):
        """Build bot live status & performance summary card."""
        card = self._make_card(parent, "🤖 BOT STATUS & PERFORMANCE")

        # ── Row 1: Status indicator + Uptime + Cycles ──
        row1 = tk.Frame(card, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, padx=10, pady=(5, 2))

        # Live / Stopped indicator
        self.bot_alive_label = tk.Label(
            row1, text="⏹ STOPPED", font=("Segoe UI", 11, "bold"),
            fg=COLORS["red"], bg=COLORS["bg_card"]
        )
        self.bot_alive_label.pack(side=tk.LEFT)

        # Uptime
        upt_f = tk.Frame(row1, bg=COLORS["bg_card"])
        upt_f.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(upt_f, text="⏱ Uptime:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        tk.Label(upt_f, textvariable=self.bot_uptime_str,
                 font=("Consolas", 10, "bold"),
                 fg=COLORS["text_bright"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(4, 0))

        # Cycles
        cyc_f = tk.Frame(row1, bg=COLORS["bg_card"])
        cyc_f.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(cyc_f, text="🔄 Cycles:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        tk.Label(cyc_f, textvariable=self.bot_cycles_str,
                 font=("Consolas", 10, "bold"),
                 fg=COLORS["text_bright"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(4, 0))

        # Last action
        act_f = tk.Frame(row1, bg=COLORS["bg_card"])
        act_f.pack(side=tk.RIGHT)
        tk.Label(act_f, text="Last:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self.bot_last_action_label = tk.Label(
            act_f, textvariable=self.bot_last_action,
            font=("Segoe UI", 9, "bold"),
            fg=COLORS["yellow"], bg=COLORS["bg_card"]
        )
        self.bot_last_action_label.pack(side=tk.LEFT, padx=(4, 0))

        # ── Row 2: Performance stats boxes ──
        row2 = tk.Frame(card, bg=COLORS["bg_card"])
        row2.pack(fill=tk.X, padx=10, pady=(2, 5))

        perf_items = [
            ("💰 Realized P/L", self.bot_realized_pnl_str, "realized_pnl"),
            ("📊 Total P/L", self.bot_total_pnl_str, "total_pnl"),
            ("🏆 Win Rate", self.bot_winrate_str, "winrate"),
        ]

        self._perf_labels = {}
        for i, (title, var, key) in enumerate(perf_items):
            f = tk.Frame(row2, bg=COLORS["bg_input"], padx=10, pady=5)
            f.grid(row=0, column=i, padx=4, sticky="ew")
            row2.columnconfigure(i, weight=1)

            tk.Label(f, text=title, font=("Segoe UI", 8),
                     fg=COLORS["text_dim"], bg=COLORS["bg_input"]).pack()
            lbl = tk.Label(f, textvariable=var, font=("Segoe UI", 13, "bold"),
                           fg=COLORS["text_bright"], bg=COLORS["bg_input"])
            lbl.pack()
            self._perf_labels[key] = lbl

        # ── Row 3: Win/Lose trade counter ──
        row3 = tk.Frame(card, bg=COLORS["bg_card"])
        row3.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.bot_trade_stats_label = tk.Label(
            row3,
            text="Trades: 0  |  ✅ Win: 0  |  ❌ Lose: 0",
            font=("Segoe UI", 9),
            fg=COLORS["text_dim"], bg=COLORS["bg_card"]
        )
        self.bot_trade_stats_label.pack(side=tk.LEFT)

        # Boss Mode status
        self.bot_boss_status_label = tk.Label(
            row3, text="", font=("Segoe UI", 9, "bold"),
            fg=COLORS["yellow"], bg=COLORS["bg_card"]
        )
        self.bot_boss_status_label.pack(side=tk.RIGHT)

    def _build_wallet_card(self, parent):
        """Build wallet / balance display card."""
        card = self._make_card(parent, "💰 WALLET BALANCE")

        # THB summary row
        thb_frame = tk.Frame(card, bg=COLORS["bg_card"])
        thb_frame.pack(fill=tk.X, padx=10, pady=(5, 2))

        tk.Label(thb_frame, text="THB Available:", font=("Segoe UI", 10),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self.wallet_thb_label = tk.Label(thb_frame, text="- (กรุณาเชื่อมต่อก่อน)",
                                          font=("Segoe UI", 14, "bold"),
                                          fg=COLORS["yellow"], bg=COLORS["bg_card"])
        self.wallet_thb_label.pack(side=tk.LEFT, padx=(10, 0))

        # Coins table
        columns = ("coin", "available", "reserved", "total", "value_thb")
        self.wallet_tree = ttk.Treeview(card, columns=columns, show="headings", height=4)

        headers = {
            "coin": ("Coin", 60),
            "available": ("Available", 110),
            "reserved": ("In Order", 90),
            "total": ("Total", 110),
            "value_thb": ("Value (THB)", 110),
        }
        for col, (text, width) in headers.items():
            self.wallet_tree.heading(col, text=text)
            self.wallet_tree.column(col, width=width, anchor="center")

        self.wallet_tree.pack(fill=tk.X, padx=10, pady=(2, 5))

        # Refresh wallet button
        tk.Button(
            card, text="🔄 Refresh Wallet", font=("Segoe UI", 9),
            bg=COLORS["accent2"], fg=COLORS["text_bright"],
            activebackground=COLORS["accent"], relief=tk.FLAT,
            cursor="hand2", command=self._refresh_wallet
        ).pack(fill=tk.X, padx=10, pady=(0, 5), ipady=2)

    def _build_indicators_card(self, parent):
        """Build technical indicators display."""
        card = self._make_card(parent, "📈 TECHNICAL INDICATORS")

        grid = tk.Frame(card, bg=COLORS["bg_card"])
        grid.pack(fill=tk.X, padx=10, pady=5)

        indicators = [
            ("RSI", self.rsi_val),
            ("MACD", self.macd_val),
            ("Bollinger", self.bb_val),
            ("EMA 21", self.ema_val),
        ]

        for i, (name, var) in enumerate(indicators):
            f = tk.Frame(grid, bg=COLORS["bg_input"], padx=10, pady=5)
            f.grid(row=0, column=i, padx=5, sticky="ew")
            grid.columnconfigure(i, weight=1)

            tk.Label(f, text=name, font=("Segoe UI", 9),
                     fg=COLORS["text_dim"], bg=COLORS["bg_input"]).pack()
            tk.Label(f, textvariable=var, font=("Segoe UI", 12, "bold"),
                     fg=COLORS["text_bright"], bg=COLORS["bg_input"]).pack()

    def _build_positions_card(self, parent):
        """Build open positions table."""
        card = self._make_card(parent, "💼 OPEN POSITIONS")

        columns = ("symbol", "entry", "current", "pnl", "amount", "sl", "tp")
        self.positions_tree = ttk.Treeview(card, columns=columns, show="headings",
                                           height=4)

        headers = {
            "symbol": ("Symbol", 80),
            "entry": ("Entry Price", 100),
            "current": ("Current", 100),
            "pnl": ("P/L %", 70),
            "amount": ("Amount", 100),
            "sl": ("Stop Loss", 90),
            "tp": ("Take Profit", 90),
        }
        for col, (text, width) in headers.items():
            self.positions_tree.heading(col, text=text)
            self.positions_tree.column(col, width=width, anchor="center")

        self.positions_tree.pack(fill=tk.X, padx=10, pady=5)

    def _build_log_card(self, parent):
        """Build log output area."""
        card = self._make_card(parent, "📝 ACTIVITY LOG")

        self.log_text = scrolledtext.ScrolledText(
            card, height=8, bg=COLORS["bg_input"], fg=COLORS["text"],
            font=("Consolas", 9), insertbackground=COLORS["text"],
            relief=tk.FLAT, state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def _build_api_settings(self, parent):
        """Build API settings panel."""
        card = self._make_card(parent, "🔑 API SETTINGS")

        # API Key
        tk.Label(card, text="API Key:", font=("Segoe UI", 9),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(anchor="w", padx=10)
        key_frame = tk.Frame(card, bg=COLORS["bg_card"])
        key_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        self.api_key_entry = tk.Entry(key_frame, font=("Consolas", 9), show="•",
                                      bg=COLORS["bg_input"], fg=COLORS["text"],
                                      insertbackground=COLORS["text"],
                                      relief=tk.FLAT)
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self._enable_paste(self.api_key_entry)
        self.api_key_show_btn = tk.Button(
            key_frame, text="SHOW", font=("Segoe UI", 8, "bold"), width=6,
            bg=COLORS["bg_input"], fg=COLORS["text_dim"],
            activebackground=COLORS["accent2"], relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._toggle_show(self.api_key_entry, self.api_key_show_btn)
        )
        self.api_key_show_btn.pack(side=tk.RIGHT, padx=(3, 0))
        tk.Button(
            key_frame, text="PASTE", font=("Segoe UI", 8, "bold"), width=6,
            bg=COLORS["accent2"], fg=COLORS["text_bright"],
            activebackground=COLORS["accent"], relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._paste_into_entry(self.api_key_entry, replace_all=True)
        ).pack(side=tk.RIGHT, padx=(3, 0))

        # API Secret
        tk.Label(card, text="API Secret:", font=("Segoe UI", 9),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(anchor="w", padx=10)
        sec_frame = tk.Frame(card, bg=COLORS["bg_card"])
        sec_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
        self.api_secret_entry = tk.Entry(sec_frame, font=("Consolas", 9), show="•",
                                         bg=COLORS["bg_input"], fg=COLORS["text"],
                                         insertbackground=COLORS["text"],
                                         relief=tk.FLAT)
        self.api_secret_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self._enable_paste(self.api_secret_entry)
        self.api_secret_show_btn = tk.Button(
            sec_frame, text="SHOW", font=("Segoe UI", 8, "bold"), width=6,
            bg=COLORS["bg_input"], fg=COLORS["text_dim"],
            activebackground=COLORS["accent2"], relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._toggle_show(self.api_secret_entry, self.api_secret_show_btn)
        )
        self.api_secret_show_btn.pack(side=tk.RIGHT, padx=(3, 0))
        tk.Button(
            sec_frame, text="PASTE", font=("Segoe UI", 8, "bold"), width=6,
            bg=COLORS["accent2"], fg=COLORS["text_bright"],
            activebackground=COLORS["accent"], relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._paste_into_entry(self.api_secret_entry, replace_all=True)
        ).pack(side=tk.RIGHT, padx=(3, 0))

        # Symbol selector
        sym_frame = tk.Frame(card, bg=COLORS["bg_card"])
        sym_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(sym_frame, text="เหรียญที่จะเทรด:", font=("Segoe UI", 9),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)

        self.symbol_var = tk.StringVar(value="BTC_THB")
        symbols = ["BTC_THB", "ETH_THB", "ADA_THB", "DOT_THB", "DOGE_THB",
                    "XRP_THB", "SOL_THB", "LINK_THB", "MATIC_THB"]
        self.symbol_combo = ttk.Combobox(sym_frame, textvariable=self.symbol_var,
                                          values=symbols, width=15, state="readonly")
        self.symbol_combo.pack(side=tk.RIGHT)
        self.symbol_combo.bind("<<ComboboxSelected>>", self._on_symbol_change)

        # Connect button
        self.connect_btn = tk.Button(
            card, text="🔗 Connect & Load Wallet", font=("Segoe UI", 10, "bold"),
            bg=COLORS["accent2"], fg=COLORS["text_bright"],
            activebackground=COLORS["accent"], relief=tk.FLAT,
            cursor="hand2", command=self._connect_exchange
        )
        self.connect_btn.pack(fill=tk.X, padx=10, pady=5, ipady=3)

        # Connection status
        self.conn_status_label = tk.Label(
            card, text="❌ ยังไม่ได้เชื่อมต่อ", font=("Segoe UI", 9),
            fg=COLORS["red"], bg=COLORS["bg_card"]
        )
        self.conn_status_label.pack(fill=tk.X, padx=10, pady=(0, 5))

    def _build_ai_card(self, parent):
        """Build AI prediction display."""
        card = self._make_card(parent, "🧠 AI PREDICTION")

        grid = tk.Frame(card, bg=COLORS["bg_card"])
        grid.pack(fill=tk.X, padx=10, pady=5)

        for i, (label, var) in enumerate([
            ("Direction", self.ai_direction),
            ("Confidence", self.ai_confidence),
            ("Predicted", self.ai_predicted),
        ]):
            f = tk.Frame(grid, bg=COLORS["bg_input"], padx=8, pady=4)
            f.grid(row=0, column=i, padx=3, sticky="ew")
            grid.columnconfigure(i, weight=1)

            tk.Label(f, text=label, font=("Segoe UI", 8),
                     fg=COLORS["text_dim"], bg=COLORS["bg_input"]).pack()
            tk.Label(f, textvariable=var, font=("Segoe UI", 10, "bold"),
                     fg=COLORS["text_bright"], bg=COLORS["bg_input"]).pack()

    def _build_controls_card(self, parent):
        """Build control buttons."""
        card = self._make_card(parent, "⚙️ CONTROLS")

        # Trading settings row
        settings = tk.Frame(card, bg=COLORS["bg_card"])
        settings.pack(fill=tk.X, padx=10, pady=5)

        # Interval
        f1 = tk.Frame(settings, bg=COLORS["bg_card"])
        f1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        interval_header = tk.Frame(f1, bg=COLORS["bg_card"])
        interval_header.pack(fill=tk.X)
        tk.Label(interval_header, text="Interval (s):", font=("Segoe UI", 8),
             fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(interval_header, "interval")
        self.interval_var = tk.StringVar(value="30")
        tk.Entry(f1, textvariable=self.interval_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(
            fill=tk.X, ipady=2)

        # Stop Loss %
        f2 = tk.Frame(settings, bg=COLORS["bg_card"])
        f2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        sl_header = tk.Frame(f2, bg=COLORS["bg_card"])
        sl_header.pack(fill=tk.X)
        tk.Label(sl_header, text="SL %:", font=("Segoe UI", 8),
             fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(sl_header, "sl")
        self.sl_var = tk.StringVar(value="1.8")
        tk.Entry(f2, textvariable=self.sl_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(
            fill=tk.X, ipady=2)

        # Take Profit %
        f3 = tk.Frame(settings, bg=COLORS["bg_card"])
        f3.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        tp_header = tk.Frame(f3, bg=COLORS["bg_card"])
        tp_header.pack(fill=tk.X)
        tk.Label(tp_header, text="TP %:", font=("Segoe UI", 8),
             fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(tp_header, "tp")
        self.tp_var = tk.StringVar(value="5.0")
        tk.Entry(f3, textvariable=self.tp_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(
            fill=tk.X, ipady=2)

        auto_buy_frame = tk.Frame(card, bg=COLORS["bg_card"])
        auto_buy_frame.pack(fill=tk.X, padx=10, pady=(2, 5))

        ab1 = tk.Frame(auto_buy_frame, bg=COLORS["bg_card"])
        ab1.pack(side=tk.LEFT, expand=True, fill=tk.X)
        auto_buy_header = tk.Frame(ab1, bg=COLORS["bg_card"])
        auto_buy_header.pack(fill=tk.X)
        tk.Label(auto_buy_header, text="Auto Buy (THB):", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(auto_buy_header, "auto_buy")
        auto_buy_entry = tk.Entry(
            ab1, textvariable=self.auto_trade_amount_var, font=("Segoe UI", 9),
            bg=COLORS["bg_input"], fg=COLORS["text"], width=10,
            relief=tk.FLAT, insertbackground=COLORS["text"]
        )
        auto_buy_entry.pack(fill=tk.X, ipady=2)
        self._enable_paste(auto_buy_entry)

        ai_manage_frame = tk.Frame(card, bg=COLORS["bg_card"])
        ai_manage_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        ai_row1 = tk.Frame(ai_manage_frame, bg=COLORS["bg_card"])
        ai_row1.pack(fill=tk.X, pady=(0, 2))
        self.ai_scale_in_check = tk.Checkbutton(
            ai_row1, text="🪜 AI Scale-In", font=("Segoe UI", 9, "bold"),
            variable=self.ai_scale_in_enabled, fg=COLORS["accent"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["accent"],
            cursor="hand2"
        )
        self.ai_scale_in_check.pack(side=tk.LEFT)
        self._create_runtime_badge(ai_row1, "ai_scale_in_enabled")
        tk.Label(ai_row1, text="Loss %:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(12, 4))
        self._create_runtime_badge(ai_row1, "ai_scale_in_loss_pct", padx=(0, 6))
        tk.Entry(ai_row1, textvariable=self.ai_scale_in_loss_pct_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(side=tk.LEFT, ipady=2)

        ai_row2 = tk.Frame(ai_manage_frame, bg=COLORS["bg_card"])
        ai_row2.pack(fill=tk.X)
        self.ai_take_profit_check = tk.Checkbutton(
            ai_row2, text="🤖 AI Take Profit", font=("Segoe UI", 9, "bold"),
            variable=self.ai_take_profit_enabled, fg=COLORS["green"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["green"],
            cursor="hand2"
        )
        self.ai_take_profit_check.pack(side=tk.LEFT)
        self._create_runtime_badge(ai_row2, "ai_take_profit_enabled")
        tk.Label(ai_row2, text="Profit %:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(12, 4))
        self._create_runtime_badge(ai_row2, "ai_take_profit_pct", padx=(0, 6))
        tk.Entry(ai_row2, textvariable=self.ai_take_profit_pct_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(side=tk.LEFT, ipady=2)

        # Boss Mode section
        boss_frame = tk.Frame(card, bg=COLORS["bg_card"])
        boss_frame.pack(fill=tk.X, padx=10, pady=(5, 0))

        self.boss_check = tk.Checkbutton(
            boss_frame, text="🏆 Boss Mode", font=("Segoe UI", 9, "bold"),
            variable=self.boss_mode, fg=COLORS["yellow"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["yellow"],
            cursor="hand2"
        )
        self.boss_check.pack(side=tk.LEFT)
        self._create_runtime_badge(boss_frame, "boss_mode")

        self.boss_status_label = tk.Label(
            boss_frame, text="ON", font=("Segoe UI", 8, "bold"),
            fg=COLORS["green"], bg=COLORS["bg_card"]
        )
        self.boss_status_label.pack(side=tk.LEFT, padx=(5, 0))
        self.boss_mode.trace_add("write", self._on_boss_toggle)

        boss_settings = tk.Frame(card, bg=COLORS["bg_card"])
        boss_settings.pack(fill=tk.X, padx=10, pady=(2, 5))

        # CutLoss %
        bf1 = tk.Frame(boss_settings, bg=COLORS["bg_card"])
        bf1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        boss_cutloss_header = tk.Frame(bf1, bg=COLORS["bg_card"])
        boss_cutloss_header.pack(fill=tk.X)
        tk.Label(boss_cutloss_header, text="CutLoss %:", font=("Segoe UI", 8),
             fg=COLORS["red"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(boss_cutloss_header, "boss_cutloss")
        tk.Entry(bf1, textvariable=self.boss_cutloss_pct_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(
            fill=tk.X, ipady=2)

        # Recovery %
        bf2 = tk.Frame(boss_settings, bg=COLORS["bg_card"])
        bf2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        boss_recovery_header = tk.Frame(bf2, bg=COLORS["bg_card"])
        boss_recovery_header.pack(fill=tk.X)
        tk.Label(boss_recovery_header, text="Recovery %:", font=("Segoe UI", 8),
             fg=COLORS["green"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(boss_recovery_header, "boss_recovery")
        tk.Entry(bf2, textvariable=self.boss_recovery_pct_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(
            fill=tk.X, ipady=2)

        # Auto re-entry section
        reentry_frame = tk.Frame(card, bg=COLORS["bg_card"])
        reentry_frame.pack(fill=tk.X, padx=10, pady=(2, 0))

        self.reentry_check = tk.Checkbutton(
            reentry_frame, text="🔁 Auto Re-Buy", font=("Segoe UI", 9, "bold"),
            variable=self.auto_reentry_enabled, fg=COLORS["accent"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["accent"],
            cursor="hand2"
        )
        self.reentry_check.pack(side=tk.LEFT)
        self._create_runtime_badge(reentry_frame, "reentry_enabled")

        reentry_settings = tk.Frame(card, bg=COLORS["bg_card"])
        reentry_settings.pack(fill=tk.X, padx=10, pady=(2, 5))

        rf1 = tk.Frame(reentry_settings, bg=COLORS["bg_card"])
        rf1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        reentry_rise_header = tk.Frame(rf1, bg=COLORS["bg_card"])
        reentry_rise_header.pack(fill=tk.X)
        tk.Label(reentry_rise_header, text="Buy Up %:", font=("Segoe UI", 8),
             fg=COLORS["green"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(reentry_rise_header, "reentry_rise")
        tk.Entry(rf1, textvariable=self.reentry_rise_pct_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(
            fill=tk.X, ipady=2)

        rf2 = tk.Frame(reentry_settings, bg=COLORS["bg_card"])
        rf2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        reentry_delay_header = tk.Frame(rf2, bg=COLORS["bg_card"])
        reentry_delay_header.pack(fill=tk.X)
        tk.Label(reentry_delay_header, text="Delay Down %:", font=("Segoe UI", 8),
             fg=COLORS["yellow"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        self._create_runtime_badge(reentry_delay_header, "reentry_delay")
        tk.Entry(rf2, textvariable=self.reentry_delay_pct_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(
            fill=tk.X, ipady=2)

        # Buttons row
        btn_frame = tk.Frame(card, bg=COLORS["bg_card"])
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_btn = tk.Button(
            btn_frame, text="▶ START BOT", font=("Segoe UI", 10, "bold"),
            bg=COLORS["text_dim"], fg=COLORS["text_bright"],
            activebackground="#00a884", relief=tk.FLAT,
            cursor="hand2", command=self._toggle_bot,
            state=tk.DISABLED
        )
        self.start_btn.pack(fill=tk.X, ipady=5, pady=(0, 5))

        btn_row2 = tk.Frame(btn_frame, bg=COLORS["bg_card"])
        btn_row2.pack(fill=tk.X)

        tk.Button(
            btn_row2, text="🧠 Train AI", font=("Segoe UI", 9),
            bg=COLORS["accent2"], fg=COLORS["text_bright"],
            activebackground=COLORS["accent"], relief=tk.FLAT,
            cursor="hand2", command=self._train_ai
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3), ipady=3)

        tk.Button(
            btn_row2, text="📊 Backtest", font=("Segoe UI", 9),
            bg=COLORS["accent2"], fg=COLORS["text_bright"],
            activebackground=COLORS["accent"], relief=tk.FLAT,
            cursor="hand2", command=self._run_backtest
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 3), ipady=3)

        tk.Button(
            btn_row2, text="🔄 Refresh", font=("Segoe UI", 9),
            bg=COLORS["accent2"], fg=COLORS["text_bright"],
            activebackground=COLORS["accent"], relief=tk.FLAT,
            cursor="hand2", command=self._manual_refresh
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0), ipady=3)

    def _build_risk_card(self, parent):
        """Build risk status display."""
        card = self._make_card(parent, "⚠️ RISK STATUS")

        self.risk_text = tk.Label(
            card, text="Daily Loss: 0 / 5,000 THB  |  Positions: 0 / 3",
            font=("Segoe UI", 9), fg=COLORS["text"],
            bg=COLORS["bg_card"], anchor="w"
        )
        self.risk_text.pack(fill=tk.X, padx=10, pady=5)

    def _build_quick_trade_card(self, parent):
        """Build manual quick trade card with Buy/Sell buttons."""
        card = self._make_card(parent, "💸 QUICK TRADE")

        # Amount input
        amt_frame = tk.Frame(card, bg=COLORS["bg_card"])
        amt_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(amt_frame, text="จำนวน (THB):", font=("Segoe UI", 9),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        trade_amt_entry = tk.Entry(
            amt_frame, textvariable=self.trade_amount_var,
            font=("Segoe UI", 10), bg=COLORS["bg_input"],
            fg=COLORS["text"], width=12, relief=tk.FLAT,
            insertbackground=COLORS["text"]
        )
        trade_amt_entry.pack(side=tk.RIGHT, ipady=2)
        self._enable_paste(trade_amt_entry)

        # Buy / Sell buttons
        btn_frame = tk.Frame(card, bg=COLORS["bg_card"])
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.quick_buy_btn = tk.Button(
            btn_frame, text="📈 BUY NOW", font=("Segoe UI", 10, "bold"),
            bg=COLORS["green"], fg=COLORS["text_bright"],
            activebackground="#00a884", relief=tk.FLAT,
            cursor="hand2", command=self._quick_buy, state=tk.DISABLED
        )
        self.quick_buy_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3), ipady=4)

        self.quick_sell_btn = tk.Button(
            btn_frame, text="📉 SELL ALL", font=("Segoe UI", 10, "bold"),
            bg=COLORS["red"], fg=COLORS["text_bright"],
            activebackground="#d63031", relief=tk.FLAT,
            cursor="hand2", command=self._quick_sell, state=tk.DISABLED
        )
        self.quick_sell_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(3, 0), ipady=4)

    def _build_pnl_card(self, parent):
        """Build real-time Profit/Loss monitoring card."""
        card = self._make_card(parent, "💹 REAL-TIME P/L")

        pnl_row = tk.Frame(card, bg=COLORS["bg_card"])
        pnl_row.pack(fill=tk.X, padx=10, pady=5)

        # P/L THB
        f1 = tk.Frame(pnl_row, bg=COLORS["bg_input"], padx=12, pady=6)
        f1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        tk.Label(f1, text="Unrealized P/L", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_input"]).pack()
        self.pnl_thb_label = tk.Label(f1, textvariable=self.realtime_pnl,
                                       font=("Segoe UI", 14, "bold"),
                                       fg=COLORS["text_bright"], bg=COLORS["bg_input"])
        self.pnl_thb_label.pack()

        # P/L %
        f2 = tk.Frame(pnl_row, bg=COLORS["bg_input"], padx=12, pady=6)
        f2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))
        tk.Label(f2, text="P/L %", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_input"]).pack()
        self.pnl_pct_label = tk.Label(f2, textvariable=self.realtime_pnl_pct,
                                       font=("Segoe UI", 14, "bold"),
                                       fg=COLORS["text_bright"], bg=COLORS["bg_input"])
        self.pnl_pct_label.pack()

        # Position summary text
        self.pnl_summary_label = tk.Label(
            card, text="ไม่มี position เปิดอยู่",
            font=("Segoe UI", 9), fg=COLORS["text_dim"],
            bg=COLORS["bg_card"], anchor="w"
        )
        self.pnl_summary_label.pack(fill=tk.X, padx=10, pady=(0, 5))

    def _build_history_card(self, parent):
        """Build trade history."""
        card = self._make_card(parent, "📜 TRADE HISTORY")

        columns = ("time", "type", "price", "pnl")
        self.history_tree = ttk.Treeview(card, columns=columns, show="headings",
                                          height=5)

        for col, (text, width) in {
            "time": ("Time", 70),
            "type": ("Type", 50),
            "price": ("Price", 90),
            "pnl": ("P/L %", 70),
        }.items():
            self.history_tree.heading(col, text=text)
            self.history_tree.column(col, width=width, anchor="center")

        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # ─── Helpers ──────────────────────────────────────────────

    def _make_card(self, parent, title: str) -> tk.Frame:
        """Create a styled card frame."""
        wrapper = tk.Frame(parent, bg=COLORS["bg"])
        wrapper.pack(fill=tk.X, pady=(0, 5))

        card = tk.Frame(wrapper, bg=COLORS["bg_card"], highlightbackground=COLORS["border"],
                        highlightthickness=1)
        card.pack(fill=tk.X)

        # Title bar
        tk.Label(card, text=title, font=("Segoe UI", 10, "bold"),
                 fg=COLORS["accent"], bg=COLORS["bg_card"],
                 anchor="w").pack(fill=tk.X, padx=10, pady=(8, 2))

        return card

    def _create_runtime_badge(self, parent, field_key: str, padx=(6, 0)) -> tk.Label:
        """Create a compact badge that shows whether a parameter is pending or already live."""
        badge = tk.Label(
            parent,
            text="READY",
            font=("Segoe UI", 7, "bold"),
            fg=COLORS["text_dim"],
            bg=COLORS["bg_input"],
            padx=6,
            pady=1,
        )
        badge.pack(side=tk.RIGHT, padx=padx)
        self._runtime_badges[field_key] = badge
        return badge

    def _setup_runtime_badges(self):
        """Track runtime-editable fields and refresh their badge state."""
        self._runtime_field_normalizers = {
            "interval": lambda: max(1, int(self.interval_var.get())),
            "sl": lambda: float(self.sl_var.get()),
            "tp": lambda: float(self.tp_var.get()),
            "auto_buy": lambda: float(self.auto_trade_amount_var.get()),
            "ai_scale_in_enabled": lambda: bool(self.ai_scale_in_enabled.get()),
            "ai_scale_in_loss_pct": lambda: float(self.ai_scale_in_loss_pct_var.get()),
            "ai_take_profit_enabled": lambda: bool(self.ai_take_profit_enabled.get()),
            "ai_take_profit_pct": lambda: float(self.ai_take_profit_pct_var.get()),
            "boss_mode": lambda: bool(self.boss_mode.get()),
            "boss_cutloss": lambda: float(self.boss_cutloss_pct_var.get()),
            "boss_recovery": lambda: float(self.boss_recovery_pct_var.get()),
            "reentry_enabled": lambda: bool(self.auto_reentry_enabled.get()),
            "reentry_rise": lambda: float(self.reentry_rise_pct_var.get()),
            "reentry_delay": lambda: float(self.reentry_delay_pct_var.get()),
        }
        tracked_vars = [
            self.interval_var,
            self.sl_var,
            self.tp_var,
            self.auto_trade_amount_var,
            self.ai_scale_in_enabled,
            self.ai_scale_in_loss_pct_var,
            self.ai_take_profit_enabled,
            self.ai_take_profit_pct_var,
            self.boss_mode,
            self.boss_cutloss_pct_var,
            self.boss_recovery_pct_var,
            self.auto_reentry_enabled,
            self.reentry_rise_pct_var,
            self.reentry_delay_pct_var,
        ]
        for variable in tracked_vars:
            variable.trace_add("write", self._on_runtime_field_change)
        self._refresh_runtime_badges()

    def _on_runtime_field_change(self, *args):
        """Refresh parameter badges when the user edits a runtime field."""
        self._refresh_runtime_badges()

    def _set_runtime_badge_state(self, badge: tk.Label, state: str):
        """Apply badge style for the current parameter state."""
        styles = {
            "LIVE": ("LIVE", COLORS["bg_input"], COLORS["green"]),
            "PENDING": ("PENDING", COLORS["bg_input"], COLORS["yellow"]),
            "EDIT": ("EDIT", COLORS["bg_input"], COLORS["yellow"]),
            "READY": ("READY", COLORS["bg_input"], COLORS["text_dim"]),
            "INVALID": ("INVALID", COLORS["bg_input"], COLORS["red"]),
        }
        text, bg, fg = styles.get(state, styles["READY"])
        badge.config(text=text, bg=bg, fg=fg)

    def _get_runtime_field_value(self, field_key: str):
        """Read a normalized value for a tracked runtime field."""
        normalizer = self._runtime_field_normalizers.get(field_key)
        if not normalizer:
            return None
        try:
            return normalizer()
        except (ValueError, tk.TclError):
            return None

    def _refresh_runtime_badges(self):
        """Update all runtime badges to show pending vs applied values."""
        for field_key, badge in self._runtime_badges.items():
            current_value = self._get_runtime_field_value(field_key)
            if current_value is None:
                self._set_runtime_badge_state(badge, "INVALID")
                continue

            has_snapshot = field_key in self._runtime_settings_snapshot
            applied_value = self._runtime_settings_snapshot.get(field_key)
            if has_snapshot and current_value == applied_value:
                self._set_runtime_badge_state(badge, "LIVE" if self.bot_running else "READY")
            else:
                self._set_runtime_badge_state(badge, "PENDING" if self.bot_running else "EDIT")

    @staticmethod
    def _enable_paste(entry: tk.Entry):
        """Enable Ctrl+V paste, Ctrl+A select-all, Ctrl+C copy, Ctrl+X cut on an Entry widget.
        Works correctly even with show='•' masked entries.
        """
        def _get_real_text():
            """Get the real text from the entry (not masked)."""
            return entry.get()

        def _paste(event=None):
            try:
                text = entry.clipboard_get()
                if not text:
                    return "break"
                text = text.strip()
                # If there's a selection, delete it first
                try:
                    if entry.selection_present():
                        entry.delete(tk.SEL_FIRST, tk.SEL_LAST)
                except tk.TclError:
                    pass
                entry.insert(tk.INSERT, text)
            except tk.TclError:
                pass
            return "break"

        def _paste_replace_all(event=None):
            """Clear field and paste — useful for API key fields."""
            try:
                text = entry.clipboard_get()
                if not text:
                    return "break"
                text = text.strip()
                entry.delete(0, tk.END)
                entry.insert(0, text)
            except tk.TclError:
                pass
            return "break"

        def _select_all(event=None):
            entry.select_range(0, tk.END)
            entry.icursor(tk.END)
            return "break"

        def _copy(event=None):
            try:
                real = _get_real_text()
                if entry.selection_present():
                    s = entry.index(tk.SEL_FIRST)
                    e = entry.index(tk.SEL_LAST)
                    text = real[s:e]
                else:
                    text = real
                if text:
                    entry.clipboard_clear()
                    entry.clipboard_append(text)
            except tk.TclError:
                pass
            return "break"

        def _cut(event=None):
            _copy(event)
            try:
                if entry.selection_present():
                    entry.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                pass
            return "break"

        entry.bind("<Control-v>", _paste)
        entry.bind("<Control-V>", _paste)
        entry.bind("<Shift-Insert>", _paste)
        entry.bind("<Control-a>", _select_all)
        entry.bind("<Control-A>", _select_all)
        entry.bind("<Control-c>", _copy)
        entry.bind("<Control-C>", _copy)
        entry.bind("<Control-x>", _cut)
        entry.bind("<Control-X>", _cut)
        # Also handle right-click paste on Windows
        def _right_click_paste(event):
            menu = tk.Menu(entry, tearoff=0, bg="#2d3436", fg="#dfe6e9",
                           activebackground="#e94560", activeforeground="#ffffff",
                           font=("Segoe UI", 9))
            menu.add_command(label="📋 Paste  (Ctrl+V)", command=_paste)
            menu.add_command(label="📋 Paste (แทนที่ทั้งหมด)", command=_paste_replace_all)
            menu.add_command(label="📑 Copy   (Ctrl+C)", command=_copy)
            menu.add_command(label="✂️ Cut    (Ctrl+X)", command=_cut)
            menu.add_command(label="🔘 Select All (Ctrl+A)", command=_select_all)
            menu.add_separator()
            menu.add_command(label="🗑️ Clear", command=lambda: entry.delete(0, tk.END))
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
        entry.bind("<Button-3>", _right_click_paste)

    @staticmethod
    def _paste_into_entry(entry: tk.Entry, replace_all: bool = True):
        """Paste clipboard text into an entry, optionally replacing existing text."""
        try:
            text = entry.clipboard_get()
        except tk.TclError:
            return

        if not text:
            return

        text = text.strip()
        entry.focus_set()
        if replace_all:
            entry.delete(0, tk.END)
            entry.insert(0, text)
        else:
            entry.insert(tk.INSERT, text)

    @staticmethod
    def _toggle_show(entry: tk.Entry, button: Optional[tk.Button] = None):
        """Toggle show/hide text in a masked Entry widget."""
        current = entry.cget("show")
        is_hidden = current == "•"
        entry.config(show="" if is_hidden else "•")
        if button is not None:
            button.config(text="HIDE" if is_hidden else "SHOW")

    def _apply_styles(self):
        """Apply ttk styles for Treeview."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Treeview",
                        background=COLORS["bg_input"],
                        foreground=COLORS["text"],
                        fieldbackground=COLORS["bg_input"],
                        font=("Segoe UI", 9),
                        rowheight=25)
        style.configure("Treeview.Heading",
                        background=COLORS["bg_card"],
                        foreground=COLORS["accent"],
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", COLORS["accent2"])])

        style.configure("TCombobox",
                        fieldbackground=COLORS["bg_input"],
                        background=COLORS["bg_card"],
                        foreground=COLORS["text"])

    def _update_clock(self):
        """Update the clock in the top bar."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=now)
        self.root.after(1000, self._update_clock)

    def _log(self, message: str, level: str = "INFO"):
        """Add message to the activity log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌",
                   "TRADE": "💰", "AI": "🧠", "SUCCESS": "✅"}
        icon = prefix.get(level, "•")

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {icon} {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ─── Actions ──────────────────────────────────────────────

    def _connect_exchange(self):
        """Connect to Bitkub exchange and load wallet."""
        if self._connect_in_progress:
            return

        api_key = self.api_key_entry.get().strip()
        api_secret = self.api_secret_entry.get().strip()
        symbol = self.symbol_var.get()

        if not api_key or not api_secret:
            messagebox.showwarning("API Error", "กรุณาใส่ API Key และ API Secret")
            return

        self.config.bitkub.api_key = api_key
        self.config.bitkub.api_secret = api_secret
        self.config.trading.symbol = symbol
        self._connect_in_progress = True
        self.connect_btn.config(state=tk.DISABLED, text="⏳ Connecting...", bg=COLORS["yellow"])
        self.conn_status_label.config(text="⏳ กำลังเชื่อมต่อ...", fg=COLORS["yellow"])
        self._log("กำลังเชื่อมต่อ Bitkub และโหลดข้อมูลเริ่มต้น...", "INFO")
        threading.Thread(target=self._do_connect_exchange, args=(symbol,), daemon=True).start()

    def _do_connect_exchange(self, symbol: str):
        """Connect to Bitkub exchange in a background thread."""
        try:
            client = BitkubClient(self.config.bitkub, self.logger)
            data_collector = MarketDataCollector(client, self.config.trading, self.logger)
            market_snapshot = self._fetch_market_snapshot(client, symbol)
            if not market_snapshot:
                raise RuntimeError("Connection failed - no ticker data")
            wallet_snapshot = self._fetch_wallet_snapshot(client)
            self.root.after(
                0,
                self._finish_connect_exchange,
                client,
                data_collector,
                wallet_snapshot,
                market_snapshot,
            )
        except Exception as e:
            self.root.after(0, self._handle_connect_error, str(e))

    def _finish_connect_exchange(
        self,
        client: BitkubClient,
        data_collector: MarketDataCollector,
        wallet_snapshot: Dict,
        market_snapshot: Dict,
    ):
        """Apply connection results on the Tkinter thread."""
        self.client = client
        self.data_collector = data_collector
        self.is_connected = True
        self._connect_in_progress = False
        self._log("Connected to Bitkub!", "SUCCESS")
        self.connect_btn.config(state=tk.NORMAL, text="✅ Connected", bg=COLORS["green"])
        self.conn_status_label.config(text="✅ เชื่อมต่อสำเร็จ", fg=COLORS["green"])
        self.start_btn.config(state=tk.NORMAL, bg=COLORS["green"])
        self.quick_buy_btn.config(state=tk.NORMAL)
        self.quick_sell_btn.config(state=tk.NORMAL)
        self._apply_wallet_snapshot(wallet_snapshot, log_success=True)
        self._apply_market_snapshot(market_snapshot)

    def _handle_connect_error(self, error_message: str):
        """Show connection errors on the Tkinter thread."""
        self._connect_in_progress = False
        self.is_connected = False
        self.connect_btn.config(state=tk.NORMAL, text="🔗 Connect & Load Wallet", bg=COLORS["accent"])
        self.conn_status_label.config(text=f"❌ {error_message}", fg=COLORS["red"])
        self._log(f"Connection error: {error_message}", "ERROR")
        messagebox.showerror("Connection Error", error_message)

    def _load_wallet(self):
        """Load and display wallet balances."""
        if not self.client or self._wallet_refresh_in_progress:
            return

        self._wallet_refresh_in_progress = True
        threading.Thread(target=self._do_load_wallet, daemon=True).start()

    def _do_load_wallet(self):
        """Fetch wallet balances in a background thread."""
        try:
            snapshot = self._fetch_wallet_snapshot(self.client)
            self.root.after(0, self._finish_load_wallet, snapshot)
        except Exception as e:
            self.root.after(0, self._handle_wallet_error, str(e))

    def _finish_load_wallet(self, snapshot: Dict):
        """Apply wallet data on the Tkinter thread."""
        self._wallet_refresh_in_progress = False
        self._apply_wallet_snapshot(snapshot, log_success=True)

    def _handle_wallet_error(self, error_message: str):
        """Handle wallet loading errors on the Tkinter thread."""
        self._wallet_refresh_in_progress = False
        self._log(f"Wallet error: {error_message}", "ERROR")

    def _fetch_wallet_snapshot(self, client: BitkubClient) -> Dict:
        """Build wallet data without touching Tkinter widgets."""
        wallet_balances = client.get_balance_detail()
        thb_info = wallet_balances.get("THB", {})
        thb_avail = thb_info.get("available", 0)
        thb_total = thb_info.get("total", 0)

        all_tickers = {}
        price_map = {}
        try:
            all_tickers = client._get("/api/market/ticker")
        except Exception:
            pass

        total_value_thb = thb_total
        coin_symbols = []
        wallet_rows = []

        for coin, info in sorted(wallet_balances.items()):
            if coin == "THB":
                continue

            avail = info.get("available", 0)
            reserved = info.get("reserved", 0)
            total = info.get("total", 0)

            ticker_key = f"{coin}_THB"
            value_thb = 0
            ticker_data = all_tickers.get(ticker_key, {})
            if ticker_data:
                last_price = float(ticker_data.get("last", 0))
                if last_price > 0:
                    price_map[ticker_key] = last_price
                value_thb = total * last_price
                total_value_thb += value_thb

            wallet_rows.append((
                coin,
                f"{avail:.8g}",
                f"{reserved:.8g}",
                f"{total:.8g}",
                f"฿{value_thb:,.2f}" if value_thb > 0 else "-",
            ))

            if total > 0:
                coin_symbols.append(f"{coin}_THB")

        popular = ["BTC_THB", "ETH_THB", "ADA_THB", "DOT_THB", "DOGE_THB",
                   "XRP_THB", "SOL_THB", "LINK_THB"]
        all_symbols = list(dict.fromkeys(coin_symbols + popular))

        return {
            "wallet_balances": wallet_balances,
            "thb_avail": thb_avail,
            "thb_total": thb_total,
            "wallet_rows": wallet_rows,
            "price_map": price_map,
            "total_value_thb": total_value_thb,
            "all_symbols": all_symbols,
            "coin_count": len([coin for coin in wallet_balances if coin != "THB"]),
        }

    def _sync_selected_position_from_wallet(self, snapshot: Optional[Dict] = None,
                                            log_sync: bool = False):
        """Adopt the currently selected wallet holding into bot tracking."""
        symbol = self.config.trading.symbol
        coin = symbol.split("_")[0]
        wallet_state = snapshot.get("wallet_balances", {}) if snapshot else self.wallet_balances
        price_map = snapshot.get("price_map", {}) if snapshot else self.wallet_price_map

        coin_info = wallet_state.get(coin, {})
        available_amount = float(coin_info.get("available", 0) or 0)
        reserved_amount = float(coin_info.get("reserved", 0) or 0)
        tracked_positions = self.strategy.get_open_positions(symbol)
        tracked_amount = sum(max(position.amount, 0.0) for position in tracked_positions)
        tolerance = max(1e-8, available_amount * 0.001)

        if available_amount <= 0:
            if reserved_amount <= 0 and tracked_positions and not self.bot_running:
                self.strategy.positions = [
                    position for position in self.strategy.positions
                    if position.symbol != symbol
                ]
                if log_sync:
                    self._log(
                        f"ล้าง position ที่ sync ไว้สำหรับ {symbol} เพราะไม่พบยอดคงเหลือในกระเป๋าแล้ว",
                        "INFO",
                    )
            return

        if tracked_positions:
            if len(tracked_positions) == 1 and abs(tracked_amount - available_amount) > tolerance:
                tracked_positions[0].amount = available_amount
                if log_sync:
                    self._log(
                        f"อัปเดต position จากกระเป๋าจริง {symbol} เป็น {available_amount:.8g}",
                        "INFO",
                    )
            return

        market_price = float(price_map.get(symbol, 0) or 0)
        if market_price <= 0:
            try:
                market_price = float(self.current_price.get().replace(",", "").split()[0])
            except (ValueError, IndexError):
                market_price = 0.0

        if market_price <= 0:
            return

        self.strategy.add_position(symbol, market_price, available_amount)
        if log_sync:
            reserved_text = (
                f" | Reserved: {reserved_amount:.8g}" if reserved_amount > 0 else ""
            )
            self._log(
                f"📋 Sync position จากกระเป๋าจริง {symbol}: {available_amount:.8g} @ {market_price:,.2f}{reserved_text}",
                "INFO",
            )

    def _apply_wallet_snapshot(self, snapshot: Dict, log_success: bool = True):
        """Update wallet widgets from pre-fetched data."""
        self.wallet_balances = snapshot.get("wallet_balances", {})
        self.wallet_price_map = snapshot.get("price_map", {})
        thb_avail = snapshot.get("thb_avail", 0)
        thb_total = snapshot.get("thb_total", 0)

        self.wallet_thb_label.config(
            text=f"฿{thb_avail:,.2f}  (รวม: ฿{thb_total:,.2f})",
            fg=COLORS["green"] if thb_avail > 0 else COLORS["red"],
        )
        self.balance_thb.set(f"{thb_avail:,.2f} THB")

        for item in self.wallet_tree.get_children():
            self.wallet_tree.delete(item)

        for row in snapshot.get("wallet_rows", []):
            self.wallet_tree.insert("", "end", values=row)

        self.total_value.set(f"{snapshot.get('total_value_thb', 0):,.2f} THB")
        self.symbol_combo.config(values=snapshot.get("all_symbols", []))
        self._sync_selected_position_from_wallet(snapshot, log_sync=log_success)

        if log_success:
            self._log(
                f"Wallet loaded: ฿{thb_avail:,.2f} THB + {snapshot.get('coin_count', 0)} coins",
                "SUCCESS",
            )
            self._log(
                f"Total portfolio value: ฿{snapshot.get('total_value_thb', 0):,.2f} THB",
                "INFO",
            )

    def _refresh_wallet(self):
        """Refresh wallet balance in background."""
        if not self.is_connected:
            messagebox.showinfo("Info", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return
        self._log("กำลังรีเฟรช Wallet...", "INFO")
        self._load_wallet()

    def _on_symbol_change(self, event=None):
        """Handle symbol change from combobox."""
        new_symbol = self.symbol_var.get()

        if self.bot_running:
            positions = self.strategy.get_open_positions()
            pending_reentry = self.reentry_waiting or self.boss_waiting_recovery
            if positions or pending_reentry:
                self.root.after(0, lambda: self.symbol_var.set(self.config.trading.symbol))
                self._log(
                    "เปลี่ยนเหรียญระหว่างรันไม่ได้ขณะยังมี position หรือสถานะรอซื้อคืน",
                    "WARN",
                )
                return

        self.config.trading.symbol = new_symbol
        self._log(f"เปลี่ยนเหรียญเป็น: {new_symbol}", "INFO")
        self._sync_selected_position_from_wallet(log_sync=True)

        if self.is_connected and self.client:
            # Update price for new symbol
            threading.Thread(target=self._update_symbol_price, daemon=True).start()

    def _update_symbol_price(self):
        """Update price display for current symbol."""
        if not self.client:
            return
        try:
            symbol = self.config.trading.symbol
            ticker = self.client.get_ticker(symbol)
            if ticker:
                price = ticker.get("last", 0)
                change = ticker.get("change", 0)
                self.root.after(0, lambda: self.current_price.set(f"{price:,.2f}"))
                self.root.after(0, lambda: self.price_change.set(f"{change:+.2f}%"))
                color = COLORS["green"] if change >= 0 else COLORS["red"]
                self.root.after(0, lambda: self.change_label.config(fg=color))
        except Exception:
            pass

    def _quick_buy(self):
        """Manual buy: place market buy order and register position for bot monitoring."""
        if not self.is_connected or not self.client:
            messagebox.showwarning("Not Connected", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return

        try:
            amount_thb = float(self.trade_amount_var.get())
        except ValueError:
            messagebox.showerror("Error", "กรุณาใส่จำนวนเงิน THB ที่ถูกต้อง")
            return

        if amount_thb < 10:
            messagebox.showwarning("Amount Too Low", "จำนวนเงินต่ำสุด 10 THB")
            return

        # Check THB balance
        thb_info = self.wallet_balances.get("THB", {})
        thb_avail = thb_info.get("available", 0)
        if amount_thb > thb_avail:
            messagebox.showwarning("Insufficient Balance",
                f"ยอด THB ไม่พอ\nต้องการ: ฿{amount_thb:,.2f}\nมี: ฿{thb_avail:,.2f}")
            return

        symbol = self.config.trading.symbol
        confirm = messagebox.askyesno("Confirm Buy",
            f"ยืนยันซื้อ {symbol} ด้วยจำนวน ฿{amount_thb:,.2f} THB?")
        if not confirm:
            return

        self._log(f"📈 กำลังซื้อ {symbol} จำนวน ฿{amount_thb:,.2f}...", "TRADE")
        threading.Thread(target=self._do_quick_buy,
                         args=(symbol, amount_thb), daemon=True).start()

    def _do_quick_buy(self, symbol: str, amount_thb: float):
        """Execute quick buy in background thread."""
        try:
            # Get current price before order
            ticker = self.client.get_ticker(symbol)
            est_price = ticker.get("last", 0) if ticker else 0

            order = self.client.create_buy_order(symbol, amount_thb)

            # Check for API error
            if "_error" in order:
                err_code = order["_error"]
                raw = order.get("_raw", {})
                error_map = {
                    2: "Missing API key",
                    3: "Invalid API key",
                    5: "IP not allowed",
                    6: "Invalid signature",
                    8: "Invalid signature",
                    10: "Invalid parameter — ตรวจสอบค่าที่ส่ง",
                    11: "Invalid symbol",
                    12: "Invalid amount — ขั้นต่ำ 10 THB",
                    13: "Invalid rate",
                    14: "Improper rate — ราคาไม่เหมาะสม",
                    15: "Amount too low",
                    16: "Not enough balance — ยอด THB ไม่พอ",
                    17: "Wallet is empty",
                    18: "Insufficient balance — ยอดไม่พอ",
                    19: "Failed to create order — ลองใหม่อีกครั้ง",
                    21: "Invalid order type",
                    30: "Limit exceeded",
                }
                err_msg = error_map.get(err_code, f"Error code: {err_code}")
                self.root.after(0, self._log,
                    f"❌ ซื้อไม่สำเร็จ: {err_msg} | Raw: {raw}", "ERROR")
                return

            if not order:
                self.root.after(0, self._log,
                    "❌ ซื้อไม่สำเร็จ - ไม่มีการตอบกลับจาก API", "ERROR")
                return

            exec_price = order.get("rate", est_price)
            crypto_amount = order.get("amount", amount_thb / exec_price if exec_price else 0)

            # If rate is 0 (market order), use estimated price
            if exec_price <= 0:
                exec_price = est_price
            if crypto_amount <= 0:
                crypto_amount = amount_thb / exec_price if exec_price else 0

            # Apply SL/TP from settings
            try:
                self.config.trading.stop_loss_pct = float(self.sl_var.get())
                self.config.trading.take_profit_pct = float(self.tp_var.get())
            except ValueError:
                pass

            # Register position for bot monitoring
            self.strategy.add_position(symbol, exec_price, crypto_amount)

            self.root.after(0, self._log,
                f"✅ ซื้อสำเร็จ! {symbol} @ {exec_price:,.2f} THB | "
                f"จำนวน: {crypto_amount:.8f} | "
                f"SL: {exec_price * (1 - self.config.trading.stop_loss_pct/100):,.2f} | "
                f"TP: {exec_price * (1 + self.config.trading.take_profit_pct/100):,.2f}",
                "SUCCESS")
            self.root.after(0, self._add_history_entry, "BUY", exec_price, 0)
            self.root.after(0, self._load_wallet)
            self.root.after(0, self._update_pnl_display, exec_price)

            # Auto-start bot if not running
            if not self.bot_running:
                self.root.after(500, self._auto_start_bot_after_buy)

        except Exception as e:
            self.root.after(0, self._log, f"❌ Buy error: {e}", "ERROR")

    def _auto_start_bot_after_buy(self):
        """Auto-start bot monitoring after manual buy."""
        if self.bot_running:
            return
        self._log("🤖 เริ่ม Bot อัตโนมัติหลังซื้อเหรียญ...", "INFO")
        self._start_bot()

    def _quick_sell(self):
        """Manual sell: sell all holdings of current coin."""
        if not self.is_connected or not self.client:
            messagebox.showwarning("Not Connected", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return

        symbol = self.config.trading.symbol
        coin = symbol.split("_")[0]
        coin_info = self.wallet_balances.get(coin, {})
        coin_avail = coin_info.get("available", 0)

        if coin_avail <= 0:
            messagebox.showinfo("No Holdings", f"ไม่มี {coin} ในกระเป๋า")
            return

        ticker = self.client.get_ticker(symbol)
        current_price = ticker.get("last", 0) if ticker else 0
        est_value = coin_avail * current_price

        confirm = messagebox.askyesno("Confirm Sell",
            f"ยืนยันขาย {coin} ทั้งหมด?\n"
            f"จำนวน: {coin_avail:.8f} {coin}\n"
            f"ราคาปัจจุบัน: ฿{current_price:,.2f}\n"
            f"มูลค่าประมาณ: ฿{est_value:,.2f}")
        if not confirm:
            return

        self._log(f"📉 กำลังขาย {coin} ทั้งหมด...", "TRADE")
        threading.Thread(target=self._do_quick_sell,
                         args=(symbol, coin_avail, current_price), daemon=True).start()

    def _do_quick_sell(self, symbol: str, amount: float, current_price: float):
        """Execute quick sell in background thread."""
        try:
            estimated_value = amount * current_price
            if estimated_value < 10:
                self.root.after(
                    0,
                    self._log,
                    f"⚠️ ขายไม่ได้เพราะมูลค่ารวมต่ำกว่า 10 THB ({estimated_value:,.2f} THB)",
                    "WARN",
                )
                return

            order = self.client.create_sell_order(symbol, amount)

            # Check for API error
            if "_error" in order:
                err_code = order["_error"]
                raw = order.get("_raw", {})
                self.root.after(0, self._log,
                    f"❌ ขายไม่สำเร็จ: error {err_code} | {raw}", "ERROR")
                return

            exit_price = current_price
            if order:
                exit_price = order.get("rate", current_price)
                if exit_price <= 0:
                    exit_price = current_price

            # Close all open positions for this symbol
            positions = self.strategy.get_open_positions(symbol)
            for pos in list(positions):
                record = self.strategy.close_position(pos, exit_price, "MANUAL_SELL")
                if record:
                    self.risk_manager.record_trade_result(record["profit_thb"])
                    # Track bot performance
                    self.bot_total_realized_pnl += record["profit_thb"]
                    self.bot_total_trades += 1
                    if record["profit_thb"] >= 0:
                        self.bot_win_trades += 1
                    else:
                        self.bot_lose_trades += 1
                    pnl_pct = record["profit_pct"]
                    self.root.after(0, self._add_history_entry, "SELL", exit_price, pnl_pct)
                    self.root.after(0, self._log,
                        f"✅ ขายสำเร็จ! @ {exit_price:,.2f} THB | P/L: {pnl_pct:+.2f}%",
                        "SUCCESS")

            if not positions:
                self.root.after(0, self._add_history_entry, "SELL", exit_price, 0)
                self.root.after(0, self._log,
                    f"✅ ขายสำเร็จ! @ {exit_price:,.2f} THB", "SUCCESS")

            self.root.after(0, self._load_wallet)
            self.root.after(0, self._update_pnl_display, exit_price)

        except Exception as e:
            self.root.after(0, self._log, f"❌ Sell error: {e}", "ERROR")

    def _toggle_bot(self):
        """Start or stop the trading bot."""
        if not self.is_connected or not self.client:
            messagebox.showwarning("Not Connected", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return

        if self.bot_running:
            self._stop_bot()
        else:
            self._start_bot()

    def _start_bot(self):
        """Start the auto trading loop."""
        if self._bot_start_in_progress:
            return

        runtime_settings = self._sync_runtime_settings(show_popup=True)
        if runtime_settings is None:
            return

        auto_buy_amount = runtime_settings["auto_buy_amount"]
        if auto_buy_amount < 10:
            messagebox.showerror("Settings Error", "Auto Buy Amount ต้องไม่น้อยกว่า 10 THB")
            return

        self._bot_start_in_progress = True
        self.start_btn.config(state=tk.DISABLED, text="⏳ STARTING...", bg=COLORS["yellow"])
        self._log("กำลังเตรียมข้อมูลและโหลดโมเดลก่อนเริ่ม Bot...", "INFO")
        threading.Thread(target=self._do_start_bot, args=(runtime_settings,), daemon=True).start()

    def _do_start_bot(self, runtime_settings: Dict[str, object]):
        """Run startup checks and model loading in a background thread."""
        try:
            wallet_snapshot = self._fetch_wallet_snapshot(self.client)
            symbol = self.config.trading.symbol
            coin = symbol.split("_")[0]
            thb_avail = wallet_snapshot.get("thb_avail", 0)
            coin_info = wallet_snapshot.get("wallet_balances", {}).get(coin, {})
            coin_total = coin_info.get("total", 0)

            if thb_avail < 10 and not coin_total:
                self.root.after(
                    0,
                    self._abort_bot_start,
                    "No Balance",
                    f"ไม่มียอดเงิน THB หรือเหรียญ {coin} ในบัญชี\nTHB Available: ฿{thb_avail:,.2f}",
                    "warning",
                )
                return

            ticker = self.client.get_ticker(symbol)
            current_price = ticker.get("last", 0) if ticker else 0
            model_messages = []
            try:
                self.lstm_predictor.load_model()
                model_messages.append(("AI", "LSTM model loaded"))
            except Exception:
                model_messages.append(("WARN", "LSTM model not found (using signals only)"))

            try:
                self.rl_agent.load_model()
                model_messages.append(("AI", "RL model loaded"))
            except Exception:
                model_messages.append(("WARN", "RL model not found"))

            startup_context = {
                "wallet_snapshot": wallet_snapshot,
                "coin": coin,
                "coin_total": coin_total,
                "thb_avail": thb_avail,
                "initial_portfolio_value": thb_avail + (coin_total * current_price),
                "model_messages": model_messages,
                "auto_buy_amount": runtime_settings["auto_buy_amount"],
            }
            self.root.after(0, self._finish_start_bot, startup_context)
        except Exception as e:
            self.root.after(0, self._abort_bot_start, "Start Error", str(e), "error")

    def _finish_start_bot(self, startup_context: Dict[str, object]):
        """Finalize bot startup on the Tkinter thread."""
        self._bot_start_in_progress = False
        self._sync_selected_position_from_wallet(startup_context["wallet_snapshot"], log_sync=True)
        self._apply_wallet_snapshot(startup_context["wallet_snapshot"], log_success=False)

        self.initial_portfolio_value = startup_context["initial_portfolio_value"]
        self.bot_running = True
        self.bot_start_time = datetime.now()
        self.bot_status.set("🟢 Running")
        self.status_label.config(fg=COLORS["green"])
        self.start_btn.config(state=tk.NORMAL, text="⏹ STOP BOT", bg=COLORS["red"])
        self.bot_alive_label.config(text="🟢 RUNNING", fg=COLORS["green"])
        self._start_uptime_timer()
        self._log("Bot STARTED", "SUCCESS")
        self._log(f"  เหรียญ: {self.config.trading.symbol}", "INFO")
        self._log(
            f"  THB Available: ฿{startup_context['thb_avail']:,.2f} | {startup_context['coin']}: {startup_context['coin_total']:.8g}",
            "INFO",
        )
        self._log(f"  Interval: {self.config.trading.trading_interval_seconds}s", "INFO")
        self._log(
            f"  SL: {self.config.trading.stop_loss_pct}% | TP: {self.config.trading.take_profit_pct}%",
            "INFO",
        )
        self._log(
            f"  AI Scale-In: {'ON' if self.config.trading.ai_scale_in_enabled else 'OFF'} @ -{self.config.trading.ai_scale_in_loss_pct}% | "
            f"AI Take Profit: {'ON' if self.config.trading.ai_take_profit_enabled else 'OFF'} @ +{self.config.trading.ai_take_profit_min_profit_pct}%",
            "INFO",
        )
        self._log(f"  Auto Buy Amount: ฿{startup_context['auto_buy_amount']:,.2f}", "INFO")

        for level, message in startup_context.get("model_messages", []):
            self._log(message, level)

        self.bot_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.bot_thread.start()

    def _abort_bot_start(self, title: str, message: str, dialog_type: str):
        """Reset UI state when bot startup fails."""
        self._bot_start_in_progress = False
        self.start_btn.config(state=tk.NORMAL, text="▶ START BOT", bg=COLORS["green"])
        self._log(f"Bot start error: {message}", "ERROR" if dialog_type == "error" else "WARN")
        if dialog_type == "warning":
            messagebox.showwarning(title, message)
        else:
            messagebox.showerror(title, message)

    def _sync_runtime_settings(self, show_popup: bool = False) -> Optional[Dict[str, object]]:
        """Read current GUI parameters and apply them immediately to runtime config."""
        try:
            interval = max(1, int(self.interval_var.get()))
            stop_loss_pct = float(self.sl_var.get())
            take_profit_pct = float(self.tp_var.get())
            ai_scale_in_loss_pct = float(self.ai_scale_in_loss_pct_var.get())
            ai_take_profit_pct = float(self.ai_take_profit_pct_var.get())
            auto_buy_amount = float(self.auto_trade_amount_var.get())
        except ValueError:
            if show_popup:
                messagebox.showerror(
                    "Settings Error",
                    "ค่า Interval, SL, TP, AI Loss, AI Profit หรือ Auto Buy ไม่ถูกต้อง",
                )
            elif not self._runtime_settings_invalid:
                self.root.after(
                    0,
                    self._log,
                    "พารามิเตอร์บางช่องไม่ถูกต้อง ระบบจะใช้ค่าล่าสุดที่ถูกต้องต่อไป",
                    "WARN",
                )
            self._runtime_settings_invalid = True
            return None

        self._runtime_settings_invalid = False

        desired_symbol = self.symbol_var.get()
        current_symbol = self.config.trading.symbol
        positions = self.strategy.get_open_positions()
        pending_reentry = self.reentry_waiting or self.boss_waiting_recovery
        can_switch_symbol = not self.bot_running or (not positions and not pending_reentry)
        applied_symbol = current_symbol

        if desired_symbol != current_symbol:
            if can_switch_symbol:
                applied_symbol = desired_symbol
            elif self.symbol_var.get() != current_symbol:
                self.root.after(0, lambda: self.symbol_var.set(current_symbol))
                self.root.after(
                    0,
                    self._log,
                    f"ยังเปลี่ยนเหรียญเป็น {desired_symbol} ไม่ได้จนกว่าจะปิด position เดิมก่อน",
                    "WARN",
                )

        self.config.trading.trading_interval_seconds = interval
        self.config.trading.stop_loss_pct = stop_loss_pct
        self.config.trading.take_profit_pct = take_profit_pct
        self.config.trading.ai_scale_in_enabled = self.ai_scale_in_enabled.get()
        self.config.trading.ai_scale_in_loss_pct = ai_scale_in_loss_pct
        self.config.trading.ai_take_profit_enabled = self.ai_take_profit_enabled.get()
        self.config.trading.ai_take_profit_min_profit_pct = ai_take_profit_pct
        self.config.trading.symbol = applied_symbol

        snapshot = {
            "symbol": applied_symbol,
            "interval": interval,
            "sl": stop_loss_pct,
            "tp": take_profit_pct,
            "auto_buy": auto_buy_amount,
            "ai_scale_in_enabled": self.config.trading.ai_scale_in_enabled,
            "ai_scale_in_loss_pct": ai_scale_in_loss_pct,
            "ai_take_profit_enabled": self.config.trading.ai_take_profit_enabled,
            "ai_take_profit_pct": ai_take_profit_pct,
            "boss_mode": self.boss_mode.get(),
            "boss_cutloss": float(self.boss_cutloss_pct_var.get()),
            "boss_recovery": float(self.boss_recovery_pct_var.get()),
            "reentry_enabled": self.auto_reentry_enabled.get(),
            "reentry_rise": float(self.reentry_rise_pct_var.get()),
            "reentry_delay": float(self.reentry_delay_pct_var.get()),
        }

        if self.bot_running and self._runtime_settings_snapshot != snapshot:
            self.root.after(
                0,
                self._log,
                f"อัปเดตพารามิเตอร์สด | Interval {interval}s | SL {stop_loss_pct}% | TP {take_profit_pct}% | Auto Buy ฿{auto_buy_amount:,.2f}",
                "INFO",
            )

        self._runtime_settings_snapshot = snapshot
        self.root.after(0, self._refresh_runtime_badges)
        return {
            "interval": interval,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "auto_buy_amount": auto_buy_amount,
            "symbol": applied_symbol,
        }

    def _stop_bot(self):
        """Stop the auto trading loop."""
        self.bot_running = False
        self.bot_status.set("⏹ Stopped")
        self.status_label.config(fg=COLORS["yellow"])
        self.start_btn.config(text="▶ START BOT", bg=COLORS["green"])
        self.bot_alive_label.config(text="⏹ STOPPED", fg=COLORS["red"])
        self._log("Bot STOPPED", "WARN")
        self._refresh_runtime_badges()

    def _trading_loop(self):
        """Background trading loop (runs in thread)."""
        while self.bot_running:
            try:
                self._sync_runtime_settings()
                self._execute_trading_cycle()
            except Exception as e:
                self.root.after(0, self._log, f"Cycle error: {e}", "ERROR")

            waited_seconds = 0
            while self.bot_running:
                self._sync_runtime_settings()
                interval = max(1, int(self.config.trading.trading_interval_seconds))
                if waited_seconds >= interval:
                    break
                time.sleep(1)
                waited_seconds += 1
                if not self.bot_running:
                    break

    def _execute_trading_cycle(self):
        """Execute one trading cycle."""
        if not self.data_collector or not self.client:
            return

        self.cycle_count += 1
        symbol = self.config.trading.symbol

        # Step 1: Fetch data
        df = self.data_collector.fetch_ohlcv(symbol)
        if df is None or df.empty:
            self.root.after(0, self._log, "No market data available", "WARN")
            return

        # Step 2: Calculate indicators
        df_ind = self.indicator_engine.add_all_indicators(df)
        signals = self.indicator_engine.get_signal_summary(df_ind)

        current_price = signals.get("price", 0)
        if current_price <= 0:
            return

        # Step 3: AI prediction
        ai_prediction = self.lstm_predictor.predict(df_ind)

        # Step 4: Update GUI
        self.root.after(0, self._update_gui_data, signals, ai_prediction)

        # Step 5: Get REAL balance from exchange
        balance_detail = self.client.get_balance_detail()
        self.wallet_balances = balance_detail
        thb_info = balance_detail.get("THB", {})
        balance = thb_info.get("available", 0.0)
        positions = self.strategy.get_open_positions(symbol)

        # Refresh wallet display every 5 cycles
        if self.cycle_count % 5 == 0:
            self.root.after(0, self._load_wallet)

        # Step 6: Boss Mode or normal SL/TP
        if self.boss_mode.get():
            action = self._check_boss_mode(symbol, current_price, balance, signals, df_ind, ai_prediction)
        else:
            action = self._check_sl_tp(symbol, current_price, balance, signals, df_ind, ai_prediction)

        if not action and positions:
            action = self._check_ai_scale_in(
                symbol, current_price, balance, positions, signals, ai_prediction, df_ind
            )

        # Step 7: If we recently sold this symbol, monitor for auto re-entry
        if not action:
            action = self._check_auto_reentry(
                symbol, current_price, balance, positions, signals, ai_prediction
            )

        # Step 8: If no SL/TP, re-entry, or Boss action, check signals
        if not action:
            risk_check = self.risk_manager.can_trade(balance, positions)
            if risk_check["allowed"]:
                action = self._check_signals(
                    symbol, signals, ai_prediction, balance, current_price
                )

        action_name = action or "HOLD"
        self.root.after(0, self._log,
                        f"Cycle #{self.cycle_count} | {symbol} @ {current_price:,.2f} | "
                        f"THB: ฿{balance:,.2f} | {action_name}",
                        "TRADE" if action else "INFO")

        # Update last action
        self.root.after(0, lambda a=action_name: self.bot_last_action.set(a))
        self.root.after(0, lambda: self.bot_cycles_str.set(str(self.cycle_count)))

        # Update positions, balance, P/L, and risk
        self.root.after(0, self._update_positions_display, current_price)
        self.root.after(0, self._update_balance_display, balance, current_price)
        self.root.after(0, self._update_pnl_display, current_price)
        self.root.after(0, self._update_risk_display, balance)
        self.root.after(0, self._update_bot_performance, current_price)

    def _on_boss_toggle(self, *args):
        """Handle boss mode toggle."""
        if self.boss_mode.get():
            self.boss_status_label.config(text="ON", fg=COLORS["green"])
            self._log("🏆 Boss Mode: ON | AI CutLoss + Buy-Back เมื่อราคาฟื้น", "SUCCESS")
        else:
            self.boss_status_label.config(text="OFF", fg=COLORS["text_dim"])
            self._clear_boss_recovery_state()
            self._log("🏆 Boss Mode: OFF", "WARN")

    def _build_rl_live_decision(self, df_ind, position: Position) -> Dict:
        """Build the RL live decision for the current position."""
        try:
            state = self.rl_agent.build_live_state(
                df_ind, has_position=True, entry_price=position.entry_price
            )
            if state is None:
                return {}
            return self.rl_agent.decide(state)
        except Exception as e:
            self.root.after(0, self._log, f"RL live decision error: {e}", "ERROR")
            return {}

    def _get_auto_buy_amount(self) -> float:
        """Get the configured auto-buy amount from the GUI."""
        try:
            return float(self.auto_trade_amount_var.get())
        except ValueError:
            return 0.0

    def _update_recovery_tracker(self, exit_price: float, current_price: float,
                                 tracked_low_price: float, recovery_pct: float) -> Dict:
        """Track the lowest post-exit price and build a flexible re-buy trigger."""
        low_price = tracked_low_price if tracked_low_price > 0 else exit_price
        if low_price <= 0:
            low_price = current_price

        low_updated = False
        if current_price > 0 and (low_price <= 0 or current_price < low_price):
            low_price = current_price
            low_updated = True

        trigger_price = low_price * (1 + max(recovery_pct, 0.0) / 100) if low_price > 0 else 0.0
        recovery_from_low_pct = (
            (current_price - low_price) / low_price * 100 if low_price > 0 else 0.0
        )
        change_from_exit_pct = (
            (current_price - exit_price) / exit_price * 100 if exit_price > 0 else 0.0
        )

        return {
            "low_price": low_price,
            "low_updated": low_updated,
            "trigger_price": trigger_price,
            "recovery_from_low_pct": recovery_from_low_pct,
            "change_from_exit_pct": change_from_exit_pct,
        }

    def _clear_boss_recovery_state(self):
        """Reset pending boss buy-back recovery tracking."""
        self.boss_waiting_recovery = False
        self.boss_last_sell_price = 0.0
        self.boss_recovery_low_price = 0.0
        self.boss_rebuy_budget_thb = 0.0

    def _arm_boss_recovery_state(self, exit_price: float, budget_thb: float):
        """Store recovery tracking state after a boss cutloss sell."""
        self.boss_last_sell_price = exit_price
        self.boss_recovery_low_price = exit_price
        self.boss_rebuy_budget_thb = max(budget_thb, 0.0)
        self.boss_waiting_recovery = True

    def _check_ai_scale_in(self, symbol: str, current_price: float, balance: float,
                           positions, signals: Dict, ai_prediction: Dict, df_ind) -> str:
        """Use AI to average down a losing position when rebound odds improve."""
        if not positions or not self.config.trading.ai_scale_in_enabled:
            return ""

        risk_check = self.risk_manager.can_trade(balance, positions)
        if not risk_check["allowed"]:
            return ""

        representative_position = min(
            positions,
            key=lambda pos: (current_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0,
        )
        rl_decision = self._build_rl_live_decision(df_ind, representative_position)
        scale_in = self.strategy.evaluate_ai_scale_in(
            positions,
            current_price,
            signals,
            ai_prediction,
            rl_decision,
        )
        if not scale_in["should_buy"]:
            return ""

        configured_amount = self._get_auto_buy_amount()
        if configured_amount < 10:
            self.root.after(
                0,
                self._log,
                f"⚠️ AI Scale-In ข้ามรอบ เพราะ Auto Buy Amount ไม่ถูกต้อง ({configured_amount:,.2f})",
                "WARN",
            )
            return "INVALID_SCALE_IN"

        sizing = self.risk_manager.calculate_position_size(
            balance, current_price, scale_in["signal_strength"]
        )
        buy_amount = min(configured_amount, sizing["position_size_thb"], balance * 0.95)
        if buy_amount < 10:
            return ""

        self._do_buy(symbol, buy_amount, current_price)
        reasons = "; ".join(scale_in["reasons"])
        self.root.after(
            0,
            self._log,
            f"🪜 AI SCALE-IN @ {current_price:,.2f} | Avg loss: -{scale_in['avg_loss_pct']:.2f}% | ซื้อเพิ่ม ฿{buy_amount:,.0f}",
            "TRADE",
        )
        self.root.after(0, self._log, f"  เหตุผล: {reasons}", "TRADE")
        return "AI_SCALE_IN"

    def _check_boss_mode(self, symbol: str, current_price: float, balance: float,
                         signals: Dict, df_ind, ai_prediction: Dict) -> str:
        """Boss Mode: AI confirms cutloss, sells to THB, then buys back on price recovery."""
        try:
            base_cutloss_pct = float(self.boss_cutloss_pct_var.get())
            base_recovery_pct = float(self.boss_recovery_pct_var.get())
        except ValueError:
            base_cutloss_pct = 0.5
            base_recovery_pct = 0.5

        adaptive_profile = self.strategy.get_adaptive_risk_profile(
            signals,
            ai_prediction,
            base_cutloss_pct=base_cutloss_pct,
            base_recovery_pct=base_recovery_pct,
        )
        cutloss_pct = adaptive_profile["cutloss_pct"]
        recovery_pct = adaptive_profile["recovery_pct"]
        rebuy_allocation_pct = adaptive_profile["rebuy_allocation_pct"]

        # Wait for price recovery after a boss cutloss before buying back.
        if self.boss_waiting_recovery and self.boss_last_sell_price > 0:
            exit_price = self.boss_last_sell_price
            tracker = self._update_recovery_tracker(
                exit_price,
                current_price,
                self.boss_recovery_low_price,
                recovery_pct,
            )
            self.boss_recovery_low_price = tracker["low_price"]

            if tracker["recovery_from_low_pct"] >= recovery_pct:
                # Price recovered enough above the cutloss exit -> buy back.
                target_budget = self.boss_rebuy_budget_thb or balance
                buy_amount = min(
                    balance * min(rebuy_allocation_pct / 100, 0.95),
                    target_budget * (rebuy_allocation_pct / 100),
                )
                if buy_amount >= 10:
                    tracked_low = self.boss_recovery_low_price
                    self._do_buy(symbol, buy_amount, current_price)
                    self._clear_boss_recovery_state()
                    self.root.after(0, self._log,
                        f"🏆 BOSS BUY-BACK @ {current_price:,.2f} | "
                        f"ราคาฟื้น {tracker['recovery_from_low_pct']:+.2f}% จาก low {tracked_low:,.2f} "
                        f"(ขายเดิม {exit_price:,.2f}) | "
                        f"ซื้อคืน ฿{buy_amount:,.0f} | AI Recovery {recovery_pct:.2f}% | Allocation {rebuy_allocation_pct:.0f}%",
                        "TRADE")
                    return "BOSS_BUYBACK"
                else:
                    self.root.after(0, self._log,
                        f"🏆 Boss ต้องการซื้อคืนแต่ THB ไม่พอ (฿{balance:,.2f})", "WARN")
            else:
                if self.cycle_count % 3 == 0:  # log every 3 cycles to avoid spam
                    self.root.after(0, self._log,
                        f"🏆 Boss รอราคาฟื้นเพื่อซื้อคืน | ราคา: {current_price:,.2f} | "
                        f"จุดขาย: {exit_price:,.2f} | low หลังขาย: {self.boss_recovery_low_price:,.2f} | "
                        f"ฟื้นจาก low: {tracker['recovery_from_low_pct']:+.2f}% "
                        f"(AI เป้าฟื้น: +{recovery_pct:.2f}% | trigger {tracker['trigger_price']:,.2f})",
                        "INFO")
            return ""

        # ── Check positions for cutloss ──
        positions = self.strategy.get_open_positions(symbol)
        scale_in_checked = False
        for pos in list(positions):
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            pnl_thb = (current_price - pos.entry_price) * pos.amount

            if self.strategy.check_stop_loss(pos, current_price):
                record = self._do_sell(pos, current_price, "STOP_LOSS_TO_THB")
                if record:
                    self._arm_boss_recovery_state(current_price, record["amount"] * current_price)
                    self.root.after(0, self._log,
                        f"🔴 BOSS BACKUP STOP LOSS @ {current_price:,.2f} | "
                        f"ขายกลับเป็น THB | ขาดทุน: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                        f"ชน SL {self.config.trading.stop_loss_pct:.4f}% ก่อน AI ยืนยันขาย",
                        "TRADE")
                    return "STOP_LOSS"

            if pnl_pct <= -cutloss_pct:
                rl_decision = self._build_rl_live_decision(df_ind, pos)
                adaptive_profile = self.strategy.get_adaptive_risk_profile(
                    signals,
                    ai_prediction,
                    pos,
                    rl_decision,
                    base_cutloss_pct=base_cutloss_pct,
                    base_recovery_pct=base_recovery_pct,
                )
                cutloss_pct = adaptive_profile["cutloss_pct"]
                recovery_pct = adaptive_profile["recovery_pct"]
                rebuy_allocation_pct = adaptive_profile["rebuy_allocation_pct"]
                hard_limit = adaptive_profile["hard_limit_pct"]
                if not scale_in_checked and abs(pnl_pct) < hard_limit:
                    scale_in_checked = True
                    scale_in_action = self._check_ai_scale_in(
                        symbol, current_price, balance, positions, signals, ai_prediction, df_ind
                    )
                    if scale_in_action:
                        return scale_in_action

                ai_cutloss = self.strategy.evaluate_ai_cut_loss(
                    pos,
                    current_price,
                    ai_prediction,
                    rl_decision,
                    min_loss_pct=cutloss_pct,
                    hard_limit_pct=hard_limit,
                )
                if ai_cutloss["should_sell"]:
                    record = self._do_sell(pos, current_price, ai_cutloss["reason"])
                    if record:
                        self._arm_boss_recovery_state(current_price, record["amount"] * current_price)
                        self.root.after(0, self._log,
                            f"🏆 AI BOSS CUTLOSS @ {current_price:,.2f} | "
                            f"ขายกลับเป็น THB | ขาดทุน: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                            f"AI CutLoss {cutloss_pct:.2f}% | Hard Limit {hard_limit:.2f}% | "
                            f"รอราคาฟื้น +{recovery_pct:.2f}% จากจุดขายเพื่อซื้อคืน {rebuy_allocation_pct:.0f}% ของงบเดิม",
                            "TRADE")
                        return "AI_BOSS_CUTLOSS"
                elif self.cycle_count % 3 == 0:
                    self.root.after(0, self._log,
                        f"🏆 Boss รอ AI ยืนยัน CutLoss | ขาดทุน {pnl_pct:+.2f}% | "
                        f"LSTM: {ai_prediction.get('direction', 'unknown')} "
                        f"({ai_prediction.get('confidence', 0):.2f})",
                        "INFO")

        # Also check normal TP (boss still takes profit)
        for pos in list(positions):
            rl_decision = self._build_rl_live_decision(df_ind, pos)
            ai_take_profit = self.strategy.evaluate_ai_take_profit(
                pos,
                current_price,
                signals,
                ai_prediction,
                rl_decision,
            )
            if ai_take_profit["should_sell"]:
                record = self._do_sell(pos, current_price, ai_take_profit["reason"])
                if record:
                    self.root.after(0, self._log,
                        f"🤖 AI TAKE PROFIT @ {current_price:,.2f} | "
                        f"ขายกลับเป็น THB | กำไร: {ai_take_profit['profit_thb']:,.2f} THB ({ai_take_profit['profit_pct']:+.2f}%)",
                        "TRADE")
                    return "AI_TAKE_PROFIT"

            if self.strategy.check_take_profit(pos, current_price):
                pnl_thb = (current_price - pos.entry_price) * pos.amount
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                record = self._do_sell(pos, current_price, "TAKE_PROFIT_TO_THB")
                if record:
                    self.root.after(0, self._log,
                        f"🟢 TAKE PROFIT @ {current_price:,.2f} | "
                        f"ขายกลับเป็น THB | กำไร: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%)",
                        "TRADE")
                    return "TAKE_PROFIT"

        return ""

    def _check_sl_tp(self, symbol: str, current_price: float, balance: float, signals: Dict, df_ind,
                     ai_prediction: Dict) -> str:
        """Check AI cut loss / stop loss / take profit for open positions."""
        scale_in_checked = False
        for pos in list(self.strategy.get_open_positions(symbol)):
            pnl_thb = (current_price - pos.entry_price) * pos.amount
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            rl_decision = self._build_rl_live_decision(df_ind, pos)
            adaptive_profile = self.strategy.get_adaptive_risk_profile(
                signals,
                ai_prediction,
                pos,
                rl_decision,
            )

            ai_take_profit = self.strategy.evaluate_ai_take_profit(
                pos,
                current_price,
                signals,
                ai_prediction,
                rl_decision,
            )
            if ai_take_profit["should_sell"]:
                record = self._do_sell(pos, current_price, ai_take_profit["reason"])
                if record:
                    self.root.after(0, self._log,
                                    f"🤖 AI TAKE PROFIT @ {current_price:,.2f} | "
                                    f"ขายกลับเป็น THB | กำไร: {ai_take_profit['profit_thb']:,.2f} THB ({ai_take_profit['profit_pct']:+.2f}%)",
                                    "TRADE")
                    return "AI_TAKE_PROFIT"

            if pnl_pct < 0 and not scale_in_checked and abs(pnl_pct) < self.config.trading.ai_cutloss_hard_limit_pct:
                scale_in_checked = True
                scale_in_action = self._check_ai_scale_in(
                    symbol, current_price, balance=balance, positions=self.strategy.get_open_positions(symbol),
                    signals=signals, ai_prediction=ai_prediction, df_ind=df_ind
                )
                if scale_in_action:
                    return scale_in_action

            ai_cutloss = self.strategy.evaluate_ai_cut_loss(
                pos,
                current_price,
                ai_prediction,
                rl_decision,
                min_loss_pct=adaptive_profile["cutloss_pct"],
                hard_limit_pct=adaptive_profile["hard_limit_pct"],
            )
            if ai_cutloss["should_sell"]:
                record = self._do_sell(pos, current_price, ai_cutloss["reason"])
                if record:
                    self.root.after(0, self._log,
                                    f"🤖 AI CUTLOSS @ {current_price:,.2f} | "
                                    f"ขายกลับเป็น THB | ขาดทุน: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                                    f"AI CutLoss {adaptive_profile['cutloss_pct']:.2f}% / Hard {adaptive_profile['hard_limit_pct']:.2f}%",
                                    "TRADE")
                    return "AI_CUTLOSS"

            if self.strategy.check_stop_loss(pos, current_price):
                record = self._do_sell(pos, current_price, "STOP_LOSS_TO_THB")
                if record:
                    self.root.after(0, self._log,
                                    f"🔴 STOP LOSS @ {current_price:,.2f} | "
                                    f"ขายกลับเป็น THB | ขาดทุน: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                                    f"ตัดขาดทุนอัตโนมัติ!", "TRADE")
                    return "STOP_LOSS"

            if self.strategy.check_take_profit(pos, current_price):
                record = self._do_sell(pos, current_price, "TAKE_PROFIT_TO_THB")
                if record:
                    self.root.after(0, self._log,
                                    f"🟢 TAKE PROFIT @ {current_price:,.2f} | "
                                    f"ขายกลับเป็น THB | กำไร: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                                    f"ทำกำไรอัตโนมัติ!", "TRADE")
                    return "TAKE_PROFIT"
        return ""

    def _check_auto_reentry(self, symbol: str, current_price: float,
                            balance: float, positions, signals: Dict,
                            ai_prediction: Dict) -> str:
        """Re-buy the same coin after an auto-sell only when price recovers.

        Rules:
        - If price falls below the last exit by Delay Down %, keep waiting.
        - If price rises above the last exit by Buy Up %, buy back automatically.
        - If we already hold the symbol, re-entry is disabled until the next sell.
        """
        if not self.auto_reentry_enabled.get():
            return ""
        if positions:
            return ""
        if not self.reentry_waiting or self.reentry_symbol != symbol:
            return ""
        if self.reentry_last_exit_price <= 0:
            return ""

        try:
            base_rise_pct = float(self.reentry_rise_pct_var.get())
            base_delay_pct = float(self.reentry_delay_pct_var.get())
        except ValueError:
            base_rise_pct = 0.5
            base_delay_pct = 1.0

        adaptive_profile = self.strategy.get_adaptive_risk_profile(
            signals,
            ai_prediction,
            base_recovery_pct=base_rise_pct,
        )
        rise_pct = adaptive_profile["recovery_pct"]
        delay_pct = max(base_delay_pct, adaptive_profile["delay_pct"])
        rebuy_allocation_pct = adaptive_profile["rebuy_allocation_pct"]

        tracker = self._update_recovery_tracker(
            self.reentry_last_exit_price,
            current_price,
            self.reentry_recovery_low_price,
            rise_pct,
        )
        self.reentry_recovery_low_price = tracker["low_price"]

        if tracker["recovery_from_low_pct"] >= rise_pct:
            buy_amount = min(
                balance * min(rebuy_allocation_pct / 100, 0.95),
                self.reentry_budget_thb * (rebuy_allocation_pct / 100),
            )
            if buy_amount >= 10:
                tracked_low = self.reentry_recovery_low_price
                exit_price = self.reentry_last_exit_price
                self._do_buy(symbol, buy_amount, current_price)
                self.reentry_waiting = False
                self.root.after(
                    0,
                    self._log,
                    f"🔁 AUTO RE-BUY {symbol} @ {current_price:,.2f} | "
                    f"ราคาฟื้น {tracker['recovery_from_low_pct']:+.2f}% จาก low {tracked_low:,.2f} "
                    f"(ขายเดิม {exit_price:,.2f}) | "
                    f"ซื้อกลับ ฿{buy_amount:,.0f} | AI Recovery {rise_pct:.2f}% | Allocation {rebuy_allocation_pct:.0f}%",
                    "TRADE",
                )
                return "AUTO_REBUY"

            self.root.after(
                0,
                self._log,
                f"🔁 ต้องการซื้อกลับ {symbol} แต่ THB ไม่พอ (มี ฿{balance:,.2f})",
                "WARN",
            )
            return "REENTRY_WAIT"

        if tracker["change_from_exit_pct"] <= -delay_pct:
            if self.cycle_count % 3 == 0:
                self.root.after(
                    0,
                    self._log,
                    f"🔁 ชะลอการซื้อ {symbol} | ราคา {tracker['change_from_exit_pct']:+.2f}% "
                    f"ต่ำกว่าจุดขาย {self.reentry_last_exit_price:,.2f} | "
                    f"low หลังขาย {self.reentry_recovery_low_price:,.2f} | "
                    f"ฟื้นจาก low {tracker['recovery_from_low_pct']:+.2f}% "
                    f"(เกณฑ์ชะลอ {delay_pct:.2f}% | trigger {tracker['trigger_price']:,.2f})",
                    "WARN",
                )
            return "DELAY_BUY"

        if self.cycle_count % 3 == 0:
            self.root.after(
                0,
                self._log,
                f"🔁 รอจังหวะ Re-Buy {symbol} | ขายเดิม {self.reentry_last_exit_price:,.2f} | "
                f"low หลังขาย {self.reentry_recovery_low_price:,.2f} | "
                f"ฟื้นจาก low {tracker['recovery_from_low_pct']:+.2f}% "
                f"(ต้องการ {rise_pct:.2f}% | trigger {tracker['trigger_price']:,.2f})",
                "INFO",
            )

        return "REENTRY_WAIT"

    def _arm_auto_reentry(self, symbol: str, exit_price: float,
                          exit_value_thb: float, reason: str):
        """Remember the last auto-sold symbol so the bot can re-buy it later."""
        if not self.auto_reentry_enabled.get():
            return
        self.reentry_waiting = True
        self.reentry_symbol = symbol
        self.reentry_last_exit_price = exit_price
        self.reentry_recovery_low_price = exit_price
        self.reentry_budget_thb = max(exit_value_thb, 0.0)
        self.reentry_last_reason = reason

    def _clear_auto_reentry(self, symbol: str = ""):
        """Clear pending auto re-entry state."""
        if symbol and self.reentry_symbol and self.reentry_symbol != symbol:
            return
        self.reentry_waiting = False
        self.reentry_symbol = ""
        self.reentry_last_exit_price = 0.0
        self.reentry_recovery_low_price = 0.0
        self.reentry_budget_thb = 0.0
        self.reentry_last_reason = ""

    def _check_signals(self, symbol, signals, ai_pred, balance, price) -> str:
        """Check buy/sell signals based on indicators and AI."""
        positions = self.strategy.get_open_positions(symbol)

        # SELL signals - always check if we have positions
        if positions:
            sold_any = False
            last_reasons = ""
            for pos in list(positions):
                sell = self.strategy.should_sell(signals, ai_pred, pos)
                if not sell["should_sell"]:
                    continue
                sold_any = True
                last_reasons = "; ".join(sell["reasons"])
                for_sell_pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                for_sell_pnl_thb = (price - pos.entry_price) * pos.amount
                self._do_sell(pos, price, "SELL_SIGNAL")
                self.root.after(0, self._log,
                    f"📉 SELL signal | P/L: {for_sell_pnl_thb:+,.2f} THB ({for_sell_pnl_pct:+.2f}%)",
                    "TRADE")
            if sold_any:
                self.root.after(0, self._log, f"  เหตุผล: {last_reasons}", "TRADE")
                return "SELL"

        # BUY signals
        buy = self.strategy.should_buy(signals, ai_pred)
        if buy["should_buy"]:
            # Allow buy if no position OR if adding to profitable position
            can_buy = not positions
            if positions:
                # Allow additional buy if existing position is in profit
                avg_pnl = sum(
                    (price - p.entry_price) / p.entry_price * 100
                    for p in positions
                ) / len(positions)
                if avg_pnl > 1.0:  # position is >1% in profit
                    can_buy = True

            if can_buy:
                try:
                    configured_amount = float(self.auto_trade_amount_var.get())
                except ValueError:
                    configured_amount = 0.0

                if configured_amount < 10:
                    self.root.after(
                        0,
                        self._log,
                        f"⚠️ Auto Buy Amount ไม่ถูกต้อง ({configured_amount:,.2f}) ต้องอย่างน้อย 10 THB",
                        "WARN",
                    )
                    return "INVALID_AUTO_BUY"

                if balance < configured_amount:
                    self.root.after(
                        0,
                        self._log,
                        f"⚠️ AI เจอสัญญาณ BUY แต่ THB ไม่พอสำหรับ Auto Buy ฿{configured_amount:,.2f} (มี ฿{balance:,.2f})",
                        "WARN",
                    )
                    return "INSUFFICIENT_AUTO_BUY"

                self._do_buy(symbol, configured_amount, price)
                reasons = "; ".join(buy["reasons"])
                self.root.after(
                    0,
                    self._log,
                    f"📈 AI BUY ฿{configured_amount:,.0f} | {reasons}",
                    "TRADE",
                )
                return "BUY"

        return ""

    def _do_buy(self, symbol: str, amount_thb: float, price: float):
        """Execute buy order."""
        if not self.client:
            return
        order = self.client.create_buy_order(symbol, amount_thb)
        if "_error" in order:
            self.root.after(0, self._log,
                f"❌ Auto-buy failed: error {order['_error']} | {order.get('_raw', '')}", "ERROR")
            return
        if order:
            crypto_amount = order.get("amount", amount_thb / price if price else 0)
            exec_price = order.get("rate", price)
            if exec_price <= 0:
                exec_price = price
            if crypto_amount <= 0:
                crypto_amount = amount_thb / exec_price if exec_price else 0
            self.strategy.add_position(symbol, exec_price, crypto_amount)
            self._clear_auto_reentry(symbol)
            self.root.after(0, self._add_history_entry,
                            "BUY", exec_price, 0)
            self.root.after(0, self._update_pnl_display, exec_price)

    def _do_sell(self, position: Position, price: float, reason: str):
        """Execute sell order back into THB."""
        if not self.client:
            return

        estimated_value = position.amount * price
        if estimated_value < 10:
            if position in self.strategy.positions:
                self.strategy.positions.remove(position)
            self.root.after(
                0,
                self._log,
                f"⚠️ ข้ามการขาย {position.symbol} เพราะมูลค่าคงเหลือเพียง {estimated_value:,.2f} THB ซึ่งต่ำกว่าขั้นต่ำ Bitkub; ตัดออกจากการติดตามของบอทแล้ว",
                "WARN",
            )
            return None

        order = self.client.create_sell_order(position.symbol, position.amount)
        if "_error" in order:
            self.root.after(0, self._log,
                f"❌ Auto-sell failed: error {order['_error']} | {order.get('_raw', '')}", "ERROR")
            return
        if not order:
            self.root.after(0, self._log,
                f"❌ Auto-sell failed: empty exchange response for {position.symbol}", "ERROR")
            return
        exit_price = price
        if order:
            exit_price = order.get("rate", price)
            if exit_price <= 0:
                exit_price = price

        record = self.strategy.close_position(position, exit_price, reason)
        if record:
            self.risk_manager.record_trade_result(record["profit_thb"])
            if self.bot_running and reason != "MANUAL_SELL":
                self._arm_auto_reentry(
                    position.symbol,
                    exit_price,
                    exit_price * position.amount,
                    reason,
                )
            # Track bot performance
            self.bot_total_realized_pnl += record["profit_thb"]
            self.bot_total_trades += 1
            if record["profit_thb"] >= 0:
                self.bot_win_trades += 1
            else:
                self.bot_lose_trades += 1
            self.root.after(0, self._add_history_entry,
                            "SELL", exit_price, record["profit_pct"])
            self.root.after(0, self._update_pnl_display, exit_price)
            return record

    # ─── GUI Updates ──────────────────────────────────────────

    def _start_uptime_timer(self):
        """Start a 1-second uptime counter on the GUI."""
        if not self.bot_running or not self.bot_start_time:
            return
        elapsed = datetime.now() - self.bot_start_time
        total_seconds = int(elapsed.total_seconds())
        h, remainder = divmod(total_seconds, 3600)
        m, s = divmod(remainder, 60)
        self.bot_uptime_str.set(f"{h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._start_uptime_timer)

    def _update_bot_performance(self, current_price: float):
        """Refresh the bot status & performance card."""
        # Realized P/L
        realized = self.bot_total_realized_pnl
        self.bot_realized_pnl_str.set(f"{realized:+,.2f} THB")
        clr_r = COLORS["green"] if realized >= 0 else COLORS["red"]
        self._perf_labels["realized_pnl"].config(fg=clr_r)

        # Total P/L = realized + unrealized
        unrealized = 0.0
        for pos in self.strategy.positions:
            unrealized += (current_price - pos.entry_price) * pos.amount
        total_pnl = realized + unrealized
        self.bot_total_pnl_str.set(f"{total_pnl:+,.2f} THB")
        clr_t = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
        self._perf_labels["total_pnl"].config(fg=clr_t)

        # Win rate
        if self.bot_total_trades > 0:
            wr = self.bot_win_trades / self.bot_total_trades * 100
            self.bot_winrate_str.set(f"{wr:.1f}%")
            clr_w = COLORS["green"] if wr >= 50 else COLORS["red"]
        else:
            self.bot_winrate_str.set("— %")
            clr_w = COLORS["text_bright"]
        self._perf_labels["winrate"].config(fg=clr_w)

        # Trade stats
        self.bot_trade_stats_label.config(
            text=f"Trades: {self.bot_total_trades}  |  "
                 f"✅ Win: {self.bot_win_trades}  |  "
                 f"❌ Lose: {self.bot_lose_trades}"
        )

        # Boss mode status
        if self.boss_mode.get():
            if self.boss_waiting_recovery:
                txt = (
                    f"🏆 Boss: รอซื้อคืน | ขาย {self.boss_last_sell_price:,.2f} | "
                    f"low {self.boss_recovery_low_price:,.2f}"
                )
            else:
                txt = "🏆 Boss: กำลังถือเหรียญ"
            self.bot_boss_status_label.config(text=txt, fg=COLORS["yellow"])
        else:
            self.bot_boss_status_label.config(text="Boss: OFF", fg=COLORS["text_dim"])

        if self.reentry_waiting and self.reentry_symbol:
            self.bot_boss_status_label.config(
                text=(
                    f"🔁 Re-Buy รอ {self.reentry_symbol} | ขาย {self.reentry_last_exit_price:,.2f} | "
                    f"low {self.reentry_recovery_low_price:,.2f}"
                ),
                fg=COLORS["accent"],
            )

    def _update_market_data(self):
        """Fetch and display current market data."""
        if not self.client or self._market_refresh_in_progress:
            return
        self._market_refresh_in_progress = True
        threading.Thread(target=self._do_update_market_data, daemon=True).start()

    def _do_update_market_data(self):
        """Fetch market data in a background thread."""
        try:
            snapshot = self._fetch_market_snapshot(self.client, self.config.trading.symbol)
            self.root.after(0, self._finish_market_update, snapshot)
        except Exception as e:
            self.root.after(0, self._handle_market_error, str(e))

    def _finish_market_update(self, snapshot: Dict):
        """Apply market data on the Tkinter thread."""
        self._market_refresh_in_progress = False
        self._apply_market_snapshot(snapshot)

    def _handle_market_error(self, error_message: str):
        """Handle market update errors on the Tkinter thread."""
        self._market_refresh_in_progress = False
        self._log(f"Market data error: {error_message}", "ERROR")

    def _fetch_market_snapshot(self, client: BitkubClient, symbol: str) -> Dict:
        """Fetch ticker data without touching Tkinter state."""
        ticker = client.get_ticker(symbol)
        if not ticker:
            return {}
        return {
            "price": ticker.get("last", 0),
            "change": ticker.get("change", 0),
        }

    def _apply_market_snapshot(self, snapshot: Dict):
        """Update market widgets from pre-fetched data."""
        if not snapshot:
            return

        price = snapshot.get("price", 0)
        change = snapshot.get("change", 0)
        self.current_price.set(f"{price:,.2f}")
        self.price_change.set(f"{change:+.2f}%")
        self.change_label.config(
            fg=COLORS["green"] if change >= 0 else COLORS["red"]
        )

        thb_info = self.wallet_balances.get("THB", {})
        balance = thb_info.get("available", 0)
        self.balance_thb.set(f"{balance:,.2f} THB")

    def _update_gui_data(self, signals: Dict, ai_prediction: Dict):
        """Update GUI with latest signals and AI data."""
        # Price
        price = signals.get("price", 0)
        self.current_price.set(f"{price:,.2f}")

        # Indicators
        rsi = signals.get("rsi", 0)
        self.rsi_val.set(f"{rsi:.1f}")

        macd = signals.get("macd", 0)
        self.macd_val.set(f"{macd:.2f}")

        bb_u = signals.get("bb_upper", 0)
        bb_l = signals.get("bb_lower", 0)
        self.bb_val.set(f"{bb_l:,.0f}-{bb_u:,.0f}")

        ema = signals.get("ema_21", 0)
        self.ema_val.set(f"{ema:,.2f}")

        # AI
        direction = ai_prediction.get("direction", "unknown")
        arrow = "⬆️ UP" if direction == "up" else "⬇️ DOWN" if direction == "down" else "➡️ N/A"
        self.ai_direction.set(arrow)

        conf = ai_prediction.get("confidence", 0)
        self.ai_confidence.set(f"{conf:.0%}")

        pred = ai_prediction.get("predicted_price", 0)
        self.ai_predicted.set(f"{pred:,.2f}")

    def _update_positions_display(self, current_price: float):
        """Update positions table."""
        # Clear existing
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)

        for pos in self.strategy.positions:
            pnl = (current_price - pos.entry_price) / pos.entry_price * 100
            self.positions_tree.insert("", "end", values=(
                pos.symbol,
                f"{pos.entry_price:,.2f}",
                f"{current_price:,.2f}",
                f"{pnl:+.2f}%",
                f"{pos.amount:.8f}",
                f"{pos.stop_loss_price:,.2f}",
                f"{pos.take_profit_price:,.2f}",
            ))

    def _update_balance_display(self, balance: float, current_price: float):
        """Update balance display."""
        total = balance
        for pos in self.strategy.positions:
            total += pos.amount * current_price

        self.balance_thb.set(f"{balance:,.2f} THB")
        self.total_value.set(f"{total:,.2f} THB")

        pnl = total - balance
        self.pnl_text.set(f"{pnl:+,.2f} THB")

    def _update_risk_display(self, balance: float):
        """Update risk status."""
        risk = self.risk_manager.get_risk_status(balance, self.strategy.positions)
        self.risk_text.config(
            text=(
                f"Daily Loss: {risk['daily_loss']:,.0f} / "
                f"{risk['max_daily_loss']:,.0f} THB  |  "
                f"Positions: {risk['open_positions']} / {risk['max_positions']}  |  "
                f"Exposure: {risk['exposure_pct']:.1f}%"
            )
        )

    def _update_pnl_display(self, current_price: float):
        """Update real-time P/L display for all open positions."""
        positions = self.strategy.positions
        if not positions:
            self.realtime_pnl.set("0.00 THB")
            self.realtime_pnl_pct.set("0.00%")
            self.pnl_thb_label.config(fg=COLORS["text_bright"])
            self.pnl_pct_label.config(fg=COLORS["text_bright"])
            self.pnl_summary_label.config(text="ไม่มี position เปิดอยู่")
            return

        total_pnl_thb = 0
        total_cost = 0
        summaries = []

        for pos in positions:
            pnl_thb = (current_price - pos.entry_price) * pos.amount
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            cost = pos.entry_price * pos.amount
            total_pnl_thb += pnl_thb
            total_cost += cost
            coin = pos.symbol.split("_")[0]
            summaries.append(
                f"{coin}: {pos.amount:.6g} @ {pos.entry_price:,.2f} → "
                f"{current_price:,.2f} ({pnl_pct:+.2f}%)"
            )

        total_pnl_pct = (total_pnl_thb / total_cost * 100) if total_cost > 0 else 0

        # Update labels
        self.realtime_pnl.set(f"{total_pnl_thb:+,.2f} THB")
        self.realtime_pnl_pct.set(f"{total_pnl_pct:+.2f}%")

        # Color: green for profit, red for loss
        color = COLORS["green"] if total_pnl_thb >= 0 else COLORS["red"]
        self.pnl_thb_label.config(fg=color)
        self.pnl_pct_label.config(fg=color)

        # Summary
        summary = " | ".join(summaries)
        self.pnl_summary_label.config(text=summary, fg=color)

    def _add_history_entry(self, trade_type: str, price: float, pnl: float):
        """Add entry to trade history table."""
        now = datetime.now().strftime("%H:%M:%S")
        self.history_tree.insert("", 0, values=(
            now, trade_type, f"{price:,.2f}", f"{pnl:+.2f}%"
        ))

    def _manual_refresh(self):
        """Manual data refresh."""
        if not self.client:
            messagebox.showinfo("Info", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return

        self._log("Manual refresh...", "INFO")
        threading.Thread(target=self._do_refresh, daemon=True).start()

    def _do_refresh(self):
        """Perform refresh in background thread."""
        if not self.data_collector:
            return
        try:
            symbol = self.config.trading.symbol
            df = self.data_collector.fetch_ohlcv(symbol)
            if df is not None and not df.empty:
                df_ind = self.indicator_engine.add_all_indicators(df)
                signals = self.indicator_engine.get_signal_summary(df_ind)
                ai_pred = self.lstm_predictor.predict(df_ind)
                self.root.after(0, self._update_gui_data, signals, ai_pred)

            self.root.after(0, self._update_market_data)
            self.root.after(0, self._log, "Data refreshed", "SUCCESS")
        except Exception as e:
            self.root.after(0, self._log, f"Refresh error: {e}", "ERROR")

    def _train_ai(self):
        """Train AI models in background."""
        if not self.client:
            messagebox.showinfo("Info", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return

        self._log("Starting AI training...", "AI")
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        """Perform AI training in background."""
        if not self.data_collector:
            return
        try:
            symbol = self.config.trading.symbol
            self.root.after(0, self._log, "Fetching training data...", "AI")

            df = self.data_collector.fetch_ohlcv(symbol, "1h", 1000)
            if df is None or df.empty:
                self.root.after(0, self._log, "No training data available", "ERROR")
                return

            df_ind = self.indicator_engine.add_all_indicators(df)
            df_ind = df_ind.dropna()

            self.root.after(0, self._log,
                            f"Training data: {len(df_ind)} bars", "AI")

            # Train LSTM
            self.root.after(0, self._log, "Training LSTM model...", "AI")
            self.lstm_predictor.train(df_ind)
            self.root.after(0, self._log, "LSTM training complete ✅", "SUCCESS")

            # Train RL
            self.root.after(0, self._log, "Training RL agent...", "AI")
            self.rl_agent.train(df_ind, episodes=200)
            self.root.after(0, self._log, "RL training complete ✅", "SUCCESS")

            self.root.after(0, self._log, "All AI models trained and saved! 🎉", "SUCCESS")

        except Exception as e:
            self.root.after(0, self._log, f"Training error: {e}", "ERROR")

    def _run_backtest(self):
        """Run backtest in background."""
        if not self.client:
            messagebox.showinfo("Info", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return

        self._log("Starting backtest...", "INFO")
        threading.Thread(target=self._do_backtest, daemon=True).start()

    def _do_backtest(self):
        """Perform backtest in background."""
        if not self.data_collector:
            return
        try:
            symbol = self.config.trading.symbol
            df = self.data_collector.fetch_ohlcv(symbol, "1h", 1000)
            if df is None or df.empty:
                self.root.after(0, self._log, "No backtest data available", "ERROR")
                return

            self.root.after(0, self._log,
                            f"Backtest data: {len(df)} bars", "INFO")

            backtester = Backtester(self.config.trading, self.config.risk, self.logger)
            result = backtester.run(df)

            # Show results
            self.root.after(0, self._log, "=" * 40, "INFO")
            self.root.after(0, self._log, "📊 BACKTEST RESULTS", "INFO")
            self.root.after(0, self._log,
                            f"  Return: {result.total_return_pct:+.2f}%", "INFO")
            self.root.after(0, self._log,
                            f"  Trades: {result.total_trades} | Win Rate: {result.win_rate:.1f}%", "INFO")
            self.root.after(0, self._log,
                            f"  Max Drawdown: {result.max_drawdown_pct:.2f}%", "INFO")
            self.root.after(0, self._log,
                            f"  Sharpe: {result.sharpe_ratio:.2f} | PF: {result.profit_factor:.2f}", "INFO")
            self.root.after(0, self._log,
                            f"  Final Balance: {result.final_balance:,.2f} THB", "INFO")
            self.root.after(0, self._log, "=" * 40, "INFO")

        except Exception as e:
            self.root.after(0, self._log, f"Backtest error: {e}", "ERROR")

    # ─── Run ──────────────────────────────────────────────────

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    app = TradingBotGUI()
    app.run()


if __name__ == "__main__":
    main()
