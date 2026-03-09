"""
Bitkub Trading Bot - GUI Dashboard (Tkinter)
หน้า GUI แสดง balance, positions, indicators, AI predictions, trade history
พร้อมปุ่มควบคุม Start/Stop bot, Train AI, Backtest
"""
import os
import re
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
from ai_model.llm_advisor import LLMBossAdvisor
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
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close_request)

        # State
        self.config = AppConfig.from_env()
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
        self.llm_boss_advisor = LLMBossAdvisor(self.config.ai, self.config.trading, self.logger)

        # Data variables
        self.current_price = tk.StringVar(value="0.00")
        self.price_change = tk.StringVar(value="0.00%")
        self.balance_thb = tk.StringVar(value="0.00 THB")
        self.total_value = tk.StringVar(value="0.00 THB")
        self.pnl_text = tk.StringVar(value="0.00 THB")
        self.bot_status = tk.StringVar(value="⏹ หยุดอยู่")
        self.rsi_val = tk.StringVar(value="-")
        self.macd_val = tk.StringVar(value="-")
        self.bb_val = tk.StringVar(value="-")
        self.ema_val = tk.StringVar(value="-")
        self.ai_direction = tk.StringVar(value="-")
        self.ai_confidence = tk.StringVar(value="-")
        self.ai_predicted = tk.StringVar(value="-")
        self.llm_trade_enabled = tk.BooleanVar(value=bool(self.config.ai.llm_enabled))
        self.paper_trade_enabled = tk.BooleanVar(value=bool(self.config.trading.paper_trade_enabled))
        self.trade_llm_status = tk.StringVar(value="LLM trade: disabled")
        self.boss_realtime_status = tk.StringVar(value="Boss: OFF")
        self.boss_realtime_detail = tk.StringVar(value="รอข้อมูล realtime...")
        self.boss_llm_status = tk.StringVar(value="LLM: disabled")
        self.risk_regime_text = tk.StringVar(value="Market regime: waiting for data")
        self.risk_guard_text = tk.StringVar(value="Position sizing: waiting for data")
        self.ui_connect_state = tk.StringVar(value="ยังไม่ได้เชื่อมต่อ Exchange")
        self.ui_mode_state = tk.StringVar(value="โหมด: LIVE")
        self.ui_focus_state = tk.StringVar(value="ขั้นตอนถัดไป: เชื่อมต่อ API และโหลด wallet")
        self.cycle_count = 0
        self.wallet_balances: Dict[str, Dict] = {}  # detailed balances
        self.wallet_price_map: Dict[str, float] = {}
        self.last_signals: Dict = {}
        self.last_ai_prediction: Dict = {}
        self.last_thb_balance = 0.0
        self.last_boss_llm_advice: Dict = {}
        self.last_trade_llm_advice: Dict = {}
        self.is_connected = False
        self.initial_portfolio_value = 0.0  # portfolio value when bot started
        self.paper_balance_thb = float(self.config.trading.paper_trade_start_balance_thb or 0.0)
        self.realtime_pnl = tk.StringVar(value="0.00 THB")
        self.realtime_pnl_pct = tk.StringVar(value="0.00%")
        self.trade_amount_var = tk.StringVar(value="100")

        # Boss Mode state
        self.boss_mode = tk.BooleanVar(value=True)
        self.boss_cutloss_pct_var = tk.StringVar(value=f"{self.config.trading.ai_cutloss_min_loss_pct:.2f}")
        self.boss_recovery_pct_var = tk.StringVar(value=f"{self.config.trading.adaptive_rebuy_floor_pct:.2f}")
        self.boss_last_sell_price = 0.0  # track price at which boss sold
        self.boss_recovery_low_price = 0.0
        self.boss_rebuy_budget_thb = 0.0
        self.boss_recovery_sell_cycle = 0
        self.boss_recovery_confirm_count = 0
        self.boss_waiting_recovery = False  # True = sold, waiting for price to recover before buy-back

        # Auto re-entry state
        self.auto_reentry_enabled = tk.BooleanVar(value=True)
        self.reentry_rise_pct_var = tk.StringVar(value=f"{self.config.trading.adaptive_rebuy_floor_pct:.2f}")
        self.reentry_delay_pct_var = tk.StringVar(value=f"{self.config.trading.adaptive_reentry_delay_floor_pct:.2f}")
        self.reentry_waiting = False
        self.reentry_symbol = ""
        self.reentry_last_exit_price = 0.0
        self.reentry_recovery_low_price = 0.0
        self.reentry_budget_thb = 0.0
        self.reentry_sell_cycle = 0
        self.reentry_confirm_count = 0
        self.reentry_last_reason = ""

        # Bot performance tracking
        self.bot_start_time: Optional[datetime] = None
        self.bot_total_realized_pnl = 0.0      # cumulative realized P/L (THB)
        self.bot_total_trades = 0               # total completed trades
        self.bot_win_trades = 0                 # winning trades
        self.bot_lose_trades = 0                # losing trades
        self.bot_last_action = tk.StringVar(value="—")
        self.bot_decision_status = tk.StringVar(value="รอประเมินสัญญาณ...")
        self.bot_decision_detail = tk.StringVar(value="ระบบจะแสดงเหตุผล BUY / HOLD / WAIT และราคาเป้าหมายที่นี่")
        self.bot_uptime_str = tk.StringVar(value="00:00:00")
        self.bot_cycles_str = tk.StringVar(value="0")
        self.bot_realized_pnl_str = tk.StringVar(value="0.00 THB")
        self.bot_unrealized_pnl_str = tk.StringVar(value="0.00 THB")
        self.bot_total_pnl_str = tk.StringVar(value="0.00 THB")
        self.bot_winrate_str = tk.StringVar(value="— %")
        self.bot_last_loss_source_str = tk.StringVar(value="ขาดทุนที่ปิดล่าสุด: ยังไม่มี")
        self.bot_unrealized_source_str = tk.StringVar(value="ลอยตัวตอนนี้: ยังไม่มี position")
        self.auto_trade_amount_var = tk.StringVar(value="100")
        self.quick_buy_mode_var = tk.StringVar(value="manual_auto")
        self.quick_buy_mode_hint = tk.StringVar(value="Manual Buy + Auto Trade: ซื้อทันทีแล้วให้บอทดูแล position ต่อ")
        self.ai_scale_in_enabled = tk.BooleanVar(value=True)
        self.ai_scale_in_loss_pct_var = tk.StringVar(value=f"{self.config.trading.ai_scale_in_loss_pct:.2f}")
        self.ai_take_profit_enabled = tk.BooleanVar(value=True)
        self.ai_take_profit_pct_var = tk.StringVar(value=f"{self.config.trading.ai_take_profit_min_profit_pct:.2f}")
        self.profit_cashout_enabled = tk.BooleanVar(value=self.config.trading.profit_cashout_enabled)
        self.profit_cashout_pct_var = tk.StringVar(value=f"{self.config.trading.profit_cashout_min_profit_pct:.2f}")
        self.fee_guard_enabled = tk.BooleanVar(value=self.config.trading.small_position_fee_guard_enabled)
        self.fee_guard_max_cost_var = tk.StringVar(value=f"{self.config.trading.small_position_fee_guard_max_cost_thb:.0f}")
        self._runtime_settings_snapshot: Dict[str, object] = {}
        self._runtime_settings_invalid = False
        self._runtime_badges: Dict[str, tk.Label] = {}
        self._runtime_field_normalizers: Dict[str, object] = {}
        self._ai_metric_labels: Dict[str, tk.Label] = {}
        self._connect_in_progress = False
        self._wallet_refresh_in_progress = False
        self._market_refresh_in_progress = False
        self._bot_start_in_progress = False
        self._paper_toggle_guard = False
        self._paper_trade_enabled_last = bool(self.paper_trade_enabled.get())
        self._active_scroll_canvas: Optional[tk.Canvas] = None
        self._mousewheel_handler_registered = False
        self._ai_model_lock = threading.RLock()
        self._ai_training_in_progress = False
        self._trade_cycle_lock = threading.Lock()
        self._runtime_apply_after_id = None
        self._scroll_columns: Dict[str, tk.Canvas] = {}
        self._collapsible_cards: Dict[str, Dict[str, object]] = {}
        self._fee_guard_history_state: Dict[str, bool] = {}

        self._build_ui()
        self._setup_runtime_badges()
        self._apply_styles()
        self.llm_trade_enabled.trace_add("write", self._on_llm_trade_toggle)
        self.paper_trade_enabled.trace_add("write", self._on_paper_trade_toggle)
        self.quick_buy_mode_var.trace_add("write", self._on_quick_buy_mode_change)
        self._sync_llm_trade_status_visual()
        self._refresh_quick_buy_mode_ui()

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
        left_outer, left = self._create_scrollable_column(body, "left", width=0)
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel
        right_outer, right = self._create_scrollable_column(body, "right", width=390)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Left: Price + Bot Status + Wallet + Indicators + Positions + Log
        self._build_price_card(left)
        self._build_bot_status_card(left)
        self._build_pnl_card(left)
        self._build_wallet_card(left)
        self._build_indicators_card(left)
        self._build_positions_card(left)
        self._build_log_card(left)

        # Right: API Settings + Quick Trade + AI + Controls + Trade History
        self._build_quick_start_card(right)
        self._build_api_settings(right)
        self._build_quick_trade_card(right)
        self._build_ai_card(right)
        self._build_controls_card(right)
        self._build_risk_card(right)
        self._build_history_card(right)

        self._refresh_quick_start_summary()

    def _create_scrollable_column(self, parent, column_name: str, width: int = 0):
        """Create a reusable scrollable column for dense dashboard sections."""
        outer = tk.Frame(parent, bg=COLORS["bg"], width=width if width > 0 else 1)
        if width > 0:
            outer.pack_propagate(False)

        toolbar = tk.Frame(outer, bg=COLORS["bg"], height=28)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        toolbar.pack_propagate(False)

        side_label = "LEFT PANEL" if column_name == "left" else "RIGHT PANEL"
        tk.Label(
            toolbar,
            text=side_label,
            font=("Segoe UI", 7, "bold"),
            fg=COLORS["text_dim"],
            bg=COLORS["bg"],
        ).pack(side=tk.LEFT, padx=(2, 0))

        tk.Button(
            toolbar,
            text="TOP",
            font=("Segoe UI", 7, "bold"),
            bg=COLORS["bg_input"],
            fg=COLORS["text_bright"],
            activebackground=COLORS["accent2"],
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda name=column_name: self._scroll_column_to(name, "top"),
        ).pack(side=tk.RIGHT, padx=(4, 0), ipady=1)

        tk.Button(
            toolbar,
            text="BOTTOM",
            font=("Segoe UI", 7, "bold"),
            bg=COLORS["bg_input"],
            fg=COLORS["text_bright"],
            activebackground=COLORS["accent"],
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda name=column_name: self._scroll_column_to(name, "bottom"),
        ).pack(side=tk.RIGHT, ipady=1)

        canvas = tk.Canvas(
            outer,
            bg=COLORS["bg"],
            highlightthickness=0,
            bd=0,
            width=width if width > 0 else 1,
        )
        scrollbar = tk.Scrollbar(
            outer,
            orient=tk.VERTICAL,
            command=canvas.yview,
            width=15,
            bg=COLORS["bg_input"],
            activebackground=COLORS["accent"],
            troughcolor=COLORS["bg_card"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            elementborderwidth=0,
        )
        content = tk.Frame(canvas, bg=COLORS["bg"])

        content.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(sidebar_window, width=event.width),
        )

        sidebar_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas._linked_scroll_canvas = canvas
        canvas._scroll_column_name = column_name
        scrollbar._linked_scroll_canvas = canvas
        content._linked_scroll_canvas = canvas
        outer._linked_scroll_canvas = canvas

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.bind("<Enter>", lambda event, target=canvas: self._set_active_scroll_canvas(target))
        canvas.bind("<Leave>", lambda event, target=canvas: self._clear_active_scroll_canvas(target))
        content.bind("<Enter>", lambda event, target=canvas: self._set_active_scroll_canvas(target))
        content.bind("<Leave>", lambda event, target=canvas: self._clear_active_scroll_canvas(target))
        scrollbar.bind("<Enter>", lambda event, target=canvas: self._set_active_scroll_canvas(target))
        scrollbar.bind("<Leave>", lambda event, target=canvas: self._clear_active_scroll_canvas(target))

        if not self._mousewheel_handler_registered:
            self.root.bind_all("<MouseWheel>", self._handle_global_mousewheel)
            self.root.bind_all("<Shift-MouseWheel>", self._handle_shift_mousewheel)
            self._mousewheel_handler_registered = True

        self._scroll_columns[column_name] = canvas

        return outer, content

    def _scroll_column_to(self, column_name: str, target: str):
        """Jump a named scrollable column to the top or bottom."""
        canvas = self._scroll_columns.get(column_name)
        if canvas is None:
            return
        canvas.yview_moveto(1.0 if target == "bottom" else 0.0)

    def _set_active_scroll_canvas(self, canvas: tk.Canvas):
        """Remember which scrollable column is currently under the pointer."""
        self._active_scroll_canvas = canvas

    def _clear_active_scroll_canvas(self, canvas: tk.Canvas):
        """Clear the active scroll target when leaving a scrollable column."""
        if self._active_scroll_canvas is canvas:
            self._active_scroll_canvas = None

    def _resolve_scroll_canvas_from_event(self, event) -> Optional[tk.Canvas]:
        """Resolve the scrollable canvas currently under the pointer."""
        widget = None
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
        except tk.TclError:
            widget = None

        while widget is not None:
            linked_canvas = getattr(widget, "_linked_scroll_canvas", None)
            if linked_canvas is not None:
                return linked_canvas
            widget = getattr(widget, "master", None)

        return self._active_scroll_canvas

    @staticmethod
    def _scroll_canvas(canvas: Optional[tk.Canvas], steps: int, mode: str = "units"):
        """Scroll a canvas safely when a valid target exists."""
        if canvas is None or steps == 0:
            return
        canvas.yview_scroll(steps, mode)

    def _handle_global_mousewheel(self, event):
        """Route mouse-wheel scrolling to the column currently under the cursor."""
        if not event.delta:
            return
        canvas = self._resolve_scroll_canvas_from_event(event)
        if canvas is None:
            return
        steps = int(-event.delta / 120)
        if steps == 0:
            steps = -1 if event.delta > 0 else 1
        self._scroll_canvas(canvas, steps, "units")

    def _handle_shift_mousewheel(self, event):
        """Allow faster page-style scrolling while holding Shift."""
        if not event.delta:
            return
        canvas = self._resolve_scroll_canvas_from_event(event)
        if canvas is None:
            return
        steps = -1 if event.delta > 0 else 1
        self._scroll_canvas(canvas, steps, "pages")

    def _build_quick_start_card(self, parent):
        """Show the simplest next steps and current app state."""
        card = self._make_card(parent, "🧭 QUICK START", "เริ่มใช้งานได้ใน 3 ขั้นตอน", collapsible=True, collapse_key="quick_start")

        status_grid = tk.Frame(card, bg=COLORS["bg_card"])
        status_grid.pack(fill=tk.X, padx=10, pady=(4, 6))

        items = [
            ("การเชื่อมต่อ", self.ui_connect_state, COLORS["yellow"]),
            ("โหมด", self.ui_mode_state, COLORS["accent"]),
            ("สิ่งที่ควรทำต่อ", self.ui_focus_state, COLORS["green"]),
        ]
        for row_index, (label_text, variable, color) in enumerate(items):
            box = tk.Frame(status_grid, bg=COLORS["bg_input"], padx=10, pady=6)
            box.grid(row=row_index, column=0, sticky="ew", pady=3)
            status_grid.columnconfigure(0, weight=1)
            tk.Label(
                box, text=label_text, font=("Segoe UI", 8),
                fg=COLORS["text_dim"], bg=COLORS["bg_input"], anchor="w"
            ).pack(fill=tk.X)
            tk.Label(
                box, textvariable=variable, font=("Segoe UI", 9, "bold"),
                fg=color, bg=COLORS["bg_input"], anchor="w", justify=tk.LEFT,
                wraplength=320
            ).pack(fill=tk.X)

        steps = tk.Frame(card, bg=COLORS["bg_card"])
        steps.pack(fill=tk.X, padx=10, pady=(0, 8))
        for step in [
            "1. Connect & Load Wallet",
            "2. ตรวจ Auto Buy, SL, TP และเลือก Paper/Live",
            "3. กด START BOT แล้วดู Decision / Risk Status",
        ]:
            tk.Label(
                steps,
                text=step,
                font=("Segoe UI", 8),
                fg=COLORS["text_dim"],
                bg=COLORS["bg_card"],
                anchor="w",
            ).pack(fill=tk.X, pady=1)

    def _refresh_quick_start_summary(self):
        """Refresh the top-right quick start state summary."""
        symbol = self.config.trading.symbol
        mode_label = "PAPER" if self._is_paper_trade_mode() else "LIVE"
        self.ui_mode_state.set(f"โหมด: {mode_label} | Symbol: {symbol}")

        if self.is_connected:
            self.ui_connect_state.set("เชื่อมต่อแล้ว พร้อมโหลดราคาและ wallet")
        else:
            self.ui_connect_state.set("ยังไม่ได้เชื่อมต่อ Exchange")

        if not self.is_connected:
            self.ui_focus_state.set("ขั้นตอนถัดไป: ใส่ API แล้วกด Connect & Load Wallet")
        elif not self.bot_running:
            self.ui_focus_state.set("ขั้นตอนถัดไป: ตรวจค่าความเสี่ยง แล้วกด START BOT")
        elif self.boss_waiting_recovery or self.reentry_waiting:
            self.ui_focus_state.set("บอทกำลังรอจังหวะซื้อคืน ดูรายละเอียดที่ Decision และ Risk Status")
        else:
            self.ui_focus_state.set("บอทกำลังทำงาน ดู Action ล่าสุด, Decision และ P/L ได้ทันที")

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
        card = self._make_card(parent, "🤖 สถานะและผลงานบอท")

        # ── Row 1: Status indicator + Uptime + Cycles ──
        row1 = tk.Frame(card, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, padx=10, pady=(5, 2))

        # Live / Stopped indicator
        self.bot_alive_label = tk.Label(
            row1, text="⏹ หยุดอยู่", font=("Segoe UI", 11, "bold"),
            fg=COLORS["red"], bg=COLORS["bg_card"]
        )
        self.bot_alive_label.pack(side=tk.LEFT)

        # Uptime
        upt_f = tk.Frame(row1, bg=COLORS["bg_card"])
        upt_f.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(upt_f, text="⏱ เวลาทำงาน:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        tk.Label(upt_f, textvariable=self.bot_uptime_str,
                 font=("Consolas", 10, "bold"),
                 fg=COLORS["text_bright"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(4, 0))

        # Cycles
        cyc_f = tk.Frame(row1, bg=COLORS["bg_card"])
        cyc_f.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(cyc_f, text="🔄 รอบทำงาน:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        tk.Label(cyc_f, textvariable=self.bot_cycles_str,
                 font=("Consolas", 10, "bold"),
                 fg=COLORS["text_bright"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(4, 0))

        # Last action
        act_f = tk.Frame(row1, bg=COLORS["bg_card"])
        act_f.pack(side=tk.RIGHT)
        tk.Label(act_f, text="ล่าสุด:", font=("Segoe UI", 8),
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
            ("💰 กำไร/ขาดทุนที่ปิดแล้ว", self.bot_realized_pnl_str, "realized_pnl"),
            ("📈 กำไร/ขาดทุนลอยตัว", self.bot_unrealized_pnl_str, "unrealized_pnl"),
            ("📊 กำไร/ขาดทุนรวม", self.bot_total_pnl_str, "total_pnl"),
            ("🏆 อัตราชนะ", self.bot_winrate_str, "winrate"),
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
            text="เทรด: 0  |  ✅ ชนะ: 0  |  ❌ แพ้: 0",
            font=("Segoe UI", 9),
            fg=COLORS["text_dim"], bg=COLORS["bg_card"]
        )
        self.bot_trade_stats_label.pack(side=tk.LEFT)

        row4 = tk.Frame(card, bg=COLORS["bg_input"])
        row4.pack(fill=tk.X, padx=10, pady=(0, 6))

        tk.Label(
            row4, textvariable=self.bot_last_loss_source_str, font=("Segoe UI", 8),
            fg=COLORS["red"], bg=COLORS["bg_input"], anchor="w", justify=tk.LEFT,
            wraplength=760
        ).pack(fill=tk.X, padx=10, pady=(6, 2))

        tk.Label(
            row4, textvariable=self.bot_unrealized_source_str, font=("Segoe UI", 8),
            fg=COLORS["text_dim"], bg=COLORS["bg_input"], anchor="w", justify=tk.LEFT,
            wraplength=760
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

        row5 = tk.Frame(card, bg=COLORS["bg_input"])
        row5.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Label(
            row5, text="สรุปเหตุผลซื้อ/รอ/ถือ", font=("Segoe UI", 8),
            fg=COLORS["text_dim"], bg=COLORS["bg_input"], anchor="w"
        ).pack(fill=tk.X, padx=10, pady=(6, 0))

        self.bot_decision_status_label = tk.Label(
            row5, textvariable=self.bot_decision_status, font=("Segoe UI", 9, "bold"),
            fg=COLORS["text_bright"], bg=COLORS["bg_input"], anchor="w", justify=tk.LEFT
        )
        self.bot_decision_status_label.pack(fill=tk.X, padx=10, pady=(0, 2))

        self.bot_decision_detail_label = tk.Label(
            row5, textvariable=self.bot_decision_detail, font=("Segoe UI", 8),
            fg=COLORS["text_dim"], bg=COLORS["bg_input"], anchor="w", justify=tk.LEFT,
            wraplength=380
        )
        self.bot_decision_detail_label.pack(fill=tk.X, padx=10, pady=(0, 6))

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

        columns = ("symbol", "entry", "current", "pnl", "pnl_thb", "amount", "sl", "tp")
        self.positions_tree = ttk.Treeview(card, columns=columns, show="headings",
                                           height=4)

        headers = {
            "symbol": ("Symbol", 74),
            "entry": ("Entry Price", 92),
            "current": ("Current", 92),
            "pnl": ("P/L %", 64),
            "pnl_thb": ("P/L THB", 88),
            "amount": ("Amount", 94),
            "sl": ("Stop Loss", 82),
            "tp": ("Take Profit", 82),
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
        card = self._make_card(parent, "🔑 API SETTINGS", collapsible=True, collapse_key="api_settings")

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

        self.symbol_var = tk.StringVar(value=self.config.trading.symbol)
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
        card = self._make_card(parent, "🧠 AI PREDICTION", "สรุปมุมมอง AI ที่มีผลต่อการเข้าออก", collapsible=True, collapse_key="ai_prediction")

        grid = tk.Frame(card, bg=COLORS["bg_card"])
        grid.pack(fill=tk.X, padx=10, pady=5)

        for i, (key, label, var) in enumerate([
            ("direction", "Direction", self.ai_direction),
            ("confidence", "Confidence", self.ai_confidence),
            ("predicted", "Predicted", self.ai_predicted),
            ("llm_trade", "LLM Trade", self.trade_llm_status),
        ]):
            f = tk.Frame(grid, bg=COLORS["bg_input"], padx=8, pady=4)
            f.grid(row=0, column=i, padx=3, sticky="ew")
            grid.columnconfigure(i, weight=1)

            tk.Label(f, text=label, font=("Segoe UI", 8),
                     fg=COLORS["text_dim"], bg=COLORS["bg_input"]).pack()
            value_label = tk.Label(
                f,
                textvariable=var,
                font=("Segoe UI", 10, "bold"),
                fg=COLORS["text_bright"],
                bg=COLORS["bg_input"],
            )
            value_label.pack()
            self._ai_metric_labels[key] = value_label

    def _build_controls_card(self, parent):
        """Build control buttons."""
        card = self._make_card(parent, "⚙️ CONTROLS", "ค่าใช้งานจริงของบอทและระบบอัตโนมัติ", collapsible=True, collapse_key="controls")

        tk.Label(
            card,
            text="Core Settings",
            font=("Segoe UI", 8, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg_card"],
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(4, 0))

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
        self.interval_var = tk.StringVar(value=str(self.config.trading.trading_interval_seconds))
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
        self.sl_var = tk.StringVar(value=f"{self.config.trading.stop_loss_pct:.2f}")
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
        self.tp_var = tk.StringVar(value=f"{self.config.trading.take_profit_pct:.2f}")
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

        tk.Label(
            card,
            text="ระบบจะ cap ขนาดไม้ซื้ออัตโนมัติตาม downtrend, ATR, และ loss streak อัตโนมัติ",
            font=("Segoe UI", 8),
            fg=COLORS["text_dim"],
            bg=COLORS["bg_card"],
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(
            card,
            text="Automation",
            font=("Segoe UI", 8, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg_card"],
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(2, 0))

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

        ai_row3 = tk.Frame(ai_manage_frame, bg=COLORS["bg_card"])
        ai_row3.pack(fill=tk.X, pady=(2, 0))
        self.profit_cashout_check = tk.Checkbutton(
            ai_row3, text="💸 Profit Cashout", font=("Segoe UI", 9, "bold"),
            variable=self.profit_cashout_enabled, fg=COLORS["yellow"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["yellow"],
            cursor="hand2"
        )
        self.profit_cashout_check.pack(side=tk.LEFT)
        self._create_runtime_badge(ai_row3, "profit_cashout_enabled")
        tk.Label(ai_row3, text="Profit %:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(12, 4))
        self._create_runtime_badge(ai_row3, "profit_cashout_pct", padx=(0, 6))
        tk.Entry(ai_row3, textvariable=self.profit_cashout_pct_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(side=tk.LEFT, ipady=2)

        ai_row4 = tk.Frame(ai_manage_frame, bg=COLORS["bg_card"])
        ai_row4.pack(fill=tk.X, pady=(2, 0))
        self.fee_guard_check = tk.Checkbutton(
            ai_row4, text="🛡 Small Fee Guard", font=("Segoe UI", 9, "bold"),
            variable=self.fee_guard_enabled, fg=COLORS["accent"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["accent"],
            cursor="hand2"
        )
        self.fee_guard_check.pack(side=tk.LEFT)
        self._create_runtime_badge(ai_row4, "fee_guard_enabled")
        tk.Label(ai_row4, text="Max THB:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(side=tk.LEFT, padx=(12, 4))
        self._create_runtime_badge(ai_row4, "fee_guard_max_cost", padx=(0, 6))
        tk.Entry(ai_row4, textvariable=self.fee_guard_max_cost_var, font=("Segoe UI", 9),
                 bg=COLORS["bg_input"], fg=COLORS["text"], width=6,
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(side=tk.LEFT, ipady=2)

        ai_row5 = tk.Frame(ai_manage_frame, bg=COLORS["bg_card"])
        ai_row5.pack(fill=tk.X, pady=(2, 0))
        self.llm_trade_check = tk.Checkbutton(
            ai_row5, text="🧠 LLM Trade Advisor", font=("Segoe UI", 9, "bold"),
            variable=self.llm_trade_enabled, fg=COLORS["yellow"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["yellow"],
            cursor="hand2"
        )
        self.llm_trade_check.pack(side=tk.LEFT)
        self._create_runtime_badge(ai_row5, "llm_trade_enabled")

        ai_row6 = tk.Frame(ai_manage_frame, bg=COLORS["bg_card"])
        ai_row6.pack(fill=tk.X, pady=(2, 0))
        self.paper_trade_check = tk.Checkbutton(
            ai_row6, text="🧪 Paper Trading", font=("Segoe UI", 9, "bold"),
            variable=self.paper_trade_enabled, fg=COLORS["accent"],
            bg=COLORS["bg_card"], selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"], activeforeground=COLORS["accent"],
            cursor="hand2"
        )
        self.paper_trade_check.pack(side=tk.LEFT)
        self._create_runtime_badge(ai_row6, "paper_trade_enabled")

        # Boss Mode section
        tk.Label(
            card,
            text="Recovery",
            font=("Segoe UI", 8, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg_card"],
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(4, 0))

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
        card = self._make_card(parent, "⚠️ RISK STATUS", "ดูว่าระบบกำลังลดไม้หรือพักซื้อเพราะอะไร", collapsible=True, collapse_key="risk_status")

        self.risk_text = tk.Label(
            card, text="Daily Loss: 0 / 3,000 THB  |  Positions: 0 / 2",
            font=("Segoe UI", 9), fg=COLORS["text"],
            bg=COLORS["bg_card"], anchor="w"
        )
        self.risk_text.pack(fill=tk.X, padx=10, pady=5)

        self.risk_regime_label = tk.Label(
            card, textvariable=self.risk_regime_text,
            font=("Segoe UI", 8, "bold"), fg=COLORS["yellow"],
            bg=COLORS["bg_card"], anchor="w", justify=tk.LEFT
        )
        self.risk_regime_label.pack(fill=tk.X, padx=10, pady=(0, 3))

        self.risk_guard_label = tk.Label(
            card, textvariable=self.risk_guard_text,
            font=("Segoe UI", 8), fg=COLORS["text_dim"],
            bg=COLORS["bg_card"], anchor="w", justify=tk.LEFT, wraplength=340
        )
        self.risk_guard_label.pack(fill=tk.X, padx=10, pady=(0, 6))

    def _build_quick_trade_card(self, parent):
        """Build manual quick trade card with Buy/Sell buttons."""
        card = self._make_card(parent, "💸 QUICK TRADE", "เลือกได้ว่าจะซื้อทันทีหรือให้บอทตัดสินใจซื้อหนึ่งรอบ", collapsible=True, collapse_key="quick_trade")

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

        mode_frame = tk.Frame(card, bg=COLORS["bg_card"])
        mode_frame.pack(fill=tk.X, padx=10, pady=(2, 6))

        tk.Label(mode_frame, text="Quick Buy Mode:", font=("Segoe UI", 8),
                 fg=COLORS["text_dim"], bg=COLORS["bg_card"]).pack(anchor="w")

        mode_options = tk.Frame(mode_frame, bg=COLORS["bg_card"])
        mode_options.pack(fill=tk.X, pady=(3, 0))

        tk.Radiobutton(
            mode_options,
            text="Manual Buy + Auto Trade",
            value="manual_auto",
            variable=self.quick_buy_mode_var,
            font=("Segoe UI", 9),
            fg=COLORS["text"],
            bg=COLORS["bg_card"],
            selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"],
            activeforeground=COLORS["text_bright"],
            cursor="hand2",
        ).pack(anchor="w")

        tk.Radiobutton(
            mode_options,
            text="Auto Trade Only",
            value="auto_only",
            variable=self.quick_buy_mode_var,
            font=("Segoe UI", 9),
            fg=COLORS["text"],
            bg=COLORS["bg_card"],
            selectcolor=COLORS["bg_input"],
            activebackground=COLORS["bg_card"],
            activeforeground=COLORS["text_bright"],
            cursor="hand2",
        ).pack(anchor="w")

        tk.Label(
            card,
            textvariable=self.quick_buy_mode_hint,
            font=("Segoe UI", 8),
            fg=COLORS["text_dim"],
            bg=COLORS["bg_card"],
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

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

    def _on_quick_buy_mode_change(self, *args):
        """Refresh the quick-buy UI when the behavior mode changes."""
        self._refresh_quick_buy_mode_ui()

    def _refresh_quick_buy_mode_ui(self):
        """Apply the selected quick-buy mode to button text and helper copy."""
        mode = self.quick_buy_mode_var.get()
        if mode == "auto_only":
            self.quick_buy_mode_hint.set(
                "Auto Trade Only: ให้บอทประเมิน BUY หนึ่งรอบทันทีโดยใช้กฎ AI, risk sizing และ Auto Buy cap ของบอท"
            )
            if hasattr(self, "quick_buy_btn"):
                self.quick_buy_btn.config(text="🤖 AUTO BUY 1X")
            return

        self.quick_buy_mode_hint.set(
            "Manual Buy + Auto Trade: ซื้อทันทีด้วยจำนวนที่กรอก แล้วเริ่มบอทให้ดูแล position ต่อ"
        )
        if hasattr(self, "quick_buy_btn"):
            self.quick_buy_btn.config(text="📈 BUY NOW")

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
        card = self._make_card(parent, "📜 TRADE HISTORY", "รายการคำสั่งล่าสุดของ session นี้", collapsible=True, collapse_key="trade_history", start_collapsed=True)

        columns = ("time", "type", "price", "pnl")
        columns = ("time", "type", "price", "pnl", "note")
        self.history_tree = ttk.Treeview(card, columns=columns, show="headings",
                                          height=5)

        for col, (text, width) in {
            "time": ("Time", 70),
            "type": ("Type", 88),
            "price": ("Price", 76),
            "pnl": ("P/L %", 64),
            "note": ("Note", 210),
        }.items():
            self.history_tree.heading(col, text=text)
            self.history_tree.column(col, width=width, anchor="center")

        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.history_tree.tag_configure("fee_guard", foreground=COLORS["yellow"])
        self.history_tree.tag_configure("sell", foreground=COLORS["red"])
        self.history_tree.tag_configure("buy", foreground=COLORS["green"])

    # ─── Helpers ──────────────────────────────────────────────

    def _make_card(self, parent, title: str, subtitle: str = "", collapsible: bool = False,
                   collapse_key: str = "", start_collapsed: bool = False) -> tk.Frame:
        """Create a styled card frame."""
        wrapper = tk.Frame(parent, bg=COLORS["bg"])
        wrapper.pack(fill=tk.X, pady=(0, 5))

        card = tk.Frame(wrapper, bg=COLORS["bg_card"], highlightbackground=COLORS["border"],
                        highlightthickness=1)
        card.pack(fill=tk.X)

        header = tk.Frame(card, bg=COLORS["bg_card"])
        header.pack(fill=tk.X, padx=10, pady=(8, 2))

        tk.Label(header, text=title, font=("Segoe UI", 10, "bold"),
                 fg=COLORS["accent"], bg=COLORS["bg_card"],
                 anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        body_wrap = tk.Frame(card, bg=COLORS["bg_card"])
        body_wrap.pack(fill=tk.X)

        if collapsible and collapse_key:
            toggle_button = tk.Button(
                header,
                text="−",
                font=("Segoe UI", 9, "bold"),
                bg=COLORS["bg_input"],
                fg=COLORS["text_bright"],
                activebackground=COLORS["accent2"],
                relief=tk.FLAT,
                cursor="hand2",
                width=2,
                command=lambda key=collapse_key: self._toggle_card_section(key),
            )
            toggle_button.pack(side=tk.RIGHT)
            self._collapsible_cards[collapse_key] = {
                "body_wrap": body_wrap,
                "button": toggle_button,
                "collapsed": False,
            }

        if subtitle:
            tk.Label(
                body_wrap,
                text=subtitle,
                font=("Segoe UI", 8),
                fg=COLORS["text_dim"],
                bg=COLORS["bg_card"],
                anchor="w",
                justify=tk.LEFT,
            ).pack(fill=tk.X, padx=10, pady=(0, 4))

        body = tk.Frame(body_wrap, bg=COLORS["bg_card"])
        body.pack(fill=tk.X)

        if collapsible and collapse_key and start_collapsed:
            self._set_card_collapsed(collapse_key, True)

        return body

    def _toggle_card_section(self, collapse_key: str):
        """Toggle a collapsible card body."""
        state = self._collapsible_cards.get(collapse_key)
        if not state:
            return
        self._set_card_collapsed(collapse_key, not bool(state.get("collapsed")))

    def _set_card_collapsed(self, collapse_key: str, collapsed: bool):
        """Apply collapsed or expanded state to a card section."""
        state = self._collapsible_cards.get(collapse_key)
        if not state:
            return
        body_wrap = state.get("body_wrap")
        button = state.get("button")
        if body_wrap is None or button is None:
            return
        if collapsed:
            body_wrap.pack_forget()
            button.config(text="+")
        else:
            body_wrap.pack(fill=tk.X)
            button.config(text="−")
        state["collapsed"] = collapsed

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
            "profit_cashout_enabled": lambda: bool(self.profit_cashout_enabled.get()),
            "profit_cashout_pct": lambda: float(self.profit_cashout_pct_var.get()),
            "fee_guard_enabled": lambda: bool(self.fee_guard_enabled.get()),
            "fee_guard_max_cost": lambda: float(self.fee_guard_max_cost_var.get()),
            "llm_trade_enabled": lambda: bool(self.llm_trade_enabled.get()),
            "paper_trade_enabled": lambda: bool(self.paper_trade_enabled.get()),
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
            self.profit_cashout_enabled,
            self.profit_cashout_pct_var,
            self.fee_guard_enabled,
            self.fee_guard_max_cost_var,
            self.llm_trade_enabled,
            self.paper_trade_enabled,
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
        self._refresh_quick_start_summary()

    def _on_runtime_field_change(self, *args):
        """Refresh parameter badges when the user edits a runtime field."""
        self._refresh_runtime_badges()
        self._refresh_quick_start_summary()
        if self.bot_running:
            self._schedule_runtime_settings_apply()

    def _schedule_runtime_settings_apply(self):
        """Debounce live runtime config updates while the bot is running."""
        if self._runtime_apply_after_id is not None:
            try:
                self.root.after_cancel(self._runtime_apply_after_id)
            except tk.TclError:
                pass
        self._runtime_apply_after_id = self.root.after(350, self._apply_runtime_settings_live)

    def _apply_runtime_settings_live(self):
        """Apply edited runtime parameters immediately without stopping auto trade."""
        self._runtime_apply_after_id = None
        if not self.bot_running:
            return
        self._sync_runtime_settings(show_popup=False)
        self._refresh_quick_start_summary()

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

    def _sync_llm_trade_status_visual(self):
        """Refresh the LLM Trade label text and color based on runtime state."""
        label = self._ai_metric_labels.get("llm_trade")
        if not label:
            return

        advice = self.last_trade_llm_advice or {}
        if not self.llm_trade_enabled.get():
            label.config(fg=COLORS["text_dim"])
            self.trade_llm_status.set("DISABLED")
            return

        if advice.get("used"):
            confidence = float(advice.get("confidence", 0.0) or 0.0)
            if confidence >= 0.8:
                color = COLORS["green"]
            elif confidence >= 0.68:
                color = COLORS["yellow"]
            else:
                color = COLORS["red"]
            label.config(fg=color)
            self.trade_llm_status.set(f"{advice.get('action', 'HOLD')} {confidence:.0%}")
            return

        if self.llm_boss_advisor.is_enabled():
            label.config(fg=COLORS["yellow"])
            self.trade_llm_status.set("STANDBY")
        else:
            has_key = bool(getattr(self.config.ai, "openai_api_key", ""))
            label.config(fg=COLORS["red"] if not has_key else COLORS["text_dim"])
            self.trade_llm_status.set("NO KEY" if not has_key else "IDLE")

    def _rebuild_llm_advisor(self):
        """Recreate the LLM advisor after runtime enable/disable changes."""
        self.llm_boss_advisor = LLMBossAdvisor(self.config.ai, self.config.trading, self.logger)

    @staticmethod
    def _truncate_ui_text(text: str, max_length: int = 72) -> str:
        """Keep runtime status text compact enough for the dashboard."""
        clean = " ".join(str(text or "").split())
        if len(clean) <= max_length:
            return clean
        return clean[: max_length - 3].rstrip() + "..."

    def _format_reason_list(self, reasons, max_items: int = 3, max_length: int = 156) -> str:
        """Flatten multiple strategy reasons into a short readable summary."""
        normalized = [
            self._translate_decision_reason(" ".join(str(reason or "").split()))
            for reason in (reasons or [])
            if str(reason or "").strip()
        ]
        if not normalized:
            return ""

        text = " | ".join(normalized[:max_items])
        if len(normalized) > max_items:
            text = f"{text} | ..."
        return self._truncate_ui_text(text, max_length)

    def _format_decision_reason_block(self, reasons, max_items: int = 3, max_length: int = 180) -> str:
        """Format reasons for the dashboard as short Thai bullet lines."""
        normalized = []
        seen = set()
        for reason in (reasons or []):
            translated = self._translate_decision_reason(" ".join(str(reason or "").split()))
            if not translated:
                continue
            dedupe_key = translated.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(self._truncate_ui_text(translated, max_length))
            if len(normalized) >= max_items:
                break

        if not normalized:
            return ""
        return "\n".join(f"- {item}" for item in normalized)

    def _join_decision_lines(self, *parts: str, max_length: int = 220) -> str:
        """Join multiple dashboard detail lines while skipping blanks."""
        lines = [str(part).strip() for part in parts if str(part or "").strip()]
        return self._truncate_ui_text("\n".join(lines), max_length)

    def _translate_decision_reason(self, reason: str) -> str:
        """Translate strategy and risk reasons into Thai for the Decision Insight panel."""
        text = " ".join(str(reason or "").split())
        if not text:
            return ""

        if text.startswith("LLM:"):
            llm_reason = text.split(":", 1)[1].strip()
            return f"LLM แนะนำ: {llm_reason}" if llm_reason else "LLM แนะนำ"

        patterns = [
            (r"^RSI oversold \((.+) < (.+)\)$", lambda m: f"RSI อยู่ในเขตขายมากเกินไป ({m.group(1)} < {m.group(2)})"),
            (r"^Price \((.+)\) below EMA21 \((.+)\)$", lambda m: f"ราคาปัจจุบัน ({m.group(1)}) ต่ำกว่า EMA21 ({m.group(2)})"),
            (r"^Price is holding near support (.+) \((.+)% above support\)$", lambda m: f"ราคายืนใกล้แนวรับ {m.group(1)} ({m.group(2).replace('above support', 'เหนือแนวรับ')})"),
            (r"^Price broke resistance (.+) and is entering momentum continuation$", lambda m: f"ราคาทะลุแนวต้าน {m.group(1)} และเริ่มต่อโมเมนตัมขาขึ้น"),
            (r"^AI predicts UP \(confidence: (.+)\)$", lambda m: f"AI มองว่าราคาจะขึ้น (ความมั่นใจ: {m.group(1)})"),
            (r"^AI expects upside (.+)%$", lambda m: f"AI คาดว่าราคามี upside {m.group(1)}%"),
            (r"^Price entered buy zone (.+)-(.+)$", lambda m: f"ราคาเข้าโซนซื้อ {m.group(1)}-{m.group(2)}"),
            (r"^RSI is within buy-zone limit \((.+) <= (.+)\)$", lambda m: f"RSI อยู่ในเกณฑ์โซนซื้อ ({m.group(1)} <= {m.group(2)})"),
            (r"^Volume support confirmed \((.+)\)$", lambda m: f"ปริมาณซื้อขายสนับสนุน ({m.group(1)})"),
            (r"^Volume too light \((.+)\), skipping weak setup$", lambda m: f"ปริมาณซื้อขายเบาเกินไป ({m.group(1)}) จึงยังข้ามจังหวะอ่อนนี้"),
            (r"^Daily loss limit reached: (.+) >= (.+) THB$", lambda m: f"ขาดทุนสะสมวันนี้ถึงลิมิตแล้ว: {m.group(1)} >= {m.group(2)} THB"),
            (r"^Daily trade limit reached: (.+) >= (.+)$", lambda m: f"จำนวนเทรดวันนี้ถึงลิมิตแล้ว: {m.group(1)} >= {m.group(2)}"),
            (r"^Consecutive loss limit reached: (.+) >= (.+)$", lambda m: f"แพ้ติดต่อกันถึงลิมิตแล้ว: {m.group(1)} >= {m.group(2)}"),
            (r"^Max open positions reached: (.+) >= (.+)$", lambda m: f"จำนวน position เปิดพร้อมกันถึงลิมิตแล้ว: {m.group(1)} >= {m.group(2)}"),
            (r"^Insufficient balance: (.+) THB$", lambda m: f"ยอดเงินไม่พอ: {m.group(1)} THB"),
            (r"^Small-position fee guard active: holding (.+) because loss (.+) THB \((.+)%\) is still within fee-noise buffer (.+) THB$", lambda m: f"หน่วงขายไม้เล็ก {m.group(1)} เพราะขาดทุน {m.group(2)} THB ({m.group(3)}%) ยังอยู่ในช่วง fee noise {m.group(4)} THB"),
        ]

        for pattern, formatter in patterns:
            match = re.match(pattern, text)
            if match:
                return formatter(match)

        literal_map = {
            "Trend filter passed: EMA9 > EMA21 > EMA50": "เทรนด์ผ่านเงื่อนไข: EMA9 > EMA21 > EMA50",
            "Short-term trend still positive": "เทรนด์ระยะสั้นยังเป็นบวก",
            "MACD bullish crossover": "MACD ตัดขึ้น",
            "Price below Bollinger lower band": "ราคาต่ำกว่าเส้นล่าง Bollinger Band",
            "Auto-buy triggered because price is inside the configured buy zone": "เข้าเงื่อนไขซื้ออัตโนมัติ เพราะราคาอยู่ในโซนซื้อที่ตั้งไว้",
            "Dip-buy setup detected: price weakened and AI allows accumulation": "เข้าเงื่อนไขซื้อย่อ: ราคาอ่อนตัวและ AI อนุญาตให้สะสม",
            "Trend-buy setup detected: price is climbing and AI allows momentum entry": "เข้าเงื่อนไขซื้อตามเทรนด์: ราคากำลังขึ้นและ AI อนุญาตให้เข้าตามโมเมนตัม",
            "Uptrend pullback entry: AI waits for support, then buys on strength": "เข้าเงื่อนไขย่อในขาขึ้น: AI รอรับแถวแนวรับแล้วค่อยซื้อเมื่อแรงกลับมา",
            "Uptrend breakout entry confirmed above resistance": "เข้าเงื่อนไขทะลุแนวต้านขาขึ้นหลังผ่านแนวต้านแล้ว",
            "Early recovery entry confirmed: market structure improved and AI found a low-chase entry": "เข้าเงื่อนไขฟื้นตัวระยะแรก: โครงสร้างตลาดเริ่มดีขึ้นและ AI หาโซนเข้าที่ไม่ไล่ราคา",
            "Counter-trend reversal allowed: strong AI rebound from support": "เข้าเงื่อนไขกลับตัวสวนเทรนด์: AI เห็นแรงเด้งจากแนวรับชัดเจน",
            "Downtrend protection active: waiting for EMA9 reclaim, bullish MACD, and strong AI volume confirmation": "ระบบกันตลาดขาลงกำลังทำงาน: รอราคา reclaim EMA9, MACD กลับตัว และปริมาณ/AI แข็งแรงก่อน",
            "Downtrend recovery confirmation passed: price reclaimed EMA9 with strong volume": "ยืนยันการฟื้นตัวในตลาดลงแล้ว: ราคา reclaim EMA9 พร้อมปริมาณซื้อขายหนุน",
            "Downtrend protection active after recent losses": "ระบบกันขาดทุนหยุดซื้อชั่วคราว เพราะยังอยู่ใน downtrend หลังแพ้ติดกัน",
            "AI is waiting for either a dip-buy near support or a trend-buy on strength": "AI ยังรอให้เกิดจังหวะซื้อย่อใกล้แนวรับ หรือซื้อเมื่อราคาแข็งแรงตามเทรนด์",
        }
        return literal_map.get(text, text)

    def _build_buy_wait_hint(self, price: float, signals: Dict, buy_signal: Dict) -> str:
        """Explain the next price area the bot is waiting for before buying."""
        ai_entry_plan = buy_signal.get("ai_entry_plan", {}) or {}
        buy_zone = buy_signal.get("buy_zone", {}) or {}
        zone_low = float(buy_zone.get("zone_low", 0.0) or 0.0)
        zone_high = float(buy_zone.get("zone_high", 0.0) or 0.0)
        support_level = float(signals.get("support_level", 0.0) or 0.0)
        resistance_level = float(signals.get("resistance_level", 0.0) or 0.0)
        preferred_entry = float(ai_entry_plan.get("preferred_entry", 0.0) or 0.0)
        trigger_price = float(ai_entry_plan.get("trigger_price", 0.0) or 0.0)
        chase_limit = float(ai_entry_plan.get("chase_limit", 0.0) or 0.0)
        plan_label = str(ai_entry_plan.get("label", "") or "")

        hints = []
        if preferred_entry > 0 and chase_limit > 0:
            if price > chase_limit:
                hints.append(f"AI รอราคาเข้ากลับโซน {preferred_entry:,.2f}-{chase_limit:,.2f}")
            elif trigger_price > 0 and price < trigger_price:
                hints.append(f"AI รอให้ราคายืนเหนือ {trigger_price:,.2f} ก่อนเข้า")
            elif plan_label:
                hints.append(f"AI มองโซนเข้า {preferred_entry:,.2f} | {plan_label}")

        if zone_low > 0 and zone_high > 0:
            if price > zone_high:
                hints.append(f"รอราคาเข้าโซนซื้อ {zone_low:,.2f}-{zone_high:,.2f}")
            elif price < zone_low:
                hints.append(f"รอราคายืนกลับเหนือ {zone_low:,.2f}")

        if support_level > 0 and price > support_level:
            hints.append(f"รอรับใกล้แนวรับ {support_level:,.2f}")

        if resistance_level > 0:
            breakout_price = resistance_level * (1 + self.config.trading.resistance_breakout_pct / 100)
            if price < breakout_price:
                hints.append(f"หรือรอทะลุแนวต้านเหนือ {breakout_price:,.2f}")

        return self._truncate_ui_text(" | ".join(hints[:2]), 120)

    def _build_decision_snapshot(self, symbol: str, current_price: float, balance: float,
                                 positions, signals: Dict, ai_prediction: Dict,
                                 action_name: str, risk_check: Optional[Dict] = None) -> Dict:
        """Build a compact user-facing summary of why the bot buys, waits, or holds."""
        if self.boss_waiting_recovery or (self.reentry_waiting and self.reentry_symbol == symbol):
            snapshot = self._get_boss_runtime_snapshot(current_price)
            return {
                "status": snapshot["title"],
                "detail": snapshot["detail"],
                "color": snapshot["color"],
            }

        buy_signal = self.strategy.should_buy(signals, ai_prediction)
        reason_text = self._format_reason_list(buy_signal.get("reasons", []))
        reason_block = self._format_decision_reason_block(buy_signal.get("reasons", []), max_items=3, max_length=104)
        wait_hint = self._build_buy_wait_hint(current_price, signals, buy_signal)
        ai_entry_plan = buy_signal.get("ai_entry_plan", {}) or {}
        preferred_entry = float(ai_entry_plan.get("preferred_entry", 0.0) or 0.0)
        score_text = f"คะแนน {float(buy_signal.get('score', 0.0) or 0.0):.2f}/{self.config.trading.min_buy_signal_score:.2f}"

        if action_name == "BUY":
            detail = self._join_decision_lines(
                f"ซื้อที่ {current_price:,.2f}",
                f"สรุป: {score_text}",
                reason_block or reason_text,
            )
            if preferred_entry > 0:
                detail = self._join_decision_lines(detail, f"AI มองจุดเข้าแถว {preferred_entry:,.2f}")
            return {"status": "📈 ซื้อ", "detail": detail, "color": COLORS["green"]}

        if action_name in {"AUTO_REBUY", "BOSS_BUYBACK"}:
            return {
                "status": "🔁 ซื้อคืน",
                "detail": f"ซื้อคืนแล้วที่ {current_price:,.2f} หลังราคาฟื้นตามเงื่อนไข",
                "color": COLORS["green"],
            }

        if action_name in {"SELL", "STOP_LOSS", "TAKE_PROFIT", "AI_CUTLOSS", "AI_TAKE_PROFIT", "AI_BOSS_CUTLOSS", "PROFIT_CASHOUT", "AI_PROFIT_CASHOUT"}:
            action_label_map = {
                "SELL": "ขาย",
                "STOP_LOSS": "ตัดขาดทุน",
                "TAKE_PROFIT": "ทำกำไร",
                "AI_CUTLOSS": "AI ตัดขาดทุน",
                "AI_TAKE_PROFIT": "AI ทำกำไร",
                "AI_BOSS_CUTLOSS": "Boss AI ตัดขาดทุน",
                "PROFIT_CASHOUT": "ถอนกำไร",
                "AI_PROFIT_CASHOUT": "AI ถอนกำไร",
            }
            return {
                "status": f"📉 {action_label_map.get(action_name, action_name)}",
                "detail": (
                    f"ถอนกำไรบางส่วนที่ {current_price:,.2f} แล้ว เหลือเงินต้นไว้ในเหรียญ"
                    if "CASHOUT" in action_name else
                    f"ปิดสถานะที่ {current_price:,.2f} แล้ว รอระบบประเมินจังหวะรอบถัดไป"
                ),
                "color": COLORS["red"] if "LOSS" in action_name or "SELL" in action_name else COLORS["yellow"],
            }

        if action_name in {"INVALID_AUTO_BUY", "INSUFFICIENT_AUTO_BUY"}:
            detail = self._join_decision_lines(
                "ยังไม่ซื้อ",
                f"สรุป: {reason_text or score_text}",
                wait_hint,
                reason_block,
            )
            return {"status": "⚠️ ซื้อไม่ได้", "detail": detail, "color": COLORS["yellow"]}

        if positions:
            position_summary = self._get_runtime_position_summary(current_price)
            detail = position_summary["label"]
            if buy_signal.get("should_buy"):
                detail = self._join_decision_lines(
                    "มีสถานะค้างอยู่แล้ว",
                    "จะซื้อเพิ่มเมื่อกำไรเฉลี่ยเกิน +1.00%",
                    detail,
                )
            elif reason_text:
                detail = self._join_decision_lines("ถืออยู่", detail, reason_block or reason_text)
            return {"status": "🟡 ถือต่อ", "detail": detail, "color": COLORS["yellow"]}

        if risk_check and not risk_check.get("allowed", True):
            risk_text = self._format_reason_list(risk_check.get("reasons", []), max_items=2, max_length=140)
            detail = self._join_decision_lines(
                "พักการซื้อ",
                risk_text or "ติดข้อจำกัดความเสี่ยง",
                wait_hint,
                reason_block,
            )
            return {"status": "⏸️ พักซื้อ", "detail": detail, "color": COLORS["yellow"]}

        detail = self._join_decision_lines(
            "ยังไม่ซื้อ",
            f"สรุป: {score_text}",
            wait_hint,
            reason_block or reason_text,
        )
        return {"status": "⏳ รอซื้อ", "detail": detail, "color": COLORS["text_bright"]}

    def _apply_decision_snapshot(self, snapshot: Dict):
        """Apply the current decision summary to the dashboard widgets."""
        self.bot_decision_status.set(snapshot.get("status", "—"))
        self.bot_decision_detail.set(snapshot.get("detail", ""))
        if hasattr(self, "bot_decision_status_label"):
            self.bot_decision_status_label.config(fg=snapshot.get("color", COLORS["text_bright"]))

    def _format_llm_status_text(self, advice: Dict, compact: bool = False) -> str:
        """Build a short UI-friendly LLM status line."""
        if not self.llm_trade_enabled.get():
            return "LLM: disabled"

        if advice.get("used"):
            text = f"LLM {advice.get('action', 'HOLD')} {float(advice.get('confidence', 0.0) or 0.0):.0%}"
            reason = self._truncate_ui_text(advice.get("reason", ""), 44 if compact else 64)
            return f"{text} | {reason}" if reason else text

        status = advice.get("status", "unavailable")
        label_map = {
            "quota": "LLM quota exceeded",
            "auth": "LLM invalid API key",
            "rate_limit": "LLM rate limited",
            "timeout": "LLM timeout",
            "backoff": "LLM standby",
            "error": "LLM unavailable",
            "unavailable": "LLM standby",
        }
        message = label_map.get(status, "LLM standby")
        if not compact:
            reason = advice.get("reason", "")
            if reason and reason not in message:
                message = f"{message} | {self._truncate_ui_text(reason, 44)}"
        return message

    def _is_paper_trade_mode(self) -> bool:
        """Return whether the bot should trade against the simulated portfolio."""
        return bool(self.paper_trade_enabled.get())

    def _get_live_thb_balance(self) -> float:
        """Read the currently cached live THB balance."""
        thb_info = self.wallet_balances.get("THB", {})
        return float(thb_info.get("available", 0.0) or 0.0)

    def _resolve_paper_start_balance(self, live_balance: float = 0.0) -> float:
        """Resolve the starting simulated THB balance from config or live wallet."""
        configured_balance = float(self.config.trading.paper_trade_start_balance_thb or 0.0)
        if configured_balance > 0:
            return configured_balance
        return max(float(live_balance or 0.0), 0.0)

    def _reset_runtime_trade_state(self):
        """Clear tracked positions and recovery state when switching trading modes."""
        self.strategy.positions = []
        self._clear_boss_recovery_state()
        self._clear_auto_reentry(self.config.trading.symbol)
        self.last_trade_llm_advice = {}
        self.last_boss_llm_advice = {}

    def _ensure_paper_balance(self, live_balance: float = 0.0, force_reset: bool = False) -> float:
        """Initialize or refresh the simulated THB balance."""
        if force_reset or self.paper_balance_thb <= 0:
            self.paper_balance_thb = self._resolve_paper_start_balance(live_balance)
        return self.paper_balance_thb

    def _get_runtime_trade_balance(self, live_balance: Optional[float] = None) -> float:
        """Return the balance source used by trading decisions."""
        if self._is_paper_trade_mode():
            return float(self.paper_balance_thb or 0.0)
        if live_balance is not None:
            return float(live_balance or 0.0)
        return self._get_live_thb_balance()

    def _on_paper_trade_toggle(self, *args):
        """Prevent mixing live and simulated positions when the mode changes."""
        if self._paper_toggle_guard:
            return

        enabled = bool(self.paper_trade_enabled.get())
        previous_enabled = self._paper_trade_enabled_last
        if enabled == previous_enabled:
            return

        if self.bot_running:
            self._paper_toggle_guard = True
            self.paper_trade_enabled.set(previous_enabled)
            self._paper_toggle_guard = False
            self._log("เปลี่ยนโหมด Live/Paper ระหว่างบอทรันไม่ได้ กรุณาหยุดบอทก่อน", "WARN")
            return

        self._paper_trade_enabled_last = enabled
        self.config.trading.paper_trade_enabled = enabled
        live_balance = self._get_live_thb_balance()
        self._reset_runtime_trade_state()

        if enabled:
            seeded_balance = self._ensure_paper_balance(live_balance, force_reset=True)
            self._log(
                f"🧪 Paper Trading: ON | เงินจำลองเริ่มต้น ฿{seeded_balance:,.2f} | จะไม่ส่งคำสั่งจริงไป Bitkub",
                "WARN",
            )
        else:
            self.paper_balance_thb = self._resolve_paper_start_balance(live_balance)
            self._log("🧪 Paper Trading: OFF | กลับไปใช้ wallet จริงสำหรับคำสั่งถัดไป", "INFO")

        display_balance = self._get_runtime_trade_balance(live_balance)
        try:
            current_price = float(self.current_price.get().replace(",", "").split()[0])
        except (ValueError, IndexError):
            current_price = 0.0
        self._update_balance_display(display_balance, current_price)
        self._update_positions_display(current_price)
        self._update_pnl_display(current_price)
        self._refresh_runtime_badges()
        self._refresh_quick_start_summary()

    def _execute_paper_buy(self, symbol: str, amount_thb: float, price: float,
                           history_type: str = "P-BUY") -> Optional[Dict[str, float]]:
        """Execute a simulated buy without sending an exchange order."""
        if price <= 0:
            self.root.after(0, self._log, "❌ Paper buy failed: invalid market price", "ERROR")
            return None

        available_balance = self._ensure_paper_balance(self._get_live_thb_balance())
        if amount_thb > available_balance:
            self.root.after(
                0,
                self._log,
                f"❌ Paper buy failed: เงินจำลองไม่พอ (ต้องการ ฿{amount_thb:,.2f} | มี ฿{available_balance:,.2f})",
                "ERROR",
            )
            return None

        buy_fee_rate = max(float(getattr(self.config.trading, "buy_fee_rate", 0.0027) or 0.0), 0.0)
        crypto_amount = (amount_thb * (1 - buy_fee_rate)) / price
        self.paper_balance_thb = max(available_balance - amount_thb, 0.0)
        self.strategy.add_position(
            symbol,
            price,
            crypto_amount,
            cost_thb=amount_thb,
            entry_fee_thb=max(amount_thb - (price * crypto_amount), 0.0),
        )
        self._clear_boss_recovery_state()
        self._clear_auto_reentry(symbol)
        self.root.after(0, self._add_history_entry, history_type, price, 0)
        self.root.after(0, self._update_pnl_display, price)
        self.root.after(0, self._update_balance_display, self.paper_balance_thb, price)
        return {
            "rate": price,
            "amount": crypto_amount,
            "cost_thb": amount_thb,
        }

    def _execute_paper_sell(self, position: Position, price: float, reason: str,
                            arm_auto_reentry: bool = True,
                            history_type: str = "P-SELL",
                            keep_principal: bool = False) -> Optional[Dict]:
        """Execute a simulated sell without sending an exchange order."""
        if price <= 0:
            self.root.after(0, self._log, "❌ Paper sell failed: invalid market price", "ERROR")
            return None

        if keep_principal:
            cashout_plan = self.strategy.evaluate_profit_cashout(position, price)
            if cashout_plan["should_cashout"]:
                record = self.strategy.cash_out_profit(
                    position,
                    price,
                    float(cashout_plan["sell_amount"] or 0.0),
                    reason,
                )
            else:
                record = self.strategy.close_position(position, price, reason)
        else:
            record = self.strategy.close_position(position, price, reason)
        if not record:
            return None

        self.paper_balance_thb += float(record.get("net_exit_value_thb", 0.0) or 0.0)
        self.risk_manager.record_trade_result(record["profit_thb"])
        if self.bot_running and arm_auto_reentry and reason != "MANUAL_SELL" and not record.get("partial_close"):
            self._arm_auto_reentry(
                position.symbol,
                price,
                float(record.get("net_exit_value_thb", 0.0) or 0.0),
                reason,
            )
        self.bot_total_realized_pnl += record["profit_thb"]
        self.bot_total_trades += 1
        if record["profit_thb"] >= 0:
            self.bot_win_trades += 1
        else:
            self.bot_lose_trades += 1
        effective_history_type = "P-CASHOUT" if record.get("partial_close") else history_type
        self.root.after(0, self._add_history_entry, effective_history_type, price, record["profit_pct"])
        self.root.after(0, self._update_pnl_display, price)
        self.root.after(0, self._update_balance_display, self.paper_balance_thb, price)
        return record

    def _on_llm_trade_toggle(self, *args):
        """Handle runtime enable/disable of the LLM trade advisor."""
        self.config.ai.llm_enabled = bool(self.llm_trade_enabled.get())
        self.last_trade_llm_advice = {}
        self.last_boss_llm_advice = {}
        self._rebuild_llm_advisor()
        self._sync_llm_trade_status_visual()
        if self.llm_trade_enabled.get():
            if getattr(self.config.ai, "openai_api_key", ""):
                self._log("🧠 LLM Trade Advisor: ON", "SUCCESS")
            else:
                self._log("🧠 LLM Trade Advisor เปิดแล้ว แต่ยังไม่มี OPENAI_API_KEY", "WARN")
        else:
            self.boss_llm_status.set("LLM: disabled")
            self.trade_llm_status.set("DISABLED")
            self._log("🧠 LLM Trade Advisor: OFF", "INFO")

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
        self._refresh_quick_start_summary()

    def _handle_connect_error(self, error_message: str):
        """Show connection errors on the Tkinter thread."""
        self._connect_in_progress = False
        self.is_connected = False
        self.connect_btn.config(state=tk.NORMAL, text="🔗 Connect & Load Wallet", bg=COLORS["accent"])
        self.conn_status_label.config(text=f"❌ {error_message}", fg=COLORS["red"])
        self._log(f"Connection error: {error_message}", "ERROR")
        self._refresh_quick_start_summary()
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
        runtime_balance = self._get_runtime_trade_balance(thb_avail)
        self.balance_thb.set(f"{runtime_balance:,.2f} THB")

        for item in self.wallet_tree.get_children():
            self.wallet_tree.delete(item)

        for row in snapshot.get("wallet_rows", []):
            self.wallet_tree.insert("", "end", values=row)

        if self._is_paper_trade_mode():
            self._update_balance_display(runtime_balance, 0.0)
        else:
            self.total_value.set(f"{snapshot.get('total_value_thb', 0):,.2f} THB")
        self.symbol_combo.config(values=snapshot.get("all_symbols", []))
        if not self._is_paper_trade_mode():
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
        self._refresh_quick_start_summary()

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
        """Handle quick-buy according to the selected execution mode."""
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

        mode = self.quick_buy_mode_var.get()
        if mode == "auto_only":
            runtime_settings = self._sync_runtime_settings(show_popup=True)
            if runtime_settings is None:
                return

            confirm = messagebox.askyesno(
                "Confirm Auto Buy",
                (
                    f"ให้บอทประเมิน BUY {self.config.trading.symbol} หนึ่งรอบทันที\n"
                    f"วงเงินสูงสุด: ฿{amount_thb:,.2f}\n"
                    f"โหมด: {'Paper Auto Buy' if self._is_paper_trade_mode() else 'Live Auto Buy'}"
                ),
            )
            if not confirm:
                return

            self._log(
                f"🤖 Quick Buy โหมด Auto Trade Only | กำลังให้บอทประเมิน {self.config.trading.symbol} หนึ่งรอบ (budget ฿{amount_thb:,.2f})...",
                "AI",
            )
            threading.Thread(
                target=self._do_quick_buy_auto_only,
                args=(self.config.trading.symbol, amount_thb),
                daemon=True,
            ).start()
            return

        thb_avail = self._get_runtime_trade_balance()
        if amount_thb > thb_avail:
            messagebox.showwarning("Insufficient Balance",
                f"ยอด THB ไม่พอ\nต้องการ: ฿{amount_thb:,.2f}\nมี: ฿{thb_avail:,.2f}")
            return

        symbol = self.config.trading.symbol
        trade_mode = "Paper Buy" if self._is_paper_trade_mode() else "Live Buy"
        action_text = "ยืนยันจำลองซื้อ" if self._is_paper_trade_mode() else "ยืนยันซื้อ"
        confirm = messagebox.askyesno("Confirm Buy",
            f"{action_text} {symbol} ด้วยจำนวน ฿{amount_thb:,.2f} THB?\nโหมด: {trade_mode}")
        if not confirm:
            return

        if self._is_paper_trade_mode():
            self._log(f"🧪 กำลังจำลองซื้อ {symbol} จำนวน ฿{amount_thb:,.2f}...", "TRADE")
        else:
            self._log(f"📈 กำลังซื้อ {symbol} จำนวน ฿{amount_thb:,.2f}...", "TRADE")
        threading.Thread(target=self._do_quick_buy,
                         args=(symbol, amount_thb), daemon=True).start()

    def _do_quick_buy(self, symbol: str, amount_thb: float):
        """Execute quick buy in background thread."""
        try:
            # Get current price before order
            ticker = self.client.get_ticker(symbol)
            est_price = ticker.get("last", 0) if ticker else 0

            if self._is_paper_trade_mode():
                paper_order = self._execute_paper_buy(symbol, amount_thb, est_price)
                if not paper_order:
                    return

                exec_price = paper_order["rate"]
                crypto_amount = paper_order["amount"]
                self.root.after(
                    0,
                    self._log,
                    f"✅ จำลองซื้อสำเร็จ! {symbol} @ {exec_price:,.2f} THB | จำนวน: {crypto_amount:.8f} | เงินจำลองคงเหลือ: ฿{self.paper_balance_thb:,.2f}",
                    "SUCCESS",
                    self._refresh_quick_start_summary()
                )

                if not self.bot_running:
                    self.root.after(0, self._auto_start_bot_after_buy)
                return

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
            position = self.strategy.add_position(
                symbol,
                exec_price,
                crypto_amount,
                cost_thb=float(order.get("cost", amount_thb) or amount_thb),
            )

            self.root.after(0, self._log,
                f"✅ ซื้อสำเร็จ! {symbol} @ {exec_price:,.2f} THB | "
                f"จำนวน: {crypto_amount:.8f} | "
                f"SL: {position.stop_loss_price:,.2f} | "
                f"TP: {position.take_profit_price:,.2f}",
                "SUCCESS")
            self.root.after(0, self._add_history_entry, "BUY", exec_price, 0)
            self.root.after(0, self._load_wallet)
            self.root.after(0, self._update_pnl_display, exec_price)

            # Auto-start bot immediately so the open position is managed by auto trade.
            if not self.bot_running:
                self.root.after(0, self._auto_start_bot_after_buy)

        except Exception as e:
            self.root.after(0, self._log, f"❌ Buy error: {e}", "ERROR")

    def _do_quick_buy_auto_only(self, symbol: str, budget_thb: float):
        """Run one immediate auto-trade buy decision without starting the bot loop."""
        if not self.data_collector or not self.client:
            self.root.after(0, self._log, "Quick auto buy ใช้งานไม่ได้ เพราะยังไม่มี market data collector", "ERROR")
            return

        try:
            with self._trade_cycle_lock:
                self.cycle_count += 1
                df = self.data_collector.fetch_ohlcv(symbol)
                if df is None or df.empty:
                    self.root.after(0, self._log, "No market data available for quick auto buy", "WARN")
                    return

                df_ind = self.indicator_engine.add_all_indicators(df)
                signals = self.indicator_engine.get_signal_summary(df_ind)
                current_price = float(signals.get("price", 0.0) or 0.0)
                if current_price <= 0:
                    self.root.after(0, self._log, "Quick auto buy ข้ามรอบ เพราะราคาปัจจุบันไม่ถูกต้อง", "WARN")
                    return

                ai_prediction = self._predict_ai(df_ind)
                self.root.after(0, self._update_gui_data, signals, ai_prediction)

                balance_detail = self.client.get_balance_detail()
                self.wallet_balances = balance_detail
                thb_info = balance_detail.get("THB", {})
                live_balance = float(thb_info.get("available", 0.0) or 0.0)
                self.last_thb_balance = live_balance
                positions = self.strategy.get_open_positions(symbol)
                balance = self._ensure_paper_balance(live_balance) if self._is_paper_trade_mode() else live_balance

                risk_check = self.risk_manager.can_trade(balance, positions, signals, ai_prediction)
                action = ""
                if not risk_check["allowed"]:
                    guard_reason = risk_check.get("reason", "Risk guard active")
                    self.root.after(0, self._log, f"🛡️ Quick auto buy ถูกบล็อก | {guard_reason}", "WARN")
                else:
                    action = self._check_buy_signal_once(
                        symbol,
                        signals,
                        ai_prediction,
                        balance,
                        current_price,
                        positions,
                        configured_amount_override=budget_thb,
                        force_feedback=True,
                    )

                positions = self.strategy.get_open_positions(symbol)

                decision_snapshot = self._build_decision_snapshot(
                    symbol,
                    current_price,
                    balance,
                    positions,
                    signals,
                    ai_prediction,
                    action or "HOLD",
                    risk_check=risk_check,
                )

            self.root.after(0, lambda: self.bot_cycles_str.set(str(self.cycle_count)))
            self.root.after(0, lambda snapshot=decision_snapshot: self._apply_decision_snapshot(snapshot))
            self.root.after(0, self._update_positions_display, current_price)
            self.root.after(0, self._update_balance_display, balance, current_price)
            self.root.after(0, self._update_pnl_display, current_price)
            self.root.after(0, self._update_risk_display, balance)
            self.root.after(0, self._load_wallet)
            self.root.after(0, self._refresh_quick_start_summary)

            if not action:
                self.root.after(0, self._log, "🤖 Quick auto buy จบรอบแล้ว | ยังไม่พบจังหวะซื้อ", "INFO")
        except Exception as e:
            self.root.after(0, self._log, f"❌ Quick auto buy error: {e}", "ERROR")

    def _auto_start_bot_after_buy(self):
        """Start auto trade immediately after a successful quick buy."""
        if self.bot_running:
            return
        self._log("🤖 Quick Buy สำเร็จ ระบบกำลังเริ่ม auto trade ทันที...", "INFO")
        self._start_bot()

    def _quick_sell(self):
        """Manual sell: sell all holdings of current coin."""
        if not self.is_connected or not self.client:
            messagebox.showwarning("Not Connected", "กรุณาเชื่อมต่อ Exchange ก่อน")
            return

        symbol = self.config.trading.symbol
        coin = symbol.split("_")[0]
        if self._is_paper_trade_mode():
            coin_avail = sum(pos.amount for pos in self.strategy.get_open_positions(symbol))
        else:
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
            f"มูลค่าประมาณ: ฿{est_value:,.2f}\n"
            f"โหมด: {'Paper Sell' if self._is_paper_trade_mode() else 'Live Sell'}")
        if not confirm:
            return

        if self._is_paper_trade_mode():
            self._log(f"🧪 กำลังจำลองขาย {coin} ทั้งหมด...", "TRADE")
        else:
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

            if self._is_paper_trade_mode():
                positions = self.strategy.get_open_positions(symbol)
                for pos in list(positions):
                    record = self._execute_paper_sell(pos, current_price, "MANUAL_SELL", arm_auto_reentry=False)
                    if record:
                        self.root.after(
                            0,
                            self._log,
                            f"✅ จำลองขายสำเร็จ! @ {current_price:,.2f} THB | P/L: {record['profit_pct']:+.2f}% | เงินจำลองคงเหลือ: ฿{self.paper_balance_thb:,.2f}",
                            "SUCCESS",
                        )
                if not positions:
                    self.root.after(0, self._add_history_entry, "P-SELL", current_price, 0)
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
            total_position_amount = sum(float(pos.amount or 0.0) for pos in positions)
            total_received_thb = float(order.get("received", 0.0) or 0.0)
            for pos in list(positions):
                position_received_thb = 0.0
                if total_received_thb > 0 and total_position_amount > 0:
                    position_received_thb = total_received_thb * (float(pos.amount or 0.0) / total_position_amount)
                record = self.strategy.close_position(
                    pos,
                    exit_price,
                    "MANUAL_SELL",
                    net_exit_value_thb=position_received_thb if position_received_thb > 0 else None,
                )
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
            paper_mode = self._is_paper_trade_mode()
            paper_start_balance = self._resolve_paper_start_balance(thb_avail)

            if not paper_mode and thb_avail < 10 and not coin_total:
                self.root.after(
                    0,
                    self._abort_bot_start,
                    "No Balance",
                    f"ไม่มียอดเงิน THB หรือเหรียญ {coin} ในบัญชี\nTHB Available: ฿{thb_avail:,.2f}",
                    "warning",
                )
                return
            if paper_mode and paper_start_balance < 10 and not self.strategy.get_open_positions(symbol):
                self.root.after(
                    0,
                    self._abort_bot_start,
                    "Paper Balance Too Low",
                    f"Paper Trading ต้องมียอดเงินจำลองอย่างน้อย 10 THB\nยอดตั้งต้น: ฿{paper_start_balance:,.2f}",
                    "warning",
                )
                return

            ticker = self.client.get_ticker(symbol)
            current_price = ticker.get("last", 0) if ticker else 0
            model_messages = []
            reload_state = self._reload_saved_ai_models()
            if reload_state["lstm"] == "saved":
                model_messages.append(("AI", "LSTM model loaded"))
            else:
                model_messages.append(("WARN", "LSTM model not found (using signals only)"))

            if reload_state["rl"] == "saved":
                model_messages.append(("AI", "RL model loaded"))
            else:
                model_messages.append(("WARN", "RL model not found"))

            startup_context = {
                "wallet_snapshot": wallet_snapshot,
                "coin": coin,
                "coin_total": coin_total,
                "thb_avail": thb_avail,
                "paper_mode": paper_mode,
                "paper_start_balance": paper_start_balance,
                "initial_portfolio_value": (
                    paper_start_balance + sum(
                        self.strategy.estimate_exit_value_thb(position.amount, current_price)
                        for position in self.strategy.get_open_positions(symbol)
                    )
                ) if paper_mode else thb_avail + self.strategy.estimate_exit_value_thb(coin_total, current_price),
                "model_messages": model_messages,
                "auto_buy_amount": runtime_settings["auto_buy_amount"],
            }
            self.root.after(0, self._finish_start_bot, startup_context)
        except Exception as e:
            self.root.after(0, self._abort_bot_start, "Start Error", str(e), "error")

    def _finish_start_bot(self, startup_context: Dict[str, object]):
        """Finalize bot startup on the Tkinter thread."""
        self._bot_start_in_progress = False
        self._apply_wallet_snapshot(startup_context["wallet_snapshot"], log_success=False)
        if startup_context.get("paper_mode"):
            if not self.strategy.get_open_positions(self.config.trading.symbol):
                self._ensure_paper_balance(startup_context.get("thb_avail", 0.0), force_reset=True)
            self._log(
                f"🧪 เริ่ม Bot ในโหมด Paper Trading | เงินจำลอง: ฿{self.paper_balance_thb:,.2f}",
                "WARN",
            )
        else:
            self._sync_selected_position_from_wallet(startup_context["wallet_snapshot"], log_sync=True)

        self.initial_portfolio_value = startup_context["initial_portfolio_value"]
        self.bot_running = True
        self.bot_start_time = datetime.now()
        self.bot_status.set("🟢 กำลังทำงาน")
        self.status_label.config(fg=COLORS["green"])
        self.start_btn.config(state=tk.NORMAL, text="⏹ STOP BOT", bg=COLORS["red"])
        self.bot_alive_label.config(text="🟢 กำลังทำงาน", fg=COLORS["green"])
        self._start_uptime_timer()
        self._log("Bot STARTED", "SUCCESS")
        self._log(f"  เหรียญ: {self.config.trading.symbol}", "INFO")
        self._log(
            (
                f"  Paper Balance: ฿{self.paper_balance_thb:,.2f}"
                if startup_context.get("paper_mode")
                else f"  THB Available: ฿{startup_context['thb_avail']:,.2f} | {startup_context['coin']}: {startup_context['coin_total']:.8g}"
            ),
            "INFO",
        )
        self._log(f"  Interval: {self.config.trading.trading_interval_seconds}s", "INFO")
        self._log(
            f"  SL: {self.config.trading.stop_loss_pct}% | TP: {self.config.trading.take_profit_pct}%",
            "INFO",
        )
        self._log(
            f"  AI Scale-In: {'ON' if self.config.trading.ai_scale_in_enabled else 'OFF'} @ -{self.config.trading.ai_scale_in_loss_pct}% | "
            f"AI Take Profit: {'ON' if self.config.trading.ai_take_profit_enabled else 'OFF'} @ +{self.config.trading.ai_take_profit_min_profit_pct}% | "
            f"Profit Cashout: {'ON' if self.config.trading.profit_cashout_enabled else 'OFF'} @ +{self.config.trading.profit_cashout_min_profit_pct}% | "
            f"Fee Guard: {'ON' if self.config.trading.small_position_fee_guard_enabled else 'OFF'} ≤ ฿{self.config.trading.small_position_fee_guard_max_cost_thb:,.0f}",
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
            profit_cashout_pct = float(self.profit_cashout_pct_var.get())
            fee_guard_max_cost = float(self.fee_guard_max_cost_var.get())
            auto_buy_amount = float(self.auto_trade_amount_var.get())
        except ValueError:
            if show_popup:
                messagebox.showerror(
                    "Settings Error",
                    "ค่า Interval, SL, TP, AI Loss, AI Profit, Profit Cashout, Fee Guard หรือ Auto Buy ไม่ถูกต้อง",
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
        self.config.trading.profit_cashout_enabled = self.profit_cashout_enabled.get()
        self.config.trading.profit_cashout_min_profit_pct = profit_cashout_pct
        self.config.trading.small_position_fee_guard_enabled = self.fee_guard_enabled.get()
        self.config.trading.small_position_fee_guard_max_cost_thb = fee_guard_max_cost
        self.config.trading.symbol = applied_symbol
        self.config.trading.paper_trade_enabled = self.paper_trade_enabled.get()

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
            "profit_cashout_enabled": self.config.trading.profit_cashout_enabled,
            "profit_cashout_pct": profit_cashout_pct,
            "fee_guard_enabled": self.config.trading.small_position_fee_guard_enabled,
            "fee_guard_max_cost": fee_guard_max_cost,
            "boss_mode": self.boss_mode.get(),
            "paper_trade_enabled": self.paper_trade_enabled.get(),
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
                f"อัปเดตพารามิเตอร์สด | Interval {interval}s | SL {stop_loss_pct}% | TP {take_profit_pct}% | Cashout {'ON' if self.profit_cashout_enabled.get() else 'OFF'} @ +{profit_cashout_pct}% | Fee Guard {'ON' if self.fee_guard_enabled.get() else 'OFF'} ≤ ฿{fee_guard_max_cost:,.0f} | Auto Buy ฿{auto_buy_amount:,.2f} | Mode {'PAPER' if self.paper_trade_enabled.get() else 'LIVE'}",
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
        self.bot_status.set("⏹ หยุดอยู่")
        self.status_label.config(fg=COLORS["yellow"])
        self.start_btn.config(text="▶ START BOT", bg=COLORS["green"])
        self.bot_alive_label.config(text="⏹ หยุดอยู่", fg=COLORS["red"])
        self._log("Bot STOPPED", "WARN")
        self._refresh_runtime_badges()
        self._refresh_quick_start_summary()

    def _on_window_close_request(self):
        """Require the bot to be stopped before the GUI window can be closed."""
        if self._bot_start_in_progress:
            messagebox.showwarning(
                "Bot Starting",
                "ระบบกำลังเริ่มบอทอยู่ กรุณารอให้เริ่มเสร็จหรือกดหยุดบอทก่อน แล้วค่อยปิดโปรแกรม",
            )
            return

        if self.bot_running:
            messagebox.showwarning(
                "Stop Bot First",
                "ต้องกด STOP BOT ให้เรียบร้อยก่อน จึงจะปิดโปรแกรมได้",
            )
            self._log("บล็อกการปิดโปรแกรม: ต้องหยุดบอทก่อน", "WARN")
            return

        self.root.destroy()

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

        with self._trade_cycle_lock:
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
            ai_prediction = self._predict_ai(df_ind)

            # Step 4: Update GUI
            self.root.after(0, self._update_gui_data, signals, ai_prediction)

            # Step 5: Get balance snapshot
            balance_detail = self.client.get_balance_detail()
            self.wallet_balances = balance_detail
            thb_info = balance_detail.get("THB", {})
            live_balance = float(thb_info.get("available", 0.0) or 0.0)
            self.last_thb_balance = live_balance
            positions = self.strategy.get_open_positions(symbol)
            if self._is_paper_trade_mode():
                balance = self._ensure_paper_balance(live_balance)
            else:
                balance = live_balance

            # Refresh wallet display every 5 cycles
            if self.cycle_count % 5 == 0:
                self.root.after(0, self._load_wallet)

            risk_check = None

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
                risk_check = self.risk_manager.can_trade(balance, positions, signals, ai_prediction)
                if risk_check["allowed"]:
                    action = self._check_signals(
                        symbol, signals, ai_prediction, balance, current_price
                    )

            action_name = action or "HOLD"
            decision_snapshot = self._build_decision_snapshot(
                symbol,
                current_price,
                balance,
                positions,
                signals,
                ai_prediction,
                action_name,
                risk_check=risk_check,
            )
            self.root.after(0, self._log,
                            f"Cycle #{self.cycle_count} | {symbol} @ {current_price:,.2f} | "
                            f"{'PAPER' if self._is_paper_trade_mode() else 'THB'}: ฿{balance:,.2f} | {action_name}",
                            "TRADE" if action else "INFO")

            # Update last action
            self.root.after(0, lambda a=action_name: self.bot_last_action.set(a))
            self.root.after(0, lambda: self.bot_cycles_str.set(str(self.cycle_count)))
            self.root.after(0, lambda snapshot=decision_snapshot: self._apply_decision_snapshot(snapshot))

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
            _, rl_agent = self._get_ai_models()
            state = rl_agent.build_live_state(
                df_ind, has_position=True, entry_price=position.entry_price
            )
            if state is None:
                return {}
            return rl_agent.decide(state)
        except Exception as e:
            self.root.after(0, self._log, f"RL live decision error: {e}", "ERROR")
            return {}

    def _get_ai_models(self):
        """Return a consistent snapshot of the active AI models."""
        with self._ai_model_lock:
            return self.lstm_predictor, self.rl_agent

    def _predict_ai(self, df_ind) -> Dict:
        """Run LSTM prediction using the currently committed model."""
        lstm_predictor, _ = self._get_ai_models()
        return lstm_predictor.predict(df_ind)

    def _commit_ai_models(self, lstm_predictor: Optional[LSTMPredictor] = None,
                          rl_agent: Optional[RLTradingAgent] = None):
        """Atomically swap the AI models used by auto trade."""
        with self._ai_model_lock:
            if lstm_predictor is not None:
                self.lstm_predictor = lstm_predictor
            if rl_agent is not None:
                self.rl_agent = rl_agent

    def _reload_saved_ai_models(self, df_ind=None) -> Dict[str, str]:
        """Load the persisted AI models and make them active for auto trade."""
        lstm_source = "unchanged"
        rl_source = "unchanged"
        reloaded_lstm = None
        reloaded_rl = None

        try:
            candidate_lstm = LSTMPredictor(self.config.ai, self.logger)
            candidate_lstm.load_model(df_ind)
            if candidate_lstm.model is not None:
                reloaded_lstm = candidate_lstm
                lstm_source = "saved"
        except Exception as e:
            self.root.after(0, self._log, f"LSTM reload failed: {e}", "WARN")

        try:
            candidate_rl = RLTradingAgent(self.config.ai, logger=self.logger)
            if candidate_rl.load_model():
                reloaded_rl = candidate_rl
                rl_source = "saved"
        except Exception as e:
            self.root.after(0, self._log, f"RL reload failed: {e}", "WARN")

        if reloaded_lstm is not None or reloaded_rl is not None:
            self._commit_ai_models(reloaded_lstm, reloaded_rl)

        return {
            "lstm": lstm_source,
            "rl": rl_source,
        }

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

    def _update_recovery_confirmation(self, current_price: float, trigger_price: float,
                                      confirm_count: int, required_confirm_cycles: int,
                                      buffer_pct: float, low_updated: bool) -> Dict:
        """Require the market to hold above the trigger before rebuying."""
        effective_trigger = trigger_price * (1 + max(buffer_pct, 0.0) / 100) if trigger_price > 0 else 0.0

        if low_updated or current_price < effective_trigger:
            confirm_count = 0
        else:
            confirm_count += 1

        return {
            "confirm_count": confirm_count,
            "required_confirm_cycles": max(required_confirm_cycles, 1),
            "effective_trigger": effective_trigger,
            "confirmed": confirm_count >= max(required_confirm_cycles, 1),
        }

    def _get_rebuy_target_price(self, exit_price: float, recovery_pct: float,
                                buffer_pct: float = 0.0,
                                low_price: Optional[float] = None) -> float:
        """Compute the effective re-buy trigger price for GUI and log text."""
        reference_price = low_price if low_price is not None and low_price > 0 else exit_price
        if reference_price <= 0:
            return 0.0

        tracker = self._update_recovery_tracker(
            exit_price,
            reference_price,
            reference_price,
            recovery_pct,
        )
        confirmation = self._update_recovery_confirmation(
            reference_price,
            tracker["trigger_price"],
            0,
            1,
            buffer_pct,
            False,
        )
        return float(confirmation["effective_trigger"])

    @staticmethod
    def _blend_numeric_advice(base_value: float, suggested_value: Optional[float], weight: float = 0.55) -> float:
        if suggested_value is None:
            return base_value
        return (base_value * (1 - weight)) + (suggested_value * weight)

    def _request_boss_llm_advice(self, stage: str, symbol: str, current_price: float,
                                 balance: float, signals: Dict, ai_prediction: Dict,
                                 adaptive_profile: Dict, position: Optional[Position] = None,
                                 tracker: Optional[Dict] = None, pnl_pct: float = 0.0,
                                 pnl_thb: float = 0.0) -> Dict:
        context = {
            "symbol": symbol,
            "current_price": round(float(current_price or 0.0), 8),
            "balance_thb": round(float(balance or 0.0), 2),
            "pnl_pct": round(float(pnl_pct or 0.0), 4),
            "pnl_thb": round(float(pnl_thb or 0.0), 2),
            "position_entry_price": round(float(position.entry_price), 8) if position else None,
            "position_amount": round(float(position.amount), 8) if position else None,
            "adaptive_profile": {
                "cutloss_pct": round(float(adaptive_profile.get("cutloss_pct", 0.0) or 0.0), 4),
                "hard_limit_pct": round(float(adaptive_profile.get("hard_limit_pct", 0.0) or 0.0), 4),
                "recovery_pct": round(float(adaptive_profile.get("recovery_pct", 0.0) or 0.0), 4),
                "rebuy_allocation_pct": round(float(adaptive_profile.get("rebuy_allocation_pct", 0.0) or 0.0), 2),
            },
            "signals": {
                "rsi": round(float(signals.get("rsi", 0.0) or 0.0), 2),
                "volume_ratio": round(float(signals.get("volume_ratio", 0.0) or 0.0), 4),
                "trend_down": bool(signals.get("trend_down", False)),
                "macd_bearish": bool(signals.get("macd_bearish", False)),
                "support_level": round(float(signals.get("support_level", 0.0) or 0.0), 8),
                "resistance_level": round(float(signals.get("resistance_level", 0.0) or 0.0), 8),
                "atr_pct": round(float(signals.get("atr_pct", 0.0) or 0.0), 4),
            },
            "ai": {
                "direction": ai_prediction.get("direction", "unknown"),
                "confidence": round(float(ai_prediction.get("confidence", 0.0) or 0.0), 4),
                "price_change_pct": round(float(ai_prediction.get("price_change_pct", 0.0) or 0.0), 4),
                "predicted_price": round(float(ai_prediction.get("predicted_price", 0.0) or 0.0), 8),
            },
        }
        if tracker:
            context["recovery"] = {
                "low_price": round(float(tracker.get("low_price", 0.0) or 0.0), 8),
                "trigger_price": round(float(tracker.get("trigger_price", 0.0) or 0.0), 8),
                "recovery_from_low_pct": round(float(tracker.get("recovery_from_low_pct", 0.0) or 0.0), 4),
                "change_from_exit_pct": round(float(tracker.get("change_from_exit_pct", 0.0) or 0.0), 4),
            }

        advice = self.llm_boss_advisor.review_boss_action(stage, context)
        advice["stage"] = stage
        self.last_boss_llm_advice = advice
        self.boss_llm_status.set(self._format_llm_status_text(advice))

        self._sync_llm_trade_status_visual()

        return advice

    def _request_trade_llm_advice(self, stage: str, symbol: str, current_price: float,
                                  balance: float, signals: Dict, ai_prediction: Dict,
                                  position: Optional[Position] = None,
                                  reasons: Optional[list] = None,
                                  signal_score: float = 0.0) -> Dict:
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
                "volume_ratio": round(float(signals.get("volume_ratio", 0.0) or 0.0), 4),
            },
            "ai": {
                "direction": ai_prediction.get("direction", "unknown"),
                "confidence": round(float(ai_prediction.get("confidence", 0.0) or 0.0), 4),
                "price_change_pct": round(float(ai_prediction.get("price_change_pct", 0.0) or 0.0), 4),
                "predicted_price": round(float(ai_prediction.get("predicted_price", 0.0) or 0.0), 8),
            },
            "reasons": reasons or [],
        }
        constraints = {
            "stage": stage,
            "prefer": "risk_reduction and clear trend confirmation",
        }
        advice = self.llm_boss_advisor.review_trade_action(stage, context, constraints)
        self.last_trade_llm_advice = advice
        if advice.get("used"):
            self.trade_llm_status.set(
                f"{advice.get('action', 'HOLD')} {advice.get('confidence', 0.0):.0%}"
            )
        elif self.llm_boss_advisor.is_enabled():
            self.trade_llm_status.set(self._format_llm_status_text(advice, compact=True))
        else:
            self.trade_llm_status.set("LLM trade: disabled")
        self._sync_llm_trade_status_visual()
        return advice

    def _apply_trade_llm_gate(self, advice: Dict, allow_action: str, block_action: str) -> Dict:
        min_conf = float(getattr(self.config.ai, "llm_override_min_confidence", 0.68) or 0.68)
        confidence = float(advice.get("confidence", 0.0) or 0.0)
        action = advice.get("action")
        return {
            "allow": advice.get("used") and action == allow_action and confidence >= min_conf,
            "block": advice.get("used") and action == block_action and confidence >= min_conf,
            "confidence": confidence,
            "reason": advice.get("reason", ""),
        }

    def _llm_is_confident(self, advice: Dict, expected_action: str) -> bool:
        min_conf = float(getattr(self.config.ai, "llm_override_min_confidence", 0.68) or 0.68)
        return (
            advice.get("used")
            and advice.get("action") == expected_action
            and float(advice.get("confidence", 0.0) or 0.0) >= min_conf
        )

    def _get_runtime_position_summary(self, current_price: float) -> Dict[str, float | int | str]:
        """Summarize the open position for the currently selected symbol."""
        symbol = self.config.trading.symbol
        positions = self.strategy.get_open_positions(symbol)
        if not positions or current_price <= 0:
            return {
                "count": 0,
                "amount": 0.0,
                "avg_entry": 0.0,
                "pnl_thb": 0.0,
                "pnl_pct": 0.0,
                "label": "ยังไม่มี position เปิดอยู่",
            }

        total_amount = sum(float(pos.amount or 0.0) for pos in positions)
        total_cost = sum(float(pos.entry_cost_thb or 0.0) for pos in positions)
        pnl_thb = sum(self.strategy.get_position_pnl(pos, current_price)["profit_thb"] for pos in positions)
        avg_entry = (total_cost / total_amount) if total_amount > 0 else 0.0
        pnl_pct = (pnl_thb / total_cost * 100) if total_cost > 0 else 0.0
        side_text = "กำไร" if pnl_thb >= 0 else "ขาดทุน"
        return {
            "count": len(positions),
            "amount": total_amount,
            "avg_entry": avg_entry,
            "pnl_thb": pnl_thb,
            "pnl_pct": pnl_pct,
            "label": f"ถือ {total_amount:.6g} | ต้นทุนเฉลี่ย {avg_entry:,.2f} | {side_text} {abs(pnl_thb):,.2f} THB ({pnl_pct:+.2f}%)",
        }

    def _get_boss_runtime_snapshot(self, current_price: float) -> Dict:
        """Build realtime Boss or Auto Re-Buy details for the GUI."""
        if self.boss_mode.get() and self.boss_waiting_recovery and self.boss_last_sell_price > 0:
            base_recovery_pct = 0.5
            try:
                base_recovery_pct = float(self.boss_recovery_pct_var.get())
            except ValueError:
                pass

            adaptive_profile = self.strategy.get_adaptive_risk_profile(
                self.last_signals,
                self.last_ai_prediction,
                base_recovery_pct=base_recovery_pct,
            )
            recovery_pct = adaptive_profile["recovery_pct"]
            allocation_pct = adaptive_profile["rebuy_allocation_pct"]
            tracker = self._update_recovery_tracker(
                self.boss_last_sell_price,
                current_price,
                self.boss_recovery_low_price,
                recovery_pct,
            )
            confirm_required = max(getattr(self.config.trading, "boss_recovery_confirm_cycles", 2), 1)
            cooldown_cycles = max(getattr(self.config.trading, "boss_recovery_cooldown_cycles", 1), 0)
            trigger_buffer_pct = max(getattr(self.config.trading, "reentry_trigger_buffer_pct", 0.05), 0.0)
            confirmation = self._update_recovery_confirmation(
                current_price,
                tracker["trigger_price"],
                self.boss_recovery_confirm_count,
                confirm_required,
                trigger_buffer_pct,
                False,
            )
            cycles_waited = max(self.cycle_count - self.boss_recovery_sell_cycle, 0)
            cooldown_left = max(cooldown_cycles - cycles_waited, 0)
            target_budget = self.boss_rebuy_budget_thb or self.last_thb_balance
            buy_budget = min(
                self.last_thb_balance * min(allocation_pct / 100, 0.95),
                target_budget * (allocation_pct / 100),
            )
            return {
                "title": "🏆 Boss รอซื้อคืน",
                "color": COLORS["yellow"],
                "detail": (
                    f"ราคา Re-Buy {confirmation['effective_trigger']:,.2f} | ตอนนี้ {current_price:,.2f} | ขาย {self.boss_last_sell_price:,.2f} | low {self.boss_recovery_low_price:,.2f}\n"
                    f"ฟื้น {tracker['recovery_from_low_pct']:+.2f}% | confirm {self.boss_recovery_confirm_count}/{confirm_required} | cooldown {cooldown_left} | งบ ฿{buy_budget:,.0f} | {self._truncate_ui_text(self.boss_llm_status.get(), 58)}"
                ),
            }

        if self.reentry_waiting and self.reentry_symbol:
            base_rise_pct = 0.5
            try:
                base_rise_pct = float(self.reentry_rise_pct_var.get())
            except ValueError:
                pass

            adaptive_profile = self.strategy.get_adaptive_risk_profile(
                self.last_signals,
                self.last_ai_prediction,
                base_recovery_pct=base_rise_pct,
            )
            recovery_pct = adaptive_profile["recovery_pct"]
            allocation_pct = adaptive_profile["rebuy_allocation_pct"]
            tracker = self._update_recovery_tracker(
                self.reentry_last_exit_price,
                current_price,
                self.reentry_recovery_low_price,
                recovery_pct,
            )
            confirm_required = max(getattr(self.config.trading, "reentry_confirm_cycles", 2), 1)
            cooldown_cycles = max(getattr(self.config.trading, "reentry_cooldown_cycles", 1), 0)
            trigger_buffer_pct = max(getattr(self.config.trading, "reentry_trigger_buffer_pct", 0.05), 0.0)
            confirmation = self._update_recovery_confirmation(
                current_price,
                tracker["trigger_price"],
                self.reentry_confirm_count,
                confirm_required,
                trigger_buffer_pct,
                False,
            )
            cycles_waited = max(self.cycle_count - self.reentry_sell_cycle, 0)
            cooldown_left = max(cooldown_cycles - cycles_waited, 0)
            buy_budget = min(
                self.last_thb_balance * min(allocation_pct / 100, 0.95),
                self.reentry_budget_thb * (allocation_pct / 100),
            )
            return {
                "title": f"🔁 Re-Buy รอ {self.reentry_symbol}",
                "color": COLORS["accent"],
                "detail": (
                    f"ราคา Re-Buy {confirmation['effective_trigger']:,.2f} | ตอนนี้ {current_price:,.2f} | ขาย {self.reentry_last_exit_price:,.2f} | low {self.reentry_recovery_low_price:,.2f}\n"
                    f"ฟื้น {tracker['recovery_from_low_pct']:+.2f}% | confirm {self.reentry_confirm_count}/{confirm_required} | cooldown {cooldown_left} | งบ ฿{buy_budget:,.0f} | {self._truncate_ui_text(self.boss_llm_status.get(), 58)}"
                ),
            }

        if self.boss_mode.get():
            position_summary = self._get_runtime_position_summary(current_price)
            return {
                "title": "🏆 Boss พร้อมทำงาน",
                "color": COLORS["green"],
                "detail": (
                    f"ราคาปัจจุบัน {current_price:,.2f} | {position_summary['label']}\n"
                    f"{self._truncate_ui_text(self.boss_llm_status.get(), 92)}"
                ),
            }

        return {
            "title": "Boss: OFF",
            "color": COLORS["text_dim"],
            "detail": f"ราคาปัจจุบัน {current_price:,.2f}",
        }

    def _clear_boss_recovery_state(self):
        """Reset pending boss buy-back recovery tracking."""
        self.boss_waiting_recovery = False
        self.boss_last_sell_price = 0.0
        self.boss_recovery_low_price = 0.0
        self.boss_rebuy_budget_thb = 0.0
        self.boss_recovery_sell_cycle = 0
        self.boss_recovery_confirm_count = 0

    def _arm_boss_recovery_state(self, exit_price: float, budget_thb: float):
        """Store recovery tracking state after a boss cutloss sell."""
        self.boss_last_sell_price = exit_price
        self.boss_recovery_low_price = exit_price
        self.boss_rebuy_budget_thb = max(budget_thb, 0.0)
        self.boss_recovery_sell_cycle = self.cycle_count
        self.boss_recovery_confirm_count = 0
        self.boss_waiting_recovery = True

    def _check_ai_scale_in(self, symbol: str, current_price: float, balance: float,
                           positions, signals: Dict, ai_prediction: Dict, df_ind) -> str:
        """Use AI to average down a losing position when rebound odds improve."""
        if not positions or not self.config.trading.ai_scale_in_enabled:
            return ""

        risk_check = self.risk_manager.can_trade(balance, positions, signals, ai_prediction)
        if not risk_check["allowed"]:
            return ""

        representative_position = min(
            positions,
            key=lambda pos: self.strategy.get_position_pnl(pos, current_price)["profit_pct"],
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
            balance,
            current_price,
            scale_in["signal_strength"],
            signals,
            ai_prediction,
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
        boss_cooldown_cycles = max(getattr(self.config.trading, "boss_recovery_cooldown_cycles", 1), 0)
        boss_confirm_cycles = max(getattr(self.config.trading, "boss_recovery_confirm_cycles", 2), 1)
        trigger_buffer_pct = max(getattr(self.config.trading, "reentry_trigger_buffer_pct", 0.05), 0.0)

        # Wait for price recovery after a boss cutloss before buying back.
        if self.boss_waiting_recovery and self.boss_last_sell_price > 0:
            exit_price = self.boss_last_sell_price
            tracker = self._update_recovery_tracker(
                exit_price,
                current_price,
                self.boss_recovery_low_price,
                recovery_pct,
            )
            llm_advice = self._request_boss_llm_advice(
                "recovery",
                symbol,
                current_price,
                balance,
                signals,
                ai_prediction,
                adaptive_profile,
                tracker=tracker,
            )
            recovery_pct = max(
                self.config.trading.adaptive_rebuy_floor_pct,
                min(
                    self.config.trading.adaptive_rebuy_ceiling_pct,
                    self._blend_numeric_advice(recovery_pct, llm_advice.get("recovery_pct")),
                ),
            )
            rebuy_allocation_pct = max(
                self.config.trading.adaptive_rebuy_min_allocation_pct,
                min(
                    self.config.trading.adaptive_rebuy_max_allocation_pct,
                    self._blend_numeric_advice(rebuy_allocation_pct, llm_advice.get("allocation_pct"), weight=0.45),
                ),
            )
            tracker = self._update_recovery_tracker(
                exit_price,
                current_price,
                self.boss_recovery_low_price,
                recovery_pct,
            )
            self.boss_recovery_low_price = tracker["low_price"]
            pending_rebuy_price = self._get_rebuy_target_price(
                exit_price,
                recovery_pct,
                trigger_buffer_pct,
                self.boss_recovery_low_price,
            )
            cycles_waited = max(self.cycle_count - self.boss_recovery_sell_cycle, 0)

            if cycles_waited < boss_cooldown_cycles:
                self.boss_recovery_confirm_count = 0
                if self.cycle_count % 3 == 0:
                    self.root.after(0, self._log,
                        f"🏆 Boss cooldown หลังขาย | รออีก {boss_cooldown_cycles - cycles_waited} รอบ | "
                        f"low หลังขาย {self.boss_recovery_low_price:,.2f} | ราคา Re-Buy {pending_rebuy_price:,.2f}",
                        "INFO")
                return ""

            confirmation = self._update_recovery_confirmation(
                current_price,
                tracker["trigger_price"],
                self.boss_recovery_confirm_count,
                boss_confirm_cycles,
                trigger_buffer_pct,
                tracker["low_updated"],
            )
            self.boss_recovery_confirm_count = confirmation["confirm_count"]

            if tracker["recovery_from_low_pct"] >= recovery_pct and confirmation["confirmed"]:
                if self._llm_is_confident(llm_advice, "WAIT") or self._llm_is_confident(llm_advice, "HOLD"):
                    if self.cycle_count % 3 == 0:
                        self.root.after(0, self._log,
                            f"🏆 Boss ชะลอ buy-back ตาม LLM | {llm_advice.get('reason', '')}",
                            "AI")
                    return ""
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
                        f"(AI เป้าฟื้น: +{recovery_pct:.2f}% | ราคา Re-Buy {pending_rebuy_price:,.2f} | "
                        f"confirm {self.boss_recovery_confirm_count}/{boss_confirm_cycles})",
                        "INFO")
            return ""

        # ── Check positions for cutloss ──
        positions = self.strategy.get_open_positions(symbol)
        scale_in_checked = False
        for pos in list(positions):
            pnl = self.strategy.get_position_pnl(pos, current_price)
            pnl_pct = pnl["profit_pct"]
            pnl_thb = pnl["profit_thb"]

            if self.strategy.check_stop_loss(pos, current_price):
                record = self._do_sell(pos, current_price, "STOP_LOSS_TO_THB", arm_auto_reentry=False)
                if record:
                    self._arm_boss_recovery_state(current_price, float(record.get("net_exit_value_thb", 0.0) or 0.0))
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
                llm_advice = self._request_boss_llm_advice(
                    "cutloss",
                    symbol,
                    current_price,
                    balance,
                    signals,
                    ai_prediction,
                    adaptive_profile,
                    position=pos,
                    pnl_pct=pnl_pct,
                    pnl_thb=pnl_thb,
                )
                cutloss_pct = max(
                    self.config.trading.adaptive_cutloss_floor_pct,
                    min(
                        self.config.trading.adaptive_cutloss_ceiling_pct,
                        self._blend_numeric_advice(cutloss_pct, llm_advice.get("cutloss_pct")),
                    ),
                )
                recovery_pct = max(
                    self.config.trading.adaptive_rebuy_floor_pct,
                    min(
                        self.config.trading.adaptive_rebuy_ceiling_pct,
                        self._blend_numeric_advice(recovery_pct, llm_advice.get("recovery_pct")),
                    ),
                )
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
                llm_force_sell = self._llm_is_confident(llm_advice, "SELL") and pnl_pct <= -cutloss_pct
                llm_hold = self._llm_is_confident(llm_advice, "HOLD")
                if llm_hold and abs(pnl_pct) < hard_limit:
                    ai_cutloss = {**ai_cutloss, "should_sell": False}
                elif llm_force_sell:
                    ai_cutloss = {**ai_cutloss, "should_sell": True, "reason": f"LLM_SELL: {llm_advice.get('reason', '')}"}
                if ai_cutloss["should_sell"]:
                    record = self._do_sell(pos, current_price, ai_cutloss["reason"], arm_auto_reentry=False)
                    if record:
                        self._arm_boss_recovery_state(current_price, float(record.get("net_exit_value_thb", 0.0) or 0.0))
                        self.root.after(0, self._log,
                            f"🏆 AI BOSS CUTLOSS @ {current_price:,.2f} | "
                            f"ขายกลับเป็น THB | ขาดทุน: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                            f"AI CutLoss {cutloss_pct:.2f}% | Hard Limit {hard_limit:.2f}% | "
                            f"รอราคาฟื้น +{recovery_pct:.2f}% จากจุดขายเพื่อซื้อคืน {rebuy_allocation_pct:.0f}% ของงบเดิม | ราคา Re-Buy {self._get_rebuy_target_price(current_price, recovery_pct, trigger_buffer_pct):,.2f}",
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
            take_profit_snapshot = self.strategy.get_position_pnl(pos, current_price)
            take_profit_pnl = take_profit_snapshot["profit_pct"]
            take_profit_thb = take_profit_snapshot["profit_thb"]
            llm_tp_advice = self._request_boss_llm_advice(
                "take_profit",
                symbol,
                current_price,
                balance,
                signals,
                ai_prediction,
                adaptive_profile,
                position=pos,
                pnl_pct=take_profit_pnl,
                pnl_thb=take_profit_thb,
            )
            if self._llm_is_confident(llm_tp_advice, "HOLD"):
                ai_take_profit = {**ai_take_profit, "should_sell": False}
            elif self._llm_is_confident(llm_tp_advice, "SELL") and take_profit_pnl >= self.config.trading.quick_profit_sell_pct:
                ai_take_profit = {
                    **ai_take_profit,
                    "should_sell": True,
                    "reason": f"LLM_TAKE_PROFIT: {llm_tp_advice.get('reason', '')}",
                }
            if ai_take_profit["should_sell"]:
                record = self._do_sell(
                    pos,
                    current_price,
                    ai_take_profit["reason"],
                    arm_auto_reentry=False,
                    keep_principal=self._should_keep_principal_on_sell(ai_take_profit["reason"]),
                )
                if record:
                    if record.get("partial_close"):
                        self.root.after(0, self._log,
                            f"💸 AI PROFIT CASHOUT @ {current_price:,.2f} | ถอนกำไร {record['net_exit_value_thb']:,.2f} THB | คงเงินต้นไว้ใน {pos.symbol}",
                            "TRADE")
                        return "AI_PROFIT_CASHOUT"
                    self.root.after(0, self._log,
                        f"🤖 AI TAKE PROFIT @ {current_price:,.2f} | "
                        f"ขายกลับเป็น THB | กำไร: {ai_take_profit['profit_thb']:,.2f} THB ({ai_take_profit['profit_pct']:+.2f}%)",
                        "TRADE")
                    return "AI_TAKE_PROFIT"

            if self.strategy.check_take_profit(pos, current_price):
                pnl = self.strategy.get_position_pnl(pos, current_price)
                pnl_thb = pnl["profit_thb"]
                pnl_pct = pnl["profit_pct"]
                record = self._do_sell(pos, current_price, "TAKE_PROFIT_TO_THB", arm_auto_reentry=False, keep_principal=True)
                if record:
                    if record.get("partial_close"):
                        self.root.after(0, self._log,
                            f"💸 PROFIT CASHOUT @ {current_price:,.2f} | ถอนกำไร {record['net_exit_value_thb']:,.2f} THB | เงินต้นยังถือในเหรียญ",
                            "TRADE")
                        return "PROFIT_CASHOUT"
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
            pnl = self.strategy.get_position_pnl(pos, current_price)
            pnl_thb = pnl["profit_thb"]
            pnl_pct = pnl["profit_pct"]
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
                record = self._do_sell(
                    pos,
                    current_price,
                    ai_take_profit["reason"],
                    keep_principal=self._should_keep_principal_on_sell(ai_take_profit["reason"]),
                )
                if record:
                    if record.get("partial_close"):
                        self.root.after(0, self._log,
                                        f"💸 AI PROFIT CASHOUT @ {current_price:,.2f} | ถอนกำไร {record['net_exit_value_thb']:,.2f} THB | คงเงินต้นไว้ในเหรียญ",
                                        "TRADE")
                        return "AI_PROFIT_CASHOUT"
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
                    trigger_buffer_pct = max(getattr(self.config.trading, "reentry_trigger_buffer_pct", 0.05), 0.0)
                    self.root.after(0, self._log,
                                    f"🤖 AI CUTLOSS @ {current_price:,.2f} | "
                                    f"ขายกลับเป็น THB | ขาดทุน: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                                    f"AI CutLoss {adaptive_profile['cutloss_pct']:.2f}% / Hard {adaptive_profile['hard_limit_pct']:.2f}% | ราคา Re-Buy {self._get_rebuy_target_price(current_price, adaptive_profile['recovery_pct'], trigger_buffer_pct):,.2f}",
                                    "TRADE")
                    return "AI_CUTLOSS"

            if self.strategy.check_stop_loss(pos, current_price):
                record = self._do_sell(pos, current_price, "STOP_LOSS_TO_THB")
                if record:
                    trigger_buffer_pct = max(getattr(self.config.trading, "reentry_trigger_buffer_pct", 0.05), 0.0)
                    self.root.after(0, self._log,
                                    f"🔴 STOP LOSS @ {current_price:,.2f} | "
                                    f"ขายกลับเป็น THB | ขาดทุน: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                                    f"ตัดขาดทุนอัตโนมัติ! | ราคา Re-Buy {self._get_rebuy_target_price(current_price, adaptive_profile['recovery_pct'], trigger_buffer_pct):,.2f}", "TRADE")
                    return "STOP_LOSS"

            if self.strategy.check_take_profit(pos, current_price):
                record = self._do_sell(pos, current_price, "TAKE_PROFIT_TO_THB", keep_principal=True)
                if record:
                    if record.get("partial_close"):
                        self.root.after(0, self._log,
                                        f"💸 PROFIT CASHOUT @ {current_price:,.2f} | ถอนกำไร {record['net_exit_value_thb']:,.2f} THB | เงินต้นยังถือในเหรียญ",
                                        "TRADE")
                        return "PROFIT_CASHOUT"
                    trigger_buffer_pct = max(getattr(self.config.trading, "reentry_trigger_buffer_pct", 0.05), 0.0)
                    self.root.after(0, self._log,
                                    f"🟢 TAKE PROFIT @ {current_price:,.2f} | "
                                    f"ขายกลับเป็น THB | กำไร: {pnl_thb:,.2f} THB ({pnl_pct:+.2f}%) | "
                                    f"ทำกำไรอัตโนมัติ! | ราคา Re-Buy {self._get_rebuy_target_price(current_price, adaptive_profile['recovery_pct'], trigger_buffer_pct):,.2f}", "TRADE")
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
            self._clear_auto_reentry(symbol)
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
        reentry_cooldown_cycles = max(getattr(self.config.trading, "reentry_cooldown_cycles", 1), 0)
        reentry_confirm_cycles = max(getattr(self.config.trading, "reentry_confirm_cycles", 2), 1)
        trigger_buffer_pct = max(getattr(self.config.trading, "reentry_trigger_buffer_pct", 0.05), 0.0)

        tracker = self._update_recovery_tracker(
            self.reentry_last_exit_price,
            current_price,
            self.reentry_recovery_low_price,
            rise_pct,
        )
        self.reentry_recovery_low_price = tracker["low_price"]
        pending_rebuy_price = self._get_rebuy_target_price(
            self.reentry_last_exit_price,
            rise_pct,
            trigger_buffer_pct,
            self.reentry_recovery_low_price,
        )
        cycles_waited = max(self.cycle_count - self.reentry_sell_cycle, 0)

        if cycles_waited < reentry_cooldown_cycles:
            self.reentry_confirm_count = 0
            if self.cycle_count % 3 == 0:
                self.root.after(
                    0,
                    self._log,
                    f"🔁 Re-Buy cooldown หลังขาย {symbol} | รออีก {reentry_cooldown_cycles - cycles_waited} รอบ | "
                    f"low หลังขาย {self.reentry_recovery_low_price:,.2f} | ราคา Re-Buy {pending_rebuy_price:,.2f}",
                    "INFO",
                )
            return "REENTRY_WAIT"

        confirmation = self._update_recovery_confirmation(
            current_price,
            tracker["trigger_price"],
            self.reentry_confirm_count,
            reentry_confirm_cycles,
            trigger_buffer_pct,
            tracker["low_updated"],
        )
        self.reentry_confirm_count = confirmation["confirm_count"]

        if tracker["recovery_from_low_pct"] >= rise_pct and confirmation["confirmed"]:
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
                    f"(เกณฑ์ชะลอ {delay_pct:.2f}% | ราคา Re-Buy {pending_rebuy_price:,.2f})",
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
                f"(ต้องการ {rise_pct:.2f}% | ราคา Re-Buy {pending_rebuy_price:,.2f} | "
                f"confirm {self.reentry_confirm_count}/{reentry_confirm_cycles})",
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
        self.reentry_sell_cycle = self.cycle_count
        self.reentry_confirm_count = 0
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
        self.reentry_sell_cycle = 0
        self.reentry_confirm_count = 0
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
                sell_snapshot = self.strategy.get_position_pnl(pos, price)
                for_sell_pnl_pct = sell_snapshot["profit_pct"]
                for_sell_pnl_thb = sell_snapshot["profit_thb"]
                sell_advice = self._request_trade_llm_advice(
                    "exit",
                    symbol,
                    price,
                    balance,
                    signals,
                    ai_pred,
                    position=pos,
                    reasons=sell.get("reasons", []),
                    signal_score=sell.get("score", 0.0),
                )
                sell_gate = self._apply_trade_llm_gate(sell_advice, allow_action="SELL", block_action="HOLD")
                should_sell = sell["should_sell"]
                if sell_gate["block"]:
                    should_sell = False
                    if self.cycle_count % 3 == 0:
                        self.root.after(
                            0,
                            self._log,
                            f"🧠 LLM ชะลอ SELL {symbol} | {sell_gate['reason']}",
                            "AI",
                        )
                elif sell_gate["allow"]:
                    should_sell = True
                    sell["reasons"] = list(sell.get("reasons", [])) + [f"LLM: {sell_gate['reason']}"]

                if not should_sell:
                    self._track_fee_guard_history(symbol, price, for_sell_pnl_pct, sell)
                    continue
                self._track_fee_guard_history(symbol, price, for_sell_pnl_pct, sell)
                sold_any = True
                last_reasons = "; ".join(sell["reasons"])
                self._do_sell(
                    pos,
                    price,
                    "SELL_SIGNAL",
                    keep_principal=self._should_keep_principal_on_sell("SELL_SIGNAL", sell),
                )
                self.root.after(0, self._log,
                    f"📉 SELL signal | P/L: {for_sell_pnl_thb:+,.2f} THB ({for_sell_pnl_pct:+.2f}%)",
                    "TRADE")
            if sold_any:
                self.root.after(0, self._log, f"  เหตุผล: {last_reasons}", "TRADE")
                return "SELL"

        return self._check_buy_signal_once(
            symbol,
            signals,
            ai_pred,
            balance,
            price,
            positions=positions,
        )

    def _check_buy_signal_once(self, symbol, signals, ai_pred, balance, price,
                               positions=None, configured_amount_override: Optional[float] = None,
                               force_feedback: bool = False) -> str:
        """Evaluate the buy side of auto trade exactly once."""
        positions = positions if positions is not None else self.strategy.get_open_positions(symbol)
        should_log_detail = force_feedback or self.cycle_count % 3 == 0

        # BUY signals
        buy = self.strategy.should_buy(signals, ai_pred)
        buy_advice = self._request_trade_llm_advice(
            "entry",
            symbol,
            price,
            balance,
            signals,
            ai_pred,
            reasons=buy.get("reasons", []),
            signal_score=buy.get("score", 0.0),
        )
        buy_gate = self._apply_trade_llm_gate(buy_advice, allow_action="BUY", block_action="SKIP")
        should_buy = buy["should_buy"]
        if buy_gate["block"]:
            should_buy = False
            if should_log_detail:
                self.root.after(
                    0,
                    self._log,
                    f"🧠 LLM ข้าม BUY {symbol} | {buy_gate['reason']}",
                    "AI",
                )
        elif buy_gate["allow"]:
            should_buy = True
            buy["reasons"] = list(buy.get("reasons", [])) + [f"LLM: {buy_gate['reason']}"]

        reason_summary = self._format_reason_list(buy.get("reasons", []), max_items=3, max_length=170)
        wait_hint = self._build_buy_wait_hint(price, signals, buy)

        if should_buy:
            # Allow buy if no position OR if adding to profitable position
            can_buy = not positions
            avg_pnl = 0.0
            if positions:
                # Allow additional buy if existing position is in profit
                avg_pnl = sum(
                    (price - p.entry_price) / p.entry_price * 100
                    for p in positions
                ) / len(positions)
                if avg_pnl > 1.0:  # position is >1% in profit
                    can_buy = True

            configured_amount = configured_amount_override
            if configured_amount is None:
                try:
                    configured_amount = float(self.auto_trade_amount_var.get())
                except ValueError:
                    configured_amount = 0.0

            if can_buy:
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

                sizing = self.risk_manager.calculate_position_size(
                    balance,
                    price,
                    buy.get("signal_strength", 0.0),
                    signals,
                    ai_pred,
                )
                buy_amount = min(configured_amount, sizing["position_size_thb"], balance * 0.95)
                if buy_amount < 10:
                    if should_log_detail:
                        note_text = self._format_reason_list(sizing.get("risk_notes", []), max_items=2, max_length=120)
                        self.root.after(
                            0,
                            self._log,
                            f"⚠️ สัญญาณ BUY มาแล้ว แต่ระบบลดขนาดไม้ตามความเสี่ยงจนต่ำกว่า 10 THB | {note_text or sizing.get('market_regime', 'Risk guard')}",
                            "WARN",
                        )
                    return "RISK_CAPPED_BUY"

                if buy_amount < configured_amount and should_log_detail:
                    note_text = self._format_reason_list(sizing.get("risk_notes", []), max_items=2, max_length=120)
                    self.root.after(
                        0,
                        self._log,
                        f"🛡️ Auto Buy ถูก cap จาก ฿{configured_amount:,.0f} เหลือ ฿{buy_amount:,.0f} | {note_text or sizing.get('market_regime', 'Risk guard')}",
                        "INFO",
                    )

                self._do_buy(symbol, buy_amount, price)
                reasons = "; ".join(buy["reasons"])
                self.root.after(
                    0,
                    self._log,
                    f"📈 AI BUY ฿{buy_amount:,.0f} | {reasons}",
                    "TRADE",
                )
                return "BUY"

            if should_log_detail:
                detail = (
                    f"⏳ ยังไม่ซื้อเพิ่ม {symbol} | มี position อยู่แล้ว | กำไรเฉลี่ย {avg_pnl:+.2f}% "
                    f"ยังไม่ถึง +1.00%"
                )
                if reason_summary:
                    detail = f"{detail} | {reason_summary}"
                self.root.after(0, self._log, detail, "INFO")
                if wait_hint:
                    self.root.after(0, self._log, f"  รอจังหวะเข้าเพิ่ม: {wait_hint}", "INFO")
            return ""

        if should_log_detail:
            score_text = f"score {float(buy.get('score', 0.0) or 0.0):.2f}/{self.config.trading.min_buy_signal_score:.2f}"
            detail = f"⏳ ยังไม่ซื้อ {symbol} | {score_text}"
            if wait_hint:
                detail = f"{detail} | {wait_hint}"
            self.root.after(0, self._log, detail, "INFO")
            if reason_summary:
                self.root.after(0, self._log, f"  เหตุผล: {reason_summary}", "INFO")

        return ""

    def _do_buy(self, symbol: str, amount_thb: float, price: float):
        """Execute buy order."""
        if self._is_paper_trade_mode():
            self._execute_paper_buy(symbol, amount_thb, price)
            return
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
            self.strategy.add_position(
                symbol,
                exec_price,
                crypto_amount,
                cost_thb=float(order.get("cost", amount_thb) or amount_thb),
            )
            self._clear_boss_recovery_state()
            self._clear_auto_reentry(symbol)
            self.root.after(0, self._add_history_entry,
                            "BUY", exec_price, 0)
            self.root.after(0, self._update_pnl_display, exec_price)

    def _should_keep_principal_on_sell(self, reason: str, sell_decision: Optional[Dict] = None) -> bool:
        """Return True when the sell should skim gains into THB instead of closing the whole position."""
        if not getattr(self.config.trading, "profit_cashout_enabled", False):
            return False
        if sell_decision is not None:
            return bool(
                (sell_decision.get("profit_lock_sell") or sell_decision.get("extended_rally_sell"))
                and not sell_decision.get("panic_sell")
            )
        upper_reason = str(reason or "").upper()
        return "TAKE_PROFIT" in upper_reason and "CUTLOSS" not in upper_reason and "STOP_LOSS" not in upper_reason

    def _do_sell(self, position: Position, price: float, reason: str,
                 arm_auto_reentry: bool = True,
                 keep_principal: bool = False):
        """Execute sell order back into THB."""
        if self._is_paper_trade_mode():
            return self._execute_paper_sell(
                position,
                price,
                reason,
                arm_auto_reentry=arm_auto_reentry,
                keep_principal=keep_principal,
            )
        if not self.client:
            return

        cashout_plan = None
        sell_amount = position.amount
        if keep_principal:
            cashout_plan = self.strategy.evaluate_profit_cashout(position, price)
            if cashout_plan["should_cashout"]:
                sell_amount = float(cashout_plan["sell_amount"] or 0.0)
            else:
                keep_principal = False
                sell_amount = position.amount

        estimated_value = sell_amount * price
        if estimated_value < 10:
            if not keep_principal and position in self.strategy.positions:
                self.strategy.positions.remove(position)
            self.root.after(
                0,
                self._log,
                (
                    f"⚠️ ข้ามการถอนกำไร {position.symbol} เพราะมูลค่าน้อยเพียง {estimated_value:,.2f} THB ซึ่งต่ำกว่าขั้นต่ำ Bitkub"
                    if keep_principal else
                    f"⚠️ ข้ามการขาย {position.symbol} เพราะมูลค่าคงเหลือเพียง {estimated_value:,.2f} THB ซึ่งต่ำกว่าขั้นต่ำ Bitkub; ตัดออกจากการติดตามของบอทแล้ว"
                ),
                "WARN",
            )
            return None

        order = self.client.create_sell_order(position.symbol, sell_amount)
        if "_error" in order:
            err = order.get("_error")
            raw = order.get("_raw", "")
            if err == 18:
                self.root.after(0, self._log,
                    f"❌ Auto-sell failed: insufficient balance (error 18) - {raw}", "ERROR")
            else:
                self.root.after(0, self._log,
                    f"❌ Auto-sell failed: error {err} | {raw}", "ERROR")
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

        if keep_principal and cashout_plan and cashout_plan["should_cashout"]:
            record = self.strategy.cash_out_profit(
                position,
                exit_price,
                float(order.get("amount", sell_amount) or sell_amount),
                reason,
                net_exit_value_thb=float(order.get("received", 0.0) or 0.0),
            )
        else:
            record = self.strategy.close_position(
                position,
                exit_price,
                reason,
                net_exit_value_thb=float(order.get("received", 0.0) or 0.0),
            )
        if record:
            self.risk_manager.record_trade_result(record["profit_thb"])
            if self.bot_running and arm_auto_reentry and reason != "MANUAL_SELL" and not record.get("partial_close"):
                self._arm_auto_reentry(
                    position.symbol,
                    exit_price,
                    float(record.get("net_exit_value_thb", 0.0) or 0.0),
                    reason,
                )
            # Track bot performance
            self.bot_total_realized_pnl += record["profit_thb"]
            self.bot_total_trades += 1
            if record["profit_thb"] >= 0:
                self.bot_win_trades += 1
            else:
                self.bot_lose_trades += 1
            history_type = "CASHOUT" if record.get("partial_close") else "SELL"
            self.root.after(0, self._add_history_entry,
                            history_type, exit_price, record["profit_pct"])
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

    def _format_last_loss_source(self) -> str:
        """Describe the latest realized losing trade shown in the status card."""
        for record in reversed(self.strategy.trade_history):
            profit_thb = float(record.get("profit_thb", 0.0) or 0.0)
            if profit_thb >= 0:
                continue
            symbol = str(record.get("symbol", "-") or "-")
            exit_price = float(record.get("exit_price", 0.0) or 0.0)
            profit_pct = float(record.get("profit_pct", 0.0) or 0.0)
            reason = str(record.get("reason", "") or "-")
            return (
                f"ขาดทุนที่ปิดล่าสุด: {symbol} {profit_thb:+,.2f} THB ({profit_pct:+.2f}%) "
                f"| ออก {exit_price:,.2f} | เหตุผล {self._truncate_ui_text(reason, 54)}"
            )
        return "ขาดทุนที่ปิดล่าสุด: ยังไม่มี"

    def _format_unrealized_source(self, current_price: float) -> str:
        """Describe the open position contributing the most to current floating P/L."""
        if not self.strategy.positions:
            return "ลอยตัวตอนนี้: ยังไม่มี position"

        ranked_positions = []
        for pos in self.strategy.positions:
            pnl = self.strategy.get_position_pnl(pos, current_price)
            ranked_positions.append((abs(float(pnl.get("profit_thb", 0.0) or 0.0)), pos, pnl))

        if not ranked_positions:
            return "ลอยตัวตอนนี้: ยังไม่มี position"

        _, position, pnl = max(ranked_positions, key=lambda item: item[0])
        profit_thb = float(pnl.get("profit_thb", 0.0) or 0.0)
        profit_pct = float(pnl.get("profit_pct", 0.0) or 0.0)
        direction_text = "ขาดทุนลอยตัว" if profit_thb < 0 else "กำไรลอยตัว"
        return (
            f"ลอยตัวตอนนี้: {position.symbol} {direction_text} {profit_thb:+,.2f} THB ({profit_pct:+.2f}%) "
            f"| เข้า {position.entry_price:,.2f} | ตอนนี้ {current_price:,.2f}"
        )

    def _track_fee_guard_history(self, symbol: str, current_price: float,
                                 pnl_pct: float, sell_decision: Optional[Dict] = None):
        """Write a single Trade History row when fee guard delays a sell for a symbol."""
        active = bool((sell_decision or {}).get("fee_guard_active"))
        if active:
            if self._fee_guard_history_state.get(symbol):
                return
            reasons = list((sell_decision or {}).get("reasons", []))
            note = f"{symbol} | {self._format_reason_list(reasons, max_items=1, max_length=104)}"
            self._add_history_entry("FEE-GUARD", current_price, pnl_pct, note=note, tag="fee_guard")
            self._fee_guard_history_state[symbol] = True
            return

        self._fee_guard_history_state.pop(symbol, None)

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
            unrealized += self.strategy.get_position_pnl(pos, current_price)["profit_thb"]
        self.bot_unrealized_pnl_str.set(f"{unrealized:+,.2f} THB")
        clr_u = COLORS["green"] if unrealized >= 0 else COLORS["red"]
        self._perf_labels["unrealized_pnl"].config(fg=clr_u)
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
              text=f"เทรด: {self.bot_total_trades}  |  "
                  f"✅ ชนะ: {self.bot_win_trades}  |  "
                  f"❌ แพ้: {self.bot_lose_trades}"
        )
        self.bot_last_loss_source_str.set(self._format_last_loss_source())
        self.bot_unrealized_source_str.set(self._format_unrealized_source(current_price))

        # Boss mode status
        snapshot = self._get_boss_runtime_snapshot(current_price)
        self.boss_realtime_status.set(snapshot["title"])
        self.boss_realtime_detail.set(snapshot["detail"])

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
        live_balance = thb_info.get("available", 0)
        balance = self._get_runtime_trade_balance(live_balance)
        self.balance_thb.set(f"{balance:,.2f} THB")
        self._update_balance_display(balance, price)

    def _update_gui_data(self, signals: Dict, ai_prediction: Dict):
        """Update GUI with latest signals and AI data."""
        self.last_signals = dict(signals)
        self.last_ai_prediction = dict(ai_prediction)

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
            pnl = self.strategy.get_position_pnl(pos, current_price)
            self.positions_tree.insert("", "end", values=(
                pos.symbol,
                f"{(pnl['entry_cost_thb'] / pos.amount) if pos.amount > 0 else pos.entry_price:,.2f}",
                f"{current_price:,.2f}",
                f"{pnl['profit_pct']:+.2f}%",
                f"{pnl['profit_thb']:+,.2f}",
                f"{pos.amount:.8f}",
                f"{pos.stop_loss_price:,.2f}",
                f"{pos.take_profit_price:,.2f}",
            ))

    def _update_balance_display(self, balance: float, current_price: float):
        """Update balance display."""
        total = balance
        for pos in self.strategy.positions:
            mark_price = current_price if current_price > 0 else pos.entry_price
            total += self.strategy.estimate_exit_value_thb(pos.amount, mark_price)

        self.balance_thb.set(f"{balance:,.2f} THB")
        self.total_value.set(f"{total:,.2f} THB")

        pnl = total - self.initial_portfolio_value if self.initial_portfolio_value > 0 else total - balance
        self.pnl_text.set(f"{pnl:+,.2f} THB")

    def _update_risk_display(self, balance: float):
        """Update risk status."""
        risk = self.risk_manager.get_risk_status(balance, self.strategy.positions)
        market_profile = self.risk_manager.get_market_risk_profile(
            self.last_signals,
            self.last_ai_prediction,
        )
        self.risk_text.config(
            text=(
                f"Daily Loss: {risk['daily_loss']:,.0f} / "
                f"{risk['max_daily_loss']:,.0f} THB  |  "
                f"Positions: {risk['open_positions']} / {risk['max_positions']}  |  "
                f"Exposure: {risk['exposure_pct']:.1f}%"
            )
        )
        self.risk_regime_text.set(
            f"Market: {market_profile['regime_label']} | Scale {market_profile['position_scale_pct']:.0f}% | ATR {market_profile['atr_pct']:.2f}%"
        )
        guard_notes = market_profile.get("reasons", [])
        guard_text = " | ".join(guard_notes[:3]) if guard_notes else "ไม่มีตัวกรองความเสี่ยงพิเศษในรอบนี้"
        guard_text = f"Loss streak: {risk['loss_streak']} | Daily trades: {risk['daily_trades']} | {guard_text}"
        self.risk_guard_text.set(self._truncate_ui_text(guard_text, 180))

        regime_color = COLORS["green"]
        if market_profile["regime"] == "downtrend":
            regime_color = COLORS["yellow"]
        elif market_profile["regime"] == "volatile_downtrend":
            regime_color = COLORS["red"]
        elif market_profile["regime"] == "high_volatility":
            regime_color = COLORS["accent"]
        self.risk_regime_label.config(fg=regime_color)

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
            pnl = self.strategy.get_position_pnl(pos, current_price)
            pnl_thb = pnl["profit_thb"]
            pnl_pct = pnl["profit_pct"]
            effective_entry = (pnl["entry_cost_thb"] / pos.amount) if pos.amount > 0 else pos.entry_price
            total_pnl_thb += pnl_thb
            total_cost += pnl["entry_cost_thb"]
            coin = pos.symbol.split("_")[0]
            summaries.append(
                f"{coin}: {pos.amount:.6g} @ {effective_entry:,.2f} → "
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

    def _add_history_entry(self, trade_type: str, price: float, pnl: float,
                           note: str = "", tag: str = ""):
        """Add entry to trade history table."""
        now = datetime.now().strftime("%H:%M:%S")
        resolved_tag = tag
        if not resolved_tag:
            if "BUY" in trade_type:
                resolved_tag = "buy"
            elif trade_type in {"SELL", "P-SELL", "CASHOUT", "P-CASHOUT"}:
                resolved_tag = "sell"
            elif "FEE-GUARD" in trade_type:
                resolved_tag = "fee_guard"
        self.history_tree.insert(
            "",
            0,
            values=(now, trade_type, f"{price:,.2f}", f"{pnl:+.2f}%", self._truncate_ui_text(note, 72)),
            tags=(resolved_tag,) if resolved_tag else (),
        )

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
                ai_pred = self._predict_ai(df_ind)
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

        if self._ai_training_in_progress:
            self._log("AI training is already running", "WARN")
            return

        self._ai_training_in_progress = True
        self._log("Starting AI training...", "AI")
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        """Perform AI training in background."""
        if not self.data_collector:
            self._ai_training_in_progress = False
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

            trained_lstm = LSTMPredictor(self.config.ai, self.logger)
            trained_rl = RLTradingAgent(self.config.ai, logger=self.logger)

            # Train LSTM
            self.root.after(0, self._log, "Training LSTM model...", "AI")
            trained_lstm.train(df_ind)
            self.root.after(0, self._log, "LSTM training complete ✅", "SUCCESS")

            # Train RL
            self.root.after(0, self._log, "Training RL agent...", "AI")
            trained_rl.train(df_ind, episodes=200)
            self.root.after(0, self._log, "RL training complete ✅", "SUCCESS")

            reload_state = self._reload_saved_ai_models(df_ind)
            if reload_state["lstm"] != "saved":
                self._commit_ai_models(lstm_predictor=trained_lstm)
            if reload_state["rl"] != "saved":
                self._commit_ai_models(rl_agent=trained_rl)

            signals = self.indicator_engine.get_signal_summary(df_ind)
            ai_pred = self._predict_ai(df_ind)
            self.root.after(0, self._update_gui_data, signals, ai_pred)

            self.root.after(
                0,
                self._log,
                "All AI models trained, saved, and applied to auto trade permanently ✅",
                "SUCCESS",
            )

        except Exception as e:
            self.root.after(0, self._log, f"Training error: {e}", "ERROR")
        finally:
            self._ai_training_in_progress = False

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
