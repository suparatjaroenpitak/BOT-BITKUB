"""
Logging System - บันทึก trade logs, errors, AI predictions
"""
import logging
import os
from datetime import datetime


def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
    """Create and configure a logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def get_trade_logger(log_file: str = "logs/trades.log") -> logging.Logger:
    """Get the trade-specific logger."""
    return setup_logger("trade", log_file)


def get_error_logger(log_file: str = "logs/errors.log") -> logging.Logger:
    """Get the error-specific logger."""
    return setup_logger("error", log_file, "ERROR")


def get_ai_logger(log_file: str = "logs/ai_predictions.log") -> logging.Logger:
    """Get the AI prediction logger."""
    return setup_logger("ai", log_file)


class TradeLogger:
    """Structured trade logging."""

    def __init__(self, log_dir: str = "logs"):
        self.trade_logger = get_trade_logger(os.path.join(log_dir, "trades.log"))
        self.error_logger = get_error_logger(os.path.join(log_dir, "errors.log"))
        self.ai_logger = get_ai_logger(os.path.join(log_dir, "ai_predictions.log"))

    def log_trade(self, action: str, symbol: str, price: float, amount: float,
                  reason: str = ""):
        """Log a trade execution."""
        self.trade_logger.info(
            f"TRADE | {action} | {symbol} | Price: {price:.2f} | "
            f"Amount: {amount:.8f} | Reason: {reason}"
        )

    def log_order(self, order_type: str, symbol: str, price: float, amount: float,
                  order_id: str = ""):
        """Log an order placement."""
        self.trade_logger.info(
            f"ORDER | {order_type} | {symbol} | Price: {price:.2f} | "
            f"Amount: {amount:.8f} | OrderID: {order_id}"
        )

    def log_stop_loss(self, symbol: str, buy_price: float, current_price: float,
                      loss_pct: float):
        """Log stop loss trigger."""
        self.trade_logger.warning(
            f"STOP_LOSS | {symbol} | BuyPrice: {buy_price:.2f} | "
            f"CurrentPrice: {current_price:.2f} | Loss: {loss_pct:.2f}%"
        )

    def log_take_profit(self, symbol: str, buy_price: float, current_price: float,
                        profit_pct: float):
        """Log take profit trigger."""
        self.trade_logger.info(
            f"TAKE_PROFIT | {symbol} | BuyPrice: {buy_price:.2f} | "
            f"CurrentPrice: {current_price:.2f} | Profit: {profit_pct:.2f}%"
        )

    def log_ai_prediction(self, symbol: str, current_price: float,
                          predicted_price: float, confidence: float,
                          direction: str):
        """Log AI prediction."""
        self.ai_logger.info(
            f"PREDICTION | {symbol} | Current: {current_price:.2f} | "
            f"Predicted: {predicted_price:.2f} | Confidence: {confidence:.4f} | "
            f"Direction: {direction}"
        )

    def log_error(self, context: str, error: Exception):
        """Log an error."""
        self.error_logger.error(f"{context} | {type(error).__name__}: {error}")

    def log_info(self, message: str):
        """Log general info."""
        self.trade_logger.info(message)
