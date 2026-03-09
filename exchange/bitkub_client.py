"""
Bitkub Exchange Client - เชื่อมต่อ Bitkub REST API โดยตรง (ไม่ใช้ ccxt)
"""
import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Any

import requests

from config import BitkubConfig
from utils.logger import TradeLogger


# Bitkub API timeframe mapping → seconds
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}

_MIN_ORDER_VALUE_THB = 10.0


class BitkubClient:
    """Client for Bitkub exchange using direct REST API v3."""

    def __init__(self, config: BitkubConfig, logger: Optional[TradeLogger] = None):
        self.config = config
        self.logger = logger or TradeLogger()
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.api_secret = config.api_secret

    # ─── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _flip_symbol(symbol: str) -> str:
        """Flip symbol format: BTC_THB <-> THB_BTC.
        Ticker & TradingView use BTC_THB; Books/Trades/Private use THB_BTC."""
        parts = symbol.split("_")
        if len(parts) == 2:
            return f"{parts[1]}_{parts[0]}"
        return symbol

    @staticmethod
    def _clean_num(n) -> int | float:
        """Remove trailing zeros: 100.0 → 100, 0.0 → 0.
        Bitkub rejects values with trailing zeros like 1000.00."""
        f = float(n)
        return int(f) if f == int(f) else f

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        """Public GET request."""
        url = f"{self.base_url}{path}"
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _signed_post(self, path: str, payload: Optional[Dict] = None) -> Any:
        """Authenticated POST request with Bitkub v3 HMAC-SHA256 signature.
        Signature = HMAC-SHA256(secret, timestamp_ms + 'POST' + path + body)
        """
        # Use server time to avoid clock skew
        try:
            srv = requests.get(f"{self.base_url}/api/v3/servertime", timeout=5)
            ts = str(srv.json())
        except Exception:
            ts = str(int(time.time() * 1000))

        if payload:
            body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        else:
            body = ""

        # v3 signing: ts_ms + method + path + body
        sig_msg = ts + "POST" + path + body
        sig = hmac.new(
            self.api_secret.encode(), sig_msg.encode(), hashlib.sha256
        ).hexdigest()

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-BTK-APIKEY": self.api_key,
            "X-BTK-SIGN": sig,
            "X-BTK-TIMESTAMP": ts,
        }
        url = f"{self.base_url}{path}"
        resp = requests.post(url, headers=headers, data=body if body else None, timeout=15)

        # Don't raise_for_status — Bitkub returns HTTP 400 with valid JSON errors
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            return {"error": -1}

        # Check for API errors
        error_code = data.get("error", 0)
        if error_code != 0:
            error_msgs = {
                1: "Invalid JSON payload",
                2: "Missing API key",
                3: "Invalid API key",
                4: "API pending for activation",
                5: "IP not allowed",
                6: "Invalid signature",
                8: "Invalid timestamp",
                11: "Invalid symbol",
                15: "Order value below minimum",
                18: "Insufficient balance / amount exceeds available",
            }
            err_text = error_msgs.get(error_code, f"Unknown error ({error_code})")
            self.logger.log_error(f"API {path}", Exception(f"Bitkub error {error_code}: {err_text}"))

        return data

    # ─── Public API ───────────────────────────────────────────────

    def get_ticker(self, symbol: str = "BTC_THB") -> Dict[str, Any]:
        """Get current ticker data for a symbol."""
        try:
            data = self._get("/api/market/ticker", params={"sym": symbol})
            info = data.get(symbol, {})
            if not info:
                # API may return without the underscore key – try all keys
                for key, val in data.items():
                    if key.replace("_", "") == symbol.replace("_", ""):
                        info = val
                        break
            return {
                "symbol": symbol,
                "last": float(info.get("last", 0)),
                "bid": float(info.get("highestBid", 0)),
                "ask": float(info.get("lowestAsk", 0)),
                "high": float(info.get("high24hr", 0)),
                "low": float(info.get("low24hr", 0)),
                "volume": float(info.get("baseVolume", 0)),
                "change": float(info.get("percentChange", 0)),
                "timestamp": int(time.time() * 1000),
            }
        except Exception as e:
            self.logger.log_error("get_ticker", e)
            return {}

    def get_orderbook(self, symbol: str = "BTC_THB", limit: int = 10) -> Dict[str, Any]:
        """Get order book for a symbol."""
        try:
            market_sym = self._flip_symbol(symbol)
            data = self._get("/api/market/books", params={"sym": market_sym, "lmt": limit})
            result = data.get("result", {})
            return {
                "bids": result.get("bids", []),
                "asks": result.get("asks", []),
                "timestamp": int(time.time() * 1000),
            }
        except Exception as e:
            self.logger.log_error("get_orderbook", e)
            return {"bids": [], "asks": []}

    def get_ohlcv(self, symbol: str = "BTC_THB", timeframe: str = "15m",
                  limit: int = 500) -> List[List]:
        """Get OHLCV candlestick data via Bitkub tradingview endpoint."""
        try:
            tf_sec = _TF_SECONDS.get(timeframe, 900)
            now = int(time.time())
            start = now - tf_sec * limit

            data = self._get("/tradingview/history", params={
                "symbol": symbol,
                "resolution": str(tf_sec // 60) if tf_sec < 86400 else "1D",
                "from": start,
                "to": now,
            })

            if data.get("s") != "ok":
                return []

            t_list = data.get("t", [])
            o_list = data.get("o", [])
            h_list = data.get("h", [])
            l_list = data.get("l", [])
            c_list = data.get("c", [])
            v_list = data.get("v", [])

            ohlcv = []
            for i in range(len(t_list)):
                ohlcv.append([
                    t_list[i] * 1000,        # timestamp ms
                    float(o_list[i]),         # open
                    float(h_list[i]),         # high
                    float(l_list[i]),         # low
                    float(c_list[i]),         # close
                    float(v_list[i]),         # volume
                ])
            return ohlcv

        except Exception as e:
            self.logger.log_error("get_ohlcv", e)
            return []

    def get_recent_trades(self, symbol: str = "BTC_THB", limit: int = 50) -> List[Dict]:
        """Get recent trades."""
        try:
            market_sym = self._flip_symbol(symbol)
            data = self._get("/api/market/trades", params={"sym": market_sym, "lmt": limit})
            result = data.get("result", [])
            return result
        except Exception as e:
            self.logger.log_error("get_recent_trades", e)
            return []

    # ─── Private API (requires API key) ──────────────────────────

    def get_balance(self) -> Dict[str, float]:
        """Get account balance (total = available + reserved)."""
        try:
            data = self._signed_post("/api/v3/market/balances")
            result = data.get("result", {})
            if not result:
                return {}
            balances: Dict[str, float] = {}
            for currency, info in result.items():
                total = float(info.get("available", 0)) + float(info.get("reserved", 0))
                if total > 0:
                    balances[currency.upper()] = total
            return balances
        except Exception as e:
            self.logger.log_error("get_balance", e)
            return {}

    def get_balance_detail(self) -> Dict[str, Dict[str, float]]:
        """Get detailed balance with available/reserved breakdown."""
        try:
            data = self._signed_post("/api/v3/market/balances")
            result = data.get("result", {})
            if not result:
                return {}
            balances: Dict[str, Dict[str, float]] = {}
            for currency, info in result.items():
                avail = float(info.get("available", 0))
                reserved = float(info.get("reserved", 0))
                if avail > 0 or reserved > 0:
                    balances[currency.upper()] = {
                        "available": avail,
                        "reserved": reserved,
                        "total": avail + reserved,
                    }
            return balances
        except Exception as e:
            self.logger.log_error("get_balance_detail", e)
            return {}

    def get_thb_balance(self) -> float:
        """Get THB balance."""
        balance = self.get_balance()
        return balance.get("THB", 0.0)

    def create_buy_order(self, symbol: str, amount_thb: float,
                         price: Optional[float] = None) -> Dict[str, Any]:
        """Create a buy order. Always uses limit order with market price if no price given."""
        try:
            if amount_thb < _MIN_ORDER_VALUE_THB:
                return {
                    "_error": 15,
                    "_raw": {
                        "error": 15,
                        "msg": f"Minimum buy amount is {_MIN_ORDER_VALUE_THB:.0f} THB",
                        "amount_thb": amount_thb,
                    },
                }

            market_sym = symbol.lower()  # btc_thb (no flip for v3)

            # Always use limit order — Bitkub market orders (rat=0) are unreliable
            if not price:
                ticker = self.get_ticker(symbol)
                if ticker and ticker.get("ask", 0) > 0:
                    price = ticker["ask"] * 1.002  # +0.2% above ask for instant fill
                elif ticker and ticker.get("last", 0) > 0:
                    price = ticker["last"] * 1.002
                else:
                    return {"_error": -2, "_raw": {"msg": "ดึงราคาไม่ได้"}}

            payload: Dict[str, Any] = {
                "sym": market_sym,
                "amt": self._clean_num(amount_thb),
                "rat": self._clean_num(round(float(price), 2)),
                "typ": "limit",
            }

            self.logger.log_info(f"BUY payload: {payload}")
            data = self._signed_post("/api/v3/market/place-bid", payload)

            # Check for API error
            error_code = data.get("error", 0)
            if error_code != 0:
                self.logger.log_error("create_buy_order",
                    Exception(f"Bitkub error {error_code}: {data}"))
                return {"_error": error_code, "_raw": data}

            result = data.get("result", {})
            if not result:
                self.logger.log_error("create_buy_order",
                    Exception(f"Empty result: {data}"))
                return {"_error": -1, "_raw": data}

            # Bitkub v3 response fields: id, typ, amt, rat, fee, cre, rec, ts
            order = {
                "id": result.get("id", ""),
                "rate": float(result.get("rat", 0)),
                "amount": float(result.get("rec", 0)),  # rec = crypto received
                "fee": float(result.get("fee", 0)),
                "cost": float(result.get("amt", amount_thb)),
            }

            self.logger.log_order(
                "BUY", symbol,
                order["rate"],
                order["amount"],
                str(order["id"]),
            )
            return order

        except Exception as e:
            self.logger.log_error("create_buy_order", e)
            return {}

    def create_sell_order(self, symbol: str, amount_crypto: float,
                          price: Optional[float] = None) -> Dict[str, Any]:
        """Create a sell order. Always uses limit order with market price if no price given."""
        try:
            market_sym = symbol.lower()  # btc_thb (no flip for v3)

            # quick pre-flight: ensure we don't try selling more crypto than available
            # (Bitkub returns code 18 in that case, which confused users).
            try:
                balances = self.get_balance_detail()
                base_coin = symbol.split("_")[0].upper()
                avail = balances.get(base_coin, {}).get("available", 0)
                if amount_crypto > avail + 1e-12:  # allow tiny rounding slack
                    return {
                        "_error": 18,
                        "_raw": {"msg": "Insufficient available balance", "available": avail, "requested": amount_crypto},
                    }
            except Exception:
                # ignore failure to fetch balance, we'll let the API return its own error
                pass

            # Always use limit order — Bitkub market orders (rat=0) are unreliable
            if not price:
                ticker = self.get_ticker(symbol)
                if ticker and ticker.get("bid", 0) > 0:
                    price = ticker["bid"] * 0.998  # -0.2% below bid for instant fill
                elif ticker and ticker.get("last", 0) > 0:
                    price = ticker["last"] * 0.998
                else:
                    return {"_error": -2, "_raw": {"msg": "ดึงราคาไม่ได้"}}

            estimated_value_thb = float(amount_crypto) * float(price)
            if estimated_value_thb < _MIN_ORDER_VALUE_THB:
                return {
                    "_error": 15,
                    "_raw": {
                        "error": 15,
                        "msg": f"Minimum sell value is {_MIN_ORDER_VALUE_THB:.0f} THB",
                        "amount_crypto": amount_crypto,
                        "estimated_value_thb": round(estimated_value_thb, 4),
                    },
                }

            payload: Dict[str, Any] = {
                "sym": market_sym,
                "amt": self._clean_num(amount_crypto),
                "rat": self._clean_num(round(float(price), 2)),
                "typ": "limit",
            }

            self.logger.log_info(f"SELL payload: {payload}")
            data = self._signed_post("/api/v3/market/place-ask", payload)

            # Check for API error
            error_code = data.get("error", 0)
            if error_code != 0:
                self.logger.log_error("create_sell_order",
                    Exception(f"Bitkub error {error_code}: {data}"))
                return {"_error": error_code, "_raw": data}

            result = data.get("result", {})
            if not result:
                self.logger.log_error("create_sell_order",
                    Exception(f"Empty result: {data}"))
                return {"_error": -1, "_raw": data}

            # Bitkub v3 response fields: id, typ, amt, rat, fee, cre, rec, ts
            order = {
                "id": result.get("id", ""),
                "rate": float(result.get("rat", 0)),
                "amount": float(result.get("amt", amount_crypto)),
                "fee": float(result.get("fee", 0)),
                "received": float(result.get("rec", 0)),  # rec = THB received
            }

            self.logger.log_order(
                "SELL", symbol,
                order["rate"],
                amount_crypto,
                str(order["id"]),
            )
            return order

        except Exception as e:
            self.logger.log_error("create_sell_order", e)
            return {}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        try:
            market_sym = symbol.lower()  # btc_thb (no flip for v3)
            payload = {
                "sym": market_sym,
                "id": order_id,
                "sd": "buy",  # will try both sides
            }
            data = self._signed_post("/api/v3/market/cancel-order", payload)
            if data.get("error", 0) != 0:
                # Try sell side
                payload["sd"] = "sell"
                data = self._signed_post("/api/v3/market/cancel-order", payload)

            self.logger.log_info(f"Order cancelled: {order_id}")
            return data.get("error", -1) == 0
        except Exception as e:
            self.logger.log_error("cancel_order", e)
            return False

    def get_open_orders(self, symbol: str = "BTC_THB") -> List[Dict]:
        """Get all open orders."""
        try:
            market_sym = symbol.lower()  # btc_thb (no flip for v3)
            data = self._signed_post("/api/v3/market/my-open-orders", {"sym": market_sym})
            return data.get("result", [])
        except Exception as e:
            self.logger.log_error("get_open_orders", e)
            return []

    def get_my_trades(self, symbol: str = "BTC_THB", limit: int = 50) -> List[Dict]:
        """Get trade history."""
        try:
            market_sym = symbol.lower()  # btc_thb (no flip for v3)
            data = self._signed_post("/api/v3/market/my-order-history", {
                "sym": market_sym,
                "lmt": limit,
            })
            return data.get("result", [])
        except Exception as e:
            self.logger.log_error("get_my_trades", e)
            return []
