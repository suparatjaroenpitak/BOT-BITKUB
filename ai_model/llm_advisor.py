import json
import time
from typing import Any, Dict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class LLMBossAdvisor:
    """Optional LLM advisor for Boss Mode and normal buy/sell decisions."""

    def __init__(self, ai_config, trading_config, logger):
        self.config = ai_config
        self.trading_config = trading_config
        self.logger = logger
        self.client = None
        self.last_request_at = 0.0
        self.last_cache_key = ""
        self.last_response: Dict[str, Any] = {}
        self.disabled_until = 0.0

        if self.is_enabled() and OpenAI is not None:
            try:
                self.client = OpenAI(
                    api_key=self.config.openai_api_key,
                    timeout=self.config.llm_timeout_seconds,
                )
            except Exception as error:
                self.logger.log_error("LLM advisor init", error)

    def is_enabled(self) -> bool:
        return bool(getattr(self.config, "llm_enabled", False) and getattr(self.config, "openai_api_key", ""))

    def is_available(self) -> bool:
        return self.is_enabled() and self.client is not None

    def review_trade_action(self, stage: str, context: Dict[str, Any],
                            constraints: Dict[str, Any] | None = None) -> Dict[str, Any]:
        fallback = {
            "enabled": self.is_enabled(),
            "used": False,
            "action": "HOLD" if stage not in {"recovery", "entry"} else ("WAIT" if stage == "recovery" else "SKIP"),
            "confidence": 0.0,
            "reason": "LLM advisor unavailable",
            "status": "unavailable",
            "error_code": "unavailable",
            "cutloss_pct": None,
            "recovery_pct": None,
            "allocation_pct": None,
            "raw": "",
        }
        if not self.is_available():
            if OpenAI is None and self.is_enabled():
                fallback["reason"] = "openai package not installed"
            elif self.is_enabled():
                fallback["reason"] = "OpenAI client not ready"
            return fallback

        cache_key = json.dumps({"stage": stage, "context": context}, sort_keys=True, ensure_ascii=True, default=str)
        interval = max(getattr(self.config, "llm_request_interval_seconds", 20), 0)
        now = time.time()
        if self.disabled_until > now:
            cached_error = dict(self.last_response) if self.last_response else dict(fallback)
            cached_error["used"] = False
            cached_error["status"] = "backoff"
            cached_error["reason"] = cached_error.get("reason", "LLM temporarily paused")
            return cached_error
        if cache_key == self.last_cache_key and self.last_response:
            return self.last_response
        if self.last_response and now - self.last_request_at < interval:
            cached = dict(self.last_response)
            cached["reason"] = f"{cached.get('reason', 'cached')} | cached"
            return cached

        system_prompt = (
            "You are a trading risk advisor for Bitkub trading decisions. "
            "Return only compact JSON with keys: action, confidence, reason, cutloss_pct, recovery_pct, allocation_pct. "
            "Allowed action values: BUY, SELL, HOLD, BUYBACK, WAIT, SKIP. "
            "Keep reason under 140 characters. Prefer risk reduction over aggressive trading."
        )
        user_prompt = json.dumps(
            {
                "stage": stage,
                "symbol": context.get("symbol"),
                "market": context,
                "constraints": constraints or {
                    "cutloss_floor_pct": self.trading_config.adaptive_cutloss_floor_pct,
                    "cutloss_ceiling_pct": self.trading_config.adaptive_cutloss_ceiling_pct,
                    "recovery_floor_pct": self.trading_config.adaptive_rebuy_floor_pct,
                    "recovery_ceiling_pct": self.trading_config.adaptive_rebuy_ceiling_pct,
                    "allocation_floor_pct": self.trading_config.adaptive_rebuy_min_allocation_pct,
                    "allocation_ceiling_pct": self.trading_config.adaptive_rebuy_max_allocation_pct,
                },
            },
            ensure_ascii=True,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                temperature=0.1,
                max_tokens=self.config.llm_max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content if response.choices else "{}"
            parsed = self._parse_response(content)
            result = {
                "enabled": True,
                "used": True,
                "action": parsed.get("action", fallback["action"]),
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                "reason": str(parsed.get("reason", "LLM response parsed"))[:140],
                "status": "ok",
                "error_code": "",
                "cutloss_pct": self._normalize_optional_float(
                    parsed.get("cutloss_pct"),
                    self.trading_config.adaptive_cutloss_floor_pct,
                    self.trading_config.adaptive_cutloss_ceiling_pct,
                ),
                "recovery_pct": self._normalize_optional_float(
                    parsed.get("recovery_pct"),
                    self.trading_config.adaptive_rebuy_floor_pct,
                    self.trading_config.adaptive_rebuy_ceiling_pct,
                ),
                "allocation_pct": self._normalize_optional_float(
                    parsed.get("allocation_pct"),
                    self.trading_config.adaptive_rebuy_min_allocation_pct,
                    self.trading_config.adaptive_rebuy_max_allocation_pct,
                ),
                "raw": content,
            }
            self.last_request_at = now
            self.last_cache_key = cache_key
            self.last_response = result
            self.disabled_until = 0.0
            return result
        except Exception as error:
            self.logger.log_error("LLM boss advisor", error)
            error_info = self._classify_error(error)
            fallback["reason"] = error_info["reason"]
            fallback["status"] = error_info["status"]
            fallback["error_code"] = error_info["code"]
            fallback["raw"] = str(error)
            self.disabled_until = now + error_info["retry_after_seconds"]
            self.last_request_at = now
            self.last_cache_key = cache_key
            self.last_response = dict(fallback)
            return fallback

    def review_boss_action(self, stage: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.review_trade_action(stage, context)

    @staticmethod
    def _parse_response(content: str) -> Dict[str, Any]:
        if not content:
            return {}
        text = content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _normalize_optional_float(value: Any, minimum: float, maximum: float):
        if value is None:
            return None
        try:
            return _clamp(float(value), minimum, maximum)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _classify_error(error: Exception) -> Dict[str, Any]:
        text = str(error)
        normalized = text.lower()

        if "insufficient_quota" in normalized or "quota" in normalized:
            return {
                "status": "quota",
                "code": "insufficient_quota",
                "reason": "LLM quota exceeded",
                "retry_after_seconds": 300,
            }
        if "invalid_api_key" in normalized or "incorrect api key" in normalized or "401" in normalized:
            return {
                "status": "auth",
                "code": "invalid_api_key",
                "reason": "LLM invalid API key",
                "retry_after_seconds": 300,
            }
        if "rate limit" in normalized or "429" in normalized:
            return {
                "status": "rate_limit",
                "code": "rate_limit",
                "reason": "LLM rate limited",
                "retry_after_seconds": 90,
            }
        if "timeout" in normalized:
            return {
                "status": "timeout",
                "code": "timeout",
                "reason": "LLM timeout",
                "retry_after_seconds": 45,
            }
        return {
            "status": "error",
            "code": "error",
            "reason": "LLM unavailable",
            "retry_after_seconds": 60,
        }