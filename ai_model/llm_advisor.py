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
            return result
        except Exception as error:
            self.logger.log_error("LLM boss advisor", error)
            fallback["reason"] = f"LLM error: {error}"
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