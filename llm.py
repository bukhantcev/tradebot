"""
LLM-петля (OpenAI Responses API) — мягкий тюнинг гиперпараметров и режима.
Здесь — безопасный каркас: все рекомендации клипуем в допустимые диапазоны.
"""
import os
import json
import time
import logging
from typing import Dict, Any

import httpx
from config import OPENAI_API_KEY

logger = logging.getLogger("LLM")

SAFE_LIMITS = {
    "risk_pct_min": 0.5,   # %
    "risk_pct_max": 1.0,
    "sl_mult_min": 1.0,
    "sl_mult_max": 2.0,
    "tp_vs_sl_min": 1.5,
    "tp_vs_sl_max": 3.0,
}

def clip(v, lo, hi):
    return max(lo, min(hi, v))

class LLMAdvisor:
    def __init__(self, budget_daily_usd: float = 3.0, model: str = "gpt-4o-mini"):
        self.budget_daily_usd = budget_daily_usd
        self.model = model
        self._spent_today = 0.0
        self._day = time.strftime("%Y-%m-%d")

    def _check_budget(self) -> bool:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._spent_today = 0.0
        return self._spent_today < self.budget_daily_usd

    def advise(self, kpis: Dict[str, Any], market_brief: Dict[str, Any]) -> Dict[str, Any]:
        # Если нет ключа или бюджет исчерпан — ничего не делаем
        if not OPENAI_API_KEY or not self._check_budget():
            return {}

        prompt = {
            "role": "user",
            "content": f"""
You are a trading coach. Based on KPIs and market brief, suggest small adjustments:
- mode in ["trend","range"]
- risk_pct in [{SAFE_LIMITS["risk_pct_min"]}, {SAFE_LIMITS["risk_pct_max"]}]
- sl_mult in [{SAFE_LIMITS["sl_mult_min"]}, {SAFE_LIMITS["sl_mult_max"]}]
- tp_vs_sl in [{SAFE_LIMITS["tp_vs_sl_min"]}, {SAFE_LIMITS["tp_vs_sl_max"]}]
Return compact JSON with fields: mode, risk_pct, sl_mult, tp_vs_sl.
KPIs: {json.dumps(kpis)}
Market: {json.dumps(market_brief)}
"""
        }
        try:
            # минимальный каркас вызова
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            payload = {
                "model": self.model,
                "input": [prompt],
                "temperature": 0.2,
            }
            r = httpx.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=20.0)
            data = r.json()
            text = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "").strip()
            # попытка распарсить json из ответа
            rec = {}
            try:
                rec = json.loads(text)
            except Exception:
                logger.debug(f"[LLM] raw: {text}")

            mode = rec.get("mode")
            if mode not in ("trend","range"):
                mode = None
            risk_pct = rec.get("risk_pct")
            if isinstance(risk_pct, (int,float)):
                risk_pct = clip(float(risk_pct), SAFE_LIMITS["risk_pct_min"], SAFE_LIMITS["risk_pct_max"])
            else:
                risk_pct = None
            sl_mult = rec.get("sl_mult")
            if isinstance(sl_mult, (int,float)):
                sl_mult = clip(float(sl_mult), SAFE_LIMITS["sl_mult_min"], SAFE_LIMITS["sl_mult_max"])
            else:
                sl_mult = None
            tp_vs_sl = rec.get("tp_vs_sl")
            if isinstance(tp_vs_sl, (int,float)):
                tp_vs_sl = clip(float(tp_vs_sl), SAFE_LIMITS["tp_vs_sl_min"], SAFE_LIMITS["tp_vs_sl_max"])
            else:
                tp_vs_sl = None

            out = {"mode": mode, "risk_pct": risk_pct, "sl_mult": sl_mult, "tp_vs_sl": tp_vs_sl}
            return {k:v for k,v in out.items() if v is not None}
        except Exception as e:
            logger.warning(f"[LLM] advise failed: {e}")
            return {}