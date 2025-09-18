import os, json, time, logging
from typing import Dict, Any
import httpx
from config import OPENAI_API_KEY

log = logging.getLogger("LLM")

SAFE = {"risk_min":0.5,"risk_max":1.0,"sl_min":1.0,"sl_max":2.0,"tpvs_min":1.5,"tpvs_max":3.0}

def _clip(v, lo, hi): return max(lo, min(hi, v))

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
        if not OPENAI_API_KEY or not self._check_budget():
            log.debug("[LLM] skip (no key or budget)")
            return {}

        prompt = {
            "role": "user",
            "content": f"""Return JSON with fields: mode in ["trend","range"], risk_pct [{SAFE['risk_min']}, {SAFE['risk_max']}], sl_mult [{SAFE['sl_min']},{SAFE['sl_max']}], tp_vs_sl [{SAFE['tpvs_min']},{SAFE['tpvs_max']}]. KPIs={json.dumps(kpis)} Market={json.dumps(market_brief)}"""
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": self.model, "input": [prompt], "temperature": 0.2}

        try:
            log.debug(f"[LLM→] {payload}")
            r = httpx.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=20.0)
            data = r.json()
            text = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "").strip()
            log.debug(f"[LLM←] {text[:400]}")
            rec = {}
            try: rec = json.loads(text)
            except Exception: pass

            mode = rec.get("mode") if rec.get("mode") in ("trend","range") else None
            rp = rec.get("risk_pct"); sl = rec.get("sl_mult"); tvs = rec.get("tp_vs_sl")
            out = {}
            if mode: out["mode"] = mode
            if isinstance(rp,(int,float)): out["risk_pct"] = _clip(float(rp), SAFE["risk_min"], SAFE["risk_max"])
            if isinstance(sl,(int,float)): out["sl_mult"]  = _clip(float(sl), SAFE["sl_min"], SAFE["sl_max"])
            if isinstance(tvs,(int,float)): out["tp_vs_sl"]= _clip(float(tvs), SAFE["tpvs_min"], SAFE["tpvs_max"])
            log.info(f"[LLM][APPLY] {out}")
            return out
        except Exception as e:
            log.warning(f"[LLM][ERR] {e}")
            return {}