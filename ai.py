# ai.py — выбор стратегии через OpenAI Chat Completions
import os
import json
import logging
from typing import Dict, Any

import requests
from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("ai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _prompt(payload: Dict[str, Any]) -> str:
    # Коротко, без токен-спама; структура ответа — строгий JSON.
    return f"""
You are a high-frequency scalping assistant. Decide the best scalping strategy now.
Return ONLY valid JSON with keys: strategy (one of: momentum, breakout, density, knife), reason, confidence (0..1).

Market snapshot (compact):
{json.dumps(payload, ensure_ascii=False) }
описание причины выбора пиши по-русски.
"""

def pick_strategy(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        log.warning("[AI] OPENAI_API_KEY missing; fallback momentum")
        return {"strategy": "momentum", "reason": "fallback (no api key)", "confidence": 0.5}

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert scalping strategist."},
            {"role": "user", "content": _prompt(payload)}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        j = resp.json()
        content = j["choices"][0]["message"]["content"]
        log.debug("[AI][RAW] %s", content)
        data = json.loads(content)
        st = data.get("strategy", "momentum")
        if st not in {"momentum", "breakout", "density", "knife"}:
            st = "momentum"
        return {
            "strategy": st,
            "reason": data.get("reason", ""),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except Exception as e:
        log.warning("[AI] error: %s", e)
        return {"strategy": "momentum", "reason": f"fallback ({e})", "confidence": 0.5}