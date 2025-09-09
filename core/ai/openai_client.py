import logging
import httpx

log = logging.getLogger("AI")

PROMPT_SYSTEM = """\
You are a quant trading assistant. You must pick exactly ONE strategy name based on the snapshot:
- Momentum
- Reversal
- Breakout
- Orderbook Density
- Knife

Return ONLY a strict JSON:
{"strategy": "<one_of_the_above>", "reason": "<short_reason>"}"""

def _build_user_prompt(snapshot: dict) -> str:
    # компактный снапшот: цены, волатильность, объемы, стакан
    return f"SNAPSHOT:\n{snapshot}"

async def ask_strategy(model: str, api_key: str, snapshot: dict) -> dict:
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": _build_user_prompt(snapshot)},
        ],
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post("https://api.openai.com/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json=payload)
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"]
        log.info("[AI][raw] %s", content)
        try:
            import json
            data = json.loads(content)
            strategy = data.get("strategy", "Momentum")
            reason = data.get("reason", "")
            log.info("[AI] chosen=%s | reason=%s", strategy, reason)
            return {"strategy": strategy, "reason": reason, "raw": content}
        except Exception:
            log.exception("[AI] parse error")
            return {"strategy": "Momentum", "reason": "fallback", "raw": content}