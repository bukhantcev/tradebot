# llm.py
import os
import json
import logging
from typing import Dict, Any

import httpx

log = logging.getLogger("LLM")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def _shorten(obj: Any, maxlen: int = 220) -> str:
    """
    Короткий человекочитаемый дамп для логов.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return (s[: maxlen] + "...") if len(s) > maxlen else s


async def ask_model(features: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Делает короткий запрос к OpenAI и возвращает решение.
    Логи только по факту: [LLM→] и [LLM←].
    Формат ответа: словарь с ключами:
      - decision: "Buy" | "Sell" | "Hold"
      - reason: краткое объяснение (строка)
    """
    if not OPENAI_API_KEY:
        log.info("[LLM] skip (no OPENAI_API_KEY) → Hold")
        return {"decision": "Hold", "reason": "no_api_key"}

    system = (
        "Ты помощник-трейдер. Дай одно из решений: Buy, Sell или Hold. "
        "Ответ возвращай JSON с ключами decision и reason. Коротко и по делу."
    )
    user = {
        "features": features,
        "context": ctx,
    }

    log.info(f"[LLM→] {_shorten(user)}")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "На основе следующих данных верни JSON: {decision: Buy|Sell|Hold, reason: string}.\n"
                    f"{json.dumps(user, ensure_ascii=False)}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 150,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Пытаемся распарсить JSON, но если не удалось — вернём raw
        decision = {"decision": "Hold", "reason": "parse_error", "raw": content}
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                d = str(parsed.get("decision", "Hold"))
                reason = str(parsed.get("reason", "") or "").strip()
                # нормализуем
                d_norm = d.capitalize()
                if d_norm not in ("Buy", "Sell", "Hold"):
                    d_norm = "Hold"
                decision = {"decision": d_norm, "reason": reason or "ok"}
        except Exception:
            # не JSON — пытаемся угадать по ключевому слову
            lc = content.lower()
            if "buy" in lc:
                decision = {"decision": "Buy", "reason": "model_text", "raw": content}
            elif "sell" in lc:
                decision = {"decision": "Sell", "reason": "model_text", "raw": content}
            else:
                decision = {"decision": "Hold", "reason": "model_text", "raw": content}

        log.info(f"[LLM←] {decision['decision']} | {decision.get('reason','')}")
        return decision

    except httpx.HTTPError as e:
        log.error(f"[LLM][HTTP] {e}")
        return {"decision": "Hold", "reason": "http_error"}
    except Exception as e:
        log.error(f"[LLM][ERR] {e}")
        return {"decision": "Hold", "reason": "error"}