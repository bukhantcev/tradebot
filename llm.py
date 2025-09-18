# llm.py
import os
import json
import logging
from typing import Dict, Any
import asyncio
import httpx
import time

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

    log.debug(f"[LLM][DEBUG] system={system}, user={user}, model={OPENAI_MODEL}, temperature=0.2")
    log.info(f"[LLM→] {_shorten(user)}")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    start_time = time.monotonic()

    max_attempts = 3
    backoff_times = [1, 2, 4]

    last_exception = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            log.info(f"[LLM][RETRY] attempt {attempt}")
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                log.debug("[LLM][HTTP→] sending request")
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                log.debug(f"[LLM][HTTP←] status={r.status_code} body={_shorten(r.text)}")
                r.raise_for_status()
                data = r.json()
            break
        except httpx.HTTPError as e:
            log.debug(f"[LLM][HTTP ERROR DEBUG] {type(e).__name__}: {e}")
            last_exception = e
            if attempt == max_attempts:
                log.error(f"[LLM][HTTP] {e}")
                elapsed = time.monotonic() - start_time
                log.info(f"[LLM←] decision=Hold reason=http_error time={elapsed:.2f}s")
                return {"decision": "Hold", "reason": "http_error"}
            await asyncio.sleep(backoff_times[attempt - 1])
        except Exception as e:
            log.debug(f"[LLM][ERROR DEBUG] {type(e).__name__}: {e}")
            last_exception = e
            if attempt == max_attempts:
                log.error(f"[LLM][ERR] {e}")
                elapsed = time.monotonic() - start_time
                log.info(f"[LLM←] decision=Hold reason=error time={elapsed:.2f}s")
                return {"decision": "Hold", "reason": "error"}
            await asyncio.sleep(backoff_times[attempt - 1])
    else:
        # Should not reach here, but in case
        elapsed = time.monotonic() - start_time
        log.info(f"[LLM←] decision=Hold reason=error time={elapsed:.2f}s")
        return {"decision": "Hold", "reason": "error"}

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    log.debug(f"[LLM][DEBUG] raw content before parse: {content}")

    # Пытаемся распарсить JSON, но если не удалось — вернём raw
    decision = {"decision": "Hold", "reason": "parse_error", "raw": content}
    try:
        parsed = json.loads(content)
        log.debug(f"[LLM][DEBUG] parsed dict: {parsed}")
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
    log.debug(f"[LLM][DEBUG] chosen decision: {decision}")

    elapsed = time.monotonic() - start_time
    log.info(f"[LLM←] decision={decision['decision']} reason={decision.get('reason','')} time={elapsed:.2f}s")
    return decision