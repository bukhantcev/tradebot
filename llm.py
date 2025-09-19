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

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))
LLM_BACKOFF = float(os.getenv("LLM_BACKOFF", "0.75"))


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
        "Ты помощник-трейдер. Верни JSON с ключами: "
        "decision (Buy|Sell|Hold), reason (string), "
        "regime ('trend'|'flat'|null), trend_side ('Buy'|'Sell'|null), "
        "mode ('trend'|'flat'|null), side ('buy'|'sell'|null). "
        "Коротко и по делу."
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
                    "На основе следующих данных верни JSON: "
                    "{decision: Buy|Sell|Hold, reason: string, regime: 'trend'|'flat'|null, trend_side: 'Buy'|'Sell'|null, mode: 'trend'|'flat'|null, side: 'buy'|'sell'|null}.\n"
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

    max_attempts = LLM_RETRIES + 1

    last_exception = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            log.info(f"[LLM][RETRY] attempt {attempt}")
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
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
                return {"decision": "Hold", "reason": "http_error", "regime": None, "trend_side": None, "mode": None, "side": None}
            await asyncio.sleep(LLM_BACKOFF * attempt)
        except Exception as e:
            log.debug(f"[LLM][ERROR DEBUG] {type(e).__name__}: {e}")
            last_exception = e
            if attempt == max_attempts:
                log.error(f"[LLM][ERR] {e}")
                elapsed = time.monotonic() - start_time
                log.info(f"[LLM←] decision=Hold reason=error time={elapsed:.2f}s")
                return {"decision": "Hold", "reason": "error", "regime": None, "trend_side": None, "mode": None, "side": None}
            await asyncio.sleep(LLM_BACKOFF * attempt)
    else:
        # Should not reach here, but in case
        elapsed = time.monotonic() - start_time
        log.info(f"[LLM←] decision=Hold reason=error time={elapsed:.2f}s")
        return {"decision": "Hold", "reason": "error", "regime": None, "trend_side": None, "mode": None, "side": None}

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    log.debug(f"[LLM][DEBUG] raw content before parse: {content}")

    # Пытаемся распарсить JSON, но если не удалось — вернём raw
    decision = {"decision": "Hold", "reason": "parse_error", "regime": None, "trend_side": None, "mode": None, "side": None, "raw": content}
    try:
        parsed = json.loads(content)
        log.debug(f"[LLM][DEBUG] parsed dict: {parsed}")
        if isinstance(parsed, dict):
            d = str(parsed.get("decision", "Hold"))
            reason = str(parsed.get("reason", "") or "").strip()
            regime = parsed.get("regime", None)
            trend_side = parsed.get("trend_side", None)

            mode = parsed.get("mode", None)
            side_field = parsed.get("side", None)

            if isinstance(regime, str):
                regime = regime.lower()
                if regime not in ("trend", "flat"):
                    regime = None
            else:
                regime = None

            if isinstance(trend_side, str):
                ts = trend_side.capitalize()
                trend_side = ts if ts in ("Buy", "Sell") else None
            else:
                trend_side = None

            if isinstance(mode, str):
                mode = mode.lower()
                if mode not in ("trend", "flat"):
                    mode = None
            else:
                mode = None

            if isinstance(side_field, str):
                s_norm = side_field.capitalize()
                side_field = s_norm if s_norm in ("Buy", "Sell") else None
            else:
                side_field = None

            # нормализация
            d_norm = d.capitalize()
            if d_norm not in ("Buy", "Sell", "Hold"):
                d_norm = "Hold"

            decision = {
                "decision": d_norm,
                "reason": reason or "ok",
                "regime": regime,
                "trend_side": trend_side,
                "mode": mode,
                "side": side_field,
            }
    except Exception:
        # не JSON — пытаемся угадать по ключевому слову
        lc = content.lower()
        if "buy" in lc:
            decision = {"decision": "Buy", "reason": "model_text", "regime": None, "trend_side": None, "mode": None, "side": None, "raw": content}
        elif "sell" in lc:
            decision = {"decision": "Sell", "reason": "model_text", "regime": None, "trend_side": None, "mode": None, "side": None, "raw": content}
        else:
            decision = {"decision": "Hold", "reason": "model_text", "regime": None, "trend_side": None, "mode": None, "side": None, "raw": content}
    log.debug(f"[LLM][DEBUG] chosen decision: {decision}")

    elapsed = time.monotonic() - start_time
    log.info(f"[LLM←] decision={decision['decision']} reason={decision.get('reason','')} time={elapsed:.2f}s")
    return decision