import os
import asyncio
import json

from bybit_client import _body_edges
from clients import openai_client, OPENAI_MODEL, USE_LOCAL_DECIDER
from helpers import ts_now, tg_send
from models import MarketData, AIDecision, Regime, Side, STATE
from prompt import prompt as pr
import pandas as pd
import re
from logger import log

# ---- Optional SSL relax (env BYBIT_VERIFY_SSL=false) ----
if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"

from dotenv import load_dotenv


# Локальная решалка
from local_ai import decide as local_decide

# ------------------ .env & logging ------------------
load_dotenv()








def ai_prompt(symbol: str, df: pd.DataFrame, md: MarketData, txt: str) -> str:
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    body_low, body_high = _body_edges(prev)
    p = {
        "symbol": symbol,
        "now": ts_now(),
        "last_price": md.last_price,
        "prev_high": float(prev["high"]),
        "prev_low": float(prev["low"]),
        "prev_body_low": body_low,
        "prev_body_high": body_high,
        "curr_open": float(curr["open"]),
        "curr_close": float(curr["close"]),
        "position": {"side": md.position.side.value, "size": md.position.size, "avg_price": md.position.avg_price},
        "balance_usdt": md.balance_usdt
    }
    return (
        "You are a trading decision engine. Return ONLY compact JSON, no extra text.\n"
        "Decide regime among: \"trend\", \"flat\", or \"hold\".\n"
        "If regime is trend or flat, also return trading side: \"Buy\" or \"Sell\".\n"
        "If regime is hold, side can be \"None\".\n"
        "Optionally you may include sl_ticks integer override.\n\n"
        f"Market snapshot:\n{json.dumps(p, ensure_ascii=False)}\n\n"
        "JSON schema:\n"
        "{ \"regime\": \"trend|flat|hold\", \"side\": \"Buy|Sell|None\", \"sl_ticks\": int|null, \"comment\": string }\n"
        f"{txt}"
    )

def _extract_json_block(text: str) -> str:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)
    if text.startswith("```") and text.endswith("```"):
        inner = text.strip('`')
        parts = inner.split('\n', 1)
        if len(parts) == 2:
            return parts[1]
    return text

def parse_ai(text: str) -> AIDecision:
    data = json.loads(_extract_json_block(text))
    return AIDecision(
        regime=Regime(data["regime"]),
        side=Side(data.get("side", "None")),
        sl_ticks=data.get("sl_ticks"),
        comment=data.get("comment", "")
    )

def parse_local_decision(obj) -> AIDecision:
    """
    Принимает как dict, так и объект (например, dataclass Decision из local_ai).
    Поля: regime, side, sl_ticks, comment. Поддерживает как Enum, так и строки.
    """
    def get_field(name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    r = get_field("regime", "hold")
    regime = r if isinstance(r, Regime) else Regime(str(r))

    s = get_field("side", "None")
    side = s if isinstance(s, Side) else Side(str(s))

    sl_ticks = get_field("sl_ticks", None)
    if sl_ticks is not None:
        try:
            sl_ticks = int(sl_ticks)
        except Exception:
            sl_ticks = None

    comment = get_field("comment", "")

    return AIDecision(regime=regime, side=side, sl_ticks=sl_ticks, comment=comment)

async def ask_ai(symbol: str, df: pd.DataFrame, md: MarketData) -> AIDecision:
    prompt = ai_prompt(symbol, df, md, txt=pr)
    log.info("[AI] request: %s", prompt.replace("\n", " ")[:500])
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content
        STATE.last_ai_text = text
        log.info("[AI] raw: %s", text)
        return parse_ai(text)
    except Exception as e:
        log.error("[AI] error: %s", e)
        await tg_send(f"🤖 <b>Ошибка ИИ</b>: {e}")
        return AIDecision(regime=Regime.HOLD, side=Side.NONE, comment="fallback")

async def get_decision(symbol: str, df: pd.DataFrame, md: MarketData) -> AIDecision:
    """
    Унифицированная точка получения решения — локально или через OpenAI.
    Вызовем ТОЛЬКО ПОСЛЕ закрытия минутной свечи (см. trading_loop).
    """
    if USE_LOCAL_DECIDER:
        try:
            obj = await asyncio.to_thread(local_decide, symbol, df, md)
            dec = parse_local_decision(obj)
            log.info("[LOCAL] %s", dec)
            return dec
        except Exception as e:
            log.error("[LOCAL] error: %s", e)
            await tg_send(f"🧠 Локальная решалка дала ошибку, переключаюсь в HOLD. {e}")
            return AIDecision(regime=Regime.HOLD, side=Side.NONE, comment="local error")
    else:
        return await ask_ai(symbol, df, md)