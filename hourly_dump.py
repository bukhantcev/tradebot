import os, json, time, shutil, tempfile, asyncio, logging, html
import re
from typing import Dict, Any, List
from config import DUMP_DIR, OPENAI_API_KEY, OPENAI_MODEL
from params_store import save_params

log = logging.getLogger("dump")
os.makedirs(DUMP_DIR, exist_ok=True)

class HourlyDump:
    def __init__(self):
        self.reset()

    def reset(self):
        self.meta = {}
        self.m1: List[dict] = []
        self.m5: List[dict] = []
        self.orderbook_snaps: List[dict] = []
        self.densities: List[dict] = []
        self.signals: List[dict] = []
        self.trades: List[dict] = []
        self.telemetry: Dict[str, Any] = {"latency_ms": {}, "errors": []}
        self.params_used: Dict[str, Any] = {}

    def add_candle(self, interval: str, c: dict):
        if interval == "1": self.m1.append(c)
        elif interval == "5": self.m5.append(c)

    def add_signal(self, s: dict): self.signals.append(s)
    def add_trade(self, t: dict): self.trades.append(t)
    def set_params_used(self, p: dict): self.params_used = p
    def set_meta(self, symbol: str, account: str, start_ts: int, end_ts: int):
        self.meta = {"symbol": symbol, "account": account,
                     "period": f"{time.strftime('%Y-%m-%dT%H:00:00Z', time.gmtime(start_ts/1000))}/"
                               f"{time.strftime('%Y-%m-%dT%H:00:00Z', time.gmtime(end_ts/1000))}"}

    def _safe_write(self, path: str, obj: dict):
        tmp_fd, tmp_path = tempfile.mkstemp()
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        shutil.move(tmp_path, path)

    def flush_to_file(self) -> str:
        fname = time.strftime("dump_%Y%m%d_%H00.json", time.gmtime())
        fpath = os.path.join(DUMP_DIR, fname)
        payload = {
            "meta": self.meta,
            "market": {"m1": self.m1, "m5": self.m5, "orderbook": {"snapshots": self.orderbook_snaps, "densities": self.densities}},
            "signals": self.signals, "trades": self.trades,
            "telemetry": self.telemetry, "params_used": self.params_used
        }
        self._safe_write(fpath, payload)
        log.info(f"Dump written to {fpath}")
        return fpath


# Smart shrinker: keep only recent tails of arrays and compact JSON to avoid TPM limits
def _shrink_dump_payload(dump_text: str, *, m1_max=300, m5_max=200, ob_snaps_max=10, dens_max=60, signals_max=120, trades_max=120) -> str:
    try:
        j = json.loads(dump_text)
        mkt = j.get("market", {})
        # slice candles
        if isinstance(mkt.get("m1"), list):
            mkt["m1"] = mkt["m1"][-m1_max:]
        if isinstance(mkt.get("m5"), list):
            mkt["m5"] = mkt["m5"][-m5_max:]
        ob = mkt.get("orderbook", {})
        if isinstance(ob.get("snapshots"), list):
            ob["snapshots"] = ob["snapshots"][-ob_snaps_max:]
        if isinstance(ob.get("densities"), list):
            ob["densities"] = ob["densities"][-dens_max:]
        # signals / trades tails
        if isinstance(j.get("signals"), list):
            j["signals"] = j["signals"][-signals_max:]
        if isinstance(j.get("trades"), list):
            j["trades"] = j["trades"][-trades_max:]
        # telemetry: keep only last errors and last 10 latency entries per key
        tel = j.get("telemetry", {})
        if isinstance(tel.get("errors"), list):
            tel["errors"] = tel["errors"][-20:]
        if isinstance(tel.get("latency_ms"), dict):
            tel["latency_ms"] = {k: v[-10:] if isinstance(v, list) else v for k, v in tel["latency_ms"].items()}
        j["telemetry"] = tel
        # write compact JSON (no spaces) to reduce tokens
        return json.dumps(j, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Fallback: if not JSON, return original text
        return dump_text


async def send_to_openai_and_update_params(dump_path: str, notifier=None):
    if not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY is empty, skip tuning")
        return
    if notifier:
        await notifier.notify("📦 Дамп часа отправлен в ИИ на анализ...")

    with open(dump_path, "r", encoding="utf-8") as f:
        dump_text = f.read()

    # Smart shrink to avoid TPM/token limits
    dump_text = _shrink_dump_payload(dump_text)
    trimmed_note = "(умно сжато для лимитов токенов)\n"

    import httpx
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    instruction = (
        "Ты — квант-аналитик-скальпер. Проанализируй журнал торговли за последний час и предложи НОВЫЕ параметры стратегии. Анализируй убыточные позиции и "
        "пытайся с помощью настройки параметров избежать их в будущем! твоя задача настроить систему на получение максимальной прибыли с минимальными убытками!\n"
        "Верни ТОЛЬКО JSON строго формата params.json (валидный JSON-объект) — без комментариев и лишнего текста."
    )

    body = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": "Ты отвечаешь исключительно валидным JSON без преамбул и пояснений."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": f"Исходный журнал ниже {trimmed_note}"},
                    {"type": "input_text", "text": dump_text}
                ]
            }
        ],
        "text": {
            "format": {"type": "json_object"}
        }
    }

    log.info(f"Sending dump to OpenAI: {dump_path}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as cli:
            r = await cli.post("https://api.openai.com/v1/responses", headers=headers, json=body)
            log.debug(f"OpenAI resp status={r.status_code} len={len(r.text)}")
            if r.status_code >= 400:
                err_text = r.text[:4000]
                try:
                    err_json = r.json()
                except Exception:
                    err_json = {}
                err_obj = err_json.get("error") or {}
                msg = (err_obj.get("message") or "")
                param = (err_obj.get("param") or "")

                # Handle TPM/rate limit by shrinking further and retrying once
                if "tokens per min" in msg.lower() or "rate_limit_exceeded" in (err_obj.get("code") or "").lower():
                    log.warning(f"Retrying OpenAI with extra shrink due to rate limit: {msg}")
                    smaller = _shrink_dump_payload(dump_text, m1_max=180, m5_max=120, ob_snaps_max=6, dens_max=40, signals_max=80, trades_max=80)
                    body_small = dict(body)
                    # Replace only the user dump content part
                    body_small["input"][1]["content"][2]["text"] = smaller
                    r2 = await cli.post("https://api.openai.com/v1/responses", headers=headers, json=body_small)
                    log.debug(f"OpenAI retry(resp-size) status={r2.status_code} len={len(r2.text)}")
                    if r2.status_code >= 400:
                        err_text2 = r2.text[:4000]
                        log.error(f"OpenAI retry 4xx/5xx body: {err_text2}")
                        if notifier:
                            safe_err2 = html.escape(err_text2)
                            await notifier.notify(f"❌ OpenAI отклонил повторный запрос ({r2.status_code}). Тело ответа:\n<pre>{safe_err2}</pre>")
                        return
                    j = r2.json()
                    # proceed with parsed response
                    goto_after_error_handling = True
                else:
                    goto_after_error_handling = False

                if not goto_after_error_handling:
                    # If the model/version doesn't support text.format or the type is wrong, retry without JSON mode
                    if (param in ("text.format", "response_format")) or ("not supported" in msg.lower()) or ("invalid type for 'text.format'" in msg.lower()):
                        log.warning(f"Retrying OpenAI without text.format due to error: {msg or err_text}")
                        body2 = {k: v for k, v in body.items() if k != "text"}
                        r2 = await cli.post("https://api.openai.com/v1/responses", headers=headers, json=body2)
                        log.debug(f"OpenAI retry resp status={r2.status_code} len={len(r2.text)}")
                        if r2.status_code >= 400:
                            err_text2 = r2.text[:4000]
                            log.error(f"OpenAI retry 4xx/5xx body: {err_text2}")
                            if notifier:
                                safe_err2 = html.escape(err_text2)
                                await notifier.notify(f"❌ OpenAI отклонил повторный запрос ({r2.status_code}). Тело ответа:\n<pre>{safe_err2}</pre>")
                            return
                        j = r2.json()
                    else:
                        log.error(f"OpenAI 4xx/5xx body: {err_text}")
                        if notifier:
                            safe_err = html.escape(err_text)
                            await notifier.notify(f"❌ OpenAI отклонил запрос ({r.status_code}). Тело ответа:\n<pre>{safe_err}</pre>")
                        return
            else:
                j = r.json()
    except httpx.RequestError as e:
        log.exception("OpenAI request failed")
        if notifier:
            await notifier.notify(f"❌ Сбой запроса к OpenAI: {html.escape(str(e))}")
        return

    text = _extract_json_text_from_openai(j)

    if notifier:
        raw_preview = html.escape(json.dumps(j, ensure_ascii=False)[:1800])
        txt_preview = html.escape((text or "(пусто)")[:1800])
        await notifier.notify(f"🧠 Ответ ИИ (raw):\n<pre>{raw_preview}</pre>")
        await notifier.notify(f"🧠 Ответ ИИ (parsed):\n<pre>{txt_preview}</pre>")

    try:
        new_params = json.loads(text)
        save_params(new_params)
        if notifier:
            await notifier.notify("✅ Параметры обновлены и применены")
    except Exception:
        log.exception("OpenAI params parsing failed")
        if notifier:
            await notifier.notify("⚠️ Ошибка разбора ответа от ИИ — параметры не обновлены")

    # удаляем отправленный дамп
    try:
        os.remove(dump_path)
    except Exception:
        log.exception("Failed to remove dump file")


# Helper to extract JSON string from OpenAI response
def _extract_json_text_from_openai(resp: dict) -> str:
    """
    Try to extract JSON string from various OpenAI response formats.
    """
    txt = resp.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    try:
        out = resp.get("output") or []
        parts = []
        for item in out:
            for c in item.get("content", []):
                t = c.get("text")
                if isinstance(t, str):
                    parts.append(t)
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass

    try:
        ch = resp.get("choices")
        if ch and "message" in ch[0] and "content" in ch[0]["message"]:
            t = ch[0]["message"]["content"]
            if isinstance(t, str) and t.strip():
                return t.strip()
    except Exception:
        pass

    dump = json.dumps(resp, ensure_ascii=False)
    m = re.search(r"```json\s*(\{.*?\})\s*```", dump, re.DOTALL)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"(\{(?:[^{}]|(?1))*\})", dump)
    if m2:
        return m2.group(1).strip()

    return ""