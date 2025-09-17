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

async def send_to_openai_and_update_params(dump_path: str, notifier=None):
    if not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY is empty, skip tuning")
        return
    if notifier:
        await notifier.notify("📦 Дамп часа отправлен в ИИ на анализ...")

    with open(dump_path, "r", encoding="utf-8") as f:
        dump_text = f.read()

    # Trim too-large dumps to avoid 400 due to payload size
    MAX_CHARS = 200_000  # ~200k chars is a safe envelope for Responses API
    trimmed_note = ""
    if len(dump_text) > MAX_CHARS:
        dump_text = dump_text[-MAX_CHARS:]
        trimmed_note = "(усечено до последних ~200k символов)\n"

    import httpx
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    instruction = (
        "Ты — квант-аналитик-скальпер. Проанализируй журнал торговли за последний час и предложи НОВЫЕ параметры стратегии.\n"
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
        "text": {"format": "json"}
    }

    log.info(f"Sending dump to OpenAI: {dump_path}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as cli:
            r = await cli.post("https://api.openai.com/v1/responses", headers=headers, json=body)
            log.debug(f"OpenAI resp status={r.status_code} len={len(r.text)}")
            if r.status_code >= 400:
                # Log body and notify, then exit gracefully without raising
                err_text = r.text[:4000]
                log.error(f"OpenAI 4xx/5xx body: {err_text}")
                if notifier:
                    safe_err = html.escape(err_text)
                    await notifier.notify(f"❌ OpenAI отклонил запрос ({r.status_code}). Тело ответа:\n<pre>{safe_err}</pre>")
                return
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