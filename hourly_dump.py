import os, json, time, shutil, tempfile, asyncio, logging, html
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

    import httpx
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "Ты — квант-аналитик-скальпер. Изучив JSON-лог, предложи НОВЫЕ параметры стратегии.\n"
        "Верни ТОЛЬКО JSON того же формата, что и params.json. Без комментариев и текста."
    )
    body = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "You are a helpful quant and return pure JSON."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": dump_text}
        ]
    }
    log.info(f"Sending dump to OpenAI: {dump_path}")
    async with httpx.AsyncClient(timeout=90) as cli:
        r = await cli.post("https://api.openai.com/v1/responses", headers=headers, json=body)
        log.debug(f"OpenAI resp status={r.status_code} len={len(r.text)}")
        r.raise_for_status()
        j = r.json()

    # Попытка достать чистый JSON
    text = ""
    try:
        # unified Responses API format may vary — пробуем наиболее частый
        text = j["output"][0]["content"][0]["text"]
    except Exception:
        # запасной путь — просто json.dumps на всякий случай
        text = ""

    if notifier:
        shown = (text or "")[:2000] or "(пустой ответ)"
        # показываем ответ ИИ целиком (усечённо) в TG
        import html as _html
        await notifier.notify(f"🧠 Ответ ИИ:\n<pre>{_html.escape(shown)}</pre>")

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