import os, json, time, shutil, tempfile, asyncio
from typing import Dict, Any, List
from config import DUMP_DIR, OPENAI_API_KEY, OPENAI_MODEL, PARAMS_PATH
from params_store import save_params

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

    def add_signal(self, s: dict):
        self.signals.append(s)

    def add_trade(self, t: dict):
        self.trades.append(t)

    def set_params_used(self, p: dict):
        self.params_used = p

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
            "candles": {"m1": self.m1, "m5": self.m5},
            "orderbook": {"snapshots": self.orderbook_snaps, "densities": self.densities},
            "signals": self.signals, "trades": self.trades,
            "telemetry": self.telemetry, "params_used": self.params_used
        }
        self._safe_write(fpath, payload)
        return fpath

async def send_to_openai_and_update_params(dump_path: str, notifier=None):
    if notifier:
        await notifier.notify("📦 Дамп часа отправлен в ИИ на анализ...")
    if not OPENAI_API_KEY:
        return  # скипаем если нет ключа
    with open(dump_path, "r", encoding="utf-8") as f:
        dump_text = f.read()

    import httpx
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "Ты — квант-аналитик. По JSON-логам скальпера за последний час предложи новые значения параметров стратегии.\n"
        "Ответ верни ТОЛЬКО в виде JSON со структурой как в params.json. Не добавляй комментарии."
    )
    body = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "You are a helpful quant."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": dump_text}
        ]
    }
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post("https://api.openai.com/v1/responses", headers=headers, json=body)
        r.raise_for_status()
        j = r.json()
    # Пытаемся достать JSON из ответа
    text = ""
    try:
        text = j["output"][0]["content"][0]["text"]
    except Exception:
        # запасной путь под различные форматы
        text = json.dumps({"error": "unexpected_openai_response"}, ensure_ascii=False)

    try:
        new_params = json.loads(text)
        save_params(new_params)
        if notifier:
            await notifier.notify("🧠 Параметры обновлены от ИИ")
    except Exception:
        if notifier:
            await notifier.notify("⚠️ Ошибка разбора ответа от ИИ")

    # удаляем отправленный дамп
    try:
        os.remove(dump_path)
    except Exception:
        pass