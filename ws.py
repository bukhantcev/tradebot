import json, asyncio, websockets, logging, time
import httpx
from typing import Callable, Dict, Any, Optional
from config import BYBIT_TESTNET

WS_PUBLIC_MAIN = "wss://stream.bybit.com/v5/public/linear"
WS_PUBLIC_TEST = "wss://stream-testnet.bybit.com/v5/public/linear"

async def preload_klines(symbol: str, on_kline, counts: dict, *, testnet: bool, category: str = "linear"):
    """Fetch historical klines from Bybit REST v5 and emit them into on_kline(interval, item).
    Uses explicit time window and multiple batches if needed. Emits oldest→newest with confirm=True.
    counts example: {"1": 300, "5": 200}
    """
    base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as cli:
        log_ws = logging.getLogger("ws")
        now_ms = int(time.time() * 1000)
        for interval, need in counts.items():
            if not need or need <= 0:
                continue
            try:
                got = 0
                acc: list[dict] = []
                # interval minutes → ms
                try:
                    i_min = int(interval)
                except Exception:
                    i_min = 1
                step_ms = i_min * 60_000
                # request a window wide enough for 'need' candles (+20% buffer)
                window_ms = int(need * step_ms * 1.2)
                start_ms = now_ms - max(window_ms, step_ms * need)
                # Bybit returns newest→oldest; we may need a couple of pages to be safe
                page_limit = min(max(need, 50), 1000)

                for _ in range(5):  # up to 5 pages
                    params = {
                        "category": category,
                        "symbol": symbol,
                        "interval": interval,
                        "limit": str(page_limit),
                        "start": str(start_ms),
                    }
                    url = f"{base}/v5/market/kline"
                    log_ws.info(f"[PRELOAD] GET {url} params={params}")
                    r = await cli.get(url, params=params)
                    log_ws.info(f"[PRELOAD] HTTP {r.status_code} len={len(r.text)} interval={interval}")
                    if r.status_code != 200:
                        log_ws.warning(f"[PRELOAD] kline {interval}m HTTP {r.status_code}: {r.text[:200]}")
                        break
                    data = r.json().get("result", {}).get("list") or []
                    if not data:
                        log_ws.warning(f"[PRELOAD] empty list for interval={interval} (symbol={symbol}) start={start_ms}")
                        break
                    # newest→oldest → oldest→newest
                    data = list(reversed(data))
                    # normalize and extend accumulator
                    appended = 0
                    for row in data:
                        try:
                            if isinstance(row, list) and len(row) >= 8:
                                item = {
                                    "start": int(row[0]),
                                    "end": int(row[1]),
                                    "open": float(row[2]),
                                    "high": float(row[3]),
                                    "low": float(row[4]),
                                    "close": float(row[5]),
                                    "volume": float(row[6]),
                                    "turnover": float(row[7]),
                                    "confirm": True,
                                }
                            elif isinstance(row, dict) and {"start","end","open","high","low","close"}.issubset(row.keys()):
                                row["confirm"] = True
                                item = row
                            else:
                                continue
                            acc.append(item)
                            appended += 1
                        except Exception:
                            continue
                    got = len(acc)
                    log_ws.info(f"[PRELOAD] page got={len(data)} appended={appended} acc={got} need={need} interval={interval}")
                    if got >= need:
                        break
                    # widen the window further back for next page (safe if acc is empty)
                    if acc:
                        start_ms = acc[0]["start"] - (step_ms * need)
                    else:
                        start_ms = start_ms - (step_ms * need)

                if got == 0:
                    log_ws.warning(f"[PRELOAD] no candles accumulated for {interval}m (symbol={symbol})")
                    continue
                # trim to the last 'need' items and emit
                acc = acc[-need:]
                for item in acc:
                    try:
                        on_kline(interval, item)
                    except Exception:
                        log_ws.exception("[PRELOAD] on_kline failed")
                log_ws.info(f"[PRELOAD] emitted {len(acc)} candles for {interval}m (symbol={symbol})")
            except Exception:
                log_ws.exception(f"[PRELOAD] failed to fetch {interval}m klines")

class PublicWS:
    def __init__(self, symbol: str, on_kline: Callable[[str, Dict[str, Any]], None], intervals=("1", "5"), deliver_only_confirm: bool = True):
        self.url = WS_PUBLIC_TEST if BYBIT_TESTNET else WS_PUBLIC_MAIN
        self.symbol = symbol
        self.on_kline = on_kline
        self.intervals = intervals
        self.deliver_only_confirm = deliver_only_confirm
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        self.log = logging.getLogger("ws")

    async def connect(self):
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stopped.set()
        if self._task:
            await asyncio.wait([self._task], timeout=2)

    async def _run(self):
        subs = [f"kline.{itv}.{self.symbol}" for itv in self.intervals]
        self.log.info(f"WS connect to {self.url} subs={subs}")
        while not self._stopped.is_set():
            try:
                async with websockets.connect(self.url, ping_interval=15, ping_timeout=10) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": subs}))
                    async for msg in ws:
                        j = json.loads(msg)
                        topic = j.get("topic", "")
                        if topic.startswith("kline."):
                            interval = topic.split(".")[1]
                            for itm in j.get("data", []):
                                # confirm may come as bool or string; normalize to bool
                                confirm = itm.get("confirm")
                                if isinstance(confirm, str):
                                    confirm = confirm.lower() == "true"
                                confirm = bool(confirm)

                                if not confirm:
                                    # interim updates while the candle is forming
                                    self.log.debug(f"[KLINE_TICK] {interval}m confirm=False start={itm.get('start')} close={itm.get('close')}")
                                    if self.deliver_only_confirm:
                                        continue

                                # Log on close of a candle (especially 1m)
                                if confirm and interval == "1":
                                    self.log.info(f"[KLINE_CLOSE] 1m start={itm.get('start')} end={itm.get('end')} close={itm.get('close')} vol={itm.get('volume')} turnover={itm.get('turnover')}")

                                self.on_kline(interval, itm)
            except Exception:
                self.log.exception("WS error")
                await asyncio.sleep(1.0)