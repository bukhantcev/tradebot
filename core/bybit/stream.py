import asyncio
import json
import logging
import ssl
import certifi
import websockets

log = logging.getLogger("WS")

PUBLIC_MAIN = "wss://stream.bybit.com/v5/public"
PUBLIC_TEST = "wss://stream-testnet.bybit.com/v5/public"

class BybitPublicStream:
    def __init__(self, symbol: str, interval: str, testnet: bool = True):
        self.symbol = symbol
        self.interval = interval
        self.ws_url = (PUBLIC_TEST if testnet else PUBLIC_MAIN) + "/linear"
        self.ws = None
        self._q = asyncio.Queue()
        self._hb_task = None
        self._runner_task = None
        self.last_price = None
        self.last_kline_close = None

    def queue(self):
        return self._q

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._runner_task = asyncio.create_task(self._runner())

    async def stop(self):
        if self._runner_task:
            self._runner_task.cancel()
        if self.ws:
            await self.ws.close()
        if self._hb_task:
            self._hb_task.cancel()

    async def _runner(self):
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        while True:
            try:
                log.info("[WS] Connecting to %s", self.ws_url)
                async with websockets.connect(self.ws_url, ssl=ssl_ctx, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                    self.ws = ws
                    await self._subscribe()
                    self._hb_task = asyncio.create_task(self._heartbeat())
                    async for msg in ws:
                        await self._on_raw(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("[WS] error: %s", e, exc_info=True)
                await asyncio.sleep(1)

    async def _subscribe(self):
        topics = [f"tickers.{self.symbol}", f"kline.{self.interval}.{self.symbol}"]
        req = {"op": "subscribe", "args": topics}
        await self.ws.send(json.dumps(req))
        log.info("[WS] Subscribed: %s", topics)

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(30)
            log.info("[WS] online %s | last_price=%s | kline_close=%s | q=%d",
                     self.symbol, self.last_price, self.last_kline_close, self._q.qsize())

    async def _on_raw(self, msg: str):
        j = json.loads(msg)
        # Debug raw (можно включать точечно при необходимости)
        # print(json.dumps(j, ensure_ascii=False))
        topic = j.get("topic")
        if topic and topic.startswith("tickers."):
            d = j.get("data", {})
            last = d.get("lastPrice") or d.get("price") or d.get("markPrice")
            if last:
                try:
                    self.last_price = float(last)
                except:  # noqa
                    pass
            await self._q.put({"type": "ticker", "data": d})
        elif topic and topic.startswith("kline."):
            bars = j.get("data") or []
            for b in bars:
                closed = bool(b.get("confirm"))
                if closed:
                    self.last_kline_close = float(b.get("close") or b.get("c"))
                    # нормализуем клик для стратегий
                    await self._q.put({"type": "kline", "data": b})