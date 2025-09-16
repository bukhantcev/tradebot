import json, asyncio, websockets
from typing import Callable, Dict, Any, Optional
from config import BYBIT_TESTNET

WS_PUBLIC_MAIN = "wss://stream.bybit.com/v5/public/linear"
WS_PUBLIC_TEST = "wss://stream-testnet.bybit.com/v5/public/linear"

class PublicWS:
    def __init__(self, symbol: str, on_kline: Callable[[str, Dict[str, Any]], None], intervals=("1", "5")):
        self.url = WS_PUBLIC_TEST if BYBIT_TESTNET else WS_PUBLIC_MAIN
        self.symbol = symbol
        self.on_kline = on_kline
        self.intervals = intervals
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._task = None
        self._stopped = asyncio.Event()

    async def connect(self):
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stopped.set()
        if self._task:
            await asyncio.wait([self._task], timeout=2)

    async def _run(self):
        subs = [f"kline.{itv}.{self.symbol}" for itv in self.intervals]
        while not self._stopped.is_set():
            try:
                async with websockets.connect(self.url, ping_interval=15, ping_timeout=10) as ws:
                    self._ws = ws
                    sub_msg = {"op": "subscribe", "args": subs}
                    await ws.send(json.dumps(sub_msg))
                    async for msg in ws:
                        j = json.loads(msg)
                        if j.get("topic","").startswith("kline."):
                            data = j.get("data", [])
                            for itm in data:
                                interval = j["topic"].split(".")[1]
                                self.on_kline(interval, itm)
            except Exception:
                await asyncio.sleep(1.0)