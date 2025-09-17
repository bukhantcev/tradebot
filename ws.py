import json, asyncio, websockets, logging
from typing import Callable, Dict, Any, Optional
from config import BYBIT_TESTNET

WS_PUBLIC_MAIN = "wss://stream.bybit.com/v5/public/linear"
WS_PUBLIC_TEST = "wss://stream-testnet.bybit.com/v5/public/linear"

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