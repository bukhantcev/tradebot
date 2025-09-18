import os
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import aiosqlite
import pandas as pd

from config import DB_PATH, PARQUET_DIR, SYMBOL
from bybit_client import BybitClient

log = logging.getLogger("DATA")

@dataclass
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    confirm: bool
    @property
    def dt_minute(self) -> int:
        return (self.ts // 60000) * 60000

DDL_1M = """
CREATE TABLE IF NOT EXISTS bars_1m(
    symbol TEXT NOT NULL,
    ts_ms INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    turnover REAL NOT NULL,
    confirm INTEGER NOT NULL,
    PRIMARY KEY(symbol, ts_ms)
);
"""
DDL_5M = """
CREATE TABLE IF NOT EXISTS bars_5m(
    symbol TEXT NOT NULL,
    ts_ms INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    turnover REAL NOT NULL,
    PRIMARY KEY(symbol, ts_ms)
);
"""

class DataStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def open(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute(DDL_1M)
        await self._db.execute(DDL_5M)
        await self._db.commit()
        log.debug("[DB] open OK")

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None
            log.debug("[DB] close OK")

    async def upsert_bar_1m(self, symbol: str, bar: Bar):
        assert self._db
        q = """
        INSERT INTO bars_1m(symbol, ts_ms, open, high, low, close, volume, turnover, confirm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts_ms) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            turnover=excluded.turnover,
            confirm=excluded.confirm;
        """
        await self._db.execute(q, (symbol, bar.dt_minute, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.turnover, int(bar.confirm)))
        log.debug(f"[DB][1m] upsert ts={bar.dt_minute} c={bar.close} conf={bar.confirm}")

    async def upsert_bar_5m(self, symbol: str, ts_ms_5m: int, ohlcv: Tuple[float, float, float, float, float, float]):
        assert self._db
        o, h, l, c, v, t = ohlcv
        q = """
        INSERT INTO bars_5m(symbol, ts_ms, open, high, low, close, volume, turnover)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts_ms) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            turnover=excluded.turnover;
        """
        await self._db.execute(q, (symbol, ts_ms_5m, o, h, l, c, v, t))
        log.debug(f"[DB][5m] upsert ts5={ts_ms_5m} c={c}")

    async def commit(self):
        assert self._db
        await self._db.commit()
        log.debug("[DB] commit")

def aggregate_5m(rows: List[Tuple[int, float, float, float, float, float, float]]) -> Dict[int, Tuple[float, float, float, float, float, float]]:
    buckets: Dict[int, List[Tuple[int, float, float, float, float, float, float]]] = {}
    for ts_ms, o, h, l, c, v, t in rows:
        ts5 = (ts_ms // (5*60*1000)) * (5*60*1000)
        buckets.setdefault(ts5, []).append((ts_ms, o, h, l, c, v, t))
    out: Dict[int, Tuple[float, float, float, float, float, float]] = {}
    for ts5, arr in buckets.items():
        arr.sort(key=lambda x: x[0])
        o = arr[0][1]; h = max(x[2] for x in arr); l = min(x[3] for x in arr)
        c = arr[-1][4]; v = sum(x[5] for x in arr); t = sum(x[6] for x in arr)
        out[ts5] = (o, h, l, c, v, t)
    log.debug(f"[AGG] 5m buckets={len(out)} from_rows={len(rows)}")
    return out

def write_parquet_daily(df: pd.DataFrame, day_utc: str, symbol: str, timeframe: str):
    if df.empty:
        log.debug("[PARQUET] skip empty")
        return
    os.makedirs(PARQUET_DIR, exist_ok=True)
    fn = os.path.join(PARQUET_DIR, f"{symbol}_{timeframe}_{day_utc}.parquet")
    dff = df.copy()
    dff["ts"] = pd.to_datetime(dff["ts_ms"], unit="ms", utc=True)
    dff.set_index("ts", inplace=True)
    dff.to_parquet(fn, engine="pyarrow")
    log.debug(f"[PARQUET] written {fn} rows={len(dff)}")

def parse_kline_payload(msg: Dict[str, Any]) -> List[Bar]:
    bars: List[Bar] = []
    if "data" not in msg:
        return bars
    for it in msg["data"]:
        try:
            b = Bar(
                ts=int(it.get("start") or it.get("t") or 0),
                open=float(it["open"]),
                high=float(it["high"]),
                low=float(it["low"]),
                close=float(it["close"]),
                volume=float(it.get("volume") or it.get("v") or 0.0),
                turnover=float(it.get("turnover") or it.get("q") or 0.0),
                confirm=bool(it.get("confirm", False)),
            )
            if b.ts > 0:
                bars.append(b)
        except Exception as e:
            log.warning(f"[WS][PARSE] skip item: {e} item={it}")
    log.debug(f"[WS][PARSE] bars={len(bars)}")
    return bars

class DataManager:
    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
        self.client = BybitClient()
        self.store = DataStore()
        self._queue: "asyncio.Queue[Bar]" = asyncio.Queue(maxsize=2000)
        self._stop = asyncio.Event()
        self._last_snapshot_day: Optional[str] = None

    async def start(self):
        asyncio.current_task().set_name("data")
        await self.store.open()
        consumer_task = asyncio.create_task(self._consumer_loop(), name="data-consumer")
        agg_task = asyncio.create_task(self._aggregator_loop(), name="data-aggregator")
        ws_task = asyncio.create_task(self._ws_loop(), name="data-ws")
        log.debug("[DATA] start")
        try:
            await asyncio.wait({consumer_task, agg_task, ws_task}, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            self._stop.set()
            for t in (consumer_task, agg_task, ws_task):
                if not t.done():
                    t.cancel()
            await self.store.commit()
            await self.store.close()
            log.debug("[DATA] stop")

    async def stop(self):
        self._stop.set()

    async def _ws_loop(self):
        topic = f"kline.1.{self.symbol}"
        while not self._stop.is_set():
            try:
                async for msg in self.client.ws_subscribe(self.client.ws_public_url, [topic], auth=False):
                    if self._stop.is_set():
                        break
                    if isinstance(msg, dict) and msg.get("topic", "").startswith("kline."):
                        bars = parse_kline_payload(msg)
                        for b in bars:
                            try:
                                self._queue.put_nowait(b)
                            except asyncio.QueueFull:
                                _ = self._queue.get_nowait()
                                await self._queue.put(b)
                    elif isinstance(msg, dict) and msg.get("op") == "subscribe":
                        log.debug(f"[WS] {msg}")
            except Exception as e:
                log.error(f"[WS] reconnect: {e}")
                await asyncio.sleep(2.0)

    async def _consumer_loop(self):
        batch = 0
        last_commit = time.time()
        while not self._stop.is_set():
            try:
                b: Bar = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                b = None  # type: ignore
            if b:
                await self.store.upsert_bar_1m(self.symbol, b)
                batch += 1
            if batch >= 50 or (time.time() - last_commit) >= 1.0:
                await self.store.commit()
                log.debug(f"[DB] batch_commit n={batch}")
                batch = 0
                last_commit = time.time()

    async def _aggregator_loop(self):
        while not self._stop.is_set():
            try:
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - 120 * 60 * 1000
                q = """
                SELECT ts_ms, open, high, low, close, volume, turnover
                FROM bars_1m
                WHERE symbol=? AND confirm=1 AND ts_ms>=?
                ORDER BY ts_ms ASC
                """
                async with self.store._db.execute(q, (self.symbol, start_ms)) as cur:
                    rows = await cur.fetchall()
                agg = aggregate_5m(rows)
                if agg:
                    for ts5, ohlcv in agg.items():
                        await self.store.upsert_bar_5m(self.symbol, ts5, ohlcv)
                    await self.store.commit()
                # parquet snapshot once per day
                utc_day = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
                if self._last_snapshot_day != utc_day:
                    async with self.store._db.execute(
                        "SELECT ts_ms, open, high, low, close, volume, turnover FROM bars_1m WHERE symbol=? AND DATE(ts_ms/1000,'unixepoch')=DATE('now') ORDER BY ts_ms"
                        , (self.symbol,)
                    ) as cur:
                        day_rows = await cur.fetchall()
                    df = pd.DataFrame(day_rows, columns=["ts_ms","open","high","low","close","volume","turnover"])
                    write_parquet_daily(df, utc_day, self.symbol, "1m")
                    self._last_snapshot_day = utc_day
            except Exception as e:
                log.error(f"[AGG] {e}")
            await asyncio.sleep(10.0)