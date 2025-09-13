import aiosqlite
from typing import List, Tuple
from config import CFG
from log import log

INIT_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS candles_1m (
    ts INTEGER PRIMARY KEY,              -- ms
    open REAL, high REAL, low REAL, close REAL, volume REAL
);

CREATE TABLE IF NOT EXISTS candles_5m (
    ts INTEGER PRIMARY KEY,              -- ms
    open REAL, high REAL, low REAL, close REAL, volume REAL
);
"""

class DB:
    def __init__(self, path: str):
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def open(self):
        self._conn = await aiosqlite.connect(self.path)
        await self._conn.executescript(INIT_SQL)
        await self._conn.commit()
        log.info("[DB] opened %s", self.path)

    async def close(self):
        if self._conn:
            await self._conn.close()
            log.info("[DB] closed")

    async def upsert_candles(self, tf: str, rows: List[Tuple[int, float, float, float, float, float]]):
        """
        rows: list of (ts_ms, open, high, low, close, volume)
        """
        table = "candles_1m" if tf == "1" else "candles_5m"
        sql = f"""
        INSERT INTO {table} (ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ts) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume;
        """
        await self._conn.executemany(sql, rows)
        await self._conn.commit()

    async def fetch_last_n(self, tf: str, n: int) -> list[tuple]:
        table = "candles_1m" if tf == "1" else "candles_5m"
        sql = f"SELECT ts, open, high, low, close, volume FROM {table} ORDER BY ts DESC LIMIT ?"
        cur = await self._conn.execute(sql, (n,))
        out = await cur.fetchall()
        await cur.close()
        return list(reversed(out))

    async def trim_to(self, tf: str, n: int):
        table = "candles_1m" if tf == "1" else "candles_5m"
        # keep only last n rows
        sql = f"""
        DELETE FROM {table}
        WHERE ts NOT IN (
            SELECT ts FROM {table} ORDER BY ts DESC LIMIT ?
        );
        """
        await self._conn.execute(sql, (n,))
        await self._conn.commit()

db = DB(CFG.db_path)