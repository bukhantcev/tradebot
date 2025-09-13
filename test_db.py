# test_db.py
# Расширенная БД и утилиты экспорта для последующего анализа ИИ

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json
import os

try:
    import pandas as pd  # для экспорта
except Exception:
    pd = None


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys = ON;

-- 1) Сводная таблица сделок (жизненный цикл позиции)
CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_open           DATETIME,            -- когда открыли позицию (биржой подтверждено)
    ts_close          DATETIME,            -- когда закрыли позицию (по факту)
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,       -- Buy / Sell
    qty               REAL NOT NULL,       -- контрактов
    avg_entry_price   REAL,                -- средняя цена входа (факт с биржи)
    avg_exit_price    REAL,                -- средняя цена выхода (факт с биржи)
    status            TEXT NOT NULL,       -- open / closed / canceled
    pnl_abs           REAL,                -- итог $-PNL по сделке
    pnl_pct           REAL,                -- итог % относительного риска/входа
    close_reason      TEXT,                -- stop_loss / take_profit / manual / liquidation / unknown
    strategy          TEXT,                -- имя стратегии/правила сигнала
    mode              TEXT,                -- testnet / mainnet
    risk_config_json  TEXT                 -- снапшот risk/config на момент входа (json)
);

-- 2) Сигналы (что именно подсказала стратегия)
CREATE TABLE IF NOT EXISTS signals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id          INTEGER,             -- может быть NULL, если позиция не открылась
    ts                DATETIME NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,       -- Buy / Sell
    entry_estimate    REAL,                -- ожидаемая цена входа по сигналу
    sl_estimate       REAL,                -- рассчитанный SL (виртуальный/биржевой)
    tp_estimate       REAL,                -- рассчитанный TP
    confidence        REAL,                -- уверенность
    strategy          TEXT,                -- имя стратегии
    features_json     TEXT,                -- фичи (RSI/MACD/режимы/уровни и т.п.) в сыром виде
    regime_1m         TEXT,                -- UP/DOWN/FLAT …
    regime_5m         TEXT,
    sr_1m_s           REAL,
    sr_1m_r           REAL,
    sr_5m_s           REAL,
    sr_5m_r           REAL,
    FOREIGN KEY(trade_id) REFERENCES trades(id) ON DELETE SET NULL
);

-- 3) Попытки размещения ордеров (чтобы разбирать 30208/10001 и т.п.)
CREATE TABLE IF NOT EXISTS order_attempts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id          INTEGER,             -- NULL до фактического открытия
    ts                DATETIME NOT NULL,
    intent            TEXT NOT NULL,       -- open / close / set_sl_tp / amend_sl_tp
    side              TEXT,                -- Buy/Sell (для open/close)
    order_type        TEXT,                -- Market/Limit/IOC и т.п.
    price             REAL,
    qty               REAL,
    stop_loss         REAL,
    take_profit       REAL,
    ret_code          INTEGER,             -- retCode bybit
    ret_msg           TEXT,                -- retMsg bybit
    payload_json      TEXT,                -- отправленный запрос
    response_json     TEXT,                -- ответ биржи
    FOREIGN KEY(trade_id) REFERENCES trades(id) ON DELETE SET NULL
);

-- 4) Исполнения/подтверждения (что реально стало позицией)
CREATE TABLE IF NOT EXISTS executions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id          INTEGER NOT NULL,
    ts                DATETIME NOT NULL,
    side              TEXT NOT NULL,
    avg_price         REAL NOT NULL,
    qty               REAL NOT NULL,
    liquidity         TEXT,                -- maker/taker если доступно
    fee               REAL,
    fee_currency      TEXT,
    raw_json          TEXT,
    FOREIGN KEY(trade_id) REFERENCES trades(id) ON DELETE CASCADE
);

-- 5) Снапшоты позиции/баланса (диагностика причин убытка: просадки, проскальзывания, маржа)
CREATE TABLE IF NOT EXISTS snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id          INTEGER,
    ts                DATETIME NOT NULL,
    last_price        REAL,
    position_side     TEXT,
    position_size     REAL,
    position_entry    REAL,
    unrealized_pnl    REAL,
    balance_available REAL,
    raw_json          TEXT,
    FOREIGN KEY(trade_id) REFERENCES trades(id) ON DELETE SET NULL
);

-- 6) Ошибки высокого уровня (исключения, сетевые ошибки, распарсенные возражения API)
CREATE TABLE IF NOT EXISTS errors (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                DATETIME NOT NULL,
    where_ctx         TEXT NOT NULL,       -- analyze_and_trade/place_order/fetch_position/telegram…
    message           TEXT NOT NULL,
    details_json      TEXT
);

-- 7) Индексирование ключевых полей для скорости аналитики
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, ts_open);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_order_attempts_ts ON order_attempts(ts);
CREATE INDEX IF NOT EXISTS idx_exec_trade ON executions(trade_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_trade_ts ON snapshots(trade_id, ts);

-- 8) Просмотр агрегированной метрики для выгрузки «на один клик»
CREATE VIEW IF NOT EXISTS v_trades_enriched AS
SELECT
    t.id                    AS trade_id,
    t.ts_open,
    t.ts_close,
    t.symbol,
    t.side,
    t.qty,
    t.avg_entry_price,
    t.avg_exit_price,
    t.status,
    t.pnl_abs,
    t.pnl_pct,
    t.close_reason,
    t.strategy,
    t.mode,
    t.risk_config_json,
    -- последний сигнал, привязанный к сделке
    s.ts                    AS signal_ts,
    s.entry_estimate,
    s.sl_estimate,
    s.tp_estimate,
    s.confidence,
    s.strategy              AS signal_strategy,
    s.features_json         AS signal_features,
    s.regime_1m,
    s.regime_5m,
    s.sr_1m_s, s.sr_1m_r,
    s.sr_5m_s, s.sr_5m_r
FROM trades t
LEFT JOIN signals s ON s.trade_id = t.id
WHERE t.ts_open IS NOT NULL;
"""


@dataclass
class RiskConfig:
    max_notional_usdt: float
    leverage: Optional[float] = None
    sr1_touch_pct: Optional[float] = None
    sr1_break_pct: Optional[float] = None
    notes: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})


class AnalyticsDB:
    """Класс для детального логирования и экспорта датасетов под ИИ."""
    def __init__(self, path: str = "analytics.db"):
        self.path = path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---------- базовые инсёрты ----------

    def new_trade(self,
                  ts_open: datetime,
                  symbol: str,
                  side: str,
                  qty: float,
                  avg_entry_price: float,
                  strategy: str,
                  mode: str,
                  risk_cfg: Optional[RiskConfig] = None) -> int:
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO trades(ts_open, symbol, side, qty, avg_entry_price, status, strategy, mode, risk_config_json)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)
            """, (ts_open, symbol, side, qty, avg_entry_price, strategy, mode,
                  risk_cfg.to_json() if risk_cfg else None))
            return cur.lastrowid

    def close_trade(self,
                    trade_id: int,
                    ts_close: datetime,
                    avg_exit_price: float,
                    pnl_abs: Optional[float],
                    pnl_pct: Optional[float],
                    close_reason: str):
        with self._conn() as c:
            c.execute("""
                UPDATE trades SET
                    ts_close = ?, avg_exit_price = ?, pnl_abs = ?, pnl_pct = ?, close_reason = ?, status = 'closed'
                WHERE id = ?
            """, (ts_close, avg_exit_price, pnl_abs, pnl_pct, close_reason, trade_id))

    def add_signal(self,
                   ts: datetime,
                   symbol: str,
                   side: str,
                   entry_estimate: Optional[float],
                   sl_estimate: Optional[float],
                   tp_estimate: Optional[float],
                   confidence: Optional[float],
                   strategy: str,
                   features: Dict[str, Any],
                   regime_1m: Optional[str] = None,
                   regime_5m: Optional[str] = None,
                   sr_1m: Optional[Tuple[float, float]] = None,
                   sr_5m: Optional[Tuple[float, float]] = None,
                   trade_id: Optional[int] = None) -> int:
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO signals(trade_id, ts, symbol, side, entry_estimate, sl_estimate, tp_estimate,
                                    confidence, strategy, features_json, regime_1m, regime_5m,
                                    sr_1m_s, sr_1m_r, sr_5m_s, sr_5m_r)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, ts, symbol, side, entry_estimate, sl_estimate, tp_estimate,
                confidence, strategy, json.dumps(features or {}),
                regime_1m, regime_5m,
                (sr_1m or (None, None))[0], (sr_1m or (None, None))[1],
                (sr_5m or (None, None))[0], (sr_5m or (None, None))[1],
            ))
            return cur.lastrowid

    def add_order_attempt(self,
                          ts: datetime,
                          intent: str,
                          side: Optional[str],
                          order_type: str,
                          price: Optional[float],
                          qty: Optional[float],
                          stop_loss: Optional[float],
                          take_profit: Optional[float],
                          ret_code: Optional[int],
                          ret_msg: Optional[str],
                          payload: Dict[str, Any],
                          response: Dict[str, Any],
                          trade_id: Optional[int] = None):
        with self._conn() as c:
            c.execute("""
                INSERT INTO order_attempts(trade_id, ts, intent, side, order_type, price, qty, stop_loss, take_profit,
                                           ret_code, ret_msg, payload_json, response_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, ts, intent, side, order_type, price, qty, stop_loss, take_profit,
                  ret_code, ret_msg, json.dumps(payload or {}), json.dumps(response or {})))

    def add_execution(self,
                      trade_id: int,
                      ts: datetime,
                      side: str,
                      avg_price: float,
                      qty: float,
                      liquidity: Optional[str] = None,
                      fee: Optional[float] = None,
                      fee_currency: Optional[str] = None,
                      raw_json: Optional[Dict[str, Any]] = None):
        with self._conn() as c:
            c.execute("""
                INSERT INTO executions(trade_id, ts, side, avg_price, qty, liquidity, fee, fee_currency, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, ts, side, avg_price, qty, liquidity, fee, fee_currency,
                  json.dumps(raw_json or {})))

    def add_snapshot(self,
                     trade_id: Optional[int],
                     ts: datetime,
                     last_price: Optional[float],
                     position_side: Optional[str],
                     position_size: Optional[float],
                     position_entry: Optional[float],
                     unrealized_pnl: Optional[float],
                     balance_available: Optional[float],
                     raw_json: Optional[Dict[str, Any]] = None):
        with self._conn() as c:
            c.execute("""
                INSERT INTO snapshots(trade_id, ts, last_price, position_side, position_size, position_entry,
                                      unrealized_pnl, balance_available, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, ts, last_price, position_side, position_size, position_entry,
                  unrealized_pnl, balance_available, json.dumps(raw_json or {})))

    def add_error(self, ts: datetime, where_ctx: str, message: str, details: Optional[Dict[str, Any]] = None):
        with self._conn() as c:
            c.execute("""
                INSERT INTO errors(ts, where_ctx, message, details_json)
                VALUES (?, ?, ?, ?)
            """, (ts, where_ctx, message, json.dumps(details or {})))

    # ---------- удобные хелперы связки сигнал → сделка ----------

    def link_signal_to_trade(self, signal_id: int, trade_id: int):
        with self._conn() as c:
            c.execute("UPDATE signals SET trade_id = ? WHERE id = ?", (trade_id, signal_id))

    # ---------- экспорт для ИИ ----------

    def export_enriched(self, out_path: str):
        """Экспорт представления v_trades_enriched в CSV/Parquet/JSONL (по расширению файла)."""
        if pd is None:
            raise RuntimeError("pandas не установлен. Установите pandas для экспорта.")

        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM v_trades_enriched", c)

        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".csv":
            df.to_csv(out_path, index=False)
        elif ext in (".parquet", ".pq"):
            df.to_parquet(out_path, index=False)
        elif ext in (".jsonl", ".ndjson"):
            df.to_json(out_path, orient="records", lines=True, force_ascii=False)
        else:
            # по умолчанию CSV
            df.to_csv(out_path, index=False)
        return len(df)

    def export_raw(self, table: str, out_path: str):
        """Экспорт любой таблицы как есть (raw snapshot)."""
        if pd is None:
            raise RuntimeError("pandas не установлен. Установите pandas для экспорта.")

        with self._conn() as c:
            df = pd.read_sql_query(f"SELECT * FROM {table}", c)

        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".csv":
            df.to_csv(out_path, index=False)
        elif ext in (".parquet", ".pq"):
            df.to_parquet(out_path, index=False)
        elif ext in (".jsonl", ".ndjson"):
            df.to_json(out_path, orient="records", lines=True, force_ascii=False)
        else:
            df.to_csv(out_path, index=False)
        return len(df)