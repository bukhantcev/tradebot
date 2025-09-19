# trader.py
import logging
from typing import Optional, Dict, Any
import os
import asyncio
from config import SYMBOL, RISK_PCT, LEVERAGE
from bybit_client import BybitClient

# все импорты логики — внутрь из пакета trader_mod
from trader_mod import account as acc
from trader_mod import risk as rk
from trader_mod import tpsl as ts
from trader_mod import extremes as ex
from trader_mod import utils as ut

log = logging.getLogger("TRADER")


class Trader:
    __lp_fail = 0

    """
    Минималистичный трейдер (фасад):
    - refresh_equity(): получить equity (USDT)
    - ensure_filters(): разово подтянуть tickSize/qtyStep/min
    - ensure_leverage(): ставит плечо только если отличается
    - open_market(side, signal): маркет-вход + TPSL
    - close_market(side, qty): маркет-выход
    Опциональный notifier: объект с notify(str).
    """

    def __init__(
        self,
        client: BybitClient,
        symbol: str = SYMBOL,
        risk_pct: float = RISK_PCT,
        leverage: float = LEVERAGE,
        notifier: Optional[Any] = None,
    ):
        # публичные поля — как раньше
        self.client = client
        self.symbol = symbol
        self.risk_pct = float(risk_pct)
        self.leverage = float(leverage)
        self.notifier = notifier

        # внутренние состояния — как раньше
        self._filters: Optional[Dict[str, float]] = None
        self.equity: float = 0.0
        self.available: float = 0.0
        self.entry_extremes: bool = os.getenv("ENTRY_EXTREMES", "0") == "1"
        try:
            self.ext_eps_ticks = int(os.getenv("EXT_EPS_TICKS", "2"))
        except Exception:
            self.ext_eps_ticks = 2

        self._minute_task = None
        self._minute_mode: str = "normal"
        self._minute_sl: float | None = None
        self._realign_task = None  # background TPSL realigner task

    # -------- УТИЛЫ (тонкие обёртки на функции из модулей) --------
    def _round_step(self, value: float, step: float) -> float: return rk._round_step(self, value, step)
    def _ceil_step(self, value: float, step: float) -> float: return rk._ceil_step(self, value, step)
    def _fmt(self, x: float) -> str: return rk._fmt(self, x)
    def _round_down_qty(self, qty: float) -> float: return rk._round_down_qty(self, qty)

    def _last_price(self) -> float: return acc._last_price(self)
    def _position_side_and_size(self) -> tuple[str | None, float]: return acc._position_side_and_size(self)
    def _opposite(self, side: str) -> str: return acc._opposite(self, side)

    def _fix_tpsl(self, side: str, price: float, sl: float, tp: float, tick: float) -> tuple[float, float]:
        return ts._fix_tpsl(self, side, price, sl, tp, tick)

    def _normalize_tpsl_with_anchor(self, side: str, base_price: float, sl: float, tp: float, tick: float) -> tuple[float, float]:
        return ts._normalize_tpsl_with_anchor(self, side, base_price, sl, tp, tick)

    def _cancel_realigner(self): ts._cancel_realigner(self)

    async def _realign_tpsl(self, side: str, desired_sl: float, desired_tp: float, tick: float, debounce: float = 0.8, max_tries: int = 30):
        await ts._realign_tpsl(self, side, desired_sl, desired_tp, tick, debounce, max_tries)

    async def _await_fill_or_retry(self, order_id: str, side: str, qty: float) -> bool:
        return await ts._await_fill_or_retry(self, order_id, side, qty)

    async def _watchdog_close_on_lastprice(self, side: str, sl_price: float, tp_price: float, check_interval: float = 0.3, max_wait: float = 3600.0) -> bool:
        return await ts._watchdog_close_on_lastprice(self, side, sl_price, tp_price, check_interval, max_wait)

    async def _wait_position_open(self, timeout: float = 10.0, interval: float = 0.3) -> bool:
        return await ts._wait_position_open(self, timeout, interval)

    async def _wait_position_flat(self, timeout: float = 3600.0, interval: float = 0.5) -> bool:
        return await ts._wait_position_flat(self, timeout, interval)

    async def _apply_sl_failsafe(self, side: str, sl: float) -> bool:
        return await ts._apply_sl_failsafe(self, side, sl)

    async def _apply_tpsl_failsafe(self, side: str, base_price: float, sl: float, tp: float) -> bool:
        return await ts._apply_tpsl_failsafe(self, side, base_price, sl, tp)

    def _order_status_brief(self, order_id: str) -> str: return ex._order_status_brief(self, order_id)
    def _place_conditional(self, side: str, trigger_price: float, qty: float, trigger_direction: int) -> Dict[str, Any]:
        return ex._place_conditional(self, side, trigger_price, qty, trigger_direction)
    def _cancel_order(self, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict[str, Any]:
        return ex._cancel_order(self, order_id, order_link_id)
    def _cancel_all_orders(self) -> Dict[str, Any]: return ex._cancel_all_orders(self)

    def _calc_qty(self, side: str, price: float, sl: float) -> float: return rk._calc_qty(self, side, price, sl)

    def set_minute_status(self, mode: str, sl_price: float | None):
        ut.set_minute_status(self, mode, sl_price)

    async def _stop_minute_logger(self):
        await ut.stop_minute_logger(self)

    # -------- ПУБЛИЧНЫЕ МЕТОДЫ (как раньше) --------
    def refresh_equity(self) -> float: return acc.refresh_equity(self)
    def ensure_filters(self) -> Dict[str, float]: return acc.ensure_filters(self)
    def ensure_leverage(self): return acc.ensure_leverage(self)

    async def _enter_extremes_with_limits(self, side: str, prev_high: float, prev_low: float, qty: float, sl: float, tp: float):
        await ex.enter_extremes_with_limits(self, side, prev_high, prev_low, qty, sl, tp)

    async def _enter_by_extremes(self, side: str, prev_high: float, prev_low: float, qty: float, sl_r: float, tp_r: float):
        await ex.enter_by_extremes(self, side, prev_high, prev_low, qty, sl_r, tp_r)

    async def open_market(self, side: str, signal: Dict[str, Any]):
        await ut.open_market(self, side, signal)

    async def close_market(self, side: str, qty: float):
        await ut.close_market(self, side, qty)