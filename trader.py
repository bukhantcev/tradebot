# trader.py
import math
import logging
from typing import Optional, Dict, Any

from config import SYMBOL, RISK_PCT, LEVERAGE
from bybit_client import BybitClient

log = logging.getLogger("TRADER")


class Trader:
    """
    Минималистичный трейдер:
    - refresh_equity(): получить equity (USDT) одной строкой
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
        self.client = client
        self.symbol = symbol
        self.risk_pct = float(risk_pct)
        self.leverage = float(leverage)
        self.notifier = notifier

        # кэш маркет-фильтров
        self._filters: Optional[Dict[str, float]] = None
        # последняя известная equity (USDT)
        self.equity: float = 0.0

    # ---------- УТИЛЫ ----------

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step + 1e-12) * step

    def _ceil_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.ceil(value / step - 1e-12) * step

    def _fmt(self, x: float) -> str:
        return f"{x:.6f}".rstrip("0").rstrip(".")

    # ---------- ПУБЛИЧНЫЕ МЕТОДЫ ----------

    def refresh_equity(self) -> float:
        """Запрос баланса без лишнего шума"""
        try:
            r = self.client.wallet_balance(account_type="UNIFIED")
            usdt = float(r["result"]["list"][0]["totalEquity"])
            self.equity = usdt
            log.info(f"[BALANCE] {usdt:.2f} USDT")
            if self.notifier:
                try:
                    import asyncio
                    asyncio.create_task(self.notifier.notify(f"💰 Баланс: {usdt:.2f} USDT"))
                except Exception:
                    pass
            return usdt
        except Exception as e:
            log.error(f"[BALANCE][ERR] {e}")
            return 0.0

    def ensure_filters(self) -> Dict[str, float]:
        """Получить tickSize / qtyStep / minOrderQty (кэшируется)"""
        if self._filters:
            return self._filters
        r = self.client.instruments_info(category="linear", symbol=self.symbol)
        it = r.get("result", {}).get("list", [{}])[0]
        tick = float(it.get("priceFilter", {}).get("tickSize", "0.1"))
        qty_step = float(it.get("lotSizeFilter", {}).get("qtyStep", "0.001"))
        min_qty = float(it.get("lotSizeFilter", {}).get("minOrderQty", "0.001"))
        self._filters = {"tickSize": tick, "qtyStep": qty_step, "minQty": min_qty}
        log.debug(f"[FILTERS] {self._filters}")
        return self._filters

    def ensure_leverage(self):
        """Ставит плечо только если отличается (подавляем 110043 как норму)"""
        try:
            pl = self.client.position_list(self.symbol)
            # Bybit может отдавать пустой список, если позиции нет
            cur_lev = float(pl.get("result", {}).get("list", [{}])[0].get("leverage") or 0.0)
            if abs(cur_lev - self.leverage) < 1e-9:
                log.debug(f"[LEV] already {cur_lev}x")
                return
        except Exception as e:
            log.debug(f"[LEV] read failed: {e}")
        r = self.client.set_leverage(self.symbol, self.leverage, self.leverage)
        rc = r.get("retCode")
        if rc in (0, 110043):  # 110043 = leverage not modified
            log.info(f"[LEV] {self.leverage}x OK")
        else:
            log.warning(f"[LEV] rc={rc} msg={r.get('retMsg')}")

    # ---------- РАСЧЁТ QTY ----------

    def _calc_qty(self, side: str, price: float, sl: float) -> float:
        """
        К-во рассчитываем по риску:
        risk_amt = equity * risk_pct
        stop_dist = |price - sl|
        qty = risk_amt / stop_dist
        затем приводим к шагу и к минимуму биржи.
        """
        f = self.ensure_filters()
        risk_amt = max(self.equity * self.risk_pct / max(self.leverage, 1.0), 1e-8)
        stop_dist = abs(price - sl)
        if stop_dist <= 0:
            return 0.0
        raw_qty = risk_amt / stop_dist
        qty = max(f["minQty"], self._round_step(raw_qty, f["qtyStep"]))
        return qty

    # ---------- ОРДЕРА ----------

    async def open_market(self, side: str, signal: Dict[str, Any]):
        """
        side: "Buy" | "Sell"
        signal: { 'sl': float, 'tp': float, 'atr': float, 'ts_ms': int, ... }
        """
        # обновим equity (если 0) и фильтры/плечо
        if self.equity <= 0:
            self.refresh_equity()
        self.ensure_filters()
        self.ensure_leverage()

        price = float(signal.get("price") or signal.get("close") or 0.0)
        if price <= 0:
            # берём близкую оценку — без отдельного REST-запроса
            price = float(signal.get("tp") or 0.0) or 1.0

        sl = float(signal.get("sl") or 0.0)
        tp = float(signal.get("tp") or 0.0)
        if sl <= 0 or tp <= 0:
            log.info("[ENTER][SKIP] bad SL/TP")
            return

        qty = self._calc_qty(side, price, sl)
        if qty <= 0:
            log.info("[ENTER][SKIP] qty=0")
            return

        # Маркет-вход
        log.info(f"[ENTER] {side} qty={self._fmt(qty)}")
        r = self.client.place_order(self.symbol, side, qty)
        if r.get("retCode") != 0:
            log.error(f"[ORDER][FAIL] {r}")
            return

        order_id = r.get("result", {}).get("orderId", "")
        log.info(f"[ORDER←] OK id={order_id}")
        if self.notifier:
            try:
                await self.notifier.notify(f"✅ {side} {self.symbol} qty={self._fmt(qty)} (id {order_id})")
            except Exception:
                pass

        # Установка TP/SL
        f = self.ensure_filters()
        tick = f["tickSize"]
        sl_r = self._ceil_step(sl, tick) if side == "Buy" else self._round_step(sl, tick)
        tp_r = self._round_step(tp, tick) if side == "Buy" else self._ceil_step(tp, tick)

        r2 = self.client.trading_stop(self.symbol, side=side, stop_loss=sl_r, take_profit=tp_r)
        if r2.get("retCode") in (0, None):
            log.info(f"[TPSL] sl={self._fmt(sl_r)} tp={self._fmt(tp_r)} OK")
            if self.notifier:
                try:
                    await self.notifier.notify(f"🎯 TP/SL set: SL {self._fmt(sl_r)} / TP {self._fmt(tp_r)}")
                except Exception:
                    pass
        else:
            log.warning(f"[TPSL][FAIL] {r2}")

    async def close_market(self, side: str, qty: float):
        """
        Закрытие позиции маркетом (на твой выбор side / qty).
        """
        if qty <= 0:
            log.info("[EXIT][SKIP] qty=0")
            return
        log.info(f"[EXIT] {side} qty={self._fmt(qty)}")
        r = self.client.place_order(self.symbol, side, qty)
        if r.get("retCode") == 0:
            oid = r.get("result", {}).get("orderId", "")
            log.info(f"[ORDER←] CLOSE OK id={oid}")
            if self.notifier:
                try:
                    await self.notifier.notify(f"❌ Close {side} {self.symbol} qty={self._fmt(qty)} (id {oid})")
                except Exception:
                    pass
        else:
            log.error(f"[ORDER][FAIL] {r}")