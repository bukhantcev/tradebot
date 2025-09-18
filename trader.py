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
        self.available: float = 0.0  # доступный баланс для маржи (USDT)

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

    def _fix_tpsl(self, side: str, price: float, sl: float, tp: float, tick: float) -> (float, float):
        """
        Нормализует SL/TP относительно рыночной цены и тика, чтобы не ловить 30208 на order.create.
          • Buy:  SL < price, TP > price  (минимум на 1 тик от цены)
          • Sell: SL > price, TP < price
        Возвращает кортеж (sl_fixed, tp_fixed) с корректировкой на 1–2 тика при необходимости.
        """
        p = float(price)
        sl_f, tp_f = float(sl), float(tp)
        t = max(float(tick), 0.0) or 0.1

        if side == "Buy":
            # SL должен быть строго ниже цены, TP — строго выше
            if sl_f >= p:
                sl_f = p - t
            if tp_f <= p:
                tp_f = p + t
            # округление в сторону, совместимую с правилами биржи
            sl_f = self._round_step(sl_f, t)  # вниз по тику
            tp_f = self._ceil_step(tp_f, t)   # вверх по тику
            # дополнительная страховка: разнести хотя бы на 1 тик
            if sl_f >= p:
                sl_f = p - 2 * t
            if tp_f <= p:
                tp_f = p + 2 * t
        else:  # Sell
            if sl_f <= p:
                sl_f = p + t
            if tp_f >= p:
                tp_f = p - t
            sl_f = self._ceil_step(sl_f, t)   # вверх по тику
            tp_f = self._round_step(tp_f, t)  # вниз по тику
            if sl_f <= p:
                sl_f = p + 2 * t
            if tp_f >= p:
                tp_f = p - 2 * t

        return sl_f, tp_f

    # ---------- ПУБЛИЧНЫЕ МЕТОДЫ ----------

    def refresh_equity(self) -> float:
        """Запрос баланса: totalEquity и totalAvailableBalance"""
        try:
            r = self.client.wallet_balance(account_type="UNIFIED")
            lst = r["result"]["list"][0]
            usdt_total = float(lst["totalEquity"])
            usdt_avail = float(lst.get("totalAvailableBalance") or lst.get("availableBalance") or usdt_total)
            self.equity = usdt_total
            self.available = usdt_avail
            log.info(f"[BALANCE] equity={usdt_total:.2f} avail={usdt_avail:.2f} USDT")
            if self.notifier:
                try:
                    import asyncio
                    asyncio.create_task(self.notifier.notify(f"💰 Баланс: {usdt_total:.2f} USDT (доступно {usdt_avail:.2f})"))
                except Exception:
                    pass
            return usdt_total
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
        mov = float(it.get("lotSizeFilter", {}).get("minOrderValue", "0") or 0.0)
        self._filters = {"tickSize": tick, "qtyStep": qty_step, "minQty": min_qty, "minNotional": mov}
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
        Считаем qty как минимум из:
          • риск-бейзд:  qty_risk = (equity * risk_pct) / stop_dist
          • по доступной марже: qty_afford = available / (price/leverage * fee_buf)
        Затем приводим к шагу qtyStep и проверяем минимальные ограничения.
        """
        f = self.ensure_filters()
        stop_dist = abs(price - sl)
        if stop_dist <= 0:
            return 0.0

        # 1) по риску (леверидж не влияет на риск в $)
        risk_amt = max(self.equity * self.risk_pct, 0.0)
        qty_risk = risk_amt / stop_dist

        # 2) по доступной марже (учтём буфер на комиссии/плавающие требования)
        FEE_BUF = 1.003
        margin_per_qty = price / max(self.leverage, 1.0)
        if margin_per_qty <= 0:
            return 0.0
        qty_afford = (self.available / (margin_per_qty * FEE_BUF)) if self.available > 0 else qty_risk

        # 3) итог
        raw = max(0.0, min(qty_risk, qty_afford))
        qty = self._round_step(raw, f["qtyStep"])
        # Минимальные ограничения
        if qty < f["minQty"]:
            return 0.0
        # Минимальная нотация (если биржа требует)
        min_notional = f.get("minNotional", 0.0) or 0.0
        if min_notional > 0 and (qty * price) < min_notional:
            return 0.0
        return qty

    async def _wait_position_open(self, timeout: float = 2.0, interval: float = 0.2) -> bool:
        """
        Короткое ожидание появления позиции (size&gt;0), чтобы fallback trading-stop не падал rc=10001.
        Возвращает True, если позиция открылась в течение timeout.
        """
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                pl = self.client.position_list(self.symbol)
                lst = pl.get("result", {}).get("list", [])
                if lst:
                    size_str = lst[0].get("size") or lst[0].get("positionValue") or ""
                    try:
                        size = float(size_str) if size_str != "" else 0.0
                    except Exception:
                        size = 0.0
                    if size > 0:
                        return True
            except Exception:
                pass
            await asyncio.sleep(interval)
        return False

    # ---------- ОРДЕРА ----------

    async def open_market(self, side: str, signal: Dict[str, Any]):
        """
        side: "Buy" | "Sell"
        signal: { 'sl': float, 'tp': float, 'atr': float, 'ts_ms': int, ... }
        """
        # обновим equity/available (если 0) и фильтры/плечо
        if self.equity <= 0 or self.available <= 0:
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
        log.info(f"[QTY] risk%={self.risk_pct*100:.2f} stop={abs(price-sl):.2f} equity={self.equity:.2f} avail={self.available:.2f} -> qty={self._fmt(qty)}")
        if qty <= 0:
            log.info("[ENTER][SKIP] qty=0")
            return

        # Подготовим округлённые SL/TP заранее (для передачи в order/create)
        f = self.ensure_filters()
        tick = f["tickSize"]
        sl_r = self._ceil_step(sl, tick) if side == "Buy" else self._round_step(sl, tick)
        tp_r = self._round_step(tp, tick) if side == "Buy" else self._ceil_step(tp, tick)
        sl_r, tp_r = self._fix_tpsl(side, price, sl_r, tp_r, tick)

        # Маркет-вход: сначала ордер БЕЗ TP/SL, затем TP/SL через trading-stop (устраняем 30208)
        log.info(f"[ENTER] {side} qty={self._fmt(qty)}")
        r = self.client.place_order(self.symbol, side, qty)
        rc = r.get("retCode")
        if rc == 0:
            order_id = r.get("result", {}).get("orderId", "")
            log.info(f"[ORDER←] OK id={order_id}")
            if self.notifier:
                try:
                    await self.notifier.notify(f"✅ {side} {self.symbol} qty={self._fmt(qty)} (id {order_id})")
                except Exception:
                    pass
        elif rc == 110007:  # ab not enough for new order — уменьшим объём и повторим один раз
            f = self.ensure_filters()
            qty2 = max(f["minQty"], self._round_step(qty * 0.9, f["qtyStep"]))
            if qty2 < f["minQty"]:
                log.error(f"[ORDER][FAIL] rc=110007 (no balance), qty too small after retry")
                return
            log.info(f"[ENTER][RETRY] reduce qty -> {self._fmt(qty2)}")
            r = self.client.place_order(self.symbol, side, qty2)
            if r.get("retCode") == 0:
                order_id = r.get("result", {}).get("orderId", "")
                log.info(f"[ORDER←] OK id={order_id}")
                if self.notifier:
                    try:
                        await self.notifier.notify(f"✅ {side} {self.symbol} qty={self._fmt(qty2)} (id {order_id})")
                    except Exception:
                        pass
                qty = qty2  # используем фактическое qty далее
            else:
                log.error(f"[ORDER][FAIL] {r}")
                return
        else:
            log.error(f"[ORDER][FAIL] {r}")
            return

        # После успешного входа — подождём открытия позиции и поставим TP/SL отдельным вызовом
        ok = await self._wait_position_open()
        if not ok:
            log.warning("[TPSL][SKIP] position not opened")
            return

        r2 = self.client.trading_stop(
            self.symbol,
            side=side,
            stop_loss=sl_r,
            take_profit=tp_r,
            tpslMode="Full",
            positionIdx=0,
        )
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