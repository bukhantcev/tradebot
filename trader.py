# app/trader.py
import asyncio
import logging
from typing import Optional, Tuple, Callable, Any, Coroutine, Awaitable

# Ожидаемые подмодули (реализация лежит в trader_mod/*)
from trader_mod.account import Account
from trader_mod.risk import Risk
from trader_mod.tpsl import TPSL
from trader_mod.extremes import Extremes
from trader_mod.utils import fmt

log = logging.getLogger("TRADER")


class Trader:
    """
    Тонкий фасад над клиентом биржи.
    Делегирует:
      - учёт/плечо/фильтры -> Account
      - расчёт количества -> Risk
      - постановка/реалайн TP/SL -> TPSL
      - экстрем-режим (лимитки по экстремумам) -> Extremes
    """

    def __init__(
        self,
        client: Any,
        symbol: str,
        leverage: int,
        risk_pct: float,
        notifier: Optional[Callable[[str], asyncio.Future | None]] = None,
        eps_ticks: int = 1,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.notifier = notifier
        self._closed = False

        # Сервисы
        self.acct = Account(client=client, symbol=symbol, leverage=leverage, notifier=notifier)
        self.risk = Risk(account=self.acct, risk_pct=risk_pct)
        self.tpsl = TPSL(
            account=self.acct,
            get_last_price=self.get_last_price,
            get_pos_side_size=self.get_pos_side_size,
            close_market=self.close_market,
            cancel_realigner=self._stop_realign_task,
            notifier=notifier,
        )
        self.ext = Extremes(
            account=self.acct,
            tpsl=self.tpsl,
            client=client,
            symbol=symbol,
            risk=self.risk,
            eps_ticks=eps_ticks,
            get_last_price=self.get_last_price,
            get_pos_side_size=self.get_pos_side_size,
            cancel_order=self.cancel_order,
            cancel_all=self.cancel_all_orders,
            set_realign_task=self._set_realign_task,
            stop_realign_task=self._stop_realign_task,
            notifier=notifier,
        )

        # Параметры инструмента
        flt = self.acct.ensure_filters()
        self.tick = float(flt.get("tickSize", 0.1))
        self.qty_step = float(flt.get("qtyStep", 0.001))

        # Реалайнер TP/SL (таск)
        self._realign_task: Optional[asyncio.Task] = None

    # -----------------------------
    # Общие хелперы
    # -----------------------------
    async def refresh_equity(self) -> float:
        equity = await self.acct.refresh_equity()
        log.info("[BALANCE] equity=%s avail=%s USDT", fmt(equity), fmt(self.acct.available))
        return equity

    async def ensure_leverage(self) -> None:
        try:
            await self.acct.ensure_leverage()
        except Exception as e:
            log.warning("[LEV] %s", str(e))

    async def get_last_price(self) -> float:
        """
        Возвращает last price. Ожидается, что client.ticker(symbol)
        вернёт dict с 'last' или 'lastPrice' (строка/число).
        """
        t = await self.client.ticker(self.symbol)
        if not t:
            raise RuntimeError("ticker() returned empty")
        last = t.get("last") or t.get("lastPrice") or t.get("last_price")
        if last is None:
            raise RuntimeError(f"ticker() has no last price fields: {t!r}")
        return float(last)

    async def get_pos_side_size(self) -> Tuple[str, float]:
        """
        Возвращает (side, size) текущей позиции:
          side: "Buy" | "Sell" | "Flat"
          size: положительное число, 0 если позиции нет
        """
        p = await self.client.position_list(self.symbol)
        if not p:
            return "Flat", 0.0

        # Ожидаем формат наподобие:
        # {'list': [{'side': 'Buy'|'Sell', 'size': '0.003', ...}], ...}
        lst = p.get("list") or []
        if not lst:
            return "Flat", 0.0

        # Берём агрегированную позицию (positionIdx=0)
        cur = lst[0]
        side = str(cur.get("side", "")).title()
        size_raw = cur.get("size") or cur.get("positionSize") or 0
        try:
            size = float(size_raw)
        except Exception:
            size = 0.0

        if size <= 0:
            return "Flat", 0.0
        return side if side in ("Buy", "Sell") else "Flat", size

    # -----------------------------
    # Управление ордерами/позициями
    # -----------------------------
    async def cancel_order(self, order_id: str) -> None:
        try:
            await self.client.cancel_order(self.symbol, order_id=order_id)
        except Exception as e:
            log.warning("[ORDER][CANCEL][ERR] %s", str(e))

    async def cancel_all_orders(self) -> None:
        try:
            await self.client.cancel_all_orders(self.symbol)
        except Exception as e:
            log.warning("[ORDER][CANCEL_ALL][ERR] %s", str(e))

    async def close_market(self, side: str, qty: float) -> bool:
        """
        Закрыть qty по рынку (IOC). side — это направление СДЕЛКИ (не позиции):
          - чтобы закрыть Buy-позицию: side="Sell"
          - чтобы закрыть Sell-позицию: side="Buy"
        """
        if qty <= 0:
            return True

        body = {
            "orderType": "Market",
            "timeInForce": "IOC",
            "positionIdx": 0,
            # reduceOnly — пусть клиент подставит сам (если поддерживает),
            # либо закрытие произойдёт встречной сделкой.
        }
        try:
            r = await self.client.place_order(
                self.symbol, side, str(qty),
                order_type=body["orderType"],
                timeInForce=body["timeInForce"],
                positionIdx=body["positionIdx"],
            )
            if r and r.get("retCode") == 0:
                oid = r["result"]["orderId"]
                log.info("[ORDER←] CLOSE OK id=%s", oid)
                return True
            log.error("[ORDER][FAIL] %s", r)
        except Exception as e:
            log.error("[ORDER][CLOSE][EXC] %s", str(e))
        return False

    # -----------------------------
    # Вход
    # -----------------------------
    async def open_market(
        self,
        side: str,
        qty: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        prev_high: Optional[float] = None,
        prev_low: Optional[float] = None,
        use_ext: Optional[bool] = None,
    ) -> None:
        """
        Универсальный вход:
          - Если use_ext=True и заданы prev_high/prev_low -> экстрем-режим (лимитные сценарии целиком в Extremes).
          - Иначе — маркет-вход (IOC), затем постановка TP/SL через trading_stop и запуск мягкого реалайнера.
        """
        if self._closed:
            log.warning("[ENTER][SKIP] trader closed")
            return

        # Экстрем-режим (лимитки у экстремумов)
        if use_ext and prev_high is not None and prev_low is not None:
            log.info("[EXT][MODE] ON  prevH=%s prevL=%s qty=%s", fmt(prev_high), fmt(prev_low), fmt(qty or 0))
            await self.ext.run_limits(
                side=side,
                prev_high=float(prev_high),
                prev_low=float(prev_low),
                qty=qty,
                sl=sl,
                tp=tp,
                tick=self.tick,
                qty_step=self.qty_step,
            )
            return

        # Обычный маркет-вход
        # 1) цена/к-во
        last = await self.get_last_price()
        if qty is None:
            if sl is None:
                raise ValueError("open_market: 'sl' required to compute risk-based qty")
            qty = self.risk.calc_qty(side=side, price=last, sl=float(sl))
        if qty < self.qty_step:
            log.info("[ENTER][SKIP] qty=%s < min step=%s", qty, self.qty_step)
            return

        # 2) маркет-ордер
        log.info("[ENTER] %s qty=%s", side, fmt(qty))
        try:
            r = await self.client.place_order(
                self.symbol,
                side,
                str(qty),
                order_type="Market",
                timeInForce="IOC",
                positionIdx=0,
            )
        except Exception as e:
            log.error("[ORDER][EXC] %s", str(e))
            return

        if not r or r.get("retCode") != 0:
            log.error("[ORDER][FAIL] %s", r)
            return

        oid = r["result"]["orderId"]
        log.info("[ORDER←] OK id=%s", oid)

        # 3) TP/SL через trading_stop (если есть хотя бы SL)
        if sl is None and tp is None:
            log.info("[TPSL][SKIP] no sl/tp specified")
            return

        # Нормализуем целевые уровни с привязкой:
        #  - SL: MarkPrice
        #  - TP: LastPrice (маркет), чтобы исполнялось по последней.
        want_sl, want_tp = self.tpsl.normalize_with_anchor(
            side=side, base=last, sl=sl, tp=tp, tick=self.tick
        )

        ok = await self._try_set_tpsl(side, want_sl, want_tp)
        if ok:
            # 4) Запустим мягкий реалайнер (в фоне подстраивает TP/SL при изменении условий)
            self._set_realign_task(self.tpsl.realign_tpsl(side, want_sl, want_tp, self.tick, self.client, self.symbol))

    async def _try_set_tpsl(self, side: str, sl: Optional[float], tp: Optional[float]) -> bool:
        if sl is None and tp is None:
            return True

        body = dict(
            category="linear",
            symbol=self.symbol,
            positionIdx=0,
            tpslMode="Full",
            slTriggerBy="MarkPrice",
            tpTriggerBy="LastPrice",
            tpOrderType="Market",
        )
        try:
            r = await self.client.trading_stop(
                symbol=self.symbol,
                side=side,
                stop_loss=sl,
                take_profit=tp,
                tpslMode=body["tpslMode"],
                tpTriggerBy=body["tpTriggerBy"],
                slTriggerBy=body["slTriggerBy"],
                tpOrderType=body["tpOrderType"],
                positionIdx=body["positionIdx"],
            )
            if r and r.get("retCode") == 0:
                log.info("[TPSL] sl=%s tp=%s OK", fmt(sl) if sl else "-", fmt(tp) if tp else "-")
                return True
            rc = (r or {}).get("retCode")
            msg = (r or {}).get("retMsg")
            log.warning("[TPSL][ERR] rc=%s msg=%s body=%s", rc, msg, body)
            return False
        except Exception as e:
            log.warning("[TPSL][EXC] %s", str(e))
            return False

    # -----------------------------
    # Реалайнер TP/SL
    # -----------------------------
    def _set_realign_task(self, coro: Optional[Coroutine[Any, Any, Any] | Awaitable[Any] | asyncio.Task]) -> None:
        """Register (or replace) background realigner task.
        Accepts a coroutine/awaitable or an already created asyncio.Task.
        """
        self._stop_realign_task()
        if coro is None:
            return
        if isinstance(coro, asyncio.Task):
            self._realign_task = coro
        else:
            self._realign_task = asyncio.create_task(coro, name="tpsl_realign")

    def _stop_realign_task(self) -> None:
        if self._realign_task and not self._realign_task.done():
            self._realign_task.cancel()
        self._realign_task = None

    # -----------------------------
    # Жизненный цикл
    # -----------------------------
    async def shutdown(self) -> None:
        self._closed = True
        self._stop_realign_task()
        # В тонком фасаде — без авто-флэттинга.
        # В нужном месте выше по стеку можно вызвать close_all/flat, если это требование UI.
        log.info("[TRADER] shutdown complete")