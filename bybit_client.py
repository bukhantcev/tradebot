import asyncio
from typing import Optional, Dict, Any

import httpx
import json

from config import CFG, Config
from log import log, mask, trunc
from utils import hmac_sha256, now_ms, exp_backoff, clamp_qty_step, fmt_to_step

class BybitClient:
    """
    REST-клиент Bybit v5 с расширенным логированием и нормализацией параметров.
    Фиксы retCode 10001:
      - Для Market ордеров не отправляем timeInForce.
      - Передаём marketUnit=baseCoin (для линеек qty в базовой монете).
      - qty/price нормализуем по шагам инструмента.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.AsyncClient(timeout=cfg.http_timeout)
        self._limiter = asyncio.Semaphore(5)

        # Заполняются из instruments-info
        self.qty_step = 0.001
        self.min_qty = 0.001
        self.tick_size = 0.1

    # --- HTTP log helper
    def _log_http(self, direction: str, method: str, path: str, payload: Dict[str, Any] | None, extra: str = ""):
        if not self.cfg.log_http:
            return
        payload = dict(payload or {})
        for k in ("api_key", "sign"):
            if k in payload:
                payload[k] = mask(str(payload[k]))
        log(f"[HTTP {direction}] {method} {path} {extra} payload={trunc(payload)}")

    async def _req(self, method: str, path: str, params: Dict[str, Any] | None = None, need_auth: bool = False):
        url = self.cfg.rest_base + path
        params = params or {}
        headers: Dict[str, str] = {}

        attempt = 0
        while True:
            async with self._limiter:
                try:
                    if need_auth:
                        ts = str(now_ms())
                        recv = "5000"
                        headers["X-BAPI-API-KEY"] = self.cfg.bybit_key
                        headers["X-BAPI-TIMESTAMP"] = ts
                        headers["X-BAPI-RECV-WINDOW"] = recv
                        headers["X-BAPI-SIGN-TYPE"] = "2"

                        if method == "GET":
                            qs = "&".join([f"{k}={params[k]}" for k in sorted(params)])
                            sign_str = ts + self.cfg.bybit_key + recv + qs
                            headers["X-BAPI-SIGN"] = hmac_sha256(self.cfg.bybit_secret, sign_str)
                            self._log_http("->", method, path, params)
                            r = await self._client.get(url, params=params, headers=headers)
                        else:
                            body_str = json.dumps(params, separators=(",", ":"), ensure_ascii=False)
                            sign_str = ts + self.cfg.bybit_key + recv + body_str
                            headers["X-BAPI-SIGN"] = hmac_sha256(self.cfg.bybit_secret, sign_str)
                            headers["Content-Type"] = "application/json"
                            self._log_http("->", method, path, params)
                            r = await self._client.post(url, content=body_str.encode("utf-8"), headers=headers)
                    else:
                        if method == "GET":
                            self._log_http("->", method, path, params)
                            r = await self._client.get(url, params=params, headers=headers)
                        else:
                            body_str = json.dumps(params, separators=(",", ":"), ensure_ascii=False)
                            headers["Content-Type"] = "application/json"
                            self._log_http("->", method, path, params)
                            r = await self._client.post(url, content=body_str.encode("utf-8"), headers=headers)
                except Exception:
                    if attempt >= self.cfg.max_retries:
                        raise
                    await asyncio.sleep(exp_backoff(attempt))
                    attempt += 1
                    continue

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                delay = float(ra) if ra else exp_backoff(attempt)
                await asyncio.sleep(delay)
                attempt += 1
                if attempt > self.cfg.max_retries:
                    r.raise_for_status()
                continue

            if r.status_code >= 500:
                if attempt >= self.cfg.max_retries:
                    r.raise_for_status()
                await asyncio.sleep(exp_backoff(attempt))
                attempt += 1
                continue

            r.raise_for_status()
            data = r.json()
            extra = f"status={r.status_code} retCode={data.get('retCode')}"
            body_to_log = data if self.cfg.log_http_bodies else {k: data.get(k) for k in ("retCode","retMsg","time","result")}
            self._log_http("<-", method, path, body_to_log, extra=extra)

            if data.get("retCode", 0) != 0:
                if attempt < self.cfg.max_retries and data.get("retCode") in (10006, 10007, 110006, 110007):
                    await asyncio.sleep(exp_backoff(attempt))
                    attempt += 1
                    continue
            return data

    # --------- Public methods:

    async def get_instruments_info(self):
        log(f"[ACTION] instruments-info {self.cfg.symbol} {self.cfg.category}")
        data = await self._req(
            "GET", "/v5/market/instruments-info",
            {"category": self.cfg.category, "symbol": self.cfg.symbol},
            need_auth=False
        )
        # кэшируем шаги
        try:
            it = data.get("result", {}).get("list", [])
            if it:
                lot = it[0].get("lotSizeFilter", {})
                pricef = it[0].get("priceFilter", {})
                self.qty_step = float(lot.get("qtyStep", self.qty_step))
                self.min_qty = float(lot.get("minOrderQty", self.min_qty))
                self.tick_size = float(pricef.get("tickSize", self.tick_size))
                log(f"[FILTERS] qtyStep={self.qty_step} minOrderQty={self.min_qty} tickSize={self.tick_size}")
        except Exception:
            pass
        return data

    async def get_kline(self, interval: str = "1", limit: int = 200, start: Optional[int] = None, end: Optional[int] = None):
        """Получение исторических свечей (REST /v5/market/kline)
        interval: строка интервала по Bybit ("1","3","5","15","30","60","120","240","360","720","D","W","M")
        limit: кол-во свечей (по умолчанию 200)
        start/end: таймштампы в миллисекундах (опционально)
        """
        log(f"[ACTION] kline interval={interval} limit={limit} start={start} end={end}")
        params: Dict[str, Any] = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "interval": str(interval),
            "limit": str(limit),
        }
        if start is not None:
            params["start"] = str(int(start))
        if end is not None:
            params["end"] = str(int(end))
        return await self._req(
            "GET", "/v5/market/kline",
            params,
            need_auth=False,
        )

    async def get_wallet_balance(self):
        log("[ACTION] wallet-balance")
        return await self._req(
            "GET", "/v5/account/wallet-balance",
            {"accountType": "UNIFIED"},
            need_auth=True
        )

    async def switch_isolated(self, trade_mode: int, buy_leverage: int, sell_leverage: int):
        """Переключение маржин-режима (0=cross, 1=isolated) c одновременной установкой плеча."""
        log(f"[ACTION] switch-isolated mode={trade_mode} {buy_leverage}/{sell_leverage}")
        return await self._req(
            "POST", "/v5/position/switch-isolated",
            {
                "category": self.cfg.category,
                "symbol": self.cfg.symbol,
                "tradeMode": str(trade_mode),
                "buyLeverage": str(buy_leverage),
                "sellLeverage": str(sell_leverage),
            },
            need_auth=True,
        )

    async def set_leverage(self, buy_leverage: int, sell_leverage: int):
        log(f"[ACTION] set-leverage {buy_leverage}/{sell_leverage}")
        # Для UTA2.0 `switch-isolated` запрещён (100028). Используем только set-leverage с header-auth + JSON.
        return await self._req(
            "POST", "/v5/position/set-leverage",
            {
                "category": self.cfg.category,
                "symbol": self.cfg.symbol,
                "buyLeverage": str(buy_leverage),
                "sellLeverage": str(sell_leverage),
            },
            need_auth=True,
        )

    async def get_position_list(self):
        log("[ACTION] position-list")
        return await self._req(
            "GET", "/v5/position/list",
            {"category": self.cfg.category, "symbol": self.cfg.symbol},
            need_auth=True
        )

    async def create_order(self, side: str, qty: float, order_type: str = "Market",
                           reduce_only: bool = False, price: Optional[float] = None):
        log(f"[ACTION] create-order {side} qty={qty} type={order_type} price={price}")
        q = clamp_qty_step(qty, self.qty_step, self.min_qty)
        if q <= 0:
            return {"retCode": -1, "retMsg": "qty_below_min"}
        qty_str = ("%.12f" % q).rstrip('0').rstrip('.')

        params = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty_str,
            "marketUnit": "baseCoin",                  # ключевой параметр
            "reduceOnly": "true" if reduce_only else "false",
        }
        if order_type == "Limit" and price is not None:
            price = fmt_to_step(price, self.tick_size)
            params["price"] = ("%.12f" % price).rstrip('0').rstrip('.')
            params["timeInForce"] = "GTC"             # для лимитных

        return await self._req("POST", "/v5/order/create", params, need_auth=True)

    async def cancel_all_orders(self):
        log("[ACTION] cancel-all-orders")
        params = {"category": self.cfg.category, "symbol": self.cfg.symbol}
        return await self._req("POST", "/v5/order/cancel-all", params, need_auth=True)

    async def set_trading_stop(
        self,
        sl: Optional[float] = None,
        take_profit: Optional[str] = None,
        takeProfit: Optional[str] = None,
        trailing_activation: Optional[str] = None,
        trailing_distance: Optional[str] = None,
        position_idx: Optional[int] = None,
        side: Optional[str] = None,
        use_trailing: Optional[bool] = None,
    ):
        """Установка биржевых SL/TP и (опционально) трейлинга через /v5/position/trading-stop.
        Требования:
        - Всегда кладём tpslMode='Full'.
        - Добавляем positionIdx и side, если заданы.
        - Если use_trailing == False — не отправляем activePrice/trailingStop.
        - Если use_trailing == True — отправляем activePrice и trailingStop, значения приходят уже округлёнными строками извне (из trader.py), без пересчёта здесь.
        - take_profit, если передан, уходит как takeProfit.
        """
        # --- Aliases ---
        # Allow calling set_trading_stop(takeProfit="...") in addition to take_profit
        if take_profit is None and takeProfit is not None:
            take_profit = takeProfit

        # use_trailing по умолчанию берём из конфига, если он не задан явно
        if use_trailing is None:
            use_trailing = bool(getattr(self.cfg, "use_trailing", False))

        log(f"[ACTION] trading-stop SL={sl} TP={take_profit} act={trailing_activation} dist={trailing_distance} posIdx={position_idx} side={side} use_trailing={use_trailing}")

        params: Dict[str, Any] = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "tpslMode": "Full",
        }

        # Привязка к позиции
        if position_idx is not None:
            params["positionIdx"] = str(int(position_idx))
        if side:
            params["side"] = side  # 'Buy'/'Sell'

        # Stop Loss
        if sl is not None:
            try:
                v = float(sl)
                t = float(getattr(self, "tick_size", 0) or 0)
                if t > 0:
                    v = fmt_to_step(v, t)
                params["stopLoss"] = ("%.12f" % float(v)).rstrip('0').rstrip('.')
            except Exception:
                params["stopLoss"] = str(sl)

        # Take Profit
        if take_profit is not None and take_profit != "":
            params["takeProfit"] = str(take_profit)

        # Trailing — только если явно включён
        if use_trailing:
            if trailing_activation is not None and trailing_activation != "":
                params["activePrice"] = str(trailing_activation)
            if trailing_distance is not None and trailing_distance != "":
                params["trailingStop"] = str(trailing_distance)
        # Если use_trailing == False — поля активатора и дистанции не добавляем вовсе

        # убрать None/пустые строки из payload на всякий случай
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        return await self._req("POST", "/v5/position/trading-stop", params, need_auth=True)

    async def close_position_market(self):
        log("[ACTION] close-position-market")
        plist = await self.get_position_list()
        li = plist.get("result", {}).get("list", [])
        if not li:
            return {"ok": True, "msg": "no position"}
        import asyncio
        tasks = []
        for p in li:
            sz = float(p.get("size", "0"))
            if sz == 0:
                continue
            side = "Sell" if p.get("side") == "Buy" else "Buy"
            tasks.append(self.create_order(side, qty=sz, order_type="Market", reduce_only=True))
        if tasks:
            res = await asyncio.gather(*tasks, return_exceptions=True)
            return {"ok": True, "results": res}
        return {"ok": True, "msg": "no position"}