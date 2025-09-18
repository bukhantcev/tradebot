import time
import hmac
import hashlib
import json
import logging
from typing import Any, Dict, Optional
import httpx
import websockets

from urllib.parse import urlencode

from config import get_bybit_keys, BYBIT_ENV

log = logging.getLogger("BYBIT")

from decimal import Decimal, ROUND_DOWN

class BybitClient:
    _filters_cache: dict[str, dict] = {}

    def _get_filters(self, symbol: str) -> dict:
        if symbol in self._filters_cache:
            return self._filters_cache[symbol]
        ins = self.instruments_info("linear", symbol)
        tick = 0.1
        qty_step = 0.001
        min_qty = 0.001
        try:
            lst = ins.get("result", {}).get("list", [])
            if lst:
                pf = lst[0].get("priceFilter", {})
                lf = lst[0].get("lotSizeFilter", {})
                tick = float(pf.get("tickSize", tick))
                qty_step = float(lf.get("qtyStep", qty_step))
                # minOrderQty может называться minOrderQty или minTradingQty в некоторых маркетах
                min_qty = float(lf.get("minOrderQty", lf.get("minTradingQty", min_qty)))
        except Exception:
            pass
        self._filters_cache[symbol] = {"tickSize": tick, "qtyStep": qty_step, "minQty": min_qty}
        return self._filters_cache[symbol]

    def _fmt_step(self, value: float, step: float) -> str:
        if step <= 0:
            return str(value)
        d_val = Decimal(str(value))
        d_step = Decimal(str(step))
        # quantize to step using ROUND_DOWN
        q = (d_val / d_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * d_step
        # strip trailing zeros
        s = format(q, 'f')
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    def _fmt_price(self, symbol: str, price: float) -> str:
        f = self._get_filters(symbol)
        return self._fmt_step(price, f["tickSize"])

    def _fmt_qty(self, symbol: str, qty: float) -> Optional[str]:
        f = self._get_filters(symbol)
        qs = max(qty, f["minQty"])  # not less than min
        s = self._fmt_step(qs, f["qtyStep"])
        # convert back to float to check strictly >= minQty after rounding
        try:
            if float(s) < f["minQty"]:
                return None
        except Exception:
            return None
        return s
    def __init__(self, recv_window: int = 5000, verify_ssl: bool = True):
        self.api_key, self.api_secret = get_bybit_keys()
        self.recv_window = recv_window
        self.verify_ssl = verify_ssl

        if BYBIT_ENV == "real":
            self.rest_url = "https://api.bybit.com"
            self.ws_public_url = "wss://stream.bybit.com/v5/public/linear"
            self.ws_private_url = "wss://stream.bybit.com/v5/private"
        else:
            self.rest_url = "https://api-testnet.bybit.com"
            self.ws_public_url = "wss://stream-testnet.bybit.com/v5/public/linear"
            self.ws_private_url = "wss://stream-testnet.bybit.com/v5/private"

        self.session = httpx.Client(verify=self.verify_ssl, timeout=15.0)
        log.debug(f"[INIT] recv_window={recv_window} verify_ssl={verify_ssl} env={BYBIT_ENV}")

    def _sign(self, payload: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None):
        url = self.rest_url + path
        ts = str(int(time.time() * 1000))
        body_str = json.dumps(body) if body else ""
        query_str = ""
        params_list = None
        if params:
            # preserve insertion order for both the URL and the signature
            params_list = list(params.items())
            query_str = urlencode(params_list)
        to_sign = ts + self.api_key + str(self.recv_window) + query_str + body_str
        sign = self._sign(to_sign)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json",
        }

        log.debug(f"[HTTP→] {method} {url} params={params} body={body}")
        # For GET, do not send a JSON body; for POST send body_str (may be empty string)
        content_payload = None if method.upper() == "GET" else body_str
        resp = self.session.request(method, url, params=params_list or None, content=content_payload, headers=headers)
        try:
            data = resp.json()
        except Exception:
            data = {"status_code": resp.status_code, "text": resp.text[:500]}

        rc = data.get("retCode")
        log.debug(f"[HTTP←] {method} {path} retCode={rc} status={resp.status_code} resp={str(data)[:500]}")
        if rc not in (0, None):
            log.error(f"[HTTP][ERROR] path={path} rc={rc} msg={data.get('retMsg')}")

        return data

    # --- Public REST endpoints ---
    def instruments_info(self, category="linear", symbol="BTCUSDT"):
        return self._request("GET", "/v5/market/instruments-info", params={"category": category, "symbol": symbol})

    def orderbook_top(self, symbol: str, category: str = "linear"):
        # depth=1 returns best bid/ask
        params = {"category": category, "symbol": symbol, "limit": 1}
        return self._request("GET", "/v5/market/orderbook", params=params)

    def server_time(self):
        return self._request("GET", "/v5/market/time")

    # --- Market Kline (REST) ---
    def kline(self, category="linear", symbol="BTCUSDT", interval="1",
              start: Optional[int]=None, end: Optional[int]=None,
              limit: int = 200, cursor: Optional[str]=None):
        params = {
            "category": category,
            "symbol": symbol,
            "interval": str(interval),
            "limit": str(limit),
        }
        if start is not None: params["start"] = str(start)
        if end is not None: params["end"] = str(end)
        if cursor is not None: params["cursor"] = cursor
        return self._request("GET", "/v5/market/kline", params=params)

    # --- Private REST endpoints ---
    def wallet_balance(self, account_type="UNIFIED"):
        return self._request("GET", "/v5/account/wallet-balance", params={"accountType": account_type})

    def set_leverage(self, symbol: str, buy_leverage: int, sell_leverage: int):
        log.debug(f"[LEVERAGE] set {symbol} -> {buy_leverage}/{sell_leverage}")
        return self._request("POST", "/v5/position/set-leverage", body={
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        })

    def place_market_safe(
        self,
        symbol: str,
        side: str,
        qty: float,
        position_idx: int = 0,
        slip_percent: float = 0.05,  # 0.05%
    ):
        """
        Отправляет рыночный ордер как Limit+IOC по цене, рассчитанной от best bid/ask с небольшим допуском,
        чтобы не ловить 30208 (прайс-лимит) на деривативах.
        """
        # 1) Получаем тик и топ стакана
        filters = self._get_filters(symbol)
        tick = filters["tickSize"]

        ob = self.orderbook_top(symbol, "linear")
        ask1 = bid1 = None
        try:
            a = ob.get("result", {}).get("a", [])
            b = ob.get("result", {}).get("b", [])
            # формат [[price, size], ...]
            ask1 = float(a[0][0]) if a and a[0] else None
            bid1 = float(b[0][0]) if b and b[0] else None
        except Exception:
            pass

        if ask1 is None or bid1 is None:
            # Если стакан не получили — отправляем чистый Market IOC
            qty_str = self._fmt_qty(symbol, qty)
            if qty_str is None:
                return {"retCode": 10001, "retMsg": "Qty invalid (below min after format)"}
            body = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_str,
                "positionIdx": position_idx,
                "timeInForce": "IOC",
            }
            return self._request("POST", "/v5/order/create", body=body)

        # 2) Считаем лимит-цену с допуском slip_percent
        k = 1.0 + (slip_percent * 0.01)
        if side == "Buy":
            price = ask1 * k
            # округление вверх к тику
            price = (int(price / tick + 0.9999999)) * tick
        else:
            price = bid1 / k
            # округление вниз к тику
            price = (int(price / tick)) * tick

        qty_str = self._fmt_qty(symbol, qty)
        if qty_str is None:
            return {"retCode": 10001, "retMsg": "Qty invalid (below min after format)"}
        price_str = self._fmt_price(symbol, price)
        # 3) Отправляем как Limit + IOC
        body = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": qty_str,
            "price": price_str,
            "positionIdx": position_idx,
            "timeInForce": "IOC",
        }
        return self._request("POST", "/v5/order/create", body=body)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        position_idx: int = 0,
        price: Optional[float] = None,
        takeProfit: Optional[str] = None,
        stopLoss: Optional[str] = None,
        tpslMode: Optional[str] = None,
        reduceOnly: Optional[bool] = None,
        timeInForce: str = "GoodTillCancel",
        tpOrderType: Optional[str] = "Market",
        slOrderType: Optional[str] = "Market",
        slippageToleranceType: Optional[str] = None,      # e.g., "0.5" for 0.50%
        slippageTolerance: Optional[str] = None,      # e.g., "0.5" for 0.50%
        preferSmart: bool = False,                         # route Market via Limit+IOC with computed price
    ):
        # Smart routing for Market: use Limit+IOC near top-of-book to avoid 30208
        if (order_type or "").lower() == "market" and preferSmart:
            return self.place_market_safe(symbol, side, qty, position_idx=position_idx)
        qty_str = self._fmt_qty(symbol, qty)
        if qty_str is None:
            return {"retCode": 10001, "retMsg": "Qty invalid (below min after format)"}

        price_str = None
        if price is not None and (order_type or "").lower() != "market":
            price_str = self._fmt_price(symbol, float(price))

        body = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty_str,
            "positionIdx": position_idx,
        }
        if price_str is not None:
            body["price"] = price_str

        if order_type.lower() == "market":
            body.pop("price", None)
            body["timeInForce"] = "IOC"
            # Optional explicit slippage tolerance for market orders
            if slippageToleranceType:
                body["slippageToleranceType"] = slippageToleranceType  # "Percent" or "TickSize"
            if slippageTolerance:
                body["slippageTolerance"] = str(slippageTolerance)
        else:
            body["timeInForce"] = timeInForce

        # TP/SL attach on order.create (v5 supports this)
        if takeProfit is not None:
            body["takeProfit"] = str(takeProfit)
            if tpOrderType is not None:
                body["tpOrderType"] = tpOrderType  # e.g., "Market"
        if stopLoss is not None:
            body["stopLoss"] = str(stopLoss)
            if slOrderType is not None:
                body["slOrderType"] = slOrderType  # e.g., "Market"
        if tpslMode is not None:
            body["tpslMode"] = tpslMode  # e.g., "Full"
        if reduceOnly is not None:
            # Bybit accepts boolean or "true"/"false" strings; we stringify to be safe
            body["reduceOnly"] = bool(reduceOnly)

        log.debug(f"[ORDER→] {body}")
        r = self._request("POST", "/v5/order/create", body=body)
        rc = r.get("retCode")
        if rc and rc != 0:
            log.error(f"[ORDER←][ERR] rc={rc} msg={r.get('retMsg')} body={body}")
        else:
            log.debug(f"[ORDER←] retCode={rc} {str(r)[:300]}")
        return r

    def position_list(self, symbol: str):
        return self._request("GET", "/v5/position/list", params={"category": "linear", "symbol": symbol})

    # --- Order status helpers ---
    def order_realtime(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None, category: str = "linear"):
        params = {"category": category, "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        return self._request("GET", "/v5/order/realtime", params=params)

    def order_history(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None, category: str = "linear"):
        params = {"category": category, "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        return self._request("GET", "/v5/order/history", params=params)

    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Возвращает {"status": str|None, "cumExecQty": float, "qty": float}
        Сначала смотрит realtime (открытые), затем history (закрытые/отменённые).
        """
        try:
            rt = self.order_realtime(symbol, order_id=order_id)
            lst = rt.get("result", {}).get("list", [])
            if lst:
                it = lst[0]
                return {
                    "status": it.get("orderStatus"),
                    "cumExecQty": float(it.get("cumExecQty") or 0.0),
                    "qty": float(it.get("qty") or 0.0),
                }
        except Exception:
            pass
        try:
            hs = self.order_history(symbol, order_id=order_id)
            lst = hs.get("result", {}).get("list", [])
            if lst:
                it = lst[0]
                return {
                    "status": it.get("orderStatus"),
                    "cumExecQty": float(it.get("cumExecQty") or 0.0),
                    "qty": float(it.get("qty") or 0.0),
                }
        except Exception:
            pass
        return {"status": None, "cumExecQty": 0.0, "qty": 0.0}

    def trading_stop(
        self,
        symbol: str,
        side: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        tpslMode: Optional[str] = None,          # "Full" or "Partial"
        positionIdx: Optional[int] = 0,          # 0 one-way, 1 long, 2 short (hedge)
        slOrderType: Optional[str] = None,       # "Market" or "Limit" (Partial)
        tpOrderType: Optional[str] = None,       # "Market" or "Limit" (Partial)
        slTriggerBy: Optional[str] = None,       # "LastPrice" (default), "MarkPrice", "IndexPrice"
        tpTriggerBy: Optional[str] = None,       # same as above
    ):
        body = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": positionIdx if positionIdx is not None else 0,
        }
        # Mode
        if tpslMode is not None:
            body["tpslMode"] = tpslMode
        # Core TP/SL values
        if stop_loss is not None:
            body["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            body["takeProfit"] = str(take_profit)
        if trailing_stop is not None:
            body["trailingStop"] = str(trailing_stop)
        # Optional order/trigger types
        if slOrderType is not None:
            body["slOrderType"] = slOrderType
        if tpOrderType is not None:
            body["tpOrderType"] = tpOrderType
        if slTriggerBy is not None:
            body["slTriggerBy"] = slTriggerBy
        if tpTriggerBy is not None:
            body["tpTriggerBy"] = tpTriggerBy

        log.debug(f"[TPSL→] {body}")
        r = self._request("POST", "/v5/position/trading-stop", body=body)
        rc = r.get("retCode")
        if rc and rc != 0:
            log.warning(f"[TPSL←][ERR] rc={rc} msg={r.get('retMsg')} body={body}")
        else:
            log.debug(f"[TPSL←] retCode={rc} {str(r)[:400]}")
        return r

    # --- WS connections ---
    async def ws_subscribe(self, url: str, topics: list[str], auth: bool = False):
        log.debug(f"[WS][CONNECT] {url} topics={topics} auth={auth}")
        async with websockets.connect(url, ping_interval=20) as ws:
            if auth:
                ts = str(int(time.time() * 1000))
                param_str = ts + self.api_key + str(self.recv_window)
                sign = self._sign(param_str)
                auth_msg = {"op": "auth", "args": [self.api_key, ts, self.recv_window, sign]}
                await ws.send(json.dumps(auth_msg))
                auth_resp = await ws.recv()
                log.debug(f"[WS][AUTH] {auth_resp}")

            sub_msg = {"op": "subscribe", "args": topics}
            await ws.send(json.dumps(sub_msg))
            log.debug(f"[WS][SUB] {sub_msg}")

            async for message in ws:
                log.debug(f"[WS][MSG] {str(message)[:400]}")
                yield json.loads(message)