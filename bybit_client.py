import time, hmac, hashlib, json, asyncio
from typing import Any, Dict, Optional
import httpx
from config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET, BYBIT_RECV_WINDOW, BYBIT_VERIFY_SSL

REST_MAIN = "https://api.bybit.com"
REST_TEST = "https://api-testnet.bybit.com"

class BybitClient:
    def __init__(self, verify_ssl: bool = BYBIT_VERIFY_SSL):
        self.base = REST_TEST if BYBIT_TESTNET else REST_MAIN
        self._client = httpx.AsyncClient(timeout=15.0, verify=verify_ssl)

    async def close(self):
        await self._client.aclose()

    def _sign(self, ts: int, recv_window: int, query: str, body: str) -> str:
        param_str = str(ts) + BYBIT_API_KEY + str(recv_window) + query + body
        return hmac.new(BYBIT_API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()

    async def _request(self, method: str, path: str, params: Dict[str, Any]=None, body: Dict[str, Any]=None, auth: bool=True):
        url = self.base + path
        params = params or {}
        body = body or {}
        headers = {"Content-Type": "application/json"}
        if auth:
            ts = int(time.time()*1000)
            query = "&".join([f"{k}={params[k]}" for k in sorted(params)]) if params else ""
            body_str = json.dumps(body) if body else ""
            sign = self._sign(ts, BYBIT_RECV_WINDOW, query, body_str)
            headers.update({
                "X-BAPI-API-KEY": BYBIT_API_KEY,
                "X-BAPI-TIMESTAMP": str(ts),
                "X-BAPI-RECV-WINDOW": str(BYBIT_RECV_WINDOW),
                "X-BAPI-SIGN": sign,
            })
        if method == "GET":
            r = await self._client.get(url, params=params, headers=headers)
        else:
            r = await self._client.post(url, params=params, json=body, headers=headers)
        r.raise_for_status()
        return r.json()

    # ======= REST wrappers =======
    async def instruments_info(self, category: str, symbol: str):
        return await self._request("GET", "/v5/market/instruments-info", {"category": category, "symbol": symbol}, auth=False)

    async def place_order(self, category: str, symbol: str, side: str, orderType: str, qty: str, price: Optional[str]=None,
                          timeInForce: Optional[str]=None, positionIdx: int=0, takeProfit: Optional[str]=None,
                          stopLoss: Optional[str]=None, tpslMode: Optional[str]=None, tpOrderType: Optional[str]=None,
                          slOrderType: Optional[str]=None, triggerBy: Optional[str]=None, reduceOnly: Optional[bool]=None,
                          closeOnTrigger: Optional[bool]=None, orderLinkId: Optional[str]=None):
        body = {
            "category": category, "symbol": symbol, "side": side, "orderType": orderType, "qty": qty,
            "positionIdx": positionIdx
        }
        if price: body["price"] = price
        if timeInForce: body["timeInForce"] = timeInForce
        if takeProfit: body["takeProfit"] = takeProfit
        if stopLoss: body["stopLoss"] = stopLoss
        if tpslMode: body["tpslMode"] = tpslMode
        if tpOrderType: body["tpOrderType"] = tpOrderType
        if slOrderType: body["slOrderType"] = slOrderType
        if triggerBy: body["tpTriggerBy"] = triggerBy; body["slTriggerBy"] = triggerBy
        if reduceOnly is not None: body["reduceOnly"] = reduceOnly
        if closeOnTrigger is not None: body["closeOnTrigger"] = closeOnTrigger
        if orderLinkId: body["orderLinkId"] = orderLinkId
        return await self._request("POST", "/v5/order/create", body=body)

    async def trading_stop(self, category: str, symbol: str, positionIdx: int, takeProfit: Optional[str]=None,
                           stopLoss: Optional[str]=None, tpOrderType: str="Market", slOrderType: str="Market",
                           tpslMode: str="Full", tpTriggerBy: str="LastPrice", slTriggerBy: str="LastPrice"):
        body = {"category": category, "symbol": symbol, "positionIdx": positionIdx,
                "tpslMode": tpslMode, "tpOrderType": tpOrderType, "slOrderType": slOrderType,
                "tpTriggerBy": tpTriggerBy, "slTriggerBy": slTriggerBy}
        if takeProfit: body["takeProfit"] = takeProfit
        if stopLoss: body["stopLoss"] = stopLoss
        return await self._request("POST", "/v5/position/trading-stop", body=body)

    async def position_list(self, category: str, symbol: str):
        return await self._request("GET", "/v5/position/list", {"category": category, "symbol": symbol})

    async def set_leverage(self, category: str, symbol: str, buyLeverage: str, sellLeverage: str):
        body = {"category": category, "symbol": symbol, "buyLeverage": buyLeverage, "sellLeverage": sellLeverage}
        return await self._request("POST", "/v5/position/set-leverage", body=body)

    async def cancel_all(self, category: str, symbol: str):
        body = {"category": category, "symbol": symbol}
        return await self._request("POST", "/v5/order/cancel-all", body=body)

    async def wallet_balance(self, accountType="UNIFIED"):
        return await self._request("GET", "/v5/account/wallet-balance", {"accountType": accountType})

    async def kline(self, category: str, symbol: str, interval: str, start: Optional[int]=None, end: Optional[int]=None, limit: int=200):
        params = {"category": category, "symbol": symbol, "interval": interval, "limit": str(limit)}
        if start: params["start"] = str(start)
        if end: params["end"] = str(end)
        return await self._request("GET", "/v5/market/kline", params, auth=False)