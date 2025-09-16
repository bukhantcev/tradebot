import time, hmac, hashlib, json, httpx, logging
from typing import Any, Dict, Optional
from config import (BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET, BYBIT_RECV_WINDOW, BYBIT_VERIFY_SSL)

REST_MAIN = "https://api.bybit.com"
REST_TEST = "https://api-testnet.bybit.com"

class BybitClient:
    def __init__(self, verify_ssl: bool = BYBIT_VERIFY_SSL):
        self.base = REST_TEST if BYBIT_TESTNET else REST_MAIN
        self._client = httpx.AsyncClient(timeout=15.0, verify=verify_ssl)
        self.log = logging.getLogger("bybit")

    async def close(self):
        await self._client.aclose()

    def _sign(self, ts: int, recv_window: int, query: str, body: str) -> str:
        param_str = str(ts) + BYBIT_API_KEY + str(recv_window) + query + body
        return hmac.new(BYBIT_API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()

    async def _request(self, method: str, path: str, params: Dict[str, Any]=None,
                       body: Dict[str, Any]=None, auth: bool=True):
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
        self.log.debug(f"HTTP {method} {path} params={params} body={body}")
        if method == "GET":
            r = await self._client.get(url, params=params, headers=headers)
        else:
            r = await self._client.post(url, params=params, json=body, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            self.log.error(f"HTTP error {e.response.status_code}: {e.response.text[:1000]}")
            raise
        self.log.debug(f"RESP {method} {path} -> {r.status_code} {r.text[:500]}")
        return r.json()

    # ===== REST wrappers =====
    async def instruments_info(self, category: str, symbol: str):
        return await self._request("GET", "/v5/market/instruments-info", {"category": category, "symbol": symbol}, auth=False)

    async def place_order(self, body: Dict[str, Any]):
        return await self._request("POST", "/v5/order/create", body=body)

    async def trading_stop(self, body: Dict[str, Any]):
        return await self._request("POST", "/v5/position/trading-stop", body=body)

    async def position_list(self, category: str, symbol: str):
        return await self._request("GET", "/v5/position/list", {"category": category, "symbol": symbol})

    async def set_leverage(self, category: str, symbol: str, buyLeverage: str, sellLeverage: str):
        return await self._request("POST", "/v5/position/set-leverage",
                                   body={"category": category, "symbol": symbol, "buyLeverage": buyLeverage, "sellLeverage": sellLeverage})

    async def cancel_all(self, category: str, symbol: str):
        return await self._request("POST", "/v5/order/cancel-all", body={"category": category, "symbol": symbol})

    async def wallet_balance(self, accountType="UNIFIED"):
        return await self._request("GET", "/v5/account/wallet-balance", {"accountType": accountType})