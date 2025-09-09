import time
import hmac
import hashlib
import httpx
import logging
import json

log = logging.getLogger("BYBIT")

BASE_MAIN = "https://api.bybit.com"
BASE_TEST = "https://api-testnet.bybit.com"

class BybitClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base = BASE_TEST if testnet else BASE_MAIN
        self.http = httpx.AsyncClient(timeout=15)

    async def _signed(self, method: str, path: str, params: dict | None = None, body: dict | None = None):
        params = params or {}
        body = body or {}
        ts = str(int(time.time() * 1000))
        recv = "5000"
        # Build string to sign per Bybit v5: ts + api_key + recv_window + (query_string OR body_json)
        query_string = str(httpx.QueryParams(params)) if params else ""
        body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""
        to_sign = query_string if method.upper() == "GET" else body_json
        payload = ts + self.api_key + recv + to_sign
        sign = hmac.new(self.api_secret, payload.encode(), hashlib.sha256).hexdigest()

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv,
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }
        headers["X-BAPI-SIGN-TYPE"] = "2"
        url = self.base + path
        req_kwargs = {"headers": headers}
        if method.upper() == "GET":
            req_kwargs["params"] = params
        else:
            req_kwargs["json"] = body
        r = await self.http.request(method, url, **req_kwargs)
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") != 0:
            log.error("Bybit error %s %s: %s", method, path, j)
            raise RuntimeError(j)
        return j

    async def public(self, path: str, params: dict | None = None):
        url = self.base + path
        r = await self.http.get(url, params=params or {})
        r.raise_for_status()
        return r.json()

    # ---- Market data
    async def klines(self, category: str, symbol: str, interval: str, limit: int = 200):
        return await self.public("/v5/market/kline", {"category": category, "symbol": symbol, "interval": interval, "limit": limit})

    async def orderbook(self, category: str, symbol: str, limit: int = 50):
        return await self.public("/v5/market/orderbook", {"category": category, "symbol": symbol, "limit": limit})

    async def wallet_balance(self, account_type: str = "UNIFIED"):
        return await self._signed("GET", "/v5/account/wallet-balance", {"accountType": account_type})

    async def instruments(self, category: str, symbol: str):
        return await self.public("/v5/market/instruments-info", {"category": category, "symbol": symbol})

    # ---- Trading
    async def create_order(self, category: str, symbol: str, side: str, qty: float, order_type: str = "Market", positionIdx: int = 0):
        body = {
            "category": category,
            "symbol": symbol,
            "side": side,  # Buy/Sell
            "orderType": order_type,
            "qty": str(qty),
            "positionIdx": positionIdx,
        }
        return await self._signed("POST", "/v5/order/create", body=body)

    async def cancel_all(self, category: str, symbol: str):
        return await self._signed("POST", "/v5/order/cancel-all", body={"category": category, "symbol": symbol})

    async def close(self):
        await self.http.aclose()

    # Filters
    async def get_instrument_filters(self, category: str, symbol: str):
        j = await self.instruments(category, symbol)
        try:
            info = j["result"]["list"][0]
            lot = info["lotSizeFilter"]
            prc = info["priceFilter"]
            filters = {
                "minOrderQty": float(lot["minOrderQty"]),
                "qtyStep": float(lot["qtyStep"]),
                "minPrice": float(prc["minPrice"]),
                "tickSize": float(prc["tickSize"]),
            }
            log.info("Filters %s: %s", symbol, filters)
            return filters
        except Exception as e:
            log.exception("Failed to parse filters: %s", e)
            return {"minOrderQty": 0.001, "qtyStep": 0.001, "minPrice": 0.1, "tickSize": 0.1}