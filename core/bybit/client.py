import time
import hmac
import hashlib
import httpx
import logging
import json
from urllib.parse import urlencode

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
        method_u = method.upper()
        params = params or {}
        body = body or {}

        ts = str(int(time.time() * 1000))
        recv = "5000"

        # Canonical query (lexicographically sorted, urlencoded, no leading '?')
        canonical_query = urlencode(sorted(params.items()), doseq=True) if params else ""

        # Canonical body (sorted keys, compact separators) only for methods with body
        if method_u in {"POST", "PUT", "PATCH", "DELETE"} and body:
            canonical_body = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        else:
            canonical_body = ""

        # Sign payload: ts + api_key + recv_window + query + body
        payload = ts + self.api_key + recv + canonical_query + canonical_body
        sign = hmac.new(self.api_secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

        # ---- debug snapshot for 10004 diagnostics
        _dbg = {
            "ts": ts,
            "api_key_prefix": (self.api_key[:6] + "…"),
            "recv": recv,
            "canonical_query": canonical_query,
            "canonical_body": canonical_body,
            "payload_preview": (payload[:200] + ("…[truncated]" if len(payload) > 200 else "")),
            "signature": sign,
        }

        log.debug(
            "[BYBIT SIGN] %s %s | qs='%s' body_len=%d",
            method_u, path, canonical_query, len(canonical_body)
        )

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv,
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json",
        }

        # Build URL with canonical query to ensure it matches the signed string
        url = self.base + path + (("?" + canonical_query) if canonical_query else "")
        req_kwargs = {"headers": headers}
        # For signed POST-like calls, send the exact canonical body bytes (sorted keys, compact) to match signature
        if method_u in {"POST", "PUT", "PATCH", "DELETE"} and canonical_body:
            req_kwargs["content"] = canonical_body.encode("utf-8")

        log.debug("[BYBIT REQ] %s %s | url=%s | body='%s'", method_u, path, url, canonical_body if canonical_body else "")

        r = await self.http.request(method_u, url, **req_kwargs)
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") != 0:
            # Extra diagnostics for signature issues
            if j.get("retCode") == 10004:
                log.error("Bybit SIGNATURE MISMATCH (%s %s): %s | debug=%s", method_u, path, j, _dbg)
            else:
                log.error("Bybit error %s %s: %s", method_u, path, j)
            raise RuntimeError(j)
        return j

    async def query_api_key(self):
        """
        Helper for debugging: signed GET to Bybit to verify that keys, recvWindow and signature formatting are valid.
        Docs: GET /v5/user/query-api
        """
        return await self._signed("GET", "/v5/user/query-api")

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