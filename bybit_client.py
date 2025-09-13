import aiohttp, asyncio, hmac, hashlib, time, json
from typing import Any, Dict, Optional
from functools import lru_cache
from config import CFG
from log import log

def _sign(payload: str) -> str:
    return hmac.new(CFG.bybit_api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

class BybitClient:

    def __init__(self):
        base = getattr(CFG, "base_url", None) or getattr(CFG, "bybit_base_url", None) or "https://api.bybit.com"
        self.base = str(base).rstrip("/")
        self.session: aiohttp.ClientSession | None = None
        self.verify_ssl = CFG.bybit_verify_ssl
        self._tick_size: Optional[float] = None

    async def open(self):
        """
        Open aiohttp session (idempotent). Respects CFG.bybit_verify_ssl.
        Logs endpoint base on success.
        """
        if self.session and not self.session.closed:
            return
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(ssl=False) if self.verify_ssl is False else None
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        log.info("[HTTP] session opened -> %s", self.base)

    async def close(self):
        """Close aiohttp session (idempotent)."""
        if self.session:
            try:
                await self.session.close()
                log.info("[HTTP] session closed")
            finally:
                self.session = None

    def _ts_ms(self) -> str:
        return str(int(time.time() * 1000))

    async def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None, auth: bool = False) -> Dict[str, Any]:
        """
        Unified HTTP requester with optional auth/signing for Bybit v5.
        - Logs outbound/inbound
        - Honors CFG.bybit_verify_ssl (set False to skip certificate verification)
        - Retries on network/JSON errors up to 3 times
        """
        await self.open()
        assert self.session is not None

        url = f"{self.base}{path}"
        params = params or {}
        data_str = json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""
        recv_window = "5000"

        headers = {
            "Content-Type": "application/json",
        }

        if auth:
            ts = self._ts_ms()
            # Bybit v5 sign string: timestamp + apiKey + recvWindow + (queryString for GET) or (body for others)
            if method.upper() == "GET":
                # Build canonical query string: key=value joined by '&', keys sorted asc, values as-is (no URL encoding)
                if params:
                    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
                    qs = "&".join(parts)
                else:
                    qs = ""
                sign_target = qs
            else:
                sign_target = data_str

            sign_payload = f"{ts}{CFG.bybit_api_key}{recv_window}{sign_target}"
            signature = _sign(sign_payload)
            headers.update({
                "X-BAPI-API-KEY": CFG.bybit_api_key,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
            })

        # for GET, body must be empty string
        method_up = method.upper()
        payload_for_log = data_str if data_str else ""
        log.info("[HTTP ->] %s %s params=%s body=%s", method_up, path, params, payload_for_log)

        # aiohttp SSL param: pass False explicitly to disable verification
        ssl_param = False if self.verify_ssl is False else None

        tries = 3
        for attempt in range(1, tries + 1):
            try:
                if method_up == "GET":
                    async with self.session.request(method_up, url, params=params, ssl=ssl_param, headers=headers) as resp:
                        text = await resp.text()
                else:
                    async with self.session.request(method_up, url, params={}, data=data_str, ssl=ssl_param, headers=headers) as resp:
                        text = await resp.text()

                try:
                    js = json.loads(text) if text else {}
                except Exception as jerr:
                    log.error("[HTTP] JSON parse error: %s text=%s", jerr, text[:500])
                    raise

                ret_code = js.get("retCode")
                log.info("[HTTP <-] %s %s status=%s retCode=%s payload=%s", method_up, path, resp.status, ret_code, str(js)[:500])
                return js
            except aiohttp.ClientError as e:
                if attempt >= tries:
                    log.error("[RETRY] %s %s attempt=%s err=%s", method_up, path, attempt, e)
                    return {"retCode": -1, "retMsg": str(e)}
                log.error("[RETRY] %s %s attempt=%s err=%s", method_up, path, attempt, e)
                await asyncio.sleep(1.0)
            except Exception as e:
                if attempt >= tries:
                    log.error("[RETRY] %s %s attempt=%s err=%s", method_up, path, attempt, e)
                    return {"retCode": -1, "retMsg": str(e)}
                log.error("[RETRY] %s %s attempt=%s err=%s", method_up, path, attempt, e)
                await asyncio.sleep(1.0)

    async def ensure_tick_size(self) -> float:
        """Fetch and cache tick size for the configured symbol."""
        if self._tick_size is not None:
            return self._tick_size
        info = await self.instruments_info(CFG.symbol)
        try:
            lst = info.get("result", {}).get("list", [])
            pf = lst[0]["priceFilter"]
            self._tick_size = float(pf.get("tickSize") or 0.1)
        except Exception:
            # sane default, but we log it; will work for BTCUSDT testnet (0.1)
            log.warning("[META] failed to parse tickSize, fallback 0.1")
            self._tick_size = 0.1
        log.info("[META] tickSize=%s", self._tick_size)
        return self._tick_size

    async def _round_price(self, price: float) -> float:
        ts = await self.ensure_tick_size()
        if ts <= 0:
            return price
        # round to nearest multiple of tick size
        return round(round(price / ts) * ts, 10)

    async def _last_price(self) -> Optional[float]:
        t = await self.tickers()
        try:
            lst = t.get("result", {}).get("list", [])
            p = float(lst[0].get("lastPrice"))
            return p
        except Exception:
            return None

    async def orderbook(self, limit: int = 1) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/v5/market/orderbook",
            {"category": CFG.category, "symbol": CFG.symbol, "limit": limit},
        )

    async def _best_aggressive_price(self, side: str) -> Optional[float]:
        """
        For Buy -> take bestAsk and add a small tick buffer.
        For Sell -> take bestBid and subtract a small tick buffer.
        """
        ob = await self.orderbook(limit=1)
        try:
            lst = ob.get("result", {}).get("a" if side == "long" else "b", [])
            # On Bybit v5, orderbook result is like {"a":[["price","size"]], "b":[["price","size"]]}
            price_str = lst[0][0]
            px = float(price_str)
            tick = await self.ensure_tick_size()
            # add/subtract 2 ticks to be safely marketable
            if side == "long":
                px = px + 2 * tick
            else:
                px = px - 2 * tick
            return await self._round_price(px)
        except Exception:
            return await self._last_price()

    # ---- endpoints ----
    async def instruments_info(self, symbol: str) -> Dict[str, Any]:
        return await self._request("GET", "/v5/market/instruments-info", {"category": CFG.category, "symbol": symbol})

    async def kline(self, interval: str, limit: int) -> Dict[str, Any]:
        return await self._request("GET", "/v5/market/kline", {"category": CFG.category, "symbol": CFG.symbol, "interval": interval, "limit": limit})

    async def tickers(self) -> Dict[str, Any]:
        return await self._request("GET", "/v5/market/tickers", {"category": CFG.category, "symbol": CFG.symbol})

    async def positions(self) -> Dict[str, Any]:
        return await self._request("GET", "/v5/position/list", {"category": CFG.category, "symbol": CFG.symbol}, auth=True)

    async def set_leverage(self, lev: float) -> Dict[str, Any]:
        body = {"category": CFG.category, "symbol": CFG.symbol, "buyLeverage": str(lev), "sellLeverage": str(lev)}
        return await self._request("POST", "/v5/position/set-leverage", body=body, auth=True)

    async def create_order(self, side: str, qty: float) -> Dict[str, Any]:
        """
        Try Market IOC first.
        If 30208 (price guard) or accepted-but-unfilled cases happen frequently, retry once with
        IOC Limit using aggressive price from best bid/ask with a small tick buffer.
        """
        body_market = {
            "category": CFG.category,
            "symbol": CFG.symbol,
            "side": "Buy" if side == "long" else "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
        }
        resp = await self._request("POST", "/v5/order/create", body=body_market, auth=True)
        if resp.get("retCode") == 0:
            return resp

        # Market rejected with max price guard: use aggressive IOC-Limit at book price +/- 2 ticks
        if resp.get("retCode") == 30208:
            px = await self._best_aggressive_price(side)
            if px is not None:
                body_limit = {
                    "category": CFG.category,
                    "symbol": CFG.symbol,
                    "side": "Buy" if side == "long" else "Sell",
                    "orderType": "Limit",
                    "qty": str(qty),
                    "price": str(px),
                    "timeInForce": "IOC",
                    "reduceOnly": False,
                }
                log.warning("[ORDER] Market rejected (30208). Retrying with IOC Limit at price=%s (aggr book)", px)
                return await self._request("POST", "/v5/order/create", body=body_limit, auth=True)

        # otherwise return the original error
        return resp

    async def wait_position_open(self, tries: int = 6, delay: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Poll /position/list a few times to confirm a position really opened.
        Returns the first non-zero position dict or None.
        """
        for i in range(tries):
            pos = await self.positions()
            lst = pos.get("result", {}).get("list", [])
            if lst:
                p = lst[0]
                try:
                    size = float(p.get("size", "0") or "0")
                except Exception:
                    size = 0.0
                if size > 0:
                    log.info("[CONFIRM-OPEN] position detected size=%s side=%s avgPrice=%s", size, p.get("side"), p.get("avgPrice"))
                    return p
            await asyncio.sleep(delay)
        log.warning("[CONFIRM-OPEN] not found on exchange after order")
        return None

    async def trading_stop(self, position_idx: int, stop_loss: float):
        # Double-check there's a non-zero position before setting SL to avoid 10001.
        pos = await self.positions()
        lst = pos.get("result", {}).get("list", [])
        if not lst:
            return {"retCode": 10001, "retMsg": "no position to set SL"}
        try:
            size = float(lst[0].get("size", "0") or "0")
        except Exception:
            size = 0.0
        if size <= 0:
            return {"retCode": 10001, "retMsg": "no position to set SL"}
        body = {
            "category": CFG.category, "symbol": CFG.symbol,
            "positionIdx": position_idx,
            "stopLoss": str(stop_loss),
            "tpslMode": "Full",
            "triggerBy": "LastPrice",
        }
        return await self._request("POST", "/v5/position/trading-stop", body=body, auth=True)

    async def close_all(self):
        # Market-close any position if size != 0
        pos = await self.positions()
        lst = pos.get("result", {}).get("list", [])
        if not lst:
            return {"closed": False, "reason": "no positions"}
        p = lst[0]
        # when in one-way mode, size field may be string; ensure float parsing works
        size = float(p.get("size", "0") or "0")
        side = p.get("side", "")
        if size == 0 or side == "":
            return {"closed": False, "reason": "flat"}
        close_side = "short" if side == "Buy" else "long"
        resp = await self.create_order(close_side, size)
        return {"closed": True, "resp": resp}

    async def wallet_unified(self) -> Dict[str, Any]:
        return await self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}, auth=True)

bybit = BybitClient()