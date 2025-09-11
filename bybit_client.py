import aiohttp
import asyncio
import ssl
import time
import hashlib
import hmac
import json
from typing import Any, Dict, Optional
from config import CFG
from log import log


# Helper to convert bools to "true"/"false" strings in dicts for Bybit API
def _convert_bools(d: dict | None):
    if not d:
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        else:
            out[k] = v
    return out

# Note: Tick rounding for stop prices is handled directly inside set_trading_stop_insurance_sl.

class BybitClient:
    def __init__(self):
        self.session: aiohttp.ClientSession | None = None
        self.tick_size: float = 0.1
        self.category: str = "linear" if str(CFG.bybit_category).lower() == "linear" else "inverse"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session and not self.session.closed:
            return self.session
        ssl_ctx = None
        if not CFG.bybit_verify_ssl:
            log.warning("[HTTP] SSL verification is DISABLED")
            ssl_ctx = False
        elif CFG.bybit_ca_bundle:
            ssl_ctx = ssl.create_default_context(cafile=CFG.bybit_ca_bundle)
        self.session = aiohttp.ClientSession(base_url=CFG.bybit_base_url, trust_env=True, connector=aiohttp.TCPConnector(ssl=ssl_ctx))
        return self.session

    def _sign(self, payload: str, ts: str) -> str:
        text = f"{ts}{CFG.bybit_api_key}5000{payload}"
        return hmac.new(CFG.bybit_api_secret.encode(), text.encode(), hashlib.sha256).hexdigest()

    async def _request(self, method: str, path: str, params: Dict[str, Any] | None = None, data: Dict[str, Any] | None = None, auth: bool = True):
        session = await self._get_session()
        url = path
        ts = str(int(time.time() * 1000))
        headers = {"Content-Type": "application/json"}
        # Convert bools to "true"/"false" for Bybit API
        params = _convert_bools(params) if isinstance(params, dict) else params
        raw_body = None
        payload_str = ""
        if method == "GET" and params:
            payload_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        elif method == "POST":
            if data is None:
                payload_str = ""
                raw_body = ""
            else:
                data = _convert_bools(data)
                payload_str = json.dumps(data, separators=(",", ":"))
                raw_body = payload_str
        if auth:
            sign = self._sign(payload_str, ts)
            headers.update({
                "X-BAPI-API-KEY": CFG.bybit_api_key,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-SIGN": sign,
                "X-BAPI-RECV-WINDOW": "5000",
                "X-BAPI-SIGN-TYPE": "2",
            })
        for attempt in range(3):
            try:
                log.info(f"[HTTP ->] {method} {path} params={params or {}} body={(payload_str if method=='POST' else '')}")
                if method == "POST":
                    async with session.request(method, url, data=raw_body, headers=headers) as resp:
                        res = await resp.json()
                else:
                    async with session.request(method, url, params=params, headers=headers) as resp:
                        res = await resp.json()
                log.info(f"[HTTP <-] {method} {path} status={resp.status} retCode={res.get('retCode')} payload={res}")
                return res
            except Exception as e:
                log.error(f"[RETRY] {method} {path} attempt={attempt+1} error={e}")
                await asyncio.sleep(1.5 * (attempt + 1))
        return None

    async def init_symbol_meta(self):
        res = await self._request("GET", "/v5/market/instruments-info", {"category": self.category, "symbol": CFG.symbol}, auth=False)
        try:
            self.tick_size = float(res["result"]["list"][0]["priceFilter"]["tickSize"])
            log.info(f"[META] tickSize={self.tick_size}")
        except Exception:
            log.error("[META] failed to parse tickSize")

    async def get_kline_5m(self, limit: int):
        return await self._request("GET", "/v5/market/kline", {"category": self.category, "symbol": CFG.symbol, "interval": "5", "limit": limit}, auth=False)

    async def get_wallet_balance(self):
        return await self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}, auth=True)

    async def get_total_equity(self):
        res = await self.get_wallet_balance()
        try:
            eq = res["result"]["list"][0]["totalEquity"]
            log.info(f"[BALANCE] totalEquity={eq}")
            return float(eq)
        except Exception:
            log.error("[BALANCE] unavailable: unexpected response: %s", res)
            return 0.0

    async def get_coin_equity(self, coin="USDT"):
        res = await self.get_wallet_balance()
        for c in res["result"]["list"][0]["coin"]:
            if c["coin"] == coin:
                log.info(f"[BALANCE] {coin} equity={c['equity']}")
                return float(c["equity"])
        return 0.0

    async def get_open_orders(self):
        """Fetch current open orders for the configured symbol."""
        params = {
            "category": self.category,
            "symbol": CFG.symbol,
        }
        return await self._request("GET", "/v5/order/realtime", params, auth=True)

    async def get_order_history(self, limit: int = 20):
        """Fetch recent order history."""
        params = {
            "category": self.category,
            "symbol": CFG.symbol,
            "limit": limit,
        }
        return await self._request("GET", "/v5/order/history", params, auth=True)

    async def get_executions(self, limit: int = 20):
        """Fetch recent trade executions (fills)."""
        params = {
            "category": self.category,
            "symbol": CFG.symbol,
            "limit": limit,
        }
        return await self._request("GET", "/v5/execution/list", params, auth=True)

    async def get_position_list(self):
        return await self._request("GET", "/v5/position/list", {"category": self.category, "symbol": CFG.symbol}, auth=True)

    async def get_positions(self, symbol: Optional[str] = None):
        """
        Backwards-compatible wrapper expected by trader:
        Returns positions list for the given symbol (defaults to CFG.symbol).
        """
        params: Dict[str, Any] = {"category": self.category}
        sym = symbol or CFG.symbol
        if sym:
            params["symbol"] = sym
        res = await self._request("GET", "/v5/position/list", params, auth=True)
        try:
            return res["result"]["list"]
        except Exception:
            log.error("[POS] unexpected response while fetching positions: %s", res)
            return []

    async def get_live_position(self, symbol: Optional[str] = None) -> Optional[dict]:
        """
        Возвращает первую найденную ОТКРЫТУЮ позицию по символу (size>0),
        либо None если позиций нет. Для hedge:
          positionIdx=1 -> long (side="Buy"), positionIdx=2 -> short (side="Sell").
        """
        sym = symbol or CFG.symbol
        try:
            positions = await self.get_positions(sym)
            if not positions:
                return None
            for p in positions:
                if p.get("symbol") != sym:
                    continue
                try:
                    size = float(p.get("size", 0))
                except Exception:
                    size = 0.0
                if size <= 0:
                    continue
                # Determine side from explicit side or positionIdx
                idx = int(p.get("positionIdx", 0)) if str(p.get("positionIdx", "0")).isdigit() else 0
                side = p.get("side")
                if not side:
                    if idx == 1:
                        side = "Buy"
                    elif idx == 2:
                        side = "Sell"
                # Parse numeric fields safely
                try:
                    avg_price = float(p.get("avgPrice", 0) or 0)
                except Exception:
                    avg_price = 0.0
                return {
                    "symbol": sym,
                    "positionIdx": idx,
                    "side": side,
                    "size": size,
                    "avgPrice": avg_price,
                    "stopLoss": p.get("stopLoss") or "",
                }
        except Exception as e:
            log.error(f"[POS] get_live_position failed: {e}")
        return None

    async def is_flat(self, symbol: Optional[str] = None) -> bool:
        """Удобный шорткат: True, если по символу нет открытой позиции на бирже."""
        return (await self.get_live_position(symbol)) is None

    async def set_leverage(self, leverage: float):
        payload = {
            "category": self.category,
            "symbol": CFG.symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        return await self._request("POST", "/v5/position/set-leverage", data=payload)

    async def place_market_order(
        self,
        *,
        side: str,
        qty: str,
        time_in_force: str = "IOC",
        reduce_only: bool = False,
        position_idx: int | None = None,
    ):
        body = {
            "category": self.category,
            "symbol": CFG.symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": time_in_force,
        }
        if reduce_only:
            body["reduceOnly"] = True  # will be converted to "true" by _convert_bools
        # Bybit v5: in one-way mode omit positionIdx; in hedge mode use 1 (Long) or 2 (Short)
        if position_idx in (1, 2):
            body["positionIdx"] = int(position_idx)
        return await self._request("POST", "/v5/order/create", data=body)

    async def set_trading_stop_insurance_sl(self, position_idx: int, stop_loss: float, side: Optional[str] = None):
        """
        Устанавливает страховочный стоп-лосс через /v5/position/trading-stop.
        Bybit v5 требует:
          - positionIdx обязательно:
              0 = one-way (side не нужен)
              1 = hedge long (side="Buy")
              2 = hedge short (side="Sell")
          - stopLoss строкой с учётом tickSize.
        """
        tick = getattr(self, "tick_size", 0.0) or 0.0
        if tick > 0:
            stop_loss_val = round(round(stop_loss / tick) * tick, 10)
        else:
            stop_loss_val = stop_loss

        payload = {
            "category": self.category,
            "symbol": CFG.symbol,
            "positionIdx": int(position_idx),
            "stopLoss": str(stop_loss_val),
            "tpslMode": "Full",
            "triggerBy": "LastPrice",
        }

        if position_idx in (1, 2):
            if side not in ("Buy", "Sell"):
                raise ValueError("For hedge positionIdx, side must be 'Buy' or 'Sell'")
            payload["side"] = side

        return await self._request("POST", "/v5/position/trading-stop", data=payload)

bybit = BybitClient()