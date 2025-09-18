import time
import hmac
import hashlib
import json
import logging
from typing import Any, Dict, Optional

import httpx
import websockets
import asyncio

from config import get_bybit_keys, BYBIT_ENV

logger = logging.getLogger("BYBIT")

class BybitClient:
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

    # --- Sign helper ---
    def _sign(self, payload: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    # --- REST request ---
    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None):
        url = self.rest_url + path
        ts = str(int(time.time() * 1000))
        body_str = json.dumps(body) if body else ""
        query_str = ""
        if params:
            query_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        to_sign = ts + self.api_key + str(self.recv_window) + query_str + body_str
        sign = self._sign(to_sign)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }

        logger.debug(f"[HTTP] {method} {url} params={params} body={body_str}")

        resp = self.session.request(method, url, params=params, content=body_str, headers=headers)
        data = resp.json()
        if data.get("retCode") != 0:
            logger.error(f"[HTTP][ERROR] {data}")
        return data

    # --- Public REST endpoints ---
    def instruments_info(self, category="linear", symbol="BTCUSDT"):
        return self._request("GET", "/v5/market/instruments-info", params={"category": category, "symbol": symbol})

    def server_time(self):
        return self._request("GET", "/v5/market/time")

    # --- Private REST endpoints ---
    def wallet_balance(self, account_type="UNIFIED"):
        return self._request("GET", "/v5/account/wallet-balance", params={"accountType": account_type})

    def set_leverage(self, symbol: str, buy_leverage: int, sell_leverage: int):
        return self._request("POST", "/v5/position/set-leverage", body={
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        })

    def place_order(self, symbol: str, side: str, qty: float, order_type="Market", position_idx=0, price: Optional[float] = None):
        body = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "GoodTillCancel",
            "positionIdx": position_idx,
        }
        if price is not None:
            body["price"] = str(price)
        return self._request("POST", "/v5/order/create", body=body)

    def position_list(self, symbol: str):
        return self._request("GET", "/v5/position/list", params={"category": "linear", "symbol": symbol})

    def trading_stop(self, symbol: str, side: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None, trailing_stop: Optional[float] = None):
        body = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": 0,
            "tpslMode": "Full",
        }
        if stop_loss:
            body["stopLoss"] = str(stop_loss)
        if take_profit:
            body["takeProfit"] = str(take_profit)
        if trailing_stop:
            body["trailingStop"] = str(trailing_stop)
        return self._request("POST", "/v5/position/trading-stop", body=body)

    # --- WS connections ---
    async def ws_subscribe(self, url: str, topics: list[str], auth: bool = False):
        async with websockets.connect(url, ping_interval=20) as ws:
            if auth:
                ts = str(int(time.time() * 1000))
                param_str = ts + self.api_key + str(self.recv_window)
                sign = self._sign(param_str)
                await ws.send(json.dumps({
                    "op": "auth",
                    "args": [self.api_key, ts, self.recv_window, sign],
                }))
                resp = await ws.recv()
                logger.info(f"[WS][AUTH] {resp}")

            sub_msg = {"op": "subscribe", "args": topics}
            await ws.send(json.dumps(sub_msg))
            logger.info(f"[WS] Subscribed: {topics}")

            async for message in ws:
                yield json.loads(message)