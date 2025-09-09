from typing import Dict, Any
from core.bybit.client import BybitClient
from config import cfg

async def gather_market_snapshot(by: BybitClient, symbol: str, category: str) -> Dict[str, Any]:
    kline_5 = await by.get_kline(symbol, interval="5", limit=200)
    kline_1 = await by.get_kline(symbol, interval="1", limit=200)
    ob = await by.get_orderbook(symbol, depth=50)
    balance = await by.get_wallet_balance()
    return {
        "symbol": symbol,
        "category": category,
        "kline_1": kline_1,
        "kline_5": kline_5,
        "orderbook": ob,
        "balance": balance
    }