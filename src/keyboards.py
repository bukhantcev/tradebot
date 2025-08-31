# src/keyboards.py
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from .config import BEST_TICKERS, RISK_PRESETS, INTERVAL_PRESETS, PAYIN_PRESETS

def main_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Старт BEST", callback_data="start_best")],
        [InlineKeyboardButton(text="🎯 Старт (выбор из 3)", callback_data="start_choose")],
        [InlineKeyboardButton(text="⏹ Стоп", callback_data="stop"), InlineKeyboardButton(text="❌ Отменить заявки", callback_data="cancel_orders")],
        [InlineKeyboardButton(text="💰 Баланс", callback_data="balance"),
         InlineKeyboardButton(text="📊 Позиции", callback_data="positions")],
        [InlineKeyboardButton(text="⚙️ Риск", callback_data="risk_menu"),
         InlineKeyboardButton(text="⏱ Интервал", callback_data="interval_menu")],
        [InlineKeyboardButton(text="➕ Пополнить", callback_data="payin_menu"),
         InlineKeyboardButton(text="ℹ️ Статус", callback_data="status")],
        [InlineKeyboardButton(text="🧾 Логи", callback_data="logs")],
    ])

def choose_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=t, callback_data=f"start_ticker:{t}") for t in BEST_TICKERS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])

def risk_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=f"{int(p*1000)/10:.1f}%", callback_data=f"risk_set:{p}") for p in RISK_PRESETS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])

def interval_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=lab, callback_data=f"interval_set:{lab}") for lab in INTERVAL_PRESETS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])

def payin_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=f"{amt//1000}k", callback_data=f"payin:{amt}") for amt in PAYIN_PRESETS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])
