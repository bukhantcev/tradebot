from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

def main_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Старт"), KeyboardButton(text="Стоп")],
            [KeyboardButton(text="Баланс"), KeyboardButton(text="Ордера")],
            [KeyboardButton(text="Статистика")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите команду…"
    )