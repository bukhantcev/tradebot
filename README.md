# Bybit LLM Trader (V5, Unified)

## Быстрый старт
1) Склонируй папку, создавай `.env` по `.env.example`.
2) `docker compose up -d --build`
3) В Telegram: отправь `/start` в чат с ботом (тот `TELEGRAM_CHAT_ID`, что в .env).

## Что внутри
- `bybit_client.py` — REST/WS V5, подпись HMAC, базовые эндпоинты.
- `data.py` — WS kline 1m → SQLite `bars_1m`, агрегация в 5m.
- `features.py` — EMA/ATR/импульс/вола.
- `ml.py` — онлайн-логистика (без внешних ML зависимостей) + drift guard (каркас).
- `strategy.py` — смешение вероятности модели и тренд-байаса 5m, SL/TP от ATR.
- `trader.py` — риск-менеджмент от equity, выставление TP/SL через `/v5/position/trading-stop`.
- `llm.py` — мягкий тюнинг гиперпараметров через OpenAI.
- `bot.py` — aiogram 3.x команды: `/start`, `/stop`, `/status`, `/close_all`.
- `main.py` — оркестрация (данные, стратегия, LLM, Telegram).

## Замечания
- Переключение **real/testnet** — переменная `BYBIT_ENV`.
- Плечо фиксировано `LEVERAGE=1`.
- Риск: `RISK_PCT` (% от equity).
- Единственная позиция одновременно, cooldown 180s.
- По умолчанию вход **market**. Maker-логика может быть добавлена в trader (post-only лимит + TTL).
- Логи: `bot_debug.log` (DEBUG), консоль — INFO.

## To-Do / апгрейды
- Точные размеры позы по `position_list.size`.
- Хранить сделки/PNL в SQLite.
- Ввести label H=10m и обучать модель онлайн с отложенной меткой.
- Ограничение бюджета OpenAI по дням (отслеживать в локальном стейте).