# (relative path: Dockerfile)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Базовые пакеты + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates tzdata bash \
    && rm -rf /var/lib/apt/lists/*

# Рабочая папка под код
WORKDIR /app

# Скрипт запуска
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Таймзона по умолчанию (можно переопределить .env)
ENV TZ=Europe/Moscow

ENTRYPOINT ["/entrypoint.sh"]