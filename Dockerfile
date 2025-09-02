FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates tzdata bash file \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh
# Срежем CRLF, проверим тип, дадим права
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh && file /entrypoint.sh

ENV TZ=Europe/Moscow

# ВАЖНО: используем bash как интерпретатор, а не полагаемся на shebang
ENTRYPOINT ["bash","/entrypoint.sh"]