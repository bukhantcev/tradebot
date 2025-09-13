FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# + добавили build deps (build-essential, wget, tar) для сборки TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates tzdata bash file build-essential wget tar \
    && rm -rf /var/lib/apt/lists/*

# --- TA-Lib via manylinux wheel (no compiling C lib) ---
RUN pip install --no-cache-dir numpy==1.26.4 TA-Lib==0.6.7

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh
# Срежем CRLF, проверим тип, дадим права
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh && file /entrypoint.sh

ENV TZ=Europe/Moscow

# ВАЖНО: используем bash как интерпретатор, а не полагаемся на shebang
ENTRYPOINT ["bash","/entrypoint.sh"]