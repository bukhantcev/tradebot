# (relative path: entrypoint.sh)
#!/usr/bin/env bash
set -euo pipefail

: "${GIT_REPO:?GIT_REPO is required (https URL or SSH)}"
: "${GIT_BRANCH:=main}"

echo "[ENTRYPOINT] repo=$GIT_REPO branch=$GIT_BRANCH"

# Если репозиторий уже есть — обновим; иначе клонируем
if [ -d "/app/.git" ]; then
  echo "[ENTRYPOINT] existing repo found; pulling latest..."
  git -C /app config --global --add safe.directory /app || true
  git -C /app fetch --all --prune
  git -C /app reset --hard "origin/${GIT_BRANCH}"
else
  echo "[ENTRYPOINT] cloning fresh repo..."
  rm -rf /app/*
  git clone --depth 1 --branch "${GIT_BRANCH}" "${GIT_REPO}" /app
fi

# Установка зависимостей
if [ -f "/app/requirements.txt" ]; then
  echo "[ENTRYPOINT] installing requirements..."
  pip install --no-cache-dir -r /app/requirements.txt
fi

# Если есть .env.example и нет .env — скопируем шаблон (не обязательно)
if [ -f "/app/.env.example" ] && [ ! -f "/app/.env" ]; then
  cp /app/.env.example /app/.env || true
fi

cd /app
echo "[ENTRYPOINT] starting bot.py ..."
exec python bot.py