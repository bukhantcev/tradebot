import logging
import sys
import asyncio

# Вытаскиваем имя текущей asyncio-задачи в логах
class TaskNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            task = asyncio.current_task()
            record.task = task.get_name() if task else "-"
        except Exception:
            record.task = "-"
        return True

FORMAT = "%(asctime)s | %(levelname)s | %(task)s | %(name)s:%(lineno)d | %(message)s"

def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # корневой DEBUG

    # Console DEBUG (всё в консоль)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(FORMAT))
    ch.addFilter(TaskNameFilter())
    root.addHandler(ch)

    # Файл DEBUG (дублируем)
    fh = logging.FileHandler("bot_debug.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(FORMAT))
    fh.addFilter(TaskNameFilter())
    root.addHandler(fh)

    # ВКЛЮЧАЕМ подробности внешних либ
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("websockets").setLevel(logging.DEBUG)
    logging.getLogger("aiogram").setLevel(logging.DEBUG)

    return root

logger = setup_logging()