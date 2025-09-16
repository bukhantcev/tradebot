import os, json, tempfile, shutil
from typing import Dict, Any
from config import PARAMS_PATH, DATA_DIR, DEFAULT_PARAMS

os.makedirs(DATA_DIR, exist_ok=True)

def load_params() -> Dict[str, Any]:
    if not os.path.exists(PARAMS_PATH):
        save_params(DEFAULT_PARAMS)
        return DEFAULT_PARAMS
    try:
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PARAMS

def save_params(params: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp()
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
        json.dump(params, tmp, ensure_ascii=False, indent=2)
    shutil.move(tmp_path, PARAMS_PATH)