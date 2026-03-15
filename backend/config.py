import os

# === Core Settings ===
FPS = 30
ALERTS_DIR = os.getenv("ALERTS_DIR", "alerts")

# === VLM Settings (Custom Endpoint) ===
VLM_ENDPOINT = ""
VLM_RESIZE_WIDTH = 960
VLM_RESIZE_HEIGHT = 960

# === Gemini Settings ===
SEED_API_KEY = os.getenv("SEED_API_KEY", "")
SEED_BASE_URL = os.getenv("SEED_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"

VLM_WORKER_COUNT = int(os.getenv("VLM_WORKER_COUNT", "1"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))


GENERAL_WORKERS = 1
VEHICLE_COLOR_WORKERS = 3

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
)
LOG_DATEFMT = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")


def configure_logging():
    # Local import keeps this function safe even if module-level imports are edited.
    import logging as _logging

    level_name = (LOG_LEVEL or "INFO").upper()
    level = getattr(_logging, level_name, _logging.INFO)
    root = _logging.getLogger()
    if not root.handlers:
        _logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    else:
        root.setLevel(level)
    return level
