# src/AutoFisher/utils/paths.py
from __future__ import annotations
import os
from datetime import datetime

# Project root = parent of the "src" directory
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

LOG_DIR = os.path.join(PROJECT_ROOT, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)

def timestamp(fmt: str = "%d%m%Y%H%M%S") -> str:
    return datetime.now().strftime(fmt)

def log_path(script_name: str) -> str:
    """Returns Logs/<script>-logs-<timestamp>.txt"""
    return os.path.join(LOG_DIR, f"{script_name}-logs-{timestamp()}.txt")