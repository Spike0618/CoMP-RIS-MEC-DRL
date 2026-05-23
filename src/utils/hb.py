"""
/


- 
-  train/eval 
"""

from __future__ import annotations

from datetime import datetime


def now_ts() -> str:
    """YYYYMMDD-HHMMSS"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def fmt_hms(sec: float) -> str:
    """ H/M/S  ETA """
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


def progress_bar(k: int, total: int, width: int = 24) -> str:
    """ ASCII """
    total = max(1, int(total))
    k = max(0, min(int(k), total))
    filled = int(width * (k / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
