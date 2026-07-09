"""mock_data.json 시드 로더 — 카탈로그·MLOps 시계열의 공통 데이터 출처."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.core.settings import get_settings


@lru_cache
def get_seed() -> dict[str, Any]:
    """mock_data.json 전체를 1회 로드해 캐시."""
    return json.loads(get_settings().mock_data_path.read_text(encoding="utf-8"))
