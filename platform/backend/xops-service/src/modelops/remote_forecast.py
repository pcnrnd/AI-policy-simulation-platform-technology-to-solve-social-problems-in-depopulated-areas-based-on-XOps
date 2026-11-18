"""원격 예측 모델(Chronos-2) — 활성 릿지 버전과 같은 학습데이터로 원격 서버의 다음 달 예측·분위수를 받아 나란히 돌려준다.

실데이터는 스냅샷의 행정동별 월 계열을 한 묶음(다변량)으로, 데모는 합성 요인 패널의 월 계열 하나를 보낸다.
주소·토큰은 설정(XOPS_REMOTE_FORECAST_URL·XOPS_REMOTE_FORECAST_TOKEN)으로만 받는다. 미설정·시간 초과·오류면
"unavailable"을 돌려주고(화면은 릿지만 보인다) 서버 로그에는 사유 종류만 남긴다. 보낸 계열은 저장하지 않는다.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.core.settings import get_settings
from src.modelops import datasets, inference
from src.realdata.features import _canonical_dongs, add_month

logger = logging.getLogger(__name__)

REMOTE_MODEL = {"model_type": "remote_forecast", "name": "시계열 예측 모델", "model": "amazon/chronos-2"}
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
TIMEOUT_SECONDS = 5.0  # 첫 호출(GPU 커널 적재)이 1초 안팎 — 넘으면 원격 예측 없음
MAX_LENGTH = 600  # 원격 서버의 계열 길이 상한과 같다


def available() -> bool:
    settings = get_settings()
    return bool(settings.remote_forecast_url and settings.remote_forecast_token.get_secret_value())


def _months(start: int, end: int) -> list[int]:
    months = [start]
    while months[-1] < end:
        months.append(add_month(months[-1], 1))
    return months


def series(document: dict[str, Any], period: int, dong_code: str | None) -> tuple[list[list[float | None]], int] | None:
    """예측월 앞까지의 월 계열 묶음과 선택 계열의 위치. 데이터 종류가 계열을 만들 수 없으면 None."""
    if "panel" in document:
        rows = sorted((row for row in document["panel"]["rows"] if row["base_ym"] < period), key=lambda row: row["base_ym"])
        return ([[float(row["y"]) for row in rows][-MAX_LENGTH:]], 0) if rows else None
    if "rows" not in document:
        return None
    by_dong: dict[str, dict[int, float]] = {}
    for row in document["rows"]:
        if row["base_ym"] < period and row.get("y") is not None:
            by_dong.setdefault(row["dong_code"], {})[row["base_ym"]] = float(row["y"])
    if dong_code not in by_dong:
        return None
    months = _months(min(min(values) for values in by_dong.values()), add_month(period, -1))[-MAX_LENGTH:]
    # 관측이 하나도 없는 동은 묶음에서 뺀다(원격 서버가 거절한다). 공백 달은 None(결측)으로 그대로 보낸다.
    codes = [code for code in _canonical_dongs() if code in by_dong and any(month in by_dong[code] for month in months)]
    if dong_code not in codes:
        return None
    return [[by_dong[code].get(month) for month in months] for code in codes], codes.index(dong_code)


def _call(matrix: list[list[float | None]], index: int) -> dict[str, Any]:
    settings = get_settings()
    body = json.dumps({"series": matrix, "prediction_length": 1, "quantile_levels": list(QUANTILE_LEVELS)}, allow_nan=False)
    request = Request(settings.remote_forecast_url.rstrip("/") + "/forecast", data=body.encode(), method="POST",
                      headers={"Content-Type": "application/json", "X-Forecast-Token": settings.remote_forecast_token.get_secret_value()})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 — 주소는 운영 설정값
            low, median, high = (float(value) for value in json.loads(response.read())["quantiles"][index][0])
    except HTTPError as exc:
        logger.warning("remote forecast unavailable: HTTP %s", exc.code)
        return {"status": "unavailable"}
    except (URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("remote forecast unavailable: %s", type(exc).__name__)
        return {"status": "unavailable"}
    if not all(math.isfinite(value) for value in (low, median, high)) or not low <= median <= high:
        logger.warning("remote forecast unavailable: invalid quantiles")
        return {"status": "unavailable"}
    return {"status": "ok", **REMOTE_MODEL, "quantile_levels": list(QUANTILE_LEVELS), "low": low, "median": median, "high": high}


def compare(mode: str, instance_id: str, version_id: str, dong_code: str | None = None) -> dict[str, Any]:
    """활성 릿지 버전의 기준 예측(조정 없음)과 같은 달의 원격 예측(10~90% 범위)."""
    artifact, context, _ = inference._context(mode, instance_id, version_id, dong_code)
    proof = context["provenance"]
    ridge, _ = inference._calculate(artifact, context["inputs"])
    remote: dict[str, Any] = {"status": "unavailable"}
    if available():
        _, document = datasets.load(mode, proof["dataset_id"], proof["dataset_hash"])
        built = series(document, proof["prediction_period"], proof["dong_code"])
        if built is not None:
            remote = _call(*built)
    return {"mode": mode, "source": mode, "provenance": proof, "ridge": {"prediction": ridge}, "remote": remote}
