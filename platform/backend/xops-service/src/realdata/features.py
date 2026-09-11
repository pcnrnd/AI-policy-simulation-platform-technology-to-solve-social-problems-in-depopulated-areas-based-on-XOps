"""실데이터 피처 엔지니어링 — 표본 t의 모든 피처는 base_ym < t 관측만 사용한다(R2-2 누출 차단).

랙이 결손이면(연속 관측 구간 밖) 표본을 만들지 않는다(R1-5). 표준화 통계는 models.py가
학습 구간 행만으로 별도 산출한다 — 여기서는 원 척도 피처만 만든다.
"""

from __future__ import annotations

import json
import math
from typing import Any

from src.core.settings import get_settings

LAG_FEATURES = ("y_lag1", "y_lag2", "y_lag3")
BASE_FEATURES = (*LAG_FEATURES, "y_yoy", "has_yoy", "month_sin", "month_cos")
_CROSS_FEATURE = "visitors_lag1"
_SALES_TARGET = "observed_sales_krw"

# build_feature_rows 최근 호출의 제외 사유별 카운트 — 인터페이스 시그니처를 바꾸지 않고
# 품질 보고(R1-5)에 노출하기 위한 보조 상태.
_last_excluded: dict[str, int] = {}


def excluded_rows_summary() -> dict[str, int]:
    """직전 `build_feature_rows` 호출의 제외 수(사유별)."""
    return dict(_last_excluded)


def _canonical_dongs() -> list[str]:
    doc = json.loads(get_settings().realdata_dong_map_path.read_text(encoding="utf-8"))
    return sorted(entry["dong_code"] for entry in doc["entries"])


def add_month(base_ym: int, delta: int) -> int:
    year, month = divmod(base_ym, 100)
    total = year * 12 + (month - 1) + delta
    year, month = divmod(total, 12)
    return year * 100 + month + 1


def _month_cycle(base_ym: int) -> tuple[float, float]:
    month = base_ym % 100
    angle = 2 * math.pi * month / 12
    return math.sin(angle), math.cos(angle)


def _group_by_dong(dataset: Any) -> dict[str, dict[int, dict[str, Any]]]:
    by_dong: dict[str, dict[int, dict[str, Any]]] = {}
    for row in dataset.rows:
        by_dong.setdefault(row["dong_code"], {})[row["base_ym"]] = row
    return by_dong


def _include_visitors_lag1(dataset: Any, by_dong: dict[str, dict[int, dict[str, Any]]]) -> bool:
    if dataset.target != _SALES_TARGET:
        return False
    return any(
        series.get(add_month(t, -1), {}).get("nonlocal_visitors") is not None
        for series in by_dong.values()
        for t in series
    )


def build_feature_rows(dataset: Any, *, for_month: int | None = None) -> list[dict[str, Any]]:
    """`dataset.rows`에서 표본을 만든다. `for_month`가 주어지면 그 달의 예측용 피처만 만든다."""
    global _last_excluded
    excluded = {"lag_missing": 0, "visitors_lag1_missing": 0}

    by_dong = _group_by_dong(dataset)
    # 소비 모델의 visitors_lag1: 실제로 단 한 건이라도 가능한지 먼저 확인(전혀 없으면 spec에서 제외)
    include_visitors_lag1 = _include_visitors_lag1(dataset, by_dong)

    targets: list[tuple[str, int]]
    if for_month is None:
        targets = sorted((code, t) for code, series in by_dong.items() for t in series)
    else:
        targets = [(code, for_month) for code in _canonical_dongs()]

    out: list[dict[str, Any]] = []
    for dong_code, t in targets:
        series = by_dong.get(dong_code, {})
        lags = [series.get(add_month(t, -k)) for k in (1, 2, 3)]
        if any(lag is None for lag in lags):
            excluded["lag_missing"] += 1
            continue
        lag1, lag2, lag3 = lags
        assert lag1 is not None and lag2 is not None and lag3 is not None  # 위 any() 가드로 보장됨

        visitors_lag1 = None
        if include_visitors_lag1:
            prev = series.get(add_month(t, -1))
            visitors_lag1 = prev.get("nonlocal_visitors") if prev else None
            if visitors_lag1 is None:
                excluded["visitors_lag1_missing"] += 1
                continue

        yoy_row = series.get(add_month(t, -12))
        month_sin, month_cos = _month_cycle(t)

        feature_row: dict[str, Any] = {
            "base_ym": t,
            "dong_code": dong_code,
            "y": series.get(t, {}).get("y"),
            "y_lag1": lag1["y"],
            "y_lag2": lag2["y"],
            "y_lag3": lag3["y"],
            "y_yoy": yoy_row["y"] if yoy_row is not None else 0,
            "has_yoy": 1 if yoy_row is not None else 0,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
        for code in _canonical_dongs():
            feature_row[f"dong_{code}"] = 1 if code == dong_code else 0
        if include_visitors_lag1:
            feature_row[_CROSS_FEATURE] = visitors_lag1
        out.append(feature_row)

    _last_excluded = excluded
    return out


def feature_names(dataset: Any) -> list[str]:
    """이 데이터셋의 학습에 쓰일 피처 이름 순서(dong one-hot 23 포함)."""
    names = list(BASE_FEATURES)
    if _include_visitors_lag1(dataset, _group_by_dong(dataset)):
        names.append(_CROSS_FEATURE)
    names.extend(f"dong_{code}" for code in _canonical_dongs())
    return names
