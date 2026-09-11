"""평가 — 시간 순차 3개월 고정 학습 평가(R2-3), MAE·RMSE·WAPE(R2-4), 전년동월 기준선.

분류 지표(Accuracy·F1)·LOO는 이 모듈에서 생성하지 않는다(계약 금지).
"""

from __future__ import annotations

from typing import Sequence

from src.realdata.errors import InsufficientData
from src.realdata.features import add_month

MIN_OBSERVED_MONTHS = 12
EVAL_HORIZON_MONTHS = 3


def _require_same_length(y_true: Sequence[float], y_pred: Sequence[float]) -> None:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true와 y_pred는 길이가 같고 비어 있지 않아야 합니다.")


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _require_same_length(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _require_same_length(y_true, y_pred)
    mean_sq_error = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return float(mean_sq_error**0.5)


def wape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """WAPE = Σ|e| / Σ|y| — Σ|y|=0이면 0.0(정의역 밖)."""
    _require_same_length(y_true, y_pred)
    denom = sum(abs(t) for t in y_true)
    if denom == 0:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / denom


def metric_bundle(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {"mae": round(mae(y_true, y_pred), 6), "rmse": round(rmse(y_true, y_pred), 6), "wape": round(wape(y_true, y_pred), 6)}


def require_min_observed_months(observed_months: set[int]) -> None:
    """선행 관측(월 수)이 12 미만이면 `InsufficientData`(R2-3)."""
    if len(observed_months) < MIN_OBSERVED_MONTHS:
        raise InsufficientData(
            f"선행 관측이 {len(observed_months)}개월로 최소 {MIN_OBSERVED_MONTHS}개월에 못 미칩니다."
        )


def eval_months(observed_end_month: int) -> list[int]:
    """평가 대상 3개월(T-2, T-1, T) — 시간순."""
    return [add_month(observed_end_month, -k) for k in (2, 1, 0)]
