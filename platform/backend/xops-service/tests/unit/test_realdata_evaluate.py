"""평가 지표·기준선·기간 단위 테스트(R2-3, R2-4) — Accuracy/F1 부재를 함께 확인한다."""

from __future__ import annotations

import pytest

from src.realdata import evaluate
from src.realdata.errors import InsufficientData


def test_mae_rmse_wape_match_hand_computation() -> None:
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]

    bundle = evaluate.metric_bundle(y_true, y_pred)

    # 손계산: |e| = [2,2,3], Σ|e|=7, Σe^2=4+4+9=17, Σ|y|=60 (지표는 소수 6자리 반올림 저장)
    assert bundle["mae"] == pytest.approx(7 / 3, abs=1e-6)
    assert bundle["rmse"] == pytest.approx((17 / 3) ** 0.5, abs=1e-6)
    assert bundle["wape"] == pytest.approx(7 / 60, abs=1e-6)


def test_metric_bundle_has_no_classification_keys() -> None:
    bundle = evaluate.metric_bundle([1.0, 2.0], [1.0, 2.0])
    assert set(bundle) == {"mae", "rmse", "wape"}


def test_wape_zero_denominator_returns_zero() -> None:
    assert evaluate.wape([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_require_min_observed_months_raises_below_threshold() -> None:
    with pytest.raises(InsufficientData):
        evaluate.require_min_observed_months(set(range(202201, 202212)))  # 11개월


def test_require_min_observed_months_passes_at_threshold() -> None:
    evaluate.require_min_observed_months(set(range(202201, 202213)))  # 12개월 — 예외 없음


def test_eval_months_returns_last_three_in_order() -> None:
    assert evaluate.eval_months(202304) == [202302, 202303, 202304]
    assert evaluate.eval_months(202312) == [202310, 202311, 202312]
