"""R4 모니터링 단위 테스트 — SHAP 재구성 오차 < 1e-6, drift none/historical/operational, evaluation pending."""

from __future__ import annotations

from typing import Any

import pytest

from src.realdata import monitoring

_MODEL_ID = "namwon-nonlocal-visitors-next-month"
_VERSION = "v1"
_FEATURES = ["y_lag1", "y_lag2"]
_MEANS = {"y_lag1": 100.0, "y_lag2": 90.0}
_STDS = {"y_lag1": 10.0, "y_lag2": 9.0}
_COEF = [1.0, 2.0]
_INTERCEPT = 5.0
_OBSERVED_END = 202310
_FORECAST_MONTH = 202311


class FakeRealdataError(Exception):
    pass


class FakeErrors:
    RealdataError = FakeRealdataError


def _manual_predict(row: dict[str, float]) -> float:
    return _INTERCEPT + sum(_COEF[i] * (row[f] - _MEANS[f]) / _STDS[f] for i, f in enumerate(_FEATURES))


class FakeModels:
    def load_artifact(self, model_id: str, version: str) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "version": version,
            "dataset_id": "ds-1",
            "features": _FEATURES,
            "coef": _COEF,
            "intercept": _INTERCEPT,
            "train_means": _MEANS,
            "train_stds": _STDS,
            "observed_end_month": _OBSERVED_END,
            "forecast_month": _FORECAST_MONTH,
        }

    def predict(self, artifact: dict[str, Any], row: dict[str, float]) -> float:
        return _manual_predict(row)


class FakeSnapshot:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def load_dataset(self, dataset_id: str) -> dict[str, Any]:
        return {"rows": self._rows}


def _patch(monkeypatch: pytest.MonkeyPatch, *, rows: list[dict[str, Any]], candidate: dict[str, Any] | None) -> None:
    monkeypatch.setattr(monitoring, "_import_models", lambda: FakeModels())
    monkeypatch.setattr(monitoring, "_import_snapshot", lambda: FakeSnapshot(rows))
    monkeypatch.setattr(monitoring, "_import_errors", lambda: FakeErrors)
    monkeypatch.setattr(monitoring.candidates_mod, "get_candidate", lambda model_id, version: candidate)


def _candidate(eval_period: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "model_id": _MODEL_ID,
        "version": _VERSION,
        "dataset_id": "ds-1",
        "metrics": {"mae": 5.0, "rmse": 6.0, "wape": 0.1, "eval_period": eval_period},
        "baseline": {"name": "yoy", "mae": 8.0, "rmse": 9.0, "wape": 0.2},
        "status": "applied",
    }


# ── evaluation ───────────────────────────────────────────────
def test_evaluation_empty_when_candidate_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, rows=[], candidate=None)
    result = monitoring.evaluation(_MODEL_ID, _VERSION)
    assert result["status"] == "empty"


def test_evaluation_pending_when_forecast_month_not_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"base_ym": 202310, "dong_code": "45190250", "y_lag1": 110.0, "y_lag2": 95.0, "y": 500.0}]
    _patch(monkeypatch, rows=rows, candidate=_candidate({"from": 202307, "to": 202309, "n": 6}))

    result = monitoring.evaluation(_MODEL_ID, _VERSION)

    assert result["status"] == "ok"
    assert result["data"]["operational"]["kind"] == "pending"
    assert result["data"]["operational"]["pending_months"] == [_FORECAST_MONTH]
    assert result["data"]["validation"]["metrics"]["mae"] == 5.0


def test_evaluation_operational_when_forecast_month_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"base_ym": _FORECAST_MONTH, "dong_code": "45190250", "y_lag1": 110.0, "y_lag2": 95.0, "y": _manual_predict({"y_lag1": 110.0, "y_lag2": 95.0})},
        {"base_ym": _FORECAST_MONTH, "dong_code": "45190260", "y_lag1": 90.0, "y_lag2": 85.0, "y": _manual_predict({"y_lag1": 90.0, "y_lag2": 85.0}) + 2.0},
    ]
    _patch(monkeypatch, rows=rows, candidate=_candidate({"from": 202307, "to": 202309, "n": 6}))

    result = monitoring.evaluation(_MODEL_ID, _VERSION)

    op = result["data"]["operational"]
    assert op["kind"] == "operational"
    assert op["n"] == 2
    assert op["metrics"]["mae"] == pytest.approx(1.0, abs=1e-6)


# ── explain ──────────────────────────────────────────────────
def test_explain_reconstruction_within_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"base_ym": 202310, "dong_code": "45190250", "y_lag1": 110.0, "y_lag2": 95.0}]
    _patch(monkeypatch, rows=rows, candidate=_candidate(None))

    result = monitoring.explain(_MODEL_ID, _VERSION, 202310, "45190250")

    assert result["status"] == "ok"
    data = result["data"]
    assert data["reconstruction_check"] < 1e-6
    assert data["within_tolerance"] is True
    assert len(data["contributions"]) == len(_FEATURES)
    assert "인과" in data["note"]


def test_explain_empty_when_row_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, rows=[], candidate=_candidate(None))
    result = monitoring.explain(_MODEL_ID, _VERSION, 202310, "45190250")
    assert result["status"] == "empty"


# ── drift ────────────────────────────────────────────────────
def test_drift_status_none_when_no_post_data_and_no_eval_period(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"base_ym": 202301, "dong_code": "45190250", "y_lag1": 100.0, "y_lag2": 90.0, "y": 10.0}]
    _patch(monkeypatch, rows=rows, candidate=_candidate(None))

    result = monitoring.drift(_MODEL_ID, _VERSION)

    assert result["status"] == "ok"
    assert result["data"]["status"] == "none"
    assert result["data"]["kind"] is None


def test_drift_kind_historical_when_only_eval_period_available(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"base_ym": 202301, "dong_code": "45190250", "y_lag1": 100.0, "y_lag2": 90.0, "y": 10.0},
        {"base_ym": 202302, "dong_code": "45190250", "y_lag1": 101.0, "y_lag2": 91.0, "y": 11.0},
        {"base_ym": 202307, "dong_code": "45190250", "y_lag1": 130.0, "y_lag2": 120.0, "y": 40.0},
        {"base_ym": 202308, "dong_code": "45190250", "y_lag1": 131.0, "y_lag2": 121.0, "y": 41.0},
    ]
    _patch(monkeypatch, rows=rows, candidate=_candidate({"from": 202307, "to": 202309, "n": 6}))

    result = monitoring.drift(_MODEL_ID, _VERSION)

    assert result["status"] == "ok"
    assert result["data"]["kind"] == "historical"
    assert result["data"]["status"] == "ok"
    assert result["data"]["target"] is not None


def test_drift_kind_operational_when_post_observed_end_data_present(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"base_ym": 202301, "dong_code": "45190250", "y_lag1": 100.0, "y_lag2": 90.0, "y": 10.0},
        {"base_ym": 202302, "dong_code": "45190250", "y_lag1": 101.0, "y_lag2": 91.0, "y": 11.0},
        {"base_ym": _FORECAST_MONTH, "dong_code": "45190250", "y_lag1": 140.0, "y_lag2": 130.0, "y": 55.0},
    ]
    _patch(monkeypatch, rows=rows, candidate=_candidate({"from": 202307, "to": 202309, "n": 6}))

    result = monitoring.drift(_MODEL_ID, _VERSION)

    assert result["status"] == "ok"
    assert result["data"]["kind"] == "operational"
    assert len(result["data"]["features"]) == len(_FEATURES)
