"""R5 분석 API 통합 테스트 — TestClient로 라우트·인증·쿼리 검증·봉투 status를 확인한다."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.realdata import analysis

_URL = "/api/v3/realdata/analysis"
_DONG = "45190250"
_MODEL_VISITORS = "namwon-nonlocal-visitors-next-month"


class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows


def _summaries() -> list[dict[str, Any]]:
    return [
        {"dataset_id": "ds-v", "spec": {"target": "nonlocal_visitors"}, "created_at": "2026-01-01T00:00:00Z", "observed_to": 202310},
        {"dataset_id": "ds-s", "spec": {"target": "observed_sales_krw"}, "created_at": "2026-01-01T00:00:00Z", "observed_to": 202312},
    ]


def _rows(base_ym: int, dong: str, value: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"base_ym": base_ym, "dong_code": dong, "y": value, **(extra or {})}


def _patch_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    visitor_rows = [_rows(ym, _DONG, 1000.0, {"local_visitors": 2000.0, "foreign_visitors": 5.0}) for ym in range(202201, 202213)] + [
        _rows(ym, _DONG, 1000.0, {"local_visitors": 2000.0, "foreign_visitors": 5.0}) for ym in range(202301, 202311)
    ]
    sales_rows = [_rows(ym, _DONG, 500_000.0, {"observed_sales_krw": 500_000.0}) for ym in range(202201, 202213)] + [
        _rows(ym, _DONG, 500_000.0, {"observed_sales_krw": 500_000.0}) for ym in range(202301, 202313)
    ]
    datasets = {"ds-v": _FakeDataset(visitor_rows), "ds-s": _FakeDataset(sales_rows)}
    monkeypatch.setattr(analysis.snapshot_mod, "list_datasets", _summaries)
    monkeypatch.setattr(analysis.snapshot_mod, "load_dataset", lambda dataset_id: datasets[dataset_id])
    monkeypatch.setattr(analysis.candidates_mod, "get_active", lambda model_id: None)


def test_analysis_requires_auth(client: TestClient) -> None:
    response = client.get(_URL, params={"dong_code": _DONG, "base_ym": 202310})
    assert response.status_code == 401


def test_analysis_rejects_invalid_dong_code(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get(_URL, params={"dong_code": "not-a-dong", "base_ym": 202310}, headers=auth_headers)
    assert response.status_code == 422


def test_analysis_rejects_unknown_model_id(client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_full_pipeline(monkeypatch)
    response = client.get(_URL, params={"dong_code": _DONG, "base_ym": 202310, "model_id": "not-a-real-model"}, headers=auth_headers)
    assert response.status_code == 422


def test_analysis_rejects_unsupported_region(client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_full_pipeline(monkeypatch)
    response = client.get(_URL, params={"region": "seoul", "dong_code": _DONG, "base_ym": 202310}, headers=auth_headers)
    assert response.status_code == 422


def test_analysis_empty_when_no_datasets(client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis.snapshot_mod, "list_datasets", lambda: [])
    response = client.get(_URL, params={"dong_code": _DONG, "base_ym": 202310}, headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "empty"
    assert body["data"] is None
    assert "datasets" in body["message"]


def test_analysis_ok_envelope_shape_with_dong_all(client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_full_pipeline(monkeypatch)
    response = client.get(_URL, params={"dong_code": "all", "base_ym": 202310}, headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    data = body["data"]
    assert set(data) == {"region", "dong_code", "base_ym", "current", "diagnosis", "forecast", "responses", "provenance"}
    assert data["diagnosis"]["rules_version"] == "rules_v1"
    assert data["forecast"]["status"] == "model_required"  # 활성 모델 없음
    assert body["provenance"]["rules_version"] == "rules_v1"


def test_analysis_forecast_model_required_when_model_id_given(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_full_pipeline(monkeypatch)
    response = client.get(
        _URL, params={"dong_code": _DONG, "base_ym": 202310, "model_id": _MODEL_VISITORS}, headers=auth_headers
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["forecast"]["models"][_MODEL_VISITORS]["status"] == "model_required"
