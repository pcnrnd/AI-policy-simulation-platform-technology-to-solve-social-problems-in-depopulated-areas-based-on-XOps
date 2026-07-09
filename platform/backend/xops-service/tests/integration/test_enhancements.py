"""보완 항목 통합 테스트 — 드리프트→재학습 자동 연결·filter 컬럼 검증·시크릿 검증."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.core.settings import Settings


# ── ⑤ 드리프트 → 재학습 자동 연결 ──
def test_drift_breach_triggers_retrain(client: TestClient) -> None:
    r = client.get(
        "/api/v3/monitoring/drift",
        params={"drifted": "true", "model_id": "population-forecast", "auto_retrain": "true"},
    ).json()
    assert r["drifted"] is True
    assert r["retrain"] is not None
    assert r["retrain"]["trigger"] == "drift"
    assert r["retrain"]["state"] in ("succeeded", "rolled_back", "debounced")


def test_no_drift_no_retrain(client: TestClient) -> None:
    r = client.get(
        "/api/v3/monitoring/drift",
        params={"drifted": "false", "model_id": "population-forecast", "auto_retrain": "true"},
    ).json()
    assert r["drifted"] is False
    assert r["retrain"] is None


def test_drift_without_auto_retrain_flag(client: TestClient) -> None:
    r = client.get("/api/v3/monitoring/drift", params={"drifted": "true"}).json()
    assert r["retrain"] is None


# ── ⑦ filter 컬럼 검증 ──
def test_unknown_filter_column_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.get(
        "/api/v3/dataops/ds_01_resident_registry",
        params={"filter": "sentiment_score > 0"},  # 이 소스에 없는 컬럼
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_known_filter_column_ok(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.get(
        "/api/v3/dataops/ds_01_resident_registry",
        params={"filter": "in_flow_count > 100"},
        headers=auth_headers,
    )
    assert r.status_code == 200


# ── ② prod 시크릿 검증 ──
def test_prod_default_secret_rejected() -> None:
    settings = Settings(environment="prod", jwt_secret="dev-only-secret-change-me")
    with pytest.raises(RuntimeError):
        settings.validate_runtime()


def test_prod_custom_secret_ok() -> None:
    Settings(environment="prod", jwt_secret="a-real-secret").validate_runtime()


def test_dev_default_secret_ok() -> None:
    Settings(environment="dev").validate_runtime()
