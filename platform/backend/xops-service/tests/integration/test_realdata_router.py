"""실데이터 라우터 골격 통합 테스트 — `health`·`dong-map` 2개 라우트만 검증(M0 범위)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.v3 import realdata as realdata_module
from src.core.settings import Settings
from src.realdata.pg_reader import ALLOWED_TABLES

_HEALTH_URL = "/api/v3/realdata/health"
_DONG_MAP_URL = "/api/v3/realdata/dong-map"


def test_health_envelope_is_error_without_pg_dsn(client: TestClient) -> None:
    """기본 설정(XOPS_PG_DSN 미설정)에서는 봉투 status=error, HTTP 200(비즈니스 상태)."""
    response = client.get(_HEALTH_URL)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "error"
    assert "XOPS_PG_DSN" in body["message"]
    assert body["data"] is None
    assert body["provenance"]["computed_at"]


def test_health_reports_table_status_when_dsn_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DSN이 있으면 allowlist 전체 테이블의 존재 여부(불리언)를 반환한다 — COUNT가 아니다."""
    settings = Settings(pg_dsn="postgresql://xops:xops@localhost:5433/xops_dataops")
    monkeypatch.setattr(realdata_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        realdata_module,
        "fetch_table_existence",
        lambda tables: {table: table != "ext_gwto_daily_trend" for table in tables},
    )

    body = client.get(_HEALTH_URL).json()

    assert body["status"] == "ok"
    assert body["data"]["pg_dsn_configured"] is True
    status = body["data"]["table_status"]
    assert set(status) == ALLOWED_TABLES  # 29개 전부 계속 보고한다(계약 축소 없음)
    assert status["ext_gwto_daily_trend"] is False
    assert all(v is True for k, v in status.items() if k != "ext_gwto_daily_trend")


def test_dong_map_requires_auth(client: TestClient) -> None:
    """인증 없이 호출하면 401 (기존 dataops data:read 정책과 동일)."""
    response = client.get(_DONG_MAP_URL)
    assert response.status_code == 401


def test_dong_map_returns_23_entries_with_auth(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """대응표 23건 전량, KT dong_code·dong_name 필드만 포함(R1-4)."""
    response = client.get(_DONG_MAP_URL, headers=auth_headers)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    entries = body["data"]["entries"]
    assert len(entries) == 23
    assert len({e["dong_code"] for e in entries}) == 23  # 중복 없이 고유
    assert all(set(e) == {"dong_name", "dong_code"} for e in entries)
    assert body["provenance"]["rules_version"] == "v1"
