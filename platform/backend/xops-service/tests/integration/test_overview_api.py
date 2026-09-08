"""Overview 롤업 API 통합 테스트 — 빈 카탈로그·시드·라이브 혼합.

DSN 없이도 200 을 돌려주고(신규 저장소 쿼리가 없으므로), 값의 출처만 `source_kind` 로
바뀌는지를 고정한다. 프론트는 이 필드로 실데이터/시드 폴백을 결정한다.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core.settings import Settings
from src.dataops import adapters as adapters_module
from src.dataops import summary as summary_module

_URL = "/api/v3/overview/summary"


class _EmptyCatalog:
    """소스가 하나도 없는 카탈로그 — 시드 파일이 비었거나 아직 적재 전인 상태."""

    @staticmethod
    def list_sources() -> list[dict[str, Any]]:
        return []


def test_seed_catalog_rollup_is_public_and_matches_catalog(client: TestClient) -> None:
    """인증 없이 조회되고, 롤업 값이 /dataops/catalog 메타데이터와 일치한다."""
    catalog = client.get("/api/v3/dataops/catalog").json()
    expected_rows = {s["id"]: (s.get("archive") or {}).get("rows") or 0 for s in catalog}

    response = client.get(_URL)  # Authorization 헤더 없음 — 공개 조회
    body = response.json()

    assert response.status_code == 200
    assert body["source_count"] == len(catalog)
    assert {s["id"]: s["archive_rows"] for s in body["sources"]} == expected_rows
    assert body["archive_rows_total"] == sum(expected_rows.values())
    # 실데이터 2건 중 하나(사회복지시설 17행)가 카탈로그 수치 그대로 실려 온다.
    assert expected_rows["ds_09_welfare_facility"] == 17
    # DSN 미설정이 기본이므로 전 소스가 In-Memory 로 degrade 한다.
    assert body["source_kind"] == "in-memory"
    assert all(s["source_kind"] == "in-memory" for s in body["sources"])
    # 기존 DataOps 봉투 관례 유지 — 버전·엔드포인트·소스별 db_adapter.
    assert body["endpoint"] == _URL
    assert body["dataops_version"] == "3.0.0-R3"
    adapters = {s["id"]: s["db_adapter"] for s in body["sources"]}
    assert adapters["ds_04_spatial_geojson"] == "PostGISAdapter (EPSG:4326)"
    assert adapters["ds_07_civil_complaints"] == "MongoAdapter (Document Store)"


def test_model_snapshot_is_composed_from_orchestration_models(client: TestClient) -> None:
    """F1·운영 버전은 /orchestration/models 와 같은 출처를 합성한 값이다."""
    models = client.get("/api/v3/orchestration/models").json()
    serving = next(m for m in models if m["model_id"] == "population-forecast")

    model = client.get(_URL).json()["model"]

    assert model["serving_version"] == serving["version"]
    assert model["f1"] == serving["metrics"]["f1"]
    assert model["metrics_source"] == serving["metrics_source"]


def test_live_mixed_catalog_reports_database_per_storage_type(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postgres DSN 만 설정하면 PostgreSQL·PostGIS 소스만 실 저장소로 표기된다."""
    settings = Settings(pg_dsn="postgresql://pg/db")
    monkeypatch.setattr(adapters_module, "get_settings", lambda: settings)

    body = client.get(_URL).json()
    kinds = {s["id"]: s["source_kind"] for s in body["sources"]}

    # 하나라도 실 DSN 이 있으면 전체는 실 저장소 평면으로 본다.
    assert body["source_kind"] == "database"
    # PostgreSQL 과 PostGIS 는 같은 인스턴스라 pg_dsn 을 공유한다.
    assert kinds["ds_01_resident_registry"] == "database"
    assert kinds["ds_04_spatial_geojson"] == "database"
    # Timescale·Mongo 는 각자 DSN 이 없어 여전히 degrade 한다.
    assert kinds["ds_05_smartfarm"] == "in-memory"
    assert kinds["ds_07_civil_complaints"] == "in-memory"
    # 행수는 여전히 카탈로그 메타데이터 — DSN 설정이 값을 바꾸지 않는다(신규 쿼리가 없다).
    catalog = client.get("/api/v3/dataops/catalog").json()
    assert body["archive_rows_total"] == sum((s.get("archive") or {}).get("rows") or 0 for s in catalog)


def test_empty_catalog_returns_zeroed_rollup(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """소스가 없어도 500 이 아니라 0 으로 채운 롤업을 돌려준다(프론트가 폴백할 수 있게)."""
    monkeypatch.setattr(summary_module, "get_catalog", _EmptyCatalog)

    response = client.get(_URL)
    body = response.json()

    assert response.status_code == 200
    assert body["source_count"] == 0
    assert body["archive_rows_total"] == 0
    assert body["sources"] == []
    assert body["source_kind"] == "in-memory"
    assert body["model"]["model_id"] == "population-forecast"


def test_unknown_serving_model_yields_null_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """레지스트리에 해당 모델이 없으면 model 은 null — 프론트는 기존 F1 표시를 그대로 쓴다."""

    class _OtherModels:
        @staticmethod
        def models() -> list[dict[str, Any]]:
            return [{"model_id": "other", "version": "v1.0", "metrics": {"f1": 0.5}}]

    monkeypatch.setattr(summary_module, "get_registry", _OtherModels)

    assert summary_module.build_overview_summary()["model"] is None


def test_non_numeric_archive_rows_degrade_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """저장된 메타데이터의 rows 가 숫자가 아니어도 합계가 깨지지 않는다."""

    class _OddCatalog:
        @staticmethod
        def list_sources() -> list[dict[str, Any]]:
            return [
                {"id": "ds_text", "source": "RDB · PostgreSQL", "archive": {"rows": "많음"}},
                {"id": "ds_none", "source": "RDB · PostgreSQL", "archive": {}},
                {"id": "ds_neg", "source": "RDB · PostgreSQL", "archive": {"rows": -5}},
                {"id": "ds_real", "source": "RDB · PostgreSQL", "archive": {"rows": 17}},
            ]

    monkeypatch.setattr(summary_module, "get_catalog", _OddCatalog)

    body = summary_module.build_overview_summary()

    assert body["archive_rows_total"] == 17
    assert [s["archive_rows"] for s in body["sources"]] == [0, 0, 0, 17]
    # label 이 없으면 id 로 대체돼 도넛 범례가 비지 않는다.
    assert body["sources"][0]["label"] == "ds_text"
