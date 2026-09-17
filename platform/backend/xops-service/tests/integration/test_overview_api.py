"""Overview 롤업 API 통합 테스트 — 빈 카탈로그·확인 불가·실측 혼합.

저장소에 닿지 못해도 200 을 돌려주고, 행수를 **실제로 세었는지**가 `source_kind` 와
`archive_rows_unknown` 으로 드러나는지를 고정한다. 프론트는 이 필드로 실데이터/시드
폴백을 결정한다. 카탈로그의 `archive.rows` 메타데이터는 더 이상 합계에 쓰이지 않는다
(L0 감사 G3 — 데모 상수가 실측치로 오인됐다).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.dataops import liveness as liveness_module
from src.dataops import summary as summary_module

_URL = "/api/v3/overview/summary"


@pytest.fixture(autouse=True)
def _clear_live_cache() -> None:
    """롤업이 liveness TTL 캐시를 공유하므로 테스트마다 비운다."""
    liveness_module.reset_cache()


class _EmptyCatalog:
    """소스가 하나도 없는 카탈로그 — 시드 파일이 비었거나 아직 적재 전인 상태."""

    @staticmethod
    def list_sources() -> list[dict[str, Any]]:
        return []


def test_rollup_is_public_and_reports_unknown_when_no_storage(client: TestClient) -> None:
    """인증 없이 조회되고, 저장소에 닿지 못하면 0이 아니라 '확인 불가'로 보고한다."""
    catalog = client.get("/api/v3/dataops/catalog").json()

    response = client.get(_URL)  # Authorization 헤더 없음 — 공개 조회
    body = response.json()

    assert response.status_code == 200
    assert body["source_count"] == len(catalog)
    # DSN 미설정이 기본이라 한 건도 세지 못한다 — 합계 0, 전건 '확인 불가'.
    assert all(s["archive_rows"] is None for s in body["sources"])
    assert body["archive_rows_total"] == 0
    assert body["archive_rows_counted"] == 0
    assert body["archive_rows_unknown"] == len(catalog)
    assert body["source_kind"] == "in-memory"
    assert all(s["source_kind"] == "in-memory" for s in body["sources"])
    # 카탈로그의 데모용 archive.rows 는 더 이상 합계에 쓰이지 않는다(L0 G3).
    seed_total = sum((s.get("archive") or {}).get("rows") or 0 for s in catalog)
    assert seed_total > 0 and body["archive_rows_total"] != seed_total
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


def test_mixed_catalog_counts_only_reachable_sources(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """일부 소스만 셀 수 있으면 그 소스만 합계에 들어가고 나머지는 '확인 불가'로 남는다."""
    reachable = {"ds_01_resident_registry": 60, "ds_09_welfare_facility": 17}
    monkeypatch.setattr(liveness_module, "_count", lambda schema: reachable.get(schema["id"]))

    body = client.get(_URL).json()
    kinds = {s["id"]: s["source_kind"] for s in body["sources"]}
    rows = {s["id"]: s["archive_rows"] for s in body["sources"]}

    assert body["source_kind"] == "database"
    assert kinds["ds_01_resident_registry"] == "database"
    assert kinds["ds_05_smartfarm"] == "in-memory"
    assert rows["ds_01_resident_registry"] == 60
    assert rows["ds_05_smartfarm"] is None
    # 합계는 실제로 센 2건만 — 데모 메타데이터(1,248,000 등)가 섞이지 않는다.
    assert body["archive_rows_total"] == 77
    assert body["archive_rows_counted"] == 2
    assert body["archive_rows_unknown"] == body["source_count"] - 2


def test_zero_rows_is_counted_not_unknown(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """테이블이 비어 0건으로 확인된 소스는 '확인 불가'가 아니라 센 값 0이다."""
    monkeypatch.setattr(liveness_module, "_count", lambda schema: 0)

    body = client.get(_URL).json()

    assert body["archive_rows_total"] == 0
    assert body["archive_rows_unknown"] == 0
    assert body["archive_rows_counted"] == body["source_count"]
    assert body["source_kind"] == "database"


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


def test_seed_archive_rows_no_longer_reach_the_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """카탈로그 메타데이터의 rows 는 값이 무엇이든 합계에 영향을 주지 않는다(L0 G3)."""

    class _OddCatalog:
        @staticmethod
        def list_sources() -> list[dict[str, Any]]:
            return [
                {"id": "ds_text", "source": "RDB · PostgreSQL", "archive": {"rows": "많음"}},
                {"id": "ds_huge", "source": "RDB · PostgreSQL", "archive": {"rows": 1_248_000}},
                {"id": "ds_neg", "source": "RDB · PostgreSQL", "archive": {"rows": -5}},
                {"id": "ds_real", "source": "RDB · PostgreSQL", "archive": {"rows": 17}},
            ]

    monkeypatch.setattr(summary_module, "get_catalog", _OddCatalog)
    # 하나만 실제로 셀 수 있는 상황 — 나머지는 확인 불가.
    monkeypatch.setattr(
        liveness_module, "_count", lambda schema: 60 if schema["id"] == "ds_huge" else None
    )

    body = summary_module.build_overview_summary()

    assert body["archive_rows_total"] == 60  # 1,248,000 이 아니라 실측 60
    assert [s["archive_rows"] for s in body["sources"]] == [None, 60, None, None]
    assert body["archive_rows_unknown"] == 3
    # label 이 없으면 id 로 대체돼 도넛 범례가 비지 않는다.
    assert body["sources"][0]["label"] == "ds_text"
