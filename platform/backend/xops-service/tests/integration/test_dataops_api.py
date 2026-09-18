"""DataOps API 통합 테스트 — 인증·CRUD·계약 대조."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    assert client.get("/").json() == {"xops": "connected"}


def test_catalog_list_and_search(client: TestClient) -> None:
    # 데모 표시 ON 경로 — 시드 7종 + 실데이터 2종(ds_08·ds_09) + 외부데이터 3종(ds_10~12)
    all_sources = client.get("/api/v3/dataops/catalog", params={"include_seed": "true"}).json()
    assert len(all_sources) == 12
    filtered = client.get(
        "/api/v3/dataops/catalog", params={"q": "MongoDB", "include_seed": "true"}
    ).json()
    assert all("MongoDB" in (s.get("source") or "") for s in filtered)


def test_catalog_default_excludes_demo_seed(client: TestClient) -> None:
    """목록 기본값 = 데모 표시 OFF — 실데이터(ds_08~12)만 남는다."""
    body = client.get("/api/v3/dataops/catalog").json()

    assert [s["id"] for s in body] == [
        "ds_08_admin_boundary",
        "ds_09_welfare_facility",
        "ds_10_bccard_dong_industry_sales",
        "ds_11_kt_namwon_monthly_dong_visitors",
        "ds_12_kt_namwon_visitors_by_sex_age",
    ]
    assert all(not s["is_seed"] for s in body)
    # 검색도 같은 목록 위에서 돈다 — 시드 태그로는 아무것도 걸리지 않는다.
    assert client.get("/api/v3/dataops/catalog", params={"q": "인구이동"}).json() == []


def test_catalog_include_seed_restores_full_list(client: TestClient) -> None:
    """데모 표시 ON — 시드 7종이 되돌아온다. 목록에서 빠졌을 뿐 지워지지 않았다는 확인."""
    body = client.get("/api/v3/dataops/catalog", params={"include_seed": "true"}).json()

    assert len(body) == 12
    assert [s["id"] for s in body if s["is_seed"]] == [
        "ds_01_resident_registry",
        "ds_02_local_welfare",
        "ds_03_industrial_factories",
        "ds_04_spatial_geojson",
        "ds_05_smartfarm",
        "ds_06_settlement_facility",
        "ds_07_civil_complaints",
    ]


def test_seed_source_stays_reachable_when_hidden_from_listing(client: TestClient) -> None:
    """목록 필터는 단건 조회·토큰 발급에 걸리지 않는다 — 데모 ON에서 고른 소스가 404가 되면 안 된다."""
    assert client.get("/api/v3/dataops/catalog/ds_01_resident_registry").status_code == 200
    assert client.post("/api/v3/dataops/token/ds_01_resident_registry").status_code == 200


def test_unknown_source_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.get("/api/v3/dataops/nope", headers=auth_headers)
    assert r.status_code == 404


def test_get_requires_auth(client: TestClient) -> None:
    r = client.get("/api/v3/dataops/ds_01_resident_registry")
    assert r.status_code == 401
    assert r.json()["status"] == 401


def test_get_contract_matches_frontend(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.get(
        "/api/v3/dataops/ds_01_resident_registry",
        params={"page": 1, "page_size": 10, "sort": "reg_date"},
        headers=auth_headers,
    )
    j = r.json()
    assert r.status_code == 200
    assert j["endpoint"] == "/api/v3/dataops/ds_01_resident_registry"
    assert j["dataops_version"] == "3.0.0-R3"
    assert j["query_language"] == "SQL"
    assert j["auth"]["scope"] == "data:read data:write"
    assert j["range_scope"] == {"column": "reg_date", "from": "20210101", "to": "20261231"}
    assert j["pagination"] == {"page": 1, "page_size": 10, "total": 1248, "total_pages": 125}
    assert set(j.keys()) >= {"archive_meta", "generated_query", "db_adapter", "sample"}


def test_mongo_source_yields_mql(client: TestClient) -> None:
    token = client.post("/api/v3/dataops/token/ds_07_civil_complaints").json()["access_token"]
    r = client.get("/api/v3/dataops/ds_07_civil_complaints", headers={"Authorization": f"Bearer {token}"})
    j = r.json()
    assert j["query_language"] == "MQL"
    assert "db.col_civil_complaints.find(" in j["generated_query"]


def test_write_requires_write_scope_and_returns_affected(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.post("/api/v3/dataops/ds_01_resident_registry", json={"data": {}}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["affected_rows"] == 1


def test_delete_and_injection_guard(client: TestClient, auth_headers: dict[str, str]) -> None:
    ok = client.request(
        "DELETE", "/api/v3/dataops/ds_01_resident_registry", params={"filter": "in_flow_count > 100"}, headers=auth_headers
    )
    assert ok.json()["affected_rows"] == 1
    bad = client.get(
        "/api/v3/dataops/ds_01_resident_registry", params={"filter": "1=1; DROP TABLE x"}, headers=auth_headers
    )
    assert bad.status_code == 400


def test_token_issue_validates_catalog(client: TestClient) -> None:
    ok = client.post("/api/v3/dataops/token/ds_01_resident_registry")
    assert ok.status_code == 200
    assert "access_token" in ok.json()

    missing = client.post("/api/v3/dataops/token/nope")
    assert missing.status_code == 404


def test_oauth2_token_usable(client: TestClient) -> None:
    grant = client.post("/api/v3/dataops/oauth2/ds_01_resident_registry").json()
    r = client.get(
        "/api/v3/dataops/ds_01_resident_registry",
        headers={"Authorization": f"Bearer {grant['access_token']}"},
    )
    assert r.status_code == 200


def test_oauth2_issue_validates_catalog_like_token(client: TestClient) -> None:
    """P-D2: 두 발급 경로가 같은 access_token을 내주므로 존재 검증도 같아야 한다."""
    missing_oauth2 = client.post("/api/v3/dataops/oauth2/nope")
    missing_token = client.post("/api/v3/dataops/token/nope")

    assert missing_oauth2.status_code == 404
    assert missing_oauth2.status_code == missing_token.status_code
    assert "access_token" not in missing_oauth2.json()

    ok = client.post("/api/v3/dataops/oauth2/ds_01_resident_registry")
    assert ok.status_code == 200
    assert "access_token" in ok.json()


def test_degrade_surfaces_reason_in_response(client, auth_headers, monkeypatch) -> None:
    """저장소 왕복이 실패하면 사유를 응답에 싣는다 — 로그에만 남기면 스텁이 실조회로 보인다."""
    from src.dataops import service as service_module

    def _boom(_schema):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(service_module, "get_adapter", _boom)
    body = client.get("/api/v3/dataops/ds_01_resident_registry", headers=auth_headers).json()
    assert body["source_kind"] == "in-memory"
    assert "connection refused" in body["source_kind_reason"]

    written = client.post("/api/v3/dataops/ds_01_resident_registry", json={"data": {}}, headers=auth_headers).json()
    assert written["source_kind_reason"]
