"""DataOps API 통합 테스트 — 인증·CRUD·계약 대조."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    assert client.get("/").json() == {"xops": "connected"}


def test_catalog_list_and_search(client: TestClient) -> None:
    all_sources = client.get("/api/v3/dataops/catalog").json()
    assert len(all_sources) == 7
    filtered = client.get("/api/v3/dataops/catalog", params={"q": "MongoDB"}).json()
    assert all("MongoDB" in (s.get("source") or "") for s in filtered)


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


def test_oauth2_token_usable(client: TestClient) -> None:
    grant = client.post("/api/v3/dataops/oauth2/ds_01_resident_registry").json()
    r = client.get(
        "/api/v3/dataops/ds_01_resident_registry",
        headers={"Authorization": f"Bearer {grant['access_token']}"},
    )
    assert r.status_code == 200
