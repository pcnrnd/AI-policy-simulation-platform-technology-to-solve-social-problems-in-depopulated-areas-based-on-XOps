"""아카이브 등록/삭제 CRUD 통합 테스트 (T2/⑥)."""

from __future__ import annotations

from fastapi.testclient import TestClient

_NEW_SOURCE = {
    "id": "ds_test_vacant_houses",
    "label": "빈집 실태조사",
    "source": "NoSQL · MongoDB",
    "object": "col_vacant_houses",
    "description": "테스트용 사용자 등록 소스",
    "tier": "Warm",
    "retention": "3년 보관",
    "tags": ["빈집", "테스트"],
    "columns": [{"name": "doc_seq", "type": "int"}, {"name": "status", "type": "string"}],
    "range": {"column": "doc_seq", "from": 1000, "to": 5000},
}


def test_register_appears_in_catalog_and_is_queryable(client: TestClient) -> None:
    created = client.post("/api/v3/dataops/catalog", json=_NEW_SOURCE)
    assert created.status_code == 201
    assert created.json()["id"] == "ds_test_vacant_houses"

    listed = {s["id"] for s in client.get("/api/v3/dataops/catalog").json()}
    assert "ds_test_vacant_houses" in listed

    # 등록 즉시 가상화 API 대상 — Mongo 유형이라 MQL 생성
    token = client.post("/api/v3/dataops/token/ds_test_vacant_houses").json()["access_token"]
    q = client.get("/api/v3/dataops/ds_test_vacant_houses", headers={"Authorization": f"Bearer {token}"}).json()
    assert q["query_language"] == "MQL"
    assert "db.col_vacant_houses.find(" in q["generated_query"]

    # 정리
    assert client.delete("/api/v3/dataops/catalog/ds_test_vacant_houses").status_code == 200


def test_duplicate_id_rejected(client: TestClient) -> None:
    payload = {**_NEW_SOURCE, "id": "ds_dup_test"}
    assert client.post("/api/v3/dataops/catalog", json=payload).status_code == 201
    dup = client.post("/api/v3/dataops/catalog", json=payload)
    assert dup.status_code == 409
    client.delete("/api/v3/dataops/catalog/ds_dup_test")


def test_delete_seed_source_forbidden(client: TestClient) -> None:
    r = client.delete("/api/v3/dataops/catalog/ds_01_resident_registry")
    assert r.status_code == 403


def test_delete_unknown_source_404(client: TestClient) -> None:
    assert client.delete("/api/v3/dataops/catalog/ghost").status_code == 404


def test_invalid_id_rejected(client: TestClient) -> None:
    bad = {**_NEW_SOURCE, "id": "bad id!"}
    assert client.post("/api/v3/dataops/catalog", json=bad).status_code == 422
