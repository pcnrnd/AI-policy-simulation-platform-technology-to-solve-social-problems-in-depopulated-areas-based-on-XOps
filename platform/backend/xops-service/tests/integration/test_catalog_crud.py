"""아카이브 등록/삭제 CRUD 통합 테스트 (T2/⑥).

등록·삭제는 `data:write` 스코프를 요구한다(조회 2개는 공개). 정상 흐름은 `auth_headers`
픽스처의 토큰을 붙이고, 무인증·위조 토큰이 실제로 401 로 막히는지를 별도로 본다.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

_CATALOG = "/api/v3/dataops/catalog"

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


def test_register_appears_in_catalog_and_is_queryable(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = client.post(_CATALOG, json=_NEW_SOURCE, headers=auth_headers)
    assert created.status_code == 201
    assert created.json()["id"] == "ds_test_vacant_houses"

    listed = {s["id"] for s in client.get(_CATALOG).json()}
    assert "ds_test_vacant_houses" in listed

    # 등록 즉시 가상화 API 대상 — Mongo 유형이라 MQL 생성
    token = client.post("/api/v3/dataops/token/ds_test_vacant_houses").json()["access_token"]
    q = client.get("/api/v3/dataops/ds_test_vacant_houses", headers={"Authorization": f"Bearer {token}"}).json()
    assert q["query_language"] == "MQL"
    assert "db.col_vacant_houses.find(" in q["generated_query"]

    # 정리
    deleted = client.delete(f"{_CATALOG}/ds_test_vacant_houses", headers=auth_headers)
    assert deleted.status_code == 200


def test_duplicate_id_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    payload = {**_NEW_SOURCE, "id": "ds_dup_test"}
    assert client.post(_CATALOG, json=payload, headers=auth_headers).status_code == 201
    dup = client.post(_CATALOG, json=payload, headers=auth_headers)
    assert dup.status_code == 409
    client.delete(f"{_CATALOG}/ds_dup_test", headers=auth_headers)


def test_delete_seed_source_forbidden(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.delete(f"{_CATALOG}/ds_01_resident_registry", headers=auth_headers)
    assert r.status_code == 403


def test_delete_unknown_source_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    assert client.delete(f"{_CATALOG}/ghost", headers=auth_headers).status_code == 404


def test_invalid_id_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    bad = {**_NEW_SOURCE, "id": "bad id!"}
    assert client.post(_CATALOG, json=bad, headers=auth_headers).status_code == 422


# ── 인증 경계 ──
def test_register_without_token_rejected(client: TestClient) -> None:
    """무인증 등록은 401 이고, 카탈로그에 아무것도 남기지 않는다."""
    r = client.post(_CATALOG, json={**_NEW_SOURCE, "id": "ds_noauth_probe"})
    assert r.status_code == 401
    assert r.json()["error"] == "AuthError"
    assert "ds_noauth_probe" not in {s["id"] for s in client.get(_CATALOG).json()}


def test_delete_without_token_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    """무인증 삭제는 401 이고, 등록된 소스가 그대로 남는다."""
    probe = {**_NEW_SOURCE, "id": "ds_delauth_probe"}
    assert client.post(_CATALOG, json=probe, headers=auth_headers).status_code == 201
    try:
        r = client.delete(f"{_CATALOG}/ds_delauth_probe")
        assert r.status_code == 401
        assert r.json()["error"] == "AuthError"
        assert client.get(f"{_CATALOG}/ds_delauth_probe").status_code == 200
    finally:
        client.delete(f"{_CATALOG}/ds_delauth_probe", headers=auth_headers)


def test_forged_token_rejected(client: TestClient) -> None:
    """서명이 맞지 않는 토큰은 401 — 인증이 시드 보호(403)보다 먼저 걸린다."""
    forged = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.e30.not-a-valid-signature"}
    assert client.post(_CATALOG, json=_NEW_SOURCE, headers=forged).status_code == 401
    assert client.delete(f"{_CATALOG}/ds_01_resident_registry", headers=forged).status_code == 401
