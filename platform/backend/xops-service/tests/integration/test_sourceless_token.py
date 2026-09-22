"""소스 무관 토큰 발급 (사용자 보고 1·2 — 등록용 토큰).

`/token/{source_id}` 는 미존재 소스에 유효 토큰을 내주지 않도록 카탈로그 존재를 검사한다(P-D2).
카탈로그 등록은 "아직 없는 소스"를 만드는 요청이라 그 검사와 충돌해, 소스를 고르기 전에는
등록용 토큰을 받을 수 없었다. 소스 표기가 없는 발급 경로를 따로 둔다.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.auth.jwt import decode_jwt

_NEW_SOURCE = {
    "id": "ds_test_sourceless",
    "label": "소스 무관 토큰 등록 테스트",
    "source": "RDB · PostgreSQL",
    "object": "tb_sourceless",
    "columns": [{"name": "row_id", "type": "INTEGER"}],
}


def test_sourceless_token_registers_a_brand_new_source(client: TestClient) -> None:
    issued = client.post("/api/v3/dataops/token")
    assert issued.status_code == 200
    token = issued.json()["access_token"]
    payload = decode_jwt(token)
    assert payload["source"] is None
    assert "data:write" in payload["scope"]

    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/v3/dataops/catalog", json=_NEW_SOURCE, headers=headers).status_code == 201
    assert client.delete(f"/api/v3/dataops/catalog/{_NEW_SOURCE['id']}", headers=headers).status_code == 200


def test_sourceless_oauth2_grant(client: TestClient) -> None:
    grant = client.post("/api/v3/dataops/oauth2")
    assert grant.status_code == 200
    assert decode_jwt(grant.json()["access_token"])["source"] is None


def test_source_bound_route_still_rejects_unknown_source(client: TestClient) -> None:
    """기존 가드는 그대로 — 미존재 소스 id 로는 여전히 토큰을 내주지 않는다."""
    assert client.post("/api/v3/dataops/token/ds_does_not_exist").status_code == 404
    assert client.post("/api/v3/dataops/oauth2/ds_does_not_exist").status_code == 404
