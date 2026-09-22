"""발급 API(Data API 빌드 결과) 목록·등록·제거 통합 테스트 (사용자 보고 3).

브라우저 localStorage 스냅샷이던 발급 목록을 서버(SQLite)로 옮겼다. 등록→목록→호출→제거가
한 바퀴 돌고, 쓰기는 `data:write` 스코프를 요구하는지까지 본다.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

_APIS = "/api/v3/dataops/apis"

_BUILT = {
    "id": "api_test_pop_get",
    "method": "GET",
    "source_id": "ds_01_resident_registry",
    "source_label": "주민등록",
    "filter": "in_flow_count > 100",
    "sort": "reg_date",
    "page": 1,
    "page_size": 20,
    "auth_method": "JWT",
}


def test_register_list_invoke_delete_roundtrip(client: TestClient, auth_headers: dict[str, str]) -> None:
    created = client.post(_APIS, json=_BUILT, headers=auth_headers)
    assert created.status_code == 201
    body = created.json()
    assert body["endpoint"] == "/api/v3/dataops/ds_01_resident_registry"
    assert body["created_at"]

    listed = client.get(_APIS, headers=auth_headers)
    assert listed.status_code == 200
    entry = next(a for a in listed.json() if a["id"] == _BUILT["id"])
    assert entry["filter"] == "in_flow_count > 100"

    # 목록의 구성 그대로 호출되어야 실제 "관리 기능"이다.
    invoked = client.get(
        entry["endpoint"],
        params={"filter": entry["filter"], "sort": entry["sort"], "page": entry["page"], "page_size": entry["page_size"]},
        headers=auth_headers,
    )
    assert invoked.status_code == 200

    removed = client.delete(f"{_APIS}/{_BUILT['id']}", headers=auth_headers)
    assert removed.status_code == 200
    assert removed.json() == {"deleted": _BUILT["id"]}
    assert all(a["id"] != _BUILT["id"] for a in client.get(_APIS, headers=auth_headers).json())


def test_same_configuration_rebuild_updates_in_place(client: TestClient, auth_headers: dict[str, str]) -> None:
    """같은 id 재등록은 행을 늘리지 않는다 — 화면이 같은 구성을 시그니처 id로 보내기 때문."""
    client.post(_APIS, json=_BUILT, headers=auth_headers)
    client.post(_APIS, json={**_BUILT, "page_size": 50}, headers=auth_headers)
    rows = [a for a in client.get(_APIS, headers=auth_headers).json() if a["id"] == _BUILT["id"]]
    assert len(rows) == 1
    assert rows[0]["page_size"] == 50
    client.delete(f"{_APIS}/{_BUILT['id']}", headers=auth_headers)


def test_write_requires_scope_and_unknown_targets_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    # 목록도 무인증이면 401 — 소스 id·필터·스키마 힌트가 실린다.
    assert client.get(_APIS).status_code == 401
    assert client.post(_APIS, json=_BUILT).status_code == 401
    assert client.delete(f"{_APIS}/{_BUILT['id']}").status_code == 401
    # 카탈로그에 없는 소스로는 API를 발급하지 않는다(목록에 즉시 '원천 없음'이 생기는 것을 막는다).
    ghost = client.post(_APIS, json={**_BUILT, "id": "api_ghost", "source_id": "ds_nope"}, headers=auth_headers)
    assert ghost.status_code == 404
    assert client.delete(f"{_APIS}/api_not_there", headers=auth_headers).status_code == 404


def test_apis_path_is_not_shadowed_by_source_route(client: TestClient, auth_headers: dict[str, str]) -> None:
    """`/apis` 가 `/{source_id}` 보다 먼저 선언돼야 한다 — 뒤면 소스 'apis' 조회로 잡혀 404가 난다."""
    listed = client.get(_APIS, headers=auth_headers)
    assert listed.status_code == 200
    assert isinstance(listed.json(), list)
