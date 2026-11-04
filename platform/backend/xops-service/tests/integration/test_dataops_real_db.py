"""실 저장소 연동 통합 테스트 — 어댑터를 주입해 HTTP 응답까지 검증.

Docker 미설치 환경에서도 "실 DB 연결 시 응답이 어떻게 달라지는가"를 고정한다.
실 컨테이너 왕복은 별도 스모크(validationNotRun)로 남긴다.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from src.core import db
from src.dataops import service as service_module
from src.dataops.catalog import get_catalog
from src.dataops.results import ExecutionResult

# 실 저장소 경로는 실데이터(real) 모드 소스에서만 열린다. 데모 시드(ds_01·ds_07)와 같은 열 구성의
# 실데이터 소스를 이 모듈 동안만 등록해 같은 계약을 검증한다(헤더 없는 요청 = real).
_SOURCE = "rt_resident_registry"
_URL = f"/api/v3/dataops/{_SOURCE}"
_MONGO_SOURCE = "rt_civil_complaints"


@pytest.fixture(autouse=True, scope="module")
def _real_sources() -> Iterator[None]:
    for source_id, seed_id in ((_SOURCE, "ds_01_resident_registry"), (_MONGO_SOURCE, "ds_07_civil_complaints")):
        schema = {**get_catalog()._seed[seed_id], "id": source_id, "is_seed": False, "mode": "real", "user_registered": True}
        db.delete_user_source(source_id)
        db.add_user_source(schema)
    yield
    for source_id in (_SOURCE, _MONGO_SOURCE):
        db.delete_user_source(source_id)


class _StubAdapter:
    """실 저장소가 응답한 것처럼 행·총계를 돌려주는 어댑터."""

    name = "PostgreSQLAdapter"

    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.requests: list[Any] = []

    def execute(self, request: Any) -> ExecutionResult:
        self.requests.append(request)
        return self._result


def _inject(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    monkeypatch.setattr(service_module, "get_adapter", lambda schema: adapter)


def test_get_returns_real_rows_when_adapter_executes(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"reg_date": "20210101", "region_code": "45190", "in_flow_count": 120, "out_flow_count": 180, "age_group": "20s"},
        {"reg_date": "20210131", "region_code": "46900", "in_flow_count": 121, "out_flow_count": 181, "age_group": "40s"},
    ]
    adapter = _StubAdapter(ExecutionResult(rows=rows, total=1873, executed=True))
    _inject(monkeypatch, adapter)

    body = client.get(_URL, params={"page": 1, "page_size": 10}, headers=auth_headers).json()

    # 실 저장소에서 읽은 행이 그대로 노출된다.
    assert body["rows"] == rows
    assert body["source_kind"] == "database"
    # 페이징 총계도 스텁(1248)이 아니라 실제 COUNT 결과를 쓴다.
    assert body["pagination"]["total"] == 1873
    assert body["pagination"]["total_pages"] == 188
    assert body["result_rows"] == 2
    # 어댑터에는 표시된 SQL과 같은 문자열이 전달된다.
    assert adapter.requests[0].sql == body["generated_query"]
    assert adapter.requests[0].method == "GET"
    # 기존 계약 필드는 그대로 유지된다.
    assert body["dataops_version"] == "3.0.0-R3"
    assert body["range_scope"]["column"] == "reg_date"


def test_get_keeps_stub_contract_when_adapter_degrades(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject(monkeypatch, _StubAdapter(ExecutionResult.not_executed("DSN 없음")))

    body = client.get(_URL, params={"page": 1, "page_size": 10}, headers=auth_headers).json()

    # 실데이터 모드에서 DB 미연결이면 지어낸 총계·행을 내지 않는다(빈 결과 + 사유) — 계약 키는 그대로다.
    assert body["pagination"] == {"page": 1, "page_size": 10, "total": None, "total_pages": None}
    assert body["source_kind"] == "in-memory"
    assert body["source_kind_reason"] == "DSN 없음"
    assert body["rows"] == []
    assert body["result_rows"] == 0


def test_adapter_exception_degrades_instead_of_500(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DB가 내려가 있어도 대시보드는 계속 동작해야 한다."""

    class _Boom:
        name = "PostgreSQLAdapter"

        def execute(self, request: Any) -> ExecutionResult:
            raise ConnectionError("connection refused")

    _inject(monkeypatch, _Boom())

    response = client.get(_URL, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["source_kind"] == "in-memory"
    assert body["pagination"]["total"] is None and body["rows"] == []


def test_write_executes_with_bound_values_and_returns_rowcount(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """본문 값이 있는 쓰기는 어댑터로 관통되고 드라이버 rowcount 가 응답에 실린다."""
    adapter = _StubAdapter(ExecutionResult(affected_rows=3, executed=True))
    _inject(monkeypatch, adapter)
    values = {"reg_date": "20260101", "region_code": "45190", "in_flow_count": 7}

    body = client.post(_URL, json={"data": values}, headers=auth_headers).json()

    assert body["affected_rows"] == 3
    assert body["source_kind"] == "database"
    # 본문 값이 그대로(파라미터 바인딩 대상으로) 어댑터에 전달된다.
    assert adapter.requests[0].method == "POST"
    assert adapter.requests[0].values == values


def test_write_degrades_to_stub_when_not_executed(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """본문 값 없는 POST 는 실행되지 않는다 — 실데이터 모드는 반영 건수를 지어내지 않는다(0)."""
    adapter = _StubAdapter(ExecutionResult.not_executed("실행 조건 미충족"))
    _inject(monkeypatch, adapter)

    body = client.post(_URL, json={"data": {}}, headers=auth_headers).json()

    assert body["affected_rows"] == 0
    assert body["source_kind"] == "in-memory"
    assert adapter.requests[0].method == "POST"


def test_delete_returns_real_rowcount_when_executed(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject(monkeypatch, _StubAdapter(ExecutionResult(affected_rows=2, executed=True)))

    body = client.delete(_URL, params={"filter": "in_flow_count > 100"}, headers=auth_headers).json()

    assert body["affected_rows"] == 2
    assert body["source_kind"] == "database"


def test_write_with_unknown_column_is_rejected_400(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """스키마에 없는 컬럼은 어댑터 도달 전에 400 — 읽기 주입 차단과 같은 경계."""
    adapter = _StubAdapter(ExecutionResult(affected_rows=1, executed=True))
    _inject(monkeypatch, adapter)

    response = client.post(_URL, json={"data": {"no_such_col": 1}}, headers=auth_headers)

    assert response.status_code == 400
    assert adapter.requests == []


def test_write_with_non_scalar_value_is_rejected_400(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """dict 값은 Mongo `$set` 연산자 주입 경로가 되므로 스칼라만 허용한다."""
    adapter = _StubAdapter(ExecutionResult(affected_rows=1, executed=True))
    _inject(monkeypatch, adapter)

    response = client.post(
        _URL, json={"data": {"in_flow_count": {"$gt": 0}}}, headers=auth_headers
    )

    assert response.status_code == 400
    assert adapter.requests == []


def test_mongo_source_rows_pass_through(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    token = client.post(f"/api/v3/dataops/token/{_MONGO_SOURCE}").json()["access_token"]
    docs = [{"seq": 25032, "region_code": "45190", "category": "교통", "sentiment_score": -0.4}]
    _inject(monkeypatch, _StubAdapter(ExecutionResult(rows=docs, total=88, executed=True)))

    body = client.get(
        f"/api/v3/dataops/{_MONGO_SOURCE}", headers={"Authorization": f"Bearer {token}"}
    ).json()

    assert body["query_language"] == "MQL"
    assert body["rows"] == docs
    assert body["pagination"]["total"] == 88
