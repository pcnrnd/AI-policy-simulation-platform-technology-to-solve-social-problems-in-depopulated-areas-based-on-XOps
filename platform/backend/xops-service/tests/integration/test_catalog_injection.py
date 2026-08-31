"""카탈로그 메타데이터 주입 방어 테스트.

`POST /dataops/catalog` 은 인증 없이 열려 있고, 등록된 object·컬럼명·range 가 생성 SQL에
그대로 조립된다. 실 저장소가 붙으면 그 SQL이 그대로 실행되므로 등록·실행 양쪽에서 막아야 한다.

각 케이스는 (1) 등록이 거부되는지 (2) 등록 검증을 우회해 저장된 행이라도 실행 단계에서
거부되는지 를 함께 본다. (2)는 이번 검증 도입 이전에 SQLite에 남은 행을 상정한 것이다.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core import db
from src.dataops.catalog import get_catalog

_CATALOG = "/api/v3/dataops/catalog"


def _payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "inj_probe",
        "source": "RDB · PostgreSQL",
        "object": "tb_probe",
        "columns": [{"name": "col_a", "type": "int"}],
    }
    body.update(overrides)
    return body


def _register(client: TestClient, **overrides: Any) -> int:
    return client.post(_CATALOG, json=_payload(**overrides)).status_code


# ── 등록 경계 ──
@pytest.mark.parametrize(
    "object_name",
    [
        "tb_x WHERE 1=1 UNION SELECT usename FROM pg_shadow",  # UNION 유출
        "tb_x; DROP TABLE tb_y",  # 스택 쿼리
        "tb_x--comment",  # 주석
        "tb_x/*c*/",
        '"tb x"',  # 인용 식별자
        "1_starts_with_digit",
        "tb_x OR 1=1",
        "a" * 64,  # 길이 상한(63) 초과
    ],
)
def test_malicious_object_name_rejected(client: TestClient, object_name: str) -> None:
    assert _register(client, object=object_name) == 422


@pytest.mark.parametrize(
    "column_name",
    [
        "a, (SELECT current_setting('is_superuser')) AS leaked",  # 서브쿼리 삽입
        "a FROM pg_shadow--",
        "a) OR (1=1",
        "a;b",
        "a'",
        "*",
        "",
    ],
)
def test_malicious_column_name_rejected(client: TestClient, column_name: str) -> None:
    assert _register(client, columns=[{"name": column_name, "type": "int"}]) == 422


@pytest.mark.parametrize(
    "range_def",
    [
        {"column": "a) OR (1=1", "from": "1", "to": "2"},  # range 컬럼 주입
        {"column": "a--", "from": "1", "to": "2"},
        {"column": "1a", "from": "1", "to": "2"},
    ],
)
def test_malicious_range_column_rejected(client: TestClient, range_def: dict[str, Any]) -> None:
    assert _register(client, range=range_def) == 422


@pytest.mark.parametrize(
    "boundary",
    [
        "x' OR '1'='1",  # 불리언 주입 (따옴표 탈출)
        "x'; DROP TABLE t--",
        "x' UNION SELECT 1--",
        "x/*c*/",
        "x)",
        "a" * 129,  # 길이 상한(128) 초과
    ],
)
def test_malicious_range_boundary_rejected(client: TestClient, boundary: str) -> None:
    """range 경계값은 SQL 리터럴로 들어가므로 따옴표·세미콜론·괄호를 배제한다."""
    status = _register(client, range={"column": "col_a", "from": boundary, "to": "z"})
    assert status == 400  # UnsafeQueryError → 400
    assert _register(client, range={"column": "col_a", "from": "a", "to": boundary}) == 400


def test_legitimate_source_still_registers(client: TestClient) -> None:
    """정상 스키마는 그대로 등록·조회된다 — 검증이 과하게 막지 않는지 확인."""
    body = _payload(
        id="inj_ok",
        object="tb_valid_source",
        columns=[{"name": "col_a", "type": "int"}, {"name": "col_b", "type": "VARCHAR(8)"}],
        range={"column": "col_a", "from": "NW-SF-001", "to": "NW-SF-128"},  # 하이픈 값은 허용
    )
    assert client.post(_CATALOG, json=body).status_code == 201
    token = client.post("/api/v3/dataops/token/inj_ok").json()["access_token"]
    got = client.get("/api/v3/dataops/inj_ok", headers={"Authorization": f"Bearer {token}"})
    assert got.status_code == 200
    assert "FROM tb_valid_source" in got.json()["generated_query"]
    client.request("DELETE", f"{_CATALOG}/inj_ok")


# ── 실행 경계 (등록 검증 우회분) ──
@pytest.mark.parametrize(
    "stored",
    [
        {"object": "tb_x WHERE 1=1 UNION SELECT usename FROM pg_shadow"},
        {"columns": [{"name": "a, (SELECT 1) AS leaked", "type": "int"}]},
        {"range": {"column": "col_a", "from": "x' OR '1'='1", "to": "z"}},
        {"range": {"column": "a) OR (1=1", "from": "1", "to": "2"}},
        {"columns": []},  # 컬럼 없는 스키마
    ],
)
def test_pre_existing_malicious_row_rejected_at_execution(
    client: TestClient, stored: dict[str, Any]
) -> None:
    """검증 도입 전에 SQLite 에 저장된 악성 행은 실행 단계에서 400 으로 막힌다."""
    schema = {
        "id": "inj_legacy",
        "label": "legacy",
        "source": "RDB · PostgreSQL",
        "object": "tb_probe",
        "columns": [{"name": "col_a", "type": "int"}],
        "user_registered": True,
    }
    schema.update(stored)
    # 스키마 검증을 우회해 직접 영속화 (과거 데이터 상정)
    db.add_user_source(schema)
    get_catalog.cache_clear()
    try:
        token = client.post("/api/v3/dataops/token/inj_legacy").json()["access_token"]
        response = client.get("/api/v3/dataops/inj_legacy", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 400
        assert response.json()["error"] == "UnsafeQueryError"
    finally:
        db.delete_user_source("inj_legacy")
        get_catalog.cache_clear()


def test_quote_in_range_value_is_escaped_by_builder() -> None:
    """검증을 통과하지 못하는 값이라도 생성기 단독으로 리터럴을 벗어나지 못한다(방어 이중화)."""
    from src.dataops.query_builder import build_sql

    sql = build_sql(
        method="GET",
        table="tb_probe",
        columns=[{"name": "col_a", "type": "int"}],
        range_={"column": "col_a", "from": "x' OR '1'='1", "to": "z"},
        filter_expr=None,
        sort=None,
        page=1,
        page_size=10,
    )
    # 따옴표가 두 배로 이스케이프돼 조건이 리터럴 안에 갇힌다.
    assert "'x'' OR ''1''=''1'" in sql
    assert "OR '1'='1'" not in sql
