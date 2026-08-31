"""실행 어댑터 단위 테스트 — DSN 라우팅·degrade·fake 커넥션 주입 실행 경로.

실 DB 없이 검증한다: Docker 미설치 환경에서도 SQL/MQL 실행 경로가 올바른 쿼리를 보내고
결과를 계약대로 되돌리는지를 가짜 커넥션으로 고정한다.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from src.core.settings import Settings
from src.dataops import adapters as adapters_module
from src.dataops.adapters import InMemoryAdapter, dsn_for, get_adapter
from src.dataops.backends import MongoAdapter, SqlAdapter
from src.dataops.query_builder import build_count_sql, build_mongo_query
from src.dataops.results import ExecutionRequest

_PG = {"id": "ds_01", "source": "RDB · PostgreSQL", "object": "tb_resident_movement"}
_POSTGIS = {"id": "ds_04_spatial_geojson", "source": "공간 DB · PostGIS", "object": "geo_grid_cells"}
_TIMESCALE = {"id": "ds_05_smartfarm", "source": "시계열 DB · TimescaleDB", "object": "ts_smartfarm_yield"}
_MONGO = {"id": "ds_07_civil_complaints", "source": "NoSQL · MongoDB", "object": "col_civil_complaints"}
_RANGE = {"column": "reg_date", "from": "20210101", "to": "20261231"}


def _request(schema: dict[str, Any], method: str = "GET", **kwargs: Any) -> ExecutionRequest:
    defaults = {"sql": f"SELECT * FROM {schema['object']};", "page": 1, "page_size": 10}
    defaults.update(kwargs)
    return ExecutionRequest(schema=schema, method=method, **defaults)  # type: ignore[arg-type]


def _with_settings(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    """adapters 모듈이 보는 설정을 교체 (get_settings 는 lru_cache 라 직접 패치한다)."""
    settings = Settings(**fields)
    monkeypatch.setattr(adapters_module, "get_settings", lambda: settings)


# ── DSN 라우팅 ──
def test_dsn_for_routes_by_source_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_settings(
        monkeypatch,
        pg_dsn="postgresql://pg/db",
        timescale_dsn="postgresql://ts/db",
        mongo_uri="mongodb://mo/db",
    )
    # PostgreSQL 과 PostGIS 는 같은 인스턴스를 쓰므로 DSN 을 공유한다.
    assert dsn_for(_PG) == "postgresql://pg/db"
    assert dsn_for(_POSTGIS) == "postgresql://pg/db"
    assert dsn_for(_TIMESCALE) == "postgresql://ts/db"
    assert dsn_for(_MONGO) == "mongodb://mo/db"


def test_get_adapter_degrades_when_dsn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_settings(monkeypatch)  # 전부 빈 문자열(기본값)
    for schema in (_PG, _POSTGIS, _TIMESCALE, _MONGO):
        adapter = get_adapter(schema)
        assert isinstance(adapter, InMemoryAdapter)
        result = adapter.execute(_request(schema))
        assert result.executed is False
        assert result.rows == [] and result.total is None
        assert "DSN" in result.reason


def test_get_adapter_selects_backend_when_dsn_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_settings(
        monkeypatch,
        pg_dsn="postgresql://pg/db",
        timescale_dsn="postgresql://ts/db",
        mongo_uri="mongodb://mo/db",
    )
    assert isinstance(get_adapter(_PG), SqlAdapter)
    assert isinstance(get_adapter(_POSTGIS), SqlAdapter)
    assert isinstance(get_adapter(_TIMESCALE), SqlAdapter)
    assert isinstance(get_adapter(_MONGO), MongoAdapter)
    # 표시용 이름은 기존 계약 그대로 유지된다.
    assert get_adapter(_POSTGIS).name == "PostGISAdapter (EPSG:4326)"
    assert get_adapter(_MONGO).name == "MongoAdapter (Document Store)"


# ── 쿼리 생성 보조 ──
def test_build_count_sql_shares_where_clause() -> None:
    sql = build_count_sql(table="tb_resident_movement", range_=_RANGE, filter_expr="in_flow_count > 100")
    assert sql.startswith("SELECT COUNT(*) AS total FROM tb_resident_movement")
    assert "reg_date BETWEEN '20210101' AND '20261231'" in sql
    assert "in_flow_count > 100" in sql


def test_build_mongo_query_maps_range_and_operators() -> None:
    assert build_mongo_query(range_=None, filter_expr=None) == {}
    assert build_mongo_query(range_={"column": "seq", "from": 1, "to": 9}, filter_expr=None) == {
        "seq": {"$gte": 1, "$lte": 9}
    }
    assert build_mongo_query(range_=None, filter_expr="sentiment_score >= 0.5") == {
        "sentiment_score": {"$gte": 0.5}
    }
    assert build_mongo_query(range_=None, filter_expr="category = '교통'") == {"category": "교통"}
    # 연산자가 없어 정규식에 걸리지 않는 식은 무시한다.
    assert build_mongo_query(range_=None, filter_expr="no_operator_here") == {}
    # 스택 쿼리 시도는 실 경로에서 safety.assert_safe_filter 가 400 으로 먼저 막는다
    # (test_dataops_api.test_delete_and_injection_guard). 여기까지 온다 해도 값이 dict 의
    # 문자열로 들어가므로 연산자 주입이 성립하지 않는다 — 무해한 리터럴로 남는다.
    assert build_mongo_query(range_=None, filter_expr="1=1; DROP TABLE x") == {"1": "1; DROP TABLE x"}


# ── fake 커넥션 주입: SQL 실행 경로 ──
class _FakeCursor:
    """psycopg 커서 흉내 — execute 로 받은 SQL 을 기록하고 미리 정한 결과를 돌려준다."""

    def __init__(self, owner: _FakeConnection) -> None:
        self._owner = owner
        self._is_count = False

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str) -> None:
        self._owner.executed.append(sql)
        self._is_count = "COUNT(*)" in sql

    @property
    def description(self) -> Sequence[tuple[str, ...]] | None:
        return None if self._is_count else [(name,) for name in self._owner.columns]

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        self._owner.fetch_sizes.append(size)
        return self._owner.rows[:size]

    def fetchone(self) -> tuple[Any, ...] | None:
        return (self._owner.total,) if self._is_count else None


class _FakeConnection:
    def __init__(self, columns: Sequence[str], rows: Sequence[tuple[Any, ...]], total: int) -> None:
        self.columns = list(columns)
        self.rows = list(rows)
        self.total = total
        self.executed: list[str] = []
        self.fetch_sizes: list[int] = []

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _InjectedSqlAdapter(SqlAdapter):
    """_connect 만 가짜로 바꾼 어댑터 — 실행 로직은 실제 코드 그대로 태운다."""

    def __init__(self, connection: _FakeConnection) -> None:
        super().__init__("PostgreSQLAdapter", "postgresql://fake/db", timeout=1.0, max_rows=100)
        self._fake = connection

    def _connect(self) -> Any:
        return self._fake


def test_sql_adapter_returns_rows_and_total_from_connection() -> None:
    conn = _FakeConnection(
        columns=["reg_date", "region_code", "in_flow_count"],
        rows=[("20210101", "45190", 120), ("20210131", "46900", 121)],
        total=1873,
    )
    schema = {**_PG, "range": _RANGE}
    request = _request(schema, sql="SELECT reg_date, region_code, in_flow_count FROM tb_resident_movement;")

    result = _InjectedSqlAdapter(conn).execute(request)

    assert result.executed is True
    assert result.total == 1873
    assert result.rows == [
        {"reg_date": "20210101", "region_code": "45190", "in_flow_count": 120},
        {"reg_date": "20210131", "region_code": "46900", "in_flow_count": 121},
    ]
    # 표시되는 SQL 그대로 실행하고, 총 행수는 같은 WHERE 로 별도 COUNT 한다.
    assert conn.executed[0] == request.sql
    assert conn.executed[1].startswith("SELECT COUNT(*) AS total FROM tb_resident_movement")
    assert "reg_date BETWEEN" in conn.executed[1]
    assert conn.fetch_sizes == [100]  # max_rows 로 잘라 가져온다


def test_sql_adapter_skips_write_methods() -> None:
    conn = _FakeConnection(columns=["a"], rows=[("x",)], total=1)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        result = _InjectedSqlAdapter(conn).execute(_request(_PG, method=method))
        assert result.executed is False
        assert "쓰기" in result.reason
    assert conn.executed == []  # 쓰기 경로에서는 저장소를 건드리지 않는다


def test_sql_adapter_jsonables_non_primitive_values() -> None:
    from datetime import date

    conn = _FakeConnection(columns=["reg_date", "geom"], rows=[(date(2021, 1, 1), object())], total=1)
    result = _InjectedSqlAdapter(conn).execute(_request(_PG))
    assert result.rows[0]["reg_date"] == "2021-01-01"
    assert isinstance(result.rows[0]["geom"], str)  # 기하·미지 타입은 문자열로 낮춘다


# ── fake 클라이언트 주입: MQL 실행 경로 ──
class _FakeMongoCursor:
    def __init__(self, documents: list[dict[str, Any]], calls: dict[str, Any]) -> None:
        self._docs = documents
        self._calls = calls

    def sort(self, key: str, direction: int) -> _FakeMongoCursor:
        self._calls["sort"] = (key, direction)
        return self

    def skip(self, count: int) -> _FakeMongoCursor:
        self._calls["skip"] = count
        return self

    def limit(self, count: int) -> _FakeMongoCursor:
        self._calls["limit"] = count
        return self

    def __iter__(self) -> Any:
        return iter(self._docs)


class _FakeCollection:
    def __init__(self, documents: list[dict[str, Any]], calls: dict[str, Any]) -> None:
        self._docs = documents
        self._calls = calls

    def find(self, query: dict[str, Any], projection: dict[str, Any]) -> _FakeMongoCursor:
        self._calls["find"] = query
        self._calls["projection"] = projection
        return _FakeMongoCursor(self._docs, self._calls)

    def count_documents(self, query: dict[str, Any]) -> int:
        self._calls["count"] = query
        return 4242


class _FakeMongoClient:
    def __init__(self, documents: list[dict[str, Any]], calls: dict[str, Any]) -> None:
        self._collection = _FakeCollection(documents, calls)
        self._calls = calls

    def get_default_database(self) -> dict[str, _FakeCollection]:
        return {"col_civil_complaints": self._collection}

    def close(self) -> None:
        self._calls["closed"] = True


class _InjectedMongoAdapter(MongoAdapter):
    def __init__(self, client: _FakeMongoClient) -> None:
        super().__init__("MongoAdapter (Document Store)", "mongodb://fake/db", timeout=1.0, max_rows=100)
        self._fake = client

    def _client(self) -> Any:
        return self._fake


def test_mongo_adapter_builds_filter_and_returns_documents() -> None:
    calls: dict[str, Any] = {}
    client = _FakeMongoClient([{"seq": 25032, "category": "교통"}], calls)
    schema = {**_MONGO, "range": {"column": "seq", "from": 25032, "to": 53024}}
    request = _request(schema, sort="seq", page=2, page_size=10)

    result = _InjectedMongoAdapter(client).execute(request)

    assert result.executed is True
    assert result.total == 4242
    assert result.rows == [{"seq": 25032, "category": "교통"}]
    # range 가 필터로 주입되고 _id 는 제외된다.
    assert calls["find"] == {"seq": {"$gte": 25032, "$lte": 53024}}
    assert calls["projection"] == {"_id": False}
    assert calls["sort"] == ("seq", -1)
    assert calls["skip"] == 10 and calls["limit"] == 10
    assert calls["closed"] is True  # 커넥션을 반드시 닫는다


def test_mongo_adapter_skips_write_methods() -> None:
    calls: dict[str, Any] = {}
    client = _FakeMongoClient([], calls)
    result = _InjectedMongoAdapter(client).execute(_request(_MONGO, method="DELETE"))
    assert result.executed is False
    assert "find" not in calls


# ── 드라이버 미설치 degrade ──
def _driver_missing(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return True
    return False


@pytest.mark.skipif(not _driver_missing("psycopg"), reason="psycopg 설치됨 — 미설치 경로 검증 불가")
def test_missing_psycopg_raises_driver_unavailable() -> None:
    """드라이버가 없으면 명시적 예외 — 서비스 경계가 이를 잡아 In-Memory 로 degrade한다."""
    from src.dataops.backends import DriverUnavailableError

    adapter = SqlAdapter("PostgreSQLAdapter", "postgresql://x/y", timeout=1.0, max_rows=10)
    with pytest.raises(DriverUnavailableError, match="psycopg"):
        adapter.execute(_request(_PG))


@pytest.mark.skipif(not _driver_missing("pymongo"), reason="pymongo 설치됨 — 미설치 경로 검증 불가")
def test_missing_pymongo_raises_driver_unavailable() -> None:
    from src.dataops.backends import DriverUnavailableError

    adapter = MongoAdapter("MongoAdapter", "mongodb://x/y", timeout=1.0, max_rows=10)
    with pytest.raises(DriverUnavailableError, match="pymongo"):
        adapter.execute(_request(_MONGO))
