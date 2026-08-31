"""실 저장소 백엔드 — psycopg(PostgreSQL·PostGIS·TimescaleDB) / pymongo(MongoDB).

드라이버는 **선택 의존성**(`pip install -e .[db]`)이며 지연 import 한다. 미설치거나 DSN이
비어 있으면 `adapters.get_adapter` 가 `InMemoryAdapter` 로 degrade하므로 이 모듈이 없어도
서비스는 그대로 동작한다(`explain.py` 의 SHAP fallback 과 같은 방침).

읽기(GET)만 실제 실행한다. 쓰기 SQL은 `query_builder` 가 `?` 플레이스홀더로 생성하고
바인딩 값이 요청 본문에 아직 연결되지 않았으며, 데모 화면의 클릭으로 실제 행이 지워지면
안 되기 때문이다. 쓰기 경로는 결정적 스텁을 유지한다(후속 과제).
"""

from __future__ import annotations

from typing import Any, Sequence

from src.core.logger import get_logger
from src.dataops.query_builder import build_count_sql, build_mongo_query
from src.dataops.results import ExecutionRequest, ExecutionResult

_logger = get_logger("xops.dataops.backend")


class DriverUnavailableError(RuntimeError):
    """드라이버 미설치 — 호출자는 In-Memory 로 degrade한다."""


def _rows_to_dicts(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    """DB 커서 결과를 JSON 직렬화 가능한 dict 목록으로."""
    return [{name: _jsonable(value) for name, value in zip(columns, row)} for row in rows]


def _jsonable(value: Any) -> Any:
    """날짜·Decimal·기하 등 드라이버 고유 타입을 문자열로 낮춘다(응답 직렬화 보장)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class SqlAdapter:
    """PostgreSQL 계열(PostGIS·TimescaleDB 포함) 어댑터 — 생성된 SQL을 그대로 실행한다.

    화면에 표시되는 `generated_query` 와 실제 실행 SQL이 동일하다는 점이 이 설계의 이점이다.
    사용자 입력은 이미 `safety.assert_safe_filter/sort/sql` 로 검증된 뒤에만 도달한다.
    """

    def __init__(self, name: str, dsn: str, *, timeout: float, max_rows: int) -> None:
        self.name = name
        self._dsn = dsn
        self._timeout = timeout
        self._max_rows = max_rows

    def _connect(self) -> Any:
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DriverUnavailableError("psycopg 미설치 — pip install -e .[db]") from exc
        # ponytail: 요청당 단발 연결. 동시성이 문제되면 psycopg_pool 로 교체(인터페이스 불변).
        return psycopg.connect(self._dsn, connect_timeout=int(self._timeout))

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """GET 만 실행. 그 밖의 메서드는 미실행으로 표시해 스텁 응답을 유지한다."""
        if request.method != "GET":
            return ExecutionResult.not_executed("쓰기 경로는 실 저장소 실행 대상이 아닙니다(후속).")

        table = request.schema["object"]
        count_sql = build_count_sql(
            table=table, range_=request.range_, filter_expr=request.filter_expr
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(request.sql)
                columns = [desc[0] for desc in (cur.description or [])]
                fetched = cur.fetchmany(self._max_rows)
                rows = _rows_to_dicts(columns, fetched)
            with conn.cursor() as cur:
                cur.execute(count_sql)
                first = cur.fetchone()
                total = int(first[0]) if first else len(rows)
        _logger.info(
            f"sql executed adapter={self.name} table={table} rows={len(rows)} total={total}"
        )
        return ExecutionResult(rows=rows, total=total, affected_rows=None, executed=True)


class MongoAdapter:
    """MongoDB 어댑터 — 표시용 MQL 문자열을 파싱하지 않고 같은 입력에서 필터 dict를 만든다."""

    def __init__(self, name: str, uri: str, *, timeout: float, max_rows: int) -> None:
        self.name = name
        self._uri = uri
        self._timeout = timeout
        self._max_rows = max_rows

    def _client(self) -> Any:
        try:
            from pymongo import MongoClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DriverUnavailableError("pymongo 미설치 — pip install -e .[db]") from exc
        return MongoClient(self._uri, serverSelectionTimeoutMS=int(self._timeout * 1000))

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """GET 만 실행. 컬렉션은 스키마의 object, DB는 URI 의 기본 DB를 쓴다."""
        if request.method != "GET":
            return ExecutionResult.not_executed("쓰기 경로는 실 저장소 실행 대상이 아닙니다(후속).")

        collection_name = request.schema["object"]
        query = build_mongo_query(range_=request.range_, filter_expr=request.filter_expr)
        client = self._client()
        try:
            collection = client.get_default_database()[collection_name]
            cursor = collection.find(query, {"_id": False})
            if request.sort:
                cursor = cursor.sort(request.sort, -1)
            limit = min(request.page_size, self._max_rows)
            documents = list(cursor.skip((request.page - 1) * request.page_size).limit(limit))
            total = int(collection.count_documents(query))
        finally:
            client.close()
        rows = [{key: _jsonable(value) for key, value in doc.items()} for doc in documents]
        _logger.info(
            f"mql executed adapter={self.name} collection={collection_name} "
            f"rows={len(rows)} total={total}"
        )
        return ExecutionResult(rows=rows, total=total, affected_rows=None, executed=True)
