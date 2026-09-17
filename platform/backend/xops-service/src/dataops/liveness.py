"""카탈로그 소스의 실적재 행수 조회 — 읽기 전용.

카탈로그는 시드·SQLite 메타데이터라서 "등록되어 있다"와 "실제로 적재되어 있다"가 다르다.
이 모듈은 소스의 `object` 를 그 소스의 저장소에서 세어 그 차이를 메운다.

세 가지 결과를 구분한다.

- `int >= 0` — 저장소에 닿아 실제로 센 행수(테이블·컬렉션이 없으면 0).
- `None` — DSN 부재·드라이버 미설치·연결 실패로 **확인하지 못함**. 0(미적재)과 구분한다.

`adapters.get_adapter` 와 같은 degrade 방침을 따르므로 DB 없이도 카탈로그 응답은 그대로다.
"""

from __future__ import annotations

from time import monotonic
from typing import Any

from src.core.logger import get_logger
from src.core.settings import get_settings
from src.dataops.adapters import dsn_for
from src.dataops.safety import assert_safe_identifier

_logger = get_logger("xops.dataops.liveness")

# ponytail: 프로세스 내 dict + TTL. 워커가 여러 개면 워커마다 따로 데우지만 조회가 읽기 전용이라 무해하다.
_CACHE_TTL_SECONDS = 60.0
_cache: dict[str, tuple[float, int | None]] = {}


def reset_cache() -> None:
    """테스트·설정 변경용 캐시 비우기."""
    _cache.clear()


def live_rows_for(schema: dict[str, Any]) -> int | None:
    """소스의 실적재 행수 — TTL 캐시. 확인 불가면 None."""
    source_id = str(schema.get("id", ""))
    now = monotonic()
    cached = _cache.get(source_id)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    counted = _count(schema)
    _cache[source_id] = (now, counted)
    return counted


def annotate(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """카탈로그 목록에 live_rows 를 덧붙인 새 목록을 만든다(원본 불변)."""
    return [{**source, "live_rows": live_rows_for(source)} for source in sources]


def _count(schema: dict[str, Any]) -> int | None:
    """저장소 유형에 맞춰 센다. 연결·드라이버 문제는 전부 None 으로 낮춘다."""
    dsn = dsn_for(schema)
    obj = schema.get("object")
    if not dsn or not obj:
        return None
    try:
        assert_safe_identifier(obj, kind="데이터 객체명")
    except Exception as exc:  # noqa: BLE001 - 카탈로그 조회가 등록 데이터 때문에 500이 되면 안 된다
        _logger.warning(f"live count skipped — 안전하지 않은 객체명 source={schema.get('id')} error={exc}")
        return None

    try:
        if "MongoDB" in str(schema.get("source", "")):
            return _count_mongo(dsn, str(obj))
        return _count_sql(dsn, str(obj))
    except Exception as exc:  # noqa: BLE001 - 드라이버 미설치·연결 실패·권한 등 전부 '확인 불가'
        _logger.warning(f"live count 실패 source={schema.get('id')} object={obj} error={exc}")
        return None


def _count_sql(dsn: str, table: str) -> int:
    """PostgreSQL 계열 — 테이블이 없으면 0, 있으면 count(*)."""
    import psycopg  # type: ignore[import-not-found]
    from psycopg import sql  # type: ignore[import-not-found]

    timeout = get_settings().db_timeout_seconds
    # 큰 테이블에서 카탈로그 응답이 늘어지지 않게 서버 쪽에서도 끊는다.
    options = f"-c statement_timeout={int(timeout * 1000)}"
    with psycopg.connect(dsn, connect_timeout=int(timeout), options=options) as conn:
        with conn.cursor() as cur:
            cur.execute("select to_regclass(%s) is not null", (table,))
            row = cur.fetchone()
            if not row or not row[0]:
                return 0
            # 식별자는 위에서 패턴 검증했고 여기서 또 드라이버로 인용한다(리터럴 조립 없음).
            cur.execute(sql.SQL("select count(*) from {}").format(sql.Identifier(table)))
            counted = cur.fetchone()
            return int(counted[0]) if counted else 0


def _count_mongo(uri: str, collection: str) -> int:
    """MongoDB — 컬렉션이 없으면 0."""
    from pymongo import MongoClient  # type: ignore[import-not-found]

    timeout = get_settings().db_timeout_seconds
    client = MongoClient(uri, serverSelectionTimeoutMS=int(timeout * 1000))
    try:
        database = client.get_default_database()
        if collection not in database.list_collection_names():
            return 0
        return int(database[collection].count_documents({}))
    finally:
        client.close()
