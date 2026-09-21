"""읽기 전용 실데이터 PG 리더 — public.ext_* 중 `ALLOWED_TABLES`에 열거된 테이블만 조회한다.

`db_max_rows`(dataops의 표시용 상한)를 적용하지 않고 `fetchmany`로 전량 스트리밍한다
(계약 R1). 쓰기 경로는 없다 — 이 모듈은 SELECT만 만든다. 테이블·컬럼명은 파라미터
바인딩이 불가능한 SQL 식별자이므로 `dataops.safety.assert_safe_identifier`로 검증하고,
테이블은 추가로 고정 allowlist에 있는지 확인한다(카탈로그 기반이 아니라 상수 고정).
"""

from __future__ import annotations

from typing import Any, Sequence

from src.core.settings import get_settings
from src.dataops.safety import assert_safe_identifier

# 고정 allowlist — 적재된 ext_* 중에서도 여기 열거된 것만 조회할 수 있다(ext_kt_nowon_*은 아직 대상이 아니다).
# 와일드카드·접두어 매칭으로 바꾸지 않는다 — 임의 테이블 접근을 막는 안전장치라 열거 형태를 유지한다.
ALLOWED_TABLES = {
    # 남원
    "ext_kt_namwon_monthly_dong_visitors",
    "ext_bccard_dong_industry_sales",
    "ext_kt_namwon_visitors_by_sex_age",
    # 강원 18시군구 관광·소비
    "ext_gwto_s1_1_nonlocal_monthly",
    "ext_gwto_s1_2_total_tourists_monthly",
    "ext_gwto_s1_3_tourists_age",
    "ext_gwto_s1_4_tourists_sex",
    "ext_gwto_s1_5_foreigners",
    "ext_gwto_datalab_foreigners",
    "ext_gwto_s1_6_lodging",
    "ext_gwto_s1_7_residence_region",
    "ext_gwto_s1_8_residence_city",
    "ext_gwto_s2_1_consumption_monthly",
    "ext_gwto_s2_2_consumption_age",
    "ext_gwto_s2_3_consumption_sex",
    "ext_gwto_s2_4_consumption_industry",
    "ext_gwto_s2_5_consumption_hour",
    "ext_gwto_s2_6_tourist_spending_power",
    "ext_gwto_s3_1_city_tourists",
    "ext_gwto_s3_2_city_age",
    "ext_gwto_s3_3_city_sex",
    "ext_gwto_s3_4_city_hour",
    "ext_gwto_s3_5_city_lodging",
    "ext_gwto_s3_6_city_region",
    "ext_gwto_s3_7_city_consumption",
    "ext_gwto_s4_1_navi_top500_prev_month",
    "ext_gwto_s4_2_navi_top500_prev_year",
    "ext_gwto_s4_3_navi_summary_500",
    "ext_gwto_daily_trend",
}

_FETCH_BATCH = 1000


class RealdataUnavailable(RuntimeError):
    """DSN 미설정·드라이버 미설치·허용되지 않은 테이블 — 시드 폴백 없이 호출자가 status=error로 노출."""


def _assert_allowed_table(table: str) -> None:
    if table not in ALLOWED_TABLES:
        raise RealdataUnavailable(f"허용되지 않는 테이블입니다: {table!r}")


def _connect() -> Any:
    settings = get_settings()
    if not settings.pg_dsn:
        raise RealdataUnavailable("XOPS_PG_DSN이 설정되지 않았습니다.")
    try:
        import psycopg  # type: ignore[import-not-found]  # ponytail: 지연 import, 선택 의존성
    except ImportError as exc:
        raise RealdataUnavailable("psycopg 미설치 — pip install psycopg[binary]") from exc
    return psycopg.connect(settings.pg_dsn, connect_timeout=int(settings.db_timeout_seconds))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _rows_to_dicts(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    return [{name: _jsonable(value) for name, value in zip(columns, row)} for row in rows]


def _fetch_streaming(sql: str, params: Sequence[Any], columns: Sequence[str]) -> list[dict[str, Any]]:
    # ponytail: 요청당 단발 연결(dataops.backends.SqlAdapter와 같은 방침).
    # 동시성이 문제되면 psycopg_pool로 교체(인터페이스 불변).
    rows: list[dict[str, Any]] = []
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            while True:
                fetched = cur.fetchmany(_FETCH_BATCH)
                if not fetched:
                    break
                rows.extend(_rows_to_dicts(columns, fetched))
    return rows


def fetch_all(
    table: str,
    columns: Sequence[str],
    *,
    where: dict[str, Any] | None = None,
    order_by: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """`table`에서 `columns`를 전량 조회한다. `where`는 등호 조건만(AND 결합)."""
    _assert_allowed_table(table)
    for column in columns:
        assert_safe_identifier(column, kind="컬럼명")
    for column in order_by:
        assert_safe_identifier(column, kind="정렬 컬럼명")

    sql = f"SELECT {', '.join(columns)} FROM {table}"
    params: list[Any] = []
    if where:
        conditions = []
        for key, value in where.items():
            assert_safe_identifier(key, kind="where 컬럼명")
            conditions.append(f"{key} = %s")
            params.append(value)
        sql += " WHERE " + " AND ".join(conditions)
    if order_by:
        sql += " ORDER BY " + ", ".join(order_by)

    return _fetch_streaming(sql, params, columns)


def fetch_count(table: str) -> int:
    """`SELECT COUNT(*)` — health 체크 등 존재 확인에 쓴다."""
    _assert_allowed_table(table)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
            return int(row[0]) if row else 0


def fetch_aggregate(
    table: str,
    group_by: Sequence[str],
    sums: Sequence[str],
) -> list[dict[str, Any]]:
    """`group_by` 단위로 `sums` 컬럼을 SUM 집계한다(R1-3 소비 합산 등에 사용)."""
    _assert_allowed_table(table)
    for column in group_by:
        assert_safe_identifier(column, kind="group_by 컬럼명")
    for column in sums:
        assert_safe_identifier(column, kind="집계 컬럼명")

    group_cols = ", ".join(group_by)
    sum_exprs = ", ".join(f"SUM({col}) AS {col}" for col in sums)
    sql = f"SELECT {group_cols}, {sum_exprs} FROM {table} GROUP BY {group_cols} ORDER BY {group_cols}"

    columns = list(group_by) + list(sums)
    return _fetch_streaming(sql, [], columns)
