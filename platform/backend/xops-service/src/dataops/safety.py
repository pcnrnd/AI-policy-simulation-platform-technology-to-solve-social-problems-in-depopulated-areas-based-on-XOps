"""SQL/쿼리 인젝션 가드 — 사용자 입력과 **카탈로그 메타데이터**를 안전 문법으로 제한.

카탈로그는 신뢰 대상이 아니다: `POST /dataops/catalog` 로 누구나 소스를 등록할 수 있고
object·컬럼명·range 가 생성 SQL에 그대로 조립되므로 UNION·불리언 주입 경로가 된다.
그래서 3중으로 막는다.

1. 등록 경계 — `schemas/dataops.py` 의 필드 패턴 (422)
2. 실행 경계 — `assert_safe_schema` (400). 이번 검증 도입 **이전에 저장된 행**과 시드까지 덮는다
3. 최종 방어 — `assert_safe_sql` (생성 SQL의 주석·스택 쿼리)

사용자 입력(filter·sort)은 종전대로 `assert_safe_filter/sort` 가 담당한다.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.exceptions import UnsafeQueryError

_FILTER_RE = re.compile(r"^(\w+)\s*(>=|<=|!=|=|>|<)\s*('[^';]*'|\"[^\";]*\"|-?\d+(?:\.\d+)?|\w+)$")
_IDENT_RE = re.compile(r"^\w+$")
_SQL_BLOCKLIST = ("--", "/*", "*/", "xp_", "\x00")

# SQL 식별자(테이블·컬럼) — 숫자로 시작하지 않는 영문/숫자/밑줄만. 인용 식별자는 허용하지 않는다.
SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Postgres 기본 식별자 상한과 같게 둔다.
MAX_IDENTIFIER_LENGTH = 63
# range 값 — 리터럴로 들어가므로 따옴표·세미콜론·괄호를 원천 배제한다(시드는 하이픈을 쓴다: NW-SF-001).
SQL_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:\- ]*$")
MAX_VALUE_LENGTH = 128
# 쓰기 본문 값 — 파라미터 바인딩으로 전달되므로 주입은 성립하지 않지만,
# 스칼라 강제(Mongo 연산자 dict 차단)와 상한·NUL 배제는 여기서 건다.
MAX_WRITE_VALUE_LENGTH = 2000


def assert_safe_filter(filter_expr: str | None, allowed_columns: set[str] | None = None) -> None:
    """사용자 filter는 `col op value` 단일 조건만 허용. 스택 쿼리/주석/함수 호출 차단.

    allowed_columns가 주어지면 filter의 대상 컬럼이 스키마에 존재하는지도 검증한다
    (소스 전환 시 이전 컬럼명 잔존으로 무효 쿼리가 생성되는 문제 예방).
    """
    if not filter_expr:
        return
    match = _FILTER_RE.match(filter_expr.strip())
    if not match:
        raise UnsafeQueryError(f"허용되지 않는 filter 식입니다: {filter_expr!r}")
    if allowed_columns is not None and match.group(1) not in allowed_columns:
        raise UnsafeQueryError(f"스키마에 없는 filter 컬럼입니다: {match.group(1)!r}")


def assert_safe_sort(sort: str | None, allowed_columns: set[str]) -> None:
    """정렬 컬럼은 스키마에 존재하는 식별자만 허용."""
    if not sort:
        return
    if not _IDENT_RE.match(sort) or sort not in allowed_columns:
        raise UnsafeQueryError(f"허용되지 않는 정렬 컬럼입니다: {sort!r}")


def assert_safe_identifier(name: Any, *, kind: str) -> None:
    """테이블·컬럼명 allowlist 검증 — 위반 시 UnsafeQueryError(400)."""
    if not isinstance(name, str) or not name:
        raise UnsafeQueryError(f"{kind}이 비어 있거나 문자열이 아닙니다: {name!r}")
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise UnsafeQueryError(f"{kind}이 너무 깁니다({len(name)} > {MAX_IDENTIFIER_LENGTH}): {name!r}")
    if not SQL_IDENTIFIER_RE.match(name):
        raise UnsafeQueryError(f"허용되지 않는 {kind}입니다(영문·숫자·밑줄만, 숫자 시작 불가): {name!r}")


def assert_safe_value(value: Any, *, kind: str) -> None:
    """range 경계값 검증 — 숫자이거나, 따옴표·세미콜론·괄호가 없는 짧은 문자열만."""
    if isinstance(value, bool) or value is None:
        raise UnsafeQueryError(f"{kind}으로 쓸 수 없는 값입니다: {value!r}")
    if isinstance(value, (int, float)):
        return
    if not isinstance(value, str):
        raise UnsafeQueryError(f"{kind}은 문자열 또는 숫자여야 합니다: {type(value).__name__}")
    if len(value) > MAX_VALUE_LENGTH:
        raise UnsafeQueryError(f"{kind}이 너무 깁니다({len(value)} > {MAX_VALUE_LENGTH})")
    if not SQL_VALUE_RE.match(value):
        raise UnsafeQueryError(f"허용되지 않는 {kind}입니다(따옴표·세미콜론·괄호 불가): {value!r}")


def assert_safe_schema(schema: dict[str, Any]) -> None:
    """실행 직전 카탈로그 스키마 검증 — 생성 SQL에 조립되는 모든 식별자·리터럴을 훑는다.

    등록 시점 검증을 통과하지 못한 과거 행(SQLite 영속화)과 시드도 여기서 걸러진다.
    """
    assert_safe_identifier(schema.get("object"), kind="object명")

    columns = schema.get("columns") or []
    if not isinstance(columns, list) or not columns:
        raise UnsafeQueryError("스키마에 컬럼 정의가 없습니다.")
    for column in columns:
        if not isinstance(column, dict):
            raise UnsafeQueryError(f"컬럼 정의가 객체가 아닙니다: {column!r}")
        assert_safe_identifier(column.get("name"), kind="컬럼명")

    range_ = schema.get("range")
    if range_ is None:
        return
    if not isinstance(range_, dict):
        raise UnsafeQueryError(f"range 가 객체가 아닙니다: {range_!r}")
    assert_safe_identifier(range_.get("column"), kind="range 컬럼명")
    assert_safe_value(range_.get("from"), kind="range 시작값")
    assert_safe_value(range_.get("to"), kind="range 종료값")


def assert_safe_write_values(values: dict[str, Any] | None, allowed_columns: set[str]) -> None:
    """쓰기 본문(data) 검증 — 컬럼명은 스키마 대조, 값은 스칼라만 허용.

    값은 SQL에 조립되지 않고 파라미터로 바인딩되지만(주입 불가), dict/list 값이
    Mongo `$set` 에 그대로 들어가면 연산자 주입이 되므로 스칼라를 강제한다.
    """
    if not values:
        return
    for key, value in values.items():
        if key not in allowed_columns:
            raise UnsafeQueryError(f"스키마에 없는 쓰기 컬럼입니다: {key!r}")
        if value is None or isinstance(value, (bool, int, float)):
            continue
        if not isinstance(value, str):
            raise UnsafeQueryError(f"쓰기 값은 스칼라(문자열·숫자·불리언·null)만 허용합니다: {key!r}")
        if len(value) > MAX_WRITE_VALUE_LENGTH:
            raise UnsafeQueryError(f"쓰기 값이 너무 깁니다({len(value)} > {MAX_WRITE_VALUE_LENGTH}): {key!r}")
        if "\x00" in value:
            raise UnsafeQueryError(f"쓰기 값에 NUL 문자가 포함되었습니다: {key!r}")


def assert_safe_sql(sql: str) -> None:
    """생성 SQL 최종 방어 — 주석 마커·스택 쿼리(비종단 세미콜론) 차단."""
    lowered = sql.lower()
    for token in _SQL_BLOCKLIST:
        if token in lowered:
            raise UnsafeQueryError(f"위험한 토큰이 쿼리에 포함되었습니다: {token!r}")
    # 세미콜론은 문장 종단 1개만 허용
    if sql.rstrip().rstrip(";").count(";") > 0:
        raise UnsafeQueryError("복수 문장(stacked query)은 허용되지 않습니다.")
