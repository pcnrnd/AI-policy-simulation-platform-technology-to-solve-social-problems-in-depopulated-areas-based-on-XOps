"""SQL/쿼리 인젝션 가드 — 사용자 입력(filter·sort)을 안전 문법으로 제한.

생성 쿼리의 테이블/컬럼명은 카탈로그(신뢰)에서 오고, 위험 표면은 사용자 filter/sort뿐이다.
따라서 입력을 먼저 검증하고(assert_safe_filter/sort), 생성된 SQL을 최종 방어(assert_safe_sql)한다.
"""

from __future__ import annotations

import re

from src.core.exceptions import UnsafeQueryError

_FILTER_RE = re.compile(r"^(\w+)\s*(>=|<=|!=|=|>|<)\s*('[^';]*'|\"[^\";]*\"|-?\d+(?:\.\d+)?|\w+)$")
_IDENT_RE = re.compile(r"^\w+$")
_SQL_BLOCKLIST = ("--", "/*", "*/", "xp_", "\x00")


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


def assert_safe_sql(sql: str) -> None:
    """생성 SQL 최종 방어 — 주석 마커·스택 쿼리(비종단 세미콜론) 차단."""
    lowered = sql.lower()
    for token in _SQL_BLOCKLIST:
        if token in lowered:
            raise UnsafeQueryError(f"위험한 토큰이 쿼리에 포함되었습니다: {token!r}")
    # 세미콜론은 문장 종단 1개만 허용
    if sql.rstrip().rstrip(";").count(";") > 0:
        raise UnsafeQueryError("복수 문장(stacked query)은 허용되지 않습니다.")
