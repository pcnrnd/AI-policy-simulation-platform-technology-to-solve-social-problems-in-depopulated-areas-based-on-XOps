"""safety 단위 테스트 — 인젝션 가드."""

from __future__ import annotations

import pytest

from src.core.exceptions import UnsafeQueryError
from src.dataops.safety import assert_safe_filter, assert_safe_sort, assert_safe_sql


@pytest.mark.parametrize("expr", ["age > 30", "region_code = '11'", "score <= 0.5", None, ""])
def test_safe_filters_pass(expr: str | None) -> None:
    assert_safe_filter(expr)


@pytest.mark.parametrize("expr", ["1=1; DROP TABLE x", "a = 1 OR 1=1", "col = 1 -- comment", "name = 'a'; DELETE"])
def test_unsafe_filters_rejected(expr: str) -> None:
    with pytest.raises(UnsafeQueryError):
        assert_safe_filter(expr)


def test_sort_must_be_known_column() -> None:
    assert_safe_sort("reg_date", {"reg_date", "age"})
    with pytest.raises(UnsafeQueryError):
        assert_safe_sort("reg_date; DROP", {"reg_date"})
    with pytest.raises(UnsafeQueryError):
        assert_safe_sort("unknown_col", {"reg_date"})


def test_assert_safe_sql_blocks_comments_and_stacking() -> None:
    assert_safe_sql("SELECT a FROM t WHERE a = 1;")
    with pytest.raises(UnsafeQueryError):
        assert_safe_sql("SELECT a FROM t; DROP TABLE t;")
    with pytest.raises(UnsafeQueryError):
        assert_safe_sql("SELECT a FROM t -- x")


# ── 쓰기 본문 값 검증 ──
def test_write_values_accept_scalars_of_known_columns() -> None:
    from src.dataops.safety import assert_safe_write_values

    cols = {"reg_date", "in_flow_count", "note"}
    assert_safe_write_values(None, cols)
    assert_safe_write_values({}, cols)
    assert_safe_write_values(
        {"reg_date": "20260101", "in_flow_count": 7, "note": None}, cols
    )


def test_write_values_reject_unknown_column_and_non_scalars() -> None:
    from src.dataops.safety import MAX_WRITE_VALUE_LENGTH, assert_safe_write_values

    cols = {"a"}
    with pytest.raises(UnsafeQueryError, match="스키마에 없는"):
        assert_safe_write_values({"b": 1}, cols)
    with pytest.raises(UnsafeQueryError, match="스칼라"):
        assert_safe_write_values({"a": {"$gt": 0}}, cols)  # Mongo 연산자 주입 차단
    with pytest.raises(UnsafeQueryError, match="스칼라"):
        assert_safe_write_values({"a": [1, 2]}, cols)
    with pytest.raises(UnsafeQueryError, match="깁니다"):
        assert_safe_write_values({"a": "x" * (MAX_WRITE_VALUE_LENGTH + 1)}, cols)
    with pytest.raises(UnsafeQueryError, match="NUL"):
        assert_safe_write_values({"a": "x\x00y"}, cols)


def test_and_combined_filters_pass() -> None:
    """AND 결합 — 2개·상한 5개, 대소문자 무시, 컬럼 allowlist 동시 적용."""
    assert_safe_filter("age > 30 AND region_code = '11'", {"age", "region_code"})
    assert_safe_filter("a = 1 and b = 2 AnD c = 3 AND d = 4 AND e = 5", {"a", "b", "c", "d", "e"})


def test_and_filters_reject_over_limit_and_unknown_column() -> None:
    from src.dataops.safety import MAX_FILTER_CONDITIONS

    over = " AND ".join(f"c{i} = {i}" for i in range(MAX_FILTER_CONDITIONS + 1))
    with pytest.raises(UnsafeQueryError, match="최대"):
        assert_safe_filter(over)
    with pytest.raises(UnsafeQueryError, match="스키마에 없는"):
        assert_safe_filter("age > 30 AND ghost = 1", {"age"})


@pytest.mark.parametrize(
    "expr",
    [
        "age > 30 OR region_code = '11'",       # OR 결합 미지원
        "age > 30 AND (b = 1 OR c = 2)",        # 괄호
        "age > 30 AND 1=1; DROP TABLE x",       # 스택 쿼리
        "age > 30 AND b = 1 -- comment",        # 주석
    ],
)
def test_and_filters_reject_non_and_combinations(expr: str) -> None:
    with pytest.raises(UnsafeQueryError):
        assert_safe_filter(expr)
