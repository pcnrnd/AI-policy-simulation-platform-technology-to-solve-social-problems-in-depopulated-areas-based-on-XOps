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
