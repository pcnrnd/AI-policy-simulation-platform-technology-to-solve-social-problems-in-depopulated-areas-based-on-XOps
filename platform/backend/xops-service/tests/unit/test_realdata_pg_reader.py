"""읽기 전용 PG 리더 단위 테스트 — fake 커넥션으로 allowlist 거부·바인딩·전량 fetch를 검증.

실 PG는 물지 않는다(기존 관행) — 실측은 워커 보고의 `python -c` 스모크로 별도 확인.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from src.core.exceptions import UnsafeQueryError
from src.core.settings import Settings
from src.realdata import pg_reader
from src.realdata.pg_reader import RealdataUnavailable


class _FakeCursor:
    """`fetchmany` 가 배치를 다 소비하면 빈 리스트를 돌려주는 스텁 커서."""

    def __init__(self, batches: list[list[tuple[Any, ...]]]) -> None:
        self._batches = list(batches)
        self.executed: list[tuple[str, list[Any]]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        self.executed.append((sql, list(params or [])))

    def fetchmany(self, _n: int) -> list[tuple[Any, ...]]:
        return self._batches.pop(0) if self._batches else []

    def fetchone(self) -> tuple[Any, ...] | None:
        batch = self._batches.pop(0) if self._batches else []
        return batch[0] if batch else None


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return self._cursor


def test_fetch_all_rejects_table_outside_allowlist() -> None:
    # 적재는 되어 있지만 등록하지 않은 노원 축제 테이블 — allowlist가 적재 목록과 별개임을 고정한다.
    with pytest.raises(RealdataUnavailable):
        pg_reader.fetch_all("ext_kt_nowon_daily_visitors", ["base_ym"])


def test_allowlist_covers_namwon_and_gwto_only() -> None:
    assert len(pg_reader.ALLOWED_TABLES) == 29
    assert len([t for t in pg_reader.ALLOWED_TABLES if t.startswith("ext_gwto_")]) == 26
    assert not [t for t in pg_reader.ALLOWED_TABLES if t.startswith("ext_kt_nowon_")]


def test_fetch_all_rejects_unsafe_column_identifier() -> None:
    with pytest.raises(UnsafeQueryError):
        pg_reader.fetch_all(
            "ext_kt_namwon_monthly_dong_visitors", ["base_ym; DROP TABLE x"]
        )


def test_fetch_all_rejects_unsafe_where_column() -> None:
    with pytest.raises(UnsafeQueryError):
        pg_reader.fetch_all(
            "ext_kt_namwon_monthly_dong_visitors",
            ["base_ym"],
            where={"dong_code; --": "x"},
        )


def test_connect_raises_when_dsn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pg_reader, "get_settings", lambda: Settings(pg_dsn=""))
    with pytest.raises(RealdataUnavailable):
        pg_reader.fetch_all("ext_kt_namwon_monthly_dong_visitors", ["base_ym"])


def test_connect_applies_statement_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """전량 조회에 LIMIT이 없으므로 실행 시간 상한은 연결 옵션으로만 걸린다."""
    captured: dict[str, Any] = {}

    class _FakePsycopg:
        @staticmethod
        def connect(dsn: str, **kwargs: Any) -> str:
            captured.update({"dsn": dsn, **kwargs})
            return "conn"

    monkeypatch.setattr(pg_reader, "get_settings", lambda: Settings(pg_dsn="postgresql://x/y"))
    monkeypatch.setitem(sys.modules, "psycopg", _FakePsycopg)

    assert pg_reader._connect() == "conn"
    assert captured["options"] == f"-c statement_timeout={pg_reader._STATEMENT_TIMEOUT_MS}"


def test_fetch_all_streams_all_batches_and_binds_where(monkeypatch: pytest.MonkeyPatch) -> None:
    batch1 = [(202301, "45190250", 10)]
    batch2 = [(202302, "45190250", 12)]
    cursor = _FakeCursor([batch1, batch2])
    monkeypatch.setattr(pg_reader, "_connect", lambda: _FakeConn(cursor))

    rows = pg_reader.fetch_all(
        "ext_kt_namwon_monthly_dong_visitors",
        ["base_ym", "dong_code", "nonlocal_visitors"],
        where={"dong_code": "45190250"},
        order_by=["base_ym"],
    )

    assert rows == [
        {"base_ym": 202301, "dong_code": "45190250", "nonlocal_visitors": 10},
        {"base_ym": 202302, "dong_code": "45190250", "nonlocal_visitors": 12},
    ]
    sql, params = cursor.executed[0]
    assert "WHERE dong_code = %s" in sql
    assert "ORDER BY base_ym" in sql
    assert params == ["45190250"]


def test_fetch_aggregate_groups_and_sums(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = [(202301, "45190250", 1_000_000)]
    cursor = _FakeCursor([batch])
    monkeypatch.setattr(pg_reader, "_connect", lambda: _FakeConn(cursor))

    rows = pg_reader.fetch_aggregate(
        "ext_bccard_dong_industry_sales",
        group_by=["base_ym", "dong_name"],
        sums=["sales_est_krw"],
    )

    assert rows == [{"base_ym": 202301, "dong_name": "45190250", "sales_est_krw": 1_000_000}]
    sql, _ = cursor.executed[0]
    assert "SUM(sales_est_krw)" in sql
    assert "GROUP BY base_ym, dong_name" in sql


def test_fetch_table_existence_binds_names_as_values(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = [("ext_bccard_dong_industry_sales", True), ("ext_kt_namwon_monthly_dong_visitors", False)]
    cursor = _FakeCursor([batch])
    monkeypatch.setattr(pg_reader, "_connect", lambda: _FakeConn(cursor))

    present = pg_reader.fetch_table_existence(
        ["ext_kt_namwon_monthly_dong_visitors", "ext_bccard_dong_industry_sales"]
    )

    assert present == {
        "ext_bccard_dong_industry_sales": True,
        "ext_kt_namwon_monthly_dong_visitors": False,
    }
    sql, params = cursor.executed[0]
    assert "to_regclass" in sql and "COUNT(" not in sql  # 카탈로그 조회 1회, 전량 스캔 없음
    assert params == [["ext_bccard_dong_industry_sales", "ext_kt_namwon_monthly_dong_visitors"]]


def test_fetch_table_existence_rejects_table_outside_allowlist() -> None:
    with pytest.raises(RealdataUnavailable):
        pg_reader.fetch_table_existence(["ext_kt_nowon_daily_visitors"])
