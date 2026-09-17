"""카탈로그 실적재 행수(live_rows) 조회 단위 테스트.

0(등록됐지만 미적재)과 None(확인 불가)을 섞지 않는 것이 이 모듈의 핵심 계약이다.
"""

from __future__ import annotations

import pytest

from src.dataops import liveness

_PG_SOURCE = {"id": "ds_x", "source": "RDB · PostgreSQL", "object": "tb_x"}


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    liveness.reset_cache()


def test_missing_dsn_reports_unknown_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """DSN이 없으면 '미적재(0)'가 아니라 '확인 불가(None)'다."""
    monkeypatch.setattr(liveness, "dsn_for", lambda schema: "")

    assert liveness.live_rows_for(_PG_SOURCE) is None


def test_driver_failure_reports_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """드라이버 미설치·연결 실패도 None으로 낮춘다(카탈로그 조회가 500이 되면 안 된다)."""
    monkeypatch.setattr(liveness, "dsn_for", lambda schema: "postgresql://nope/nope")

    def _boom(dsn: str, table: str) -> int:
        raise RuntimeError("psycopg 미설치")

    monkeypatch.setattr(liveness, "_count_sql", _boom)

    assert liveness.live_rows_for(_PG_SOURCE) is None


def test_unsafe_object_name_is_not_queried(monkeypatch: pytest.MonkeyPatch) -> None:
    """식별자 패턴을 벗어난 객체명은 조회 자체를 하지 않는다."""
    monkeypatch.setattr(liveness, "dsn_for", lambda schema: "postgresql://x/y")
    called: list[str] = []
    monkeypatch.setattr(liveness, "_count_sql", lambda dsn, table: called.append(table) or 1)

    result = liveness.live_rows_for({"id": "ds_bad", "source": "RDB · PostgreSQL", "object": "tb_x; drop table y"})

    assert result is None
    assert called == []


def test_counts_are_cached_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """같은 소스를 반복 조회해도 TTL 안에서는 저장소를 한 번만 친다."""
    monkeypatch.setattr(liveness, "dsn_for", lambda schema: "postgresql://x/y")
    calls: list[str] = []

    def _count(dsn: str, table: str) -> int:
        calls.append(table)
        return 42

    monkeypatch.setattr(liveness, "_count_sql", _count)

    assert liveness.live_rows_for(_PG_SOURCE) == 42
    assert liveness.live_rows_for(_PG_SOURCE) == 42
    assert len(calls) == 1


def test_cache_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """TTL이 지나면 다시 센다."""
    monkeypatch.setattr(liveness, "dsn_for", lambda schema: "postgresql://x/y")
    monkeypatch.setattr(liveness, "_CACHE_TTL_SECONDS", 0.0)
    calls: list[str] = []
    monkeypatch.setattr(liveness, "_count_sql", lambda dsn, table: calls.append(table) or 7)

    liveness.live_rows_for(_PG_SOURCE)
    liveness.live_rows_for(_PG_SOURCE)

    assert len(calls) == 2


def test_annotate_does_not_mutate_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """annotate는 새 dict를 만든다 — 카탈로그 시드가 오염되면 안 된다."""
    monkeypatch.setattr(liveness, "dsn_for", lambda schema: "postgresql://x/y")
    monkeypatch.setattr(liveness, "_count_sql", lambda dsn, table: 3)
    original = dict(_PG_SOURCE)

    annotated = liveness.annotate([original])

    assert annotated[0]["live_rows"] == 3
    assert "live_rows" not in original


def test_zero_rows_is_distinct_from_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """테이블이 없거나 비어 있으면 0 — None과 구분된다."""
    monkeypatch.setattr(liveness, "dsn_for", lambda schema: "postgresql://x/y")
    monkeypatch.setattr(liveness, "_count_sql", lambda dsn, table: 0)

    assert liveness.live_rows_for(_PG_SOURCE) == 0
