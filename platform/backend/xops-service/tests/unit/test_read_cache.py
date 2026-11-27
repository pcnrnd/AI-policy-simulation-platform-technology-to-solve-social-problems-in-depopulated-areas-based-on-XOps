"""Data API 조회 결과 캐시 — DataService 실행부에서 어댑터 호출 횟수로 검증한다."""

from __future__ import annotations

from typing import Any

import pytest

from src.dataops import read_cache
from src.dataops import service as service_module
from src.dataops.read_cache import ReadCache
from src.dataops.results import ExecutionRequest, ExecutionResult
from src.dataops.service import DataService

_SCHEMA: dict[str, Any] = {
    "id": "rt_cache_source",
    "source": "RDB · PostgreSQL",
    "object": "cache_table",
    "columns": [{"name": "row_id", "type": "INTEGER"}, {"name": "value", "type": "INTEGER"}],
}
_OTHER = {**_SCHEMA, "id": "rt_cache_other", "object": "cache_other"}
_PAYLOAD = {"sub": "tester", "scope": "data:read data:write"}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _CountingAdapter:
    """호출 횟수를 세고, 실행할 때마다 다른 값을 돌려주는 어댑터."""

    name = "PostgreSQLAdapter"

    def __init__(self, executed: bool = True, fail: bool = False) -> None:
        self.calls: list[ExecutionRequest] = []
        self._executed = executed
        self._fail = fail

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls.append(request)
        if self._fail:
            raise RuntimeError("연결 실패")
        if not self._executed:
            return ExecutionResult.not_executed("미연결")
        if request.method != "GET":
            return ExecutionResult(affected_rows=1, executed=True)
        return ExecutionResult(rows=[{"row_id": 1, "value": len(self.calls)}], total=1, executed=True)


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(read_cache, "store", ReadCache(clock=fake))
    return fake


def _use(monkeypatch: pytest.MonkeyPatch, adapter: _CountingAdapter) -> _CountingAdapter:
    monkeypatch.setattr(service_module, "get_adapter", lambda _schema: adapter)
    return adapter


def _get(schema: dict[str, Any] = _SCHEMA, **kwargs: Any) -> dict[str, Any]:
    return DataService().execute(method="GET", schema=schema, payload=_PAYLOAD, **kwargs)


def test_same_read_hits_adapter_once(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    adapter = _use(monkeypatch, _CountingAdapter())
    first = _get(filter_expr="value > 0", sort="value", page=1, page_size=10)
    second = _get(filter_expr="value > 0", sort="value", page=1, page_size=10)
    assert len(adapter.calls) == 1
    assert first["rows"] == second["rows"]
    assert second["source_kind"] == "database"


def test_different_conditions_are_separate(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    adapter = _use(monkeypatch, _CountingAdapter())
    _get(filter_expr="value > 0")
    _get(filter_expr="value > 1")
    _get(filter_expr="value > 0", sort="value")
    _get(filter_expr="value > 0", page=2)
    _get(filter_expr="value > 0", page_size=5)
    _get(_OTHER, filter_expr="value > 0")
    assert len(adapter.calls) == 6


def test_write_invalidates_source(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    adapter = _use(monkeypatch, _CountingAdapter())
    _get()
    _get(_OTHER)
    DataService().execute(method="POST", schema=_SCHEMA, payload=_PAYLOAD, values={"row_id": 2, "value": 3})
    after = _get()
    _get(_OTHER)  # 다른 소스의 보관 결과는 그대로
    assert [c.method for c in adapter.calls] == ["GET", "GET", "POST", "GET"]
    assert after["rows"][0]["value"] == 4


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_every_write_method_invalidates(monkeypatch: pytest.MonkeyPatch, clock: _Clock, method: str) -> None:
    adapter = _use(monkeypatch, _CountingAdapter())
    _get()
    DataService().execute(method=method, schema=_SCHEMA, payload=_PAYLOAD, filter_expr="row_id = 1",
                          values=None if method == "DELETE" else {"value": 9})
    _get()
    assert [c.method for c in adapter.calls] == ["GET", method, "GET"]


def test_expires_after_ttl(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    adapter = _use(monkeypatch, _CountingAdapter())
    _get()
    clock.now += 59.9
    _get()
    assert len(adapter.calls) == 1
    clock.now += 0.1
    _get()
    assert len(adapter.calls) == 2


@pytest.mark.parametrize("adapter", [_CountingAdapter(executed=False), _CountingAdapter(fail=True)])
def test_failures_are_not_cached(monkeypatch: pytest.MonkeyPatch, clock: _Clock, adapter: _CountingAdapter) -> None:
    adapter.calls.clear()
    _use(monkeypatch, adapter)
    _get()
    _get()
    assert len(adapter.calls) == 2
    assert len(read_cache.store) == 0


def test_entry_limit_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(read_cache, "store", ReadCache(clock=clock, max_entries=3))
    adapter = _use(monkeypatch, _CountingAdapter())
    for page in (1, 2, 3, 4):
        _get(page=page)
    assert len(read_cache.store) == 3
    _get(page=4)  # 남아 있다
    assert len(adapter.calls) == 4
    _get(page=1)  # 가장 오래된 항목은 지워졌다
    assert len(adapter.calls) == 5


def test_caller_cannot_mutate_cached_rows(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    _use(monkeypatch, _CountingAdapter())
    first = _get()
    first["rows"][0]["value"] = "변조"
    first["rows"].append({"row_id": 99})
    second = _get()
    second["rows"][0]["row_id"] = -1
    third = _get()
    assert third["rows"] == [{"row_id": 1, "value": 1}]


def test_cache_does_not_mix_callers(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    """어댑터 결과는 호출자와 무관하다. 보관 결과를 써도 응답의 인증 정보는 각 호출자 것이다."""
    adapter = _use(monkeypatch, _CountingAdapter())
    a = DataService().execute(method="GET", schema=_SCHEMA, payload={"sub": "a", "scope": "data:read"})
    b = DataService().execute(method="GET", schema=_SCHEMA, payload={"sub": "b", "scope": "data:read"})
    assert len(adapter.calls) == 1
    assert a["auth"]["sub"] == "a" and b["auth"]["sub"] == "b"


def test_data_mode_is_part_of_key(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    adapter = _use(monkeypatch, _CountingAdapter())
    monkeypatch.setattr(read_cache, "current_mode", lambda: "real")
    _get()
    monkeypatch.setattr(read_cache, "current_mode", lambda: "mock")
    _get()
    assert len(adapter.calls) == 2


def test_read_started_before_write_is_not_stored(clock: _Clock) -> None:
    """쓰기 전에 시작해 쓰기 뒤에 끝난 조회의 결과는 보관하지 않는다."""
    store = read_cache.store
    request = ExecutionRequest(schema=_SCHEMA, method="GET", sql="SELECT 1")
    key = read_cache.key_for(request)
    generation = store.generation("rt_cache_source")
    store.invalidate("rt_cache_source")
    store.put(key, ExecutionResult(rows=[{"row_id": 1}], total=1, executed=True), generation)
    assert store.get(key) is None


def test_auth_still_checked_before_cache(client: Any, auth_headers: dict[str, str],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """보관 결과가 있어도 토큰 없는 요청은 401이다(권한 확인이 캐시 조회보다 먼저)."""
    from src.core import db

    schema = {**_SCHEMA, "id": "rt_cache_api", "is_seed": False, "mode": "real", "user_registered": True}
    db.delete_user_source(schema["id"])
    db.add_user_source(schema)
    try:
        adapter = _use(monkeypatch, _CountingAdapter())
        url = f"/api/v3/dataops/{schema['id']}"
        assert client.get(url, headers=auth_headers).status_code == 200
        assert client.get(url, headers=auth_headers).status_code == 200
        assert len(adapter.calls) == 1
        assert client.get(url).status_code == 401
    finally:
        db.delete_user_source(schema["id"])
