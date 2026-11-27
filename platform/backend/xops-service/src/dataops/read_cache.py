"""Data API 조회 결과의 프로세스 내 캐시 — 같은 조회를 반복할 때 저장소 왕복을 줄인다.

- 읽기(GET) 중 저장소에서 실제로 실행된 결과(`executed=True`)만 보관한다. 미실행·실패는 보관하지 않는다.
- 키는 결과를 바꾸는 값 전부다: 데이터 모드, 소스 id·저장소 유형·객체명, 생성 쿼리(범위·필터·정렬·
  페이지가 반영된 문자열)와 필터·정렬·페이지·페이지 크기. 어댑터 실행은 호출자(토큰)와 무관하므로
  호출자는 키에 넣지 않는다 — 권한 확인은 라우터 의존성에서 서비스 호출 전에 끝난다.
- 같은 소스에 쓰기(POST·PUT·PATCH·DELETE)를 시도하면 그 소스의 보관 결과를 모두 지운다.
- 보관 시간은 liveness 와 같은 60초, 항목 수 상한을 넘으면 오래 보관한 것부터 지운다.
- uvicorn 워커가 여러 개면 워커마다 따로 데운다. Data API를 거치지 않고 저장소에 쓴 변경은
  보관 시간 동안 늦게 보일 수 있다.
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from time import monotonic
from typing import Callable, Hashable

from src.core.data_mode import current_mode
from src.dataops.results import ExecutionRequest, ExecutionResult

_TTL_SECONDS = 60.0  # liveness._CACHE_TTL_SECONDS 와 같은 값
_MAX_ENTRIES = 256


def key_for(request: ExecutionRequest) -> tuple[Hashable, ...]:
    """조회 결과를 결정하는 값으로 키를 만든다. 첫 원소는 무효화 기준인 소스 id."""
    schema = request.schema
    return (
        str(schema.get("id", "")),
        current_mode(),
        str(schema.get("source", "")),
        str(schema.get("object", "")),
        request.sql,
        request.filter_expr,
        request.sort,
        request.page,
        request.page_size,
    )


class ReadCache:
    """TTL·상한·잠금을 갖춘 조회 결과 보관소. 시간 함수는 테스트에서 주입한다."""

    def __init__(
        self,
        *,
        ttl_seconds: float = _TTL_SECONDS,
        max_entries: int = _MAX_ENTRIES,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[Hashable, ...], tuple[float, ExecutionResult]] = OrderedDict()
        # 소스별 세대 번호 — 쓰기 전에 시작한 조회가 쓰기 뒤에 끝나 옛 결과를 보관하는 것을 막는다.
        self._generations: dict[str, int] = {}

    def generation(self, source_id: str) -> int:
        """조회 시작 시점의 세대 번호. `put` 에 그대로 넘긴다."""
        with self._lock:
            return self._generations.get(source_id, 0)

    def get(self, key: tuple[Hashable, ...]) -> ExecutionResult | None:
        """보관 결과의 복사본. 없거나 보관 시간이 지났으면 None."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, result = entry
            if self._clock() - stored_at >= self._ttl:
                del self._entries[key]
                return None
            return copy.deepcopy(result)  # 호출자가 행을 바꿔도 보관본은 그대로

    def put(self, key: tuple[Hashable, ...], result: ExecutionResult, generation: int) -> None:
        """실행된 결과만 복사해 보관한다. 조회 중 같은 소스에 쓰기가 있었으면 보관하지 않는다."""
        if not result.executed:
            return
        source_id = str(key[0])
        snapshot = copy.deepcopy(result)
        with self._lock:
            if self._generations.get(source_id, 0) != generation:
                return
            self._entries[key] = (self._clock(), snapshot)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def invalidate(self, source_id: str) -> None:
        """소스의 보관 결과를 모두 지우고 세대를 올린다."""
        with self._lock:
            self._generations[source_id] = self._generations.get(source_id, 0) + 1
            for key in [k for k in self._entries if k[0] == source_id]:
                del self._entries[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._generations.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


store = ReadCache()


def reset_cache() -> None:
    """테스트·설정 변경용 캐시 비우기."""
    store.clear()
