"""어댑터 실행 요청·결과 계약 — 서비스와 백엔드가 공유하는 자료형.

`adapters` 와 `backends` 가 서로를 import 하지 않도록 자료형만 여기 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionRequest:
    """저장소 실행에 필요한 최소 입력.

    렌더된 SQL(`sql`)은 화면에 표시되는 것과 동일한 문자열이며 이미 안전 검증을 거쳤다.
    Mongo 백엔드는 `sql` 을 쓰지 않고 range·filter 로 필터 dict를 직접 만든다.
    """

    schema: dict[str, Any]
    method: str
    sql: str
    filter_expr: str | None = None
    sort: str | None = None
    page: int = 1
    page_size: int = 20

    @property
    def range_(self) -> dict[str, Any] | None:
        """스키마의 적재 범위 — 어댑터가 쿼리에 자동 주입한다."""
        value: dict[str, Any] | None = self.schema.get("range")
        return value


@dataclass(frozen=True)
class ExecutionResult:
    """저장소 실행 결과. `executed=False` 면 서비스가 결정적 스텁 응답을 유지한다."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    total: int | None = None
    affected_rows: int | None = None
    executed: bool = False
    reason: str = ""

    @classmethod
    def not_executed(cls, reason: str) -> ExecutionResult:
        """실행하지 않았음을 이유와 함께 표시."""
        return cls(executed=False, reason=reason)
