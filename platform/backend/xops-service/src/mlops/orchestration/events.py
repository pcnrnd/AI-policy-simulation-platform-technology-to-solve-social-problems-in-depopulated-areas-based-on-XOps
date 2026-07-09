"""재학습 이벤트 + EventBus (model_id별 debounce).

드리프트/성능저하 감지가 RetrainEvent를 발생시킨다. 짧은 간격의 중복 트리거를 막되
(retrain_min_interval_minutes), 수동(manual) 이벤트는 항상 통과시킨다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RetrainEvent:
    """재학습 트리거 이벤트."""

    model_id: str
    trigger: str = "manual"  # "drift" | "performance" | "manual"
    candidate_metrics: dict[str, float] = field(default_factory=dict)
    candidate_latency_ms: float | None = None
    created_at: datetime = field(default_factory=_utcnow)


class EventBus:
    """model_id별 최소 간격 debounce. manual은 무조건 통과."""

    def __init__(self, min_interval_minutes: float) -> None:
        self._min_interval = timedelta(minutes=min_interval_minutes)
        self._last_accepted: dict[str, datetime] = {}

    def accept(self, event: RetrainEvent, now: datetime | None = None) -> bool:
        """이벤트 수용 여부. 수용 시 마지막 수용 시각 갱신."""
        current = now or _utcnow()
        if event.trigger == "manual":
            self._last_accepted[event.model_id] = current
            return True
        last = self._last_accepted.get(event.model_id)
        if last is not None and current - last < self._min_interval:
            return False
        self._last_accepted[event.model_id] = current
        return True

    def snapshot(self) -> dict[str, Any]:
        return {mid: ts.isoformat() for mid, ts in self._last_accepted.items()}
