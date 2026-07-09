"""MLOps 오케스트레이션 요청/응답 DTO."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EventRequest(BaseModel):
    """재학습 이벤트 트리거 요청."""

    model_id: str
    trigger: str = Field("manual", pattern="^(manual|drift|performance)$")
    candidate_metrics: dict[str, float] = Field(default_factory=dict)
    candidate_latency_ms: float | None = None
