"""MLOps 오케스트레이션 요청/응답 DTO."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EventRequest(BaseModel):
    """재학습 이벤트 트리거 요청."""

    model_id: str
    trigger: str = Field("manual", pattern="^(manual|drift|performance)$")
    candidate_metrics: dict[str, float] = Field(default_factory=dict)
    candidate_latency_ms: float | None = None
    # 등록 파이프라인이 띄운 실행이면 그 id — 실행 이력이 파이프라인별로 조회된다.
    pipeline_id: str | None = None


class PipelineCreateRequest(BaseModel):
    """ML 재학습 파이프라인 등록 요청."""

    # id는 실행 이력·URL 경로 식별자라 영숫자·하이픈·밑줄로 제한한다.
    id: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    name: str = Field(..., min_length=1, max_length=80)
    model_id: str
    trigger_policy: str = Field("수동", max_length=200)
    experiment: str = Field("", max_length=80)


class PipelineRunRequest(BaseModel):
    """등록 파이프라인 실행 요청 — 트리거만 받고 대상 모델은 등록 정의를 따른다."""

    trigger: str = Field("manual", pattern="^(manual|drift|performance)$")
    candidate_latency_ms: float | None = None
