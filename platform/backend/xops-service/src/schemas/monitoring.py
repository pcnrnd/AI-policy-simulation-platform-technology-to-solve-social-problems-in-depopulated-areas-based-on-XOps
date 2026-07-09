"""MLOps 모니터링 요청/응답 DTO."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegressionInput(BaseModel):
    """회귀 지표 계산 입력."""

    y_true: list[float] = Field(min_length=1)
    y_pred: list[float] = Field(min_length=1)


class ClassificationInput(BaseModel):
    """분류 지표 계산 입력."""

    y_true: list[int] = Field(min_length=1)
    y_pred: list[int] = Field(min_length=1)
    positive: int = 1


class DriftInput(BaseModel):
    """드리프트 판정 입력 — 참조/현재 분포."""

    reference: list[float] = Field(min_length=1)
    current: list[float] = Field(min_length=1)


class OutlierInput(BaseModel):
    """이상치 탐지 입력."""

    values: list[float] = Field(min_length=2)


class ExplainInput(BaseModel):
    """설명가능성 입력 — feature별 기여도 시퀀스."""

    contributions: dict[str, list[float]]
