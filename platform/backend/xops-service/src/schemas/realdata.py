"""실데이터 연계 응답 봉투·요청 DTO (M0 골격) — A·B·C 라우트가 공용으로 쓴다."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# 식별자(영문·숫자·밑줄·하이픈) — SQL 식별자로 조립되지 않는 앱 레벨 ID라
# dataops.safety의 SQL_IDENTIFIER_RE(하이픈 불가)와 달리 하이픈을 허용한다.
_APP_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"

T = TypeVar("T")


class Provenance(BaseModel):
    """응답 값의 출처 — 데이터셋·모델 버전·규칙 버전과 계산 시각(R6)."""

    dataset_id: str | None = None
    model_id: str | None = None
    version: str | None = None
    observed_end_month: int | None = None
    rules_version: str | None = None
    computed_at: datetime


class Envelope(BaseModel, Generic[T]):
    """공통 응답 봉투 — status로 비즈니스 상태를 분리한다(HTTP 코드는 요청/서버 오류 전용)."""

    status: Literal["ok", "insufficient_data", "empty", "model_required", "pending", "error"]
    message: str | None = None
    data: T | None = None
    provenance: Provenance | None = None


class DatasetCreateRequest(BaseModel):
    """POST /realdata/datasets 요청 골격 — model_id 또는 target으로 스펙을 고른다(R1-6)."""

    model_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=_APP_ID_PATTERN)
    target: Literal["nonlocal_visitors", "observed_sales_krw"] | None = None
    spec_version: str = Field(default="v1", min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.\-]+$")


class TrainingRunRequest(BaseModel):
    """POST /realdata/training-runs 요청 골격(R3-1)."""

    model_id: str = Field(min_length=1, max_length=128, pattern=_APP_ID_PATTERN)
    dataset_id: str = Field(min_length=1, max_length=64, pattern=r"^ds-[0-9a-f]{12}$")
