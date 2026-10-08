"""실데이터 연계 응답 봉투·요청 DTO (M0 골격) — A·B·C 라우트가 공용으로 쓴다."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

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


class RestoreRequest(BaseModel):
    """POST /realdata/models/{model_id}/restore/{version} 요청 본문(R3-4) — 사유는 선택."""

    note: str | None = Field(default=None, max_length=500)


# ── 응답 payload(타 부서 연동 계약) ─────────────────────────────
# 라우트는 dict를 그대로 돌려준다(응답 JSON 불변). 아래 타입은 OpenAPI `responses` 문서화용이고,
# 실제 응답이 이 구조와 필드 단위로 같은지는 tests/integration/test_response_contracts.py가 검사한다.
# 예제에 null을 쓰지 않는다 — FastAPI가 OpenAPI를 exclude_none으로 내보내 그 키가 사라진다.


class RealdataModelEntry(BaseModel):
    """`GET /realdata/models` 의 `data[]` 항목 — 계약 고정 모델 2종의 활성 버전 상태."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_id": "namwon-observed-sales-next-month",
                    "target": "observed_sales_krw",
                    "active_version": "v20260918-3f9a2c",
                    "applied_at": "2026-09-18T07:12:44.120931+00:00",
                    "previous_version": "v20260911-8d1e04",
                    "retrain_needed": False,
                }
            ]
        }
    )

    model_id: str
    target: Literal["nonlocal_visitors", "observed_sales_krw"]
    active_version: str | None = Field(description="활성(반영) 버전. 반영 전이면 null")
    applied_at: str | None = Field(description="반영 시각(ISO 8601, UTC). 반영 전이면 null")
    previous_version: str | None = Field(description="직전 활성 버전(복원 대상). 없으면 null")
    retrain_needed: bool = Field(description="재학습 필요 판정(캐시). 활성 모델이 없으면 false")


class ForecastErrorMetrics(BaseModel):
    """예측 오차 지표 — 값이 작을수록 좋다. 계산하지 못한 값은 null."""

    mae: float | None
    rmse: float | None
    wape: float | None = Field(
        description="가중 절대 오차율 = Σ|오차| / Σ|관측|. 비율값이며 1을 넘을 수 있다(백분율 표시는 ×100)"
    )


class BaselineMetrics(ForecastErrorMetrics):
    """같은 평가 구간에서 단순 기준선(전년 동월, 없으면 전월)의 오차."""

    name: str = Field(description="기준선 이름. 현행 `yoy_or_prev_month`")


class EvalPeriod(BaseModel):
    """학습 시 순차 평가 구간."""

    model_config = ConfigDict(populate_by_name=True)

    from_: int = Field(alias="from", description="평가 시작 월(YYYYMM)")
    to: int = Field(description="평가 끝 월(YYYYMM)")
    n: int = Field(description="평가 행 수")


class ValidationEvaluation(BaseModel):
    """학습 시 검증 평가(후보에 저장된 값)."""

    kind: Literal["validation"]
    metrics: ForecastErrorMetrics
    baseline: BaselineMetrics
    eval_period: EvalPeriod | None


class OperationalScored(BaseModel):
    """반영 후 예측 대상 월의 관측이 도착해 계산한 운영 평가."""

    kind: Literal["operational"]
    base_ym: int = Field(description="예측 대상 월(YYYYMM)")
    n: int = Field(description="채점한 행(행정동) 수")
    metrics: ForecastErrorMetrics


class OperationalPending(BaseModel):
    """예측 대상 월의 관측이 아직 없어 운영 평가를 기다리는 상태."""

    kind: Literal["pending"]
    pending_months: list[int] = Field(description="관측을 기다리는 월(YYYYMM)")


class OperationalError(BaseModel):
    """아티팩트·데이터셋을 읽지 못해 운영 평가를 계산하지 못한 상태(HTTP는 200)."""

    kind: Literal["error"]
    message: str


class RealdataEvaluation(BaseModel):
    """`GET /realdata/models/{model_id}/evaluation` 의 `data` — 검증 평가와 운영 평가."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "validation": {
                        "kind": "validation",
                        "metrics": {"mae": 1520433.2, "rmse": 2210871.9, "wape": 0.118},
                        "baseline": {"name": "yoy_or_prev_month", "mae": 1890312.5, "rmse": 2604410.3, "wape": 0.147},
                        "eval_period": {"from": 202506, "to": 202508, "n": 69},
                    },
                    "operational": {"kind": "pending", "pending_months": [202510]},
                }
            ]
        }
    )

    validation: ValidationEvaluation
    operational: Annotated[
        OperationalScored | OperationalPending | OperationalError, Field(discriminator="kind")
    ]
