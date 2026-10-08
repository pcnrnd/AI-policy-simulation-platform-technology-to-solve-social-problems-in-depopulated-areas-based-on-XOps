"""Overview 응답 DTO — `GET /overview/summary` 롤업.

라우트는 dict를 그대로 돌려준다(응답 JSON 불변). 이 타입은 OpenAPI `responses` 문서화용이고,
실제 응답과 필드 단위로 같은지는 tests/integration/test_response_contracts.py가 검사한다.
예제에 null을 쓰지 않는다 — FastAPI가 OpenAPI를 exclude_none으로 내보내 그 키가 사라진다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal["database", "in-memory"]


class OverviewSourceRollup(BaseModel):
    """소스 1건 — 실제로 센 적재 행수와 출처 표기."""

    id: str
    label: str
    source: str | None
    db_adapter: str
    archive_rows: int | None = Field(description="저장소에서 직접 센 행수(60초 캐시). 확인하지 못하면 null(0과 구분)")
    storage_tier: str | None
    loaded_at: str | None
    source_kind: SourceKind = Field(description="행수를 실제로 셌으면 database, 못 셌으면 in-memory")
    user_registered: bool


class OverviewModelSnapshot(BaseModel):
    """Overview F1 카드가 보는 운영 모델(`population-forecast`)의 현행 버전과 지표."""

    model_id: str
    serving_version: str | None
    f1: float | None
    metrics_source: str | None = Field(description="지표 출처 — seed(시드)·trained(학습 실측) 등")


class OverviewSummary(BaseModel):
    """카탈로그 롤업(소스 수·적재 행수) + 현행 모델 지표. 공개 조회."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": 200,
                    "method": "GET",
                    "endpoint": "/api/v3/overview/summary",
                    "dataops_version": "3.0.0-R3",
                    "source_kind": "database",
                    "source_count": 1,
                    "archive_rows_total": 2496,
                    "archive_rows_counted": 1,
                    "archive_rows_unknown": 0,
                    "sources": [
                        {
                            "id": "ds_11_kt_namwon_monthly_dong_visitors",
                            "label": "KT 남원시 행정동별 월별 방문자",
                            "source": "PostgreSQL",
                            "db_adapter": "PostgreSQLAdapter",
                            "archive_rows": 2496,
                            "storage_tier": "hot",
                            "loaded_at": "2026-09-11",
                            "source_kind": "database",
                            "user_registered": False,
                        }
                    ],
                    "model": {
                        "model_id": "population-forecast",
                        "serving_version": "v3.1",
                        "f1": 0.8421,
                        "metrics_source": "trained",
                    },
                }
            ]
        }
    )

    status: int = Field(description="스캐폴드 호환 표기(항상 200). 판정에 쓰지 않는다")
    method: str
    endpoint: str
    dataops_version: str
    source_kind: SourceKind = Field(description="한 소스라도 실제로 셌으면 database")
    source_count: int
    archive_rows_total: int = Field(description="확인한 소스의 행수 합계(확인 못 한 소스는 빠진다)")
    archive_rows_counted: int
    archive_rows_unknown: int = Field(description="행수를 확인하지 못한 소스 수")
    sources: list[OverviewSourceRollup]
    model: OverviewModelSnapshot | None = Field(description="데모 표시 OFF에서 지표가 시드뿐이면 null")
