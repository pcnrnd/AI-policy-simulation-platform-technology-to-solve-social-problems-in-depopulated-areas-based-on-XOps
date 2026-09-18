"""Overview 엔드포인트 — 카탈로그 롤업(소스 수·아카이브 행수) + 현행 모델 지표.

조회 전용 공개 메타데이터라 인증을 요구하지 않는다 (GET /dataops/catalog 와 같은 정책).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from src.dataops.summary import build_overview_summary

router = APIRouter(prefix="/overview", tags=["overview"])


@router.get("/summary")
def overview_summary(
    include_seed: bool = Query(False, description="데모 시드 소스·시드 지표 모델까지 포함 — 데모 표시 ON 전용"),
) -> dict[str, Any]:
    """Overview 지표 카드·아카이브 도넛이 쓰는 롤업.

    기본값은 데모 표시 OFF(`include_seed=false`) — 카탈로그 목록과 같은 규칙이다.
    """
    return build_overview_summary(include_seed=include_seed)
