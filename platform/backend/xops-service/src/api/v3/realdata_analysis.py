"""Retired diagnostic endpoint; authentication is retained for older consumers."""
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from src.api.dependencies import require_auth

router = APIRouter(prefix="/realdata", tags=["realdata-analysis"])


@router.get("/analysis", deprecated=True, responses={410: {"description": "진단 기능 종료"}})
def get_analysis(_: dict[str, Any] = Depends(require_auth("data:read"))) -> dict[str, Any]:
    raise HTTPException(410, "실데이터 진단 기능이 종료되었습니다. 등록한 모델의 요인분석과 시뮬레이션을 이용하세요.")
