"""MLOps 오케스트레이션 엔드포인트 — 재학습 이벤트/파이프라인/모델 상태.

모델 스토어·오케스트레이터·실행 이력은 ModelRegistry(공유 싱글톤)가 보유한다.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter

from src.mlops.orchestration.registry import get_registry
from src.schemas.orchestration import EventRequest

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


@router.get("/models")
def list_models() -> list[dict[str, Any]]:
    """등록된 운영 모델과 현재 버전/지표."""
    return get_registry().models()


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    """파이프라인 실행 이력 (최신 우선)."""
    return [asdict(r) for r in reversed(get_registry().runs())]


@router.post("/events")
def trigger_event(body: EventRequest) -> dict[str, Any]:
    """재학습 이벤트 발생 → 상태머신 실행."""
    run = get_registry().trigger(
        model_id=body.model_id,
        trigger=body.trigger,
        candidate_metrics=body.candidate_metrics,
        candidate_latency_ms=body.candidate_latency_ms,
    )
    return asdict(run)
