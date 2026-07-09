"""MLOps 오케스트레이션 엔드포인트 — 재학습 이벤트/파이프라인/모델 상태.

인프로세스 모델 스토어를 시드로 두고, 이벤트가 오면 상태머신을 태워 승급/롤백을 반영한다.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter

from src.core.exceptions import SourceNotFoundError
from src.mlops.orchestration.events import RetrainEvent
from src.mlops.orchestration.orchestrator import Orchestrator, PipelineRun
from src.schemas.orchestration import EventRequest

router = APIRouter(prefix="/orchestration", tags=["orchestration"])
_orchestrator = Orchestrator()
_runs: list[PipelineRun] = []

# 인프로세스 모델 스토어 (프론트 레지스트리 3종 대응). 실제 Model Store 연동은 후속.
_MODEL_STORE: dict[str, dict[str, Any]] = {
    "population-forecast": {
        "version": "v3.0-R3",
        "next_version": "v3.1",
        "metrics": {"accuracy": 0.892, "f1": 0.884, "precision": 0.891, "recall": 0.878, "mse": 0.041, "mae": 0.125},
    },
    "living-population": {
        "version": "v2.4",
        "next_version": "v2.5",
        "metrics": {"accuracy": 0.861, "f1": 0.852, "precision": 0.858, "recall": 0.847, "mse": 0.058, "mae": 0.147},
    },
    "settlement-demand": {
        "version": "v1.7",
        "next_version": "v1.8",
        "metrics": {"accuracy": 0.834, "f1": 0.821, "precision": 0.829, "recall": 0.814, "mse": 0.071, "mae": 0.166},
    },
}


@router.get("/models")
def list_models() -> list[dict[str, Any]]:
    """등록된 운영 모델과 현재 버전/지표."""
    return [{"model_id": mid, **info} for mid, info in _MODEL_STORE.items()]


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    """파이프라인 실행 이력 (최신 우선)."""
    return [asdict(r) for r in reversed(_runs)]


@router.post("/events")
def trigger_event(body: EventRequest) -> dict[str, Any]:
    """재학습 이벤트 발생 → 상태머신 실행. 승급 성공 시 스토어 버전 갱신."""
    model = _MODEL_STORE.get(body.model_id)
    if model is None:
        raise SourceNotFoundError(f"등록된 모델이 아닙니다: {body.model_id}")

    event = RetrainEvent(
        model_id=body.model_id,
        trigger=body.trigger,
        candidate_metrics=body.candidate_metrics,
        candidate_latency_ms=body.candidate_latency_ms,
    )
    run = _orchestrator.handle_event(
        event,
        current_metrics=model["metrics"],
        current_version=model["version"],
        candidate_version=model["next_version"],
    )
    _runs.append(run)

    if run.state == "succeeded":
        model["version"] = model["next_version"]

    return asdict(run)
