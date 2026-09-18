"""MLOps 오케스트레이션 엔드포인트 — 재학습 이벤트/파이프라인/모델 상태.

모델 스토어·오케스트레이터·실행 이력은 ModelRegistry(공유 싱글톤)가 보유한다.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query

from src.core import db
from src.core.exceptions import SourceNotFoundError
from src.mlops.orchestration.registry import get_registry
from src.schemas.orchestration import EventRequest, PipelineCreateRequest, PipelineRunRequest

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


# 목록 GET 3개의 `include_seed` 는 기본 false — 데모 표시 OFF가 기본 상태다.
# 시드 데이터는 지우지 않고 목록에서만 빠지며, 데모 ON 화면이 `include_seed=true` 로 되살린다.
# 단건 경로(`/pipelines/{id}/run`, `/runs/{id}/logs`)와 쓰기 경로는 이 필터를 적용하지 않는다.
_INCLUDE_SEED = Query(False, description="데모 시드(기본 파이프라인·시드 지표 모델)까지 포함 — 데모 표시 ON 전용")


@router.get("/models")
def list_models(include_seed: bool = _INCLUDE_SEED) -> list[dict[str, Any]]:
    """등록된 운영 모델과 현재 버전/지표."""
    return get_registry().models(include_seed=include_seed)


# ── ML 파이프라인 등록 ──────────────────────────────────────
@router.get("/pipelines")
def list_pipelines(include_seed: bool = _INCLUDE_SEED) -> list[dict[str, Any]]:
    """등록된 재학습 파이프라인 (등록이 없으면 빈 목록)."""
    return get_registry().pipelines(include_seed=include_seed)


@router.post("/pipelines", status_code=201)
def register_pipeline(body: PipelineCreateRequest) -> dict[str, Any]:
    """재학습 파이프라인 등록 → SQLite 영속화."""
    return get_registry().register_pipeline(
        {**body.model_dump(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    )


@router.delete("/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id: str) -> dict[str, str]:
    """등록 파이프라인 삭제. 실행 이력은 보존한다."""
    get_registry().delete_pipeline(pipeline_id)
    return {"deleted": pipeline_id}


@router.post("/pipelines/{pipeline_id}/run")
def run_pipeline(pipeline_id: str, body: PipelineRunRequest | None = None) -> dict[str, Any]:
    """등록 파이프라인 실행 → 실행 레코드·단계 상태·로그가 SQLite에 남는다."""
    registry = get_registry()
    pipeline = registry.get_pipeline(pipeline_id)
    request = body or PipelineRunRequest()
    run = registry.trigger(
        model_id=pipeline["model_id"],
        trigger=request.trigger,
        candidate_latency_ms=request.candidate_latency_ms,
        pipeline_id=pipeline_id,
    )
    return asdict(run)


# ── 실행 이력·로그 ──────────────────────────────────────────
@router.get("/runs")
def list_runs(
    pipeline_id: str | None = Query(None, description="파이프라인별 이력만"),
    include_seed: bool = _INCLUDE_SEED,
) -> list[dict[str, Any]]:
    """파이프라인 실행 이력 (최신 우선)."""
    return list(reversed(get_registry().runs(pipeline_id, include_seed=include_seed)))


@router.get("/runs/{run_id}/logs")
def get_run_logs(run_id: str) -> dict[str, Any]:
    """실행 하나의 저장된 로그 라인."""
    run = db.get_run(run_id)
    if run is None:
        raise SourceNotFoundError(f"실행 기록을 찾을 수 없습니다: {run_id}")
    return {"run_id": run_id, "state": run.get("state"), "logs": run.get("logs", [])}


@router.post("/events")
def trigger_event(body: EventRequest) -> dict[str, Any]:
    """재학습 이벤트 발생 → 상태머신 실행."""
    run = get_registry().trigger(
        model_id=body.model_id,
        trigger=body.trigger,
        candidate_metrics=body.candidate_metrics,
        candidate_latency_ms=body.candidate_latency_ms,
        pipeline_id=body.pipeline_id,
    )
    return asdict(run)
