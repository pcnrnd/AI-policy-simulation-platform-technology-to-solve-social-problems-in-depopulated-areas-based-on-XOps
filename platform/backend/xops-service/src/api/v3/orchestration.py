"""MLOps 오케스트레이션 엔드포인트 — 재학습 이벤트/파이프라인/모델 상태.

모델 스토어·오케스트레이터·실행 이력은 ModelRegistry(공유 싱글톤)가 보유한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import optional_auth
from src.core import db
from src.core.exceptions import SourceNotFoundError
from src.mlops.monitoring import realdata_bridge
from src.mlops.orchestration.registry import RUNNING_STATE, get_registry, recover_stale_runs
from src.schemas.orchestration import EventRequest, PipelineCreateRequest, PipelineRunRequest

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


@router.on_event("startup")
def _recover_stale_runs_on_startup() -> None:
    """기동 시 진행 중으로 남은 실행을 마감한다(realdata 학습 job과 같은 방침)."""
    recover_stale_runs()


# 목록 GET 3개의 `include_seed` 는 기본 false — 데모 표시 OFF가 기본 상태다.
# 시드 데이터는 지우지 않고 목록에서만 빠지며, 데모 ON 화면이 `include_seed=true` 로 되살린다.
# 단건 경로(`/pipelines/{id}/run`, `/runs/{id}/logs`)와 쓰기 경로는 이 필터를 적용하지 않는다.
_INCLUDE_SEED = Query(False, description="데모 시드(기본 파이프라인·시드 지표 모델)까지 포함 — 데모 표시 ON 전용")

# 데모 표시 OFF에서 실데이터 학습 job·모델을 같은 스키마로 함께 내려준다. 실데이터는
# `data:read` 토큰이 있는 호출에만 싣는다 — 토큰 없는 기존 공개 GET은 계약이 그대로다.
_OPTIONAL_AUTH = Depends(optional_auth("data:read"))


def _with_realdata(include_seed: bool, auth: dict[str, Any] | None) -> bool:
    return not include_seed and auth is not None


@router.get("/models")
def list_models(
    include_seed: bool = _INCLUDE_SEED,
    auth: dict[str, Any] | None = _OPTIONAL_AUTH,
) -> list[dict[str, Any]]:
    """등록된 운영 모델과 현재 버전/지표."""
    rows = get_registry().models(include_seed=include_seed)
    if _with_realdata(include_seed, auth):
        rows = rows + realdata_bridge.models()
    return rows


# ── ML 파이프라인 등록 ──────────────────────────────────────
@router.get("/pipelines")
def list_pipelines(
    include_seed: bool = _INCLUDE_SEED,
    auth: dict[str, Any] | None = _OPTIONAL_AUTH,
) -> list[dict[str, Any]]:
    """등록된 재학습 파이프라인 (등록이 없으면 빈 목록)."""
    rows = get_registry().pipelines(include_seed=include_seed)
    if _with_realdata(include_seed, auth):
        rows = rows + realdata_bridge.pipelines()
    return rows


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


@router.post("/pipelines/{pipeline_id}/run", status_code=202)
def run_pipeline(pipeline_id: str, body: PipelineRunRequest | None = None) -> dict[str, Any]:
    """등록 파이프라인 실행 접수 — `/events` 와 같은 백그라운드 경로(모델별 락을 함께 쓴다)."""
    registry = get_registry()
    pipeline = registry.get_pipeline(pipeline_id)
    request = body or PipelineRunRequest()
    run_id = registry.start_trigger(
        model_id=pipeline["model_id"],
        trigger=request.trigger,
        candidate_latency_ms=request.candidate_latency_ms,
        pipeline_id=pipeline_id,
    )
    return db.get_run(run_id) or {"run_id": run_id, "state": RUNNING_STATE}


# ── 실행 이력·로그 ──────────────────────────────────────────
@router.get("/runs")
def list_runs(
    pipeline_id: str | None = Query(None, description="파이프라인별 이력만"),
    include_seed: bool = _INCLUDE_SEED,
    auth: dict[str, Any] | None = _OPTIONAL_AUTH,
) -> list[dict[str, Any]]:
    """파이프라인 실행 이력 (최신 우선)."""
    rows = list(reversed(get_registry().runs(pipeline_id, include_seed=include_seed)))
    if _with_realdata(include_seed, auth):
        rows = rows + realdata_bridge.runs(pipeline_id)
    return rows


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """실행 하나의 현재 상태 — 접수(202) 뒤 종결까지 폴링하는 경로."""
    run = db.get_run(run_id)
    if run is None:
        raise SourceNotFoundError(f"실행 기록을 찾을 수 없습니다: {run_id}")
    return run


@router.get("/runs/{run_id}/logs")
def get_run_logs(run_id: str, auth: dict[str, Any] | None = _OPTIONAL_AUTH) -> dict[str, Any]:
    """실행 하나의 저장된 로그 라인. 실데이터 job은 job 레코드의 상태 전이를 라인으로 만든다."""
    run = db.get_run(run_id)
    if run is None:
        if auth is not None:
            realdata = realdata_bridge.run_logs(run_id)
            if realdata is not None:
                return realdata
        raise SourceNotFoundError(f"실행 기록을 찾을 수 없습니다: {run_id}")
    return {"run_id": run_id, "state": run.get("state"), "logs": run.get("logs", [])}


@router.post("/events", status_code=202)
def trigger_event(body: EventRequest) -> dict[str, Any]:
    """재학습 이벤트 접수 → 백그라운드 실행. 결과는 `GET /runs/{run_id}` 로 확인한다.

    학습·평가·배포가 요청 스레드를 붙잡지 않고, 진행 중 상태가 저장돼 프로세스가 죽어도
    실행 흔적이 남는다(기동 시 `recover_stale_runs`가 마감).
    """
    run_id = get_registry().start_trigger(
        model_id=body.model_id,
        trigger=body.trigger,
        candidate_metrics=body.candidate_metrics,
        candidate_latency_ms=body.candidate_latency_ms,
        pipeline_id=body.pipeline_id,
    )
    return db.get_run(run_id) or {"run_id": run_id, "state": RUNNING_STATE}
