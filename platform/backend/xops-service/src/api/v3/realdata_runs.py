"""실데이터 연계 R3(실행·후보·반영)·R4(모니터링) 라우터 — prefix `/realdata`.

M0 골격 라우터(`src.api.v3.realdata`, health·dong-map)와 별도 파일로 분리해 소유 경계를
지킨다(B 소유). 같은 prefix로 include하며 서브 경로가 겹치지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import require_auth
from src.realdata import candidates, jobs, monitoring
from src.realdata.candidates import ApplyRejected, CandidateNotFound
from src.realdata.jobs import JobConflict
from src.schemas.realdata import Envelope, Provenance, RestoreRequest, TrainingRunRequest

router = APIRouter(prefix="/realdata", tags=["realdata"])

# 계약(G0 §2) 고정 모델 2종 — model_id → 타깃 컬럼.
MODEL_TARGETS: dict[str, str] = {
    "namwon-nonlocal-visitors-next-month": "nonlocal_visitors",
    "namwon-observed-sales-next-month": "observed_sales_krw",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _envelope(status: str, *, message: str | None = None, data: Any = None, **provenance: Any) -> dict[str, Any]:
    return Envelope[Any](
        status=status,  # type: ignore[arg-type]
        message=message,
        data=data,
        provenance=Provenance(computed_at=_now(), **provenance),
    ).model_dump(mode="json")


def _require_known_model(model_id: str) -> None:
    if model_id not in MODEL_TARGETS:
        raise HTTPException(status_code=404, detail=f"알 수 없는 model_id: {model_id}")


def _decided_by(payload: dict[str, Any]) -> str:
    return str(payload.get("sub") or payload.get("source") or "api")


@router.on_event("startup")
def _recover_stale_jobs_on_startup() -> None:
    """서버 재시작 시 활성 상태로 남은 job을 failed(error="restart")로 마감한다(R3-1).

    main.py는 건드리지 않는 것이 소유 경계라, FastAPI의 (deprecated지만 여전히 동작하는)
    `router.on_event("startup")`을 썼다 — `APIRouter.include_router`가 on_startup 콜백을
    상위(app)까지 전파하므로(router.py → api_router → main.py 순으로 include_router가
    이미 호출됨) main.py 수정 없이 실제 uvicorn 기동 시 정상 등록된다. 다만 `TestClient(app)`을
    컨텍스트 매니저 없이 쓰는 이 저장소의 기존 관행에서는 lifespan이 트리거되지 않아
    통합 테스트로 이 경로를 검증하려면 `with TestClient(app) as client:`가 필요하다
    (완료 보고에 근거 명시)."""
    jobs.recover_stale_jobs()


# ── R3-1 실행 ──────────────────────────────────────────────
@router.post("/training-runs", status_code=202)
def create_training_run(
    body: TrainingRunRequest,
    _: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    _require_known_model(body.model_id)
    try:
        job_id = jobs.start_training(body.model_id, body.dataset_id)
    except JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    job = jobs.get_job(job_id)
    return _envelope("ok", data=job, model_id=body.model_id, dataset_id=body.dataset_id)


@router.get("/training-runs/{job_id}")
def get_training_run(
    job_id: str,
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job을 찾을 수 없습니다: {job_id}")
    status = "insufficient_data" if job["state"] == "failed" and (job.get("error") or "").startswith("insufficient_data") else "ok"
    return _envelope(status, data=job, model_id=job["model_id"], dataset_id=job["dataset_id"])


@router.get("/training-runs")
def list_training_runs(
    model_id: str | None = Query(None),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    rows = jobs.list_jobs(model_id)
    return _envelope("empty" if not rows else "ok", data=rows, model_id=model_id)


# ── 모델·후보 조회 ────────────────────────────────────────
@router.get("/models")
def list_models(_: dict[str, Any] = Depends(require_auth("data:read"))) -> dict[str, Any]:
    entries = []
    for model_id, target in MODEL_TARGETS.items():
        active = candidates.get_active(model_id)
        entries.append(
            {
                "model_id": model_id,
                "target": target,
                "active_version": active["version"] if active else None,
                "applied_at": active["applied_at"] if active else None,
                "previous_version": active["previous_version"] if active else None,
                "retrain_needed": candidates.retrain_needed(model_id) if active else False,
            }
        )
    return _envelope("ok", data=entries)


@router.get("/models/{model_id}/candidates")
def list_candidates(
    model_id: str,
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    _require_known_model(model_id)
    rows = candidates.list_candidates(model_id)
    return _envelope("empty" if not rows else "ok", data=rows, model_id=model_id)


# ── R3-3/3-4 반영·복원 ────────────────────────────────────
@router.post("/models/{model_id}/candidates/{version}/apply")
def apply_candidate(
    model_id: str,
    version: str,
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    _require_known_model(model_id)
    try:
        active = candidates.apply(model_id, version, _decided_by(payload))
    except CandidateNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApplyRejected as exc:
        raise HTTPException(status_code=400, detail={"reasons": exc.reasons}) from exc
    return _envelope("ok", data=active, model_id=model_id, version=version)


@router.post("/models/{model_id}/restore/{version}")
def restore_candidate(
    model_id: str,
    version: str,
    body: RestoreRequest | None = None,
    payload: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    _require_known_model(model_id)
    try:
        active = candidates.restore(model_id, version, _decided_by(payload), note=body.note if body else None)
    except CandidateNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApplyRejected as exc:
        raise HTTPException(status_code=400, detail={"reasons": exc.reasons}) from exc
    return _envelope("ok", data=active, model_id=model_id, version=version)


# ── R4 모니터링 ───────────────────────────────────────────
def _resolve_version(model_id: str, version: str | None) -> str | None:
    if version:
        return version
    active = candidates.get_active(model_id)
    return active["version"] if active else None


@router.get("/models/{model_id}/evaluation")
def get_evaluation(
    model_id: str,
    version: str | None = Query(None),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    _require_known_model(model_id)
    resolved = _resolve_version(model_id, version)
    if resolved is None:
        return _envelope("model_required", message="활성 모델이 없습니다.", model_id=model_id)
    result = monitoring.evaluation(model_id, resolved)
    return _envelope(result["status"], message=result.get("message"), data=result.get("data"), model_id=model_id, version=resolved)


@router.get("/models/{model_id}/explain")
def get_explain(
    model_id: str,
    base_ym: int = Query(...),
    dong_code: str = Query(...),
    version: str | None = Query(None),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    _require_known_model(model_id)
    resolved = _resolve_version(model_id, version)
    if resolved is None:
        return _envelope("model_required", message="활성 모델이 없습니다.", model_id=model_id)
    result = monitoring.explain(model_id, resolved, base_ym, dong_code)
    return _envelope(result["status"], message=result.get("message"), data=result.get("data"), model_id=model_id, version=resolved)


@router.get("/models/{model_id}/drift")
def get_drift(
    model_id: str,
    version: str | None = Query(None),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    _require_known_model(model_id)
    resolved = _resolve_version(model_id, version)
    if resolved is None:
        return _envelope("model_required", message="활성 모델이 없습니다.", model_id=model_id)
    result = monitoring.drift(model_id, resolved)
    return _envelope(result["status"], message=result.get("message"), data=result.get("data"), model_id=model_id, version=resolved)
