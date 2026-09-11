"""실데이터 데이터셋 API(R1) — 스냅샷 생성·목록·조회. 소유: A(G0 §7)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import require_auth
from src.realdata import snapshot
from src.realdata.errors import DatasetNotFound, MappingError
from src.realdata.pg_reader import RealdataUnavailable
from src.schemas.realdata import DatasetCreateRequest, Envelope, Provenance

router = APIRouter(prefix="/realdata", tags=["realdata-datasets"])

_MAX_ROWS = 5000


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _envelope(status: str, *, message: str | None = None, data: Any = None, **provenance: Any) -> dict[str, Any]:
    return Envelope[Any](
        status=status,  # type: ignore[arg-type]
        message=message,
        data=data,
        provenance=Provenance(computed_at=_now(), **provenance),
    ).model_dump(mode="json")


def _resolve_target(body: DatasetCreateRequest) -> str:
    if body.target:
        return body.target
    if body.model_id:
        for target, model_id in snapshot.TARGETS.items():
            if model_id == body.model_id:
                return target
        raise HTTPException(422, f"알 수 없는 model_id입니다: {body.model_id!r}")
    raise HTTPException(422, "target 또는 model_id 중 하나가 필요합니다.")


@router.post("/datasets")
def create_dataset_route(
    body: DatasetCreateRequest,
    _: dict[str, Any] = Depends(require_auth("data:write")),
) -> dict[str, Any]:
    """스냅샷 생성(R1-6). 매핑 실패·PG 장애는 status=error(시드 폴백 없음)."""
    target = _resolve_target(body)
    try:
        record = snapshot.create_dataset(target, spec_version=body.spec_version)
    except MappingError as exc:
        return _envelope("error", message=f"{exc} (미매핑: {exc.unmapped})")
    except RealdataUnavailable as exc:
        return _envelope("error", message=str(exc))
    return _envelope(
        "ok",
        data=record.to_summary(),
        dataset_id=record.dataset_id,
        model_id=record.model_id,
        observed_end_month=record.observed_to,
    )


@router.get("/datasets")
def list_datasets_route(_: dict[str, Any] = Depends(require_auth("data:read"))) -> dict[str, Any]:
    """생성된 스냅샷 목록(요약, rows 제외)."""
    return _envelope("ok", data={"datasets": snapshot.list_datasets()})


@router.get("/datasets/{dataset_id}")
def get_dataset_route(
    dataset_id: str,
    include_rows: bool = Query(default=False),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    """스냅샷 요약+quality. `include_rows=true`면 rows 포함(최대 5,000행, 초과 시 절단 표시)."""
    try:
        record = snapshot.load_dataset(dataset_id)
    except DatasetNotFound as exc:
        raise HTTPException(404, str(exc)) from exc

    data = record.to_summary()
    message = None
    if include_rows:
        rows = record.rows
        if len(rows) > _MAX_ROWS:
            message = f"rows가 {len(rows)}행이라 {_MAX_ROWS}행으로 절단했습니다."
            rows = rows[:_MAX_ROWS]
        data["rows"] = rows
    return _envelope(
        "ok",
        message=message,
        data=data,
        dataset_id=record.dataset_id,
        model_id=record.model_id,
        observed_end_month=record.observed_to,
    )
