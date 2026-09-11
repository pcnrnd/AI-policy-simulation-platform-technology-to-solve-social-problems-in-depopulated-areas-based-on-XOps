"""실데이터 연계 R5(분석·진단·예측·대응 후보) 라우터 — prefix `/realdata`. 소유: C(G0 §7)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import require_auth
from src.realdata import analysis, rules, snapshot
from src.schemas.realdata import Envelope, Provenance

router = APIRouter(prefix="/realdata", tags=["realdata-analysis"])

_DONG_CODE_PATTERN = r"^(all|\d{8})$"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _envelope(status: str, *, message: str | None = None, data: Any = None, **provenance: Any) -> dict[str, Any]:
    return Envelope[Any](
        status=status,  # type: ignore[arg-type]
        message=message,
        data=data,
        provenance=Provenance(computed_at=_now(), **provenance),
    ).model_dump(mode="json")


@router.get("/analysis")
def get_analysis(
    region: str = Query("namwon"),
    dong_code: str = Query(..., pattern=_DONG_CODE_PATTERN),
    base_ym: int = Query(..., ge=190001, le=999912),
    model_id: str | None = Query(None),
    _: dict[str, Any] = Depends(require_auth("data:read")),
) -> dict[str, Any]:
    """`GET /realdata/analysis`(R5) — 현황·진단·예측·대응 후보를 한 번에 반환한다."""
    if region != "namwon":
        raise HTTPException(422, f"지원하지 않는 region입니다: {region!r} (namwon만 지원)")
    if model_id is not None and model_id not in snapshot.TARGETS.values():
        raise HTTPException(422, f"알 수 없는 model_id입니다: {model_id!r}")

    result = analysis.analyze(region=region, dong_code=dong_code, base_ym=base_ym, model_id=model_id)
    return _envelope(
        result["status"],
        message=result.get("message"),
        data=result.get("data"),
        rules_version=rules.RULES_VERSION,
    )
